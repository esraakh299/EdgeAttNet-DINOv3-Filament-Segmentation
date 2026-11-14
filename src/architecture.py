"""Architecture extracted verbatim from edgeattnet-dinov3-postprocessing.ipynb cell 10."""
import torch
import torch.nn as nn
import torch.nn.functional as F
ACTIVE_NORM='group'


def norm2d(channels):
    return nn.GroupNorm(8, channels) if ACTIVE_NORM == "group" else nn.BatchNorm2d(channels)


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), norm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), norm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class EdgeGuidedMHSA(nn.Module):
    """EG-MHSA: Q = x + E', K = x + E', V = x  (E' = 1x1-conv-projected, resized edge prior).
    4 heads, no positional encoding, residual + LayerNorm + dropout."""

    def __init__(self, dim=512, heads=4, dropout=0.1):
        super().__init__()
        self.edge_proj = nn.Conv2d(1, dim, 1)          # E -> E' (own projection per block)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, z, edge):                        # z: B,C,H',W'   edge: B,1,H,W
        B, C, H, W = z.shape
        e = F.interpolate(edge, size=(H, W), mode="bilinear", align_corners=False)
        e = self.edge_proj(e).flatten(2).transpose(1, 2)   # B,N,C
        x = z.flatten(2).transpose(1, 2)                   # B,N,C
        out, _ = self.attn(x + e, x + e, x, need_weights=False)
        x = self.norm(x + self.drop(out))
        return x.transpose(1, 2).reshape(B, C, H, W)


def main_logits(output):
    return output['logits'] if isinstance(output,dict) else output


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-Base encoder that returns the four maps EdgeAttNet's decoder expects.

    stage | stride | ConvNeXt ch -> adapter ch
    0     | 4      | 128  -> 64                  | s1
    1     | 8      | 256 -> 128                 | s2
    2     | 16     | 512 -> 256                 | s3
    3     | 32     | 1024 -> 512                 | z (input of the edge-attention blocks)

    timm_name is the timm weights tag to build (DINOv3 LVD-1689M); the weights themselves
    are loaded by the worker from the verified file (load_pretrained).
    """
    def __init__(self, timm_name):
        super().__init__()
        import timm
        # Full timm model, same layout as the saved weight files, so strict loading works.
        # Its final head.norm is never used in forward(); it stays only so that
        # the saved weights load strictly (every key must match).
        self.body = timm.create_model(timm_name, pretrained=False, num_classes=0)
        # Adapters: a 1x1 conv + norm that changes the channel count to what the decoder expects.
        # Match the encoder's actual feature channels, as in the training notebook.
        in_chs = [f["num_chs"] for f in self.body.feature_info]
        out_chs = [64, 128, 256, 512]
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), norm2d(cout))
            for cin, cout in zip(in_chs, out_chs)])


    def load_pretrained(self, state):
        """Strict load of a saved state dict: any missing or extra key raises an error."""
        return self.body.load_state_dict(state, strict=True)

    def forward(self, x):
        x = self.body.stem(x)                      # patchify: stride 4
        feats = []
        for stage, adapt in zip(self.body.stages, self.adapters):
            x = stage(x)                           # stages 1-3 halve the resolution first
            feats.append(adapt(x))
        return feats                               # [s1, s2, s3, z]


class ConvNeXtEdgeAttNet(nn.Module):
    """EdgeAttNet with a ConvNeXt-base encoder (DINOv3 weights).

    The four adapted DINOv3 encoder stages feed the edge-guided decoder.
    The full-resolution image skip and auxiliary heads are trained for this task.
    """
    def __init__(self, timm_name):
        super().__init__()
        self.encoder=ConvNeXtEncoder(timm_name)
        self.full_stem=DoubleConv(1,64)
        self.s0_conv=nn.Sequential(nn.Conv2d(64,64,3,stride=2,padding=1,bias=False),norm2d(64),nn.ReLU(inplace=True))
        self.edge_conv=nn.Conv2d(1,1,3,padding=1)
        self.eg1=EdgeGuidedMHSA(512,4,.1);self.eg2=EdgeGuidedMHSA(512,4,.1)
        self.dec4=DoubleConv(512+256,256)
        self.dec3=DoubleConv(256+128,128)
        self.dec2=DoubleConv(128+64,64)
        self.dec1=DoubleConv(64+64,64)
        self.dec0=DoubleConv(64+64,64)
        self.head=nn.Conv2d(64,1,1);self.aux2=nn.Conv2d(64,1,1);self.aux3=nn.Conv2d(64,1,1)
        # ImageNet mean/std: what the DINOv3 weights expect (asserted in the weights cell).
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    @staticmethod
    def join(x,skip):return torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1)
    def forward(self,x):
        raw=self.full_stem(x);e=torch.sigmoid(self.edge_conv(x))
        rgb=(x.expand(-1,3,-1,-1)-self.rgb_mean)/self.rgb_std
        s1,s2,s3,z=self.encoder(rgb)
        s0=self.s0_conv(raw)
        z=self.eg2(self.eg1(z,e),e)
        d=self.dec4(self.join(z,s3));d=self.dec3(self.join(d,s2))
        d=self.dec2(self.join(d,s1));a3=self.aux3(d)
        d=self.dec1(self.join(d,s0));a2=self.aux2(d)
        d=self.dec0(self.join(d,raw))
        return dict(logits=self.head(d),aux=[a3,a2])


def build_model():
    return ConvNeXtEdgeAttNet("convnext_base.dinov3_lvd1689m")
