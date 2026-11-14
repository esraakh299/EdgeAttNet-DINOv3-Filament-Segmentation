import os,sys,json,time,math,hashlib,random,copy,gc,argparse
from pathlib import Path
import numpy as np,pandas as pd,scipy.ndimage as ndi
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def atomic_json(p,obj):
    p=Path(p);tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(p)
def atomic_torch(p,obj):
    p=Path(p);tmp=Path(str(p)+'.tmp');torch.save(obj,tmp);tmp.replace(p)
def state_digest(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def seed_everything(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def configure(hp):
    global HP,CROP,RECIPES,BOUNDARY_BCE_GAIN,AUX_WEIGHT,SKELETON_RECALL_WEIGHT,EMA_DECAY,TV_ALPHA,TV_BETA
    HP=hp;CROP=hp['crop'];RECIPES={'balanced':hp['recipe']}
    BOUNDARY_BCE_GAIN=hp['boundary'];AUX_WEIGHT=hp['aux'];SKELETON_RECALL_WEIGHT=0.;EMA_DECAY=hp['ema']
    TV_ALPHA,TV_BETA=.4,.6
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

class CachedBase:
    def __init__(self,names,cache,allowed):
        self.file_names=list(names);self.cache=Path(cache);self.allowed=set(allowed);self.fg_coords=None
        assert set(names)<=self.allowed
    def __len__(self):return len(self.file_names)
    @staticmethod
    def _to_float(a):return torch.from_numpy(np.asarray(a,dtype=np.float32)/255.)
    def _record(self,i):
        n=self.file_names[i];assert n in self.allowed
        raw=np.load(self.cache/(n+'.image.npy'),mmap_mode='r')
        sem=np.load(self.cache/(n+'.semantic.npy'),mmap_mode='r')
        # No spine supervision is active; this broadcast array carries no labels.
        spine=np.broadcast_to(np.zeros((1,*raw.shape),np.uint8),sem.shape)
        return raw,sem,spine

def norm2d(channels):
    return nn.GroupNorm(8, channels) if ACTIVE_NORM == "group" else nn.BatchNorm2d(channels)

ACTIVE_NORM="group"

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


class UNetEdgeTransformer(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, chs=(64, 128, 256, 512), heads=4, dropout=0.1, with_spine=False, aux=True):
        super().__init__()
        c1, c2, c3, c4 = chs
        self.enc1, self.enc2 = DoubleConv(in_ch, c1), DoubleConv(c1, c2)
        self.enc3, self.enc4 = DoubleConv(c2, c3), DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.edge_conv = nn.Conv2d(in_ch, 1, 3, padding=1)  # edge prior branch: 3x3 conv + sigmoid
        self.bottleneck = DoubleConv(c4, c4)
        self.eg1 = EdgeGuidedMHSA(c4, heads, dropout)
        self.eg2 = EdgeGuidedMHSA(c4, heads, dropout)
        self.up4, self.dec4 = nn.ConvTranspose2d(c4, c4, 2, stride=2), DoubleConv(2 * c4, c4)
        self.up3, self.dec3 = nn.ConvTranspose2d(c4, c3, 2, stride=2), DoubleConv(2 * c3, c3)
        self.up2, self.dec2 = nn.ConvTranspose2d(c3, c2, 2, stride=2), DoubleConv(2 * c2, c2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c2, c1, 2, stride=2), DoubleConv(2 * c1, c1)
        self.head = nn.Conv2d(c1, out_ch, 1)
        self.aux = aux
        if aux:
            self.aux3 = nn.Conv2d(c3,1,1)
            self.aux2 = nn.Conv2d(c2,1,1)
        self.with_spine = with_spine
        if with_spine:
            self.spine_head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        edge = torch.sigmoid(self.edge_conv(x))
        s1 = self.enc1(x); s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2)); s4 = self.enc4(self.pool(s3))
        z = self.bottleneck(self.pool(s4))
        z = self.eg2(self.eg1(z, edge), edge)
        d = self.dec4(torch.cat([self.up4(z), s4], 1))
        d = self.dec3(torch.cat([self.up3(d), s3], 1))
        a3=self.aux3(d) if self.aux else None
        d = self.dec2(torch.cat([self.up2(d), s2], 1))
        a2=self.aux2(d) if self.aux else None
        d = self.dec1(torch.cat([self.up1(d), s1], 1))
        seg_logits = self.head(d)
        if self.aux:
            return dict(logits=seg_logits,aux=[a3,a2])
        if self.with_spine:
            return seg_logits, self.spine_head(d)
        return seg_logits


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-Small encoder that returns the four maps EdgeAttNet's decoder expects.

    stage | stride | ConvNeXt ch -> adapter ch | plays the role of (ResNet34)
    0     | 4      | 96  -> 64                  | s1
    1     | 8      | 192 -> 128                 | s2
    2     | 16     | 384 -> 256                 | s3
    3     | 32     | 768 -> 512                 | z (input of the edge-attention blocks)

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
    """EdgeAttNet with a ConvNeXt-Base encoder (DINOv3 weights).

    Same as the ResNet34 OOF notebooks' model except for the encoder: same decoder, same two
    edge-attention blocks, same full-resolution raw skip, same deep supervision.
    ConvNeXt has no 1/2-resolution stage, so the s0 skip is built by a stride-2
    conv on the raw full-resolution branch. Those layers are new (trained from
    scratch), so this skip differs from the ResNet34 model, whose s0 is pretrained.
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

def build_model(norm='group',aux=True):return ConvNeXtEdgeAttNet(HP['encoder'])

def main_logits(output):
    return output['logits'] if isinstance(output,dict) else output

def segmentation_loss(logits,targets,stage='dice',boundary=True):
    # Every annotator gets equal weight; no hard union/consensus target.
    y=targets.float();z=logits.float().expand(-1,y.shape[1],-1,-1)
    bce=F.binary_cross_entropy_with_logits(z,y,reduction='none')
    if boundary and BOUNDARY_BCE_GAIN:
        outer=F.max_pool2d(y,5,stride=1,padding=2)
        inner=-F.max_pool2d(-y,5,stride=1,padding=2)
        w=1+BOUNDARY_BCE_GAIN*(outer-inner)
        bce=(bce*w).sum((-2,-1))/w.sum((-2,-1)).clamp_min(1)
    else: bce=bce.mean((-2,-1))
    p=z.sigmoid();tp=(p*y).sum((-2,-1))
    if stage=='dice': score=(2*tp+1e-6)/(p.sum((-2,-1))+y.sum((-2,-1))+1e-6)
    else:
        fp=(p*(1-y)).sum((-2,-1));fn=((1-p)*y).sum((-2,-1))
        score=(tp+1e-6)/(tp+TV_ALPHA*fp+TV_BETA*fn+1e-6)
    return (bce+1-score).mean()

def train_loss(output,targets,spines,stage,epoch):
    loss=segmentation_loss(main_logits(output),targets,stage)
    if isinstance(output,dict) and AUX_WEIGHT:
        for aux in output['aux']:
            y=F.interpolate(targets.float(),size=aux.shape[-2:],mode='area')
            loss=loss+(AUX_WEIGHT/len(output['aux']))*segmentation_loss(aux,y,stage,boundary=False)
    if SKELETON_RECALL_WEIGHT:
        p=main_logits(output).float().sigmoid().expand_as(spines)
        den=spines.sum((-2,-1));valid=den>0
        rec=(p*spines).sum((-2,-1))/den.clamp_min(1)
        if valid.any(): loss=loss+SKELETON_RECALL_WEIGHT*min(1,epoch/10)*(1-rec[valid]).mean()
    return loss

class ModelEMA:
    def __init__(self,net): self.module=copy.deepcopy(net).eval();self.updates=0;self.module.requires_grad_(False)
    @torch.no_grad()
    def update(self,net):
        self.updates+=1;d=min(EMA_DECAY,(1+self.updates)/(10+self.updates))
        source=net.state_dict()
        for k,v in self.module.state_dict().items():
            src=source[k].detach()
            if torch.is_floating_point(v): v.lerp_(src,1-d)
            else: v.copy_(src)

def cpu_state(net): return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}

class TargetedCropDataset(Dataset):
    def __init__(self,base,spec):
        self.base,self.spec=base,spec
        self.recipe=RECIPES[spec['recipe']]
        self.epoch=0; self.strength=1.0
    def __len__(self): return len(self.base)
    def set_epoch(self,epoch,strength=1.0): self.epoch,self.strength=int(epoch),float(strength)
    def _paste(self,image,sem,spine):
        # One real donor common to all donor annotators, never from validation.
        j=random.randrange(len(self.base))
        raw,ss,sk=self.base._record(j)
        common=np.all(ss>127,axis=0)
        labs,n=ndi.label(common)
        areas=np.bincount(labs.ravel())
        ids=np.flatnonzero((areas>=200)&(areas<=800)); ids=ids[ids>0]
        if not len(ids): return image,sem,spine
        cid=int(random.choice(ids)); sl=ndi.find_objects((labs==cid).astype(np.uint8))[0]
        y0=max(0,sl[0].start-8);y1=min(raw.shape[0],sl[0].stop+8)
        x0=max(0,sl[1].start-8);x1=min(raw.shape[1],sl[1].stop+8)
        mask=(labs[y0:y1,x0:x1]==cid)
        if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
            return image,sem,spine
        donor=raw[y0:y1,x0:x1].astype(np.float32)/255.
        residual=np.minimum(donor-ndi.gaussian_filter(donor,4),0)
        soft=ndi.gaussian_filter(mask.astype(np.float32),0.7)
        residual=torch.from_numpy(residual*soft)
        dm=torch.from_numpy(mask); ds=torch.from_numpy(np.any(sk[:,y0:y1,x0:x1]>127,axis=0)&mask)
        h,w=mask.shape; H,W=image.shape[-2:]
        if h+16>=H or w+16>=W: return image,sem,spine
        occupied=(sem>0.5).any(0)
        # Require an 8px gap; avoid joining unrelated instance labels in a semantic task.
        for _ in range(15):
            yy=random.randint(8,H-h-8);xx=random.randint(8,W-w-8)
            if occupied[yy-8:yy+h+8,xx-8:xx+w+8].any(): continue
            target=image[0,yy:yy+h,xx:xx+w]
            if target[dm].mean()<.15 or (target[dm]<.07).any(): continue
            image[:,yy:yy+h,xx:xx+w]+=residual*random.uniform(.8,1.1)
            sem[:,yy:yy+h,xx:xx+w]=torch.maximum(sem[:,yy:yy+h,xx:xx+w],dm.float())
            spine[:,yy:yy+h,xx:xx+w]=torch.maximum(spine[:,yy:yy+h,xx:xx+w],ds.float())
            break
        return image,sem,spine
    def _sample(self,i):
        raw,ss,sk=self.base._record(i); H,W=raw.shape
        r=self.recipe; strength=self.strength
        scale=random.uniform(*r['scale_range']) if random.random()<r['p_zoom']*strength else 1.0
        side=min(H,W,max(16,round(CROP/scale)))
        top=random.randint(0,H-side);left=random.randint(0,W-side)
        choose=random.random(); center=None
        if choose < 0.50:
            boxes=instance_boxes[self.base.file_names[i]]
            small=[v for v in boxes if 200<=v[2]<=800]
            pool=small if small and random.random()<.5 else boxes
            if pool:
                yy,xx,_=random.choice(pool);center=(yy,xx)
        elif choose < .70:
            if self.base.fg_coords is None: ys,xs=np.nonzero(np.any(ss>127,axis=0))
            else: ys,xs=self.base.fg_coords[i]
            if len(ys):
                k=random.randrange(len(ys));center=(ys[k],xs[k])
        if center is not None:
            # Center remains in the middle half of the crop; avoids just a tip.
            top=int(np.clip(center[0]-side/2+random.uniform(-.2,.2)*side,0,H-side))
            left=int(np.clip(center[1]-side/2+random.uniform(-.2,.2)*side,0,W-side))
        sl=(slice(top,top+side),slice(left,left+side))
        image=self.base._to_float(raw[sl]).unsqueeze(0)
        sem=(self.base._to_float(ss[:,sl[0],sl[1]])>.5).float()
        spine=(self.base._to_float(sk[:,sl[0],sl[1]])>.5).float()
        if side!=CROP:
            image=F.interpolate(image[None],size=(CROP,CROP),mode='bilinear',align_corners=False)[0]
            sem=F.interpolate(sem[None],size=(CROP,CROP),mode='nearest')[0]
            spine=F.interpolate(spine[None],size=(CROP,CROP),mode='nearest')[0]
        k=random.randrange(4);flip=random.random()<.5
        image,sem,spine=[torch.rot90(t,k,(-2,-1)) for t in (image,sem,spine)]
        if flip: image,sem,spine=[t.flip(-1) for t in (image,sem,spine)]
        if r['p_paste'] and random.random()<r['p_paste']*strength:
            image,sem,spine=self._paste(image,sem,spine)
        # At most ONE global brightness/contrast/gamma transform per sample.
        disk=(image>.04).float()
        if random.random()<r['p_photo']*strength:
            kind=random.choice(('gain','contrast','gamma'))
            if kind=='gain': image=image*random.uniform(1-.12*strength,1+.12*strength)
            elif kind=='gamma': image=image.clamp(0,1)**random.uniform(1-.12*strength,1+.12*strength)
            else:
                mu=(image*disk).sum()/disk.sum().clamp_min(1)
                image=(mu+(image-mu)*random.uniform(1-.15*strength,1+.15*strength))*disk
        if random.random()<.25*strength:
            cloud=torch.randn(1,1,4,4)
            cloud=F.interpolate(cloud,size=(CROP,CROP),mode='bicubic',align_corners=False)[0]
            cloud=cloud/cloud.abs().max().clamp_min(1)
            image=image*(1+cloud*r['cloud_amp']*strength)
        if random.random()<r['p_blur']*strength:
            image=F.avg_pool2d(F.pad(image[None],(1,1,1,1),mode='reflect'),3,stride=1)[0]
        if random.random()<r['p_noise']*strength:
            image=image+torch.randn_like(image)*random.uniform(.001,.005)*strength*disk
        return dict(image=image.clamp(0,1).contiguous(),semantic_masks=sem.contiguous(),
                    spine_masks=spine.contiguous(),file_name=self.base.file_names[i])
    def __getitem__(self,i):
        # Reproducible independent of worker scheduling; permits epoch resume.
        sample_seed=int(self.spec['seed'])+1000003*self.epoch+9176*int(i)
        py_state=random.getstate();np_state=np.random.get_state();t_state=torch.get_rng_state()
        random.seed(sample_seed); np.random.seed(sample_seed % 2**32)
        torch.default_generator.manual_seed(sample_seed)
        try: return self._sample(i)
        finally:
            random.setstate(py_state);np.random.set_state(np_state);torch.set_rng_state(t_state)



def train_fold(job,device_override=None):
    global instance_boxes
    manifest=json.loads(Path(job['manifest']).read_text());configure(manifest['hyperparameters'])
    fold=manifest['folds'][job['fold']];folder=Path(job['output']);folder.mkdir(parents=True,exist_ok=True)
    allowed=set(fold['training_names']);hold=set(fold['heldout_names']);original_val=set(manifest['validation_names'])
    assert allowed and allowed.isdisjoint(hold|original_val)
    sig=hashlib.sha256((manifest['signature']+f'/fold{job["fold"]}').encode()).hexdigest()
    done=folder/'run_complete.json'
    if done.exists():
        record=json.loads(done.read_text());assert record['signature']==sig
        assert sha256_file(folder/'fixed_final_ema.pth')==record['checkpoint_sha256']
        print('Already complete fold',job['fold'],flush=True);return
    device=torch.device(device_override or 'cuda:0')
    if device.type=='cuda':assert torch.cuda.device_count()==1;torch.cuda.set_device(device)
    seed_everything(fold['seed'])
    net=build_model('group',True)
    # DINOv3 encoder: file must exist, match the recorded SHA-256 and load strictly (no random fallback).
    assert Path(job['pretrained']).is_file() and sha256_file(job['pretrained'])==job['pretrained_sha256'],'DINOv3 encoder file missing or hash mismatch'
    state=torch.load(job['pretrained'],map_location='cpu',weights_only=True)
    net.encoder.load_pretrained(state);del state
    net=net.to(device);ema=ModelEMA(net)
    initial_digest=state_digest(net.state_dict())
    enc=list(net.encoder.body.parameters());eid={id(v) for v in enc}
    optimizer=torch.optim.AdamW([dict(params=enc,lr=HP['lr']*HP['encoder_lr_mult']),dict(params=[v for v in net.parameters() if id(v) not in eid],lr=HP['lr'])],weight_decay=HP['weight_decay'])
    base=CachedBase(fold['training_names'],job['cache'],allowed)
    instance_boxes={n:job['boxes'][n] for n in allowed}
    ds=TargetedCropDataset(base,dict(seed=fold['seed'],recipe='balanced'))
    updates=math.ceil(len(ds)/HP['accum']);total=updates*HP['epochs'];warm=updates*HP['warmup_epochs']
    def factor(step):
        if step<warm:return .1+.9*step/max(warm,1)
        phase=min(1,(step-warm)/max(total-warm,1))
        return HP['min_lr_ratio']+(1-HP['min_lr_ratio'])*.5*(1+math.cos(math.pi*phase))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    history=[];start=1;resume=folder/'last_resume.pth'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        assert saved['signature']==sig and saved['initialization']=='DINOv3_ConvNeXtBase_LVD1689M'
        assert saved['training_names']==fold['training_names'] and saved['heldout_names']==fold['heldout_names']
        net.load_state_dict(saved['model']);ema.module.load_state_dict(saved['ema']);ema.updates=saved['ema_updates']
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        history=saved['history'];start=saved['epoch']+1;initial_digest=saved['initial_state_digest']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'],device)
        del saved;gc.collect()
        print(f'fold {job["fold"]}: resume epoch {start}',flush=True)
    elif job.get('preflight',True):
        # No optimizer step and no model-selection labels. Restore RNG afterward.
        cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
        c=HP['crop'];x=torch.zeros((1,1,c,c),device=device);y=torch.zeros((1,3,c,c),device=device)
        net.train()
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):out=net(x)
        loss=train_loss(out,y,torch.zeros_like(y),'dice',1);loss.backward();optimizer.zero_grad(set_to_none=True)
        net.eval()
        full=int(job.get('full_res',2048))
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            q=main_logits(net(torch.zeros((1,1,full,full),device=device)))
            assert q.shape[-2:]==(full,full)
        del x,y,out,loss,q
        torch.set_rng_state(cr)
        if gr is not None:torch.cuda.set_rng_state(gr,device);torch.cuda.empty_cache()
        print('PREFLIGHT OK: real training backward + native full-frame inference',flush=True)
    def save_resume(epoch):
        atomic_torch(resume,dict(model=cpu_state(net),ema=cpu_state(ema.module),ema_updates=ema.updates,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
            signature=sig,epoch=epoch,history=history,initialization='DINOv3_ConvNeXtBase_LVD1689M',initial_state_digest=initial_digest,
            training_names=fold['training_names'],heldout_names=fold['heldout_names'],
            original_validation_used=False,heldout_labels_used=False,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
    if start==1:save_resume(0)
    hardware=dict(device=str(device),visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU test',torch_version=torch.__version__)
    atomic_json(folder/'hardware.json',hardware)
    for ep in range(start,HP['epochs']+1):
        remaining=job['deadline']-time.time();estimate=max([r['seconds'] for r in history[-3:]] or [900])*1.3
        if remaining<job['reserve']+estimate:
            atomic_json(folder/'run_status.json',dict(status='paused',epoch=ep-1,signature=sig))
            print('NORMAL BUDGET PAUSE; complete epoch checkpoint saved',flush=True);return
        t0=time.perf_counter();ds.set_epoch(ep,min(1,.3+.7*ep/10))
        loader=DataLoader(ds,batch_size=1,shuffle=True,num_workers=job.get('num_workers',0),
            generator=torch.Generator().manual_seed(fold['seed']+ep),pin_memory=device.type=='cuda')
        net.train();optimizer.zero_grad(set_to_none=True);losses=[];skipped=0;seen=[]
        for step,b in enumerate(loader):
            name=b['file_name'][0];assert name in allowed and name not in hold|original_val;seen.append(name)
            x=b['image'].to(device);y=b['semantic_masks'].to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):out=net(x)
            loss=train_loss(out,y,b['spine_masks'].to(device),'dice',ep)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            first=(step//HP['accum'])*HP['accum'];divisor=min(HP['accum'],len(loader)-first)
            scaler.scale(loss/divisor).backward();losses.append(float(loss.detach()))
            if (step+1)%HP['accum']==0 or step+1==len(loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(net.parameters(),HP['clip'])
                old=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()>=old:ema.update(net);scheduler.step()
                else:skipped+=1
                optimizer.zero_grad(set_to_none=True)
        assert set(seen)==allowed and len(seen)==len(allowed),'Unexpected training exposure'
        del loader,x,y,out,loss,b
        rec=dict(epoch=ep,train_loss=float(np.mean(losses)),seconds=time.perf_counter()-t0,
                 lr=float(optimizer.param_groups[-1]['lr']),amp_skipped=skipped,training_images=len(seen),
                 original_validation_exposures=0,heldout_exposures=0)
        history.append(rec);atomic_json(folder/'history.json',history)
        pd.DataFrame(history).to_csv(folder/'epoch_metrics.csv',index=False);save_resume(ep)
        print(f'fold {job["fold"]} epoch {ep}/{HP["epochs"]} | train loss {rec["train_loss"]:.5f} | {rec["seconds"]:.0f}s | LR {rec["lr"]:.2e} | AMP skips {skipped}',flush=True)
        if job.get('stop_after_epoch')==ep:return # local resume regression only
    payload=dict(model=cpu_state(ema.module),norm='group',aux=True,kind='ema',fold=job['fold'],
        epoch=HP['epochs'],signature=sig,manifest_signature=manifest['signature'],
        initialization='DINOv3_ConvNeXtBase_LVD1689M',initial_state_digest=initial_digest,training_names=fold['training_names'],
        heldout_names=fold['heldout_names'],original_validation_used=False,heldout_labels_used=False,
        selection='fixed_final_epoch_EMA')
    checkpoint=folder/'fixed_final_ema.pth';atomic_torch(checkpoint,payload)
    atomic_json(done,dict(signature=sig,fold=job['fold'],epochs=HP['epochs'],checkpoint_sha256=sha256_file(checkpoint),
                         initialization='DINOv3_ConvNeXtBase_LVD1689M',encoder_sha256=job['pretrained_sha256'],original_validation_used=False,heldout_labels_used=False))
    atomic_json(folder/'run_status.json',dict(status='complete',epoch=HP['epochs'],signature=sig))
    # Completed fold no longer needs a 3-state optimizer resume payload.
    if job.get('delete_completed_resume',True):resume.unlink(missing_ok=True)
    print('COMPLETE fold',job['fold'],flush=True)


@torch.inference_mode()
def infer_one(net,x,mode='full'):
    if mode=='full':
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            z=main_logits(net(x))
        return z.float().sigmoid()
    assert mode=='tile'
    H,W=x.shape[-2:]; tile=INFER_TILE; stride=tile-INFER_OVERLAP
    def starts(n):
        if n<=tile:return [0]
        return sorted(set(list(range(0,n-tile+1,stride))+[n-tile]))
    out=torch.zeros((1,1,H,W),device=x.device,dtype=torch.float32)
    den=torch.zeros_like(out)
    for y in starts(H):
        for xx in starts(W):
            patch=x[...,y:min(y+tile,H),xx:min(xx+tile,W)]
            h,w=patch.shape[-2:]
            patch=F.pad(patch,(0,(-w)%16,0,(-h)%16),mode='replicate')
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                p=main_logits(net(patch)).float().sigmoid()[...,:h,:w]
            # Nonzero edge weights also cover the outer image boundary.
            wy=torch.hann_window(h,periodic=False,device=x.device).clamp_min(.05)
            wx=torch.hann_window(w,periodic=False,device=x.device).clamp_min(.05)
            weight=wy[:,None]*wx[None,:]
            out[...,y:y+h,xx:xx+w]+=p*weight;den[...,y:y+h,xx:xx+w]+=weight
    assert bool((den>0).all())
    return out/den

@torch.inference_mode()
def predict_probability(net,x,mode='full',tta=False):
    views=[(False,0)] if not tta else [(f,k) for f in (False,True) for k in range(4)]
    avg=torch.zeros_like(x,dtype=torch.float32)
    for flip,k in views:
        view=torch.rot90(x.flip(-1) if flip else x,k,(-2,-1))
        p=torch.rot90(infer_one(net,view,mode),-k,(-2,-1))
        if flip:p=p.flip(-1)
        avg+=p/len(views)
    return avg



def infer_fold(job,device_override=None):
    global device
    manifest=json.loads(Path(job['manifest']).read_text());configure(manifest['hyperparameters'])
    fold=manifest['folds'][job['fold']];device=torch.device(device_override or 'cuda:0')
    if device.type=='cuda':assert torch.cuda.device_count()==1;torch.cuda.set_device(device)
    checkpoint=Path(job['checkpoint']);digest=sha256_file(checkpoint)
    assert digest==job['checkpoint_sha256']
    expected=hashlib.sha256((manifest['signature']+f'/fold{job["fold"]}').encode()).hexdigest()
    if job.get('checkpoint_format')=='model_only':
        # Inference-only tensor state_dict + independently packaged provenance.
        # The provenance binds the excluded images to this specific fold/model.
        meta_path=Path(job['checkpoint_metadata'])
        meta=json.loads(meta_path.read_text())
        assert meta['model_sha256']==digest, 'Model-only state_dict checksum mismatch'
        assert meta['signature']==expected and meta['manifest_signature']==manifest['signature']
        assert meta['fold']==job['fold']
        assert meta['epoch']==manifest['hyperparameters']['epochs'] and meta['selection']=='fixed_final_epoch_EMA'
        assert meta['initialization']=='DINOv3_ConvNeXtBase_LVD1689M'
        assert not meta['original_validation_used'] and not meta['heldout_labels_used']
        assert meta['training_names']==fold['training_names'] and meta['heldout_names']==fold['heldout_names']
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert isinstance(state,dict) and len(state)>300 and all(torch.is_tensor(t) for t in state.values())
        verified_train=meta['training_names'];verified_hold=meta['heldout_names']
    else:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert ck['signature']==expected and ck['manifest_signature']==manifest['signature']
        assert ck['epoch']==manifest['hyperparameters']['epochs'] and ck['selection']=='fixed_final_epoch_EMA'
        assert ck['initialization']=='DINOv3_ConvNeXtBase_LVD1689M' and not ck['original_validation_used'] and not ck['heldout_labels_used']
        assert ck['training_names']==fold['training_names'] and ck['heldout_names']==fold['heldout_names']
        state=ck['model'];verified_train=ck['training_names'];verified_hold=ck['heldout_names']
    assert set(verified_train).isdisjoint(verified_hold+manifest['validation_names'])
    net=build_model('group',True).to(device);net.load_state_dict(state,strict=True);net.eval()
    del state
    output=Path(job['output']);output.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256((expected+digest+str(bool(job['tta8']))+'/full/native/float16').encode()).hexdigest()
    done=0
    for name in fold['heldout_names']:
        assert name not in fold['training_names'] and name not in manifest['validation_names']
        path=output/(name+'.npy');marker=output/(name+'.json')
        if path.exists() and marker.exists():
            meta=json.loads(marker.read_text())
            if meta.get('signature')==signature and meta.get('map_sha256')==sha256_file(path):done+=1;continue
        if time.time()>job['deadline']-job['reserve']-120:
            print(f'NORMAL INFERENCE PAUSE fold {job["fold"]}; {done}/{len(fold["heldout_names"])} maps saved',flush=True);return
        start=time.perf_counter();img_path=Path(job['images'])/name
        assert sha256_file(img_path)==manifest['image_hashes'][name]
        raw=np.asarray(Image.open(img_path).convert('L'))
        x=torch.from_numpy(raw.astype(np.float32)/255.)[None,None].to(device)
        p=predict_probability(net,x,'full',bool(job['tta8']))[0,0].cpu().numpy().astype(np.float16)
        assert p.shape==raw.shape and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
        tmp=Path(str(path)+'.tmp')
        with tmp.open('wb') as f:np.save(f,p,allow_pickle=False)
        tmp.replace(path)
        atomic_json(marker,dict(signature=signature,map_sha256=sha256_file(path),checkpoint_sha256=digest,
            manifest_signature=manifest['signature'],fold=job['fold'],file_name=name,
            image_sha256=manifest['image_hashes'][name],tta8=bool(job['tta8']),mode='full',shape=list(raw.shape),
            image_excluded_from_model_training=True))
        done+=1
        if done%25==0 or done==len(fold['heldout_names']):print(f'OOF fold {job["fold"]}: {done}/{len(fold["heldout_names"])} | last image {time.perf_counter()-start:.1f}s',flush=True)
        del x,p,raw
    print('OOF inference complete fold',job['fold'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('job');ap.add_argument('--device',default=None);a=ap.parse_args()
    job=json.loads(Path(a.job).read_text())
    try:
        if job['mode']=='train':train_fold(job,a.device)
        elif job['mode']=='infer':infer_fold(job,a.device)
        else:raise ValueError('Unknown worker mode')
    except Exception as exc:
        folder=Path(job['output']);folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/'worker_error.json',dict(error=f'{type(exc).__name__}: {exc}'))
        raise
