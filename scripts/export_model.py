"""Convert a trusted full EMA snapshot to an inference-only PyTorch state_dict.

Usage: python scripts/export_model.py --source /path/to/dice_best_ema.pth --output models/edgeattnet_dinov3_base_best.pth
"""
import argparse,json,hashlib
from pathlib import Path
import torch

def main():
    ap=argparse.ArgumentParser(description='Export model-only weights (no optimizer, scheduler or training history).')
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('models/edgeattnet_dinov3_base_best.pth'))
    a=ap.parse_args()
    # Weights-only is safe for basic tensor dictionaries and the supplied checkpoint wrapper.
    blob=torch.load(a.source,map_location='cpu',weights_only=True)
    state=blob['model'] if isinstance(blob,dict) and 'model' in blob else blob
    assert isinstance(state,dict) and len(state)>300
    assert all(torch.is_tensor(t) for t in state.values())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({k:v.cpu().contiguous() for k,v in state.items()},a.output)
    check=torch.load(a.output,map_location='cpu',weights_only=True)
    assert len(check)==len(state) and all(torch.equal(v,check[k]) for k,v in state.items())
    def digest(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    manifest={'architecture':'dino_convnext_base','format':'pytorch_state_dict',
              'tensors':len(state),'sha256':digest(a.output),'size_bytes':a.output.stat().st_size,
              'source_sha256':digest(a.source)}
    (a.output.with_suffix('.json')).write_text(json.dumps(manifest,indent=2))
    print('Exported:',a.output, '\n', json.dumps(manifest,indent=2))
if __name__=='__main__':main()
