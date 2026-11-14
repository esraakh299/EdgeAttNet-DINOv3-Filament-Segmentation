"""Export four OOF inference-only EMA weights with fold-specific provenance.

For each fold:
  python scripts/export_oof_models.py --source /path/to/fold_0/fixed_final_ema.pth --out models/oof

The .pth is a pure state_dict. The .json proves which training images were
excluded and binds the artifact to the original four-fold training manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True,help='Original fixed_final_ema.pth wrapper')
    ap.add_argument('--out',type=Path,default=Path('models/oof'))
    ap.add_argument('--manifest',type=Path,default=None,help='Original oof_training_manifest.json (recommended)')
    a=ap.parse_args()
    ck=torch.load(a.source,map_location='cpu',weights_only=True)
    required=['model','fold','epoch','signature','manifest_signature','training_names','heldout_names',
              'selection','initialization','original_validation_used','heldout_labels_used']
    if not isinstance(ck,dict) or any(k not in ck for k in required):
        raise ValueError('Not an original fixed-final EMA fold snapshot; missing required provenance fields')
    if ck['selection']!='fixed_final_epoch_EMA' or ck['initialization']!='DINOv3_ConvNeXtBase_LVD1689M' \
       or ck['original_validation_used'] or ck['heldout_labels_used']:
        raise ValueError('Fold is not the original leakage-safe fixed-epoch EMA model')
    f=ck['fold'];assert f in range(4)
    if a.manifest:
        m=json.loads(a.manifest.read_text())
        sig=hashlib.sha256((m['signature']+f'/fold{f}').encode()).hexdigest()
        assert ck['manifest_signature']==m['signature'] and ck['signature']==sig
        assert ck['training_names']==m['folds'][f]['training_names']
        assert ck['heldout_names']==m['folds'][f]['heldout_names']
        assert ck['epoch']==m['hyperparameters']['epochs']
    state=ck['model']
    assert isinstance(state,dict) and len(state)>300 and all(torch.is_tensor(t) for t in state.values())
    a.out.mkdir(parents=True,exist_ok=True)
    target=a.out/f'fold_{f}_model.pth'
    torch.save({k:v.detach().cpu().contiguous() for k,v in state.items()},target)
    new=torch.load(target,map_location='cpu',weights_only=True)
    assert len(state)==len(new) and all(torch.equal(v,new[k]) for k,v in state.items())
    provenance={k:ck[k] for k in required if k!='model'}
    provenance.update(format='pytorch_state_dict',model_sha256=sha256(target),
                      source_sha256=sha256(a.source),tensor_count=len(new))
    sidecar=target.with_suffix('.json')
    sidecar.write_text(json.dumps(provenance,indent=2))
    print('Exported:',target,'\nProvenance:',sidecar)

if __name__=='__main__':main()
