"""Audit and index all 566 original leakage-safe OOF probability maps.

Example:
  python scripts/audit_oof_coverage.py \
    --pair-01 /kaggle/input/oof-01/DINO_OOF_Folds01 \
    --pair-23 /kaggle/input/oof-23/DINO_OOF_Folds23 \
    --out /kaggle/working/oof_verified

Output: oof_index.json. No 4+ GB array copying is required.
"""
import argparse
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.oof_helpers import sha256_file, map_signature


def audit(pair01,pair23,out):
    roots={0:Path(pair01),1:Path(pair01),2:Path(pair23),3:Path(pair23)}
    manifests={f:json.loads((r/'oof_training_manifest.json').read_text()) for f,r in roots.items()}
    sigs={m['signature'] for m in manifests.values()}
    if len(sigs)!=1:raise ValueError('OOF 01 and 23 manifests differ; they cannot be mixed')
    m=manifests[0]
    names=set(m['training_names']);vals=set(m['validation_names'])
    if len(names)!=566 or len(vals)!=141 or names & vals:
        raise ValueError('Unexpected fixed split; expected 566 training / 141 distinct validation physical images')
    records=[];covered=set();checkpoint_by_fold={}
    for fold in range(4):
        root=roots[fold]
        members=m['folds'][fold]
        if (set(members['training_names']) & set(members['heldout_names'])) or \
           (set(members['training_names']) & vals) or (set(members['heldout_names']) & vals):
            raise ValueError(f'Leakage detected in fold {fold}')
        for name in members['heldout_names']:
            if name in covered:raise ValueError('A training image has two OOF maps: '+name)
            p=root/'prob_oof'/f'fold_{fold}'/(name+'.npy')
            j=p.with_suffix('.json')
            if not p.is_file() or not j.is_file():
                raise FileNotFoundError(f'Incomplete fold {fold}: {p} or its JSON metadata is missing')
            meta=json.loads(j.read_text())
            digest=meta.get('checkpoint_sha256','')
            if fold in checkpoint_by_fold and checkpoint_by_fold[fold]!=digest:
                raise ValueError(f'Mixed checkpoint hashes within fold {fold}')
            checkpoint_by_fold[fold]=digest
            if not (meta.get('fold')==fold and meta.get('file_name')==name and
                    meta.get('signature')==map_signature(m,fold,digest) and
                    meta.get('manifest_signature')==m['signature'] and
                    meta.get('image_sha256')==m['image_hashes'][name] and
                    meta.get('image_excluded_from_model_training') is True and
                    meta.get('tta8') is True and meta.get('mode')=='full' and
                    meta.get('map_sha256')==sha256_file(p)):
                raise ValueError('OOF provenance or checksum mismatch: '+str(p))
            arr=np.load(p,mmap_mode='r',allow_pickle=False)
            if arr.ndim!=2 or arr.dtype!=np.float16 or arr.shape!=(2048,2048):
                raise ValueError('Unexpected OOF probability shape/type: '+str(p))
            records.append(dict(file_name=name,fold=fold,map_path=str(p.resolve()),
                                metadata_path=str(j.resolve()),sha256=meta['map_sha256']))
            covered.add(name)
    if covered!=names:raise ValueError(f'OOF coverage incomplete: {len(covered)}/566. Missing {sorted(names-covered)[:8]}')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    index=dict(version='edgeattnet_dinov3_base_oof_index_v1',manifest_signature=m['signature'],
               training_images=len(covered),validation_images=len(vals),folds=4,
               description='All four original held-out DINOv3 Base fold predictions; no validation images or in-sample outputs.',
               records=sorted(records,key=lambda x:x['file_name']))
    output=out/'oof_index.json';output.write_text(json.dumps(index,indent=2))
    print('Verified:',len(records),'genuine held-out maps (4 folds / 566 physical training images)')
    print('Index:',output)
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pair-01',type=Path,required=True)
    ap.add_argument('--pair-23',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();audit(a.pair_01,a.pair_23,a.out)

if __name__=='__main__':main()
