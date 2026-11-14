"""OOF fold artifact verification and GPU job orchestration.

This module does not tune hyperparameters, alter fold assignments or use held-out
annotations for model selection. Provenance requirements prevent accidental
in-sample maps from entering the CatBoost training data.
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def fold_signature(manifest, fold):
    return hashlib.sha256((manifest['signature']+f'/fold{fold}').encode()).hexdigest()


def map_signature(manifest, fold, weight_sha):
    return hashlib.sha256((fold_signature(manifest,fold)+weight_sha+str(True)+'/full/native/float16').encode()).hexdigest()


def discover_fold_weights(root, manifest, folds, metadata_root=None):
    """Discover original fixed-final EMA wrappers or model-only + sidecar files.

    Original wrappers are accepted for private research sessions; exported
    model-only state_dict files are the preferred public committee artifacts.
    """
    root=Path(root)
    if not root.exists():raise FileNotFoundError(f'OOF model attachment root missing: {root}')
    found={}
    for fold in folds:
        wrappers=[p for p in root.rglob('fixed_final_ema.pth') if p.parent.name==f'fold_{fold}']
        singles=list(root.rglob(f'fold_{fold}_model.pth'))
        candidates=[]
        for p in wrappers:candidates.append(dict(checkpoint=p,format='snapshot',sha256=sha256_file(p)))
        for p in singles:
            meta=p.with_suffix('.json')
            if not meta.is_file():
                from src.oof_provenance import generate_sidecar
                target = Path(metadata_root)/meta.name if metadata_root is not None else meta
                meta = generate_sidecar(p, fold, target)
            d=json.loads(meta.read_text())
            digest=sha256_file(p)
            if not (d.get('model_sha256')==digest and d.get('manifest_signature')==manifest['signature']
                    and d.get('signature')==fold_signature(manifest,fold) and d.get('fold')==fold
                    and d.get('training_names')==manifest['folds'][fold]['training_names']
                    and d.get('heldout_names')==manifest['folds'][fold]['heldout_names']
                    and d.get('epoch')==manifest['hyperparameters']['epochs']
                    and d.get('selection')=='fixed_final_epoch_EMA'
                    and not d.get('original_validation_used',True) and not d.get('heldout_labels_used',True)):
                raise ValueError(f'OOF model-only provenance mismatch for fold {fold}: {p}')
            artifact_format = 'snapshot' if d.get('format') == 'pytorch_ema_snapshot' else 'model_only'
            candidates.append(dict(checkpoint=p,format=artifact_format,sha256=digest,metadata=meta))
        # A source may include its model-only export and the original full wrapper.
        # Prefer the smaller state_dict, but disallow conflicting versions.
        grouped={c['sha256']:c for c in candidates}
        if not grouped:
            raise FileNotFoundError(
                f'OOF fold {fold} model not found under {root}. Attach original '
                f'DINO_OOF_Folds01/23 fold_{fold}/fixed_final_ema.pth or '
                f'exported fold_{fold}_model.pth + fold_{fold}_model.json.')
        exported=[v for v in grouped.values() if v['format']=='model_only']
        selected=exported if exported else list(grouped.values())
        if len(selected)!=1:
            raise RuntimeError(f'Ambiguous OOF fold {fold} weights under {root}: '
                               +', '.join(str(x['checkpoint']) for x in selected))
        found[fold]=selected[0]
    return found


def copy_resume(source, output, manifest, folds):
    source=Path(source)
    prior=json.loads((source/'oof_training_manifest.json').read_text())
    if prior['signature']!=manifest['signature']:
        raise ValueError('Resume manifest differs from current folds/splits/recipe')
    for f in folds:
        for rel in [f'fold_{f}', f'prob_oof/fold_{f}']:
            if (source/rel).exists():shutil.copytree(source/rel,Path(output)/rel,dirs_exist_ok=True)


def install_precomputed_maps(root, output, manifest, fold, weight_info):
    """Re-use original per-image FP16 OOF arrays only if fully hash/provenance verified."""
    root=Path(root)
    if weight_info['format']!='snapshot':
        # Original maps are signed against the original snapshot hash; model-only
        # state dict bytes differ, so regenerate signed maps for this artifact.
        return False
    dirs=[p for p in root.rglob(f'fold_{fold}') if p.parent.name=='prob_oof']
    if len(dirs)>1:
        raise RuntimeError(f'Multiple OOF map folders for fold {fold}: {dirs}. Attach one unambiguous artifact set.')
    if not dirs:return False
    source=dirs[0]
    names=manifest['folds'][fold]['heldout_names']
    sig=map_signature(manifest,fold,weight_info['sha256'])
    pairs=[]
    for name in names:
        p=source/(name+'.npy');marker=source/(name+'.json')
        if not (p.is_file() and marker.is_file()):return False
        m=json.loads(marker.read_text())
        if not (m.get('signature')==sig and m.get('manifest_signature')==manifest['signature']
                and m.get('checkpoint_sha256')==weight_info['sha256'] and m.get('fold')==fold
                and m.get('file_name')==name and m.get('image_sha256')==manifest['image_hashes'][name]
                and m.get('image_excluded_from_model_training') is True
                and m.get('mode')=='full' and m.get('tta8') is True
                and m.get('map_sha256')==sha256_file(p)):
            return False
        pairs.append((p,marker))
    dest=Path(output)/'prob_oof'/f'fold_{fold}';dest.mkdir(parents=True,exist_ok=True)
    for p,marker in pairs:
        target=dest/p.name;dst_marker=dest/marker.name
        if not target.is_file() or sha256_file(target)!=sha256_file(p):shutil.copy2(p,target)
        if not dst_marker.is_file() or dst_marker.read_bytes()!=marker.read_bytes():shutil.copy2(marker,dst_marker)
    return True


def verify_pair_completion(root,manifest,folds):
    root=Path(root);status=[]
    for f in folds:
        names=manifest['folds'][f]['heldout_names'];folder=root/'prob_oof'/f'fold_{f}'
        complete=True
        for name in names:
            p=folder/(name+'.npy');meta=folder/(name+'.json')
            if not p.is_file() or not meta.is_file():complete=False;break
            d=json.loads(meta.read_text())
            if not (d.get('file_name')==name and d.get('fold')==f
                    and d.get('signature')==map_signature(manifest,f,d.get('checkpoint_sha256',''))
                    and d.get('manifest_signature')==manifest['signature']
                    and d.get('image_excluded_from_model_training') is True
                    and d.get('map_sha256')==sha256_file(p)):
                complete=False;break
        status.append(dict(fold=f,complete=complete,heldout_images=len(names)))
    return status


def launch_pair_jobs(mode,worker,output,manifest,folds,deadline,cache,images,encoder,encoder_sha,boxes,checkpoints=None):
    """Launch original GPU workers: 2 GPUs in parallel, or one GPU sequentially.

    Training uses exactly two GPUs in the source notebook, but pretrained
    inference also supports a single Kaggle GPU (one fold at a time).
    """
    import torch
    count=torch.cuda.device_count()
    if count<1:raise RuntimeError('No CUDA device available')
    output=Path(output);jobs=[]
    def collect(jobs):
        while any(j['proc'].poll() is None for j in jobs):
            for j in jobs:
                with j['path'].open() as fh:
                    fh.seek(j['pos']);new=fh.read();j['pos']=fh.tell()
                if new:print(new,end='',flush=True)
            time.sleep(5)
        for j in jobs:
            j['handle'].close()
            with j['path'].open() as fh:
                fh.seek(j['pos']);print(fh.read(),end='',flush=True)
            if j['proc'].returncode:
                raise RuntimeError(f'OOF {mode} fold {j["fold"]} failed (exit {j["proc"].returncode}); see {j["path"]}')
    for start in range(0,len(folds),count):
        active=[]
        for gpu,f in enumerate(folds[start:start+count]):
            folder=output/f'fold_{f}';folder.mkdir(parents=True,exist_ok=True)
            job=dict(mode=mode,fold=f,manifest=str(output/'oof_training_manifest.json'),
                     output=str(folder if mode=='train' else output/'prob_oof'/f'fold_{f}'),
                     cache=str(cache),boxes=boxes if mode=='train' else {},images=str(images),
                     deadline=deadline,reserve=600,num_workers=0,delete_completed_resume=False,
                     pretrained=str(encoder),pretrained_sha256=str(encoder_sha),tta8=True)
            if mode=='infer':
                rec=checkpoints[f]
                job.update(checkpoint=str(rec['checkpoint']),checkpoint_sha256=rec['sha256'],checkpoint_format=rec['format'])
                if rec.get('metadata'):job['checkpoint_metadata']=str(rec['metadata'])
            jp=output/f'{mode}_{f}.json';jp.write_text(json.dumps(job,indent=2,default=str))
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',
                                               OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            lp=output/f'{mode}_fold{f}.log';handle=lp.open('w')
            proc=subprocess.Popen([sys.executable,str(worker),str(jp)],stdout=handle,
                                  stderr=subprocess.STDOUT,env=env)
            active.append(dict(proc=proc,path=lp,handle=handle,pos=0,fold=f))
        collect(active)
