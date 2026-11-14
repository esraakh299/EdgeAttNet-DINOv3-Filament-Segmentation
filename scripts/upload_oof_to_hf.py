"""Upload verified four-fold OOF model-only artifacts to an existing HF model repository.

Requires `HF_TOKEN` in the environment for write access. No token is stored
in source files or notebooks. Run --dry-run first.
"""
import argparse
import json
import os
from pathlib import Path
import hashlib

DEFAULT_REPO='youssefElghanam191/edgeattnet-dinov3-convnext-base'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as fd:
        for block in iter(lambda:fd.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models-dir',type=Path,required=True,help='Folder with all four exported .pth/.json pairs')
    ap.add_argument('--repo',default=DEFAULT_REPO)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    for f in range(4):
        stem=f'fold_{f}_model'
        weight=args.models_dir/(stem+'.pth')
        meta=args.models_dir/(stem+'.json')
        if not (weight.is_file() and meta.is_file()):
            raise FileNotFoundError(f'Missing pretrained OOF pair for fold {f}: {weight}, {meta}')
        obj=json.loads(meta.read_text())
        if obj.get('fold')!=f or obj.get('model_sha256')!=sha256(weight):
            raise ValueError(f'Invalid OOF artifact identity/hash: {stem}')
        if obj.get('selection')!='fixed_final_epoch_EMA' or obj.get('original_validation_used') is not False or obj.get('heldout_labels_used') is not False:
            raise ValueError(f'Unsafe or incorrect fold provenance: {stem}')
        print(stem,weight.stat().st_size,'bytes; hash verified')
    if args.dry_run:
        print('Dry run complete: eight files can be published in oof/.')
        return
    if not os.environ.get('HF_TOKEN'):
        raise EnvironmentError('Set HF_TOKEN securely for the upload only (not for public inference).')
    from huggingface_hub import HfApi
    HfApi(token=os.environ['HF_TOKEN']).upload_folder(
        repo_id=args.repo, repo_type='model',
        folder_path=str(args.models_dir),path_in_repo='oof',
        allow_patterns=['fold_?_model.pth','fold_?_model.json'],
        commit_message='Publish four verified training-only OOF EMA models')
    print('Uploaded all four OOF pairs to',args.repo+'/tree/main/oof')

if __name__=='__main__':
    main()
