"""Run exact original DINOv3-Base OOF fold training or inference using saved models."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folds',required=True,choices=['01','23'])
    ap.add_argument('--mode',choices=['pretrained','train'],default='pretrained')
    ap.add_argument('--models-root',type=Path,default=Path('/kaggle/input'),
                    help='Original full fold snapshots or model-only weights with provenance')
    ap.add_argument('--data-root',type=Path,default=None)
    ap.add_argument('--splits-root',type=Path,default=None)
    ap.add_argument('--encoder',type=Path,default=None,help='Original DINOv3 Base encoder for training mode only')
    ap.add_argument('--resume-root',type=Path,default=None)
    ap.add_argument('--output',type=Path,default=None)
    ap.add_argument('--hours',type=float,default=9.5)
    ap.add_argument('--no-reuse-maps',action='store_true')
    ap.add_argument('--hf-repo',default='youssefElghanam191/edgeattnet-dinov3-convnext-base')
    ap.add_argument('--no-hf-download',action='store_true')
    a=ap.parse_args()
    if a.mode=='train' and a.encoder is None:
        ap.error('Training mode needs --encoder (the original DINOv3 ConvNeXt-Base pretrained encoder weights).')
    env=os.environ.copy()
    settings={'EDGE_OOF_FOLDS':a.folds,'EDGE_OOF_MODE':a.mode,
              'EDGE_OOF_MODELS_ROOT':str(a.models_root),'EDGE_OOF_REUSE_MAPS':'0' if a.no_reuse_maps else '1',
              'EDGE_OOF_HOURS':str(a.hours),
              'EDGE_OOF_HF_REPO':a.hf_repo,'EDGE_OOF_HF_DOWNLOAD':'0' if a.no_hf_download else '1'}
    for flag,value in [('EDGE_OOF_ENCODER',a.encoder),('EDGE_OOF_DATA_ROOT',a.data_root),
                       ('EDGE_OOF_SPLITS_ROOT',a.splits_root),('EDGE_OOF_RESUME_ROOT',a.resume_root),
                       ('EDGE_OOF_OUT',a.output)]:
        if value is not None:settings[flag]=str(value)
    env.update(settings)
    subprocess.run([sys.executable,str(ROOT/'src'/'oof_pair_runtime.py')],env=env,cwd=str(ROOT),check=True)

if __name__=='__main__':main()
