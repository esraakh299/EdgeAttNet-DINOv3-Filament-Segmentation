"""Run the original DINOv3 ConvNeXt-Base training implementation unchanged.

Requires original Kaggle competition data, curated splits, training-only masks,
and trusted DINOv3 ConvNeXt-Base encoder weights. Fine-tunes the segmentation
model for 55 epochs, with original optimizer/EMA/snapshot selection. The encoder
starts from self-supervised DINOv3 weights, not random initialization.
"""
import argparse,subprocess,sys,os,time,hashlib
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--encoder',type=Path,required=True,help='Official pretrained DINOv3 ConvNeXt-Base state_dict')
    ap.add_argument('--data-root',type=Path,help='Exact competition dataset root; defaults to Kaggle discovery')
    ap.add_argument('--hours',type=float,default=9.5)
    ap.add_argument('--output-root',type=Path,default=Path('/kaggle/working/EdgeAttNet_DUAL_v1'))
    a=ap.parse_args()
    if not a.encoder.is_file():raise FileNotFoundError(f'Provide DINOv3 encoder weights: {a.encoder}')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Original training requires a CUDA GPU')
    a.output_root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from scripts.prepare_training_assets import prepare
    mask_root = prepare()
    split_root = Path(__file__).resolve().parents[1]/'data'/'splits'
    worker=Path(__file__).resolve().parents[1]/'src'/'training_worker.py'
    assert worker.is_file()
    env=os.environ.copy()
    env['EDGE_TRAIN_OUTPUT_ROOT'] = str(a.output_root.resolve())
    if a.data_root is not None:
        env['EDGE_TRAIN_DATA_ROOT'] = str(a.data_root.resolve())
    env.update(EDGE_TRAIN_SPLITS_ROOT=str(split_root),EDGE_TRAIN_MASKS_ROOT=str(mask_root),DUAL_VARIANT='dino_convnext_base',DUAL_START=str(time.time()),
               DUAL_BUDGET=str(a.hours*3600),DUAL_PRETRAINED=str(a.encoder.resolve()),
               DUAL_PRETRAIN_SHA=digest(a.encoder),DUAL_RESUME=str(a.output_root),
               CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
               PYTHONUNBUFFERED='1')
    log=a.output_root/'committee_training.log'
    with log.open('a') as f:
        subprocess.run([sys.executable,str(worker)],cwd=str(Path(__file__).resolve().parents[1]),env=env,
                       stdout=f,stderr=subprocess.STDOUT,check=True)
    best=a.output_root/'dino_convnext_base'/'dice_best_ema.pth'
    complete=a.output_root/'dino_convnext_base'/'run_complete.json'
    if not complete.is_file():raise RuntimeError(f'Training has not finished all scheduled epochs. Inspect {log} and resume; a partial best EMA must not be used as a completed run.')
    if not best.is_file():raise RuntimeError(f'Completed run has no best EMA. Inspect {log}.')
    print('EMA model:',best)
    print('Training log:',log)
if __name__=='__main__':main()
