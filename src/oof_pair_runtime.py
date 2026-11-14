"""Original four-fold DINOv3 ConvNeXt-Base OOF method, driven by notebooks/04 and 05.

Train routines, model/OOF manifest parameters, temporal embargo and image
preparation are sourced from the two original user-supplied fold notebooks.
Pretrained mode verifies fold provenance instead of retraining.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,subprocess,importlib.util,json,hashlib,time,zipfile,shutil,re,gc,math
from pathlib import Path
from collections import defaultdict,Counter
needed={'numpy':'numpy','pandas':'pandas','scipy':'scipy','PIL':'pillow','pyarrow':'pyarrow',
        'pycocotools':'pycocotools','torch':'torch','torchvision':'torchvision','cv2':'opencv-python-headless',
        'sklearn':'scikit-learn','matplotlib':'matplotlib','joblib':'joblib','IPython':'ipython',
        'timm':'timm','huggingface_hub':'huggingface_hub'}
missing=[p for m,p in needed.items() if importlib.util.find_spec(m) is None]
if missing:raise ImportError('Missing training dependencies: '+', '.join(missing)+'; install requirements-train.txt before starting')
import numpy as np,pandas as pd,scipy.ndimage as ndi
from PIL import Image
from pycocotools import mask as mask_utils
import torch,cv2,matplotlib.pyplot as plt,sklearn,joblib
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from threadpoolctl import threadpool_limits
from IPython.display import display,Markdown
cv2.setNumThreads(1)


INPUT_ROOT='/kaggle/input';KAGGLE_INPUT=Path(INPUT_ROOT)
DATA_ROOT_OVERRIDE=os.environ.get('EDGE_OOF_DATA_ROOT','');SPLIT_ROOT_OVERRIDE=os.environ.get('EDGE_OOF_SPLITS_ROOT','')
FOLDS_TO_TRAIN=tuple(int(i) for i in os.environ.get('EDGE_OOF_FOLDS','01'))
TRAIN_OOF_FROM_SCRATCH=(os.environ.get('EDGE_OOF_MODE','pretrained')=='train')
MODEL_INPUT_ROOT=Path(os.environ.get('EDGE_OOF_MODELS_ROOT','/kaggle/input'))
OOF_HF_REPO=os.environ.get('EDGE_OOF_HF_REPO','youssefElghanam191/edgeattnet-dinov3-convnext-base')
OOF_HF_DOWNLOAD=(os.environ.get('EDGE_OOF_HF_DOWNLOAD','1')=='1')
REUSE_PRECOMPUTED_MAPS=(os.environ.get('EDGE_OOF_REUSE_MAPS','1')=='1')
RESUME_ROOT=os.environ.get('EDGE_OOF_RESUME_ROOT','') # exact prior pair output folder
SESSION_HOURS=float(os.environ.get('EDGE_OOF_HOURS','9.5'))
PRETRAINED_PATH=os.environ.get('EDGE_OOF_ENCODER','') # only needed for fold training
HF_SECRET_NAME='dinov3' # Kaggle secret holding the Hugging Face read token
ENCODER_TAG='convnext_base.dinov3_lvd1689m'
OUT=Path(os.environ.get('EDGE_OOF_OUT', '/kaggle/working/DINO_OOF_Folds'+''.join(map(str,FOLDS_TO_TRAIN))));OUT.mkdir(parents=True,exist_ok=True)
TRAIN_CACHE_ROOT=Path(os.environ.get('EDGE_OOF_CACHE_ROOT','/kaggle/temp/DINO_OOF_training_cache'));TRAIN_CACHE_ROOT.mkdir(parents=True,exist_ok=True)
EXPECTED_TRAIN_IMAGES,EXPECTED_VAL_IMAGES=566,141
N_FOLDS=4;FOLD_EPOCHS=55;FOLD_EMBARGO_DAYS=2.;SEED=42
OOF_VERSION='dino_oof_v1'
# Identifies the model code family; encoder weights are verified separately by file SHA-256.
MODEL_CODE_DIGEST=hashlib.sha256(b'ConvNeXtEdgeAttNet|dino_oof_worker_v1|'+ENCODER_TAG.encode()).hexdigest()
HP=dict(encoder=ENCODER_TAG,encoder_lr_mult=.1,epochs=55,crop=1024,lr=2e-4,weight_decay=1e-4,accum=2,warmup_epochs=3,min_lr_ratio=.1,clip=1.,ema=.99,boundary=1.,aux=.15,recipe=dict(scale_range=[.8,1.2],p_zoom=.45,p_photo=.55,cloud_amp=.08,p_blur=.06,p_noise=.10,p_paste=0.))
STARTED=time.perf_counter()
def log(x):print(f'[{(time.perf_counter()-STARTED)/60:.1f}m] {x}',flush=True)
def atomic_json(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(p)
def sha256_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def unique_json(paths,label):
 groups={sha256_file(p):p for p in paths};assert len(groups)==1,f'{label}: expected one distinct source, found {paths}'
 p=next(iter(groups.values()));return p,json.loads(p.read_text())
def digest_json(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


if DATA_ROOT_OVERRIDE:DATA_ROOT=Path(DATA_ROOT_OVERRIDE)
else:
    roots=sorted({p.resolve() for p in KAGGLE_INPUT.rglob('MAGFiLO_1.0_Kaggle_2026') if p.is_dir()})
    if len(roots)!=1:raise RuntimeError(f'Attach official competition data or set DATA_ROOT_OVERRIDE. Found {roots}')
    DATA_ROOT=roots[0]
image_dirs=sorted({p.parent for p in (DATA_ROOT/'train').rglob('*') if p.is_file() and p.suffix.lower() in {'.jpeg','.jpg','.png'}})
assert len(image_dirs)==1,'Expected one original training image folder'
TRAIN_IMAGES_DIR=image_dirs[0]
candidate_json=[]
for p in (DATA_ROOT/'train').rglob('*.json'):
    try:
        d=json.loads(p.read_text())
        if isinstance(d,dict) and {'images','annotations'}<=set(d):candidate_json.append(p)
    except (ValueError,UnicodeDecodeError):pass
COCO_PATH,coco=unique_json(candidate_json,'Original COCO annotation JSON')
REPO_SPLITS=Path(__file__).resolve().parents[1]/'data'/'splits'
if SPLIT_ROOT_OVERRIDE:PROCESSED_DIR=Path(SPLIT_ROOT_OVERRIDE)
elif (REPO_SPLITS/'physical_images_split.parquet').is_file() and (REPO_SPLITS/'annotation_sets_split.parquet').is_file():
    PROCESSED_DIR=REPO_SPLITS
else:
    pairs=defaultdict(list)
    for p in KAGGLE_INPUT.rglob('physical_images_split.parquet'):
        q=p.parent/'annotation_sets_split.parquet'
        if q.is_file():pairs[(sha256_file(p),sha256_file(q))].append(p.parent)
    if not pairs:
        archives=[]
        for p in KAGGLE_INPUT.rglob('*.zip'):
            try:
                with zipfile.ZipFile(p) as z:
                    if all(any(Path(n).name==key for n in z.namelist()) for key in ['physical_images_split.parquet','annotation_sets_split.parquet']):archives.append(p)
            except zipfile.BadZipFile:pass
        if len(archives)!=1:raise RuntimeError('Attach original seg-data with both split parquet tables, directly or in its original ZIP.')
        PROCESSED_DIR=OUT/'split_metadata';PROCESSED_DIR.mkdir(exist_ok=True)
        with zipfile.ZipFile(archives[0]) as z:
            for key in ['physical_images_split.parquet','annotation_sets_split.parquet']:
                names=[n for n in z.namelist() if Path(n).name==key];assert len(names)==1
                (PROCESSED_DIR/key).write_bytes(z.read(names[0]))
    else:
        assert len(pairs)==1,'Multiple different original splits found; set SPLIT_ROOT_OVERRIDE'
        PROCESSED_DIR=next(iter(pairs.values()))[0]
physical_df=pd.read_parquet(PROCESSED_DIR/'physical_images_split.parquet')
annotation_sets_df=pd.read_parquet(PROCESSED_DIR/'annotation_sets_split.parquet')
train_physical=physical_df[physical_df.split=='train'].reset_index(drop=True)
val_physical=physical_df[physical_df.split=='val'].reset_index(drop=True)
assert physical_df.file_name.is_unique
assert len(train_physical)==EXPECTED_TRAIN_IMAGES and len(val_physical)==EXPECTED_VAL_IMAGES
TRAIN_NAMES=train_physical.file_name.astype(str).tolist();VAL_NAMES=val_physical.file_name.astype(str).tolist()
assert set(TRAIN_NAMES).isdisjoint(VAL_NAMES)
assert set(annotation_sets_df.file_name)==set(physical_df.file_name)
for split,names in [('train',TRAIN_NAMES),('val',VAL_NAMES)]:
    assert (annotation_sets_df.loc[annotation_sets_df.file_name.isin(names),'split']==split).all()
annotations_by_file={str(n):g.reset_index(drop=True) for n,g in annotation_sets_df.groupby('file_name')}
SPLIT_DIGEST=hashlib.sha256((sha256_file(PROCESSED_DIR/'physical_images_split.parquet')+
    sha256_file(PROCESSED_DIR/'annotation_sets_split.parquet')+sha256_file(COCO_PATH)).encode()).hexdigest()
coco_image_by_id={im['id']:im for im in coco['images']}
coco_anns_by_image=defaultdict(list)
for a in coco['annotations']:coco_anns_by_image[a['image_id']].append(a)
def segmentation_to_mask(seg,h,w):
    if isinstance(seg,list):
        if not seg:return np.zeros((h,w),bool)
        r=mask_utils.merge(mask_utils.frPyObjects(seg,h,w))
    elif isinstance(seg,dict) and isinstance(seg.get('counts'),list):r=mask_utils.frPyObjects(seg,h,w)
    else:r=seg
    a=mask_utils.decode(r)
    if a.ndim==3:a=a.any(2)
    assert a.shape==(h,w)
    return a.astype(bool)
def build_gt_for_name(name):
    result=[]
    for rid in annotations_by_file[name].image_record_id:
        im=coco_image_by_id[rid];lab=np.zeros((int(im['height']),int(im['width'])),np.int32)
        anns=coco_anns_by_image.get(rid,[])
        for j,a in enumerate(anns,1):lab[segmentation_to_mask(a['segmentation'],*lab.shape)]=j
        result.append((str(rid),lab,len(anns)))
    return result
log(f'Original split retained: {len(TRAIN_NAMES)} training / {len(VAL_NAMES)} validation images. All annotators retained.')


def make_oof_manifest():
    ordered=physical_df.copy()
    ordered['timestamp']=pd.to_datetime(ordered.file_name.str[:14],format='%Y%m%d%H%M%S')
    ordered=ordered.sort_values('timestamp').reset_index(drop=True)
    # Form temporal clusters across ALL observations, before excluding validation.
    # Group assignment uses timestamps only, never mask values.
    ordered['time_cluster']=(ordered.timestamp.diff().isna() | (ordered.timestamp.diff()>pd.Timedelta(days=FOLD_EMBARGO_DAYS))).cumsum()
    source_col=next((c for c in ['group','temporal_group','group_id'] if c in ordered),None)
    parent={int(g):int(g) for g in ordered.time_cluster.unique()}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    if source_col:
        for _,g in ordered.dropna(subset=[source_col]).groupby(source_col):
            ids=list(map(int,g.time_cluster.unique()))
            for b in ids[1:]:parent[find(b)]=find(ids[0])
    ordered['group']=[find(int(g)) for g in ordered.time_cluster]
    roster=ordered[ordered.file_name.isin(TRAIN_NAMES)].reset_index(drop=True)
    assert roster.group.nunique()>=N_FOLDS,'Too few temporal training groups for requested folds; do not substitute a random component/image split.'
    assignment={};folds=[]
    for f,(tr,va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(roster,groups=roster.group)):
        fit=roster.iloc[tr].copy();hold=roster.iloc[va].copy()
        ta=fit.timestamp.to_numpy(dtype='datetime64[ns]');tb=hold.timestamp.to_numpy(dtype='datetime64[ns]')
        gap=np.min(np.abs(ta[:,None]-tb[None,:]),axis=1)/np.timedelta64(1,'D')
        excluded=fit.loc[gap<=FOLD_EMBARGO_DAYS];fit=fit.loc[gap>FOLD_EMBARGO_DAYS]
        assert len(fit)>0 and set(fit.file_name).isdisjoint(hold.file_name)
        assert not (set(fit.file_name)|set(hold.file_name))&set(VAL_NAMES)
        assert set(fit.group).isdisjoint(hold.group)
        for n in hold.file_name:assert n not in assignment;assignment[n]=f
        folds.append(dict(fold=f,seed=SEED+f,training_names=sorted(fit.file_name),heldout_names=sorted(hold.file_name),
                          embargo_names=sorted(excluded.file_name),initialization='DINOv3_ConvNeXtBase_LVD1689M',fixed_epochs=FOLD_EPOCHS,
                          checkpoint_selection='fixed_final_epoch55_EMA; no heldout/validation labels inspected'))
    assert set(assignment)==set(TRAIN_NAMES)
    hashes={}
    for j,n in enumerate(TRAIN_NAMES):
        hashes[n]=sha256_file(TRAIN_IMAGES_DIR/n)
        if (j+1)%50==0:log(f'Training image fingerprints {j+1}/{len(TRAIN_NAMES)}')
    body=dict(version=OOF_VERSION,split_digest=SPLIT_DIGEST,training_names=sorted(TRAIN_NAMES),
        validation_names=sorted(VAL_NAMES),folds=folds,image_hashes=hashes,hyperparameters=HP,
        n_folds=N_FOLDS,embargo_days=FOLD_EMBARGO_DAYS,model_code_digest=MODEL_CODE_DIGEST,
        note='Every OOF image is excluded from the model that predicts it. The original 141 validation images are excluded from ALL new model and filter fits. Prior review/selection still limits independence of final development evaluation.')
    body['signature']=digest_json(body)
    roster['fold']=[assignment[n] for n in roster.file_name]
    roster.to_csv(OUT/'training_fold_membership.csv',index=False)
    p=OUT/'oof_training_manifest.json'
    if p.exists():assert json.loads(p.read_text())['signature']==body['signature'],'Existing output uses different OOF settings; choose a fresh OUTPUT_ROOT.'
    atomic_json(p,body)
    return body
OOF_MANIFEST=make_oof_manifest()
display(pd.DataFrame([dict(fold=f['fold'],train=len(f['training_names']),heldout=len(f['heldout_names']),embargo=len(f['embargo_names']),epochs=f['fixed_epochs']) for f in OOF_MANIFEST['folds']]))


# DINOv3 ConvNeXt-Base encoder weights -> one plain state-dict file in OUT, hash-verified by the workers.
# Parent stays on CPU. Workers never need Internet or the token.
ENCODER_FILE=OUT/'dino_convnext_encoder.pth'
def _ensure_encoder():
    import timm
    global PRETRAINED_PATH
    if not PRETRAINED_PATH:
        found=[p for p in KAGGLE_INPUT.rglob('dino_convnext_encoder.pth')]
        if found:PRETRAINED_PATH=str(found[0])
    if PRETRAINED_PATH:
        path=Path(PRETRAINED_PATH);log(f'Using attached encoder file {path}')
    else:
        from kaggle_secrets import UserSecretsClient
        from huggingface_hub import login
        login(UserSecretsClient().get_secret(HF_SECRET_NAME))
        model=timm.create_model(ENCODER_TAG,pretrained=True,num_classes=0)
        cfg=model.pretrained_cfg
        # The model hard-codes ImageNet mean/std; the weights must expect the same.
        assert tuple(cfg['mean'])==(0.485,0.456,0.406) and tuple(cfg['std'])==(0.229,0.224,0.225),'normalisation of these weights differs from the model'
        path=ENCODER_FILE;torch.save(model.state_dict(),path);del model
    # Strict re-load into a fresh, randomly initialised model: file complete, correct architecture/tag.
    fresh=timm.create_model(ENCODER_TAG,pretrained=False,num_classes=0)
    fresh.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);del fresh
    return path,sha256_file(path)
if TRAIN_OOF_FROM_SCRATCH:
    PRETRAINED_FILE,PRETRAINED_SHA=_ensure_encoder()
    log(f'DINOv3 encoder {ENCODER_TAG} | {PRETRAINED_FILE} | sha256 {PRETRAINED_SHA[:12]}')
else:
    PRETRAINED_FILE,PRETRAINED_SHA='', ''
    log('Pretrained-fold mode: no gated DINOv3 encoder download is required.')



# ─── Original training data preparation (training mode only) ───────────────────
from src.oof_helpers import discover_fold_weights, verify_pair_completion, launch_pair_jobs
from src.oof_helpers import install_precomputed_maps, copy_resume, sha256_file

worker=OUT/'dino_oof_worker.py'
worker.write_text((Path(__file__).resolve().parent/'oof_training_worker.py').read_text())
SESSION_START=time.time();deadline=SESSION_START+SESSION_HOURS*3600
if not torch.cuda.is_available():raise RuntimeError('OOF training/inference requires a CUDA GPU')
if TRAIN_OOF_FROM_SCRATCH and torch.cuda.device_count()<2:
    raise RuntimeError('Original fold training was configured for T4 x2; select two GPUs for training mode.')

if TRAIN_OOF_FROM_SCRATCH:
    boxes={};cache=TRAIN_CACHE_ROOT
    for j,name in enumerate(TRAIN_NAMES):
        with Image.open(TRAIN_IMAGES_DIR/name) as im:raw=np.asarray(im.convert('L'))
        gts=build_gt_for_name(name)
        semantic=np.stack([(g>0).astype(np.uint8)*255 for _,g,_ in gts])
        np.save(cache/(name+'.image.npy'),raw,allow_pickle=False)
        np.save(cache/(name+'.semantic.npy'),semantic,allow_pickle=False)
        boxes[name]=[]
        for _,g,gn in gts:
            for cid,sl in enumerate(ndi.find_objects(g),1):
                if sl is not None:
                    yy,xx=np.nonzero(g[sl]==cid)
                    boxes[name].append((float(yy.mean()+sl[0].start),float(xx.mean()+sl[1].start),len(yy)))
        if (j+1)%100==0 or j+1==len(TRAIN_NAMES):log(f'Training data prepared {j+1}/{len(TRAIN_NAMES)}')
    if RESUME_ROOT:copy_resume(Path(RESUME_ROOT),OUT,OOF_MANIFEST,FOLDS_TO_TRAIN)
    launch_pair_jobs('train',worker,OUT,OOF_MANIFEST,FOLDS_TO_TRAIN,deadline,
                     TRAIN_CACHE_ROOT,TRAIN_IMAGES_DIR,PRETRAINED_FILE,PRETRAINED_SHA,boxes)
    for f in FOLDS_TO_TRAIN:
        if not (OUT/f'fold_{f}'/'run_complete.json').exists():
            raise RuntimeError(f'Fold {f} is not complete; resume from whole output. OOF inference was not run.')
    checkpoints={f:{'checkpoint':OUT/f'fold_{f}'/'fixed_final_ema.pth','format':'snapshot',
                    'sha256':sha256_file(OUT/f'fold_{f}'/'fixed_final_ema.pth')} for f in FOLDS_TO_TRAIN}
else:
    if RESUME_ROOT:copy_resume(Path(RESUME_ROOT),OUT,OOF_MANIFEST,FOLDS_TO_TRAIN)
    from src.hf_oof import acquire_oof_weights
    try:
        checkpoints=discover_fold_weights(MODEL_INPUT_ROOT,OOF_MANIFEST,FOLDS_TO_TRAIN,metadata_root=OUT/'oof_metadata')
    except FileNotFoundError:
        # Local artifacts take priority; incomplete sets can be completed from HF.
        if not OOF_HF_DOWNLOAD:
            raise
        hf_root=OUT/'hf_pretrained'
        log(f'Local OOF weights incomplete; fetching missing fold models from {OOF_HF_REPO}')
        acquire_oof_weights(hf_root,FOLDS_TO_TRAIN,repo_id=OOF_HF_REPO)
        # If a user supplies one fold locally, keep that fold; fetch only absent folds.
        # The staged remote set is independently and strictly verified.
        checkpoints={}
        for f in FOLDS_TO_TRAIN:
            try:
                checkpoints.update(discover_fold_weights(MODEL_INPUT_ROOT,OOF_MANIFEST,(f,),metadata_root=OUT/'oof_metadata'))
            except FileNotFoundError:
                checkpoints.update(discover_fold_weights(hf_root,OOF_MANIFEST,(f,)))
    for f,rec in checkpoints.items():log(f'Loaded fold {f} [{rec["format"]}] {rec["checkpoint"]}')

folds_needing_inference=[]
for f in FOLDS_TO_TRAIN:
    if (not TRAIN_OOF_FROM_SCRATCH and REUSE_PRECOMPUTED_MAPS and
        install_precomputed_maps(MODEL_INPUT_ROOT,OUT,OOF_MANIFEST,f,checkpoints[f])):
        log(f'Fold {f}: verified original OOF maps reused (no inference)')
    else:
        folds_needing_inference.append(f)

if folds_needing_inference:
    launch_pair_jobs('infer',worker,OUT,OOF_MANIFEST,tuple(folds_needing_inference),deadline,
                     TRAIN_CACHE_ROOT,TRAIN_IMAGES_DIR,PRETRAINED_FILE,PRETRAINED_SHA,{},checkpoints)
status=verify_pair_completion(OUT,OOF_MANIFEST,FOLDS_TO_TRAIN)
atomic_json(OUT/'pair_status.json',dict(folds=status,complete=all(x['complete'] for x in status),
                                      mode='train' if TRAIN_OOF_FROM_SCRATCH else 'pretrained'))
if not all(x['complete'] for x in status):
    raise RuntimeError('OOF pair is incomplete. Attach the entire saved output and rerun/resume. '+str(status))
print('OOF PAIR COMPLETE:',FOLDS_TO_TRAIN,'maps:',sum(x['heldout_images'] for x in status),'output:',OUT,flush=True)
