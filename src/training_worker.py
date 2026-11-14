"""Selected DINOv3 ConvNeXt-Base training worker.

Reconstructed from the original finalized 55-epoch EMA run; other backbone
experiments are excluded from this submission package.
"""
SOURCE_SHA='dino-convnext-v1 (baseline worker 35459712...)'
# Install requirements-train.txt before launching this worker.
import importlib.util, subprocess, sys
import json, random, time, os, gc, hashlib, zipfile, shutil, math, copy, platform
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import psutil
try:
    import optuna
except ImportError:
    optuna = None
try:
    from skimage.segmentation import watershed
except ImportError:
    watershed = None
print('torch',torch.__version__,'Optuna',getattr(optuna,'__version__','unavailable: seeded random fallback'),
      'watershed',watershed is not None)

from IPython.display import display


SEED = 42
TAG = 'EdgeAttNet_DUAL_v1'
KAGGLE_INPUT = Path('/kaggle/input')
WORKING_DIR = Path('/kaggle/working')
WORKING_DIR.mkdir(exist_ok=True,parents=True)
OUT = Path(os.environ.get('EDGE_TRAIN_OUTPUT_ROOT', str(WORKING_DIR / TAG)))
OUT.mkdir(exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Normal first run: same datasets, fresh weights, one 55+10 epoch run.
COMPLETED_RUN_ROOT = ''        # post_only: absolute attached output folder containing run folders
MODE = 'train_then_post'        # 'post_only' to reuse a completed run
RUN_SPECS = [dict(name='balanced', seed=42, recipe='balanced', norm='group')]
EPOCHS_DICE, EPOCHS_TVERSKY = 55, 0
CROP = 1024
ACCUM_STEPS = 2               # variable annotator counts keep physical batch = 1
LR_DICE, LR_TVERSKY = 2e-4, 2e-5
MIN_LR_RATIO = 0.10
WARMUP_EPOCHS = 3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
EMA_DECAY = 0.99
TOP_K_PER_RUN = 3             # snapshots of EMA and matching raw weights
MIN_SNAPSHOT_GAP = 3          # epochs, limits near-identical snapshots
KEEP_FROM_EPOCH = 12
# Fixed instance decoding for PQ monitoring only; never tuned in the epoch loop.
MONITOR_THRESHOLD = 0.70
MONITOR_MIN_AREA = 400
# EMA checkpoints/stage transitions use pooled-count PQ on tuning images only.
# V3 grouping, watershed, TTA and PP parameter search remain post-training.
BOUNDARY_BCE_GAIN = 1.0        # weight 2 in a 5px boundary band, 1 elsewhere
AUX_WEIGHT = 0.15             # two intermediate decoder heads: 0.075 each
SKELETON_RECALL_WEIGHT = 0.0  # optional 0.03 ablation; baseline prioritizes outlines
TV_ALPHA, TV_BETA = 0.4, 0.6

CACHE_TRAIN, CACHE_VAL = False, False
CACHE_RAM_RESERVE_GB = 5.0
NUM_WORKERS = 0
TRAIN_ENABLED = True
RESUME_RUN = ''              # run name + path to its last_resume.pth, if interrupted
RESUME_PATH = ''
RESUME_BEST_DIR = ''          # if checkpoint came from input, attach its snapshot folder
# Resume is epoch-exact with deterministic per-sample augmentation. CUDA kernels
# can still be nondeterministic; checkpoints record the full experiment signature.

EXTERNAL_CHECKPOINTS = []

# Validation is partitioned by contiguous time blocks: tuning and confirmation.
# Confirmation is never used for choosing thresholds, ensembles, TTA or checkpoints.
TUNE_FRACTION = 0.67
VAL_BLOCK_SIZE = 7
SEARCH_SEED = 2026
PP_TRIALS = 70                # AFTER training, on cached probabilities only
PP_TIMEOUT_SECONDS = 2400
COARSE_THRESHOLDS = (0.45, 0.60, 0.70, 0.80, 0.85, 0.90)
TTA_TOP_N = 3                 # compare matching single/ensemble with and without TTA
COMPARE_TILED = True          # compare full 2048 with overlapping 1024 inference
INFER_TILE, INFER_OVERLAP = 1024, 256
GENERATE_SUBMISSION = True
RUN_NOTEBOOK_SMOKE = False
BOOTSTRAP_REPEATS = 300
ZIP_WEIGHTS = False           # weights already saved; avoid duplicate multi-GB ZIP

# Moderate augmentation, D4 exact geometry. Real FOV zoom avoids artificial borders.
RECIPES = {
 'balanced': dict(scale_range=(0.80,1.20),p_zoom=0.45,p_photo=0.55,cloud_amp=0.08,
                  p_blur=0.06,p_noise=0.10,p_paste=0.0),
}
assert MODE in {'train_then_post','post_only'}
assert len({s['name'] for s in RUN_SPECS}) == len(RUN_SPECS)
assert CROP % 16 == 0 and INFER_TILE % 16 == 0
assert ACCUM_STEPS >= 1 and 0 <= INFER_OVERLAP < INFER_TILE

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def atomic_json(path,obj):
    tmp=Path(str(path)+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,default=str)); tmp.replace(path)
def atomic_torch(path,obj):
    tmp=Path(str(path)+'.tmp'); torch.save(obj,tmp); tmp.replace(path)
seed_everything(SEED)
torch.set_num_threads(2)
print('worker output',OUT,flush=True)
if device.type != 'cuda': print('GPU is required for the full training run; CPU supports smoke checks.')

SESSION_START=float(os.environ.get('DUAL_START',time.time()))
SESSION_BUDGET=float(os.environ.get('DUAL_BUDGET',9.5*3600))
ACTIVE_VARIANT=os.environ.get('DUAL_VARIANT','dino_convnext_base')
assert ACTIVE_VARIANT=='dino_convnext_base', 'Only the submitted DINOv3 ConvNeXt-Base variant is supported'
PRETRAINED_ENCODER_PATH=os.environ.get('DUAL_PRETRAINED','')
# The submission uses exactly one architecture and one training recipe.
RESUME_ROOT=os.environ.get('DUAL_RESUME','')
BOUNDARY_HEAD_WEIGHT=.10
ENCODER_LR_MULT=.10
class BudgetPause(Exception):pass


explicit_data_root = os.environ.get("EDGE_TRAIN_DATA_ROOT")
data_roots = ([Path(explicit_data_root)] if explicit_data_root else
              [p for p in KAGGLE_INPUT.rglob("MAGFiLO_1.0_Kaggle_2026") if p.is_dir()])
if len(data_roots) != 1:
    raise RuntimeError(f"Expected exactly one MAGFiLO_1.0_Kaggle_2026 folder, found {len(data_roots)}.")

DATA_ROOT = data_roots[0]
TRAIN_DIR = DATA_ROOT / "train"

train_image_files = sorted(
    p for p in TRAIN_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg", ".png"}
)
train_image_dirs = sorted({p.parent for p in train_image_files})
if len(train_image_dirs) != 1:
    raise RuntimeError(f"Expected one training-image directory, found: {train_image_dirs}")
TRAIN_IMAGES_DIR = train_image_dirs[0]

# Original competition COCO JSON, used for true per-filament PQ ground truth.
coco_candidates = []
for p in TRAIN_DIR.rglob("*.json"):
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and {"images", "annotations"}.issubset(obj):
            coco_candidates.append((p, obj))
    except Exception:
        pass

if len(coco_candidates) != 1:
    raise RuntimeError(
        f"Expected exactly one COCO annotation JSON under {TRAIN_DIR}, "
        f"found {[str(p) for p, _ in coco_candidates]}"
    )

COCO_PATH, coco = coco_candidates[0]

# Some Kaggle dataset versions attach the processed tables as a ZIP. Extract only when direct files are missing.
if not any(KAGGLE_INPUT.rglob('physical_images_split.parquet')):
    zip_candidates = [p for p in KAGGLE_INPUT.rglob('*.zip') if p.is_file()]
    found = []
    for archive in zip_candidates:
        try:
            with zipfile.ZipFile(archive) as z:
                members = z.namelist()
                if any(Path(x).name == 'physical_images_split.parquet' for x in members) and any(Path(x).name == 'annotation_sets_split.parquet' for x in members):
                    found.append(archive)
        except zipfile.BadZipFile:
            continue
    if len(found) == 1:
        out = WORKING_DIR / 'attached_preprocessing'
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(found[0]) as z:
            for filename in ('physical_images_split.parquet','annotation_sets_split.parquet'):
                paths = [x for x in z.namelist() if Path(x).name == filename]
                assert len(paths)==1, (filename, paths)
                (out / filename).write_bytes(z.read(paths[0]))
        print('Recovered saved split tables from ZIP:', found[0])
    elif len(found)>1:
        raise RuntimeError(f'Multiple ZIPs contain split metadata: {found}. Attach one original preprocessing output.')

# Processed output of 01_preprocessing.
working_processed = WORKING_DIR / "processed"
REPO_SPLITS = Path(__file__).resolve().parents[1] / 'data' / 'splits'
ENV_SPLITS = os.environ.get('EDGE_TRAIN_SPLITS_ROOT', '')
if ENV_SPLITS and (Path(ENV_SPLITS)/'physical_images_split.parquet').exists():
    PROCESSED_DIR = Path(ENV_SPLITS)
elif (REPO_SPLITS/'physical_images_split.parquet').exists():
    PROCESSED_DIR = REPO_SPLITS
elif ((WORKING_DIR / "attached_preprocessing" / "physical_images_split.parquet").exists()):
    PROCESSED_DIR = WORKING_DIR / "attached_preprocessing"
elif (working_processed / "physical_images_split.parquet").exists():
    PROCESSED_DIR = working_processed
else:
    split_files = list(KAGGLE_INPUT.rglob("physical_images_split.parquet"))
    if len(split_files)>1:
        grouped={(sha256_file(q),sha256_file(q.parent/'annotation_sets_split.parquet')) for q in split_files}
        if len(grouped)==1:split_files=[sorted(split_files)[0]]
    if len(split_files) != 1:
        raise RuntimeError(
            "Original physical_images_split.parquet NOT FOUND. "
            "Attach 01_preprocessing saved output (split tables + paired masks). "
            f"Matches found: {split_files}"
        )
    PROCESSED_DIR = split_files[0].parent

# Masks and split tables may be in SEPARATE Kaggle datasets.
# Kaggle labels (e.g. masks_, seg_data) do not guarantee matching disk slugs.
# Locate folders by their contents rather than guessing /kaggle/input/datasets/... paths.
mask_roots = set()
if os.environ.get('EDGE_TRAIN_MASKS_ROOT'):
    mask_roots.add(Path(os.environ['EDGE_TRAIN_MASKS_ROOT']))
for parent in (PROCESSED_DIR, PROCESSED_DIR / "masks", PROCESSED_DIR / "masks" / "masks"):
    if (parent / "semantic").is_dir() and (parent / "spine").is_dir():
        mask_roots.add(parent)
for semantic_dir in KAGGLE_INPUT.rglob("semantic"):
    if semantic_dir.is_dir() and (semantic_dir.parent / "spine").is_dir():
        mask_roots.add(semantic_dir.parent)

# Confirm a candidate with actual mask filenames from the metadata.
# This prevents accidentally taking a different attached dataset's masks.
_mask_metadata = pd.read_parquet(PROCESSED_DIR / "annotation_sets_split.parquet", columns=["mask_file"])
_mask_names = _mask_metadata["mask_file"].dropna().astype(str).tolist()
assert _mask_names, "No mask filenames in annotation_sets_split.parquet"
_probe = list(dict.fromkeys(
    [_mask_names[0], _mask_names[len(_mask_names)//2], _mask_names[-1]]
))
matching_mask_roots = [
    root for root in sorted(mask_roots)
    if all((root / sub / name).is_file() for name in _probe
           for sub in ("semantic", "spine"))
]
if len(matching_mask_roots) != 1:
    raise FileNotFoundError(
        "Expected exactly one mask dataset matching split metadata. "
        f"Candidate roots: {[str(p) for p in sorted(mask_roots)]}; "
        f"matching: {[str(p) for p in matching_mask_roots]}; "
        f"example mask: {_probe[0]}. "
        "Attach the dataset containing semantic/ and spine/ PNG folders."
    )
MASK_ROOT = matching_mask_roots[0]
SEMANTIC_MASK_DIR = MASK_ROOT / "semantic"
SPINE_MASK_DIR = MASK_ROOT / "spine"
missing_masks = [
    str(path) for name in set(_mask_names)
    for path in (SEMANTIC_MASK_DIR / name, SPINE_MASK_DIR / name)
    if not path.is_file()
]
if missing_masks:
    raise FileNotFoundError(
        f"{len(missing_masks)} mask paths are missing; examples: {missing_masks[:5]}"
    )
print("Split tables:", PROCESSED_DIR)
print("Masks root  :", MASK_ROOT)
print("Verified", len(set(_mask_names)), "paired semantic/spine masks.")

print("Training images:", TRAIN_IMAGES_DIR)
print("Processed data :", PROCESSED_DIR)
print("COCO annotations:", COCO_PATH.name)

physical_df = pd.read_parquet(PROCESSED_DIR / "physical_images_split.parquet")
annotation_sets_df = pd.read_parquet(PROCESSED_DIR / "annotation_sets_split.parquet")

train_physical = physical_df[physical_df["split"] == "train"].reset_index(drop=True)
val_physical = physical_df[physical_df["split"] == "val"].reset_index(drop=True)

annotations_by_file = {
    file_name: group.reset_index(drop=True)
    for file_name, group in annotation_sets_df.groupby("file_name")
}

# leakage guard: no image may appear on both sides
assert set(train_physical["file_name"]).isdisjoint(set(val_physical["file_name"])), "train/val overlap!"

print("Train physical images:", len(train_physical), "(expected 566)")
print("Val physical images  :", len(val_physical), "(expected 141)")
print("Annotation sets      :", annotation_sets_df["split"].value_counts().to_dict())
assert len(train_physical)==566 and len(val_physical)==141, 'This is NOT the original 566/141 split.'
assert len(annotation_sets_df)==1154, 'Annotation-set count differs from the original curated split.'
# Reconcile every annotator to its physical-image split, not only image names.
assert set(annotation_sets_df.file_name) == set(physical_df.file_name)
for split,table in [('train',train_physical),('val',val_physical)]:
    rows=annotation_sets_df[annotation_sets_df.file_name.isin(table.file_name)]
    assert (rows['split']==split).all(), 'Annotator split does not match physical split.'
assert physical_df.file_name.is_unique
# The supplied metadata defines the temporal split; this notebook never regenerates it.
SPLIT_DIGEST=hashlib.sha256((sha256_file(PROCESSED_DIR/'physical_images_split.parquet')+
                          sha256_file(PROCESSED_DIR/'annotation_sets_split.parquet')+sha256_file(COCO_PATH)).encode()).hexdigest()

VAL_IMAGE_DIGEST=hashlib.sha256("".join(sha256_file(TRAIN_IMAGES_DIR/n) for n in val_physical.file_name).encode()).hexdigest()


class FilamentDataset(Dataset):
    """One physical image with ALL its annotators; cache read-only uint8 arrays in RAM.

    A normal __getitem__ returns exactly the old float32 [0,1] tensor interface.
    get_crop() slices uint8 arrays BEFORE float32 conversion (much cheaper).
    """
    def __init__(self, physical_df, annotations_by_file, image_dir,
                 semantic_mask_dir, spine_mask_dir, cache=False):
        self.physical_df = physical_df.reset_index(drop=True)
        self.file_names = self.physical_df["file_name"].tolist()
        self.annotations_by_file = annotations_by_file
        self.image_dir = image_dir
        self.semantic_mask_dir = semantic_mask_dir
        self.spine_mask_dir = spine_mask_dir
        self.cache = None
        self.fg_coords = None
        if cache:
            self.build_cache()

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def _read_uint8(path, grayscale=False):
        with Image.open(path) as im:
            array = np.asarray(im.convert("L") if grayscale else im)
        if array.dtype != np.uint8 or array.ndim != 2:
            raise ValueError(f"Expected 2D uint8 image/mask at {path}, got {array.shape}/{array.dtype}")
        return array

    def _read_record(self, idx):
        name = self.file_names[idx]
        image = self._read_uint8(self.image_dir / name, grayscale=True)
        semantics, spines = [], []
        for mask_file in self.annotations_by_file[name]["mask_file"]:
            semantics.append(self._read_uint8(self.semantic_mask_dir / mask_file))
            spines.append(self._read_uint8((OUT/'boundary_targets'/mask_file) if ACTIVE_VARIANT=='boundary' and (OUT/'boundary_targets'/mask_file).is_file() else (self.spine_mask_dir/mask_file)))
        return image, np.stack(semantics), np.stack(spines)

    def build_cache(self):
        import psutil
        # Conservative estimate. 1 channel per physical image + 2 channels
        # per annotator record, 1 byte/pixel at 2048 squared.
        n_annotations = sum(len(self.annotations_by_file[name]) for name in self.file_names)
        approx_bytes = (len(self) + 2 * n_annotations) * (2048 ** 2)
        available = psutil.virtual_memory().available
        reserve = CACHE_RAM_RESERVE_GB * 1024 ** 3
        if available < approx_bytes + reserve:
            print(
                f"Estimated cache {approx_bytes / 2**30:.1f} GiB, available "
                f"{available / 2**30:.1f} GiB; needs {CACHE_RAM_RESERVE_GB:.1f} GiB "
                "free reserve. Falling back to uncached disk reads."
            )
            return
        t0 = time.perf_counter()
        self.cache = []
        self.fg_coords = []
        total_bytes = 0
        for i in range(len(self)):
            image, semantic, spine = self._read_record(i)
            # True foreground pixel positions. Same row-major sampling distribution
            # as (semantic_masks>0.5).any(0).nonzero() in the old cropper.
            yy, xx = np.nonzero(np.any(semantic > 127.5, axis=0))
            yy, xx = yy.astype(np.int32), xx.astype(np.int32)
            self.cache.append((image, semantic, spine))
            self.fg_coords.append((yy, xx))
            total_bytes += image.nbytes + semantic.nbytes + spine.nbytes + yy.nbytes + xx.nbytes
        print(f"Cached {len(self)} images + annotators in "
              f"{time.perf_counter()-t0:.1f}s; {total_bytes / 2**30:.2f} GiB of uint8 arrays/coords")

    def _record(self, idx):
        return self.cache[idx] if self.cache is not None else self._read_record(idx)

    @staticmethod
    def _to_float(sample):
        return torch.from_numpy(np.asarray(sample, dtype=np.float32) / 255.0)

    def __getitem__(self, idx):
        image, semantics, spines = self._record(idx)
        return {
            "image": self._to_float(image).unsqueeze(0),
            "semantic_masks": self._to_float(semantics),
            "spine_masks": self._to_float(spines),
            "file_name": self.file_names[idx],
        }

    def get_crop(self, idx, crop, p_fg):
        image, semantics, spines = self._record(idx)
        H, W = image.shape
        if crop > H or crop > W:
            raise ValueError(f"crop={crop} exceeds {(H,W)}")

        top = random.randint(0, H - crop)
        left = random.randint(0, W - crop)
        if random.random() < p_fg:
            if self.fg_coords is None:
                ys, xs = np.nonzero(np.any(semantics > 127.5, axis=0))
            else:
                ys, xs = self.fg_coords[idx]
            if len(ys):
                k = random.randrange(len(ys))
                top = int(min(max(int(ys[k]) - random.randint(0, crop - 1), 0), H - crop))
                left = int(min(max(int(xs[k]) - random.randint(0, crop - 1), 0), W - crop))

        sl = (slice(top, top + crop), slice(left, left + crop))
        return {
            "image": self._to_float(image[sl]).unsqueeze(0),
            "semantic_masks": self._to_float(semantics[:, sl[0], sl[1]]),
            "spine_masks": self._to_float(spines[:, sl[0], sl[1]]),
            "file_name": self.file_names[idx],
        }


# Compact train-only instance metadata. No full instance-map cache required.
coco_image_by_id = {im['id']:im for im in coco['images']}
coco_anns_by_image = defaultdict(list)
for a in coco['annotations']: coco_anns_by_image[a['image_id']].append(a)
instance_boxes = {}
for name in train_physical.file_name:
    items=[]
    for rid in annotations_by_file[name].image_record_id:
        for ann in coco_anns_by_image.get(rid,[]):
            x,y,w,h = ann['bbox']
            if w>0 and h>0:
                items.append((y+h/2,x+w/2,float(ann.get('area',w*h))))
    instance_boxes[name] = items

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

def make_dataset(physical,cache=False):
    return FilamentDataset(physical,annotations_by_file,TRAIN_IMAGES_DIR,
                           SEMANTIC_MASK_DIR,SPINE_MASK_DIR,cache=cache)
train_dataset=make_dataset(train_physical,CACHE_TRAIN)
val_dataset=make_dataset(val_physical,CACHE_VAL)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=NUM_WORKERS,
                      pin_memory=device.type=='cuda',worker_init_fn=seed_worker)


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





def main_logits(output):
    return output['logits'] if isinstance(output,dict) else output




class ConvNeXtEncoder(nn.Module):
    """DINOv3 ConvNeXt-Base encoder with four multiscale feature adapters.

    stage | stride | ConvNeXt ch -> adapter ch
    0     | 4      | 128 -> 64                  | s1
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
        # Input channels of the 4 stages, read from the model itself, so the same
        # The four stage dimensions come from the selected Base encoder metadata.
        in_chs = [f["num_chs"] for f in self.body.feature_info]
        out_chs = [64, 128, 256, 512]   # what the decoder expects (unchanged)
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
    """DINOv3 ConvNeXt-Base encoder and edge-guided decoder.

    Four adapted encoder stages feed two edge-guided attention blocks,
    a full-resolution skip, and deep-supervision outputs.
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

ACTIVE_VARIANT='dino_convnext_base'
def build_model(norm='group', aux=True):
    assert norm == 'group' and aux
    return ConvNeXtEdgeAttNet('convnext_base.dinov3_lvd1689m')


def boundary_loss(logits,target):
    y=target.float();z=logits.float().expand_as(y)
    positives=y.sum((-2,-1),keepdim=True)
    negatives=(1-y).sum((-2,-1),keepdim=True)
    pos_weight=(negatives/positives.clamp_min(1)).clamp(1,20)
    weights=1+(pos_weight-1)*y
    bce=(F.binary_cross_entropy_with_logits(z,y,reduction='none')*weights).sum((-2,-1))/weights.sum((-2,-1)).clamp_min(1)
    p=z.sigmoid();dice=1-(2*(p*y).sum((-2,-1))+1e-6)/(p.sum((-2,-1))+y.sum((-2,-1))+1e-6)
    # Empty annotations receive BCE supervision; avoid empty-mask Dice pathologies.
    valid=(positives[...,0,0]>0).float()
    return (.5*bce+.5*dice*valid).mean()


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
    if isinstance(output,dict) and 'boundary_logits' in output:
        loss=loss+BOUNDARY_HEAD_WEIGHT*boundary_loss(output['boundary_logits'],spines)
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


S = np.ones((3, 3))                                    # 8-connectivity


def pq_from_counts(sum_iou, tp, fp, fn):
    d = tp + 0.5 * fp + 0.5 * fn
    return sum_iou / d if d > 0 else float("nan")


def counts_lab(pl, pn, gl, gn):
    """(sum_iou, tp, fp, fn) for one labelled prediction vs one labelled ground truth."""
    if pn == 0 or gn == 0:
        return 0.0, 0, pn, gn
    both = (pl > 0) & (gl > 0)
    inter = np.bincount(gl[both].astype(np.int64) * (pn + 1) + pl[both],
                        minlength=(gn + 1) * (pn + 1)).reshape(gn + 1, pn + 1)
    ap = np.bincount(pl.ravel(), minlength=pn + 1)
    ag = np.bincount(gl.ravel().astype(np.int64), minlength=gn + 1)
    iou = (inter / np.maximum(ag[:, None] + ap[None, :] - inter, 1))[1:, 1:]
    m = iou > 0.5                                      # IoU > 0.5 makes matches unique
    tp = int(m.sum())
    return float(iou[m].sum()), tp, pn - tp, gn - tp


def pq_counts(pred_bin, gt_bin):
    """Same as counts_lab, starting from two binary masks."""
    pl, pn = ndi.label(pred_bin, structure=S)
    gl, gn = ndi.label(gt_bin, structure=S)
    return counts_lab(pl, pn, gl, gn)


def clean(pb, min_size):
    """Label a binary prediction and drop blobs smaller than min_size pixels."""
    pl, pn = ndi.label(pb, structure=S)
    if pn == 0:
        return pl, 0
    keep = np.bincount(pl.ravel(), minlength=pn + 1) >= min_size
    keep[0] = False
    remap = np.cumsum(keep) * keep                     # kept blobs get ids 1..k
    return remap[pl], int(keep.sum())

coco_image_by_id = {im["id"]: im for im in coco["images"]}

coco_anns_by_image = defaultdict(list)
for ann in coco["annotations"]:
    coco_anns_by_image[ann["image_id"]].append(ann)


def segmentation_to_mask(segmentation, height, width):
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation,dict) and isinstance(segmentation.get('counts'),list):
        rle=mask_utils.frPyObjects(segmentation,height,width)
    else:
        rle = segmentation
    return mask_utils.decode(rle).astype(bool)


def build_instance_label(image_record_id):
    im = coco_image_by_id[image_record_id]
    height, width = int(im["height"]), int(im["width"])
    anns = coco_anns_by_image.get(image_record_id, [])

    label = np.zeros((height, width), dtype=np.int16)

    for instance_id, ann in enumerate(anns, start=1):
        mask = segmentation_to_mask(ann["segmentation"], height, width)
        label[mask] = instance_id

    return label, len(anns)


def build_gt_labels(physical_table):
    labels = []

    for file_name in physical_table["file_name"]:
        rows = annotations_by_file[file_name]
        per_annotator = []

        for _, row in rows.iterrows():
            image_record_id = row["image_record_id"]
            per_annotator.append(build_instance_label(image_record_id))

        labels.append(per_annotator)

    return labels


t0 = time.time()
gt_lab = build_gt_labels(val_physical)

assert len(gt_lab) == len(val_dataset)
print(
    f"{len(gt_lab)} validation images built in {time.time() - t0:.0f}s | "
    f"{sum(len(x) for x in gt_lab)} annotator sets | "
    f"{sum(gn for gls in gt_lab for _, gn in gls)} filament instances"
)# Day groups if filename dates are available, otherwise contiguous filename blocks.
# This is an internal tuning/confirmation partition, NOT a new train/val split.
import re
names=val_physical.file_name.astype(str).tolist()
def date_key(name):
    m=re.search(r'(?:19|20)\d{6}',name)
    return m.group(0) if m else name
keys=sorted({date_key(n) for n in names})
block_for={k:i//VAL_BLOCK_SIZE for i,k in enumerate(keys)}
blocks=sorted(set(block_for.values()))
rng=np.random.default_rng(SEARCH_SEED); order=rng.permutation(blocks)
nt=max(1,min(len(blocks)-1,round(len(blocks)*TUNE_FRACTION)))
assert len(blocks)>=2,'Need at least two validation blocks for confirmation.'
tune_blocks=set(order[:nt].tolist())
TUNE_IDX=[i for i,n in enumerate(names) if block_for[date_key(n)] in tune_blocks]
CONFIRM_IDX=[i for i in range(len(names)) if i not in set(TUNE_IDX)]
assert set(TUNE_IDX).isdisjoint(CONFIRM_IDX) and len(TUNE_IDX)+len(CONFIRM_IDX)==len(names)
assert len(CONFIRM_IDX)>=10,'Confirmation partition too small; reduce VAL_BLOCK_SIZE.'
CONFIRM_BLOCKS=[block_for[date_key(names[i])] for i in CONFIRM_IDX]
atomic_json(OUT/(os.environ.get('DUAL_VARIANT','boundary')+'_validation_partition.json'),dict(tune_files=[names[i] for i in TUNE_IDX],
           confirmation_files=[names[i] for i in CONFIRM_IDX],seed=SEARCH_SEED,block_size=VAL_BLOCK_SIZE,
           note='Confirmation not used in any training selection or PP/TTA/ensemble tuning. Nearby dates may remain correlated.'))
print('Tuning images',len(TUNE_IDX),'confirmation images',len(CONFIRM_IDX))


def descriptors_from_labels(labels, n):
    desc = {}
    objects = ndi.find_objects(labels)
    for cid, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        yy, xx = np.nonzero(labels[sl] == cid)
        yy = yy.astype(np.float64) + sl[0].start
        xx = xx.astype(np.float64) + sl[1].start
        pts = np.column_stack([yy, xx])
        center = pts.mean(axis=0)

        if len(pts) < 3:
            axis = np.array([0.0, 1.0])
            pmin = pmax = 0.0
        else:
            centered = pts - center
            cov = centered.T @ centered / max(len(pts)-1, 1)
            vals, vecs = np.linalg.eigh(cov)
            axis = vecs[:, np.argmax(vals)]
            proj = centered @ axis
            pmin, pmax = float(proj.min()), float(proj.max())

        axis = axis / (np.linalg.norm(axis) + 1e-9)
        e1 = center + axis * pmin
        e2 = center + axis * pmax
        desc[cid] = (axis, e1, e2)
    return desc


def acute_angle_deg(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    d = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def line_samples(prob_support, p0, p1):
    dist = float(np.linalg.norm(p1 - p0))
    steps = max(2, int(np.ceil(dist)) + 1)
    yy = np.rint(np.linspace(p0[0], p1[0], steps)).astype(np.int32)
    xx = np.rint(np.linspace(p0[1], p1[1], steps)).astype(np.int32)
    yy = np.clip(yy, 0, prob_support.shape[0]-1)
    xx = np.clip(xx, 0, prob_support.shape[1]-1)
    return prob_support[yy, xx]


def build_pair_features(labels, n, prob=None):
    desc = descriptors_from_labels(labels, n)
    ids = sorted(desc)
    features = []

    support = None
    if prob is not None:
        support = ndi.maximum_filter(prob.astype(np.float32), size=3)

    for ii in range(len(ids)):
        a = ids[ii]
        va, a1, a2 = desc[a]

        for jj in range(ii+1, len(ids)):
            b = ids[jj]
            vb, b1, b2 = desc[b]

            tilt = acute_angle_deg(va, vb)
            endpoint_pairs = [(a1,b1),(a1,b2),(a2,b1),(a2,b2)]
            distances = [float(np.linalg.norm(q-p)) for p,q in endpoint_pairs]
            k = int(np.argmin(distances))
            dist = distances[k]

            p0, q0 = endpoint_pairs[k]
            connector = q0 - p0
            if np.linalg.norm(connector) < 1e-6:
                connect_angle = 0.0
            else:
                connect_angle = max(
                    acute_angle_deg(va, connector),
                    acute_angle_deg(vb, connector)
                )

            bridge_mean = np.nan
            bridge_frac = np.nan
            if support is not None:
                vals = line_samples(support, p0, q0)
                bridge_mean = float(vals.mean()) if len(vals) else 0.0
                bridge_frac = float((vals >= 0.15).mean()) if len(vals) else 0.0

            features.append(
                (a, b, dist, tilt, connect_angle, bridge_mean, bridge_frac)
            )
    return features


def merge_from_features(labels, n, pair_features,
                        max_dist, max_tilt, max_connect_angle,
                        bridge_mean_min=None, bridge_frac_min=None):
    if n <= 1:
        return labels.astype(np.int32, copy=False), int(n)

    parent = list(range(n+1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, dist, tilt, conn, bridge_mean, bridge_frac in pair_features:
        if dist > max_dist or tilt > max_tilt or conn > max_connect_angle:
            continue
        if bridge_mean_min is not None and bridge_mean < bridge_mean_min:
            continue
        if bridge_frac_min is not None and bridge_frac < bridge_frac_min:
            continue
        union(a, b)

    roots = {}
    mapping = np.zeros(n+1, dtype=np.int32)
    next_id = 0

    for cid in range(1, n+1):
        root = find(cid)
        if root not in roots:
            next_id += 1
            roots[root] = next_id
        mapping[cid] = roots[root]

    return mapping[labels], next_id


def instance_stats(labels, n, prob):
    sizes = np.bincount(labels.ravel(), minlength=n+1)
    elong = np.ones(n+1, dtype=np.float32)
    mean_conf = np.zeros(n+1, dtype=np.float32)

    if n:
        ids = np.arange(1, n+1)
        mean_conf[1:] = ndi.mean(prob.astype(np.float32), labels=labels, index=ids)

    objects = ndi.find_objects(labels)
    for cid, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        yy, xx = np.nonzero(labels[sl] == cid)
        if len(xx) < 3:
            elong[cid] = 1.0
            continue

        pts = np.column_stack([yy.astype(np.float64), xx.astype(np.float64)])
        pts -= pts.mean(axis=0, keepdims=True)
        cov = pts.T @ pts / max(len(pts)-1, 1)
        vals = np.linalg.eigvalsh(cov)
        vals = np.maximum(vals, 1e-6)
        elong[cid] = float(np.sqrt(vals[-1] / vals[0]))

    return sizes, elong, mean_conf



def relabel_keep(labels, keep):
    keep = np.array(keep, dtype=bool, copy=True)
    keep[0] = False
    remap = np.cumsum(keep)*keep
    return remap[labels].astype(np.int32,copy=False), int(keep.sum())

def evaluate_one_label(labels,n,gt_image):
    total=np.zeros(4,dtype=np.float64)
    for gl,gn in gt_image:
        total+=counts_lab(labels,n,gl,gn)
    return total

def metrics_from_counts(total):
    s,tp,fp,fn=total
    d=tp+0.5*fp+0.5*fn
    return dict(pq=float(s/d) if d else 0.0,
                sq=float(s/tp) if tp else 0.0,
                rq=float(tp/d) if d else 0.0,
                tp=int(tp),fp=int(fp),fn=int(fn))# Same PCA endpoints and angle rules, but only construct nearby pairs.
from scipy.spatial import cKDTree
def build_pair_features_bounded(labels,n,prob,max_dist):
    desc=descriptors_from_labels(labels,n)
    if len(desc)<2:return []
    points=[];owners=[]
    for cid,(_,p1,p2) in desc.items():
        points.extend([p1,p2]);owners.extend([cid,cid])
    pairs={tuple(sorted((owners[i],owners[j]))) for i,j in cKDTree(points).query_pairs(max_dist)
           if owners[i]!=owners[j]}
    support=ndi.maximum_filter(prob.astype(np.float32),size=3) if prob is not None else None
    features=[]
    for a,b in sorted(pairs):
        va,a1,a2=desc[a];vb,b1,b2=desc[b]
        endpoints=[(a1,b1),(a1,b2),(a2,b1),(a2,b2)]
        distances=[float(np.linalg.norm(q-p)) for p,q in endpoints]
        k=int(np.argmin(distances));dist=distances[k];p0,q0=endpoints[k]
        connector=q0-p0
        connect=0. if np.linalg.norm(connector)<1e-6 else max(acute_angle_deg(va,connector),acute_angle_deg(vb,connector))
        mean=frac=np.nan
        if support is not None:
            vals=line_samples(support,p0,q0);mean=float(vals.mean());frac=float((vals>=.15).mean())
        features.append((a,b,dist,acute_angle_deg(va,vb),connect,mean,frac))
    return features


DEFAULT_PP=dict(threshold=.70,min_area=400,compact=False,compact_area=500,min_elong=2.75,
                method='cc',grow_delta=.10,seed_threshold=.85,seed_area=60,
                merge_dist=0,merge_tilt=45,merge_connect=90,bridge_mean=.20,
                close_radius=0,open_radius=0,hole_area=0,small_conf=0.0)

def disk_structure(radius):
    yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
    return yy*yy+xx*xx <= radius*radius

def process_probability(prob,config):
    c={**DEFAULT_PP,**config};p=np.asarray(prob,dtype=np.float32)
    binary=p>c['threshold']
    if c['method']=='hysteresis':
        # Select whole low-threshold components containing high-confidence seeds.
        support=p>max(.05,c['threshold']-c['grow_delta'])
        labels,n=ndi.label(support,structure=S)
        has_seed=np.zeros(n+1,dtype=bool);has_seed[np.unique(labels[binary])]=True;has_seed[0]=False
        binary=has_seed[labels]
    if c['close_radius']:binary=ndi.binary_closing(binary,structure=disk_structure(c['close_radius']))
    if c['open_radius']:binary=ndi.binary_opening(binary,structure=disk_structure(c['open_radius']))
    if c['hole_area']:
        holes=ndi.binary_fill_holes(binary)&~binary;hl,hn=ndi.label(holes,structure=S)
        sizes=np.bincount(hl.ravel());keep=sizes<=c['hole_area'];keep[0]=False
        binary=binary|keep[hl]
    labels,n=ndi.label(binary,structure=S)
    if c['method']=='watershed' and watershed is not None and n:
        seeds=(p>max(c['seed_threshold'],c['threshold']+.02))&binary
        markers,m=clean(seeds,c['seed_area'])
        # Every low-confidence base component receives a seed, so it is not erased.
        next_id=m
        for cid,sl in enumerate(ndi.find_objects(labels),1):
            if sl is None:continue
            region=labels[sl]==cid
            if not (markers[sl][region]>0).any():
                coords=np.argwhere(region);k=int(np.argmax(p[sl][region]))
                yy,xx=coords[k]+np.array([sl[0].start,sl[1].start]);next_id+=1;markers[yy,xx]=next_id
        labels=watershed(-p,markers=markers,mask=binary,connectivity=S,watershed_line=False).astype(np.int32)
        n=next_id
    # Plain/legacy path filters before merging exactly as the original V3 did.
    areas=np.bincount(labels.ravel(),minlength=n+1)
    labels,n=relabel_keep(labels,areas>=c['min_area'])
    if c['merge_dist'] and n>1:
        # Probability-supported PCA endpoint grouping; no pixels painted across gaps.
        features=build_pair_features_bounded(labels,n,p if c['bridge_mean']>0 else None,c['merge_dist'])
        labels,n=merge_from_features(labels,n,features,c['merge_dist'],c['merge_tilt'],c['merge_connect'],
                                    bridge_mean_min=c['bridge_mean'] if c['bridge_mean']>0 else None)
    if (c['compact'] or c['small_conf']>0) and n:
        areas,elong,conf=instance_stats(labels,n,p)
        keep=np.ones(n+1,dtype=bool);keep[0]=False
        if c['compact']:keep &= ~((areas<c['compact_area'])&(elong<c['min_elong']))
        if c['small_conf']>0:keep &= ~((areas<800)&(conf<c['small_conf']))
        labels,n=relabel_keep(labels,keep)
    return labels,n

# This metric is the supplied notebook's pooled-count PQ over annotators.
# Confirm official competition aggregation separately before comparing its absolute
# value to the leaderboard. All revisions retain this local metric for comparisons.
def score_paths(paths,config,indices,weights=None,return_counts=False):
    if paths and isinstance(paths[0],(str,Path)):paths=[paths]
    weights=np.asarray(weights if weights is not None else np.ones(len(paths)),dtype=float)
    assert weights.shape==(len(paths),) and np.all(weights>=0) and weights.sum()>0
    weights=weights/weights.sum();counts=[]
    for i in indices:
        p=sum(w*np.load(group[i]).astype(np.float32) for w,group in zip(weights,paths) if w>0)
        labels,n=process_probability(p,config);counts.append(evaluate_one_label(labels,n,gt_lab[i]))
    counts=np.asarray(counts)
    result=metrics_from_counts(counts.sum(0))
    return (result,counts) if return_counts else result


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

@torch.inference_mode()
def monitor_epoch(net):
    net.eval();soft=[];hard=[];bces=[];totals=np.zeros(4,dtype=np.float64)
    tune_indices=set(TUNE_IDX)
    for i,b in enumerate(val_loader):
        if i not in tune_indices: continue
        if i%25==0:print('monitor validation index',i,flush=True)
        x=b['image'].to(device,non_blocking=True)
        p=predict_probability(net,x)[0,0].cpu().numpy()
        # Fixed, lightweight decoding; no geometry or threshold/area search here.
        labels,n=clean(p>MONITOR_THRESHOLD,MONITOR_MIN_AREA)
        totals+=evaluate_one_label(labels,n,gt_lab[i])
        y=b['semantic_masks'].numpy()[0]
        inter=(p[None]*y).sum((1,2))
        soft.append(float(((2*inter+1e-6)/(p.sum()+y.sum((1,2))+1e-6)).mean()))
        pred=p>.5
        hard.append(float(((2*(pred[None]*y).sum((1,2))+1e-6)/(pred.sum()+y.sum((1,2))+1e-6)).mean()))
        clipped=np.clip(p,1e-6,1-1e-6)
        bces.append(float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean()))
    assert soft, 'No tuning validation images were evaluated.'
    metrics=metrics_from_counts(totals)
    return dict(selection_score=metrics['pq'],**metrics,val_soft_dice=float(np.mean(soft)),
                val_hard_dice=float(np.mean(hard)),val_bce=float(np.mean(bces)))


def config_signature(spec):
    config=dict(spec=spec,recipe=RECIPES[spec['recipe']],split=SPLIT_DIGEST,
                epochs=[EPOCHS_DICE,EPOCHS_TVERSKY],crop=CROP,accum=ACCUM_STEPS,
                lr=[LR_DICE,LR_TVERSKY],min_lr=MIN_LR_RATIO,warmup=WARMUP_EPOCHS,
                weight_decay=WEIGHT_DECAY,grad_clip=GRAD_CLIP,ema=EMA_DECAY,
                aux=AUX_WEIGHT,boundary=BOUNDARY_BCE_GAIN,skeleton=SKELETON_RECALL_WEIGHT,
                tv=[TV_ALPHA,TV_BETA],selection="tune_fixed_CC_PQ",
                variant=ACTIVE_VARIANT,boundary_head_weight=BOUNDARY_HEAD_WEIGHT,encoder_lr_mult=ENCODER_LR_MULT,pretrained_sha=os.environ.get('DUAL_PRETRAIN_SHA',''),source=SOURCE_SHA,
                monitor_threshold=MONITOR_THRESHOLD,monitor_min_area=MONITOR_MIN_AREA)
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()

def train_run(spec):
    run_dir=OUT/spec['name'];run_dir.mkdir(exist_ok=True)
    sig=config_signature(spec);seed_everything(spec['seed'])
    net=build_model(spec['norm'],aux=True)
    if RESUME_RUN!=spec['name']:
        # DINOv3 encoder: the file must exist, match its recorded SHA-256 and load strictly (no random fallback).
        assert Path(PRETRAINED_ENCODER_PATH).is_file(),'Missing verified pretrained encoder file'
        assert sha256_file(PRETRAINED_ENCODER_PATH)==os.environ['DUAL_PRETRAIN_SHA'],'Encoder file does not match its recorded SHA-256'
        state=torch.load(PRETRAINED_ENCODER_PATH,map_location='cpu',weights_only=True)
        net.encoder.load_pretrained(state);del state
    net=net.to(device);ema=ModelEMA(net)
    ds=TargetedCropDataset(train_dataset,spec)
    history=[];snapshots=[];best={'dice':-1.,'tversky':-1.};saved=None
    if RESUME_RUN==spec['name']:
        assert Path(RESUME_PATH).is_file(),'Set RESUME_PATH to last_resume.pth'
        saved=torch.load(RESUME_PATH,map_location='cpu',weights_only=False)
        assert saved['signature']==sig,'Resume config/split differs; use a new experiment.'
        net.load_state_dict(saved['model']);ema.module.load_state_dict(saved['ema'])
        ema.updates=saved['ema_updates'];history=saved['history'];snapshots=saved['snapshots'];best=saved['best']
        if RESUME_BEST_DIR:
            for p in Path(RESUME_BEST_DIR).glob('*.pth'):
                if p.name!='last_resume.pth':shutil.copy2(p,run_dir/p.name)
        for snap in snapshots:
            for kind in ['ema','raw']: assert (run_dir/snap[kind]).is_file(),'Attach snapshot folder for resume'
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def store_weights(path,state,ep,stage,pq,kind):
        atomic_torch(path,dict(model=state,norm=spec['norm'],aux=True,run=spec['name'],kind=kind,
                              epoch=ep,stage=stage,monitor_score=pq,monitor_pq=pq,
                              selection_metric='tune_fixed_CC_PQ',monitor_model='ema',
                              monitor_config=dict(threshold=MONITOR_THRESHOLD,min_area=MONITOR_MIN_AREA),
                              signature=sig,split=SPLIT_DIGEST,variant=ACTIVE_VARIANT,boundary_head_weight=BOUNDARY_HEAD_WEIGHT,source=SOURCE_SHA))
    for stage,epochs,base_lr in [('dice',EPOCHS_DICE,LR_DICE),('tversky',EPOCHS_TVERSKY,LR_TVERSKY)]:
        if not epochs:continue
        if saved and saved['stage']=='tversky' and stage=='dice': continue
        stage_resume=bool(saved and saved['stage']==stage)
        if stage=='tversky' and not stage_resume:
            p=run_dir/'dice_best_ema.pth';assert p.exists()
            net.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model'])
            ema=ModelEMA(net)
            print('Fine-tune from best-PQ Dice-stage EMA; reduce augmentation for final shape fitting.')
        # Low learning rate for the pretrained encoder; standard rate for new decoder/adapters.
        enc=list(net.encoder.body.parameters());ids={id(v) for v in enc}
        groups=[dict(params=enc,lr=base_lr*ENCODER_LR_MULT),
                dict(params=[v for v in net.parameters() if id(v) not in ids],lr=base_lr)]
        optimizer=torch.optim.AdamW(groups,lr=base_lr,weight_decay=WEIGHT_DECAY)
        updates_per_epoch=math.ceil(len(ds)/ACCUM_STEPS)
        total_steps=updates_per_epoch*epochs
        warm_steps=updates_per_epoch*(WARMUP_EPOCHS if stage=='dice' else 1)
        def lr_factor(step):
            if step<warm_steps:return .1+.9*step/max(warm_steps,1)
            progress=min(1,(step-warm_steps)/max(1,total_steps-warm_steps))
            return MIN_LR_RATIO+(1-MIN_LR_RATIO)*.5*(1+math.cos(math.pi*progress))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
        scaler=torch.amp.GradScaler('cuda',enabled=True)
        start=1
        if stage_resume:
            optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler']);start=saved['stage_epoch']+1
        for ep in range(start,epochs+1):
            estimate=max(900.,history[-1]['seconds']*1.3) if history else 1800.
            if time.time()-SESSION_START+estimate+600>SESSION_BUDGET:
                raise BudgetPause('Paused before next epoch; use last_resume.pth to continue.')
            t0=time.perf_counter();glob_ep=ep+(EPOCHS_DICE if stage=='tversky' else 0)
            # No worker persists across epochs: ds.set_epoch is visible in workers.
            strength=min(1,.3+.7*ep/10) if stage=='dice' else .35
            ds.set_epoch(glob_ep,strength)
            loader=DataLoader(ds,batch_size=1,shuffle=True,num_workers=NUM_WORKERS,
                              generator=torch.Generator().manual_seed(spec['seed']+glob_ep),
                              worker_init_fn=seed_worker,pin_memory=True,persistent_workers=False)
            print(f'{spec["name"]}: starting epoch {ep}/{epochs}',flush=True)
            net.train();optimizer.zero_grad(set_to_none=True);losses=[]
            for step,b in enumerate(loader):
                x=b['image'].to(device,non_blocking=True);y=b['semantic_masks'].to(device,non_blocking=True)
                sk=b['spine_masks'].to(device,non_blocking=True)
                with torch.autocast('cuda',dtype=torch.float16):output=net(x)
                loss=train_loss(output,y,sk,stage,glob_ep)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; stopped before saving.')
                if step%100==0:print(f'{spec["name"]} epoch {ep}: batch {step+1}/{len(loader)} loss {float(loss):.4f}',flush=True)
                # Correct normalization for the short final accumulation group.
                first=(step//ACCUM_STEPS)*ACCUM_STEPS
                divisor=min(ACCUM_STEPS,len(loader)-first)
                scaler.scale(loss/divisor).backward();losses.append(float(loss.detach()))
                if (step+1)%ACCUM_STEPS==0 or step+1==len(loader):
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(net.parameters(),GRAD_CLIP)
                    old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()>=old_scale:ema.update(net);scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            del x,y,sk,output,loss,b
            vm=monitor_epoch(ema.module)
            rec=dict(run=spec['name'],epoch=glob_ep,stage=stage,stage_epoch=ep,
                     train_loss=float(np.mean(losses)),lr=float(optimizer.param_groups[-1]['lr']),encoder_lr=float(optimizer.param_groups[0]['lr']),
                     aug_strength=strength,seconds=time.perf_counter()-t0,**vm)
            history.append(rec);pq=rec['selection_score']
            raw_state=ema_state=None
            if pq>best[stage]:
                best[stage]=pq;ema_state=cpu_state(ema.module);raw_state=cpu_state(net)
                store_weights(run_dir/f'{stage}_best_ema.pth',ema_state,glob_ep,stage,pq,'ema')
                store_weights(run_dir/f'{stage}_best_raw.pth',raw_state,glob_ep,stage,pq,'raw')
            if glob_ep>=KEEP_FROM_EPOCH:
                new=dict(epoch=glob_ep,quality=pq,stage=stage,ema=f'epoch_{glob_ep:03d}_ema.pth',
                         raw=f'epoch_{glob_ep:03d}_raw.pth')
                selected=[]
                for snap in sorted(snapshots+[new],key=lambda s:s['quality'],reverse=True):
                    if all(abs(snap['epoch']-q['epoch'])>=MIN_SNAPSHOT_GAP for q in selected):
                        selected.append(snap)
                    if len(selected)==TOP_K_PER_RUN:break
                if new in selected:
                    if ema_state is None:ema_state=cpu_state(ema.module);raw_state=cpu_state(net)
                    for kind,state in [('ema',ema_state),('raw',raw_state)]:
                        store_weights(run_dir/new[kind],state,glob_ep,stage,pq,kind)
                for old in snapshots:
                    if old not in selected:
                        for kind in ['ema','raw']:(run_dir/old[kind]).unlink(missing_ok=True)
                snapshots=selected
            atomic_json(run_dir/'snapshots.json',snapshots)
            atomic_json(run_dir/'history.json',history)
            pd.DataFrame(history).to_csv(run_dir/'epoch_metrics.csv',index=False)
            payload=dict(variant=ACTIVE_VARIANT,source=SOURCE_SHA,model=cpu_state(net),ema=cpu_state(ema.module),ema_updates=ema.updates,
                         optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                         signature=sig,stage=stage,stage_epoch=ep,history=history,snapshots=snapshots,best=best,
                         python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all())
            atomic_torch(run_dir/'last_resume.pth',payload)
            history[-1]['seconds']=time.perf_counter()-t0
            del payload,raw_state,ema_state,loader
            print(f"{spec['name']} {stage} {ep:02d}/{epochs} | EMA PQ {pq:.5f} | "
                  f"SQ {vm['sq']:.4f} RQ {vm['rq']:.4f} | TP/FP/FN {vm['tp']}/{vm['fp']}/{vm['fn']} | "
                  f"soft/hard Dice {vm['val_soft_dice']:.4f}/{vm['val_hard_dice']:.4f} | "
                  f"BCE {vm['val_bce']:.5f} | "
                  f"LR {rec['lr']:.2e} | {rec['seconds']:.0f}s",flush=True)
        # Keep the stage endpoint too; final PQ can prefer a different calibration.
        for kind,nnn in [('ema',ema.module),('raw',net)]:
            store_weights(run_dir/f'{stage}_last_{kind}.pth',cpu_state(nnn),len(history),stage,
                          history[-1]['selection_score'],kind)
        saved=None
    del net,ema;gc.collect();torch.cuda.empty_cache()
    atomic_json(run_dir/'run_complete.json',dict(signature=sig,epochs=len(history)))
    return history


ACTIVE_VARIANT=os.environ['DUAL_VARIANT']
spec=dict(name=ACTIVE_VARIANT,seed=42,recipe='balanced',norm='group')
RUN_SPECS=[spec]
if RESUME_ROOT:
    path=Path(RESUME_ROOT)/ACTIVE_VARIANT/'last_resume.pth'
    if path.exists():
        RESUME_RUN=ACTIVE_VARIANT;RESUME_PATH=str(path);RESUME_BEST_DIR=str(path.parent)
        print('Resuming',path,flush=True)
# Local interrupted run takes priority over attached resume; do not copy a file over itself.
local=OUT/ACTIVE_VARIANT/'last_resume.pth'
if local.exists():RESUME_RUN=ACTIVE_VARIANT;RESUME_PATH=str(local);RESUME_BEST_DIR=''
# Base-model gradient smoke test before launching full training.
seed_everything(123)
sm=build_model().to(device).train();xx=torch.rand(1,1,64,64,device=device);yy=torch.zeros(1,2,64,64,device=device);yy[...,12:36,20:30]=1
bb=yy-F.max_pool2d(-yy,3,1,1).neg() # inner boundary
out=sm(xx);assert out['logits'].shape==xx.shape
ls=train_loss(out,yy,bb,'dice',1);ls.backward();assert torch.isfinite(ls)
assert sm.head.weight.grad.abs().sum()>0
# Verify gradient flow through the selected encoder and learned adapters.
assert sm.encoder.body.stem[0].weight.grad.abs().sum()>0
assert all(a[0].weight.grad.abs().sum()>0 for a in sm.encoder.adapters)
print('PASS: actual variant shapes, finite loss, head/backbone gradients',flush=True)
del sm,xx,yy,bb,out,ls;gc.collect();torch.cuda.empty_cache()
try:
    marker=OUT/ACTIVE_VARIANT/'run_complete.json'
    if marker.exists():
        assert json.loads(marker.read_text())['signature']==config_signature(spec)
        print('Already complete',ACTIVE_VARIANT,flush=True)
    else:train_run(spec)
except BudgetPause as e:
    atomic_json(OUT/ACTIVE_VARIANT/'paused.json',dict(reason=str(e),resume='last_resume.pth'))
    print(str(e),flush=True)
