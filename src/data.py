"""Original physical split and COCO multi-annotator validation records."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

SUFFIXES={'.jpg','.jpeg','.png'}

def discover_dataset(root):
    root=Path(root)
    if (root/'train').is_dir() and (root/'test').is_dir():return root
    candidates=sorted(p for p in root.rglob('MAGFiLO_1.0_Kaggle_2026') if (p/'train').is_dir() and (p/'test').is_dir())
    if len(candidates)!=1:raise FileNotFoundError(f'Expected exactly one competition dataset root, found {len(candidates)}: {candidates[:4]}')
    return candidates[0]

def image_folder(root, split):
    top=Path(root)/split
    paths=[p for p in top.rglob('*') if p.is_file() and p.suffix.lower() in SUFFIXES]
    groups={p.parent for p in paths}
    if len(groups)!=1:raise FileNotFoundError(f'Expected one image folder for {split}, found {groups}')
    return next(iter(groups))

def test_images(dataset):
    paths=sorted(p for p in (Path(dataset)/'test').rglob('*') if p.suffix.lower() in SUFFIXES and p.is_file())
    if len(paths)!=180:raise ValueError(f'Expected 180 official test images, found {len(paths)}')
    return paths

def load_split(root):
    root=Path(root)
    a=root/'physical_images_split.parquet';b=root/'annotation_sets_split.parquet'
    if not (a.is_file() and b.is_file()):
        dirs=sorted({p.parent for p in root.rglob('physical_images_split.parquet') if (p.parent/'annotation_sets_split.parquet').is_file()})
        if len(dirs)!=1:raise FileNotFoundError(f'Expected exactly one curated split directory, found {dirs[:5]}')
        a=dirs[0]/a.name;b=dirs[0]/b.name
    physical=pd.read_parquet(a);annotations=pd.read_parquet(b)
    assert set(physical.split.unique())=={'train','val'}
    assert (physical.split=='train').sum()==566 and (physical.split=='val').sum()==141
    assert set(physical.file_name)==set(annotations.file_name)
    assert physical.file_name.is_unique
    for split in ['train','val']:
        names=set(physical.loc[physical.split==split,'file_name'])
        assert set(annotations.loc[annotations.split==split,'file_name'])==names
    return physical,annotations

def find_coco(dataset):
    candidates=[]
    for p in (Path(dataset)/'train').rglob('*.json'):
        try:
            obj=json.loads(p.read_text())
            if isinstance(obj,dict) and {'images','annotations'}<=set(obj):candidates.append((p,obj))
        except (UnicodeDecodeError,ValueError):pass
    if len(candidates)!=1:raise FileNotFoundError(f'Expected one original COCO JSON, found {len(candidates)}')
    return candidates[0][1]

class AnnotationIndex:
    def __init__(self,coco,annotation_sets):
        self.images={im['id']:im for im in coco['images']}
        self.anns=defaultdict(list)
        for ann in coco['annotations']:self.anns[ann['image_id']].append(ann)
        self.rows={name:group for name,group in annotation_sets.groupby('file_name')}
    def labels_for(self,file_name):
        if file_name not in self.rows:raise KeyError(f'No annotator records for {file_name}')
        result=[]
        for row in self.rows[file_name].itertuples():
            rid=row.image_record_id
            im=self.images[rid];height,width=int(im['height']),int(im['width'])
            lab=np.zeros((height,width),dtype=np.int32)
            anns=self.anns.get(rid,[])
            for idx,ann in enumerate(anns,1):
                seg=ann['segmentation']
                if isinstance(seg,list):
                    if not seg:continue
                    rle=mask_utils.merge(mask_utils.frPyObjects(seg,height,width))
                elif isinstance(seg,dict) and isinstance(seg.get('counts'),list):
                    rle=mask_utils.frPyObjects(seg,height,width)
                else:rle=seg
                mask=mask_utils.decode(rle)
                if mask.ndim==3:mask=mask.any(axis=2)
                lab[mask.astype(bool)]=idx
            result.append((lab,len(anns)))
        return result
