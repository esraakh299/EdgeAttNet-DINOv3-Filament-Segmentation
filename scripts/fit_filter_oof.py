"""Fit the locked 31-feature CatBoost regressor using training-only OOF maps.

Input contract: one file <image_stem>.npz per physical TRAIN image, each
containing p (2048x2048). These MUST come from models that excluded the
corresponding image from training. This tool does not create OOF maps.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from catboost import CatBoostRegressor
from src.data import discover_dataset,image_folder,find_coco,load_split,AnnotationIndex
from src.filtering import hyst_cfg,process_probability,candidate_rows,ALL_FEATURES

def make_targets(lab,n,gts):
    # Each predicted component gets its sum of IoU>0.5 matches across annotators.
    area=np.bincount(lab.ravel(),minlength=n+1)
    scores=np.zeros(n,dtype=float)
    for gl,gn in gts:
        gt_area=np.bincount(gl.ravel(),minlength=gn+1)
        positive=(lab>0)&(gl>0)
        cross=np.bincount(gl[positive].astype(np.int64)*(n+1)+lab[positive],minlength=(gn+1)*(n+1)).reshape(gn+1,n+1)
        iou=cross[1:,1:]/np.maximum(gt_area[1:,None]+area[None,1:]-cross[1:,1:],1)
        match=iou>0.5
        scores+=np.sum(iou*match,axis=0)
    return scores/max(len(gts),1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--splits',type=Path,required=True)
    ap.add_argument('--oof',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    dataset=discover_dataset(a.dataset)
    physical,annotations=load_split(a.splits)
    idx=AnnotationIndex(find_coco(dataset),annotations)
    image_root=image_folder(dataset,'train')
    names=physical.loc[physical.split=='train','file_name'].astype(str).tolist()
    index_path=a.oof/'oof_index.json'
    oof_index=None
    if index_path.is_file():
        from src.oof_helpers import sha256_file
        index=json.loads(index_path.read_text())
        if index.get('version')!='edgeattnet_dinov3_base_oof_index_v1' or index.get('training_images')!=566:
            raise ValueError('Invalid verified OOF index: '+str(index_path))
        oof_index={r['file_name']:r for r in index['records']}
        if len(oof_index)!=566 or set(oof_index)!=set(names):
            raise ValueError('OOF index must cover exactly the fixed 566 training physical images')
        print('Using original four-fold, fully verified OOF index:',index_path)
    tables=[]
    cfg=hyst_cfg(.15,.90,150)
    for i,name in enumerate(names):
        if oof_index is not None:
            rec=oof_index[name]
            path=Path(rec['map_path'])
            if not path.is_file() or sha256_file(path)!=rec['sha256']:
                raise ValueError(f'Missing or corrupted OOF map for {name}: {path}')
            p=np.load(path,allow_pickle=False).astype(np.float32)
        else:
            path=a.oof/(Path(name).stem+'.npz')
            if not path.is_file():raise FileNotFoundError(f'Missing OOF probability map for training image {name}: {path}')
            with np.load(path) as z:p=z['p'].astype(np.float32)
        if p.shape!=(2048,2048):raise ValueError(f'Wrong probability shape: {path}')
        with Image.open(image_root/name) as im:raw=np.asarray(im.convert('L'),dtype=np.float32)/255
        lab,n=process_probability(p,cfg)
        if n:
            df=pd.DataFrame(candidate_rows(lab,n,p,raw),columns=['label','area']+ALL_FEATURES)
            df['target_u']=make_targets(lab,n,idx.labels_for(name))
            tables.append(df)
        if (i+1)%50==0:print('OOF feature extraction:',i+1,'/',len(names),flush=True)
    if not tables:raise ValueError('No OOF instances were produced')
    df=pd.concat(tables,ignore_index=True)
    X=df[ALL_FEATURES].to_numpy(np.float32)
    reg=CatBoostRegressor(iterations=550,depth=5,learning_rate=.035,l2_leaf_reg=6,
                          loss_function='RMSE',thread_count=2,verbose=False,random_seed=a.seed)
    reg.fit(X,df.target_u.to_numpy(np.float32))
    a.out.mkdir(parents=True,exist_ok=True)
    joblib.dump(reg,a.out/'final_component_regressor_f.joblib')
    # Published decision rule; no validation tuning. Produces trained artifacts,
    # not a promise of byte-identical weights to the original without its folds.
    choice={'low':0.15,'seed':0.90,'algorithm':'cat_d5','min_area':150,'cutoff':0.25}
    (a.out/'locked_filter_f.json').write_text(json.dumps({'choice':choice,'features':ALL_FEATURES,
                   'provenance':'Training-only out-of-fold probabilities; original frozen decoding and filter settings'},indent=2))
    print('Saved filter artifacts; OOF rows:',len(df))
if __name__=='__main__':main()
