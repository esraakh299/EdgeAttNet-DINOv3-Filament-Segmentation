"""Submission inference: exact TTA8 geometry, hysteresis and learned selection."""
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from src.architecture import main_logits
from src.filtering import (hyst_cfg,process_probability,candidate_rows,ALL_FEATURES,score_algo,
                           evaluate_one_label,metrics_from_counts)

@torch.inference_mode()
def tta8_probability(net,path,device):
    with Image.open(path) as im:
        if (im.width,im.height)!=(2048,2048):raise ValueError(f'Expected a 2048² input: {path}')
        raw=np.asarray(im.convert('L'),dtype=np.float32)/255.0
    x=torch.from_numpy(raw.copy())[None,None].to(device)
    avg=torch.zeros_like(x,dtype=torch.float32)
    for flip in (False,True):
        for k in range(4):
            view=torch.rot90(x.flip(-1) if flip else x,k,(-2,-1))
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=(device.type=='cuda')):
                output=net(view)
            prob=torch.rot90(main_logits(output).float().sigmoid(),-k,(-2,-1))
            avg+=(prob.flip(-1) if flip else prob)/8
    p=avg[0,0].cpu().numpy().astype(np.float16).astype(np.float32)
    return p,raw

def select_instances(p,raw,regressor,choice):
    cfg=hyst_cfg(choice['low'],choice['seed'],150)
    lab,n=process_probability(p,cfg)
    if not n:return lab,[],n
    candidates=pd.DataFrame(candidate_rows(lab,n,p,raw),columns=['label','area']+ALL_FEATURES)
    keep=(candidates.area.to_numpy()>=int(choice['min_area']))
    if choice['algorithm']=='no_filter':pass
    elif choice['algorithm']=='p_mean':keep &= candidates.p_mean.to_numpy()>=float(choice['cutoff'])
    else:keep &= score_algo(regressor,candidates)>=float(choice['cutoff'])
    kept=[int(x) for x in candidates.label[keep]]
    return lab,kept,n

def relabel_selected(lab,kept):
    # Exactly match source labels when comparing with multi-annotator GT.
    lookup=np.zeros(int(lab.max())+1,dtype=np.int32)
    for idx,k in enumerate(kept,1):lookup[k]=idx
    return lookup[lab],len(kept)

def run_validation(net,regressor,choice,paths,gt_index,device,out_dir):
    from tqdm.auto import tqdm
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    total=np.zeros(4,dtype=np.float64);rows=[]
    for path in tqdm(paths,desc='Validation · 141 images'):
        p,raw=tta8_probability(net,path,device)
        lab,keep,n=select_instances(p,raw,regressor,choice)
        pred,npred=relabel_selected(lab,keep)
        item=evaluate_one_label(pred,npred,gt_index.labels_for(path.name))
        total+=item
        rows.append({'file_name':path.name,'candidate_count':int(n),'accepted_count':int(npred),
                     'sum_iou':float(item[0]),'tp':int(item[1]),'fp':int(item[2]),'fn':int(item[3])})
        del p,raw,lab,pred
    metrics=metrics_from_counts(total)
    pd.DataFrame(rows).to_csv(out_dir/'validation_per_image.csv',index=False)
    pd.DataFrame([metrics]).to_csv(out_dir/'validation_metrics.csv',index=False)
    return metrics

def run_test(net,regressor,choice,paths,device,out_dir):
    from tqdm.auto import tqdm
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    rows=[];empty=[];total_candidates=0
    for path in tqdm(paths,desc='Test · 180 images'):
        p,raw=tta8_probability(net,path,device)
        lab,keep,n=select_instances(p,raw,regressor,choice)
        total_candidates+=n
        if not keep:empty.append(path.name)
        for idx,k in enumerate(keep,1):
            mask=np.asfortranarray((lab==k).astype(np.uint8))
            rle=mask_utils.encode(mask)
            rows.append((f'{path.stem}_{idx}',rle['counts'].decode('ascii')))
        del p,raw,lab
    table=pd.DataFrame(rows,columns=['filament_id','segmentation_rle'])
    assert len(table)>0 and table.filament_id.is_unique
    path=out_dir/'submission.csv'
    table.to_csv(path,index=False)
    return path,{'test_images':len(paths),'filaments':len(table),'empty_images':empty,'candidates':int(total_candidates)}
