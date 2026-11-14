"""Original hysteresis decoding and 31 feature calculations extracted from base-hyst-learned-filters.ipynb cells 2 and 4. No tuning or training."""
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
import cv2
from threadpoolctl import threadpool_limits
S=np.ones((3,3),dtype=np.uint8)
watershed=None
DEFAULT_PP=dict(threshold=.55,min_area=400,compact=True,compact_area=500,min_elong=2.75,method='cc',grow_delta=.1,seed_threshold=.85,seed_area=60,merge_dist=40,merge_tilt=45,merge_connect=90,bridge_mean=0.,close_radius=0,open_radius=0,hole_area=0,small_conf=0.)
FEATURES=['log_area', 'log_bbox_h', 'log_bbox_w', 'extent', 'elongation', 'solidity', 'perimeter_ratio', 'max_thickness', 'mean_thickness', 'n_pieces', 'p_mean', 'p_std', 'p_min', 'p_max', 'p_q10', 'p_q50', 'p_q90', 'p_q99', 'log_p_sum', 'p_edge_mean', 'p_ring_mean', 'p_ring_max', 'nearest_centroid_distance', 'relative_area', 'image_candidate_count']
EXTRA_FEATURES=['filament_darkness','local_contrast','disk_center_dist','centroid_y','centroid_x','border_contact']
ALL_FEATURES=FEATURES+EXTRA_FEATURES
CANDIDATE_MIN_AREA=150


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


def clean(pb, min_size):
    """Label a binary prediction and drop blobs smaller than min_size pixels."""
    pl, pn = ndi.label(pb, structure=S)
    if pn == 0:
        return pl, 0
    keep = np.bincount(pl.ravel(), minlength=pn + 1) >= min_size
    keep[0] = False
    remap = np.cumsum(keep) * keep                     # kept blobs get ids 1..k
    return remap[pl], int(keep.sum())


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
                tp=int(tp),fp=int(fp),fn=int(fn))


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


def component_features(labels,n,p,raw):
    if n==0:return []
    sizes,elong,conf=instance_stats(labels,n,p)
    centers=np.array(ndi.center_of_mass(np.ones_like(labels,dtype=np.uint8),labels,np.arange(1,n+1)))
    nn=np.full(n,float(np.hypot(*labels.shape)))
    if n>1:nn=cKDTree(centers).query(centers,k=2)[0][:,1]
    total_area=max(float(sizes[1:].sum()),1.)
    rows=[]
    for cid,sl in enumerate(ndi.find_objects(labels),1):
        assert sl is not None,'Non-contiguous saved labels'
        mask=(labels[sl]==cid).astype(np.uint8)
        ys,xs=np.nonzero(mask);area=len(xs);hh,ww=mask.shape
        pr=p[sl][mask>0]
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        hull=cv2.convexHull(np.concatenate(contours)) if contours else None
        hull_area=float(cv2.contourArea(hull)) if hull is not None else float(area)
        perimeter=sum(cv2.arcLength(c,True) for c in contours)
        padded=cv2.copyMakeBorder(mask,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
        dt=cv2.distanceTransform(padded,cv2.DIST_L2,3)[1:-1,1:-1]
        # A ring excludes OTHER predicted instances as well as this instance.
        y0,x0=sl[0].start,sl[1].start;y1,x1=sl[0].stop,sl[1].stop
        pad=12;ya,xa=max(0,y0-pad),max(0,x0-pad)
        yb,xb=min(labels.shape[0],y1+pad),min(labels.shape[1],x1+pad)
        sub=(labels[ya:yb,xa:xb]==cid).astype(np.uint8)
        dil=cv2.dilate(sub,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))>0
        ring=dil & (labels[ya:yb,xa:xb]==0)
        edge=(mask>0)&~(cv2.erode(padded,np.ones((3,3),np.uint8))[1:-1,1:-1]>0)
        ring_p=p[ya:yb,xa:xb][ring]
        row=dict(label=cid,log_area=float(np.log1p(area)),log_bbox_h=float(np.log1p(hh)),log_bbox_w=float(np.log1p(ww)),
            extent=area/(hh*ww),elongation=float(elong[cid]),solidity=min(1.,area/max(hull_area,1.)),
            perimeter_ratio=perimeter**2/(4*np.pi*max(area,1)),max_thickness=float(dt.max()),
            mean_thickness=float(dt[mask>0].mean()),n_pieces=int(ndi.label(mask,structure=S)[1]),
            p_mean=float(pr.mean()),p_std=float(pr.std()),p_min=float(pr.min()),p_max=float(pr.max()),
            p_q10=float(np.quantile(pr,.1)),p_q50=float(np.quantile(pr,.5)),p_q90=float(np.quantile(pr,.9)),p_q99=float(np.quantile(pr,.99)),
            log_p_sum=float(np.log1p(pr.sum())),p_edge_mean=float(p[sl][edge].mean()) if edge.any() else float(pr.mean()),
            p_ring_mean=float(ring_p.mean()) if ring_p.size else 0.,p_ring_max=float(ring_p.max()) if ring_p.size else 0.,
            nearest_centroid_distance=float(nn[cid-1])/max(labels.shape),relative_area=area/total_area,image_candidate_count=n)
        rows.append(row)
    return rows

def hyst_cfg(low,seed,min_area):
    # In process_probability: threshold = SEED, floor = threshold - grow_delta.
    return {**DEFAULT_PP,'method':'hysteresis','threshold':float(seed),'grow_delta':round(float(seed)-float(low),4),
            'min_area':int(min_area),'compact':False,'merge_dist':0,'bridge_mean':0.,'small_conf':0.}


def extra_features(labels,n,raw):
    H,W=labels.shape;out=[]
    for cid,sl in enumerate(ndi.find_objects(labels),1):
        assert sl is not None,'Non-contiguous labels'
        ya,xa=max(0,sl[0].start-10),max(0,sl[1].start-10);yb,xb=min(H,sl[0].stop+10),min(W,sl[1].stop+10)
        sub=labels[ya:yb,xa:xb]==cid;ring=ndi.binary_dilation(sub,iterations=9)&~sub;rw=raw[ya:yb,xa:xb]
        local_mean=float(rw[sub].mean());bg=float(rw[ring].mean()) if ring.any() else float(rw.mean())
        ys,xs=np.nonzero(labels[sl]==cid);cy=sl[0].start+float(ys.mean());cx=sl[1].start+float(xs.mean())
        out.append(dict(filament_darkness=local_mean,local_contrast=bg-local_mean,
            disk_center_dist=float(np.hypot((cx-W/2)/(W/2),(cy-H/2)/(H/2))),centroid_y=cy/H,centroid_x=cx/W,
            border_contact=int(sl[0].start==0 or sl[1].start==0 or sl[0].stop==H or sl[1].stop==W)))
    return out


def candidate_rows(lab,n,p,raw):
    if n==0:return []
    feats=component_features(lab,n,p,None);ext=extra_features(lab,n,raw);areas=np.bincount(lab.ravel(),minlength=n+1)
    for r,e in zip(feats,ext):r.update(e);r['area']=int(areas[r['label']])
    return feats


def score_algo(model,df):
    if not len(df):return np.zeros(0)
    with threadpool_limits(limits=2):return np.clip(model.predict(df[ALL_FEATURES].to_numpy(np.float32)),0,1)

