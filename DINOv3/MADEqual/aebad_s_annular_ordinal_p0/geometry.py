"""Only annular query data may enter fitting or candidate selection."""
import numpy as np
from . import pixels as px
from .protocol import CONFIG,OUTER_PAIRS,version_matrix
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import storage as st

def apply(T,points):return np.asarray(points)@np.asarray(T)[:,:2].T+np.asarray(T)[:,2]
def corners(box):
    x,y=box;return np.array([[x,y],[x+63,y],[x,y+63],[x+63,y+63]],np.float64)
def admissible(T):
    if not np.isfinite(T).all() or np.linalg.det(T[:,:2])<=0:return False
    s=np.linalg.svd(T[:,:2],compute_uv=False)
    return bool(s.min()>=.75 and s.max()<=1.33)

def prepare_outer(gray,box):
    result=px.anchors(gray,box);pairs=OUTER_PAIRS+np.array(box)
    valid=px.annular(pairs,box,gray.shape).all(1);pairs=pairs[valid]
    values=px.ring_sample(gray,pairs,box) if len(pairs) else np.empty((0,2),np.float32)
    return dict(result,validation_pairs=pairs,validation_bits=values[:,1]>values[:,0],shape=gray.shape)

def ref_support_valid(T,center,version,box,query_shape,reference_shape):
    M=version_matrix(version)
    y,x=np.mgrid[-8:9,-8:9];points=np.c_[x.ravel(),y.ravel()]@M.T+center
    coords,weights,valid=px.support(points,reference_shape)
    inverse=(coords-T[:,2])@np.linalg.inv(T[:,:2]).T
    return bool(valid.all() and np.all(px.annular(inverse,box,query_shape)|(weights==0)))

def leave_direction_stability(T,q,r,directions,box):
    present=np.unique(directions);max_shift=0.
    for direction in present:
        keep=directions!=direction;X=np.c_[q[keep],np.ones(keep.sum())]
        if keep.sum()<3 or np.linalg.matrix_rank(X)<3:return None
        alternate=np.linalg.lstsq(X,r[keep],rcond=None)[0].T
        if not admissible(alternate):return None
        delta=(apply(alternate,corners(box))-apply(T,corners(box)))@np.linalg.inv(T[:,:2]).T
        max_shift=max(max_shift,float(np.linalg.norm(delta,axis=1).max()))
    return max_shift

def validation_cost(info,T,gray,box):
    pairs=info['validation_pairs']
    if not len(pairs):return None,0
    mapped=apply(T,pairs);coords,weights,valid=px.support(mapped,gray.shape)
    inverse=(coords-T[:,2])@np.linalg.inv(T[:,:2]).T
    valid &= np.all(px.annular(inverse,box,info['shape'])|(weights==0),axis=-1)
    good=valid.all(1)
    if not good.any():return None,0
    values,_=px.sample(gray,mapped[good]);bits=values[:,1]>values[:,0]
    return float(np.mean(bits!=info['validation_bits'][good],dtype=np.float32)),int(good.sum())

def fit_image(info,matches,reference,box,stats):
    if len({m['anchor'] for m in matches})<6:
        stats['insufficient_same_image']+=1;return []
    q=np.array([info['points'][m['anchor']] for m in matches],np.float64)
    r=np.array([m['point'] for m in matches],np.float64);aid=np.array([m['anchor'] for m in matches])
    groups=info['directions'][aid];n=len(matches)
    # Neither a whole-query hash nor its center enters the random seed.
    rng=st.rng('annular_hypotheses_v1',42,list(box),reference['instance_id'])
    triples=np.array([rng.choice(n,3,replace=False) for _ in range(128)])
    X=np.concatenate((q[triples],np.ones((128,3,1))),axis=2)
    good=(abs(np.linalg.det(X))>1e-6)
    triples=triples[good];X=X[good]
    if not len(X):stats['degenerate']+=1;return []
    transforms=np.swapaxes(np.linalg.solve(X,r[triples]),1,2)
    # Batch the tiny native solves/reprojection; do not loop 128 times merely
    # to discover that only the generating three points support a hypothesis.
    stats['hypotheses']+=len(transforms)
    matrix=transforms[:,:,:2];singular=np.linalg.svd(matrix,compute_uv=False)
    regular=(np.linalg.det(matrix)>0)&(singular.min(1)>=.75)&(singular.max(1)<=1.33)
    stats['degenerate']+=int((~regular).sum())
    predicted=np.einsum('hij,nj->hni',matrix,q)+transforms[:,None,:,2]
    errors=np.linalg.norm(predicted-r[None],axis=2);indices=np.arange(len(transforms))
    unique=np.unique(aid);nearest=[]
    for anchor in unique:
        options=np.flatnonzero(aid==anchor);nearest.append(options[errors[:,options].argmin(1)])
    nearest=np.stack(nearest,axis=1);within=errors[indices[:,None],nearest]<=2
    directional=groups[nearest]
    covered=sum(np.any(within&(directional==d),axis=1) for d in range(4))
    eligible=np.flatnonzero(regular&(within.sum(1)>=6)&(covered>=3))
    accepted=[];gray=None
    for index in eligible:
        T=transforms[index];triplet=triples[index];chosen=nearest[index,within[index]].tolist()
        if not all(ref_support_valid(T,r[j],matches[j]['version'],box,info['shape'],reference['shape']) for j in triplet):continue
        chosen=[j for j in chosen if ref_support_valid(T,r[j],matches[j]['version'],box,info['shape'],reference['shape'])]
        if len(chosen)<6 or len(np.unique(groups[chosen]))<3:continue
        heldout=[j for j in chosen if aid[j] not in aid[triplet]]
        if len(heldout)<3:continue
        stable=leave_direction_stability(T,q[chosen],r[chosen],groups[chosen],box)
        if stable is None or stable>1:
            stats['geometry_unstable']+=1;continue
        if gray is None:gray=reference['read_gray']()
        cost,pairs=validation_cost(info,T,gray,box)
        if cost is None or cost>.25:stats['outer_validation_rejected']+=1;continue
        accepted.append(dict(ref=reference['index'],instance_id=reference['instance_id'],kind=reference['kind'],
            T=T.tolist(),outer_cost=cost,inliers=len(chosen),heldout_inliers=len(heldout),
            directions=int(len(np.unique(groups[chosen]))),corner_shift_query_pixels=stable,validation_pairs=pairs))
    return accepted

def deduplicate(candidates,box):
    result=[]
    for h in sorted(candidates,key=lambda x:(x['outer_cost'],-x['inliers'],x['instance_id'],x['kind'],str(x['T']))):
        mapped=apply(np.asarray(h['T']),corners(box))
        if any(h['instance_id']==k['instance_id'] and np.linalg.norm(mapped-apply(np.asarray(k['T']),corners(box)),axis=1).max()<=.5 for k in result):continue
        result.append(h)
    return result

def candidate_groups(matches,references):
    grouped={}
    for match in matches:
        ref=references[match['ref']];key=(ref['kind'],ref['instance_id'])
        grouped.setdefault(key,[]).append(match)
    source=[key for key in grouped if key[0]=='Source']
    source.sort(key=lambda k:(-len({m['anchor'] for m in grouped[k]}),np.mean([m['hamming'] for m in grouped[k]]),k[1]))
    saturated=len(source)>8
    chosen=source[:8]+sorted(k for k in grouped if k[0]=='Target')[:8]
    result=[]
    for key in chosen:
        unique={}
        for m in sorted(grouped[key],key=lambda m:(m['hamming'],m['ref'],m['version'])):
            unique.setdefault((m['anchor'],*m['point']),m)
        group=list(unique.values());ref=references[min(m['ref'] for m in group)]
        result.append((group,ref))
    return result,saturated

def assemble(qualified,box,stats,pools=None):
    source,both=pools if pools is not None else (deduplicate([h for h in qualified if h['kind']=='Source'],box),deduplicate(qualified,box))
    def competing(pool):
        close=[h for h in pool if h['outer_cost']<=pool[0]['outer_cost']+.05] if pool else []
        return dict(candidates=close if len(close)<=12 else [],near_best_count=len(close),search_unresolved=len(close)>12)
    return dict(box=list(box),source=competing(source),multi=competing(both),single=dict(candidates=both[:1],search_unresolved=False),
        stats=stats,qualified=len(qualified),GT_used=False,query_center_used=False)

def freeze(info,matches,references,box):
    """No query pixels, center residual, P, labels or GT are accepted here."""
    stats=dict(hypotheses=0,degenerate=0,insufficient_same_image=0,geometry_unstable=0,outer_validation_rejected=0)
    groups,saturated=candidate_groups(matches,references);stats['source_image_budget_saturated']=saturated;qualified=[]
    for group,ref in groups:qualified.extend(fit_image(info,group,ref,box,stats))
    return assemble(qualified,box,stats)
