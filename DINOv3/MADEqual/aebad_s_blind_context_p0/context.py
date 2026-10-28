"""CPU math and input masking; synthesis API has NO original-query argument."""
import hashlib
from functools import lru_cache
import numpy as np
from .protocol import CONFIG, VERSION
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import storage as st


def select_source(rows):
    population=[r for r in rows if not r['calibration']]
    if len(population)!=455:raise ValueError('original 455 construction images required')
    ordered=sorted(population,key=lambda r:(hashlib.sha256(('42|blind_source|'+r['relative_path']).encode()).hexdigest(),r['relative_path']))
    return ordered[:CONFIG['source_images']]


def select_positions(positions, cap, image, purpose):
    p=np.sort(np.asarray(positions,np.int64))
    if len(np.unique(p))!=len(p):raise ValueError('duplicate physical positions')
    if len(p)>cap:p=np.sort(st.rng(VERSION,42,purpose,image).choice(p,cap,replace=False))
    return p


def phases(grid):
    y,x=np.indices(grid)
    return (((y//5)%2)*2+(x//5)%2).astype(np.int8)


def patch_mask(grid, phase):
    if phase not in range(4):raise ValueError('four phases only')
    return phases(grid)==phase


def masked_input(processed, phase):
    """Normalized [C,H,W] or [B,C,H,W]; zero BEFORE any host crop/split."""
    h,w=processed.shape[-2:]
    if h%16 or w%16:raise ValueError('processed input must align with physical 16px patches')
    pixels=np.repeat(np.repeat(patch_mask((h//16,w//16),phase),16,0),16,1)
    if isinstance(processed,np.ndarray):
        value=processed.copy();value[...,pixels]=0
    else:
        import torch
        value=processed.clone();value[...,torch.as_tensor(pixels,device=value.device)]=0
    return value


@lru_cache(maxsize=4)
def projection(layer, dimension=1280):
    return (st.rng(VERSION,42,'projection',layer).standard_normal((dimension,128))/np.sqrt(128)).astype(np.float32)


def describe(features, matrix):
    x=np.asarray(features,np.float32)@matrix
    return (x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),np.float32(1e-12))).astype(np.float32)


def initial_head(layer):
    r=st.rng(VERSION,42,'head',layer)
    return dict(Wq=(r.standard_normal((128,32))/np.sqrt(128)).astype(np.float32),
                Wk=(r.standard_normal((128,32))/np.sqrt(128)).astype(np.float32),bias=np.zeros(2,np.float32))


def weights(query_context, candidate_context, types, valid, state):
    """No q, NN distance, global token, RGB or confidence can enter this API."""
    q=query_context@state['Wq'];k=candidate_context@state['Wk']
    q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),np.float32(1e-12))
    k/=np.maximum(np.linalg.norm(k,axis=-1,keepdims=True),np.float32(1e-12))
    logit=np.sum(q[:,None]*k,axis=-1,dtype=np.float32)/np.float32(.1)+state['bias'][types]
    if not np.all(valid.any(axis=1)):raise ValueError('no available normal reference')
    logit=np.where(valid,logit,-np.inf);logit-=logit.max(axis=1,keepdims=True)
    a=np.exp(logit).astype(np.float32);a/=a.sum(axis=1,keepdims=True,dtype=np.float32)
    return a


def synthesize(query_context, dictionary, candidate_ids, state):
    valid=candidate_ids>=0;idx=np.maximum(candidate_ids,0)
    a=weights(query_context,dictionary['keys'][idx],dictionary['types'][idx],valid,state)
    value=np.sum(a[:,:,None]*dictionary['values'][idx],axis=1,dtype=np.float32)
    return value,a


def detection(query, reference, union):
    difference=np.asarray(query,np.float32)-np.asarray(reference,np.float32)
    d=np.sqrt(np.sum(difference*difference,axis=-1,dtype=np.float32)).astype(np.float32)/np.float32(1280)
    return np.minimum(np.asarray(union,np.float32),d),d


def top_candidates(query_context,dictionary,exclude_image=None,chunk=128):
    """Exact normalized-context dot search, physical dictionary order breaks ties.

    CPU reference used in tests. Production uses the identical exhaustive blocked
    search on resident FP32 Torch keys (runtime.ContextIndex), without full QxD.
    """
    q=np.asarray(query_context,np.float32);keys=dictionary['keys'];images=dictionary['images'];types=dictionary['types']
    excluded=np.full(len(q),-1,np.int64) if exclude_image is None else np.broadcast_to(exclude_image,(len(q),))
    out=np.full((len(q),12),-1,np.int64)
    groups=[np.flatnonzero(types==0)]+[np.flatnonzero((types==1)&(images==i)) for i in sorted(set(images[types==1]))]
    if len(groups)!=9:raise ValueError('exactly K8 Target dictionaries required')
    for start in range(0,len(q),chunk):
        end=min(start+chunk,len(q));column=0
        for gi,group in enumerate(groups):
            k=4 if gi==0 else 1
            score=q[start:end]@keys[group].T
            score=np.where(images[group][None]!=excluded[start:end,None],score,-np.inf)
            for _ in range(k):
                j=np.argmax(score,axis=1);v=score[np.arange(len(j)),j];ok=np.isfinite(v)
                out[start:end,column]=np.where(ok,group[j],-1)
                score[np.arange(len(j)),j]=-np.inf;column+=1
    return out
