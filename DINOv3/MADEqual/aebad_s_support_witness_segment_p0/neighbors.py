"""Blocked direct Euclidean top-k; deterministic original-row tie handling.

CPU is the real production SciPy path, not a mock of Torch. CUDA is explicitly
selected and lazily imported. Both use FP64 direct cdist (no norm/dot formula).
"""
import numpy as np


def topk(query,reference,k=5,*,query_chunk=64,reference_chunk=1024,device='cpu',progress=None):
    q,r=np.asarray(query),np.asarray(reference)
    if q.ndim!=2 or r.ndim!=2 or q.shape[1]!=r.shape[1] or min(len(q),len(r))<1 or len(r)<k or k<1:
        raise ValueError('complete banks with at least k reference rows required')
    if query_chunk<1 or reference_chunk<1 or not np.isfinite(q).all() or not np.isfinite(r).all():raise ValueError('invalid blocked NN input')
    if device=='cpu':
        from scipy.spatial.distance import cdist
        def candidates(a,b,offset):
            d=cdist(a.astype(np.float64),b.astype(np.float64),metric='euclidean')
            ids=np.broadcast_to(np.arange(offset,offset+len(b),dtype=np.int64),d.shape)
            order=np.lexsort((ids,d),axis=1)[:,:k]
            return np.take_along_axis(ids,order,axis=1),np.take_along_axis(d,order,axis=1)
    else:
        import torch
        dev=torch.device(device)
        if dev.type!='cuda':raise ValueError('select cpu or explicit CUDA device')
        torch.cuda.set_device(dev)
        def candidates(a,b,offset):
            with torch.inference_mode():
                aa=torch.as_tensor(a.astype(np.float64),device=dev);bb=torch.as_tensor(b.astype(np.float64),device=dev)
                d=torch.cdist(aa,bb,p=2,compute_mode='donot_use_mm_for_euclid_dist')
                # Reference block is in original row order. Stable sort resolves
                # exact ties on GPU; transfer only its <=k candidates to CPU.
                order=torch.argsort(d,dim=1,stable=True)[:,:k]
                return order.cpu().numpy()+offset,torch.gather(d,1,order).cpu().numpy()
    indices=np.empty((len(q),k),np.int64);distances=np.empty((len(q),k),np.float64)
    for start in range(0,len(q),query_chunk):
        a=q[start:start+query_chunk];best_i=np.empty((len(a),0),np.int64);best_d=np.empty((len(a),0),np.float64)
        for off in range(0,len(r),reference_chunk):
            ids,d=candidates(a,r[off:off+reference_chunk],off)
            if not np.isfinite(d).all():raise FloatingPointError('nonfinite neighbor distance')
            # Lexicographic distance then ORIGINAL feature row, including block boundaries.
            joined_i=np.concatenate((best_i,ids),axis=1);joined_d=np.concatenate((best_d,d),axis=1)
            order=np.lexsort((joined_i,joined_d),axis=1)[:,:k]
            best_i=np.take_along_axis(joined_i,order,axis=1);best_d=np.take_along_axis(joined_d,order,axis=1)
        indices[start:start+len(a)]=best_i;distances[start:start+len(a)]=best_d
        if progress is not None:progress(start+len(a),len(q))
    return indices,distances


def tables(banks,**kwargs):
    result={}
    for a in sorted(banks):
        for c in sorted(banks):
            if c!=a:result[a,c]=topk(banks[a],banks[c],**kwargs)
    return result
