"""Source-only, one PCG64 initialization, bounded Euclidean Lloyd iterations."""
import numpy as np


def assign(x,centers,chunk=2048):
    x=np.asarray(x,np.float32);centers=np.asarray(centers,np.float32);out=np.empty(len(x),np.int16)
    for start in range(0,len(x),chunk):
        # Grouping only: FP32 Euclidean squared distances, no detection-unit change.
        a=x[start:start+chunk];dist=np.sum(a*a,axis=1)[:,None]+np.sum(centers*centers,axis=1)[None,:]-2*(a@centers.T)
        if not np.isfinite(dist).all():raise FloatingPointError('nonfinite Source clustering distances')
        out[start:start+len(a)]=np.argmin(dist,axis=1)
    return out


def kmeans(bank,seed=42,groups=32,cap=16384,iterations=30):
    bank=np.asarray(bank,np.float32)
    if bank.ndim!=2 or len(bank)<groups:raise ValueError('insufficient Source bank rows')
    rng=np.random.Generator(np.random.PCG64(seed));ids=np.sort(rng.choice(len(bank),min(cap,len(bank)),replace=False))
    x=np.array(bank[ids],copy=True);initial=rng.choice(len(x),groups,replace=False);centers=x[initial].copy();old=None;empty=0
    for step in range(iterations):
        labels=assign(x,centers)
        if old is not None and np.array_equal(old,labels):break
        old=labels.copy()
        for g in range(groups):
            if np.any(labels==g):centers[g]=x[labels==g].mean(axis=0,dtype=np.float32)
            else:empty+=1  # Retain center: no random retry or unbounded loop.
    return centers,assign(bank,centers),dict(iterations=step+1,empty_cluster_events=empty,
        sampled_rows=len(ids),initial_bank_rows=ids[initial].tolist(),seed=seed,
        initialization='single_PCG64_without_replacement',empty_policy='retain_previous_center')
