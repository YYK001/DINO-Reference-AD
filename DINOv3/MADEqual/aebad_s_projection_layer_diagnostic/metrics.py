"""Exact CPU ROC/AP, complete pixels, float32 scores without quantization."""
from __future__ import annotations
import numpy as np


def resource_bytes(count):
    # argsort indices, source mmap pages, gathered scores/labels, chunk work and margin.
    return int(count)*32+(512<<20)


def exact_binary(labels,scores,*,memory_budget_bytes,chunk_size=1_000_000):
    y=np.asarray(labels);s=np.asarray(scores)
    if (y.ndim!=1 or s.shape!=y.shape or not y.size or np.any((y!=0)&(y!=1))
            or not np.isfinite(s).all()): raise ValueError('complete finite binary metric arrays required')
    positives=int(y.sum(dtype=np.int64));negatives=len(y)-positives
    if not positives or not negatives:
        return dict(AUROC=None,AP=1. if positives else None,valid=False,AUROC_valid=False,AP_valid=bool(positives),
                    pixels=len(y),positive_pixels=positives,negative_pixels=negatives,
                    status='all_positive_AP_defined_AUROC_undefined' if positives else 'no_positive_GT_not_computable')
    if resource_bytes(len(y))>memory_budget_bytes:
        raise MemoryError(f'exact pixel sort needs conservative {resource_bytes(len(y))} bytes, budget={memory_budget_bytes}; no approximation')
    # Quicksort does not require stable ties: ties are aggregated exactly below.
    order=np.argsort(s,kind='quicksort')
    sorted_s=s[order];sorted_y=y[order];del order
    pos_before=0;neg_before=0;wins=0.;ap_sum=0.;pending_score=None;pending_p=0;pending_n=0
    def consume(p,n):
        nonlocal pos_before,neg_before,wins,ap_sum
        if not len(p):return
        bp=pos_before+np.cumsum(p,dtype=np.int64)-p
        bn=neg_before+np.cumsum(n,dtype=np.int64)-n
        wins+=float(np.sum(p.astype(float)*(bn+.5*n)))
        # At threshold score v, include the WHOLE tied group with score >= v.
        ap_sum+=float(np.sum(p.astype(float)*(positives-bp)/(len(y)-bp-bn)))
        pos_before+=int(p.sum());neg_before+=int(n.sum())
    for start in range(0,len(y),chunk_size):
        v=sorted_s[start:start+chunk_size];lab=sorted_y[start:start+chunk_size]
        offsets=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]
        values=v[offsets];counts=np.diff(np.r_[offsets,len(v)])
        p=np.add.reduceat(lab.astype(np.int64),offsets);n=counts-p
        if pending_score is not None:
            if values[0]==pending_score:p[0]+=pending_p;n[0]+=pending_n
            else:consume(np.array([pending_p],np.int64),np.array([pending_n],np.int64))
        consume(p[:-1],n[:-1])
        pending_score=values[-1];pending_p=int(p[-1]);pending_n=int(n[-1])
    consume(np.array([pending_p],np.int64),np.array([pending_n],np.int64))
    return dict(AUROC=wins/(positives*negatives),AP=ap_sum/positives,valid=True,AUROC_valid=True,AP_valid=True,pixels=len(y),
                positive_pixels=positives,negative_pixels=negatives,status='exact_all_pixels_cpu_tie_grouped')
