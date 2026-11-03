"""Source-only RMS and calibration, no labels in fitting or prediction."""
import numpy as np
from .protocol import BASE,SHARED,OWN,PRIMARY,CAP,FREE,BRANCHES


def squared_sums(features):
    sums=[];n=None
    for x in features:
        if x.ndim!=2 or x.dtype!=np.float32:raise ValueError('complete original FP32 feature rows required')
        if n is None:n=x.shape[0]
        if x.shape[0]!=n:raise ValueError('layer physical patch populations differ')
        total=0.
        for start in range(0,n,256):
            block=np.asarray(x[start:start+256],np.float64)
            if not np.isfinite(block).all():raise ValueError('nonfinite Source feature')
            total+=float(np.sum(block*block,dtype=np.float64))
        sums.append(total)
    return np.array(sums),n


def rms_from_sums(sums,total_patches,dimension=1280):
    value=np.sqrt(np.asarray(sums,np.float64)/(total_patches*dimension))
    if value.shape!=(4,) or not np.isfinite(value).all() or np.any(value<=0):raise ValueError('invalid complete Source RMS')
    return value


def calibration(records):
    if len(records)!=66 or any(r['kind']!='Source' for r in records):raise ValueError('exactly 66 independent Source calibration observations required')
    result={}
    for key in ('P','cap','free'):
        x=np.array([r[key] for r in records],np.float64)
        if not np.isfinite(x).all():raise ValueError('nonfinite normal calibration')
        mu=float(x.mean());sigma=float(np.sqrt(np.mean((x-mu)**2)))
        threshold=float(64*np.finfo(np.float64).eps*max(1.,np.max(abs(x))))
        result[key]=dict(mean=mu,std=sigma,threshold=threshold,status='valid' if sigma>threshold else 'degenerate')
    result['GT_used']=False;return result


def scores(P,cap,free,model):
    raw=dict(P=float(P),cap=float(cap),free=float(free))
    if not all(np.isfinite(v) for v in raw.values()):raise ValueError('nonfinite scoring input')
    result={BASE:raw['P'],CAP:raw['cap'],FREE:raw['free']};blocked={}
    def z(k,v):return (v-model[k]['mean'])/model[k]['std']
    for branch,cover,scale in ((SHARED,'free','cap'),(OWN,'free','free'),(PRIMARY,'cap','cap')):
        invalid=[k for k in ('P',scale) if model[k]['status']!='valid']
        if invalid:blocked[branch]='degenerate normal scale: '+','.join(invalid)
        else:result[branch]=z('P',P)+z(scale,raw[cover])
    if not all(np.isfinite(v) for v in result.values()):raise ValueError('nonfinite standardized score')
    return result,blocked
