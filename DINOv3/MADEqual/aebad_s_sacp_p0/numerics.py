"""FP32 residuals, uncentered shared representation, small FP64 solves."""
import numpy as np
from .protocol import CONFIG


def rank(values):
    values=np.asarray(values)
    return int(np.sum(values>max(float(values.max(initial=0))*CONFIG['rank_rtol'],np.finfo(np.float32).tiny)))


def euclidean_basis(w):
    w=np.asarray(w,np.float64)
    if not w.size:return np.empty((w.shape[0],0),np.float32)
    _,s,_=np.linalg.svd(w,full_matrices=False);n=rank(s)
    if n<w.shape[1]:return None
    q,_=np.linalg.qr(w,mode='reduced')
    return np.asarray(q,np.float32)


def shared_basis(a,seed):
    """Randomized uncentered SVD; large arrays remain FP32. No covariance D x D."""
    a=np.asarray(a,np.float32);limit=min(CONFIG['common_rank'],*a.shape)
    if not limit or not np.any(a):return np.empty((a.shape[1],0),np.float32),dict(rank=0,retained_energy=0.)
    width=min(limit+CONFIG['svd_oversampling'],*a.shape)
    omega=np.random.Generator(np.random.PCG64(seed)).standard_normal((a.shape[1],width)).astype(np.float32)
    q=np.linalg.qr(a@omega,mode='reduced')[0].astype(np.float32)
    for _ in range(CONFIG['svd_power_iterations']):
        z=np.linalg.qr(a.T@q,mode='reduced')[0].astype(np.float32)
        q=np.linalg.qr(a@z,mode='reduced')[0].astype(np.float32)
    _,s,v=np.linalg.svd(q.T@a,full_matrices=False);p=min(limit,rank(s))
    total=float(np.sum(a*a,dtype=np.float64));retained=float(np.sum(s[:p].astype(np.float64)**2))
    return np.ascontiguousarray(v[:p].T,dtype=np.float32),dict(rank=p,retained_energy=retained/total if total else 0.,
        SVD='randomized_FP32_uncentered_QR_power2_small_SVD',sampled_energy=total)


def generalized(cs,ct):
    p=len(cs);tau_s=float(np.trace(cs)/p);tau_0=float(np.trace(cs+ct)/(2*p))
    b=.9*cs+(.1*tau_s+1e-6*tau_0)*np.eye(p)
    lower=np.linalg.cholesky(b)
    h=np.linalg.solve(lower,ct);h=np.linalg.solve(lower,h.T).T;h=(h+h.T)*.5
    gamma,v=np.linalg.eigh(h);order=np.argsort(-gamma,kind='stable')
    return gamma[order],np.linalg.solve(lower.T,v[:,order]),dict(B_condition=float(np.linalg.cond(b)),tau_S=tau_s,tau_0=tau_0)


def weighted(blocks):
    """A class has unit mass: 1/2 per fold, equal valid images, then equal rows."""
    result=[];counts={}
    for fold in ('A','B'):
        local=[b for b in blocks if b['fold']==fold and len(b['delta'])]
        if len({b['image'] for b in local})!=len(local):raise ValueError('duplicate image in one class/fold/group')
        counts[fold]=len(local)
        for b in local:result.append((np.asarray(b['delta'],np.float32),1/(2*len(local)*len(b['delta']))))
    return result,counts


def fit_group(source,target,seed):
    sw,sc=weighted(source);tw,tc=weighted(target);counts=dict(Source=sc,Target=tc,
        Source_unique_images=len({b['image'] for b in source}),Target_unique_images=len({b['image'] for b in target}))
    dim=(source+target)[0]['delta'].shape[1] if source or target else CONFIG['dimension']
    contributions=[]
    for kind,blocks,number in (('Source',source,sc),('Target',target,tc)):
        for b in blocks:
            if len(b['delta']):contributions.append(dict(kind=kind,fold=b['fold'],image=b['image'],rows=len(b['delta']),
                within_class_image_mass=1/(2*number[b['fold']]),within_class_row_mass=1/(2*number[b['fold']]*len(b['delta']))))
    counts['image_contributions']=contributions
    empty=np.zeros((3,dim,CONFIG['directions']),np.float32)
    def fallback(reason,**extra):return empty,dict(status='UNION',reason=reason,directions=0,**counts,**extra)
    if any(n<CONFIG['min_images_per_class_fold'] for n in (*sc.values(),*tc.values())):return fallback('fewer_than_4_images_in_a_class_fold')
    for a,w in sw+tw:
        if a.ndim!=2 or a.shape[1]!=dim or not np.isfinite(a).all():raise ValueError('nonfinite/invalid fitting differences')
    stack=np.concatenate([a*np.float32(np.sqrt(.5*w)) for a,w in sw+tw])
    p,diagnostic=shared_basis(stack,seed)
    if not p.shape[1]:return fallback('zero_joint_energy',**diagnostic)
    def moment(rows):
        c=np.zeros((p.shape[1],p.shape[1]),np.float64);mean=np.zeros(dim,np.float64);energy=0.
        for a,w in rows:
            z=(a@p).astype(np.float64);c+=w*(z.T@z)
            mean+=w*np.sum(a,axis=0,dtype=np.float64);energy+=w*float(np.sum(a*a,dtype=np.float64))
        me=float(mean@mean)
        return c,dict(total_energy=energy,mean_energy=me,centered_energy=max(0.,energy-me))
    cs,sd=moment(sw);ct,td=moment(tw);diagnostic.update(Source_energy=sd,Target_energy=td)
    if np.trace(cs+ct)<=0:return fallback('zero_joint_energy',**diagnostic)
    gamma,v,numeric=generalized(cs,ct);diagnostic.update(numeric,gamma=gamma.tolist())
    k=min(CONFIG['directions'],int(np.sum(gamma>1)))
    if not k:return fallback('no_gamma_above_1',**diagnostic)
    eigens=[]
    for c in (ct,cs):
        e,w=np.linalg.eigh(c);order=np.argsort(-e,kind='stable')
        if rank(np.maximum(e,0))<k:return fallback('direction_control_rank_insufficient_joint_fallback',**diagnostic)
        eigens.append(w[:,order[:k]])
    bases=[euclidean_basis(p.astype(np.float64)@w) for w in (v[:,:k],*eigens)]
    if any(u is None or u.shape[1]!=k for u in bases):return fallback('Euclidean_QR_rank_deficient_joint_fallback',**diagnostic)
    result=empty.copy()
    for j,u in enumerate(bases):result[j,:,:k]=u
    error=max(float(np.max(np.abs(u.T@u-np.eye(k)))) for u in bases)
    if error>1e-5:raise FloatingPointError('Euclidean orthogonality check failed')
    return result,dict(status='effective',reason='',directions=k,orthogonality_error=error,**counts,**diagnostic)


def correct(union,residual,groups,bases,ranks,strength=.5):
    union=np.asarray(union,np.float32)
    if not 0<=strength<=1:raise ValueError('invalid strength')
    if strength==0:return union.copy(),np.zeros_like(union),dict(roundoff_clipped=0,max_eta=0.)
    residual=np.asarray(residual,np.float32);groups=np.asarray(groups)
    if (residual.shape[0]!=len(union) or groups.shape!=union.shape or not np.isfinite(residual).all()
            or (groups<0).any() or (groups>=len(ranks)).any() or not np.isfinite(union).all() or (union<0).any()):raise ValueError('invalid residual geometry')
    eta=np.zeros_like(union);denom=np.sum(residual*residual,axis=1,dtype=np.float32)
    for g in np.unique(groups):
        k=int(ranks[g]);use=np.flatnonzero((groups==g)&(denom>0))
        if not k or not len(use):continue
        z=residual[use]@bases[g,:,:k]
        eta[use]=np.sum(z*z,axis=1,dtype=np.float32)/denom[use]
    lo=float(eta.min(initial=0));hi=float(eta.max(initial=0));tol=CONFIG['eta_roundoff_tolerance']
    if lo < -tol or hi > 1+tol or not np.isfinite(eta).all():raise FloatingPointError('eta outside FP32 roundoff range')
    clipped=int(np.sum((eta<0)|(eta>1)));eta=np.clip(eta,0,1)
    result=union*np.sqrt(np.float32(1)-np.float32(strength)*eta)
    return result.astype(np.float32),eta,dict(roundoff_clipped=clipped,max_eta=hi)


def matched_uniform(union,main):
    union=np.asarray(union,np.float32);main=np.asarray(main,np.float32)
    total=float(union.sum(dtype=np.float64));delta=float((union.astype(np.float64)-main).sum())
    if delta < -1e-7 or total<0:raise ValueError('invalid nonnegative distance reduction')
    a=delta/total if total else 0.
    if not 0<=a<=1:raise ValueError('invalid uniform reduction coefficient')
    result=np.float32(1.-a)*union
    actual=float((union.astype(np.float64)-result).sum());error=abs(actual-delta)
    bound=float(np.finfo(np.float32).eps*total)
    if error>max(1e-12,bound):raise FloatingPointError('uniform total reduction mismatch')
    return result,dict(coefficient=a,main_total_reduction=delta,uniform_total_reduction=actual,absolute_error=error,
        FP32_sum_roundoff_bound=bound,matching='within_recorded_FP32_sum_roundoff')
