"""Only compact medians/regression use FP64; original SEGMENT and CH1 are reused."""
import numpy as np
from .protocol import BASE,SOURCE,TARGET
from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import scatter
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import from_z,z_values
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics,ch1_score,pair_errors


def observation(values,positions,grid,shape,mad,encoding='standardized_z'):
    a=np.asarray(values)
    if a.dtype!=np.float32 or a.shape!=(4,int(np.prod(grid))) or not np.isfinite(a).all() or min(grid)<3:
        raise ValueError('complete finite four-layer FP32 native SEGMENT required')
    # Reuse physical mapping checks; padding/missing/overlap positions rejected.
    scatter(a,positions,tuple(grid))
    if encoding=='raw_distance':
        _,peak,local=statistics(a,positions,grid,shape,np.asarray(mad['patch']['medians'],np.float32),np.asarray(mad['patch']['scales'],np.float32))
        z=z_values(a,mad['patch'])
    elif encoding=='standardized_z':
        z=a;_,peak,local=from_z(z,positions,grid,shape)
    else:raise ValueError('explicit raw_distance or standardized_z encoding required')
    med=np.median(z.astype(np.float64),axis=1)
    return dict(B=float(med.mean()),P=ch1_score(peak,local,mad['image']),layer_medians=med.tolist(),
        peak=peak.tolist(),local=local.tolist(),grid=list(grid),positions_count=len(positions))


def validate_observation(r,mad):
    for key in ('B','P','layer_medians','peak','local'):
        if not np.isfinite(np.asarray(r[key],np.float64)).all():raise ValueError('nonfinite B/P statistics')
    if any(np.shape(r[k])!=(4,) for k in ('layer_medians','peak','local')):raise ValueError('four-layer statistics required')
    if abs(r['B']-float(np.mean(r['layer_medians'],dtype=np.float64)))>1e-12:raise ValueError('B median reduction differs')
    if abs(r['P']-ch1_score(r['peak'],r['local'],mad['image']))>1e-5:raise ValueError('CH1 statistics differ')
    if min(r['grid'])<3 or r['positions_count']!=int(np.prod(r['grid'])):raise ValueError('incomplete normal grid')


def vectors(b,p,n=None):
    b=np.asarray(b,np.float64);p=np.asarray(p,np.float64)
    if b.ndim!=1 or p.shape!=b.shape or not len(b) or (n is not None and len(b)!=n) or not np.isfinite(b).all() or not np.isfinite(p).all():
        raise ValueError('finite nonempty B/P population required')
    return b,p


def target_moments(b,p,model):
    b,p=vectors(b,p)
    if not np.isfinite(p.mean()):raise ValueError('nonfinite Target mean')
    if model['degenerate']:return dict(target_count=len(b),mu_xT=None,mu_PT=float(p.mean()),c_T=None,v_T=None,
        beta_ST_unconstrained=None,beta_ST=0.,Target_constraint_triggered=False,status='Source_scale_degenerate')
    x=(b-model['mu_BS'])/model['sigma_BS'];xc=x-x.mean();pc=p-p.mean()
    covariance=float(np.mean(xc*pc));variance=float(np.mean(xc*xc))
    unconstrained=(covariance+model['beta_S'])/(variance+1.)
    if not np.isfinite([covariance,variance,unconstrained]).all():raise ValueError('nonfinite Target moments')
    return dict(target_count=len(b),mu_xT=float(x.mean()),mu_PT=float(p.mean()),c_T=covariance,v_T=variance,
        beta_ST_unconstrained=unconstrained,beta_ST=max(0.,unconstrained),Target_constraint_triggered=unconstrained<0,status='valid')


def fit(bs,ps,bt,pt):
    bs,ps=vectors(bs,ps,66);bt,pt=vectors(bt,pt,8)
    mu=float(bs.mean());sigma=float(np.sqrt(np.mean((bs-mu)**2)))
    floor=float(64*np.finfo(np.float64).eps*max(1.,float(np.abs(bs).max())))
    model=dict(mu_BS=mu,sigma_BS=sigma,mu_PS=float(ps.mean()),scale_threshold=floor,degenerate=sigma<=floor,
        cov_BS_PS=float(np.mean((bs-mu)*(ps-ps.mean()))),Source_count=66,ddof=0,ridge=1.)
    if not np.isfinite([mu,sigma,model['mu_PS'],model['cov_BS_PS'],floor]).all():raise ValueError('nonfinite Source moments')
    if model['degenerate']:model.update(mu_xS=None,c_S=None,v_S=None,beta_S_unconstrained=None,beta_S=0.,Source_constraint_triggered=False)
    else:
        x=(bs-mu)/sigma;xc=x-x.mean();cov=float(np.mean(xc*(ps-ps.mean())));var=float(np.mean(xc*xc))
        slope=cov/var
        if not np.isfinite([cov,var,slope]).all() or var<=0:raise ValueError('nonfinite Source moments')
        model.update(mu_xS=float(x.mean()),c_S=cov,v_S=var,beta_S_unconstrained=slope,beta_S=max(0.,slope),Source_constraint_triggered=slope<0)
    return dict(model,**target_moments(bt,pt,model))


def scores(b,p,model):
    b,p=vectors(b,p)
    x=np.zeros_like(b) if model['degenerate'] else (b-model['mu_BS'])/model['sigma_BS']
    out={BASE:p.copy()}
    for branch,key in ((SOURCE,'beta_S'),(TARGET,'beta_ST')):
        # Strict original score preservation, including beta=0/degenerate cases.
        out[branch]=p.copy() if model['degenerate'] or model[key]==0 else p-model[key]*x
        if not np.isfinite(out[branch]).all():raise ValueError('nonfinite corrected score')
    return out,x


def influence(bt,pt,model):
    bt,pt=vectors(bt,pt,8)
    return [dict(deleted_LOO_index=j,analysis='remove_one_statistic_not_a_new_support_set',
        **target_moments(np.delete(bt,j),np.delete(pt,j),model)) for j in range(8)]


def ranking(labels,original,corrected):
    before=pair_errors(labels,original);after=pair_errors(labels,corrected)
    return dict(pairs=before.size,improved_pairs=int((after<before).sum()),worsened_pairs=int((after>before).sum()),
        unchanged_pairs=int((after==before).sum()),original_error_mass=float(before.sum()),corrected_error_mass=float(after.sum()),
        net_error_change=float(after.sum()-before.sum()),ties_half_error=True,
        interpretation='normal_anomaly_ranking_pairs_not_independent_samples_or_fixed_threshold_errors')
