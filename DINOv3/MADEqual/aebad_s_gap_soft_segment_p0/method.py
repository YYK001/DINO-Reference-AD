"""Label-free fixed formula; original distances are already Euclidean/1280."""
import numpy as np
from DINOv3.MADEqual.aebad_s_support_distance_diagnostic.profiles import native

BASES = ('UNION_K8', 'SEGMENT_K8', 'SEGMENT_PAIR_GATE_K8')
GAP, GAP2, UNIFORM = NEW = ('GAP_SOFT_SEGMENT', 'GAP2_SOFT_SEGMENT', 'MATCHED_UNIFORM_SEGMENT')
BRANCHES = BASES + NEW
LAYERS = (7, 15, 23, 31)
VERSION = 'gap_soft_segment_fixed_v1'
PROTOCOL = dict(tool=VERSION, branches=list(BRANCHES), layers=list(LAYERS),
    populations=dict(background=[297,85,212], illumination=[265,67,198], view=[364,84,280]),
    seed=42, supports=8, distance_unit='already_euclidean_over_1280_once',
    audit_scale='float32(Source_patch_scale)+float32(1e-8)',
    gaps='profiles.native: float64(t_sorted[k-1])-t_sorted[0], divided by audit_scale',
    h='(gap2+gap4+gap8)/3', D='maximum(float64(union)-segment,0)',
    r='D/audit_scale', w='h/(h+r) if h+r>0 else 0',
    distance='float32(float64(union)-w*D)', gap2_only='h=gap2, all other operations unchanged',
    uniform_beta='sum_float64(w*D)/sum_float64(D), per image/layer, all native patches; zero denominator => 0',
    uniform_distance='float32(float64(union)-beta*D)',
    segment_atol=1e-7, segment_rtol=1e-6, score_atol=1e-5,
    localization='original Source patch MAD; native float32 four-layer mean then original float32 bilinear',
    image_score='frozen Source CH1 peak/local half-half; CURRENT supplement',
    GT_used_by_prediction=False, labels_used_by_prediction=False,
    margin_tolerance=1e-6, full_native_grid=True, parameter_search=False)


def compute(source, target, segment, medians, scales):
    """No sample identity, label, GT, coordinates or cross-image state needed."""
    s,t,v=np.asarray(source),np.asarray(target),np.asarray(segment)
    if s.ndim!=2 or s.shape[0]!=4 or t.shape!=(8,*s.shape) or v.shape!=s.shape:
        raise ValueError('complete four-layer/eight-support native distances required')
    if np.asarray(medians).shape!=(4,) or np.asarray(scales).shape!=(4,):
        raise ValueError('original four-layer patch MAD required')
    u=np.minimum(s,t.min(axis=0));out={b:np.empty_like(u) for b in NEW};weights=[];weights2=[];rows=[]
    for j,l in enumerate(LAYERS):
        # Production diagnostic validation, tolerance and gap operation order.
        p=native(s[j],t[:,j],v[j],medians[j],scales[j])
        a=(np.asarray([scales[j]],np.float32)+1e-8)[0]
        signed=u[j].astype(np.float64)-v[j];D=np.maximum(signed,0);r=D/a
        h=(p['gap2']+p['gap4']+p['gap8'])/3
        def weight(h):
            w=np.zeros_like(h);np.divide(h,h+r,out=w,where=h+r>0);return w
        w=weight(h);w2=weight(p['gap2'])
        total=float(D.sum(dtype=np.float64));requested=float((w*D).sum(dtype=np.float64))
        beta=requested/total if total>0 else 0.
        for b,z in ((GAP,w),(GAP2,w2),(UNIFORM,beta)):
            out[b][j]=(u[j].astype(np.float64)-z*D).astype(np.float32)
            if np.any(out[b][j]>u[j]) or np.any(out[b][j]<np.minimum(v[j],u[j])):
                raise FloatingPointError('soft-distance containment failed')
        amounts={b:float((u[j].astype(np.float64)-out[b][j]).sum(dtype=np.float64)) for b in NEW}
        bound=float(np.spacing(u[j]).astype(np.float64).sum())+1e-12
        if abs(amounts[GAP]-amounts[UNIFORM])>bound:
            raise FloatingPointError('matched native total exceeds float32 rounding bound')
        effective=p['effective_reduction'];record=dict(layer=l,patches=len(D),effective_reduction_patches=int(effective.sum()),
            tolerated_negative_reductions=int((signed<0).sum()),negative_reduction_min=float(signed.min()),
            original_positive_reduction_total=total,beta=beta,requested_main_reduction_total=requested,
            uniform_matching_abs_error=abs(amounts[GAP]-amounts[UNIFORM]),uniform_matching_rounding_bound=bound,
            matching_scope='this image/layer ALL native patches; not pixel or image-score matching')
        for b in NEW:record[b+'_actual_reduction_total']=amounts[b]
        for name,z in (('w',w),('w_gap2',w2)):
            z=z[effective];record[name+'_valid_count']=len(z)
            record[name+'_mean']=float(z.mean()) if len(z) else None
            for q in (0,.1,.25,.5,.75,.9,1):
                record[name+'_q'+str(int(q*100))]=float(np.quantile(z,q)) if len(z) else None
        rows.append(record);weights.append(w);weights2.append(w2)
    return out,rows,np.stack(weights),np.stack(weights2)
