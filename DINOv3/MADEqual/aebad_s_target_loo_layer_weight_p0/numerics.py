"""CPU FP32 maps, original p/q order, float64 small normal-image reductions."""
import numpy as np
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import from_z
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics

# Positive four-term normalization: one FP64->FP32 rounding per weight,
# followed by at most three FP32 additions. gamma_4 bounds the combined error.
# This is a validation bound only; do not renormalize or change the FP32 map.
_FP32_UNIT_ROUNDOFF=np.finfo(np.float32).eps/2
WEIGHT_SUM_ATOL=4*_FP32_UNIT_ROUNDOFF/(1-4*_FP32_UNIT_ROUNDOFF)


def normal_statistics(raw,positions,grid,shape,patch):
    raw=np.asarray(raw,np.float32)
    medians=np.asarray(patch['medians'],np.float32);scales=np.asarray(patch['scales'],np.float32)
    _,p,q=statistics(raw,positions,grid,shape,medians,scales)
    z=(raw-np.asarray(patch['medians'],np.float32)[:,None])/np.asarray(patch['scales'],np.float32)[:,None]
    return np.stack([np.quantile(z,.99,axis=1,method='linear'),p,q],axis=1)


def moments(source,target):
    source=np.asarray(source,np.float64);target=np.asarray(target,np.float64)
    if source.shape!=(66,4,3) or target.shape!=(8,4,3) or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('finite 66x4x3 Source and 8x4x3 Target required')
    anchor=np.median(source,axis=0)
    return anchor,np.mean(np.maximum(source-anchor,0)**2,axis=0),np.mean(np.maximum(target-anchor,0)**2,axis=0)


def weights(ss,tt,gamma):
    if gamma not in (0.,.5):raise ValueError('only the two fixed gammas are defined')
    # gamma=0 does not evaluate Target input, even if it contains NA.
    variance=np.asarray(ss,np.float64) if gamma==0 else (1-gamma)*np.asarray(ss,np.float64)+gamma*np.asarray(tt,np.float64)
    if variance.shape!=(4,3) or not np.isfinite(variance).all() or (variance<0).any():raise ValueError('invalid positive-tail second moments')
    scale=np.sqrt(variance);floored=scale<1e-8;scale=np.maximum(scale,1e-8)
    inverse=1/scale
    return inverse/inverse.sum(axis=0),scale,floored


def weighted(z,positions,grid,shape,w):
    z=np.asarray(z,np.float32);w=np.asarray(w,np.float32)
    if w.shape!=(4,3) or not np.isfinite(w).all() or (w<=0).any() or not np.allclose(w.sum(0,dtype=np.float32),1,atol=WEIGHT_SUM_ATOL,rtol=0):raise ValueError('four positive normalized weights per statistic required')
    # p/q are derived from ORIGINAL layer z; never from u-weighted maps.
    _,p,q=from_z(z,positions,grid,shape)
    physical=np.empty_like(z);physical[:,positions]=z;physical=physical.reshape(4,*grid)
    if np.array_equal(w[:,0],np.full(4,.25,np.float32)):native=physical.mean(0,dtype=np.float32)
    else:native=np.sum(physical*w[:,0,None,None],axis=0,dtype=np.float32)
    from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import bilinear
    image=bilinear(native,shape)
    return image,dict(PEAK_LOCAL_WEIGHT=float(.5*np.sum(w[:,1]*p,dtype=np.float32)+.5*np.sum(w[:,2]*q,dtype=np.float32)),WEIGHTED_MAP_MAX=float(image.max()))


def reference_rows(supports,heldout):
    if heldout not in range(len(supports)):raise ValueError('invalid heldout support')
    chosen=[r for i,r in enumerate(supports) if i!=heldout]
    if len(chosen)!=7 or any(r['relative_path']==supports[heldout]['relative_path'] for r in chosen):raise ValueError('whole-image K7 exclusion required')
    return chosen
