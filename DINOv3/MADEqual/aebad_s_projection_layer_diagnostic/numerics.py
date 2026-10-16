"""NumPy equivalents of frozen MAD, half-pixel bilinear and adaptive max masks."""
from __future__ import annotations
import numpy as np

LAYERS=(7,15,23,31)
RESPONSES=(*LAYERS,'FUSED')
SCORE_ATOL=1e-5


def bilinear(value,shape):
    """Float32 bilinear, align_corners=False; never PIL resize for responses.

    Source: superadd_host.interpolate_score and spatial.interpolation_stencil.
    Torch is not imported. Operation rounding may differ from the CPU ATen kernel.
    """
    a=np.asarray(value)
    if a.ndim!=2 or a.dtype!=np.float32 or not np.isfinite(a).all() or min(shape)<=0:
        raise ValueError('finite float32 grid and positive output shape required')
    if tuple(a.shape)==tuple(shape): return a.copy()
    axes=[]
    for size,out in zip(a.shape,shape,strict=True):
        x=np.maximum(np.float32(0),np.float32(size/out)*(np.arange(out,dtype=np.float32)+np.float32(.5))-np.float32(.5))
        lower=np.floor(x).astype(np.int64);upper=np.minimum(lower+1,size-1)
        weight=x-lower.astype(np.float32)
        axes.append((lower,upper,weight))
    (y0,y1,fy),(x0,x1,fx)=axes
    top=a[y0[:,None],x0]*(1-fx)+a[y0[:,None],x1]*fx
    bottom=a[y1[:,None],x0]*(1-fx)+a[y1[:,None],x1]*fx
    return np.ascontiguousarray(top*(1-fy[:,None])+bottom*fy[:,None],dtype=np.float32)


def adaptive_max(mask,grid):
    """Original adaptive_max_pool2d start=floor/end=ceil, including overlaps."""
    mask=np.asarray(mask,bool)
    if mask.ndim!=2 or min(grid)<=0: raise ValueError('2D mask and positive grid required')
    # Integral image gives exact any() over every overlapping adaptive bin.
    dtype=np.uint32 if mask.size<=np.iinfo(np.uint32).max else np.uint64
    integral=np.pad(mask.astype(dtype),((1,0),(1,0)))
    integral.cumsum(0,dtype=dtype,out=integral);integral.cumsum(1,dtype=dtype,out=integral)
    ranges=[]
    for size,out in zip(mask.shape,grid,strict=True):
        i=np.arange(out,dtype=np.int64)
        ranges.append((i*size//out,((i+1)*size+out-1)//out))
    (y0,y1),(x0,x1)=ranges
    counts=integral[y1[:,None],x1]-integral[y0[:,None],x1]-integral[y1[:,None],x0]+integral[y0[:,None],x0]
    return counts>0


def reconstruct(raw,positions,grid,shape,medians,scales):
    p=np.asarray(positions);d=np.asarray(raw)
    n=int(np.prod(grid))
    if (p.dtype!=np.int64 or p.shape!=(n,) or not np.array_equal(np.sort(p),np.arange(n))
            or d.dtype!=np.float32 or d.shape!=(4,n) or not np.isfinite(d).all() or np.any(d<0)
            or medians.shape!=(4,) or scales.shape!=(4,) or np.any(scales<=0)):
        raise ValueError('valid distances, physical bijection and original MAD required')
    physical=np.empty_like(d);physical[:,p]=d
    # Same float32 arithmetic as relation_oracle_cache.extract.mad_calibrated.
    z=((physical.reshape(4,*grid)-medians[:,None,None])/scales[:,None,None]).astype(np.float32)
    maps={l:bilinear(z[j],shape) for j,l in enumerate(LAYERS)}
    maps['FUSED']=bilinear(z.mean(axis=0),shape)
    layer_mean=np.stack([maps[l] for l in LAYERS]).mean(axis=0)
    error=float(np.abs(layer_mean.astype(float)-maps['FUSED']).max())
    return z,maps,error


def check(rows,sid,kind,actual,expected,*,atol=SCORE_ATOL,rtol=0,**fields):
    a,b=np.asarray(actual),np.asarray(expected)
    okay=a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=atol,rtol=rtol)
    error=float(np.abs(a.astype(float)-b).max()) if a.shape==b.shape and a.size else None
    rows.append(dict(sample_id=sid,check_kind=kind,verified=bool(okay),max_abs_error=error,atol=atol,rtol=rtol,**fields))
    return bool(okay)
