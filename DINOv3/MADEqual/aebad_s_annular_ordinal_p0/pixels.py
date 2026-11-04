"""Pointwise pixels and explicit sampling supports; no feature receptive fields."""
import numpy as np
from .protocol import CONFIG,PAIRS,VARIANTS,version_offsets,version_matrix

def grayscale(rgb):
    a=np.asarray(rgb,np.float32)
    if a.ndim!=3 or a.shape[2]!=3:raise ValueError('RGB raster required')
    return (a[:,:,0]*np.float32(.299)+a[:,:,1]*np.float32(.587)+a[:,:,2]*np.float32(.114))/np.float32(255.)

def windows(shape):
    def axis(n):
        if n<64:return []
        return sorted(set([*range(0,n-63,32),n-64]))
    return [(x,y) for y in axis(shape[0]) for x in axis(shape[1])]

def annular(points,box,shape):
    p=np.asarray(points);x,y=box;px,py=p[...,0],p[...,1];h,w=shape
    return ((px>=max(0,x-32))&(px<min(w,x+96))&(py>=max(0,y-32))&(py<min(h,y+96))&
        ~((px>=x-8)&(px<x+72)&(py>=y-8)&(py<y+72)))

def support(points,shape):
    p=np.asarray(points,np.float32);h,w=shape;valid=np.isfinite(p).all(-1)&(p[...,0]>=0)&(p[...,0]<=w-1)&(p[...,1]>=0)&(p[...,1]<=h-1)
    safe=np.nan_to_num(p,nan=0,posinf=0,neginf=0);xx=np.clip(safe[...,0],0,w-1);yy=np.clip(safe[...,1],0,h-1)
    x0=np.floor(xx).astype(np.int32);y0=np.floor(yy).astype(np.int32);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    wx=xx-x0;wy=yy-y0
    coords=np.stack((np.stack((x0,y0),-1),np.stack((x1,y0),-1),np.stack((x0,y1),-1),np.stack((x1,y1),-1)),-2)
    weights=np.stack(((1-wx)*(1-wy),wx*(1-wy),(1-wx)*wy,wx*wy),-1).astype(np.float32)
    return coords,weights,valid

def sample(gray,points):
    coords,weights,valid=support(points,gray.shape)
    values=gray[coords[...,1],coords[...,0]]
    # Same operation order as dense index preparation.
    p=np.asarray(points,np.float32);wx=p[...,0]-np.floor(p[...,0]);wy=p[...,1]-np.floor(p[...,1])
    value=(values[...,0]*(1-wx)+values[...,1]*wx)*(1-wy)+(values[...,2]*(1-wx)+values[...,3]*wx)*wy
    return np.asarray(value,np.float32),valid

def ring_sample(gray,points,box):
    coords,weights,valid=support(points,gray.shape)
    valid &= np.all(annular(coords,box,gray.shape)|(weights==0),axis=-1)
    if not valid.all():raise ValueError('query sampling support crosses protected center or visible annulus')
    return sample(gray,points)[0]

def directions(points,box):
    d=np.asarray(points,np.float32)-np.array([box[0]+31.5,box[1]+31.5],np.float32)
    return np.where(abs(d[:,0])>=abs(d[:,1]),np.where(d[:,0]>=0,0,2),np.where(d[:,1]>=0,1,3))

def anchors(gray,box):
    x,y=box;h,w=gray.shape
    xx,yy=np.meshgrid(np.arange(max(0,x-32)+4,min(w,x+96),8),np.arange(max(0,y-32)+4,min(h,y+96),8))
    p=np.c_[xx.ravel(),yy.ravel()].astype(np.int32)
    corners=p[:,None,:]+np.array([[-8,-8],[8,-8],[-8,8],[8,8]])
    ok=np.all(annular(corners,box,gray.shape),1)
    # Entire square, not merely its corner samples, must avoid the guard box.
    ok &= ((p[:,0]+8<x-8)|(p[:,0]-8>=x+72)|(p[:,1]+8<y-8)|(p[:,1]-8>=y+72))
    p=p[ok];q=directions(p,box);contrast=np.zeros(len(p),np.float32)
    if len(p):
        local=ring_sample(gray,p[:,None,:]+np.array([(a,b) for b in (-1,0,1) for a in (-1,0,1)]),box)
        contrast=local.max(1)-local.min(1)
    chosen=[]
    for d in range(4):
        ids=np.flatnonzero((q==d)&(contrast>0));ids=ids[np.lexsort((p[ids,0],p[ids,1],-contrast[ids]))]
        chosen.extend(ids[:4])
    p=p[chosen];q=q[chosen]
    if len(p):
        values=ring_sample(gray,p[:,None,None,:]+PAIRS[None],box)
        bits=np.packbits(values[:,:,1]>values[:,:,0],axis=1,bitorder='little')
    else:bits=np.empty((0,16),np.uint8)
    return dict(points=p,bits=bits,directions=q,contrast=contrast[chosen])

def grid_spec(shape,version):
    offsets=np.array([[-8,-8],[-8,8],[8,-8],[8,8]],np.float32)@version_matrix(version).T;h,w=shape
    lo=offsets.min(0);hi=offsets.max(0)
    x0=int(np.ceil(-lo[0]/8)*8);y0=int(np.ceil(-lo[1]/8)*8)
    nx=max(0,int(np.floor((w-1-hi[0]-x0)/8))+1);ny=max(0,int(np.floor((h-1-hi[1]-y0)/8))+1)
    return dict(x0=x0,y0=y0,nx=nx,ny=ny,count=nx*ny,version=version)

def describe_grid(gray,spec):
    """128 vectorized strided operations, no Python loop over normal pixels."""
    x0,y0,nx,ny=(spec[k] for k in ('x0','y0','nx','ny'));out=np.zeros((ny,nx,16),np.uint8)
    if not nx*ny:return out.reshape(0,16)
    def plane(offset):
        dx,dy=offset;ix=int(np.floor(dx));iy=int(np.floor(dy));wx=np.float32(dx-ix);wy=np.float32(dy-iy)
        def read(a,b):return gray[y0+iy+b:y0+iy+b+ny*8:8,x0+ix+a:x0+ix+a+nx*8:8]
        # Exact integer last-pixel samples do not need a +1 neighbor.
        a=read(0,0);b=read(1,0) if wx else a;c=read(0,1) if wy else a;d=read(1,1) if wx and wy else b if wx else c
        return (a*(1-wx)+b*wx)*(1-wy)+(c*(1-wx)+d*wx)*wy
    for bit,pair in enumerate(version_offsets(spec['version'])):
        out[:,:,bit//8] |= (plane(pair[1])>plane(pair[0])).astype(np.uint8)<<np.uint8(bit%8)
    return out.reshape(-1,16)
