"""Global physical grid, exact receptive halo and area-preserving output labels."""
import numpy as np
from . import storage as st
from .protocol import CONFIG,VERSION
from .synthesis import area_resize


def bounds(center,grid,halo=None):
    halo=CONFIG['halo'] if halo is None else halo;y0,x0,y1,x1=center
    return [max(0,y0-halo),max(0,x0-halo),min(grid[0],y1+halo),min(grid[1],x1+halo)]


def plan(mask,grid,item_id):
    coverage=area_resize(mask,grid);size=CONFIG['tile_center'];positive=[];normal=[]
    for y in range(0,grid[0],size):
        for x in range(0,grid[1],size):
            center=[y,x,min(grid[0],y+size),min(grid[1],x+size)]
            (positive if np.any(coverage[y:center[2],x:center[3]]>0) else normal).append(center)
    # Cover EVERY synthetic positive cell; add up to four fixed normal centers.
    order=st.rng(VERSION,'normal_tiles',item_id).permutation(len(normal))[:(len(normal) if not positive else 4)]
    chosen=[(c,True) for c in positive]+[(normal[int(i)],False) for i in order]
    if not chosen:raise ValueError('no training tile')
    return [dict(tile=i,center=c,bounds=bounds(c,grid),positive=p) for i,(c,p) in enumerate(chosen)]


def output_window(center,grid,shape):
    intervals=[]
    for (lo,hi),n,m in zip(((center[0],center[2]),(center[1],center[3])),grid,shape):
        coord=np.maximum(np.float32(0),np.float32(n/m)*(np.arange(m,dtype=np.float32)+.5)-.5)
        owner=np.minimum(np.floor(coord+.5).astype(int),n-1)
        indices=np.flatnonzero((owner>=lo)&(owner<hi))
        if not len(indices):raise ValueError('no valid output pixels for tile center')
        intervals.append((int(indices[0]),int(indices[-1]+1)))
    return [intervals[0][0],intervals[1][0],intervals[0][1],intervals[1][1]]


def interpolate_center(logits,origin,grid,shape,center):
    import torch
    import torch.nn.functional as F
    y0,x0,y1,x1=output_window(center,grid,shape)
    yy=torch.arange(y0,y1,device=logits.device,dtype=torch.float32)
    xx=torch.arange(x0,x1,device=logits.device,dtype=torch.float32)
    yy=torch.clamp((yy+.5)*(grid[0]/shape[0])-.5,min=0)-origin[0]
    xx=torch.clamp((xx+.5)*(grid[1]/shape[1])-.5,min=0)-origin[1]
    gy,gx=torch.meshgrid((yy+.5)*2/logits.shape[-2]-1,(xx+.5)*2/logits.shape[-1]-1,indexing='ij')
    sampling=torch.stack((gx,gy),-1)[None].expand(logits.shape[0],-1,-1,-1)
    return F.grid_sample(logits.float(),sampling,mode='bilinear',padding_mode='border',align_corners=False),[y0,x0,y1,x1]
