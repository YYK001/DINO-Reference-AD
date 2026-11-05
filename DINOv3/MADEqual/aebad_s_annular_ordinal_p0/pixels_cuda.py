"""Eager tensor kernels, FP32 sampling in the CPU operation order; no padding evidence."""
import numpy as np
import torch
from .protocol import PAIRS,OUTER_PAIRS,version_offsets

def divide_by_constant(value,divisor):
    """Device denominator avoids CUDA's CPU-scalar reciprocal-multiply shortcut."""
    return value/value.new_full((),divisor)

def grayscale(rgb):
    a=rgb.to(torch.float32)
    return divide_by_constant(a[...,0]*.299+a[...,1]*.587+a[...,2]*.114,255.)

def support(points,shape):
    p=points.to(torch.float32);h,w=shape
    valid=torch.isfinite(p).all(-1)&(p[...,0]>=0)&(p[...,0]<=w-1)&(p[...,1]>=0)&(p[...,1]<=h-1)
    safe=torch.nan_to_num(p,nan=0.,posinf=0.,neginf=0.)
    xx=safe[...,0].clamp(0,w-1);yy=safe[...,1].clamp(0,h-1)
    x0=xx.floor().long();y0=yy.floor().long();x1=(x0+1).clamp_max(w-1);y1=(y0+1).clamp_max(h-1)
    wx=xx-x0;wy=yy-y0
    coords=torch.stack((torch.stack((x0,y0),-1),torch.stack((x1,y0),-1),torch.stack((x0,y1),-1),torch.stack((x1,y1),-1)),-2)
    weights=torch.stack(((1-wx)*(1-wy),wx*(1-wy),(1-wx)*wy,wx*wy),-1)
    return coords,weights,valid

def sample(gray,points):
    coords,weights,valid=support(points,gray.shape);values=gray[coords[...,1],coords[...,0]]
    p=points.float();fraction=p-p.floor();wx=fraction[...,0];wy=fraction[...,1]
    result=(values[...,0]*(1-wx)+values[...,1]*wx)*(1-wy)+(values[...,2]*(1-wx)+values[...,3]*wx)*wy
    return result,valid

def annular(points,boxes,shape):
    h,w=shape;x=boxes[...,0];y=boxes[...,1];xx=points[...,0];yy=points[...,1]
    return ((xx>=torch.maximum(x-32,torch.zeros_like(x)))&(xx<torch.minimum(x+96,torch.full_like(x,w)))&
        (yy>=torch.maximum(y-32,torch.zeros_like(y)))&(yy<torch.minimum(y+96,torch.full_like(y,h)))&
        ~((xx>=x-8)&(xx<x+72)&(yy>=y-8)&(yy<y+72)))

def packbits(bits):
    n=bits.shape[-1]
    if n%8:bits=torch.nn.functional.pad(bits,(0,8-n%8))
    weights=(1<<torch.arange(8,device=bits.device,dtype=torch.int32))
    return (bits.reshape(*bits.shape[:-1],-1,8).to(torch.int32)*weights).sum(-1).to(torch.uint8)

def outer_batch(gray,boxes):
    """Fixed 16x16 grid, stable per-direction top4 for a batch of windows."""
    device=gray.device;boxes=torch.as_tensor(boxes,device=device,dtype=torch.int64);n=len(boxes)
    gy,gx=torch.meshgrid(torch.arange(16,device=device),torch.arange(16,device=device),indexing='ij')
    grid=torch.stack((gx,gy),-1).reshape(1,256,2)*8
    start=torch.maximum(boxes-32,torch.zeros_like(boxes))+4;points=start[:,None]+grid
    square=torch.tensor([[-8,-8],[8,-8],[-8,8],[8,8]],device=device)
    ok=annular(points[:,:,None]+square,boxes[:,None,None],gray.shape).all(-1)
    x=boxes[:,0,None];y=boxes[:,1,None]
    ok &= ((points[...,0]+8<x-8)|(points[...,0]-8>=x+72)|(points[...,1]+8<y-8)|(points[...,1]-8>=y+72))
    d=points.float()-(boxes.float()[:,None]+31.5)
    direction=torch.where(d[...,0].abs()>=d[...,1].abs(),torch.where(d[...,0]>=0,0,2),torch.where(d[...,1]>=0,1,3))
    local=torch.tensor([(a,b) for b in (-1,0,1) for a in (-1,0,1)],device=device)
    values,_=sample(gray,points[ok][:,None]+local)
    contrast=torch.zeros(points.shape[:2],device=device);contrast[ok]=values.amax(-1)-values.amin(-1)
    chosen=[];chosen_valid=[]
    for quadrant in range(4):
        score=torch.where(ok&(direction==quadrant)&(contrast>0),contrast,torch.full_like(contrast,-torch.inf))
        # Grid order is y then x. Stable sort preserves the original tie rule.
        ids=torch.argsort(score,dim=1,descending=True,stable=True)[:,:4]
        chosen.append(ids);chosen_valid.append(torch.gather(score,1,ids).isfinite())
    ids=torch.cat(chosen,1);valid=torch.cat(chosen_valid,1)
    p=torch.gather(points,1,ids[:,:,None].expand(-1,-1,2));dirs=torch.gather(direction,1,ids)
    pair=torch.as_tensor(PAIRS,device=device);v,_=sample(gray,p[valid][:,None,None]+pair)
    bits=torch.zeros((n,16,16),device=device,dtype=torch.uint8);bits[valid]=packbits(v[...,1]>v[...,0])
    outer=torch.as_tensor(OUTER_PAIRS,device=device)[None]+boxes[:,None,None]
    outer_valid=annular(outer,boxes[:,None,None],gray.shape).all(-1)
    ov,_=sample(gray,outer[outer_valid]);outer_bits=torch.zeros(outer_valid.shape,device=device,dtype=torch.bool)
    outer_bits[outer_valid]=ov[...,1]>ov[...,0]
    return dict(points=p,bits=bits,anchor_valid=valid,directions=dirs,contrast=torch.gather(contrast,1,ids),
        validation_pairs=outer,validation_bits=outer_bits,validation_valid=outer_valid,shape=tuple(gray.shape))

def outer_to_host(info):
    """One compact batch transfer for CPU scheduling/RNG; no center values."""
    keys=('points','bits','anchor_valid','directions','contrast','validation_pairs','validation_bits','validation_valid')
    host={k:info[k].detach().cpu().numpy() for k in keys};result=[]
    for j in range(len(host['points'])):
        keep=host['anchor_valid'][j];pairs=host['validation_valid'][j]
        result.append(dict(points=host['points'][j,keep],bits=host['bits'][j,keep],directions=host['directions'][j,keep],
            contrast=host['contrast'][j,keep],validation_pairs=host['validation_pairs'][j,pairs],
            validation_bits=host['validation_bits'][j,pairs],shape=info['shape']))
    return result

def describe_grid(gray,spec,chunk=8192):
    """Match the CPU dense descriptor's fixed fractional offsets, not rounded absolute offsets."""
    offsets=torch.as_tensor(version_offsets(spec['version']),device=gray.device)
    integer=offsets.floor().long();fraction=offsets-integer
    parts=[]
    for start in range(0,spec['count'],chunk):
        ids=torch.arange(start,min(spec['count'],start+chunk),device=gray.device)
        p=torch.stack((spec['x0']+(ids%spec['nx'])*8,spec['y0']+(ids//spec['nx'])*8),-1)
        xy=p[:,None,None]+integer;wx=fraction[...,0];wy=fraction[...,1]
        def read(dx,dy):return gray[(xy[...,1]+dy).clamp_max(gray.shape[0]-1),(xy[...,0]+dx).clamp_max(gray.shape[1]-1)]
        a=read(0,0);b=read(1,0);c=read(0,1);d=read(1,1)
        value=(a*(1-wx)+b*wx)*(1-wy)+(c*(1-wx)+d*wx)*wy
        parts.append(packbits(value[...,1]>value[...,0]))
    return torch.cat(parts) if parts else torch.empty((0,16),device=gray.device,dtype=torch.uint8)
