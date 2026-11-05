"""Shared candidate sampling, frozen normal expectations, then batched query readout."""
import torch
from . import pixels_cuda as px,geometry_cuda as geo
from .protocol import OFFSETS
SETS=('source','single','multi')

def normal_expectations(frozen,references,boxes,get_image,device,candidate_batch=16):
    n=len(boxes);counts=[];refs=[];candidates=[];assignments=[]
    for w,item in enumerate(frozen):
        lookup={}
        for b,name in enumerate(SETS):
            pool=item[name];items=pool['candidates'];ids={references[h['ref']]['instance_id'] for h in items}
            if any(h['instance_id']!=references[h['ref']]['instance_id'] for h in items):raise ValueError('candidate reference instance differs')
            active=not pool.get('search_unresolved',False) and bool(items) and (name=='single' or len(ids)>=2)
            counts.append(len(items) if active else 0);refs.append(len(ids))
            if not active:continue
            for h in items:
                key=(h['ref'],tuple(v for row in h['T'] for v in row))
                if key not in lookup:lookup[key]=len(candidates);candidates.append((w,h));assignments.append([])
                assignments[lookup[key]].append(w*3+b)
    positive=torch.zeros((n*3,24,64,64),device=device,dtype=torch.int32);valid_count=torch.zeros_like(positive)
    counts=torch.tensor(counts,device=device,dtype=torch.int32);refs=torch.tensor(refs,device=device,dtype=torch.int16)
    yy,xx=torch.meshgrid(torch.arange(64,device=device),torch.arange(64,device=device),indexing='ij')
    z=torch.stack((xx,yy),-1);offset=torch.as_tensor(OFFSETS,device=device)
    neighbor=z[None]+offset[:,None,None];inside=((neighbor>=0)&(neighbor<64)).all(-1)
    groups={}
    for i,(_,h) in enumerate(candidates):groups.setdefault(h['ref'],[]).append(i)
    for ref,indices in groups.items():
        image=get_image(ref)
        for lo in range(0,len(indices),candidate_batch):
            selection=indices[lo:lo+candidate_batch]
            Ts=torch.tensor([candidates[i][1]['T'] for i in selection],device=device,dtype=torch.float64)
            box=torch.tensor([boxes[candidates[i][0]] for i in selection],device=device,dtype=torch.float64)
            base=z[None].double()+box[:,None,None];other=neighbor[None].double()+box[:,None,None,None]
            v0,ok0=px.sample(image,geo.apply(Ts,base));v1,ok1=px.sample(image,geo.apply(Ts,other))
            bits=v1>v0[:,None];valid=ok0[:,None]&ok1&inside[None]
            source=[];destination=[]
            for j,i in enumerate(selection):
                for d in assignments[i]:source.append(j);destination.append(d)
            src=torch.tensor(source,device=device);dst=torch.tensor(destination,device=device)
            valid_count.index_add_(0,dst,valid[src].int());positive.index_add_(0,dst,(bits[src]&valid[src]).int())
    all_valid=(valid_count==counts[:,None,None,None])&(counts[:,None,None,None]>0)
    same=(positive==0)|(positive==counts[:,None,None,None]);agreed=all_valid&same
    return dict(expected=(positive>0).reshape(n,3,24,64,64),agreed=agreed.reshape(n,3,24,64,64),
        disagreement=(all_valid&~same).reshape(n,3,24,64,64),refs=refs.reshape(n,3),normal_expectation_frozen=True)

def query_evidence(gray,normal,boxes):
    if normal.get('normal_expectation_frozen') is not True:raise ValueError('normal expectations must be frozen first')
    device=gray.device;n=len(boxes);box=torch.as_tensor(boxes,device=device)
    yy,xx=torch.meshgrid(torch.arange(64,device=device),torch.arange(64,device=device),indexing='ij')
    centers=gray[box[:,None,None,1]+yy,box[:,None,None,0]+xx]
    neighbor=torch.stack((xx,yy),-1)[None]+torch.as_tensor(OFFSETS,device=device)[:,None,None]
    inside=((neighbor>=0)&(neighbor<64)).all(-1)
    other=centers[:,neighbor[...,1].clamp(0,63),neighbor[...,0].clamp(0,63)]
    bits=other>centers[:,None]
    count=(normal['agreed']&inside[None,None]&((bits[:,None])!=normal['expected'])).sum(2).to(torch.uint8)
    e=px.divide_by_constant(count.float(),24.);valid=normal['agreed'].any(2)
    blocks=torch.zeros((n,3,62,62),device=device);valid_blocks=torch.ones_like(blocks,dtype=torch.bool)
    for dy in range(3):
        for dx in range(3):blocks=blocks+e[:,:,dy:dy+62,dx:dx+62];valid_blocks &= valid[:,:,dy:dy+62,dx:dx+62]
    blocks=px.divide_by_constant(blocks,9.);score=torch.where(valid_blocks,blocks,torch.zeros_like(blocks)).flatten(2).amax(2)
    return dict(scores=score,evidence_present=valid_blocks.flatten(2).any(2),valid_blocks=valid_blocks.sum((2,3)),
        flip_counts=count,missing_mask_packed=px.packbits((~normal['agreed']).flatten(2)),
        agreement_fraction=normal['agreed'].double().mean((2,3,4)).float(),
        disagreement_fraction=normal['disagreement'].double().mean((2,3,4)).float(),
        distinct_references=normal['refs'],evidence=e)
