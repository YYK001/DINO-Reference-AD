"""Batched tensor geometry; CPU schedules groups and the unchanged fixed random triples."""
import numpy as np
import torch
from . import geometry as cpu,pixels_cuda as px
from .protocol import version_matrix
st=cpu.st

def apply(T,p):return torch.einsum('h...j,hij->h...i',p,T[:,:,:2])+T[:,None,:,2].reshape(len(T),*([1]*(p.ndim-2)),2)

def regular(T):
    singular=torch.linalg.svdvals(T[...,:2])
    return (torch.linalg.det(T[...,:2])>0)&(singular.amin(-1)>=.75)&(singular.amax(-1)<=1.33)&torch.isfinite(T).all((-1,-2))

def support_valid(T,center,version,boxes,qshape,rshape):
    device=T.device;yy,xx=torch.meshgrid(torch.arange(-8,9,device=device),torch.arange(-8,9,device=device),indexing='ij')
    grid=torch.stack((xx,yy),-1).reshape(-1,2).double()
    matrices=torch.as_tensor(np.stack([version_matrix(i) for i in range(9)]),device=device,dtype=torch.float64)[version]
    p=torch.einsum('pj,hkij->hkpi',grid,matrices)+center[:,:,None]
    coords,weights,valid=px.support(p,rshape)
    inverse=torch.einsum('hkpvj,hij->hkpvi',coords.double()-T[:,None,None,None,:,2],torch.linalg.inv(T[:,:,:2]))
    annulus=px.annular(inverse,boxes[:,None,None,None],qshape)|(weights==0)
    return valid.all(-1)&annulus.all((-1,-2))

def stability(T,q,r,keep,directions,boxes):
    n=len(T);device=T.device;quadrants=torch.arange(4,device=device)
    active=keep[:,None,:]&(directions[:,None,:]!=quadrants[None,:,None])
    present=(keep[:,None,:]&(directions[:,None,:]==quadrants[None,:,None])).any(-1)
    X=torch.cat((q,torch.ones((n,16,1),device=device,dtype=torch.float64)),-1)[:,None]*active[:,:,:,None]
    Y=r[:,None]*active[:,:,:,None]
    rank=torch.linalg.matrix_rank(X);valid=(active.sum(-1)>=3)&(rank==3)
    safe=torch.zeros_like(X);safe[:,:,:3,:]=torch.eye(3,device=device,dtype=torch.float64)
    X=torch.where(valid[:,:,None,None],X,safe);Y=torch.where(valid[:,:,None,None],Y,torch.zeros_like(Y))
    alternate=torch.linalg.lstsq(X,Y,driver='gels').solution.transpose(-1,-2)
    valid &= regular(alternate)
    offsets=torch.tensor([[0,0],[63,0],[0,63],[63,63]],device=device,dtype=torch.float64)
    corners=boxes[:,None,:]+offsets;original=apply(T,corners)
    mapped=torch.einsum('hpj,hdij->hdpi',corners,alternate[...,:2])+alternate[:,:,None,:,2]
    delta=torch.einsum('hdpj,hij->hdpi',mapped-original[:,None],torch.linalg.inv(T[:,:,:2]))
    change=delta.norm(dim=-1).amax(-1)
    return torch.where((valid|~present).all(1),torch.where(present,change,torch.zeros_like(change)).amax(1),torch.full((n,),torch.inf,device=device))

def outer_cost(info,T,boxes,gray,qshape):
    p=info['pairs'];mapped=apply(T,p);coords,weights,valid=px.support(mapped,gray.shape)
    inverse=torch.einsum('hpuvj,hij->hpuvi',coords.double()-T[:,None,None,None,:,2],torch.linalg.inv(T[:,:,:2]))
    valid &= (px.annular(inverse,boxes[:,None,None,None],qshape)|(weights==0)).all(-1)
    good=valid.all(-1)&info['valid'];values,_=px.sample(gray,mapped[good])
    predicted=torch.zeros(good.shape,device=T.device,dtype=torch.bool);predicted[good]=values[...,1]>values[...,0]
    flipped=predicted!=info['bits'];count=good.sum(1)
    cost=(flipped&good).sum(1).float()/count.clamp_min(1).float()
    return cost,count

def deduplicate_pools(pools,boxes,device):
    """GPU corner pair distances; CPU only performs stable greedy bookkeeping."""
    result=[[] for _ in pools];groups=[]
    for i,pool in enumerate(pools):
        ordered=sorted(pool,key=lambda h:(h['outer_cost'],-h['inliers'],h['instance_id'],h['kind'],str(h['T'])))
        by_id={}
        for h in ordered:by_id.setdefault(h['instance_id'],[]).append(h)
        groups.extend((i,group) for group in by_id.values())
    for start in range(0,len(groups),16):
        batch=groups[start:start+16];width=max(len(g) for _,g in batch)
        Ts=np.zeros((len(batch),width,2,3));corner=np.zeros((len(batch),4,2))
        for j,(i,group) in enumerate(batch):Ts[j,:len(group)]=[h['T'] for h in group];corner[j]=cpu.corners(boxes[i])
        tensor=torch.as_tensor(Ts,device=device,dtype=torch.float64);c=torch.as_tensor(corner,device=device,dtype=torch.float64)
        mapped=torch.einsum('bpj,bhij->bhpi',c,tensor[...,:2])+tensor[:,:,None,:,2]
        close=((mapped[:,:,None]-mapped[:,None]).norm(dim=-1).amax(-1)<=.5).cpu().numpy()
        for j,(i,group) in enumerate(batch):
            kept=[]
            for k,h in enumerate(group):
                if not any(close[j,k,other] for other in kept):kept.append(k);result[i].append(h)
    for i in range(len(result)):result[i].sort(key=lambda h:(h['outer_cost'],-h['inliers'],h['instance_id'],h['kind'],str(h['T'])))
    return result

def freeze_batch(infos,matches,references,boxes,get_image,device,group_batch=8,hypothesis_batch=32,triple_cache=None):
    jobs=[];stats=[];qualified=[[] for _ in boxes]
    for window,(info,match,box) in enumerate(zip(infos,matches,boxes)):
        groups,saturated=cpu.candidate_groups(match,references)
        stats.append(dict(hypotheses=0,degenerate=0,insufficient_same_image=0,geometry_unstable=0,outer_validation_rejected=0,source_image_budget_saturated=saturated))
        for group,ref in groups:
            if len({m['anchor'] for m in group})<6:stats[-1]['insufficient_same_image']+=1;continue
            key=(tuple(box),ref['instance_id'],len(group))
            if triple_cache is not None and key in triple_cache:
                triples=triple_cache[key];triple_cache.move_to_end(key)
            else:
                rng=st.rng('annular_hypotheses_v1',42,list(box),ref['instance_id'])
                triples=np.array([rng.choice(len(group),3,replace=False) for _ in range(128)])
                if triple_cache is not None:
                    triple_cache[key]=triples
                    while len(triple_cache)>65536:triple_cache.popitem(last=False)
            jobs.append(dict(window=window,info=info,matches=group,ref=ref,box=box,triples=triples))
    jobs.sort(key=lambda j:(j['ref']['index'],j['window']))
    for begin in range(0,len(jobs),group_batch):
        batch=jobs[begin:begin+group_batch];B=len(batch);M=max(len(j['matches']) for j in batch)
        if hasattr(get_image,'prefetch'):get_image.prefetch(list(dict.fromkeys(j['ref']['index'] for j in batch)))
        q=np.zeros((B,M,2));r=np.zeros_like(q);aid=np.zeros((B,M),np.int64);directions=np.zeros_like(aid);versions=np.zeros_like(aid);valid=np.zeros((B,M),bool)
        pairs=np.zeros((B,128,2,2));bits=np.zeros((B,128),bool);pair_valid=np.zeros_like(bits)
        for b,j in enumerate(batch):
            for k,m in enumerate(j['matches']):q[b,k]=j['info']['points'][m['anchor']];r[b,k]=m['point'];aid[b,k]=m['anchor'];directions[b,k]=j['info']['directions'][m['anchor']];versions[b,k]=m['version'];valid[b,k]=True
            n=len(j['info']['validation_pairs']);pairs[b,:n]=j['info']['validation_pairs'];bits[b,:n]=j['info']['validation_bits'];pair_valid[b,:n]=True
        q=torch.as_tensor(q,device=device);r=torch.as_tensor(r,device=device);aid=torch.as_tensor(aid,device=device);directions=torch.as_tensor(directions,device=device)
        versions=torch.as_tensor(versions,device=device);valid=torch.as_tensor(valid,device=device);triples=torch.as_tensor(np.stack([j['triples'] for j in batch]),device=device)
        bx=torch.arange(B,device=device)[:,None,None];tq=q[bx,triples];tr=r[bx,triples]
        X=torch.cat((tq,torch.ones((*tq.shape[:-1],1),device=device,dtype=torch.float64)),-1)
        good=torch.linalg.det(X).abs()>1e-6
        safe=torch.where(good[:,:,None,None],X,torch.eye(3,device=device,dtype=torch.float64))
        T=torch.linalg.solve(safe,tr).transpose(-1,-2);reg=regular(T)&good
        error=(torch.einsum('bhij,bnj->bhni',T[...,:2],q)+T[:,:,None,:,2]-r[:,None]).norm(dim=-1)
        error=torch.where(valid[:,None],error,torch.full_like(error,torch.inf));index=aid[:,None].expand(-1,128,-1)
        minimum=torch.full((B,128,16),torch.inf,device=device,dtype=torch.float64).scatter_reduce_(2,index,error,reduce='amin')
        near=minimum.gather(2,index);candidate=torch.where((error==near)&valid[:,None],torch.arange(M,device=device),M)
        nearest=torch.full((B,128,16),M,device=device,dtype=torch.int64).scatter_reduce_(2,index,candidate,reduce='amin')
        within=minimum<=2;dirs=directions[:,None].expand(-1,128,-1).gather(2,nearest.clamp_max(M-1))
        covered=sum((within&(dirs==d)).any(-1).int() for d in range(4))
        eligible=(reg&(within.sum(-1)>=6)&(covered>=3)).nonzero()
        counts=torch.stack((good.sum(1),(good&~reg).sum(1)),1).cpu().numpy()
        for b,j in enumerate(batch):
            stats[j['window']]['hypotheses']+=int(counts[b,0]);stats[j['window']]['degenerate']+=int(counts[b,1])+(0 if counts[b,0] else 1)
        # Process accepted-hypothesis batches by normal image to reuse one uploaded raster.
        for b,j in enumerate(batch):
            selected=eligible[eligible[:,0]==b,1]
            if not len(selected):continue
            gray=get_image(j['ref']['index'])
            for first in range(0,len(selected),hypothesis_batch):
                hi=selected[first:first+hypothesis_batch];trans=T[b,hi];N=len(hi);box=torch.tensor(j['box'],device=device,dtype=torch.float64).expand(N,-1)
                ni=nearest[b,hi].clamp_max(M-1);keep=within[b,hi];centers=r[b,ni];ver=versions[b,ni]
                tri=triples[b,hi];trip_valid=support_valid(trans,r[b,tri],versions[b,tri],box,j['info']['shape'],gray.shape).all(1)
                keep=keep&support_valid(trans,centers,ver,box,j['info']['shape'],gray.shape)
                di=directions[b,ni];aq=aid[b,ni];ta=aid[b,tri]
                held=keep&~(aq[:,:,None]==ta[:,None,:]).any(-1)
                enough=trip_valid&(keep.sum(1)>=6)&(sum((keep&(di==d)).any(1).int() for d in range(4))>=3)&(held.sum(1)>=3)
                stable=stability(trans,q[b,ni],centers,keep,di,box)
                unstable=enough&(~torch.isfinite(stable)|(stable>1))
                outer=dict(pairs=torch.as_tensor(pairs[b],device=device).expand(N,-1,-1,-1),bits=torch.as_tensor(bits[b],device=device).expand(N,-1),valid=torch.as_tensor(pair_valid[b],device=device).expand(N,-1))
                cost,pair_count=outer_cost(outer,trans,box,gray,j['info']['shape'])
                active=enough&~unstable;accepted=active&(pair_count>0)&(cost<=.25)
                values=torch.stack((cost,keep.sum(1),held.sum(1),sum((keep&(di==d)).any(1).int() for d in range(4)),stable,pair_count),1)[accepted].cpu().numpy()
                matrices=trans[accepted].cpu().numpy();status=torch.stack((unstable.sum(),(active&~accepted).sum())).cpu().numpy()
                s=stats[j['window']];s['geometry_unstable']+=int(status[0]);s['outer_validation_rejected']+=int(status[1])
                for matrix,v in zip(matrices,values):qualified[j['window']].append(dict(ref=j['ref']['index'],instance_id=j['ref']['instance_id'],kind=j['ref']['kind'],T=matrix.tolist(),outer_cost=float(v[0]),inliers=int(v[1]),heldout_inliers=int(v[2]),directions=int(v[3]),corner_shift_query_pixels=float(v[4]),validation_pairs=int(v[5])))
    pools=[];pool_boxes=[]
    for pool,box in zip(qualified,boxes):pools.extend(([h for h in pool if h['kind']=='Source'],pool));pool_boxes.extend((box,box))
    unique=deduplicate_pools(pools,pool_boxes,device)
    return [cpu.assemble(pool,box,s,pools=(unique[2*i],unique[2*i+1])) for i,(pool,box,s) in enumerate(zip(qualified,boxes,stats))]
