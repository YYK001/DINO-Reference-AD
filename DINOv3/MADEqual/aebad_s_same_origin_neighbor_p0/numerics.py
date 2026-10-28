"""FP32 formulas; physical neighbors never mean adjacent compact/bank rows.

No GT, image label, software identity, or model is an input to these kernels.
"""
import numpy as np
from .protocol import A, B, C


def physical(values, positions, grid):
    p=np.asarray(positions)
    if p.dtype!=np.int64 or not np.array_equal(np.sort(p),np.arange(np.prod(grid))):
        raise ValueError('complete physical position permutation required')
    if len(values)!=len(p):raise ValueError('feature/position count differs')
    out=np.empty_like(values);out[p]=values
    return out.reshape(*grid,values.shape[-1])


def neighbors(grid, positions):
    h,w=map(int,grid);p=np.asarray(positions,np.int64)
    if (p<0).any() or (p>=h*w).any():raise ValueError('physical index out of range')
    y,x=p//w,p%w;valid=(y>0)&(y<h-1)&(x>0)&(x<w-1)
    ids=p[:,None]+np.array([-w,w,-1,1],np.int64)
    return np.clip(ids,0,h*w-1),valid


def deltas(full, positions):
    """full is [H,W,D] in physical order; invalid rows carry zero placeholders."""
    ids,valid=neighbors(full.shape[:2],positions);flat=full.reshape(-1,full.shape[-1])
    value=flat[ids]-flat[np.asarray(positions),None,:]
    value[~valid]=0
    return value.astype(np.float32),valid


def stable_smallest(distances, indices, k):
    """NumPy reference lexicographic selection, also used by CPU tests."""
    d=np.asarray(distances,np.float32);ids=np.broadcast_to(indices,d.shape)
    order=np.argsort(ids,axis=1,kind='stable')
    d0=np.take_along_axis(d,order,1)
    order=np.take_along_axis(order,np.argsort(d0,axis=1,kind='stable')[:,:k],1)
    return np.take_along_axis(d,order,1),np.take_along_axis(ids,order,1)


def exact_topk_numpy(q,m,k=16,query_chunk=32,reference_chunk=128):
    """Small CPU reference; production uses blocked torch.cdist FP32."""
    result=[];winners=[]
    for start in range(0,len(q),query_chunk):
        x=np.asarray(q[start:start+query_chunk],np.float32)
        bd=np.empty((len(x),0),np.float32);bi=np.empty((len(x),0),np.int64)
        for lo in range(0,len(m),reference_chunk):
            block=np.asarray(m[lo:lo+reference_chunk],np.float32)
            d=np.sqrt(np.sum((x[:,None]-block[None])**2,axis=-1,dtype=np.float32))
            ids=np.broadcast_to(np.arange(lo,lo+len(block),dtype=np.int64),d.shape)
            bd,bi=stable_smallest(np.concatenate([bd,d],1),np.concatenate([bi,ids],1),min(k,bd.shape[1]+d.shape[1]))
        result.append(bd/np.float32(q.shape[1]));winners.append(bi)
    return np.concatenate(result),np.concatenate(winners)


def residuals(q,query_delta,centers,memory_delta):
    """Candidates [Q,K,D], deltas [Q,K,4,D]; division follows each norm."""
    dim=np.float32(q.shape[-1]);center=q[:,None,:]-centers
    absolute=np.sqrt(np.sum(center*center,axis=-1,dtype=np.float32))/dim
    rel=query_delta[:,None,:,:]-memory_delta
    nb=center[:,:,None,:]+rel
    rel=np.mean(np.sqrt(np.sum(rel*rel,axis=-1,dtype=np.float32))/dim,axis=-1,dtype=np.float32)
    nb=np.mean(np.sqrt(np.sum(nb*nb,axis=-1,dtype=np.float32))/dim,axis=-1,dtype=np.float32)
    return absolute,nb,rel


def joint(absolute,nb,rel,candidate_valid,query_valid,union_z,state):
    """Already standardized outputs. No second patch MAD and no clipping."""
    za=(absolute-np.float32(state['abs']['median']))/np.float32(state['abs']['scale'])
    zn=(nb-np.float32(state['nb']['median']))/np.float32(state['nb']['scale'])
    zr=(rel-np.float32(state['rel']['median']))/np.float32(state['rel']['scale'])
    good=np.asarray(query_valid)&np.asarray(candidate_valid)[:,0]
    ja=np.float32(.5)*za+np.float32(.5)*zn
    jr=np.float32(.5)*za+np.float32(.5)*zr
    masked=np.where(candidate_valid,jr,np.float32(np.inf))
    # Ascending absolute distance/global row is also the stable joint-cost tie rule.
    selected=masked.argmin(1)
    values={A:np.where(good,ja[:,0],union_z).astype(np.float32),
            B:np.where(good,jr[:,0],union_z).astype(np.float32),
            C:np.where(good,masked[np.arange(len(masked)),selected],union_z).astype(np.float32)}
    return values,dict(patches=len(good),query_boundary=int((~np.asarray(query_valid)).sum()),
        top1_boundary=int((~np.asarray(candidate_valid)[:,0]).sum()),fallback=int((~good).sum()),
        excluded_top16_candidates=int((~candidate_valid[good]).sum()),effective_joint_positions=int(good.sum()))


def torch_smallest(d,ids,k):
    """Fast top-k, explicitly repair cutoff ties, then lexicographic order."""
    import torch
    ids=ids.expand_as(d);k=min(k,d.shape[1]);v,ix=torch.topk(d,k,dim=1,largest=False,sorted=False)
    cutoff=v.max(1).values;picked=torch.gather(ids,1,ix)
    # Repair every cutoff tie on device, with fixed-size tensors. No per-block
    # GPU->CPU boolean/synchronization and no full-matrix stable sort.
    sentinel=torch.iinfo(torch.int64).max
    tie_ids=torch.where(d==cutoff[:,None],ids,sentinel)
    tid=torch.topk(tie_ids,k,dim=1,largest=False,sorted=False).values
    tv=torch.where(tid!=sentinel,cutoff[:,None],torch.inf)
    v=torch.cat([torch.where(v<cutoff[:,None],v,torch.inf),tv],1)
    picked=torch.cat([picked,tid],1)
    order=picked.argsort(dim=1,stable=True);v=torch.gather(v,1,order);picked=torch.gather(picked,1,order)
    order=v.argsort(dim=1,stable=True)[:,:k]
    return torch.gather(v,1,order),torch.gather(picked,1,order)


class Search:
    """One layer resident once per stage. Separate per-support minima share scans."""
    def __init__(self,centers,delta,valid,offsets,device,query_chunk=64,reference_chunk=8192):
        import torch
        self.torch=torch;self.device=device;self.qchunk=query_chunk;self.rchunk=reference_chunk
        self.m=torch.tensor(centers,dtype=torch.float32,device=device)
        self.delta=torch.tensor(delta,dtype=torch.float32,device=device)
        self.valid=torch.tensor(valid,dtype=torch.bool,device=device);self.offsets=offsets

    def top(self,q,k=16):
        t=self.torch;out=[];ids=[];per=[];perids=[]
        with t.inference_mode():
            for start in range(0,len(q),self.qchunk):
                x=t.tensor(q[start:start+self.qchunk],dtype=t.float32,device=self.device)
                bd=t.empty((len(x),0),device=self.device);bi=t.empty((len(x),0),device=self.device,dtype=t.int64)
                pd=[];pi=[]
                for a,b in zip(self.offsets[:-1],self.offsets[1:]):
                    best=t.full((len(x),),float('inf'),device=self.device);winner=t.full((len(x),),b,device=self.device,dtype=t.int64)
                    for lo in range(a,b,self.rchunk):
                        hi=min(b,lo+self.rchunk)
                        d=t.cdist(x,self.m[lo:hi],compute_mode='use_mm_for_euclid_dist')
                        dv,di=d.min(1);di+=lo;improve=(dv<best)|((dv==best)&(di<winner))
                        best=t.where(improve,dv,best);winner=t.where(improve,di,winner)
                        v,j=torch_smallest(d,t.arange(lo,hi,device=self.device)[None],k)
                        bd,bi=torch_smallest(t.cat([bd,v],1),t.cat([bi,j],1),k)
                    pd.append(best);pi.append(winner)
                out.append((bd/np.float32(x.shape[1])).cpu().numpy());ids.append(bi.cpu().numpy())
                per.append((t.stack(pd)/np.float32(x.shape[1])).cpu().numpy());perids.append(t.stack(pi).cpu().numpy())
        return np.concatenate(out),np.concatenate(ids),np.concatenate(per,1),np.concatenate(perids,1)

    def components(self,q,qdelta,indices):
        """Bounded [query_chunk,16,4,D]; no full-image candidate tensor."""
        t=self.torch;parts=[[],[]]
        with t.inference_mode():
            for lo in range(0,len(q),self.qchunk):
                ids=t.tensor(indices[lo:lo+self.qchunk],device=self.device)
                x=t.tensor(q[lo:lo+len(ids)],device=self.device)
                dq=t.tensor(qdelta[lo:lo+len(ids)],device=self.device)
                diff=dq[:,None]-self.delta[ids]
                rel=t.linalg.vector_norm(diff,dim=-1)/np.float32(x.shape[1])
                nb=t.linalg.vector_norm((x[:,None]-self.m[ids])[:,:,None]+diff,dim=-1)/np.float32(x.shape[1])
                parts[0].append(nb.mean(-1).cpu().numpy());parts[1].append(rel.mean(-1).cpu().numpy())
        return np.concatenate(parts[0]),np.concatenate(parts[1])
