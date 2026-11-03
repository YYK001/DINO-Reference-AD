"""Deterministic spatial agglomeration. Original physical patches count once."""
import heapq
import math
import numpy as np


def physical(features,positions,grid):
    n=int(np.prod(grid));pos=np.asarray(positions)
    if len(grid)!=2 or min(grid)<1 or pos.shape!=(n,) or pos.dtype.kind not in 'iu' or not np.array_equal(np.sort(pos),np.arange(n)):
        raise ValueError('complete unique physical grid positions required; memory rows/overlapping tokens are invalid')
    order=np.argsort(pos);out=[]
    for x in features:
        x=np.asarray(x)
        if x.dtype!=np.float32 or x.ndim!=2 or x.shape[0]!=n or not np.isfinite(x).all():
            raise ValueError('finite complete FP32 layer features required')
        out.append(np.array(x[order],dtype=np.float32,copy=True))
    if len(out)!=4 or len({x.shape for x in out})!=1:raise ValueError('four aligned physical layers required')
    return np.stack(out,axis=1)  # physical patch, layer, channel


def distances_to(means,a,others,rms):
    """FP32 direct differences, no cancellation-prone squared norm subtraction."""
    result=[]
    for start in range(0,len(others),128):
        ids=others[start:start+128]
        delta=means[a][None]/rms[None,:,None]-means[ids]/rms[None,:,None]
        norms=np.sqrt(np.sum(delta*delta,axis=2,dtype=np.float32))
        result.extend((np.mean(norms,axis=1,dtype=np.float32)/np.float32(math.sqrt(means.shape[2]))).tolist())
    return result


def partition(features,positions,grid,rms,target=256):
    means=physical(features,positions,grid);n,_,dim=means.shape
    rms=np.asarray(rms,np.float32)
    if rms.shape!=(4,) or not np.isfinite(rms).all() or np.any(rms<=0):raise ValueError('four positive Source RMS scales required')
    if not 1<=target<=n:raise ValueError('not enough valid physical patches for fixed region count')
    h,w=grid;neighbors=[set() for _ in range(n)]
    for p in range(n):
        if p%w:neighbors[p].add(p-1)
        if p%w+1<w:neighbors[p].add(p+1)
        if p>=w:neighbors[p].add(p-w)
        if p+w<n:neighbors[p].add(p+w)
    counts=np.ones(n,np.int64);parent=np.arange(n,dtype=np.int64)
    version=np.zeros(n,np.int64);active=np.ones(n,bool);heap=[]
    # Batch initial edges; never form N x N distances or all feature differences.
    pairs=[(p,q) for p in range(n) for q in sorted(neighbors[p]) if p<q]
    for start in range(0,len(pairs),128):
        block=pairs[start:start+128];u,v=np.array(block).T
        delta=means[u]/rms[None,:,None]-means[v]/rms[None,:,None]
        d=np.mean(np.sqrt(np.sum(delta*delta,axis=2,dtype=np.float32)),axis=1,dtype=np.float32)/np.float32(math.sqrt(dim))
        heap.extend((float(x),int(a),int(b),0,0) for x,(a,b) in zip(d,block))
    heapq.heapify(heap)
    def valid(item):
        _,a,b,va,vb=item
        return active[a] and active[b] and version[a]==va and version[b]==vb
    for step in range(n-target):
        while heap and not valid(heap[0]):heapq.heappop(heap)
        if not heap:raise ValueError('physical adjacency disconnected before fixed region count')
        _,a,b,_,_=heapq.heappop(heap)  # a < b; kept root is minimum patch index.
        total=counts[a]+counts[b]
        means[a]=(means[a]*np.float32(counts[a])+means[b]*np.float32(counts[b]))/np.float32(total)
        counts[a]=total;parent[b]=a;active[b]=False;version[a]+=1
        joined=(neighbors[a]|neighbors[b])-{a,b}
        for q in joined:neighbors[q].discard(b);neighbors[q].add(a)
        neighbors[a]=joined;neighbors[b].clear()
        ids=sorted(joined)
        for q,d in zip(ids,distances_to(means,a,ids,rms)):
            u,v=sorted((a,q));heapq.heappush(heap,(d,u,v,int(version[u]),int(version[v])))
        if step and step%2048==0:
            heap=[item for item in heap if valid(item)];heapq.heapify(heap)
    roots=np.flatnonzero(active);labels=np.empty(n,np.int32)
    root_index={int(r):i for i,r in enumerate(roots)}
    for i in range(n):
        j=i
        while parent[j]!=j:j=int(parent[j])
        labels[i]=root_index[j]
    return dict(labels=labels,counts=counts[roots],area=counts[roots].astype(np.float64)/n,
        means=means[roots].transpose(1,0,2).copy(),roots=roots,grid=np.array(grid,np.int64))


def distance_matrix(template,query,rms,device='cpu'):
    a,b=np.asarray(template,np.float32),np.asarray(query,np.float32)
    r=np.asarray(rms,np.float32)
    if a.ndim!=3 or b.ndim!=3 or a.shape[0]!=4 or b.shape[0]!=4 or a.shape[2]!=b.shape[2] or r.shape!=(4,):
        raise ValueError('aligned four layer region means required')
    if not all(np.isfinite(x).all() for x in (a,b,r)) or np.any(r<=0):raise ValueError('invalid region descriptors')
    result=np.zeros((a.shape[1],b.shape[1]),np.float32)
    if device=='cpu':
        layers=np.empty((4,a.shape[1],b.shape[1]),np.float32)
        for l in range(4):
            x=a[l]/r[l];y=b[l]/r[l]
            for i in range(a.shape[1]):
                diff=x[i][None]-y
                layers[l,i]=np.sqrt(np.sum(diff*diff,axis=1,dtype=np.float32))
        result=np.mean(layers,axis=0,dtype=np.float32)/np.float32(math.sqrt(a.shape[2]))
    else:
        import torch
        # Stock PyTorch only. FP32 direct cdist, no TF32/MM distance identity.
        with torch.inference_mode():
            x=torch.as_tensor(a,device=device);y=torch.as_tensor(b,device=device)
            scale=torch.as_tensor(r,device=device)
            d=torch.cdist(x/scale[:,None,None],y/scale[:,None,None],p=2,compute_mode='donot_use_mm_for_euclid_dist')
            result=(d.mean(0)/math.sqrt(a.shape[2])).cpu().numpy()
    if not np.isfinite(result).all() or np.any(result<0):raise ValueError('invalid matching distances')
    return result
