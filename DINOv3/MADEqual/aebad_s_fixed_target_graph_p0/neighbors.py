"""One direct FP64 distance block updates BOTH directed support Top100 tables."""
import numpy as np


def bidirectional(query,reference,k=100,*,device='cpu',query_chunk=128,reference_chunk=2048,progress=None):
    q,r=np.asarray(query),np.asarray(reference)
    if min(len(q),len(r),k,query_chunk,reference_chunk)<1 or q.shape[1]!=r.shape[1]:
        raise ValueError('nonempty compatible support banks and positive blocks required')
    nq,nr=len(q),len(r);kq,kr=min(k,nr),min(k,nq)
    if not np.isfinite(q).all() or not np.isfinite(r).all():raise ValueError('nonfinite support feature')
    if device=='cpu':
        from scipy.spatial.distance import cdist
        qi=np.full((nq,kq),-1,np.int64);qd=np.full((nq,kq),np.inf)
        ri=np.full((nr,kr),-1,np.int64);rd=np.full((nr,kr),np.inf)
        def merge(ids,dists,new_i,new_d,keep):
            ids=np.concatenate((ids,new_i),axis=1);dists=np.concatenate((dists,new_d),axis=1)
            order=np.argsort(dists,axis=1,kind='stable')[:,:keep]
            return np.take_along_axis(ids,order,1),np.take_along_axis(dists,order,1)
        for u in range(0,nq,query_chunk):
            stop=min(u+query_chunk,nq)
            for v in range(0,nr,reference_chunk):
                end=min(v+reference_chunk,nr)
                d=cdist(q[u:stop].astype(np.float64),r[v:end].astype(np.float64),'euclidean')
                ix=np.argsort(d,axis=1,kind='stable')[:,:kq]
                qi[u:stop],qd[u:stop]=merge(qi[u:stop],qd[u:stop],ix+v,np.take_along_axis(d,ix,1),kq)
                ix=np.argsort(d.T,axis=1,kind='stable')[:,:kr]
                ri[v:end],rd[v:end]=merge(ri[v:end],rd[v:end],ix+u,np.take_along_axis(d.T,ix,1),kr)
            if progress:progress(stop,nq)
        return (qi,qd),(ri,rd)
    import torch
    dev=torch.device('cpu' if device=='torch:cpu' else device)
    if dev.type=='cuda':torch.cuda.set_device(dev)
    with torch.inference_mode():
        q=torch.as_tensor(q,dtype=torch.float64,device=dev);r=torch.as_tensor(r,dtype=torch.float64,device=dev)
        qi=torch.full((nq,kq),-1,dtype=torch.int64,device=dev);qd=torch.full((nq,kq),torch.inf,dtype=torch.float64,device=dev)
        ri=torch.full((nr,kr),-1,dtype=torch.int64,device=dev);rd=torch.full((nr,kr),torch.inf,dtype=torch.float64,device=dev)
        def merge(ids,dists,new_i,new_d,keep):
            ids=torch.cat((ids,new_i),1);dists=torch.cat((dists,new_d),1)
            order=torch.argsort(dists,dim=1,stable=True)[:,:keep]
            return torch.gather(ids,1,order),torch.gather(dists,1,order)
        for u in range(0,nq,query_chunk):
            stop=min(u+query_chunk,nq)
            for v in range(0,nr,reference_chunk):
                end=min(v+reference_chunk,nr)
                d=torch.cdist(q[u:stop],r[v:end],compute_mode='donot_use_mm_for_euclid_dist')
                ix=torch.argsort(d,dim=1,stable=True)[:,:kq]
                qi[u:stop],qd[u:stop]=merge(qi[u:stop],qd[u:stop],ix+v,torch.gather(d,1,ix),kq)
                ix=torch.argsort(d.T,dim=1,stable=True)[:,:kr]
                ri[v:end],rd[v:end]=merge(ri[v:end],rd[v:end],ix+u,torch.gather(d.T,1,ix),kr)
            if progress:progress(stop,nq)
        return (qi.cpu().numpy(),qd.cpu().numpy()),(ri.cpu().numpy(),rd.cpu().numpy())
