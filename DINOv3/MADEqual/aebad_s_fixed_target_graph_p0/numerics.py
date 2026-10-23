"""Exact finite-pool geometry, blocked CPU / explicitly selected CUDA.

No candidate truncation. FP64 GEMM identities have conservative forward-error
intervals; potentially winning/threshold-ambiguous terms use direct vectors.
"""
import numpy as np


class Backend:
    def __init__(self, device='cpu'):
        self.device = device
        self.torch = None
        if device != 'cpu':
            import torch
            self.torch = torch
            self.dev = torch.device('cpu' if device=='torch:cpu' else device)
            if self.dev.type != 'cuda' and device!='torch:cpu': raise ValueError('explicit cpu or cuda:N required')
            if self.dev.type=='cuda':torch.cuda.set_device(self.dev)
        self.direct_pairs = 0

    def array(self, a):
        if self.torch is None: return np.asarray(a, np.float64)
        return self.torch.as_tensor(a, dtype=self.torch.float64, device=self.dev)

    def numpy(self, a):
        return np.asarray(a) if self.torch is None else a.detach().cpu().numpy()

    def squared(self, q, r):
        """Squared endpoint distances plus an absolute FP64 error bound."""
        if self.torch is None:
            qn=np.einsum('ij,ij->i',q,q); rn=np.einsum('ij,ij->i',r,r)
            cross=q@r.T
        else:
            qn=(q*q).sum(1); rn=(r*r).sum(1); cross=q@r.T
        d2=qn[:,None]+rn[None,:]-2*cross
        # Cauchy bounds the absolute dot-product sum. Deliberately conservative.
        error=32*(q.shape[1]+8)*np.finfo(np.float64).eps*(
            qn[:,None]+rn[None,:]+2*(qn[:,None]*rn[None,:])**.5)
        if self.torch is None:
            if (d2 < -error).any():raise FloatingPointError('squared distance below FP64 error interval')
            rows,cols=np.where(d2<=error)
        else:
            if (d2 < -error).any().item():raise FloatingPointError('squared distance below FP64 error interval')
            rows,cols=self.torch.where(d2<=error)
        if len(rows):
            delta=q[rows]-r[cols]
            direct=(delta*delta).sum(-1)
            d2[rows,cols]=direct
            self.direct_pairs += len(rows)
        return d2,error


def direct_projection(q,u,v):
    """Independent reference: subtract original vectors, never norm identities."""
    q,u,v=(np.asarray(x,np.float64) for x in (q,u,v))
    direction=v-u; c2=np.sum(direction*direction,axis=-1)
    alpha=np.divide(np.sum((q-u)*direction,axis=-1),c2,
                    out=np.zeros(np.broadcast_shapes(q.shape[:-1],c2.shape)),where=c2>0)
    alpha=np.clip(alpha,0.,1.)
    residual=q-(u+alpha[...,None]*direction)
    return np.sqrt(np.sum(residual*residual,axis=-1)),alpha


def scan_segments(query,nodes,edges,masks,union,*,device='cpu',query_chunk=128,
                  edge_chunk=2048,reference_chunk=2048,progress=None,prepared=None):
    """Every query against every retained edge in each mask; no Top2 shortcut.

    Endpoints are evaluated once/query block. For each edge block/mask the
    minimum upper bound identifies ALL possible winners, independently refined
    with direct projection. Thus cancellation cannot create a winning zero.
    Return distances in HOST units and original canonical edge indices.
    """
    query=np.asarray(query);nodes=np.asarray(nodes);edges=np.asarray(edges,np.int64).reshape(-1,2)
    union=np.asarray(union,np.float32)
    if len(query)!=len(union) or query.ndim!=2 or nodes.ndim!=2 or query.shape[1]!=nodes.shape[1]:
        raise ValueError('query/node/union geometry mismatch')
    if not all(np.isfinite(x).all() for x in (query,nodes,union)) or (union<0).any():
        raise ValueError('nonfinite or negative graph input')
    masks=np.asarray(masks,bool).reshape(-1,len(edges)) if len(edges) else np.zeros((len(masks),0),bool)
    if len(edges) and (edges.min()<0 or edges.max()>=len(nodes)): raise ValueError('invalid edge endpoint')
    if min(query_chunk,edge_chunk,reference_chunk)<1: raise ValueError('positive blocks required')
    result=np.broadcast_to(union,(len(masks),len(query))).copy()
    winners=np.full(result.shape,-1,np.int64)
    if not len(edges) or not masks.any(): return result,winners,dict(direct_pairs=0,endpoint_pairs=0)
    if device!='cpu':
        return _scan_torch(query,nodes,edges,masks,union,device,query_chunk,edge_chunk,reference_chunk,progress,prepared)
    backend=Backend(device);x=backend.array(nodes)
    c2=np.empty(len(edges),np.float64)
    for start in range(0,len(edges),edge_chunk):
        es=edges[start:start+edge_chunk]
        c=nodes[es[:,1]].astype(np.float64)-nodes[es[:,0]].astype(np.float64)
        c2[start:start+len(es)]=np.einsum('ij,ij->i',c,c)
    if (c2<=0).any(): raise ValueError('zero-length edge must be removed before retrieval')
    for start in range(0,len(query),query_chunk):
        stop=min(start+query_chunk,len(query));q=backend.array(query[start:stop]); nq=stop-start
        # One endpoint-distance table is reused for every configuration/branch.
        endpoint=np.empty((nq,len(nodes)),np.float64); bounds=np.empty_like(endpoint)
        for r in range(0,len(nodes),reference_chunk):
            a,e=backend.squared(q,x[r:r+reference_chunk])
            endpoint[:,r:r+reference_chunk]=backend.numpy(a);bounds[:,r:r+reference_chunk]=backend.numpy(e)
        best=(union[start:stop].astype(np.float64)*1280)**2
        best=np.broadcast_to(best,(len(masks),nq)).copy()
        chosen=np.full(best.shape,-1,np.int64)
        for e in range(0,len(edges),edge_chunk):
            es=edges[e:e+edge_chunk];cc=c2[e:e+edge_chunk]
            active=masks[:,e:e+len(es)]
            if not active.any(): continue
            aa=endpoint[:,es[:,0]];bb=endpoint[:,es[:,1]]
            alpha=np.clip((aa+cc-bb)/(2*cc),0.,1.)
            square=(1-alpha)*aa+alpha*bb-alpha*(1-alpha)*cc
            error=16*(bounds[:,es[:,0]]+bounds[:,es[:,1]]+
                      np.finfo(np.float64).eps*(np.abs(aa)+np.abs(bb)+cc))
            if np.any(square < -error): raise FloatingPointError('segment square below error bound')
            needed=np.zeros(square.shape,bool)
            for k,mask in enumerate(active):
                if not mask.any(): continue
                upper=np.minimum(best[k],np.min(np.where(mask,square+error,np.inf),axis=1))
                needed |= mask & (square-error<=upper[:,None])
            exact=np.full(square.shape,np.inf)
            qr,er=np.where(needed)
            # Direct reference is executed on the selected device, in bounded batches.
            for z in range(0,len(qr),256):
                qi,ei=qr[z:z+256],er[z:z+256];u=x[es[ei,0]];v=x[es[ei,1]]
                direction=v-u;delta=q[qi]-u
                fraction=(delta*direction).sum(-1)/(direction*direction).sum(-1)
                fraction=np.clip(fraction,0,1) if backend.torch is None else fraction.clamp(0,1)
                residual=delta-fraction[:,None]*direction
                exact[qi,ei]=backend.numpy((residual*residual).sum(-1))
            backend.direct_pairs+=len(qr)
            for k,mask in enumerate(active):
                value=np.where(mask,exact,np.inf);idx=np.argmin(value,axis=1);d=value[np.arange(nq),idx]
                improve=d<best[k]
                # Canonical edge order resolves edge ties; UNION wins an exact tie.
                best[k,improve]=d[improve];chosen[k,improve]=e+idx[improve]
        converted=(np.sqrt(best)/1280).astype(np.float32)
        result[:,start:stop]=np.minimum(result[:,start:stop],converted)
        winners[:,start:stop]=chosen
        if progress: progress(stop,len(query))
    return result,winners,dict(direct_pairs=backend.direct_pairs,endpoint_pairs=len(query)*len(nodes))


class PreparedSearch:
    """Frozen graph tensors shared by consecutive queries; no model/GT inputs."""
    def __init__(self,nodes,edges,masks,device,edge_chunk=2048):
        self.backend=Backend(device);self.device=device
        torch=self.backend.torch
        if torch is None:raise ValueError('PreparedSearch is the explicitly selected Torch path')
        self.x=self.backend.array(nodes)
        self.edges=torch.as_tensor(edges,dtype=torch.int64,device=self.backend.dev)
        if edge_chunk<1:raise ValueError('positive edge block required')
        # Never materialize E x 1280 directions: a large legal graph can have
        # millions of edges. Only node vectors and scalar edge lengths persist.
        self.c2=torch.empty(len(edges),dtype=torch.float64,device=self.backend.dev)
        for start in range(0,len(edges),edge_chunk):
            es=self.edges[start:start+edge_chunk]
            c=self.x[es[:,1]]-self.x[es[:,0]]
            self.c2[start:start+len(es)]=(c*c).sum(-1)
        if (self.c2<=0).any().item():raise ValueError('zero edge must be removed')
        self.active=torch.as_tensor(masks,dtype=torch.bool,device=self.backend.dev)
        self.shape=(nodes.shape,edges.shape,masks.shape)


def _scan_torch(query,nodes,edges,masks,union,device,query_chunk,edge_chunk,reference_chunk,progress,prepared=None):
    """The production CUDA hot loop; torch:cpu tests execute this SAME code.

    Endpoint tables, edge distances, interval pruning, and direct refinements
    remain on device. Only final block minima and winning edge IDs go to CPU.
    """
    prepared=prepared or PreparedSearch(nodes,edges,masks,device,edge_chunk=edge_chunk)
    if prepared.device!=device or prepared.shape!=(nodes.shape,edges.shape,masks.shape):raise ValueError('prepared graph scope differs')
    backend=prepared.backend;before=backend.direct_pairs;torch=backend.torch;dev=backend.dev
    result=np.broadcast_to(union,(len(masks),len(query))).copy();winners=np.full(result.shape,-1,np.int64)
    with torch.inference_mode():
        x=prepared.x;es_all=prepared.edges;c2=prepared.c2;active_all=prepared.active
        for start in range(0,len(query),query_chunk):
            stop=min(start+query_chunk,len(query));q=backend.array(query[start:stop]);nq=len(q)
            endpoint=torch.empty((nq,len(nodes)),dtype=torch.float64,device=dev);bounds=torch.empty_like(endpoint)
            for r in range(0,len(nodes),reference_chunk):
                d,e=backend.squared(q,x[r:r+reference_chunk]);endpoint[:,r:r+reference_chunk]=d;bounds[:,r:r+reference_chunk]=e
            best=backend.array((union[start:stop].astype(np.float64)*1280)**2)[None].expand(len(masks),-1).clone()
            chosen=torch.full(best.shape,-1,dtype=torch.int64,device=dev)
            for e in range(0,len(edges),edge_chunk):
                es=es_all[e:e+edge_chunk];cc=c2[e:e+len(es)];active=active_all[:,e:e+len(es)]
                if not masks[:,e:e+len(es)].any():continue
                aa=endpoint[:,es[:,0]];bb=endpoint[:,es[:,1]]
                alpha=((aa+cc-bb)/(2*cc)).clamp(0,1)
                square=(1-alpha)*aa+alpha*bb-alpha*(1-alpha)*cc
                error=16*(bounds[:,es[:,0]]+bounds[:,es[:,1]]+
                    np.finfo(np.float64).eps*(aa.abs()+bb.abs()+cc))
                if (square < -error).any().item():raise FloatingPointError('segment square below FP64 interval')
                needed=torch.zeros(square.shape,dtype=torch.bool,device=dev)
                # Batch the 45 normal-LOO configurations, bounded to eight at
                # a time. Same per-mask minima, fewer tiny kernel launches.
                for first in range(0,len(masks),8):
                    last=min(first+8,len(masks));group=active[first:last]
                    upper=torch.minimum(best[first:last],torch.where(group[:,None,:],
                        (square+error)[None],torch.inf).min(2).values)
                    needed|=(group[:,None,:]&((square-error)[None]<=upper[:,:,None])).any(0)
                exact=torch.full_like(square,torch.inf);coords=torch.nonzero(needed)
                for z in range(0,len(coords),256):
                    qi,ei=coords[z:z+256].unbind(1);u=x[es[ei,0]];direction=x[es[ei,1]]-u
                    delta=q[qi]-u;fraction=((delta*direction).sum(-1)/(direction*direction).sum(-1)).clamp(0,1)
                    residual=delta-fraction[:,None]*direction;exact[qi,ei]=(residual*residual).sum(-1)
                backend.direct_pairs+=len(coords)
                for first in range(0,len(masks),8):
                    last=min(first+8,len(masks));group=active[first:last]
                    distance,idx=torch.where(group[:,None,:],exact[None],torch.inf).min(2)
                    improve=distance<best[first:last]
                    best[first:last]=torch.where(improve,distance,best[first:last])
                    chosen[first:last]=torch.where(improve,e+idx,chosen[first:last])
            result[:,start:stop]=np.minimum(result[:,start:stop],backend.numpy((best.sqrt()/1280).float()))
            winners[:,start:stop]=backend.numpy(chosen)
            if progress:progress(stop,len(query))
    return result,winners,dict(direct_pairs=backend.direct_pairs-before,endpoint_pairs=len(query)*len(nodes),
        backend='Torch FP64 device-resident exact interval/direct refinement')
