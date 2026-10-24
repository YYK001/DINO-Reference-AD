"""FP32 graph search: one edge pass and fused device-local candidate refinement.

CUDA uses Triton explicitly, without a silent eager/FP64 fallback. CPU is a
small-data reference for local tests, not a claim of CUDA kernel validation.
"""
import time
import numpy as np
from DINOv3.MADEqual.aebad_s_fixed_graph_acceleration_p0.search import strict_screen, gamma

ENGINE = 'fp32_single_edge_pass_fused_refinement_v1'
EDGE_TILE = 512


def direct_fp32(q, u, v):
    """Reference with explicit FP32 subtract/multiply/reduce, no norm identity."""
    q,u,v=(np.asarray(a,np.float32) for a in (q,u,v))
    direction=v-u;delta=q-u
    length=np.sum(direction*direction,axis=-1,dtype=np.float32)
    if not np.isfinite(length).all() or (length<=0).any():
        raise FloatingPointError('active lengths not representable as positive FP32 squares')
    alpha=np.clip(np.sum(delta*direction,axis=-1,dtype=np.float32)/length,np.float32(0),np.float32(1))
    residual=delta-alpha[...,None]*direction
    return np.sum(residual*residual,axis=-1,dtype=np.float32)


class PreparedSearch:
    def __init__(self,nodes,edges,masks,device,edge_chunk=2048):
        import torch
        self.torch=torch;self.device=device
        self.dev=torch.device('cpu' if device=='torch:cpu' else device)
        if self.dev.type not in ('cpu','cuda'):raise ValueError('explicit CPU test or CUDA required')
        nodes=np.asarray(nodes);edges=np.asarray(edges,np.int64).reshape(-1,2);masks=np.asarray(masks,bool)
        if nodes.ndim!=2 or nodes.dtype!=np.float32 or not np.isfinite(nodes).all():
            raise ValueError('finite original FP32 nodes required')
        if masks.ndim!=2 or masks.shape[1]!=len(edges) or not 1<=len(masks)<=3 or edge_chunk<1:
            raise ValueError('one to three aligned masks and positive chunks required')
        if len(edges) and (edges.min()<0 or edges.max()>=len(nodes)):raise ValueError('invalid endpoints')
        self.original_nodes=nodes;self.original_edges=edges.copy();self.original_masks=masks.copy()
        # Deduplicate identical branch masks; expand results back to original
        # branch order after reduction. No query-dependent candidate removal.
        distinct,self.inverse=np.unique(masks,axis=0,return_inverse=True)
        self.groups=len(distinct)
        self.canonical=np.flatnonzero(distinct.any(0))
        self.node_ids,compact=np.unique(edges[self.canonical].ravel(),return_inverse=True)
        self.x=torch.as_tensor(nodes[self.node_ids].copy(),device=self.dev)
        self.edges=torch.as_tensor(compact.reshape(-1,2),device=self.dev)
        self.eu=self.edges[:,0].contiguous();self.ev=self.edges[:,1].contiguous()
        self.ids=torch.as_tensor(self.canonical,device=self.dev)
        self.active=torch.as_tensor(distinct[:,self.canonical].copy(),device=self.dev)
        flags=np.zeros(len(self.canonical),np.int32)
        for k in range(self.groups):flags|=distinct[k,self.canonical].astype(np.int32)<<k
        self.flags=torch.as_tensor(flags,device=self.dev)
        self.c2=torch.empty(len(self.canonical),dtype=torch.float32,device=self.dev)
        self.norms=torch.empty(len(self.x),dtype=torch.float32,device=self.dev)
        # A common support-only origin reduces cancellation in screening.
        # Formal projections below still use original raw rows, never these
        # shifted rows. The conservative interval covers FP32 subtraction.
        self.origin=self.x[0].clone() if len(self.x) else torch.zeros(nodes.shape[1],device=self.dev)
        self.centered=self.x-self.origin
        with torch.inference_mode():
            for r in range(0,len(self.x),edge_chunk):
                x=self.centered[r:r+edge_chunk];self.norms[r:r+len(x)]=(x*x).sum(-1)
            for r in range(0,len(self.edges),edge_chunk):
                es=self.edges[r:r+edge_chunk];delta=self.x[es[:,1]]-self.x[es[:,0]]
                self.c2[r:r+len(es)]=(delta*delta).sum(-1)
            if not torch.isfinite(self.norms).all().item() or not torch.isfinite(self.c2).all().item() or (self.c2<=0).any().item():
                raise FloatingPointError('active geometry outside representable FP32 squared range')
        self.gamma=np.float32(gamma(nodes.shape[1]+8,np.finfo(np.float32).eps))


def interval(torch,aa,bb,cc,qn,nu,nv,g):
    alpha=((aa+cc-bb)/(2*cc)).clamp(0,1)
    square=(1-alpha)*aa+alpha*bb-alpha*(1-alpha)*cc
    error=8*g*(2*qn+nu+nv+2*qn.sqrt()*(nu.sqrt()+nv.sqrt())+cc)
    error=error+4*np.finfo(np.float32).tiny
    usable=torch.isfinite(square)&torch.isfinite(error)
    lower=torch.where(usable,(square-error).clamp_min(0),0.)
    upper=torch.where(usable,(square+error).clamp_min(0),torch.inf)
    return lower,upper


def _cpu_block(p,q,cross,qn,anchor):
    """Production FP32 mathematical reference; no CUDA/Triton is claimed here."""
    t=p.torch;best=anchor[None].expand(p.groups,-1).clone()
    winner=t.full(best.shape,-1,dtype=t.int64,device=p.dev);refined=0
    for first in range(0,len(p.edges),EDGE_TILE):
        es=p.edges[first:first+EDGE_TILE];cc=p.c2[first:first+len(es)]
        nu,nv=p.norms[es[:,0]],p.norms[es[:,1]]
        aa=qn[:,None]+nu-2*cross[:,es[:,0]];bb=qn[:,None]+nv-2*cross[:,es[:,1]]
        low,high=interval(t,aa,bb,cc,qn[:,None],nu,nv,p.gamma)
        needed=t.zeros_like(low,dtype=t.bool)
        active=p.active[:,first:first+len(es)]
        for k in range(p.groups):
            upper=t.minimum(anchor,t.where(active[k],high,t.inf).min(1).values)
            needed|=active[k]&(low<=upper[:,None])
        coords=t.nonzero(needed);exact=t.full_like(low,t.inf)
        for r in range(0,len(coords),2048):
            qi,ei=coords[r:r+2048].unbind(1)
            u=p.x[es[ei,0]];direction=p.x[es[ei,1]]-u;delta=q[qi]-u
            alpha=((delta*direction).sum(-1)/(direction*direction).sum(-1)).clamp(0,1)
            residual=delta-alpha[:,None]*direction
            exact[qi,ei]=(residual*residual).sum(-1)
        refined+=len(coords)
        for k in range(p.groups):
            value,idx=t.where(active[k],exact,t.inf).min(1)
            take=value<best[k]
            best[k]=t.where(take,value,best[k]);winner[k]=t.where(take,p.ids[first+idx],winner[k])
    return best,winner,refined


def scan_segments(query,nodes,edges,masks,union,*,device='cpu',query_chunk=128,edge_chunk=2048,
                  reference_chunk=2048,progress=None,prepared=None):
    import torch
    started=time.perf_counter();query=np.asarray(query);nodes=np.asarray(nodes)
    edges=np.asarray(edges,np.int64).reshape(-1,2);masks=np.asarray(masks,bool);union=np.asarray(union)
    if (query.ndim!=2 or nodes.ndim!=2 or query.shape[1]!=nodes.shape[1] or query.dtype!=np.float32 or
            union.dtype!=np.float32 or union.shape!=(len(query),) or not np.isfinite(query).all() or
            not np.isfinite(union).all() or (union<0).any()):raise ValueError('finite FP32 query/union required')
    if min(query_chunk,edge_chunk,reference_chunk)<1:raise ValueError('positive execution chunks required')
    p=prepared or PreparedSearch(nodes,edges,masks,device,edge_chunk)
    if (p.device!=device or p.original_nodes is not nodes or not np.array_equal(p.original_edges,edges)
            or not np.array_equal(p.original_masks,masks)):raise ValueError('prepared scope differs')
    result=np.broadcast_to(union,(len(masks),len(query))).copy();winners=np.full(result.shape,-1,np.int64)
    diagnostics=dict(backend=ENGINE,geometry_dtype='FP32',full_edge_passes=1,original_edges=len(edges),
                     active_edges=len(p.edges),original_nodes=len(nodes),active_nodes=len(p.x),
                     original_masks=len(masks),distinct_masks=p.groups,endpoint_pairs=len(query)*len(p.x),
                     direct_pairs=0,endpoint_gemms=0,fused_edge_launches=0,
                     execution='CUDA_Triton' if p.dev.type=='cuda' else 'Torch_CPU_reference')
    if not len(p.edges):
        diagnostics['seconds']=time.perf_counter()-started;return result,winners,diagnostics
    if p.dev.type=='cuda':
        # No dependency download and no silent return to the old slow path.
        from .kernels import fused_scan
    with torch.inference_mode(),strict_screen(torch,p.dev):
        for first in range(0,len(query),query_chunk):
            stop=min(first+query_chunk,len(query));q=torch.as_tensor(query[first:stop],device=p.dev).contiguous()
            screen=q-p.origin;qn=(screen*screen).sum(-1)
            anchor=(torch.as_tensor(union[first:stop],device=p.dev)*1280).square()
            if not torch.isfinite(qn).all().item() or not torch.isfinite(anchor).all().item():
                raise FloatingPointError('query/anchor outside representable FP32 squared range')
            # One wide GEMM per query block, not dozens of 2048-node GEMMs.
            cross=screen@p.centered.T;diagnostics['endpoint_gemms']+=1
            if p.dev.type=='cuda':
                distance,win,refined=fused_scan(p,q,cross,qn,anchor,EDGE_TILE)
                diagnostics['fused_edge_launches']+=1
            else:distance,win,refined=_cpu_block(p,q,cross,qn,anchor)
            values=(distance.sqrt()/1280).float().cpu().numpy()
            chosen=win.cpu().numpy()[p.inverse];values=values[p.inverse]
            # An FP32 square/sqrt round trip can differ by one ULP. Preserve
            # the exact stored UNION for no winner/empty masks and ties.
            improve=(chosen>=0)&(values<union[None,first:stop])
            result[:,first:stop]=np.where(improve,values,union[None,first:stop])
            winners[:,first:stop]=np.where(improve,chosen,-1)
            diagnostics['direct_pairs']+=int(refined)
            if progress:progress(stop,len(query))
    diagnostics['seconds']=time.perf_counter()-started
    return result,winners,diagnostics
