"""Lazy CUDA-only Triton kernels. Not imported by preflight or CPU tests.

One program handles one query and one edge tile. Candidate selection and all
FP32 direct refinements stay inside that GPU program. No Python per-edge
launch, .item(), nonzero(), or CPU transfer occurs in the edge traversal.
"""
import triton
import triton.language as tl


@triton.jit
def _scan(Q,X,EU,EV,C2,NORM,FLAGS,IDS,CROSS,QN,ANCHOR,OUT,WIN,COUNT,
          D:tl.constexpr,N:tl.constexpr,E:tl.constexpr,NB:tl.constexpr,
          GAMMA:tl.constexpr,B:tl.constexpr,BD:tl.constexpr):
    qi=tl.program_id(0);block=tl.program_id(1)
    lane=tl.arange(0,B);edge=block*B+lane;valid=edge<E
    u=tl.load(EU+edge,valid,other=0);v=tl.load(EV+edge,valid,other=0)
    cc=tl.load(C2+edge,valid,other=1.)
    flag=tl.load(FLAGS+edge,valid,other=0)
    qnorm=tl.load(QN+qi);nu=tl.load(NORM+u);nv=tl.load(NORM+v)
    aa=qnorm+nu-2.*tl.load(CROSS+qi*N+u)
    bb=qnorm+nv-2.*tl.load(CROSS+qi*N+v)
    alpha=tl.minimum(1.,tl.maximum(0.,(aa+cc-bb)/(2.*cc)))
    square=(1.-alpha)*aa+alpha*bb-alpha*(1.-alpha)*cc
    error=8.*GAMMA*(2.*qnorm+nu+nv+2.*tl.sqrt(qnorm)*(tl.sqrt(nu)+tl.sqrt(nv))+cc)+4.7019774e-38
    usable=(tl.abs(square)<float('inf'))&(error<float('inf'))
    lo=tl.where(usable,tl.maximum(0.,square-error),0.)
    hi=tl.where(usable,tl.maximum(0.,square+error),float('inf'))
    anchor=tl.load(ANCHOR+qi)
    upper0=tl.minimum(anchor,tl.min(tl.where((flag&1)!=0,hi,float('inf')),0))
    upper1=tl.minimum(anchor,tl.min(tl.where((flag&2)!=0,hi,float('inf')),0))
    upper2=tl.minimum(anchor,tl.min(tl.where((flag&4)!=0,hi,float('inf')),0))
    needed=valid&((((flag&1)!=0)&(lo<=upper0))|(((flag&2)!=0)&(lo<=upper1))|(((flag&4)!=0)&(lo<=upper2)))
    best0=anchor;best1=anchor;best2=anchor
    win0=tl.full((),-1,tl.int64);win1=tl.full((),-1,tl.int64);win2=tl.full((),-1,tl.int64)
    checked=tl.full((),0,tl.int32)
    dim=tl.arange(0,BD)
    while tl.sum(needed.to(tl.int32),0)>0:
        chosen=tl.min(tl.where(needed,lane,2147483647),0)
        e=block*B+chosen
        ui=tl.load(EU+e);vi=tl.load(EV+e)
        uvec=tl.load(X+ui*D+dim,dim<D,other=0.)
        vvec=tl.load(X+vi*D+dim,dim<D,other=0.)
        qvec=tl.load(Q+qi*D+dim,dim<D,other=0.)
        direction=vvec-uvec;delta=qvec-uvec
        fraction=tl.minimum(1.,tl.maximum(0.,tl.sum(delta*direction,0)/tl.sum(direction*direction,0)))
        residual=delta-fraction*direction;value=tl.sum(residual*residual,0)
        canonical=tl.load(IDS+e);f=tl.load(FLAGS+e)
        take0=((f&1)!=0)&(value<best0);take1=((f&2)!=0)&(value<best1);take2=((f&4)!=0)&(value<best2)
        best0=tl.where(take0,value,best0);win0=tl.where(take0,canonical,win0)
        best1=tl.where(take1,value,best1);win1=tl.where(take1,canonical,win1)
        best2=tl.where(take2,value,best2);win2=tl.where(take2,canonical,win2)
        checked+=1
        needed=needed&(lane!=chosen)&((((flag&1)!=0)&(lo<=best0))|(((flag&2)!=0)&(lo<=best1))|(((flag&4)!=0)&(lo<=best2)))
    base=(qi*NB+block)*3
    tl.store(OUT+base,best0);tl.store(OUT+base+1,best1);tl.store(OUT+base+2,best2)
    tl.store(WIN+base,win0);tl.store(WIN+base+1,win1);tl.store(WIN+base+2,win2)
    tl.store(COUNT+qi*NB+block,checked)


def fused_scan(p,q,cross,qn,anchor,block):
    torch=p.torch;nq=len(q);nb=triton.cdiv(len(p.edges),block)
    distance=torch.empty((nq,nb,3),device=p.dev,dtype=torch.float32)
    winners=torch.empty((nq,nb,3),device=p.dev,dtype=torch.int64)
    count=torch.empty((nq,nb),device=p.dev,dtype=torch.int32)
    _scan[(nq,nb)](q,p.x,p.eu,p.ev,p.c2,p.norms,p.flags,p.ids,cross,qn,anchor,
                   distance,winners,count,q.shape[1],len(p.x),len(p.edges),nb,float(p.gamma),
                   block,triton.next_power_of_2(q.shape[1]),num_warps=4,enable_fp_fusion=False)
    # These are small block minima, not another traversal of the edge pool.
    values,where=distance[:,:,:p.groups].min(1)
    selected=winners[:,:,:p.groups].gather(1,where[:,None,:]).squeeze(1)
    return values.T,selected.T,int(count.sum().item())
