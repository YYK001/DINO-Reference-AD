"""Optional independent-process stock Torch CPU check. Never initializes CUDA."""
import argparse
import json
from pathlib import Path
import numpy as np
from . import numerics as nm,runtime as rt


def run():
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CPU validation must not initialize CUDA')
    class Search:
        def query(self,q):return torch.tensor(q,device='cpu')
        def reference(self,b,key):return torch.tensor(b,device='cpu')
    cases=[]
    for seed,k,n in ((0,9,17),(1,9,33),(2,1,8)):
        rng=np.random.default_rng(seed);q=rng.normal(size=(n,1280)).astype(np.float32)
        banks=[rng.normal(size=(11+s,1280)).astype(np.float32) for s in range(k)]
        indices=[];distances=[]
        for bank in banks:
            d=nm.norm(q[:,None]-bank[None]);i=d.argmin(1);indices.append(i);distances.append(d[np.arange(n),i]/1280)
        ids=np.asarray(indices,np.int64);ds=np.asarray(distances,np.float32)
        x,s,_=nm.translate(q,banks,ids,ds)
        y,t,_=rt.translated_gpu(q,banks,ids,ds,Search(),list(range(k)),5)
        error=max(float(np.abs(x[b]-y[b]).max()) for b in x)
        for b in x:np.testing.assert_allclose(x[b],y[b],atol=1e-7,rtol=1e-6)
        for key in ('zero_fixed','zero_reselect'):np.testing.assert_allclose(s[key],t[key],atol=1e-7,rtol=1e-6)
        np.testing.assert_allclose(s['delta_U'],t['delta_U'],atol=1e-7,rtol=1e-6)
        cases.append(dict(seed=seed,sources=k,positions=n,max_distance_abs_error=error,verified=True))
    # Genuine MM-vs-subtraction cancellation, not an invalid NN or missing bank.
    from DINOv3.relation_reliability.superadd_host import _exact_top1_chunked
    rng=np.random.default_rng(19);cancellation=[]
    for offset in (0.,1.,8.):
        q=(rng.normal(0,.1,(4,1280))+offset).astype(np.float32)
        bank=q+rng.normal(0,.01,q.shape).astype(np.float32)
        exact,ids=_exact_top1_chunked(torch.tensor(q,device='cpu'),torch.tensor(bank,device='cpu'))
        old=exact.numpy()/np.float32(1280);ii=ids.numpy()
        direct=nm.norm(q-bank[ii])/np.float32(1280)
        ref=np.linalg.norm(q.astype(np.float64)-bank[ii].astype(np.float64),axis=1)/1280
        np.testing.assert_allclose(direct,ref,atol=1e-7,rtol=1e-6)
        cancellation.append(dict(offset=offset,max_MM_vs_direct_abs_error=float(np.abs(old-direct).max()),
            max_direct_vs_small_FP64_reference_error=float(np.abs(direct-ref).max()),
            positions_rejected_by_old_cross_path_gate=int(np.sum(np.abs(old-direct)>1e-7+1e-6*np.abs(old)))))
    if not any(r['positions_rejected_by_old_cross_path_gate'] for r in cancellation):raise AssertionError('cancellation reproduction absent')
    q=np.array([[0.],[10.],[20.]],np.float32);banks=[np.array([[0.]],np.float32),np.array([[1.]],np.float32)]
    ids=np.zeros((2,3),np.int64);ds=np.array([[0.,0.,0.],[1.,1.,1.]],np.float32)
    vector,vs,vd=nm.translate(q,banks,ids,ds)
    tensor,ts,td=rt.translated_gpu(q,banks,ids,ds,Search(),[0,1],2)
    np.testing.assert_array_equal(ts['union_source'],[0,1,1]);np.testing.assert_array_equal(ts['union_source'],vs['union_source'])
    for b in vector:np.testing.assert_allclose(vector[b],tensor[b],atol=1e-7,rtol=1e-6)
    assert td['legacy_source_switch_count']==vd['legacy_source_switch_count']==2
    if torch.cuda.is_initialized():raise RuntimeError('unexpected CUDA initialization')
    return dict(status='stock_torch_CPU_verified',torch=torch.__version__,device='cpu',cuda_initialized=False,cases=cases,
                synthetic_MM_cancellation=cancellation,direct_selection_case=dict(winners=ts['union_source'].tolist(),legacy_switch_count=2,verified=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();r=run()
    text=json.dumps(r,indent=2)
    if a.output:a.output.write_text(text+'\n',encoding='utf8')
    print(text)
