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
        np.testing.assert_allclose(s['delta_U'],t['delta_U'],atol=1e-7,rtol=1e-6)
        cases.append(dict(seed=seed,sources=k,positions=n,max_distance_abs_error=error,verified=True))
    if torch.cuda.is_initialized():raise RuntimeError('unexpected CUDA initialization')
    return dict(status='stock_torch_CPU_verified',torch=torch.__version__,device='cpu',cuda_initialized=False,cases=cases)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();r=run()
    text=json.dumps(r,indent=2)
    if a.output:a.output.write_text(text+'\n',encoding='utf8')
    print(text)
