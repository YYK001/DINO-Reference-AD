"""Optional stock tensor-logic validation ON CPU; never initialize CUDA."""
import argparse
from pathlib import Path
import time
import numpy as np
from . import numerics as nm
from .inputs import st


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('torch_cpu_validation.json'))
    args=parser.parse_args(argv)
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only verification expected no initialized CUDA')
    rng=np.random.default_rng(66);records=[];started=time.perf_counter()
    for kind in ('unequal_normal_counts','extreme_tail','lower_queries'):
        s=rng.normal(size=(4,31)).astype(np.float32);t=rng.normal(size=(8,4,31)).astype(np.float32)
        refs=[(rng.normal(size=(n,4)).astype(np.float32),rng.normal(size=(n,4)).astype(np.float32)) for n in (1,3,5,7,2,8,4,6)]
        if kind=='extreme_tail':s+=1000;t+=1000
        cpu=nm._numpy(s,t,refs,3,2);stock=nm._torch(s,t,refs,'cpu',3,2)
        error=float(np.max(np.abs(cpu.astype(float)-stock)))
        if not np.allclose(cpu,stock,atol=1e-6,rtol=1e-6):raise ValueError('stock tensor CPU mismatch: '+kind)
        if kind=='lower_queries':
            lower=nm._torch(s-1,t-1,refs,'cpu',5,3)
            if not np.all(lower>=stock-1e-6):raise ValueError('lower query decreased log probability')
        if torch.cuda.is_initialized():raise RuntimeError('unexpected CUDA initialization')
        records.append(dict(case=kind,max_log_probability_abs_error=error,verified=True))
    result=dict(status='stock_torch_batched_logic_CPU_verified',torch=torch.__version__,device='cpu',cuda_initialized=False,
                cases=records,seconds=time.perf_counter()-started,limitation='not CUDA backend or real assets validation')
    st.write_json(args.output,result);print(result)
    return 0


if __name__=='__main__':raise SystemExit(main())
