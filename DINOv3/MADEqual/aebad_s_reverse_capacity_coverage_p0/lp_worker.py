"""Persistent cuOpt process in the isolated solver environment. JSON stdin jobs."""
import json
import os
import sys
import time
import subprocess
from pathlib import Path
import numpy as np
from . import inputs as ip,gpu_lp,lp_v2
st=ip.st


def memory():
    out=dict(peak_RSS_bytes=None,GPU_memory_bytes_after_batch=None,GPU_identity=None)
    try:
        import resource
        out['peak_RSS_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    except ImportError:pass
    try:
        r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=3)
        values=[float(line.split(',')[1])*2**20 for line in r.stdout.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
        if values:out['GPU_memory_bytes_after_batch']=int(sum(values))
        r=subprocess.run(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],
            capture_output=True,text=True,timeout=3)
        if r.returncode==0:out['GPU_identity']=r.stdout.strip()
    except (OSError,ValueError,subprocess.TimeoutExpired):pass
    return out


def main():
    if sys.argv[1:]==['--versions']:
        print(json.dumps(dict(python=sys.version,packages=gpu_lp.packages())));return 0
    device=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not (device.isdigit() or device.startswith('GPU-')) or ',' in device:raise ValueError('one explicitly selected physical GPU required')
    solver=None
    for line in sys.stdin:
        if not line.strip():continue
        job=st.sealed_json(Path(json.loads(line)['job']));response=Path(job['response']);started=time.perf_counter()
        try:
            if job['protocol']!=lp_v2.PROTOCOL:raise ValueError('worker numeric protocol differs')
            if solver is None:solver=lp_v2.Solver(job['cpu_workers'])
            problems=[]
            for item in job['items']:
                data,record=st.get(item['bundle'])
                if record['seal']!=item['bundle_seal']:raise ValueError('distance inputs changed')
                problems.append(dict(id=item['id'],distance=data['distance'],a=data['a'],b=data['b'],
                    reference_objective=item.get('reference_objective')))
            solved,timing=solver.solve_certified(problems);records=[]
            for item,result in zip(job['items'],solved):
                proof=st.put(Path(item['proof']),dict(primal=result['primal'],dual=result['dual']),
                    gpu_lp.finite_record(dict(protocol=lp_v2.PROTOCOL,dependencies=item['dependencies'],
                        audit=result['audit'],solver=timing['solver'],settings=solver.parameters,versions=gpu_lp.packages())))
                records.append(dict(id=item['id'],proof=item['proof'],seal=proof['seal'],audit=result['audit']))
            result=dict(status='job_complete',records=records,solver=timing['solver'],settings=solver.parameters,
                versions=gpu_lp.packages(),resource=memory(),error=None)
        except Exception as e:result=dict(status='job_failed',records=[],error=f'{type(e).__name__}: {e}')
        # Includes reads, initialization when needed, model build, transfers,
        # solve, independent audit and proof writes; no success-only timing.
        st.sealed_json(response,gpu_lp.finite_record(dict(job=job['seal'],seconds=time.perf_counter()-started,**result)))
        print(f'[mixed LP worker] {result["status"]} pairs={len(result["records"])} seconds={time.perf_counter()-started:.3f}',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
