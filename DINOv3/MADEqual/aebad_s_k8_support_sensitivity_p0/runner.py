"""Explicit staged CLI. Default is read-only preflight without Torch imports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from . import inputs as ip
from .selection import DOMAINS
from .storage import json_bytes


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=(
        'preflight','prepare','capture','predict','evaluate','evaluate-aupro','summarize'))
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[3])
    for name in ('formal-dir','image-state','data','memory','checkpoint','dinov3','official','output-root'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--domains',nargs='+',choices=DOMAINS,default=list(DOMAINS))
    p.add_argument('--cpu-workers',type=int,default=4)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--capture-min-free-vram-gib',type=float,default=20)
    p.add_argument('--aupro-allocator-limit-gib',type=float,default=1.5)
    p.add_argument('--memory-budget-gib',type=float,default=8)
    p.add_argument('--work-dir',type=Path)
    p.add_argument('--inventory-output',type=Path,help='explicit opt-in inventory file, only at preflight')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if len(set(args.domains))!=len(args.domains): raise ValueError('duplicate domains')
    if not 1<=args.cpu_workers<=8: raise ValueError('CPU workers must be 1..8')
    if not 8<=args.capture_min_free_vram_gib<=128: raise ValueError('explicit VRAM admission must be finite 8..128 GiB')
    if not 0<args.memory_budget_gib<1024 or not 0<args.aupro_allocator_limit_gib<=128:
        raise ValueError('positive finite resource budgets required')
    paths={k:Path(v).resolve() for k,v in ip.defaults(args.project_root.resolve()).items()}
    for name in paths:
        value=getattr(args,name,None)
        if value is not None: paths[name]=value.resolve()
    ctx=None; started=time.perf_counter()
    try:
        if args.stage=='preflight':
            ctx=ip.load(paths) if paths['output_root'].exists() else None
            result=ip.inventory(paths,ctx)
            result['existing_evidence']=ip.local_evidence(paths['project'])
            if ctx:
                result['resume_records']={kind:sum(k.startswith(kind+'/') for k in ctx['ledger'].state['records'])
                    for kind in ('support','capture','prediction','prediction_freeze','cpu','aupro')}
                result['identity']=ctx['identity']
            if args.inventory_output:
                args.inventory_output.parent.mkdir(parents=True,exist_ok=True)
                if args.inventory_output.exists(): raise ValueError('inventory output exists; no overwrite')
                args.inventory_output.write_bytes(json_bytes(result))
        elif args.stage=='prepare':
            ctx=ip.prepare(paths); result=dict(status='manifests_frozen',identity=ctx['identity'])
        else:
            if args.inventory_output: raise ValueError('--inventory-output only applies to preflight')
            ctx=ip.load(paths)
            if args.stage=='capture':
                from .capture import run
                result=run(ctx,args.domains,args.device,args.capture_min_free_vram_gib)
            elif args.stage=='predict':
                from .prediction import run
                result=run(ctx,args.domains,args.cpu_workers)
            elif args.stage=='evaluate':
                from .evaluation import run_cpu
                result=run_cpu(ctx,args.domains,workers=args.cpu_workers,memory_budget=int(args.memory_budget_gib*2**30),
                    work_dir=args.work_dir or paths['project']/'downloads/k8_support_sensitivity_scratch')
            elif args.stage=='evaluate-aupro':
                from .evaluation import run_aupro
                result=run_aupro(ctx,args.domains,device=args.device,allocator_limit_gib=args.aupro_allocator_limit_gib)
            else:
                from .reporting import run
                result=run(ctx)
        result.update(stage=args.stage,seconds=time.perf_counter()-started,torch_imported='torch' in sys.modules)
        if ctx and args.stage!='preflight':
            from .evaluation import resources
            result.update(resources())
            result['retained_output_bytes']=sum(p.stat().st_size for p in paths['output_root'].rglob('*') if p.is_file())
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            key='invocations/'+stamp
            ctx['ledger'].commit(key,result,{key+'.json':json_bytes(result)})
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),flush=True)
        return 2 if args.stage=='preflight' and result.get('status','').startswith(('blocked','missing')) else 0
    except Exception as error:
        failure=dict(stage=args.stage,status='stage_failed',error=f'{type(error).__name__}: {error}',
                     seconds=time.perf_counter()-started,checks=getattr(error,'checks',[]))
        # Original records are never rewritten on failure. A fresh failure unit is
        # append-only; orphan detection errors are printed without touching ledger.
        if ctx and 'ledger' in ctx:
            try:
                ctx['ledger'].verify()
                key='failures/'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                ctx['ledger'].commit(key,failure,{key+'.json':json_bytes(failure)})
            except Exception: pass
        print(json.dumps(failure,ensure_ascii=False,indent=2),flush=True)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
