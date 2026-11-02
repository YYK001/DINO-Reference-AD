"""Default inventory only; explicit CPU statistics or authorized supplementation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from . import inputs as ip,pipeline,reporting
from DINOv3.MADEqual.aebad_s_shared_residual_translation_p0.runner import pack,cost,overlap


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight','smoke','prepare-normal','fit','collect-query','predict','evaluate','summarize','pack','all'))
    p.add_argument('--parent-run',type=Path,default=ip.ROOT/'server_results/aebad_target_loo_two_K8_20261030_013433')
    p.add_argument('--run-root',type=Path,default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_conditional_image_score_preflight')
    for name in ('cache-root','source-bank','memory42-root','memory43-root','raw-cache','sacp-run','support-feature-root',
                 'feature-manifest','statistics-provider','response-provider','retrieval-provider','checkpoint'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--data',type=Path,default=ip.ROOT/'datasets/AeBAD.zip')
    p.add_argument('--official-root',type=Path,default=ip.ROOT/'external_baselines/SuperADD')
    p.add_argument('--dinov3-root',type=Path,default=ip.ROOT/'external_baselines/dinov3')
    p.add_argument('--device',default='cpu')
    p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8)
    p.add_argument('--reference-cache-gib',type=float,default=8.)
    for name in ('supplement-retrieval','extract-missing','allow-over-12h'):p.add_argument('--'+name,action='store_true')
    return p


def normalize(a):
    for k,v in vars(a).copy().items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    a.cache_root=a.cache_root or a.run_root.with_name(a.run_root.name+'_cache')
    # Metadata only: defaults refer to the current machine's frozen parent assets.
    p=ip.parent(a)
    a.raw_cache=a.raw_cache or Path(p.inputs['raw_cache'])
    a.sacp_run=a.sacp_run or Path(p.inputs['parent_run'])
    protected=[v.resolve() for k,v in vars(a).items() if isinstance(v,Path) and k not in ('run_root','cache_root')]
    protected += [Path(p.inputs['source']['path']).resolve(),Path(p.inputs['parent_cache']).resolve()]
    if overlap(a.run_root,a.cache_root) or any(overlap(x,y) for x in (a.run_root,a.cache_root) for y in protected):
        raise ValueError('new output/cache must be separate from parent assets and each other')
    if a.reference_cache_gib<=0 or (a.device!='cpu' and (not a.device.startswith('cuda:') or not a.device[5:].isdigit())):
        raise ValueError('one explicit device and positive resident cache size required')
    a._started=time.perf_counter();return a


def execute(a):
    if a.stage=='pack':return pack(a)
    if a.stage=='preflight':return ip.preflight(a)
    ip.freeze(a)
    if a.stage=='smoke':return pipeline.smoke(a)
    if a.stage=='prepare-normal':return pipeline.run_observations(a,'normal')
    if a.stage=='collect-query':return pipeline.run_observations(a,'query')
    if a.stage in ('fit','predict','evaluate'):return getattr(pipeline,a.stage)(a)
    if a.stage=='summarize':return reporting.summarize(a)
    raise ValueError('unsupported stage')


def main(argv=None):
    args=parser().parse_args(argv)
    # Pack does not require parent entities to remain mounted.
    if args.stage=='pack':
        args.run_root=args.run_root.resolve();print(json.dumps(pack(args)));return 0
    try:a=normalize(args)
    except FileNotFoundError as e:
        if args.stage!='preflight':raise
        if any(overlap(args.run_root,v.resolve()) for k,v in vars(args).items() if isinstance(v,Path) and k not in ('run_root','cache_root')):
            raise ValueError('preflight output overlaps an input') from e
        args.run_root.mkdir(parents=True,exist_ok=True)
        result=dict(status='blocked',error=str(e),torch_imported='torch' in sys.modules)
        ip.st.write_json(args.run_root/'preflight.json',result)
        ip.st.write_csv(args.run_root/'missing_inputs.csv',[dict(role='parent_metadata',reason=str(e))])
        ip.st.write_csv(args.run_root/'asset_inventory.csv',[])
        print(json.dumps(result,ensure_ascii=False));return 2
    a.run_root.mkdir(parents=True,exist_ok=True)
    if a.stage=='all':
        ip.freeze(a)
        if (a.supplement_retrieval or a.extract_missing) and not (a.run_root/'smoke.json').exists():
            raise ValueError('run explicit real-asset smoke before full supplementation')
        pipeline.budget(a)
        common=list(sys.argv[2:] if argv is None else argv[1:])
        for stage in ('prepare-normal','fit','collect-query','predict','evaluate','summarize','pack'):
            log=a.run_root/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            print(f'STAGE={stage} LOG={log}',flush=True)
            with log.open('a',encoding='utf-8') as f:
                child=subprocess.Popen([sys.executable,'-B','-u','-m',__package__+'.runner',stage,*common],stdout=f,stderr=subprocess.STDOUT)
                while True:
                    try:rc=child.wait(timeout=30);break
                    except subprocess.TimeoutExpired:print(f'[heartbeat] conditional stage={stage}',flush=True)
            print(f'STAGE={stage} EXIT_CODE={rc}',flush=True)
            if rc:
                print(json.dumps(pack(a)),flush=True);return rc
        return 0
    tick=time.perf_counter()
    try:result=execute(a)
    except Exception as e:
        cost(a,tick,dict(status='stage_failed',error=f'{type(e).__name__}: {e}'));raise
    cost(a,tick,result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 2 if result.get('status') in ('missing_inputs','blocked','pending') else 0


if __name__=='__main__':raise SystemExit(main())
