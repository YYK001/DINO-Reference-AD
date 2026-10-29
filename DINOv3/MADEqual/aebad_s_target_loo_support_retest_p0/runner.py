"""Default CPU inventory. All retrieval, extraction and ADEval require explicit stages."""
import argparse,json,os,sys,time,tarfile,subprocess,threading
from pathlib import Path
from . import inputs as ip
from .inputs import st
from .protocol import *


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight',*STAGES,'aggregate','pack','all','all-settings'))
    p.add_argument('--domain',choices=DOMAINS,default='background');p.add_argument('--support-seed',type=int,choices=SEEDS,default=42)
    for name in ('parent-run','parent-cache','raw-run','raw-cache','feature-manifest','legacy-provider'):p.add_argument('--'+name,type=Path)
    for name in ('weight42-run','weight42-cache'):p.add_argument('--'+name,action='append',default=[],help='DOMAIN=PATH')
    p.add_argument('--run-root',type=Path,default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_target_loo_two_K8_v1')
    p.add_argument('--cache-root',type=Path);p.add_argument('--evaluation-scratch',type=Path)
    p.add_argument('--data',type=Path,default=ip.ROOT/'datasets/AeBAD.zip');p.add_argument('--checkpoint',type=Path)
    p.add_argument('--official-root',type=Path,default=ip.ROOT/'external_baselines/SuperADD');p.add_argument('--dinov3-root',type=Path,default=ip.ROOT/'external_baselines/dinov3')
    p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8)
    p.add_argument('--supplement-retrieval',action='store_true');p.add_argument('--extract-missing',action='store_true');p.add_argument('--allow-over-12h',action='store_true')
    return p


def normalize(a):
    for k,v in vars(a).copy().items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    for k,suffix in [('cache_root','_cache'),('evaluation_scratch','_pixels')]:setattr(a,k,(getattr(a,k) or a.run_root.with_name(a.run_root.name+suffix)).resolve())
    for raw,attr in [('weight42_run','weight42_runs'),('weight42_cache','weight42_caches')]:
        mapping={}
        for value in getattr(a,raw):
            d,path=value.split('=',1)
            if d not in DOMAINS or d in mapping:raise ValueError('one explicit DOMAIN=PATH per provider')
            mapping[d]=Path(path).resolve()
        setattr(a,attr,mapping)
    protected=[x for x in (a.parent_run,a.parent_cache,a.raw_run,a.raw_cache,a.data,a.checkpoint,a.feature_manifest,a.legacy_provider,a.official_root,a.dinov3_root,*a.weight42_runs.values(),*a.weight42_caches.values()) if x]
    if a.parent_run and (a.parent_run/'assets.json').exists():protected.append(Path(st.read_json(a.parent_run/'assets.json')['references']['source']['path']).resolve())
    def overlap(x,y):return x==y or x.is_relative_to(y) or y.is_relative_to(x)
    destinations=(a.run_root,a.cache_root,a.evaluation_scratch)
    if any(overlap(x,y) for x in destinations for y in protected) or any(overlap(x,y) for i,x in enumerate(destinations) for y in destinations[i+1:]):raise ValueError('outputs/cache/scratch must be isolated from read-only parents')
    if a.extract_missing and not a.supplement_retrieval:raise ValueError('explicit extraction also requires explicit retrieval')
    return a


def pack(a):
    dest=a.run_root.with_name(a.run_root.name+'_review.tar.gz')
    with tarfile.open(dest,'w:gz',compresslevel=1) as t:
        for root,dirs,files in os.walk(a.run_root,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(root)/d).is_symlink() and d not in ('scratch','pixel_scratch')]
            for name in files:
                f=Path(root)/name
                if f.suffix in ('.json','.csv','.md','.log','.txt') and not f.is_symlink():t.add(f,arcname=a.run_root.name+'/'+f.relative_to(a.run_root).as_posix(),recursive=False)
    return dict(archive=str(dest),bytes=dest.stat().st_size,large_arrays_scratch_excluded=True)


def execute(a):
    if a.stage=='preflight':return ip.preflight(a)
    if a.stage=='pack':return pack(a)
    if a.stage=='aggregate':
        from .reporting import aggregate
        return aggregate(a)
    if set(a.weight42_runs)!=set(DOMAINS):raise ValueError('three explicit --weight42-run providers required')
    st.sealed_json(a.run_root/'protocol.json',CONFIG);ip.freeze(a)
    if a.stage in ('prepare-normal-stats','fit-weights','predict'):
        from . import pipeline
        return getattr(pipeline,a.stage.replace('-','_'))(a)
    if a.stage in ('evaluate','aupro'):
        from . import evaluation
        return getattr(evaluation,a.stage)(a)
    from .reporting import summarize
    return summarize(a)


def cost(a,tick,status):
    import psutil
    mem=psutil.Process().memory_info();peak=getattr(mem,'peak_wset',mem.rss)
    try:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except ImportError:pass
    gpu={};torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        prior=sys.modules.get('DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime')
        previous=getattr(prior,'PRIOR_GPU_PEAK',{})
        gpu=dict(peak_cuda_allocated_bytes=max(torch.cuda.max_memory_allocated(),previous.get('allocated',0),getattr(a,'peak_cuda_allocated_bytes',0)),peak_cuda_reserved_bytes=max(torch.cuda.max_memory_reserved(),previous.get('reserved',0),getattr(a,'peak_cuda_reserved_bytes',0)))
    st.write_json(ip.setting(a)/'costs'/f'{a.stage}_{time.time_ns()}.json',dict(stage=a.stage,domain=a.domain,support_seed=a.support_seed,status=status,seconds=time.perf_counter()-tick,process_peak_rss_bytes=peak,feature_forwards=getattr(a,'feature_forwards',0),exact_execution=getattr(a,'exact_execution',{}),new_array_payload_bytes=__import__('DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.storage',fromlist=['NEW_ARRAY_BYTES']).NEW_ARRAY_BYTES,**gpu))


def orchestrate(a,tokens):
    # Each unit is a fresh process so large feature/model references are released.
    tasks=[(d,s,stage) for d in DOMAINS for s in SEEDS for stage in STAGES] if a.stage=='all-settings' else [(a.domain,a.support_seed,stage) for stage in STAGES]
    filtered=[];i=0
    while i<len(tokens):
        if tokens[i] in ('--domain','--support-seed'):i+=2;continue
        if tokens[i].startswith(('--domain=','--support-seed=')):i+=1;continue
        filtered.append(tokens[i]);i+=1
    for d,seed,stage in tasks:
        log=a.run_root/d/f'seed{seed}'/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        st.write_json(a.run_root/'pipeline_status.json',dict(status='running',domain=d,support_seed=seed,stage=stage))
        print(f'DOMAIN={d} SUPPORT_SEED={seed} STAGE={stage} LOG={log}',flush=True)
        with log.open('a',encoding='utf-8') as f,subprocess.Popen([sys.executable,'-B','-u','-m',__package__+'.runner',stage,*filtered,'--domain',d,'--support-seed',str(seed)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
            for line in child.stdout:f.write(line);f.flush();print(line,end='',flush=True)
            rc=child.wait()
        print(f'STAGE={stage} EXIT_CODE={rc}',flush=True)
        if rc:
            st.write_json(a.run_root/'pipeline_status.json',dict(status='failed',domain=d,support_seed=seed,stage=stage,exit_code=rc))
            print(json.dumps(pack(a)));return rc
    if a.stage=='all-settings':
        rc=subprocess.call([sys.executable,'-B','-u','-m',__package__+'.runner','aggregate',*filtered])
        if rc:
            st.write_json(a.run_root/'pipeline_status.json',dict(status='failed',stage='aggregate',exit_code=rc))
            print(json.dumps(pack(a)));return rc
    st.write_json(a.run_root/'pipeline_status.json',dict(status='complete',scope=a.stage))
    print(json.dumps(pack(a)));return 0


def main(argv=None):
    tokens=list(sys.argv[1:] if argv is None else argv);a=normalize(parser().parse_args(tokens));a._started=time.perf_counter()
    if a.stage in ('all','all-settings'):return orchestrate(a,tokens[1:])
    if a.stage=='preflight':
        from collections import Counter
        r=execute(a);brief={k:v for k,v in r.items() if k not in ('reusable','missing','unconfirmed')}
        brief.update(reusable_by_role=dict(Counter(x.get('role','other') for x in r.get('reusable',[]))),missing_by_role=dict(Counter(x.get('role','other') for x in r.get('missing',[]))),details=str(a.run_root/'preflight.json'))
        print(json.dumps(brief,ensure_ascii=False,indent=2));return 2 if r['status']=='blocked' else 0
    a.run_root.mkdir(parents=True,exist_ok=True);tick=time.perf_counter();stop=threading.Event()
    def heartbeat():
        while not stop.wait(30):print(f'[heartbeat] {a.domain}/seed{a.support_seed} stage={a.stage} seconds={time.perf_counter()-tick:.0f}',flush=True)
    t=threading.Thread(target=heartbeat,daemon=True);t.start();status='failed'
    try:
        r=execute(a);status='complete';print(json.dumps(r,ensure_ascii=False,indent=2));return 2 if r.get('status')=='pending' else 0
    finally:
        stop.set();t.join()
        if a.stage not in ('pack','aggregate'):cost(a,tick,status)


if __name__=='__main__':raise SystemExit(main())
