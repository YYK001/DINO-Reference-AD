"""Inventory by default; no connection, installs, model or GPU in preflight."""
import argparse,os,sys,time,json,tarfile,subprocess,threading
from pathlib import Path
from . import inputs as ip
from .inputs import st
from .protocol import CONFIG,STAGES


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight',*STAGES,'all','pack'))
    for name in ('parent-run','parent-cache','raw-run','raw-cache','evaluation-scratch') :p.add_argument('--'+name,type=Path)
    p.add_argument('--run-root',type=Path,default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_target_loo_weight_seed42_v1')
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--data',type=Path,default=ip.ROOT/'datasets/AeBAD.zip')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--official-root',type=Path,default=ip.ROOT/'external_baselines/SuperADD')
    p.add_argument('--dinov3-root',type=Path,default=ip.ROOT/'external_baselines/dinov3')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8)
    p.add_argument('--supplement-retrieval',action='store_true')
    p.add_argument('--extract-missing',action='store_true')
    p.add_argument('--allow-over-12h',action='store_true',help='only after explicit user approval')
    return p


def normalize(a):
    for k,v in vars(a).copy().items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    for k,suffix in [('cache_root','_cache'),('evaluation_scratch','_pixels')]:setattr(a,k,(getattr(a,k) or a.run_root.with_name(a.run_root.name+suffix)).resolve())
    protected=[p for p in (a.parent_run,a.parent_cache,a.raw_run,a.raw_cache,a.data,a.checkpoint,a.official_root,a.dinov3_root) if p]
    if a.parent_run and (a.parent_run/'assets.json').exists():protected.append(Path(st.read_json(a.parent_run/'assets.json')['references']['source']['path']).resolve())
    def overlap(x,y):return x==y or x.is_relative_to(y) or y.is_relative_to(x)
    destinations=[a.run_root,a.cache_root,a.evaluation_scratch]
    if any(overlap(x,y) for x in destinations for y in protected) or any(overlap(x,y) for i,x in enumerate(destinations) for y in destinations[i+1:]):raise ValueError('output/cache/scratch must be isolated from parent and inputs')
    if a.extract_missing and not a.supplement_retrieval:raise ValueError('--extract-missing also requires explicit --supplement-retrieval')
    return a


def pack(a):
    archive=a.run_root.with_name(a.run_root.name+'_review.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:
        for root,dirs,files in os.walk(a.run_root,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(root)/d).is_symlink() and d not in ('scratch','pixel_scratch')]
            for name in files:
                f=Path(root)/name
                if f.suffix in ('.json','.csv','.md','.log','.txt') and not f.is_symlink():t.add(f,arcname=a.run_root.name+'/'+f.relative_to(a.run_root).as_posix(),recursive=False)
    return dict(archive=str(archive),bytes=archive.stat().st_size,large_arrays_scratch_excluded=True)


def execute(a):
    if a.stage=='preflight':return ip.preflight(a)
    if a.stage=='pack':return pack(a)
    ip.freeze(a)
    st.sealed_json(a.run_root/'protocol.json',CONFIG)
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
    process=psutil.Process();peak=getattr(process.memory_info(),'peak_wset',process.memory_info().rss)
    try:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except ImportError:pass
    gpu={};torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():gpu=dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    st.write_json(a.run_root/'costs'/f'{a.stage}_{time.time_ns()}.json',dict(stage=a.stage,status=status,seconds=time.perf_counter()-tick,process_peak_rss_bytes=peak,feature_forwards=getattr(a,'feature_forwards',0),new_array_payload_bytes=__import__('DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.storage',fromlist=['NEW_ARRAY_BYTES']).NEW_ARRAY_BYTES,**gpu))
    st.write_csv(a.run_root/'stage_costs.csv',[st.read_json(p) for p in sorted((a.run_root/'costs').glob('*.json'))])


def main(argv=None):
    a=normalize(parser().parse_args(argv));a._started=time.perf_counter()
    if a.stage=='all':
        tokens=list(sys.argv[1:] if argv is None else argv);tokens=tokens[1:]
        a.run_root.mkdir(parents=True,exist_ok=True)
        for stage in STAGES:
            log=a.run_root/'logs'/(stage+'.log');log.parent.mkdir(exist_ok=True)
            print(f'STAGE={stage} LOG={log}',flush=True)
            with log.open('a',encoding='utf-8') as f:
                with subprocess.Popen([sys.executable,'-B','-u','-m',__package__+'.runner',stage,*tokens],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
                    for line in child.stdout:
                        f.write(line);f.flush();print(line,end='',flush=True)
                    rc=child.wait()
            print(f'STAGE={stage} EXIT_CODE={rc}',flush=True)
            if rc:print(json.dumps(pack(a)));return rc
        print(json.dumps(pack(a)));return 0
    if a.stage=='preflight':
        r=execute(a);print(json.dumps(r,ensure_ascii=False,indent=2));return 0 if r['status']=='inputs_available' else 2
    a.run_root.mkdir(parents=True,exist_ok=True);tick=time.perf_counter();stop=threading.Event()
    def heartbeat():
        while not stop.wait(30):print(f'[heartbeat] stage={a.stage} seconds={time.perf_counter()-tick:.0f}',flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start();status='failed'
    try:
        r=execute(a);status='complete';print(json.dumps(r,ensure_ascii=False,indent=2))
        return 2 if r.get('status')=='pending' else 0
    finally:stop.set();thread.join();cost(a,tick,status)


if __name__=='__main__':raise SystemExit(main())
