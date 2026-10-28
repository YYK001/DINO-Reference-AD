"""Default inventory only. Every GPU/model stage is an explicit invocation."""
import argparse,json,os,sys,time,subprocess,tarfile
from pathlib import Path
from . import inputs as ip
from .inputs import st
from .protocol import STAGES,CONFIG


class BudgetPause(RuntimeError):pass


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight',*STAGES,'all','pack','smoke'))
    for name in ('parent-run','parent-cache','raw-cache','feature-root','feature-manifest','catalog','source-bank','selected-indices','source-neighborhood','augmented-grids'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--run-root',type=Path,default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_same_origin_neighbor_seed42_v1')
    for name in ('cache-root','evaluation-scratch','build-scratch'):p.add_argument('--'+name,type=Path)
    p.add_argument('--data',type=Path,default=ip.ROOT/'datasets/AeBAD.zip')
    p.add_argument('--checkpoint',type=Path,default=ip.ROOT/'external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
    p.add_argument('--official-root',type=Path,default=ip.ROOT/'external_baselines/SuperADD')
    p.add_argument('--dinov3-root',type=Path,default=ip.ROOT/'external_baselines/dinov3')
    p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8)
    p.add_argument('--query-chunk',type=int,default=64);p.add_argument('--reference-chunk',type=int,default=8192)
    p.add_argument('--rebuild-source',action='store_true',help='explicit independent original Source rebuild; recompute baselines')
    p.add_argument('--baseline-mode',choices=('auto','fresh'),default='auto',help='fresh recomputes baselines with the SAME bank; does not rebuild Source')
    p.add_argument('--allow-over-12h',action='store_true',help='use ONLY after explicit user approval of exceeding experiment budget')
    return p


def overlap(a,b):return a==b or a.is_relative_to(b) or b.is_relative_to(a)


def isolation(a):
    protected=[p for p in (a.parent_run,a.parent_cache,a.raw_cache,a.feature_root,a.feature_manifest,a.catalog,a.source_bank,a.selected_indices,a.source_neighborhood,a.augmented_grids,a.data,a.checkpoint,a.official_root,a.dinov3_root) if p is not None]
    # Also protect a Source path discovered through parent metadata, even when
    # the user did not repeat it as --source-bank.
    if a.parent_run and (a.parent_run/'assets.json').exists():
        protected.append(Path(st.read_json(a.parent_run/'assets.json')['references']['source']['path']))
    destinations=[a.run_root,a.cache_root,a.evaluation_scratch,a.build_scratch]
    if any(overlap(d,p.resolve()) for d in destinations for p in protected):raise ValueError('output/scratch overlaps read-only parent/input')
    if any(overlap(d,e) for i,d in enumerate(destinations) for e in destinations[i+1:]):raise ValueError('run/cache/evaluation/build scratch must be separate')


def normalize(a):
    for k,v in vars(a).copy().items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    for key,suffix in [('cache_root','_cache'),('evaluation_scratch','_pixels'),('build_scratch','_build')]:
        setattr(a,key,(getattr(a,key) or a.run_root.with_name(a.run_root.name+suffix)).resolve())
    if a.query_chunk<1 or a.reference_chunk<1:raise ValueError('positive execution chunks required')
    isolation(a);return a


def enforce_time_budget(a,stage,done,total):
    elapsed=time.perf_counter()-getattr(a,'_started',time.perf_counter())
    previous=sum(st.read_json(p)['seconds'] for p in (a.run_root/'costs').glob('*.json'))
    unit_elapsed=time.perf_counter()-a._unit_started if hasattr(a,'_unit_started') else elapsed
    fixed=max(0.,elapsed-unit_elapsed)
    projected=previous+fixed+unit_elapsed*total/max(done,1)
    # Remaining-stage central estimates are explicit and conservative, not zeros.
    tail={'prepare-source':2.5,'prepare-target':2.25,'calibrate':1.5,'predict':.25,'evaluate':.1,'aupro':0}.get(stage,0)*3600
    if not a.allow_over_12h and (previous+elapsed>12*3600 or (done>=min(5,total) and projected+tail>12*3600)):
        value=dict(status='paused_time_budget',stage=stage,done=done,total=total,elapsed_hours=(previous+elapsed)/3600,
            projected_total_hours=(projected+tail)/3600,note='resume at committed image/job; >12h needs explicit user approval; no resource admission gate')
        st.write_json(a.run_root/'budget_pause.json',value)
        raise BudgetPause(json.dumps(value))


def cost(a,tick,result,status):
    import psutil
    rss=psutil.Process().memory_info();peak=getattr(rss,'peak_wset',rss.rss)
    try:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except ImportError:pass
    gpu={};torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():gpu=dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    logical=allocated=0;seen=set()
    for root,dirs,files in os.walk(a.cache_root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(root)/d).is_symlink()]
        for name in files:
            f=Path(root)/name
            if f.is_symlink():continue
            info=f.stat();token=(info.st_dev,info.st_ino)
            if token in seen:continue
            seen.add(token);logical+=info.st_size;allocated+=getattr(info,'st_blocks',0)*512
    st.write_json(a.run_root/'costs'/f'{a.stage}_{time.time_ns()}.json',dict(stage=a.stage,status=status,seconds=time.perf_counter()-tick,
        process_peak_rss_bytes=peak,new_array_payload_bytes=__import__('DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.storage',fromlist=['NEW_ARRAY_BYTES']).NEW_ARRAY_BYTES,
        cache_retained_logical_bytes=logical,cache_retained_allocated_bytes=allocated if os.name!='nt' else None,
        disk_scope='new own cache only; excludes read-only external providers; scratch/candidate temporary peak reported separately',
        original_four_layer_forwards=getattr(a,'_new_forwards',0),**gpu))


def pack(a):
    p=a.run_root.with_name(a.run_root.name+'_review.tar.gz')
    with tarfile.open(p,'w:gz',compresslevel=1) as t:
        for root,dirs,files in os.walk(a.run_root,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(root)/d).is_symlink() and d not in ('pixel_scratch','scratch')]
            for name in files:
                f=Path(root)/name
                if f.suffix in ('.json','.csv','.md','.txt','.log','.png') and not f.is_symlink():t.add(f,arcname=a.run_root.name+'/'+f.relative_to(a.run_root).as_posix(),recursive=False)
    return dict(archive=str(p),bytes=p.stat().st_size,large_arrays_and_scratch_excluded=True)


def execute(a):
    if a.stage in ('prepare-source','prepare-target'):
        from . import runtime
        return runtime.prepare_source(a) if a.stage=='prepare-source' else runtime.prepare_target(a)
    if a.stage in ('calibrate','predict'):
        from . import pipeline
        return getattr(pipeline,a.stage)(a)
    if a.stage=='smoke':
        from .smoke import smoke
        return smoke(a)
    if a.stage in ('evaluate','aupro'):
        from . import evaluation
        return getattr(evaluation,a.stage)(a)
    from .reporting import summarize
    return summarize(a)


def wait_for_child(child,reader,stage,heartbeat_seconds=30):
    """Wait on process lifetime, not the stdout reader (which may finish first)."""
    started=time.monotonic()
    while True:
        try:
            child.wait(timeout=heartbeat_seconds)
            break
        except subprocess.TimeoutExpired:
            print(f'[heartbeat] stage={stage} seconds={time.monotonic()-started:.0f}',flush=True)
    reader.join()


def main(argv=None):
    a=normalize(parser().parse_args(argv));a._started=time.perf_counter()
    if a.stage=='preflight':
        r=ip.preflight(a);st.write_json(a.run_root/'preflight.json',r)
        st.write_csv(a.run_root/'missing_inputs.csv',r['missing']);st.write_csv(a.run_root/'feature_inventory.csv',r.get('feature_references',[])+[dict(x,status='original_forward_needed') for x in r.get('features_to_extract',[])])
        print(json.dumps(r,ensure_ascii=False,indent=2));return 0 if r['status']=='inventory_ready_runtime_proof_pending' else 2
    if a.stage=='pack':print(json.dumps(pack(a)));return 0
    if a.stage=='all':
        # Fresh subprocess per stage releases the original host/reference tensors.
        base=list(argv if argv is not None else sys.argv[1:]);base=base[1:] if base and base[0]=='all' else base
        st.write_json(a.run_root/'pipeline_status.json',dict(status='running',success=False))
        try:
            for stage in STAGES:
                log=a.run_root/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                print(f'STAGE={stage} LOG={log}',flush=True)
                with log.open('a',encoding='utf-8') as f:
                    child=subprocess.Popen([sys.executable,'-B','-u','-m',__package__+'.runner',stage,*base],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                    import threading
                    def copy():
                        for line in child.stdout:f.write(line);f.flush();print(line,end='',flush=True)
                    reader=threading.Thread(target=copy,daemon=True);reader.start()
                    wait_for_child(child,reader,stage)
                if child.returncode:
                    status='paused_time_budget' if child.returncode==75 else 'stage_failed'
                    st.write_json(a.run_root/'pipeline_status.json',dict(status=status,stage=stage,success=False,exit_code=child.returncode));return child.returncode
            result=st.read_json(a.run_root/'status.json');ok=result['status']=='all_five_metrics_complete'
            st.write_json(a.run_root/'pipeline_status.json',dict(status='complete' if ok else 'pending',success=ok,seconds=time.perf_counter()-a._started))
            return 0 if ok else 2
        finally:print(json.dumps(pack(a)),flush=True)
    if a.stage!='prepare-source':ip.freeze(a)
    tick=time.perf_counter();status='complete';result={}
    try:
        result=execute(a);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True);return 0
    except BudgetPause as e:
        status='paused_time_budget';print(str(e),flush=True);return 75
    except Exception:
        status='failed';raise
    finally:cost(a,tick,result,status)


if __name__=='__main__':raise SystemExit(main())
