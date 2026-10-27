"""Independent staged tool. Default read-only preflight never imports Torch."""
import argparse,json,os,sys,time,subprocess,threading,tarfile
from pathlib import Path
from importlib.metadata import version
from . import inputs as ip
from .inputs import st
from .protocol import CONFIG
STAGES=('prepare','fit','predict','evaluate','evaluate-aupro','summarize')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight',*STAGES,'all','pack'))
    p.add_argument('--run-root',type=Path,default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_s_sacp_seed42_v1')
    p.add_argument('--cache-root',type=Path);p.add_argument('--evaluation-scratch',type=Path)
    p.add_argument('--recent-run',type=Path,action='append',default=[])
    p.add_argument('--data',type=Path,default=ip.ROOT/'datasets/AeBAD.zip')
    p.add_argument('--checkpoint',type=Path,default=ip.ROOT/'external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
    p.add_argument('--official-root',type=Path,default=ip.ROOT/'external_baselines/SuperADD')
    p.add_argument('--dinov3-root',type=Path,default=ip.ROOT/'external_baselines/dinov3')
    p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8)
    p.add_argument('--reference-cache-gib',type=float,default=6.)
    p.add_argument('--inventory-dir',type=Path,help='explicit metadata-only preflight report directory')
    return p


def overlap(a,b):return a==b or a.is_relative_to(b) or b.is_relative_to(a)


def isolation(a):
    protected=[a.data,a.checkpoint,a.official_root,a.dinov3_root,*a.recent_run]
    for parent in a.recent_run:
        if (parent/'inputs.json').exists():
            value=st.sealed_json(parent/'inputs.json')
            if value.get('cache'):protected.append(Path(value['cache']).resolve())
    if (a.run_root/'assets.json').exists():
        assets=ip.assets(a)
        protected.extend(Path(v['path']).resolve() for v in assets['normal_features'].values())
        protected.extend(Path(v['path']).resolve() for v in assets['references'].values() if isinstance(v,dict) and 'path' in v)
    for dest in (a.run_root,a.cache_root,a.evaluation_scratch):
        if any(overlap(dest,p.resolve()) for p in protected):raise ValueError('output/scratch overlaps read-only parent/input')
    if overlap(a.cache_root,a.evaluation_scratch) or a.evaluation_scratch==a.run_root or a.run_root.is_relative_to(a.evaluation_scratch):raise ValueError('scratch must not contain run or overlap cache')
    if a.cache_root==a.run_root or a.run_root.is_relative_to(a.cache_root):raise ValueError('cache cannot contain run')


def normalize(a):
    a.run_root=a.run_root.resolve();a.cache_root=(a.cache_root or a.run_root/'cache').resolve();a.evaluation_scratch=(a.evaluation_scratch or a.run_root/'pixel_scratch').resolve()
    for k in ('data','checkpoint','official_root','dinov3_root'):setattr(a,k,getattr(a,k).resolve())
    a.recent_run=[p.resolve() for p in a.recent_run]
    if a.device not in ('cpu','cuda') and not (a.device.startswith('cuda:') and a.device[5:].isdigit()):raise ValueError('one explicitly selected device required')
    if not 0<=a.reference_cache_gib<=64:raise ValueError('invalid reference cache budget')
    isolation(a);return a


def validate(a):
    value=st.sealed_json(a.run_root/'inputs.json')
    if value['config']!=CONFIG or value['job_plan']!=ip.plan(a) or value['cache']!=str(a.cache_root):raise ValueError('fixed SACP config/cache changed')
    if st.sealed_json(a.run_root/'assets.json')['seal']!=value['assets_seal'] or st.digest(ip.catalog(a))!=value['catalog']:raise ValueError('frozen asset/catalog metadata changed')
    for name,h in value['code'].items():
        if st.sha(Path(__file__).with_name(name))!=h:raise ValueError('SACP code changed since prepare')
    for name in ('numpy','scipy','Pillow'):
        if version(name)!=ip.assets(a)['references']['extraction']['software'][name]:raise ValueError('frozen map software changed')


def run_stage(a):
    from . import runtime,prediction,evaluation,reporting
    if a.stage=='prepare':
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=a.cpu_workers,user_api='blas'):return runtime.prepare(a)
    validate(a)
    if a.stage=='fit':
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=a.cpu_workers,user_api='blas'):return runtime.fit(a)
    if a.stage=='predict':return prediction.predict(a)
    if a.stage.startswith('evaluate'):return evaluation.evaluate(a,only_aupro=a.stage=='evaluate-aupro')
    return reporting.summarize(a)


def common(a):
    result=[]
    for k in ('run_root','cache_root','evaluation_scratch','data','checkpoint','official_root','dinov3_root','device','cpu_workers','reference_cache_gib'):result.extend(('--'+k.replace('_','-'),str(getattr(a,k))))
    for p in a.recent_run:result.extend(('--recent-run',str(p)))
    return result


def pack(a):
    dest=a.run_root.with_name(a.run_root.name+'_review.tar.gz');allowed={'.json','.csv','.md','.log','.txt','.png','.svg'}
    with tarfile.open(dest,'w:gz',compresslevel=1) as tar:
        for root,dirs,files in os.walk(a.run_root,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(root)/d).is_symlink() and not any((Path(root)/d).resolve()==p or (Path(root)/d).resolve().is_relative_to(p) for p in (a.cache_root,a.evaluation_scratch))]
            for name in files:
                p=Path(root)/name
                if p.suffix in allowed and not p.is_symlink():tar.add(p,arcname=a.run_root.name+'/'+str(p.relative_to(a.run_root)).replace('\\','/'),recursive=False)
        for folder in ('clusters','subspaces'):
            root=a.cache_root/folder
            if root.exists():tar.add(root,arcname=a.run_root.name+'/frozen_fit/'+folder)
    return dict(archive=str(dest),bytes=dest.stat().st_size,large_prediction_and_raw_Q_cache_excluded=True)


def job_gpu_peaks(result,current):
    # ADEval resets counters per job. Aggregate only jobs computed THIS invocation.
    values=[v for v in result if isinstance(v,dict) and v.get('runtime_reused') is False] if isinstance(result,list) else []
    return {key:max([current.get(key,0)]+[v.get(key,0) for v in values]) for key in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes')}


def all_stages(a):
    start=time.monotonic();stage='preflight';success=False;error=None
    try:
        r=ip.preflight(a)
        if r['status']=='blocked':print(json.dumps(r,ensure_ascii=False));raise ValueError('missing raw assets; no reconstruction')
        for stage in STAGES:
            st.write_json(a.run_root/'pipeline_status.json',dict(status='running',stage=stage,success=False))
            log=a.run_root/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True);done=threading.Event();tick=time.monotonic()
            def heartbeat():
                while not done.wait(30):print(f'[heartbeat] stage={stage} seconds={time.monotonic()-tick:.0f}',flush=True)
            thread=threading.Thread(target=heartbeat,daemon=True);thread.start();print(f'STAGE={stage} LOG={log}',flush=True)
            try:
                with log.open('a',encoding='utf-8') as f:
                    process=subprocess.Popen([sys.executable,'-B','-u','-m',__spec__.name,stage,*common(a)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                    try:
                        for line in process.stdout:print(line,end='',flush=True);f.write(line);f.flush()
                        rc=process.wait()
                    except BaseException:process.terminate();process.wait();raise
                if rc:raise RuntimeError(stage+' failed: '+str(rc))
            finally:done.set();thread.join()
        success=st.read_json(a.run_root/'status.json')['status']=='all_five_metrics_complete'
        if not success:raise RuntimeError('pending metrics remain')
        return dict(success=True,status='all_five_metrics_complete',seconds=time.monotonic()-start)
    except BaseException as e:error=type(e).__name__+': '+str(e);raise
    finally:
        st.write_json(a.run_root/'pipeline_status.json',dict(status='complete' if success else 'failed',stage=stage,success=success,error=error,seconds=time.monotonic()-start))
        print(json.dumps(pack(a)),flush=True)


def main(argv=None):
    a=normalize(parser().parse_args(argv));st.configure('fast')
    if a.stage=='preflight':
        r=ip.preflight(a)
        if a.inventory_dir:
            a.inventory_dir.mkdir(parents=True,exist_ok=True);st.write_json(a.inventory_dir/'preflight.json',r)
            st.write_csv(a.inventory_dir/'missing_inputs.csv',r.get('missing',[]));st.write_csv(a.inventory_dir/'repeated_work.csv',r.get('repeated_work',[]))
        print(json.dumps(r,ensure_ascii=False,indent=2));return 2 if r['status']=='blocked' else 0
    if a.stage=='all':result=all_stages(a)
    elif a.stage=='pack':result=pack(a)
    else:
        start=time.monotonic();result=run_stage(a)
        import psutil
        info=psutil.Process().memory_info();gpu={};torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():gpu=dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        if a.stage=='evaluate-aupro':gpu=job_gpu_peaks(result,gpu)
        try:
            import resource
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        except ImportError:peak=getattr(info,'peak_wset',info.rss)
        st.write_json(a.run_root/'costs'/f'{a.stage}_{time.time_ns()}.json',dict(stage=a.stage,seconds=time.monotonic()-start,process_peak_rss_bytes=peak,**gpu,result=result))
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
