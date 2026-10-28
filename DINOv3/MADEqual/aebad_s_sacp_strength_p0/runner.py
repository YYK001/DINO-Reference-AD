"""Default read-only preflight; CPU cached replay/evaluation, explicit GPU AUPRO."""
import argparse,json,os,sys,time,threading,subprocess,tarfile
from pathlib import Path
from . import inputs as ip
from .inputs import st
from .boundary import cpu_only,COUNTERS

STAGES=('replay','evaluate','evaluate-aupro','summarize')


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',nargs='?',default='preflight',choices=('preflight',*STAGES,'all','pack'))
    p.add_argument('--parent-run',type=Path);p.add_argument('--parent-cache',type=Path)
    p.add_argument('--run-root',type=Path,default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_s_sacp_strength_l1_v1')
    p.add_argument('--cache-root',type=Path);p.add_argument('--evaluation-scratch',type=Path);p.add_argument('--inventory-dir',type=Path)
    p.add_argument('--data',type=Path,default=ip.ROOT/'datasets/AeBAD.zip');p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8)
    p.add_argument('--device',default='cuda:0',help='used only by evaluate-aupro; one visible GPU')
    return p


def overlap(a,b):return a==b or a.is_relative_to(b) or b.is_relative_to(a)


def isolation(a):
    protected=[p for p in (a.parent_run,a.parent_cache,a.data) if p is not None]
    for target in (a.run_root,a.cache_root,a.evaluation_scratch,a.inventory_dir):
        if target is not None and any(overlap(target,p) for p in protected):raise ValueError('new output/scratch overlaps read-only parent/GT input')
    if overlap(a.cache_root,a.evaluation_scratch) or a.evaluation_scratch==a.run_root or a.run_root.is_relative_to(a.evaluation_scratch):raise ValueError('scratch cannot contain run or overlap new cache')
    if a.cache_root==a.run_root or a.run_root.is_relative_to(a.cache_root):raise ValueError('new cache cannot contain run')


def normalize(a):
    for name in ('parent_run','parent_cache','run_root','data','inventory_dir'):
        if getattr(a,name) is not None:setattr(a,name,getattr(a,name).resolve())
    a.cache_root=(a.cache_root or a.run_root/'cache').resolve();a.evaluation_scratch=(a.evaluation_scratch or a.run_root/'pixel_scratch').resolve()
    if a.device not in ('cpu','cuda') and not (a.device.startswith('cuda:') and a.device[5:].isdigit()):raise ValueError('one explicit AUPRO device required')
    isolation(a);return a


def pack(a):
    dest=a.run_root.with_name(a.run_root.name+'_review.tar.gz');allowed={'.json','.csv','.md','.log','.txt'}
    with tarfile.open(dest,'w:gz',compresslevel=1) as tar:
        for root,dirs,files in os.walk(a.run_root,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(root)/d).is_symlink() and not any((Path(root)/d).resolve()==p or (Path(root)/d).resolve().is_relative_to(p) for p in (a.cache_root,a.evaluation_scratch))]
            for name in files:
                path=Path(root)/name
                if path.suffix in allowed and not path.is_symlink():tar.add(path,arcname=a.run_root.name+'/'+path.relative_to(a.run_root).as_posix(),recursive=False)
    return dict(archive=str(dest),bytes=dest.stat().st_size,parent_cache_and_new_large_arrays_and_scratch_excluded=True)


def command_args(a):
    result=[]
    for name in ('parent_run','parent_cache','run_root','cache_root','evaluation_scratch','data','cpu_workers','device'):
        if getattr(a,name) is not None:result.extend(('--'+name.replace('_','-'),str(getattr(a,name))))
    return result


def run_stage(a):
    from . import prediction,evaluation,reporting
    if a.stage=='replay':return prediction.replay(a)
    ip.validate(a)
    if a.stage=='evaluate':
        result=evaluation.evaluate(a);reporting.summarize(a);return result
    if a.stage=='evaluate-aupro':return evaluation.evaluate(a,only_aupro=True)
    return reporting.summarize(a)


def save_cost(a,start,result):
    import psutil
    rss=psutil.Process().memory_info();torch=sys.modules.get('torch');gpu={}
    if torch is not None and torch.cuda.is_initialized():gpu=dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    if a.stage=='evaluate-aupro':
        fresh=[r for r in result if not r['runtime_reused']]
        for key in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes'):gpu[key]=max([gpu.get(key,0)]+[r.get(key,0) for r in fresh])
    try:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except ImportError:peak=getattr(rss,'peak_wset',rss.rss)
    st.write_json(a.run_root/'costs'/f'{a.stage}_{time.time_ns()}.json',dict(stage=a.stage,seconds=time.monotonic()-start,process_peak_rss_bytes=peak,
        torch_imported='torch' in sys.modules,**gpu,**COUNTERS,result=result))


def all_stages(a):
    start=time.monotonic();stage='preflight';success=False;error=None
    try:
        with cpu_only():inventory=ip.preflight(a)
        if inventory['status']=='blocked':
            show_inventory(a,inventory);raise ValueError('missing/incompatible cached parent inputs; no fallback computation')
        for stage in STAGES:
            st.write_json(a.run_root/'pipeline_status.json',dict(status='running',stage=stage,success=False))
            log=a.run_root/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True);stop=threading.Event();tick=time.monotonic()
            def heartbeat():
                while not stop.wait(30):print(f'[heartbeat] stage={stage} seconds={time.monotonic()-tick:.0f}',flush=True)
            thread=threading.Thread(target=heartbeat,daemon=True);thread.start();print(f'STAGE={stage} LOG={log}',flush=True)
            try:
                with log.open('a',encoding='utf-8') as f:
                    proc=subprocess.Popen([sys.executable,'-B','-u','-m',__spec__.name,stage,*command_args(a)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                    try:
                        for line in proc.stdout:print(line,end='',flush=True);f.write(line);f.flush()
                        rc=proc.wait()
                    except BaseException:proc.terminate();proc.wait();raise
                if rc:raise RuntimeError(stage+' failed: '+str(rc))
            finally:stop.set();thread.join()
        success=st.read_json(a.run_root/'status.json')['status']=='all_five_metrics_complete'
        if not success:raise RuntimeError('pending metrics remain')
        return dict(status='all_five_metrics_complete',success=True,seconds=time.monotonic()-start)
    except BaseException as e:error=type(e).__name__+': '+str(e);raise
    finally:
        st.write_json(a.run_root/'pipeline_status.json',dict(status='complete' if success else 'failed',stage=stage,success=success,error=error,seconds=time.monotonic()-start))
        print(json.dumps(pack(a)),flush=True)


def show_inventory(a,result):
    if a.inventory_dir:
        a.inventory_dir.mkdir(parents=True,exist_ok=True);st.write_json(a.inventory_dir/'preflight.json',result)
        st.write_csv(a.inventory_dir/'missing_inputs.csv',result['missing']);st.write_csv(a.inventory_dir/'required_cache_files.csv',result['required_cache_files'])
    compact={k:v for k,v in result.items() if k not in ('missing','required_cache_files')}
    compact.update(missing_count=len(result['missing']),missing_preview=result['missing'][:10]);print(json.dumps(compact,ensure_ascii=False,indent=2))


def main(argv=None):
    a=normalize(parser().parse_args(argv));st.configure('fast')
    if a.stage=='preflight':
        with cpu_only():result=ip.preflight(a);show_inventory(a,result)
        return 2 if result['status']=='blocked' else 0
    if a.stage=='pack':result=pack(a)
    elif a.stage=='all':result=all_stages(a)
    else:
        start=time.monotonic()
        if a.stage=='evaluate-aupro':result=run_stage(a)
        else:
            with cpu_only():result=run_stage(a)
        save_cost(a,start,result)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
