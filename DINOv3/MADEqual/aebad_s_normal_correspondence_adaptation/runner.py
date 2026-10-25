"""Default read-only inventory. Compute/all are explicit, single-device stages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
from . import inputs as ip,storage as st
from .protocol import CONFIG

STAGES=('prepare','match','train-source','train-target','transform-memory','calibrate','predict','evaluate','evaluate-aupro','summarize')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight',*STAGES,'pack','all'))
    p.add_argument('--run-root',type=Path,default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_normal_correspondence_v1')
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--data',type=Path,default=ip.ROOT/'datasets/AeBAD.zip')
    p.add_argument('--checkpoint',type=Path,default=ip.ROOT/'external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
    p.add_argument('--official-root',type=Path,default=ip.ROOT/'external_baselines/SuperADD')
    p.add_argument('--dinov3-root',type=Path,default=ip.ROOT/'external_baselines/dinov3')
    p.add_argument('--recent-run',type=Path,action='append',default=[],help='repeat for the TWO migration-era runs only; no automatic scanning')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8)
    p.add_argument('--query-mode',choices=('stream','disk'),default='stream')
    p.add_argument('--evaluation-scratch',type=Path)
    p.add_argument('--no-write',action='store_true',help='preflight is always read-only')
    return p


def normalize(a):
    a.run_root=a.run_root.resolve();a.cache_root=(a.cache_root or a.run_root/'cache').resolve()
    a.evaluation_scratch=(a.evaluation_scratch or a.run_root/'pixel_scratch').resolve()
    for k in ('data','checkpoint','official_root','dinov3_root'):setattr(a,k,getattr(a,k).resolve())
    a.recent_run=[p.resolve() for p in a.recent_run]
    check_isolation(a)
    return a


def overlaps(left,right):
    left,right=Path(left).resolve(),Path(right).resolve()
    return left==right or left.is_relative_to(right) or right.is_relative_to(left)


def check_isolation(a):
    """Read metadata only; protect inputs and named parent caches, including symlinks."""
    protected=[Path(p).resolve() for k in ('data','checkpoint','official_root','dinov3_root')
               if (p:=getattr(a,k,None)) is not None]
    pending=list(getattr(a,'recent_run',[]));seen=set()
    while pending:
        root=Path(pending.pop()).resolve()
        if root in seen:continue
        seen.add(root);protected.append(root)
        setup=root/'setup.json'
        if setup.is_file():
            record=st.read_json(setup);configuration=record.get('configuration',{})
            cache=record.get('cache') or configuration.get('cache_root') or configuration.get('cache')
            if cache:protected.append(Path(cache).resolve())
            # Only the explicit recorded parent chain, never a directory scan.
            if configuration.get('parent'):pending.append(Path(configuration['parent']))
    run,cache,scratch=(Path(getattr(a,k)).resolve() for k in ('run_root','cache_root','evaluation_scratch'))
    for destination in (run,cache,scratch):
        if any(overlaps(destination,p) for p in protected):
            raise ValueError('output/cache/evaluation scratch overlaps read-only input or recent asset: '+str(destination))
    if overlaps(scratch,cache) or scratch==run or run.is_relative_to(scratch):
        raise ValueError('evaluation scratch must be separate from cache and cannot equal/contain run-root')


def validate_runtime_inputs(a):
    frozen=st.sealed_json(a.run_root/'inputs.json')
    if frozen['configuration']!=CONFIG or frozen['query_mode']!=a.query_mode:raise ValueError('frozen configuration changed')
    if frozen['cache_root']!=str(a.cache_root):raise ValueError('cache location changed; use the original run/cache pair')
    for relative,sha in frozen['code'].items():
        if st.sha(ip.ROOT/relative)!=sha:raise ValueError('implementation changed after prepare: '+relative)
    if frozen['host_config']!=st.read_json(a.official_root/'config.json'):raise ValueError('host configuration changed')
    from importlib.metadata import version
    for name,expected in frozen['software'].items():
        if version(name)!=expected:raise ValueError('runtime software changed: '+name)
    for key,expected in frozen['external_code'].items():
        name,relative=key.split('/',1);root=a.dinov3_root if name=='dinov3' else a.official_root
        if st.sha(root/relative)!=expected:raise ValueError('external extraction source changed: '+key)
    prepared=st.sealed_json(a.run_root/'prepared.json')
    if prepared['identity']!=frozen['identity'] or st.read_json(a.cache_root/'source_original/manifest.json')['seal']!=prepared['source_seal']:
        raise ValueError('prepared Source bank identity changed')
    c=ip.catalog(a)
    for row in c['source']+sum(c['supports'].values(),[]):
        if st.read_json(ip.feature_path(a,row)/'manifest.json')['seal']!=prepared['features'][row['relative_path']]:
            raise ValueError('prepared normal features changed')


def run_stage(a):
    if a.stage=='prepare':
        from .runtime import prepare
        return prepare(a)
    validate_runtime_inputs(a)
    if a.stage=='match':
        from .matching import match
        return match(a)
    if a.stage in ('train-source','train-target'):
        from .training import train
        return train(a,source=a.stage=='train-source')
    if a.stage in ('transform-memory','calibrate'):
        from .calibration import transform_memory,calibrate
        return (transform_memory if a.stage=='transform-memory' else calibrate)(a)
    if a.stage=='predict':
        from .prediction import predict
        return predict(a)
    if a.stage in ('evaluate','evaluate-aupro'):
        from .evaluation import evaluate
        return evaluate(a,only_aupro=a.stage=='evaluate-aupro')
    from .reporting import summarize
    return summarize(a)


def pack(a):
    destination=a.run_root.with_name(a.run_root.name+'_review.tar.gz')
    allowed={'.json','.csv','.tsv','.md','.log','.txt','.png','.svg','.exitcode'}
    with tarfile.open(destination,'w:gz',compresslevel=1) as archive:
        for root,dirs,files in os.walk(a.run_root,followlinks=False):
            dirs[:]=[d for d in dirs if 'scratch' not in d.lower() and d!='cache' and not (Path(root)/d).is_symlink()]
            for name in files:
                p=Path(root)/name
                if p.suffix in allowed and not p.is_symlink():archive.add(p,arcname=a.run_root.name+'/'+p.relative_to(a.run_root).as_posix(),recursive=False)
    return dict(archive=str(destination),bytes=destination.stat().st_size,large_arrays_and_checkpoints_excluded=True)


def common(a):
    values=[]
    for k in ('run_root','cache_root','data','checkpoint','official_root','dinov3_root','device','cpu_workers','query_mode','evaluation_scratch'):
        values.extend(('--'+k.replace('_','-'),str(getattr(a,k))))
    for p in a.recent_run:values.extend(('--recent-run',str(p)))
    return values


def all_stages(a):
    a.run_root.mkdir(parents=True,exist_ok=True);started=time.monotonic();success=False
    try:
        inventory=ip.preflight(a)
        print(json.dumps(inventory,ensure_ascii=False),flush=True)
        if inventory['status']=='blocked':raise ValueError('missing input/dependency; see preflight inventory')
        for stage in STAGES:
            log=a.run_root/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            print(f'STAGE={stage} START LOG={log}',flush=True)
            state=threading.Event();tick=time.monotonic()
            def heartbeat():
                while not state.wait(30):print(f'[heartbeat] stage={stage} seconds={time.monotonic()-tick:.0f}',flush=True)
            thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
            try:
                with log.open('a',encoding='utf-8') as f:
                    process=subprocess.Popen([sys.executable,'-B','-u','-m',__spec__.name,stage,*common(a)],
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                    try:
                        for line in process.stdout:print(line,end='',flush=True);f.write(line);f.flush()
                        rc=process.wait()
                    except BaseException:
                        process.terminate();process.wait();raise
                print(f'STAGE={stage} EXIT_CODE={rc}',flush=True)
                if rc:raise RuntimeError(stage+' failed; retained '+str(log))
            finally:state.set();thread.join()
        result=st.read_json(a.run_root/'status.json');success=result['status']=='all_five_metrics_complete'
        if not success:raise RuntimeError('pipeline stages returned but pending metrics remain')
        return dict(status='all_five_metrics_complete',success=True,seconds=time.monotonic()-started)
    finally:
        st.write_json(a.run_root/'pipeline_status.json',dict(success=success,seconds=time.monotonic()-started))
        print(json.dumps(pack(a),ensure_ascii=False),flush=True)


def main(argv=None):
    a=normalize(parser().parse_args(argv))
    if a.stage=='preflight':
        result=ip.preflight(a);print(json.dumps(result,ensure_ascii=False,indent=2));return 2 if result['status']=='blocked' else 0
    if a.no_write:raise ValueError('--no-write is only available for preflight')
    if a.stage=='all':result=all_stages(a)
    elif a.stage=='pack':result=pack(a)
    else:
        started=time.monotonic()
        try:result=run_stage(a)
        except Exception as e:
            st.write_json(a.run_root/'errors'/(a.stage+'.json'),dict(error=type(e).__name__+': '+str(e),stage=a.stage));raise
        import psutil
        cuda=sys.modules.get('torch');gpu={}
        if cuda is not None and cuda.cuda.is_initialized():
            from .runtime import PRIOR_GPU_PEAK
            gpu=dict(peak_cuda_allocated_bytes=max(PRIOR_GPU_PEAK['allocated'],cuda.cuda.max_memory_allocated()),
                peak_cuda_reserved_bytes=max(PRIOR_GPU_PEAK['reserved'],cuda.cuda.max_memory_reserved()))
        info=psutil.Process().memory_info()
        try:
            import resource
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if sys.platform!='darwin' else 1)
        except ImportError:peak=getattr(info,'peak_wset',info.rss)
        value=dict(stage=a.stage,seconds=time.monotonic()-started,process_peak_rss_bytes=peak,
            new_array_payload_bytes=st.NEW_ARRAY_BYTES,**gpu,result=result)
        stamp=time.time_ns();st.write_json(a.run_root/'costs'/f'{a.stage}_{stamp}.json',value)
        result=value
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
