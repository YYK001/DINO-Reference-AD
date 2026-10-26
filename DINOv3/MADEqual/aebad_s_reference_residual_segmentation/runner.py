"""Explicit staged local/workstation entry; default read-only inventory."""
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

STAGES=('prepare','synthesize','prepare-training','train-source','train-target','calibrate','predict','evaluate','evaluate-aupro','summarize')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight',*STAGES,'smoke','pack','all'))
    p.add_argument('--run-root',type=Path,default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_reference_residual_v1')
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--data',type=Path,default=ip.ROOT/'datasets/AeBAD.zip')
    p.add_argument('--checkpoint',type=Path,default=ip.ROOT/'external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
    p.add_argument('--official-root',type=Path,default=ip.ROOT/'external_baselines/SuperADD')
    p.add_argument('--dinov3-root',type=Path,default=ip.ROOT/'external_baselines/dinov3')
    p.add_argument('--recent-run',type=Path,action='append',default=[],help='explicit migration-era original assets only; repeatable')
    p.add_argument('--asset-manifest',type=Path,action='append',default=[],help='explicit raw_assets.json, not transformed memory')
    p.add_argument('--device',default='cuda:0',help='one device; physical GPU selected by CUDA_VISIBLE_DEVICES')
    p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8)
    p.add_argument('--query-mode',choices=('stream','disk'),default='stream')
    p.add_argument('--evaluation-scratch',type=Path)
    p.add_argument('--sampling-scratch',type=Path,help='SSD private hot-layer staging for the two training-fold banks')
    p.add_argument('--no-write',action='store_true',help='preflight always read-only')
    p.add_argument('--reference-cache-gib',type=float,default=4.,help='bounded Exact reference tensor LRU; not a capacity admission check')
    p.add_argument('--profile',choices=('compact_v1','full_v1'),default='compact_v1')
    p.add_argument('--audit-mode',choices=('fast','full'),default='fast',help='fast trusts local payloads; full explicitly enables SHA and direct NN audits')
    p.add_argument('--smoke-mode',choices=('constructed','assets','prepared'),default='constructed')
    p.add_argument('--smoke-updates',type=int,default=30)
    return p


def overlaps(left,right):
    left,right=Path(left).resolve(),Path(right).resolve()
    return left==right or left.is_relative_to(right) or right.is_relative_to(left)


def check_isolation(a):
    protected=[Path(getattr(a,k)).resolve() for k in ('data','checkpoint','official_root','dinov3_root') if getattr(a,k,None) is not None]
    pending=list(getattr(a,'recent_run',[]));seen=set()
    providers=Path(a.run_root)/'eligible_raw_providers.json'
    if providers.exists():
        for r in st.sealed_json(providers)['providers']:
            pending.append(Path(r['run']))
            if r.get('cache'):protected.append(Path(r['cache']).resolve())
    while pending:
        root=Path(pending.pop()).resolve()
        if root in seen:continue
        seen.add(root);protected.append(root)
        for name in ('inputs.json','setup.json'):
            p=root/name
            if not p.is_file():continue
            r=st.read_json(p);config=r.get('configuration',{})
            cache=r.get('cache_root') or r.get('cache') or config.get('cache_root') or config.get('cache')
            if cache:protected.append(Path(cache).resolve())
            if config.get('parent'):pending.append(Path(config['parent']))
    for p in getattr(a,'asset_manifest',[]):
        protected.append(Path(p).resolve())
        if Path(p).exists():
            r=st.read_json(p)
            if r.get('source_bank',{}).get('path'):protected.append(Path(r['source_bank']['path']).resolve())
            for item in r.get('normal_features',{}).values():protected.append(Path(item['path']).resolve())
            for item in r.get('queries',{}).values():protected.append(Path(item['path']).resolve())
    run,cache,scratch=(Path(getattr(a,k)).resolve() for k in ('run_root','cache_root','evaluation_scratch'))
    for dest in (run,cache,scratch):
        if any(overlaps(dest,p) for p in protected):raise ValueError('output/cache/scratch overlaps read-only input: '+str(dest))
    if overlaps(scratch,cache) or scratch==run or run.is_relative_to(scratch):raise ValueError('scratch must be separate from cache and cannot equal/contain run-root')
    if cache==run or run.is_relative_to(cache):raise ValueError('cache cannot equal/contain the report/model run-root')
    sampling=Path(a.sampling_scratch).resolve()
    if any(overlaps(sampling,p) for p in protected) or overlaps(sampling,cache) or overlaps(sampling,scratch) or sampling==run or run.is_relative_to(sampling):
        raise ValueError('sampling scratch must be isolated from parents/cache/evaluation scratch')


def normalize(a):
    a.run_root=a.run_root.resolve();a.cache_root=(a.cache_root or a.run_root/'cache').resolve()
    a.evaluation_scratch=(a.evaluation_scratch or a.run_root/'pixel_scratch').resolve()
    a.sampling_scratch=(a.sampling_scratch or a.run_root/'sampling_scratch').resolve()
    for k in ('data','checkpoint','official_root','dinov3_root'):setattr(a,k,getattr(a,k).resolve())
    for k in ('recent_run','asset_manifest'):setattr(a,k,[p.resolve() for p in getattr(a,k)])
    if ',' in a.device or a.device not in ('cpu','cuda') and not (a.device.startswith('cuda:') and a.device[5:].isdigit()):raise ValueError('one CPU/CUDA device required')
    if not 0<=a.reference_cache_gib<=64 or not 3<=a.smoke_updates<=1000:raise ValueError('invalid execution cache size or smoke updates')
    check_isolation(a);return a


def validate_implementation(frozen):
    """Small source files only; allow the registered execution-only revision."""
    current={p:st.sha(ip.ROOT/p) for p in frozen['code']}
    mismatch={p:sha for p,sha in frozen['code'].items() if current[p]!=sha}
    if not mismatch:return
    prefix='DINOv3/MADEqual/aebad_s_reference_residual_segmentation/'
    saved={p:sha for p,sha in frozen['code'].items() if p.startswith(prefix)}
    record=st.read_json(Path(__file__).with_name('execution_compatibility.json'))
    pinned=all((current[p] if p in current else st.sha(ip.ROOT/p))==sha for p,sha in record['execution_revision_code'].items())
    if saved==record['code'] and pinned and all(p.startswith(prefix) for p in mismatch):return
    legacy=st.read_json(Path(__file__).with_name('legacy_full_v1_compatibility.json'))['code']
    if (frozen.get('config')==CONFIG and pinned and all(p.startswith(prefix) and legacy.get(Path(p).name)==sha for p,sha in mismatch.items())):return
    raise ValueError('implementation changed beyond registered execution revision: '+next(iter(mismatch)))


def validate_runtime_inputs(a):
    frozen=st.sealed_json(a.run_root/'inputs.json')
    cfg=ip.config(a);ip.plan(a)
    if (frozen['config']!=(CONFIG if ip.legacy(a) else cfg) or frozen['cache']!=str(a.cache_root)
            or frozen['query_mode']!=a.query_mode):raise ValueError('frozen runtime configuration differs')
    if not ip.legacy(a):
        if st.sealed_json(a.run_root/'resolved_config.json')['seal']!=frozen['resolved_config_seal'] or st.sealed_json(a.run_root/'job_plan.json')['seal']!=frozen['job_plan_seal']:raise ValueError('resolved config/job plan seal changed')
    if st.digest(ip.catalog(a))!=frozen['catalog']:raise ValueError('frozen image/support catalog changed')
    validate_implementation(frozen)
    e=frozen['extraction']
    offline=a.stage in ('evaluate','evaluate-aupro','summarize')
    if offline:
        from importlib.metadata import version
        for name in ('numpy','scipy','Pillow'):
            if version(name)!=e['software'][name]:raise ValueError('CPU map software changed: '+name)
        return  # Frozen predictions/maps need neither bank nor extractor entities.
    if e['host_config']!=st.read_json(a.official_root/'config.json'):raise ValueError('frozen host configuration changed')
    from importlib.metadata import version
    for name,expected in e['software'].items():
        if version(name)!=expected:raise ValueError('runtime software changed: '+name)
    for key,sha in e['external_code'].items():
        name,relative=key.split('/',1);root=a.dinov3_root if name=='dinov3' else a.official_root
        if st.sha(root/relative)!=sha:raise ValueError('extraction code changed: '+key)
    prepared=st.sealed_json(a.run_root/'prepared.json')
    if prepared['identity']!=frozen['identity'] or st.sealed_json(a.cache_root/'source_original/manifest.json')['seal']!=prepared['source_seal']:raise ValueError('formal Source bank identity changed')
    if st.full_audit():
        for item in prepared['normal_features'].values():
            if st.sealed_json(Path(item['path'])/'manifest.json')['seal']!=item['seal']:raise ValueError('full normal grid identity changed')


def run_stage(a):
    if a.stage=='smoke':
        from .smoke import smoke
        return smoke(a)
    if a.stage=='prepare':
        if (a.run_root/'prepared.json').exists():
            validate_runtime_inputs(a)
            return dict(status='original_assets_reused',payload_SHA_requested=st.full_audit(),payload_headers_checked_when_consumed=True)
        from .runtime import prepare
        return prepare(a)
    validate_runtime_inputs(a)
    if a.stage=='synthesize':
        p=a.run_root/'synthetic_manifest.json'
        if p.exists():
            r=st.sealed_json(p)
            if r['identity']!=ip.identity(a):raise ValueError('fixed synthetic pool identity differs')
            return dict(status='fixed_synthetic_pool_reused',slots=len(r['items']),invalid=r['invalid'])
        from .runtime import synthesize
        return synthesize(a)
    if a.stage=='prepare-training':
        p=a.run_root/'training_manifest.json'
        if p.exists():
            r=st.sealed_json(p)
            if r['identity']!=ip.identity(a) or r['status']!='all_fixed_training_queries_prepared':raise ValueError('training cache identity differs')
            return dict(status='training_queries_reused',queries=len(r['records']))
        from .training_data import prepare_training
        return prepare_training(a)
    if a.stage in ('train-source','train-target'):
        from .training import train
        return train(a,source=a.stage=='train-source')
    if a.stage=='calibrate':
        from .calibration import calibrate
        return calibrate(a)
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
            dirs[:]=[d for d in dirs if 'scratch' not in d.lower() and d!='cache' and not (Path(root)/d).is_symlink()
                and not any((Path(root)/d).resolve()==p or (Path(root)/d).resolve().is_relative_to(p) for p in (a.cache_root,a.evaluation_scratch,a.sampling_scratch))]
            for name in files:
                p=Path(root)/name
                if p.suffix in allowed and not p.is_symlink():archive.add(p,arcname=a.run_root.name+'/'+p.relative_to(a.run_root).as_posix(),recursive=False)
    return dict(archive=str(destination),bytes=destination.stat().st_size,checkpoints_and_large_arrays_excluded=True)


def common(a):
    values=[]
    for k in ('run_root','cache_root','data','checkpoint','official_root','dinov3_root','device','cpu_workers','query_mode','evaluation_scratch','sampling_scratch','reference_cache_gib','profile','audit_mode'):
        values.extend(('--'+k.replace('_','-'),str(getattr(a,k))))
    for k in ('recent_run','asset_manifest'):
        for p in getattr(a,k):values.extend(('--'+k.replace('_','-'),str(p)))
    return values


def all_stages(a):
    a.run_root.mkdir(parents=True,exist_ok=True);started=time.monotonic();success=False;current='preflight';error=None
    try:
        inventory=ip.preflight(a);print(json.dumps(inventory,ensure_ascii=False),flush=True)
        if inventory['status']=='blocked':raise ValueError('missing input/dependency; see preflight inventory')
        for current in STAGES:
            st.write_json(a.run_root/'pipeline_status.json',dict(status='running',stage=current,success=False))
            log=a.run_root/'logs'/(current+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            print(f'STAGE={current} START LOG={log}',flush=True);done=threading.Event();tick=time.monotonic()
            def heartbeat():
                while not done.wait(30):print(f'[heartbeat] stage={current} seconds={time.monotonic()-tick:.0f}',flush=True)
            thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
            try:
                with log.open('a',encoding='utf-8') as f:
                    process=subprocess.Popen([sys.executable,'-B','-u','-m',__spec__.name,current,*common(a)],stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                    try:
                        for line in process.stdout:print(line,end='',flush=True);f.write(line);f.flush()
                        rc=process.wait()
                    except BaseException:process.terminate();process.wait();raise
                print(f'STAGE={current} EXIT_CODE={rc}',flush=True)
                if rc:raise RuntimeError(current+' failed; retained '+str(log))
            finally:done.set();thread.join()
        success=st.read_json(a.run_root/'status.json')['status']=='all_five_metrics_complete'
        if not success:raise RuntimeError('pending metrics remain')
        return dict(status='all_five_metrics_complete',success=True,seconds=time.monotonic()-started)
    except BaseException as e:error=type(e).__name__+': '+str(e);raise
    finally:
        st.write_json(a.run_root/'pipeline_status.json',dict(status='complete' if success else 'failed',stage=current,error=error,success=success,seconds=time.monotonic()-started))
        print(json.dumps(pack(a),ensure_ascii=False),flush=True)


def main(argv=None):
    a=normalize(parser().parse_args(argv))
    st.configure(a.audit_mode)
    if a.stage=='preflight':
        r=ip.preflight(a);print(json.dumps(r,ensure_ascii=False,indent=2));return 2 if r['status']=='blocked' else 0
    if a.no_write:raise ValueError('--no-write is preflight only')
    st.write_json(a.run_root/'execution_sessions'/f'{time.time_ns()}_{a.stage}.json',dict(
        stage=a.stage,audit_mode=a.audit_mode,payload_read_SHA=a.audit_mode=='full',
        direct_CPU_NN_recomputation=a.audit_mode=='full',frozen_inputs_rewritten=False))
    if a.stage=='all':result=all_stages(a)
    elif a.stage=='pack':result=pack(a)
    else:
        tick=time.monotonic()
        try:result=run_stage(a)
        except Exception as e:
            st.write_json(a.run_root/'errors'/(a.stage+'.json'),dict(stage=a.stage,error=type(e).__name__+': '+str(e)));raise
        import psutil
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import storage as old_storage,runtime as old_runtime
        info=psutil.Process().memory_info();torch=sys.modules.get('torch');gpu={}
        if torch is not None and torch.cuda.is_initialized():gpu=dict(peak_cuda_allocated_bytes=max(old_runtime.PRIOR_GPU_PEAK['allocated'],torch.cuda.max_memory_allocated()),
            peak_cuda_reserved_bytes=max(old_runtime.PRIOR_GPU_PEAK['reserved'],torch.cuda.max_memory_reserved()))
        try:
            import resource
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if sys.platform!='darwin' else 1)
        except ImportError:peak=getattr(info,'peak_wset',info.rss)
        result=dict(stage=a.stage,seconds=time.monotonic()-tick,process_peak_rss_bytes=peak,new_array_payload_bytes=old_storage.NEW_ARRAY_BYTES,**gpu,result=result)
        st.write_json(a.run_root/'costs'/f'{a.stage}_{time.time_ns()}.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
