"""Read-only inventory by default; actual compute stages must be explicit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from . import inputs as ip
from .inputs import st
from .protocol import *


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight','smoke','predict','evaluate','aupro','summarize','pack','all'))
    p.add_argument('--parent-run',type=Path,default=ip.ROOT/'server_results/aebad_target_loo_two_K8_20261030_013433')
    p.add_argument('--sacp-run',type=Path,default=ip.ROOT/'server_results/aebad_s_sacp_seed42_20261028_011454')
    p.add_argument('--run-root',type=Path,default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_shared_residual_translation_v1')
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--evaluation-scratch',type=Path)
    for name in ('raw-cache','memory42-root','source-bank','support-feature-root','feature-manifest','nn-provider','checkpoint'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--data',type=Path,default=ip.ROOT/'datasets/AeBAD.zip')
    p.add_argument('--official-root',type=Path,default=ip.ROOT/'external_baselines/SuperADD')
    p.add_argument('--dinov3-root',type=Path,default=ip.ROOT/'external_baselines/dinov3')
    p.add_argument('--device',default='cpu')
    p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8)
    p.add_argument('--vector-chunk',type=int,default=1024)
    p.add_argument('--reference-cache-gib',type=float,default=8.)
    for name in ('supplement-retrieval','extract-missing','benchmark-metrics','allow-over-12h'):
        p.add_argument('--'+name,action='store_true')
    return p


def overlap(a,b):return a==b or a.is_relative_to(b) or b.is_relative_to(a)


def normalize(a):
    for k,v in vars(a).copy().items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    a.cache_root=a.cache_root or a.run_root.with_name(a.run_root.name+'_cache')
    a.evaluation_scratch=a.evaluation_scratch or a.run_root.with_name(a.run_root.name+'_pixels')
    protected=[v for k,v in vars(a).items() if isinstance(v,Path) and k not in ('run_root','cache_root','evaluation_scratch')]
    # Inherited defaults are assets too, even when --source-bank/raw-cache was
    # omitted. Reject overlap BEFORE making output directories or cost records.
    parent_inputs=a.parent_run/'inputs.json'
    if a.stage!='pack' and parent_inputs.exists():
        inherited=st.sealed_json(parent_inputs)
        for value in (inherited.get('source',{}).get('path'),inherited.get('parent_cache'),inherited.get('raw_cache')):
            if value:protected.append(Path(value).resolve())
    writable=(a.run_root,a.cache_root,a.evaluation_scratch)
    if any(overlap(x,y) for i,x in enumerate(writable) for y in writable[i+1:]) or any(overlap(x,y) for x in writable for y in protected):raise ValueError('output/cache/scratch must be disjoint from each other and parent assets')
    if a.vector_chunk<1 or a.reference_cache_gib<=0:raise ValueError('positive blocks/cache size required')
    # A single logical device only; never enumerate/enable a second GPU.
    if a.device!='cpu' and (not a.device.startswith('cuda:') or not a.device[5:].isdigit()):raise ValueError('one cpu or cuda:index device required')
    a.memory43_root=None;a._started=time.perf_counter();return a


def pack(a):
    dest=a.run_root.with_name(a.run_root.name+'_review.tar.gz')
    with tarfile.open(dest,'w:gz',compresslevel=1) as archive:
        for root,dirs,files in os.walk(a.run_root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in ('scratch','pixel_scratch','__pycache__') and not (Path(root)/d).is_symlink()]
            for name in files:
                path=Path(root)/name
                if path.is_symlink() or path.suffix not in ('.json','.csv','.md','.txt','.log','.png'):continue
                archive.add(path,arcname=a.run_root.name+'/'+path.relative_to(a.run_root).as_posix(),recursive=False)
    return dict(archive=str(dest),bytes=dest.stat().st_size,large_arrays_and_scratch_excluded=True)


def execute(a):
    if a.stage=='pack':return pack(a)
    if a.stage=='preflight':return ip.preflight(a)
    ip.freeze(a)
    if a.stage=='smoke':
        from .smoke import run
        return run(a)
    if a.stage=='predict':
        from .prediction import predict
        from .smoke import launch_check
        launch_check(a);return predict(a)
    if a.stage in ('evaluate','aupro'):
        from . import evaluation
        return getattr(evaluation,a.stage)(a)
    if a.stage=='summarize':
        from .reporting import summarize
        return summarize(a)
    raise ValueError('stage not dispatched')


def cost(a,tick,result):
    import psutil
    from importlib.metadata import version,PackageNotFoundError
    process=psutil.Process();info=process.memory_info()
    peak=getattr(info,'peak_wset',info.rss)
    if os.name!='nt':
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    record=dict(stage=a.stage,seconds=time.perf_counter()-tick,process_peak_rss_bytes=peak,result=result,
                feature_forwards=getattr(a,'feature_forwards',0),torch_imported='torch' in sys.modules)
    original_storage=sys.modules.get('DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.storage')
    record['new_array_payload_bytes']=getattr(original_storage,'NEW_ARRAY_BYTES',0)
    record['disk_counter_note']='new NPY writes in this subprocess; not total filesystem usage or IO bytes'
    record['versions_without_import']={}
    for name in ('numpy','scipy','Pillow','torch','torchvision','adeval'):
        try:record['versions_without_import'][name]=version(name)
        except PackageNotFoundError:record['versions_without_import'][name]=None
    if 'torch' in sys.modules:
        t=sys.modules['torch']
        if t.cuda.is_initialized():
            host=sys.modules.get('DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime')
            prior=getattr(host,'PRIOR_GPU_PEAK',{})
            record.update(peak_cuda_allocated_bytes=max(t.cuda.max_memory_allocated(),prior.get('allocated',0)),
                          peak_cuda_reserved_bytes=max(t.cuda.max_memory_reserved(),prior.get('reserved',0)),
                          device=a.device,compiled_cuda=t.version.cuda,GPU=t.cuda.get_device_name(t.cuda.current_device()))
    st.write_json(a.run_root/'costs'/f'{a.stage}_{time.time_ns()}.json',record)


def main(argv=None):
    a=normalize(parser().parse_args(argv));a.run_root.mkdir(parents=True,exist_ok=True)
    if a.stage=='all':
        from .smoke import launch_check
        ip.freeze(a);launch_check(a)
        common=list(sys.argv[2:] if argv is None else argv[1:])
        for stage in ('predict','evaluate','aupro','summarize','pack'):
            log=a.run_root/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            print(f'STAGE={stage} LOG={log}',flush=True)
            with log.open('a',encoding='utf-8') as f:
                process=subprocess.Popen([sys.executable,'-B','-u','-m',__package__+'.runner',stage,*common],stdout=f,stderr=subprocess.STDOUT)
                while True:
                    try:rc=process.wait(timeout=30);break
                    except subprocess.TimeoutExpired:print(f'[heartbeat] residual stage={stage}',flush=True)
            print(f'STAGE={stage} EXIT_CODE={rc}',flush=True)
            if rc:return rc
        return 0
    tick=time.perf_counter()
    try:result=execute(a)
    except Exception as e:
        cost(a,tick,dict(status='stage_failed',error=type(e).__name__+': '+str(e)));raise
    cost(a,tick,result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 2 if a.stage=='preflight' and result['status']!='inputs_available' else 0


if __name__=='__main__':raise SystemExit(main())
