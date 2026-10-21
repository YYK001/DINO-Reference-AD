"""Whole-run Kaggle orchestration; invoked only by the supplied notebook cell.

No notebook kernel model/GPU imports. Every stage has its own subprocess, and
the final raw archive retains all eight per-support distances for future CPU use.
"""
from __future__ import annotations
from datetime import datetime,timezone,timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import uuid

OLD='DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.runner'
NEW='DINOv3.MADEqual.aebad_s_support_distance_crossdomain.runner'


def say(text):
    print(datetime.now(timezone(timedelta(hours=8))).strftime('[%H:%M:%S] ')+str(text),flush=True)


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def capture_progress(run):
    roots=[run/'cache',*[run/'capture_shards'/d for d in ('illumination','view')]]
    return len({(p.parent.name,p.name) for root in roots for p in (root/'raw_cache').glob('*/*.npz')})


def command(label,cmd,project,run,env,*,cancel_event=None,allowed_returncodes=()):
    """Flush subprocess output and independent 30-second progress heartbeats."""
    say('STAGE='+label+' START');started=time.monotonic();last=started
    with (run/(label+'.log')).open('x',encoding='utf-8') as log:
        p=subprocess.Popen(cmd,cwd=project,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,encoding='utf-8',errors='replace',bufsize=1)
        lines=queue.Queue()
        def read():
            try:
                for line in p.stdout:lines.put(line)
            finally:lines.put(None)
        thread=threading.Thread(target=read,daemon=True);thread.start();done=False
        try:
            while not done:
                if cancel_event is not None and cancel_event.is_set():raise RuntimeError(label+' cancelled after peer failure')
                try:
                    line=lines.get(timeout=1)
                    if line is None:done=True
                    else:log.write(line);log.flush();say(line.rstrip())
                except queue.Empty:pass
                now=time.monotonic()
                if now-last>=30:
                    n=capture_progress(run)
                    say(f'[heartbeat] stage={label} elapsed={now-started:.0f}s capture_files={n}/629')
                    last=now
            rc=p.wait();(run/(label+'.exitcode')).write_text(str(rc)+'\n')
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            thread.join();p.stdout.close()
            (run/(label+'.exitcode')).write_text(str(p.returncode)+'\n')
    say(f'STAGE={label} EXIT_CODE={rc} SECONDS={time.monotonic()-started:.1f}')
    if rc and rc not in allowed_returncodes:raise RuntimeError(f'{label} failed: exit_code={rc}; see {run/(label+".log")}')
    return rc


def asset(root,preferred,name,directory=False):
    first=root/preferred
    if (first.is_dir() if directory else first.is_file()):return first
    candidates=[p for p in root.rglob(name) if (p.is_dir() if directory else p.is_file())]
    if len(candidates)!=1:raise FileNotFoundError(f'{name}: expected one Input candidate, found {list(map(str,candidates))}')
    return candidates[0]


def inputs(root):
    frozen=asset(root,'datasets/yky2026/frozen/formal','formal',True)
    return dict(formal=frozen,state=asset(root,'datasets/yky2026/frozen/image_calibration_state.json','image_calibration_state.json'),
        memory=asset(root,'datasets/yky2026/temp1-pt/official_memory.pt','official_memory.pt'),
        checkpoint=asset(root,'datasets/yky2026/checkpoint-dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
                         'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'),
        data=asset(root,'datasets/yky2026/aebad888/AeBAD/AeBAD_S','AeBAD_S',True))


def pack(run,state):
    """One small result archive + one portable raw/reference archive; no Q/banks."""
    files=[p for p in run.rglob('*') if p.is_file()]
    def portable(p):
        parts=p.relative_to(run).parts
        if p.suffix=='.tmp' or '_scratch' in '/'.join(parts):return False
        if parts[0]=='cache':return len(parts)>1 and (parts[1]=='raw_cache' or len(parts)==2)
        if parts[0]=='prediction':return p.suffix!='.npz'
        if parts[0]=='capture_shards':
            if p.suffix in ('.json','.csv'):return True
            # On failure before merge, retain every completed shard raw too.
            return len(parts)>3 and parts[2]=='raw_cache' and p.suffix=='.npz' and not (run/'cache'/Path(*parts[2:])).is_file()
        return parts[0] in ('cpu','aupro','diagnostic','frozen_inputs','background_reference') or len(parts)==1
    raw=[p for p in files if portable(p)]
    state['raw_archive']=None;state['raw_archive_error']=None
    if any('raw_cache' in p.relative_to(run).parts for p in raw):
        archive=run.parent/(run.name+'_RAW_AND_REFERENCES.tar')
        try:
            say('PACKING_RAW='+str(archive))
            with tarfile.open(archive,'x') as t:
                for p in raw:t.add(p,arcname=run.name+'/'+p.relative_to(run).as_posix(),recursive=False)
            state['raw_archive']=str(archive);state['raw_archive_bytes']=archive.stat().st_size
        except Exception as exc:state['raw_archive_error']=f'{type(exc).__name__}: {exc}'
    state['experiment_success']=bool(state['computation_success'] and state['raw_archive'])
    write(run/'pipeline_status.json',state)
    summary=run.parent/(run.name+('_SUCCESS_SUMMARY.tar.gz' if state['experiment_success'] else '_FAILED_SUMMARY.tar.gz'))
    with tarfile.open(summary,'x:gz',compresslevel=1) as t:
        for p in run.rglob('*'):
            if not p.is_file() or p.suffix=='.npz' or '_scratch' in p.as_posix():continue
            if p.relative_to(run).parts[:2] in (('cache','raw_cache'),('cache','support_features')):continue
            t.add(p,arcname=run.name+'/'+p.relative_to(run).as_posix(),recursive=False)
        # Small calibration NPZ is provenance, not a model/distance cache.
        for p in (run/'frozen_inputs').rglob('*.npz'):
            t.add(p,arcname=run.name+'/'+p.relative_to(run).as_posix(),recursive=False)
    say('SUMMARY_ARCHIVE='+str(summary));say('RAW_ARCHIVE='+str(state['raw_archive']))
    say('EXPERIMENT_SUCCESS='+str(state['experiment_success']))
    say('RUN_DIR='+str(run))
    return summary


def main(project,overlay_bytes=None,overlay_sha256=None,run_root=None,input_root=Path('/kaggle/input'),*,capture_gpus=2):
    project=Path(project).resolve()
    if not project.is_relative_to(Path('/kaggle/working')):raise ValueError('this orchestrator is for explicit Kaggle execution only')
    if capture_gpus not in (1,2):raise ValueError('capture_gpus must explicitly be 1 or 2')
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    run=Path(run_root or '/kaggle/working')/('aebad_full_crossdomain_'+stamp+'_'+uuid.uuid4().hex[:6]);run.mkdir(parents=True)
    if overlay_bytes is not None:
        import hashlib
        if hashlib.sha256(overlay_bytes).hexdigest()!=overlay_sha256:raise ValueError('wrong code overlay digest')
        (run/'code_overlay.zip').write_bytes(overlay_bytes)
    state=dict(computation_success=False,experiment_success=False,failed_stage=None,stages=[],overlay_sha256=overlay_sha256)
    started=time.monotonic();env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    say('RUN_DIR='+str(run));label='initialization'
    try:
        commit=subprocess.check_output(['git','-C',str(project),'rev-parse','HEAD'],text=True).strip()
        state['git_commit']=commit
        from . import inputs as ip,reporting
        # Validate this small reference before any expensive model work.
        background=Path(__file__).with_name('background_discovery_catalog.json')
        reporting.verified_catalog(background)
        shutil.copyfile(background,run/'background_discovery_catalog.json')
        background=run/'background_discovery_catalog.json'
        write(run/'code_manifest.json',dict(git_commit=commit,overlay_sha256=overlay_sha256,
            source='GitHub clone' if overlay_bytes is None else 'GitHub clone plus explicit notebook overlay',
            files={p.relative_to(project).as_posix():ip.sha(p) for d in ('aebad_s_support_distance_crossdomain','aebad_s_support_distance_diagnostic')
                   for p in (project/'DINOv3/MADEqual'/d).glob('*.py')}))
        # Keep preinstalled Torch/NumPy. Install missing ancillary dependencies
        # without resolving/upgrading core packages; actual imports are checked.
        mapping={'timm':'timm==1.0.29','omegaconf':'omegaconf==2.3.0','einops':'einops','termcolor':'termcolor',
            'torchmetrics':'torchmetrics','psutil':'psutil','scikit-image':'scikit-image','matplotlib':'matplotlib',
            'pandas':'pandas','opencv-python':'opencv-python','scikit-learn':'scikit-learn','scipy':'scipy','Pillow':'Pillow',
            'tqdm':'tqdm'}
        missing=[]
        for package,spec in mapping.items():
            try:importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:missing.append(spec)
        try:adeval=importlib.metadata.version('adeval')
        except importlib.metadata.PackageNotFoundError:adeval=None
        if adeval!='1.1.0':missing.append('adeval==1.1.0')
        label='dependencies'
        if missing:command(label,[sys.executable,'-m','pip','install','--no-cache-dir','--no-deps',*missing],project,run,env)
        label='inputs'
        found=inputs(Path(input_root));frozen=run/'frozen_inputs';(frozen/'formal').mkdir(parents=True)
        for name,h in ip.old.read_json(project/'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/frozen_catalog.json')['formal_sha256'].items():
            p=found['formal']/name
            if ip.sha(p)!=h:raise ValueError('frozen Input mismatch: '+str(p))
            shutil.copyfile(p,frozen/'formal'/name)
        shutil.copyfile(found['state'],frozen/'image_calibration_state.json')
        config=dict(commit=commit,paths=dict(project_root=str(project),formal_dir=str(frozen/'formal'),
            image_state=str(frozen/'image_calibration_state.json'),data=str(found['data']),memory=str(found['memory']),
            checkpoint=str(found['checkpoint']),dinov3=str(project/'external_baselines/dinov3'),cache_root=str(run/'cache')),
            device='cuda:0',cpu_workers=4,capture_gpus=capture_gpus,
            capacity_estimates_enforced=False,model_parallelism='two domain hosts; capacity observations only; no pooled VRAM' if capture_gpus==2 else 'one original host',
            capture_devices={'view':'cuda:0','illumination':'cuda:1'} if capture_gpus==2 else {'both':'cuda:0'},
            memory_budget_gib=12,capture_min_free_vram_gib=14)
        write(run/'run_config.json',config)
        environment='''
import importlib,importlib.metadata,json,os,shutil,sys
from pathlib import Path
for n in ('numpy','scipy','PIL','sklearn','psutil','torch','torchvision','timm','einops','omegaconf','cv2','skimage','matplotlib','pandas','torchmetrics','termcolor','tqdm','adeval'):
 importlib.import_module(n);print('IMPORT_OK:',n,flush=True)
import torch
assert torch.cuda.is_available(),'Kaggle accelerator must be enabled'
print('TORCH:',torch.__version__,'CUDA:',torch.version.cuda,flush=True)
for i in range(torch.cuda.device_count()):
 print('GPU',i,torch.cuda.get_device_name(i),torch.cuda.mem_get_info(i),flush=True)
from DINOv3.relation_reliability.superadd_host import install_import_paths
project_root=Path.cwd();dino_root=project_root/'external_baselines/dinov3'
install_import_paths(project_root,project_root/'external_baselines/SuperADD',dino_root)
importlib.import_module('industrial.model');importlib.import_module('dinov3.hub.backbones')
assert importlib.metadata.version('adeval')=='1.1.0'
from DINOv3.relation_reliability.official_metrics import validate_official_metric_backend
validate_official_metric_backend()
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain.resources import available_ram,memory_snapshot
print('RAM_ACCOUNTING:',json.dumps(memory_snapshot()),flush=True)
print('AVAILABLE_RAM_GiB:',available_ram()/2**30,'DISK_FREE_GiB:',shutil.disk_usage('/kaggle/working').free/2**30,flush=True)
'''
        environment+=f'''\nassert torch.cuda.device_count()>={capture_gpus}, 'requested GPU count unavailable; no silent fallback'
print('ENVIRONMENT_OK; capacity observed only; no model constructed',flush=True)
'''
        label='environment';command(label,[sys.executable,'-B','-u','-c',environment],project,run,env)
        common=[]
        for key,value in config['paths'].items():common+=['--'+key.replace('_','-'),value]
        common+=['--official',str(project/'external_baselines/SuperADD'),'--domain','both','--device','cuda:0',
                 '--cpu-workers','4','--capture-min-free-vram-gib','14']
        def stage(name,*args):
            command(name,[sys.executable,'-B','-u','-m','DINOv3.MADEqual.aebad_s_support_distance_crossdomain.execution_worker',name,*common,*args],project,run,env)
            state['stages'].append(name);write(run/'pipeline_status.json',state)
        label='preflight';stage(label,'--no-write')
        label='capture'
        if capture_gpus==1:stage(label)
        else:
            from . import parallel_capture
            parallel_capture.run(parallel_capture.commands(common,run),project,run,env,command)
            state['stages'].append('capture');label='merge-capture'
            paths={k:Path(v) for k,v in config['paths'].items() if k!='project_root'}
            paths.update(project=project,official=project/'external_baselines/SuperADD')
            ctx=ip.old.context(paths,verify_data=False)
            parallel_capture.merge(ctx,{d:run/'capture_shards'/d for d in ('illumination','view')},run/'cache')
            state['stages'].append(label);write(run/'pipeline_status.json',state)
        if ip.old.read_json(run/'cache/cache_status.json')['status']!='capture_complete':raise ValueError('capture not complete')
        label='predict';stage(label,'--output-dir',str(run/'prediction'))
        label='evaluate';stage(label,'--prediction-dir',str(run/'prediction'),'--output-dir',str(run/'cpu'),'--memory-budget-gib','12')
        label='evaluate-aupro';stage(label,'--prediction-dir',str(run/'prediction'),'--cpu-dir',str(run/'cpu'),
            '--output-dir',str(run/'aupro'),'--aupro-allocator-limit-gib','1.5')
        if not ip.old.read_json(run/'aupro/status.json')['all_metrics_complete']:raise ValueError('AUPRO/metrics incomplete')
        label='diagnostic'
        command(label,[sys.executable,'-B','-u','-m','DINOv3.MADEqual.aebad_s_support_distance_crossdomain.execution_worker','diagnostic','analyze','--parent-run',str(run),
            '--data',str(found['data']),'--background',str(background),'--cpu-workers','4',
            '--output-dir',str(run/'diagnostic')],project,run,env)
        s=ip.old.read_json(run/'diagnostic/status.json')
        if s['status']!='cross_domain_summary_complete':raise ValueError('new diagnostic incomplete')
        state['stages'].append(label);state['computation_success']=True
    except BaseException as exc:
        state.update(failed_stage=label,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        say(state['traceback'])
    finally:
        state['seconds']=time.monotonic()-started
        summary=pack(run,state)
    if not state['experiment_success']:raise RuntimeError(f'Pipeline incomplete; outputs retained: {run}; summary={summary}')
    say('ALL_DONE')
    return run
