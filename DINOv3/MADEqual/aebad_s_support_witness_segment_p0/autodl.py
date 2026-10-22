"""User-launched AutoDL pipeline: recover supports, cached experiment, review tar.

No SSH, download, environment installation, resource admission or query inference.
Explicit one/two GPU selection. Shutdown is opt-in and successful-completion-only.
"""
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


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',required=True,type=Path);p.add_argument('--data',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--dinov3',type=Path,default=ip.parent.ROOT/'external_baselines/dinov3')
    p.add_argument('--devices',nargs='+',default=['cuda:0']);p.add_argument('--cpu-workers',type=int,default=8)
    p.add_argument('--query-chunk',type=int,default=64);p.add_argument('--reference-chunk',type=int,default=1024)
    p.add_argument('--shutdown-on-success',action='store_true')
    return p


def run(stage,cmd,root):
    log=root/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    print(f'STAGE={stage} START; LOG={log}',flush=True)
    with log.open('a',encoding='utf-8') as f:
        process=subprocess.Popen(cmd,cwd=ip.parent.ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        def pump():
            for line in process.stdout:
                f.write(line);f.flush();print(line,end='',flush=True)
        reader=threading.Thread(target=pump,daemon=True);reader.start()
        while process.poll() is None:
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:print(f'[heartbeat] stage={stage} seconds={time.perf_counter()-started:.0f}',flush=True)
        reader.join();rc=process.returncode
    (root/'logs'/(stage+'.exitcode')).write_text(str(rc)+'\n',encoding='ascii')
    print(f'STAGE={stage} EXIT_CODE={rc} SECONDS={time.perf_counter()-started:.1f}',flush=True)
    if rc:raise RuntimeError(f'{stage} failed: exit_code={rc}; preserved {log}')


def pack(root,success):
    destination=root.parent/(root.name+('_review.tar.gz' if success else '_FAILED_review.tar.gz'))
    # Review is intentionally separate from the potentially large feature,
    # neighbor, connection and prediction caches; all originals remain intact.
    if destination.exists():raise FileExistsError('prior review archive preserved; choose a fresh run or archive name')
    with tarfile.open(destination,'x:gz',compresslevel=1) as archive:
        for p in sorted(root.rglob('*')):
            if (p.is_file() and p.suffix in ('.json','.csv','.md','.log','.exitcode','.png')
                and not any(x in ('features','_scratch','scratch') for x in p.relative_to(root).parts)):
                archive.add(p,arcname=root.name+'/'+p.relative_to(root).as_posix(),recursive=False)
    return destination


ENVIRONMENT=r'''
import importlib,json
from importlib.metadata import version
for name in ('numpy','scipy','PIL','sklearn','psutil','matplotlib','torch','torchvision','timm','omegaconf','adeval'):
    importlib.import_module(name);print('IMPORT_OK:',name,flush=True)
if version('adeval')!='1.1.0':raise ValueError('adeval==1.1.0 required; existing environment not changed')
from DINOv3.relation_reliability.official_metrics import validate_official_metric_backend
validate_official_metric_backend()
print('ENVIRONMENT_OK; no model constructed',flush=True)
'''


def main(argv=None):
    a=parser().parse_args(argv)
    if len(a.devices) not in (1,2) or len(set(a.devices))!=len(a.devices) or any(not d.startswith('cuda:') for d in a.devices):
        raise ValueError('choose one/two distinct explicit CUDA devices')
    if not 1<=a.cpu_workers<=8:raise ValueError('cpu-workers must be 1..8')
    root=a.run_root.resolve()
    # Same strict overlap rule as the underlying stages, BEFORE creating logs.
    protected=[*ip.parent.defaults().values(),*ip.parent.bg.defaults(ip.parent.ROOT).values(),a.data,a.checkpoint,a.dinov3]
    for p in protected:
        p=Path(p).resolve()
        if root==p or root in p.parents or p in root.parents:raise ValueError('run root overlaps original input: '+str(p))
    config=dict(code={p.name:st.sha(p) for p in Path(__file__).parent.glob('*.py')},
        data=str(a.data.resolve()),checkpoint=str(a.checkpoint.resolve()),dinov3=str(a.dinov3.resolve()),
        devices=a.devices,cpu_workers=a.cpu_workers,query_chunk=a.query_chunk,reference_chunk=a.reference_chunk)
    if root.exists():
        if not (root/'launcher_config.json').is_file() or st.read_json(root/'launcher_config.json')!=config:raise ValueError('run configuration differs; use fresh directory')
    else:root.mkdir(parents=True);st.write_json(root/'launcher_config.json',config)
    print('RUN_ROOT='+str(root),flush=True)
    if a.shutdown_on_success:print('AUTO_SHUTDOWN=only after full success AND successful review packaging; failure stays powered on',flush=True)
    output=root/'experiment';recovered=root/'recovered_supports';module=__package__+'.runner'
    common=['--output-dir',str(output),'--data',str(a.data),'--recovered-supports',str(recovered)]
    recovery=['--checkpoint',str(a.checkpoint),'--dinov3',str(a.dinov3)]
    devices=['--devices',*a.devices];chunks=['--query-chunk',str(a.query_chunk),'--reference-chunk',str(a.reference_chunk)]
    stage_args=[('recover-support-preflight','recover-support',common+recovery+['--no-write']),
        ('recover-support','recover-support',common+recovery+devices),('verify-support','verify-support',common),
        ('preflight','preflight',common+['--no-write']),('prepare-support','prepare-support',common+devices+chunks),
        ('support-loo','support-loo',common),('predict','predict',common),
        ('evaluate','evaluate',common+['--cpu-workers',str(a.cpu_workers)]),
        ('evaluate-aupro','evaluate-aupro',common+devices),('summarize','summarize',common)]
    started=time.perf_counter();state=dict(status='running',success=False,query_forward_calls=0)
    error=None;archive=None
    try:
        state['stage']='environment';run('environment',[sys.executable,'-B','-u','-c',ENVIRONMENT],root)
        for label,stage,args in stage_args:
            state['stage']=label;st.write_json(root/'pipeline_status.json',state)
            run(label,[sys.executable,'-B','-u','-m',module,stage,*args],root)
        result=st.read_json(output/'status.json')
        if (result.get('status')!='all_requested_stages_complete' or result.get('all_metrics_complete') is not True
            or result.get('pending') or result.get('numerical_failed_checks')!=0):raise ValueError('metrics/LOO/verification pending; no success shutdown')
        state.update(status='all_requested_stages_complete',success=True)
    except Exception as exc:
        error=exc;state.update(status='failed_outputs_retained',success=False,error=f'{type(exc).__name__}: {exc}')
    finally:
        state['seconds']=time.perf_counter()-started;st.write_json(root/'pipeline_status.json',state)
        try:
            archive=pack(root,state['success']);print('REVIEW_ARCHIVE='+str(archive),flush=True)
        except Exception as exc:
            state.update(status='pack_failed_outputs_retained',success=False,pack_error=f'{type(exc).__name__}: {exc}')
            st.write_json(root/'pipeline_status.json',state)
        rc=0 if state['success'] and archive is not None else 2
        (root.parent/(root.name+'.exitcode')).write_text(str(rc)+'\n',encoding='ascii')
        print(f'PIPELINE_EXIT_CODE={rc}\nELAPSED_SECONDS={time.perf_counter()-started:.1f}\nRUN_ROOT={root}',flush=True)
        if error:print('ERROR='+str(error),flush=True)
    if rc==0 and a.shutdown_on_success:
        print('All requested checks and packaging passed. Shutting down AutoDL now.',flush=True)
        subprocess.run(['shutdown','-h','now'],check=True)
    return rc


if __name__=='__main__':raise SystemExit(main())
