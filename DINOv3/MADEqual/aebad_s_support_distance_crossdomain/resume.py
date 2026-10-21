"""Resume frozen predictions from a saved Notebook Output, without a model.

Input folders stay read-only. Only metadata is copied; raw and prediction
directories are linked to the saved output. Each computation has a fresh process.
"""
import argparse
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from . import inputs as ip,kaggle,resources
from .resource_policy import observation_only


def context(root,data):
    paths=ip.old.defaults(ip.ROOT)
    paths.update(formal_dir=root/'frozen_inputs/formal',image_state=root/'frozen_inputs/image_calibration_state.json',
                 cache_root=root/'cache',data=Path(data))
    # No model/memory assets or repeated image scan at resume construction.
    # Actual prediction, map, score and GT checks in the original evaluator stay.
    return ip.old.context(paths,verify_data=False)


def prepare(source,out,data):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('fresh resume output required; never overwrite the failed run')
    if out==source or source in out.parents or out in source.parents:raise ValueError('resume output overlaps source')
    cache=ip.old.read_json(source/'cache/cache_status.json')
    pred=ip.old.read_json(source/'prediction/status.json')
    pm=ip.old.read_json(source/'prediction/prediction_manifest.json')
    frozen=ip.old.read_json(source/'cache/input_manifest.json')
    prediction_frozen=ip.old.read_json(source/'prediction/input_manifest.json')
    if (cache['status']!='capture_complete' or pred['status']!='predictions_frozen' or pred['images']!=629
        or pm['images']!=629 or len(cache['records'])!=629 or len(pm['records'])!=629
        or frozen!=prediction_frozen or ip.digest(frozen['frozen'])!=cache['identity']
        or any(s['identity']!=cache['identity'] for s in (frozen,pred,pm))
        or pred['prediction_manifest_sha256']!=ip.sha(source/'prediction/prediction_manifest.json')):
        raise ValueError('source capture/prediction is incomplete, incompatible or modified')
    catalog=ip.old.read_json(Path(ip.old.__file__).with_name('frozen_catalog.json'))
    expected={(d,s['sample_id']) for d,v in catalog['domains'].items() for s in v['samples']}
    if ({(r['domain'],r['sample_id']) for r in cache['records']}!=expected
        or {(r['domain'],r['sample_id']) for r in pm['records']}!=expected):
        raise ValueError('source is not the original complete 629-image population')
    out.mkdir(parents=True)
    shutil.copytree(source/'frozen_inputs',out/'frozen_inputs')
    # No copy of 629 raws, 629 predictions or unused support-feature entities.
    for name in ('cache','prediction'):(out/name).symlink_to(source/name,target_is_directory=True)
    ctx=context(out,data)
    if ctx['identity']!=cache['identity']:raise ValueError('current original method code differs; source identity not rewritten')
    config=ip.old.read_json(source/'run_config.json')
    source_config=json.loads(json.dumps(config))
    config['paths']={('project_root' if k=='project' else k):str(v) for k,v in ctx['paths'].items()}
    config.update(resume_source=str(source),model_inference=False,feature_extraction=False,capacity_estimates_enforced=False)
    ip.write_json(out/'run_config.json',config)
    with observation_only() as audit:ip.write_json(out/'capacity_policy.json',audit)
    ip.write_json(out/'resume_source.json',dict(source=str(source),original_config=source_config,
        source_metadata_sha256={name:ip.sha(source/name) for name in
            ('cache/cache_status.json','cache/input_manifest.json','prediction/status.json','prediction/prediction_manifest.json','run_config.json')},
        model_inference=False,feature_extraction=False,capacity_estimates_enforced=False,
        original_raw_and_prediction_arrays_copied=False))
    return out


def pack(root,success):
    archive=root.parent/(root.name+('_SUCCESS_SUMMARY.tar.gz' if success else '_FAILED_SUMMARY.tar.gz'))
    with tarfile.open(archive,'x:gz',compresslevel=1) as t:
        for directory,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in ('cache','prediction') and not d.startswith('_scratch')]
            for name in files:
                p=Path(directory)/name;t.add(p,arcname=root.name+'/'+p.relative_to(root).as_posix(),recursive=False)
        # Portable metadata only. Never archive links to external Input folders.
        for name in ('cache','prediction'):
            for p in (root/name).iterdir():
                if p.is_file() and p.suffix in ('.json','.csv','.md'):
                    t.add(p,arcname=root.name+'/'+name+'/'+p.name,recursive=False)
    return archive


def worker(stage,root,data,workers,device,budget):
    ctx=context(root,data)
    print('RESOURCE_OBSERVATION_ONLY:',json.dumps(resources.memory_snapshot()),flush=True)
    with observation_only():
        if stage=='evaluate':
            from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import analysis
            analysis.run(ctx,root/'prediction',root/'cpu',workers,int(budget*2**30))
        elif stage=='evaluate-aupro':
            from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import aupro
            aupro.run(ctx,root/'prediction',root/'cpu',root/'aupro',device,1.5)
            if not ip.old.read_json(root/'aupro/status.json')['all_metrics_complete']:raise ValueError('AUPRO incomplete')
        else:
            from . import runner,reporting
            catalog=Path(__file__).with_name('background_discovery_catalog.json');reporting.verified_catalog(catalog)
            rc=runner.main(['analyze','--parent-run',str(root),'--data',str(data),'--background',str(catalog),
                           '--cpu-workers',str(workers),'--output-dir',str(root/'diagnostic')])
            if rc or ip.old.read_json(root/'diagnostic/status.json')['status']!='cross_domain_summary_complete':
                raise ValueError('diagnostic incomplete')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('all','evaluate','evaluate-aupro','diagnostic'))
    parser.add_argument('--source-run',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--cpu-workers',type=int,choices=(1,2,3,4),default=4)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--memory-budget-gib',type=float,default=12)
    args=parser.parse_args(argv);root=args.output_dir.resolve()
    if args.stage!='all':
        worker(args.stage,root,args.data,args.cpu_workers,args.device,args.memory_budget_gib);return 0
    if args.source_run is None:parser.error('--source-run is required for all')
    prepare(args.source_run,root,args.data)
    state=dict(status='resume_running',completed=[],model_inference=False,feature_extraction=False,capacity_estimates_enforced=False)
    started=time.monotonic();label='initialization';success=False
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    try:
        for label in ('evaluate','evaluate-aupro','diagnostic'):
            command=[sys.executable,'-B','-u','-m',__package__+'.resume',label,'--output-dir',str(root),
                '--data',str(args.data),'--cpu-workers',str(args.cpu_workers),'--device',args.device,'--memory-budget-gib',str(args.memory_budget_gib)]
            kaggle.command('resume_'+label,command,ip.ROOT,root,env)
            state['completed'].append(label);ip.write_json(root/'resume_status.json',state)
        success=True;state['status']='all_requested_metrics_and_diagnostic_complete'
    except Exception as exc:
        state.update(status='resume_failed',failed_stage=label,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state.update(seconds=time.monotonic()-started,experiment_success=success)
        ip.write_json(root/'resume_status.json',state)
        try:archive=pack(root,success)
        except Exception as exc:
            state.update(computation_success=success,experiment_success=False,archive_error=f'{type(exc).__name__}: {exc}')
            ip.write_json(root/'resume_status.json',state);raise
        print('SUMMARY_ARCHIVE='+str(archive),flush=True)
        print('RAW_CACHE_NOT_DUPLICATED; retain the original RAW_AND_REFERENCES.tar',flush=True)
        print('EXPERIMENT_SUCCESS='+str(success),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
