"""Default read-only preflight. Explicit AutoDL stages, no automatic rebuild."""
from __future__ import annotations
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
from . import inputs as ip


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight','recover-calibration','capture','predict','evaluate','evaluate-aupro','summarize','package'))
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[3])
    for key in ('data','formal-metalparts','formal-pcb','source-root','official','dinov3','checkpoint','output-root','source-raw-root','feature-root','image-state-root'):
        p.add_argument('--'+key,type=Path)
    p.add_argument('--category',choices=('both',*ip.CATEGORIES),default='both')
    p.add_argument('--shift',default='all');p.add_argument('--scope',choices=('both','target','source'),default='both')
    p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-workers',type=int,default=4)
    p.add_argument('--memory-budget-gib',type=float,default=8);p.add_argument('--min-free-vram-gib',type=float,default=20)
    p.add_argument('--aupro-allocator-gib',type=float,default=1.5);p.add_argument('--evaluation-tag',default='v1')
    p.add_argument('--summary-dir',type=Path);p.add_argument('--archive',type=Path)
    p.add_argument('--raw-category',choices=ip.CATEGORIES);p.add_argument('--no-write',action='store_true')
    return p


def estimated_grid(sample):
    # Only a resource estimate. Real extraction supplies the actual stitched grid.
    h=int(sample['image_height']*.625);w=int(sample['image_width']*.625)
    if min(h,w)<640:
        factor=640/min(h,w);h=math.ceil(h*factor/16)*16;w=math.ceil(w*factor/16)*16
    return h//16,w//16


def inventory(ctx):
    from . import calibration
    import shutil
    rows=[];errors=[];source_pending=[];catalog=ctx['cat'];root=ctx['root']
    try:
        # Split-wise read-only discovery lets absent Source evaluation remain
        # ARD-pending while independently validating all Target groups.
        from external_baselines.superadd_external.dataset_adapter import _robust_split,ROBUSTAD_LAYOUT
        cat=ctx['category'];prefix,shifts=ROBUSTAD_LAYOUT[cat];base=ctx['paths']['data']/cat
        train=[r for r in _robust_split(cat,base/(prefix+'_train'),'source','source_train',evaluation=False) if r.label==0]
        _,cal=ip.split_source_normal(train,8)
        canonical=lambda rr:[Path(r.path).resolve().relative_to(ctx['paths']['data']).as_posix() for r in rr]
        if canonical(train)!=[s['sample_id'] for s in catalog['train']] or canonical(cal)!=[s['sample_id'] for s in catalog['calibration']]:raise ValueError('original Source metadata order/split differs')
        expected={s['sample_id']:s for s in catalog['source_evaluation']}
        for v in catalog['shifts'].values():expected.update({s['sample_id']:s for s in v['samples']+v['supports']})
        discovered=[]
        for i,shift in [(0,'source'),*sorted(shifts.items())]:
            try:
                rr=_robust_split(cat,base/(prefix+'_test'+str(i)),'source' if i==0 else 'target',shift,evaluation=True)
            except FileNotFoundError as exc:
                if i!=0:raise
                source_pending.append(str(exc));continue
            if canonical(rr)!=[s['sample_id'] for s in catalog['source_evaluation']] and i==0:raise ValueError('Source evaluation order differs')
            discovered.extend(rr)
        actual={Path(r.path).resolve().relative_to(ctx['paths']['data']).as_posix():r for r in discovered}
        required={k for k,v in expected.items() if not source_pending or v['shift']!='source'}
        if set(actual)!=required:raise ValueError('evaluation/support metadata identities differ')
        for rel,r in actual.items():
            if r.label!=expected[rel]['label'] or r.shift!=expected[rel]['shift']:raise ValueError('metadata labels/shift differ')
        for s in catalog['train']+list(expected.values()):
            if source_pending and s['shift']=='source':continue
            p=ctx['paths']['data']/s['relative_path']
            if ip.sha(p)!=s['image_sha256']:raise ValueError('image bytes differ: '+str(p))
            if s.get('mask_path'):
                try:
                    if ip.sha(ctx['paths']['data']/s['mask_path'])!=s['gt_sha256']:raise ValueError('GT bytes differ')
                except FileNotFoundError as exc:
                    if s['shift']!='source':raise
                    source_pending.append(str(exc))
    except (OSError,ValueError,KeyError,RuntimeError) as exc:errors.append(str(exc))
    state=ip.ledger(ctx) if (root/'inputs.json').is_file() else {'records':[]}
    cached={(r['kind'],r['sample_id']) for r in state['records']}
    image_state='missing'
    if (root/'image_calibration_state.json').is_file():
        try:calibration.load_state(ctx);image_state='verified_category_Source_image_MAD'
        except (OSError,ValueError,KeyError) as exc:errors.append(str(exc))
    raw=0;support=0;prediction=0
    for shift,d in catalog['shifts'].items():
        rows.append(dict(category=ctx['category'],shift=shift,normal=d['normal_count'],anomaly=d['anomaly_count'],
            images=len(d['samples']),supports=len(d['supports']),cached_queries=sum(('target',s['sample_id']) in cached for s in d['samples'])))
        nr=sum(math.prod(estimated_grid(s)) for s in d['samples']);sr=sum(math.prod(estimated_grid(s)) for s in d['supports'])
        raw+=nr*656;support+=sr*4*1280*4;prediction+=nr*256
    assets=[]
    disk_path=root
    while not disk_path.exists():disk_path=disk_path.parent
    source_rows=sum(math.prod(estimated_grid(s)) for s in catalog['calibration']+catalog['source_evaluation'])
    max_pixels=max(sum((s['image_height']//4)*(s['image_width']//4) for s in group) for group in [catalog['source_evaluation'],*[v['samples'] for v in catalog['shifts'].values()]])
    for role,p in [('original_Source_bank',ctx['bank']),('DINOv3_checkpoint',ctx['paths']['checkpoint']),('DINOv3_official_entry',ctx['paths']['dinov3']/'hubconf.py')]:
        assets.append(dict(role=role,path=str(p),local_or_current_runtime_exists=p.is_file(),
            running_server_status='must_check_on_AutoDL',missing_action='stop_needed_GPU_stage_no_rebuild_or_download'))
    candidates=[]
    raw_root=ctx['paths']['project']/'server_results/relation_oracle_seed42_cache_and_audit/mvtec_robustad_seed42_v2'
    raw_root=ctx['paths'].get('source_raw_root',raw_root)
    if (raw_root/'categories/robustad'/ctx['category']/'complete.json').is_file():
        index=set()
        for p in (raw_root/'categories/robustad'/ctx['category']/'images').rglob('*.npz'):
            with np.load(p,allow_pickle=False) as f:index.add(ip.canonical(str(f['image_path'].item()),ctx['category']))
        candidates.append(dict(root=str(raw_root),calibration_candidates=sum(s['sample_id'] in index for s in catalog['calibration']),
            Source_evaluation_candidates=sum(s['sample_id'] in index for s in catalog['source_evaluation']),
            Target_Source_distance_candidates=sum(s['sample_id'] in index for d in catalog['shifts'].values() for s in d['samples']),
            validation='headers/paths discovered; original bank-dependent identity still requires runtime validation',
            selected_by_argument='source_raw_root' in ctx['paths']))
    return dict(category=ctx['category'],identity=ctx['identity'],extraction_identity=ctx['extraction_identity'],shifts=rows,
        Source_train=180,memory_fit=157,calibration=23,Source_evaluation=len(catalog['source_evaluation']),
        cached_calibration=sum(('calibration',s['sample_id']) in cached for s in catalog['calibration']),
        cached_Source_evaluation=sum(('source',s['sample_id']) in cached for s in catalog['source_evaluation']),
        image_calibration=image_state,assets=assets,identity_errors=errors,ARD_pending_missing_Source_inputs=source_pending,
        legacy_Source_cache_candidates=candidates,
        original_bank_historical_byte_hash_available=False,bank_first_enrollment_required=not (root/'inputs.json').is_file(),
        resource_estimate=dict(raw_uncompressed_bytes=raw,support_feature_bytes=support,prediction_uncompressed_bytes=prediction,
            total_new_persistent_estimate_bytes=raw+support+prediction,Source_builder_scratch_bytes=0,available_RAM_bytes=ip.available_ram(),
            Source_observation_estimate_bytes=source_rows*56,max_CPU_scratch_estimate_bytes=max_pixels*13+(512<<20),
            max_exact_pixel_working_estimate_bytes=max_pixels*32+(512<<20),disk_free_bytes=shutil.disk_usage(disk_path).free,
            interpretation='fresh-run target persistent estimate, before compression; Source and sequential CPU scratch listed separately; existing caches/bank/checkpoint and archive copies not included; not measured peaks'),
        GT_decoded=False,model_loaded=False,CUDA_initialized=False,output_created=False)


def main(argv=None):
    args=parser().parse_args(argv)
    if not 1<=args.cpu_workers<=8:raise ValueError('cpu-workers must be 1..8')
    if any(not math.isfinite(v) or v<=0 for v in (args.memory_budget_gib,args.min_free_vram_gib,args.aupro_allocator_gib)):raise ValueError('positive finite resource settings required')
    if args.no_write and args.stage!='preflight':raise ValueError('--no-write is only valid for preflight')
    ip.safe(args.evaluation_tag)
    if args.category=='both' and args.shift!='all':raise ValueError('select one category for an individual shift')
    paths=ip.defaults(args.project_root)
    for k in (*paths,'source_raw_root','feature_root','image_state_root'):
        if hasattr(args,k) and getattr(args,k) is not None:paths[k]=getattr(args,k)
    categories=ip.CATEGORIES if args.category=='both' else (args.category,)
    if args.stage=='preflight':
        from importlib.metadata import version,PackageNotFoundError
        versions={}
        for name in ('numpy','scipy','Pillow','scikit-learn','psutil','torch','torchvision','timm','omegaconf','adeval'):
            try:versions[name]=version(name)
            except PackageNotFoundError:versions[name]=None
        reports=[];errors=[]
        for cat in categories:
            try:reports.append(inventory(ip.context(paths,cat)))
            except (OSError,ValueError,KeyError) as exc:errors.append(dict(category=cat,error=str(exc)))
        blocked=bool(errors) or any(r['identity_errors'] for r in reports)
        missing=any(not a['local_or_current_runtime_exists'] for r in reports for a in r['assets']) or any(v is None for v in versions.values())
        result=dict(tool=ip.TOOL,status='blocked' if blocked or missing else 'filesystem_preflight_passed',protocol=ip.PROTOCOL,
            categories=reports,errors=errors,torch_imported='torch' in sys.modules,execution_environment='user_selected_AutoDL',
            dependency_versions=versions,missing_dependencies=[n for n,v in versions.items() if v is None],
            note='missing local bank is not evidence of missing server bank; no remote check performed')
        if result['torch_imported']:raise RuntimeError('preflight unexpectedly imported Torch')
        print(json.dumps(result,ensure_ascii=False,indent=2));return 2 if blocked or missing else 0
    contexts=[ip.context(paths,c,enroll=True) for c in categories]
    if args.stage in ('summarize','package'):
        from . import reporting
        out=args.summary_dir or Path(paths['output_root'])/('summary_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        if args.stage=='summarize':print('SUMMARY_OUTPUT='+str(reporting.summarize(contexts,args.evaluation_tag,out)));return 0
        if args.summary_dir is None or args.archive is None:raise ValueError('package requires --summary-dir and --archive')
        print(json.dumps(reporting.package(contexts,out,args.archive,args.raw_category),ensure_ascii=False,indent=2));return 0
    from . import calibration,runtime,prediction,evaluation
    for ctx in contexts:
        shifts=ip.selected(ctx,args.shift)
        if args.stage=='recover-calibration':calibration.recover(ctx,args.device,args.min_free_vram_gib);continue
        if args.stage=='capture':runtime.capture(ctx,shifts,args.scope,args.device,args.min_free_vram_gib);continue
        groups=(['source'] if args.scope in ('source','both') else [])+(shifts if args.scope in ('target','both') else [])
        for group in groups:
            if args.scope=='both' and group=='source' and {r['sample_id'] for r in ip.ledger(ctx)['records'] if r['kind']=='source'}!={s['sample_id'] for s in ctx['cat']['source_evaluation']}:
                print(f'ARD_PENDING={ctx["category"]}: Source evaluation cache missing; continuing Target-only work',flush=True);continue
            if args.scope=='both' and group=='source' and args.stage in ('evaluate','evaluate-aupro'):
                missing=[s['sample_id'] for s in ctx['cat']['source_evaluation'] if s.get('mask_path') and not (ctx['paths']['data']/s['mask_path']).is_file()]
                if missing:
                    print(f'ARD_PENDING={ctx["category"]}: {len(missing)} Source GT inputs missing; continuing Target-only work',flush=True);continue
            if args.stage=='predict':prediction.run(ctx,group,args.cpu_workers)
            elif args.stage=='evaluate':evaluation.evaluate(ctx,group,args.evaluation_tag,args.cpu_workers,int(args.memory_budget_gib*2**30))
            else:evaluation.evaluate_aupro(ctx,group,args.evaluation_tag,args.device,args.aupro_allocator_gib)
    print('STAGE_COMPLETE='+args.stage);return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(json.dumps(dict(tool=ip.TOOL,status='stage_failed',error=f'{type(exc).__name__}: {exc}'),ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
