"""Default read-only preflight. No model/CUDA imports in CPU commands."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from . import inputs as ip


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight','capture','predict','evaluate','evaluate-aupro','summarize'))
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[3])
    for flag in ('formal-dir','image-state','data','memory','official','checkpoint','dinov3','cache-root','distance-reference-root'):
        p.add_argument('--'+flag,type=Path)
    p.add_argument('--domain',choices=('both',*ip.DOMAINS),default='both')
    p.add_argument('--query-feature-dir',type=Path);p.add_argument('--support-feature-dir',type=Path)
    p.add_argument('--reuse-cache-root',type=Path)
    p.add_argument('--output-dir',type=Path);p.add_argument('--prediction-dir',type=Path);p.add_argument('--cpu-dir',type=Path)
    p.add_argument('--result-dir',type=Path,action='append',default=[])
    p.add_argument('--cpu-workers',type=int,default=4);p.add_argument('--memory-budget-gib',type=float,default=8)
    p.add_argument('--device',default='cuda:0');p.add_argument('--aupro-allocator-limit-gib',type=float,default=1.5)
    p.add_argument('--no-write',action='store_true',help='preflight only; default preflight is always read-only')
    return p


def resources(ctx):
    from .method import BRANCHES
    from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.metrics import resource_bytes
    import math
    domains={};support_total=0;raw_total=0;prediction_total=0
    for domain,d in ctx['catalog']['domains'].items():
        def patches(s):return math.ceil(s['image_height']*.625/16)*math.ceil(s['image_width']*.625/16)
        rows=sum(patches(s) for s in d['samples']);support_rows=sum(patches(s) for s in d['supports'])
        pixels=sum(s['image_height']//4*(s['image_width']//4) for s in d['samples'])
        raw=rows*656;prediction=rows*256;support=support_rows*4*1280*4
        domains[domain]=dict(approximate_query_rows=rows,approximate_support_feature_bytes=support,
            approximate_uncompressed_raw_bytes=raw,approximate_prediction_bytes=prediction,exact_pixels=pixels,
            pixel_scratch_bytes=pixels*(1+4*len(BRANCHES)),exact_pixel_sort_budget_bytes=resource_bytes(pixels))
        raw_total+=raw;prediction_total+=prediction;support_total+=support
    return dict(domains=domains,approximate_retained_bytes=raw_total+prediction_total+support_total,
        capture_free_VRAM_guard_bytes=20*2**30,retrieval_only_free_VRAM_guard_bytes=8*2**30,
        capture_free_RAM_guard_bytes=12*2**30,CPU_domains_processed_serially=True,
        interpretation='estimates, not measured peaks; patch counts are conservative geometry estimates; actual geometry comes from original host; no protocol reductions',
        available_ram_bytes=ip.common.available_ram())


def preflight(ctx,args):
    r=ip.inventory(ctx,args.query_feature_dir,args.support_feature_dir)
    r['resources']=resources(ctx);r['GT_decoded']=False
    r['source_calibration_66_required']=False;r['output_created']=False
    needed_memory=any(v['reusable_complete_cache_images']!=v['heldout_images'] for v in r['domains'].values())
    needed_model=any(v['missing_query_extraction_images'] or (v['reusable_complete_cache_images']!=v['heldout_images'] and v['missing_support_indices']) for v in r['domains'].values())
    r['capture_blockers']=[a for a in r['assets'] if a['status']!='reusable' and
        ((a['input']=='memory' and needed_memory) or (a['input']!='memory' and needed_model))]
    r['torch_imported']='torch' in sys.modules
    if r['torch_imported']:raise RuntimeError('preflight unexpectedly imported Torch')
    r['readiness']='blocked' if r['identity_mismatches'] or r['capture_blockers'] else 'ready_for_explicit_capture_or_CPU_prediction'
    return r


def main(argv=None):
    args=parser().parse_args(argv)
    paths=ip.defaults(args.project_root)
    for key in (*paths,'distance_reference_root'):
        if hasattr(args,key) and getattr(args,key) is not None:paths[key]=getattr(args,key)
    if args.cpu_workers<1 or args.cpu_workers>8:raise ValueError('bounded CPU workers must be between 1 and 8')
    if args.memory_budget_gib<=0:raise ValueError('positive exact-metric RAM budget required')
    # No GT read by prediction/capture. Preflight checks GT byte identity only;
    # evaluation alone decodes masks, after verified full prediction freeze.
    ctx=ip.context(paths,verify_gt=args.stage=='preflight')
    domains=list(ip.DOMAINS) if args.domain=='both' else [args.domain]
    if args.stage=='preflight':
        report=preflight(ctx,args);print(json.dumps(report,ensure_ascii=False,indent=2))
        return 2 if report['readiness']=='blocked' else 0
    if args.no_write:raise ValueError('--no-write is only valid with preflight')
    if args.stage!='capture' and args.output_dir is None:raise ValueError('explicit fresh --output-dir required')
    output=args.output_dir
    if output is not None and output.exists():raise FileExistsError('output must be new; existing results remain unchanged')
    try:
        if args.stage=='capture':
            from .capture import run
            output=run(ctx,domains,device_name=args.device,query_features=args.query_feature_dir,
                support_features=args.support_feature_dir,reuse_root=args.reuse_cache_root)
        elif args.stage=='predict':
            from .prediction import run
            output=run(ctx,output,domains,args.cpu_workers)
        elif args.stage=='evaluate':
            if args.prediction_dir is None:raise ValueError('--prediction-dir required')
            from .analysis import run
            output=run(ctx,args.prediction_dir,output,args.cpu_workers,int(args.memory_budget_gib*2**30))
        elif args.stage=='evaluate-aupro':
            if args.prediction_dir is None or args.cpu_dir is None:raise ValueError('--prediction-dir and --cpu-dir required')
            from .aupro import run
            output=run(ctx,args.prediction_dir,args.cpu_dir,output,args.device,args.aupro_allocator_limit_gib)
        else:
            from .summary import run
            output=run(ctx,args.result_dir,output)
    except Exception as error:
        if args.stage!='capture' and output is not None and (output/'input_manifest.json').is_file():
            # Only this invocation's fresh directory is eligible for failure status.
            ip.write_json(output/'status.json',dict(tool=ip.TOOL,stage=args.stage,status='stage_failed',identity=ctx['identity'],error=f'{type(error).__name__}: {error}'))
        raise
    print('OUTPUT='+str(output));return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps(dict(tool=ip.TOOL,status='stage_failed',error=f'{type(error).__name__}: {error}'),ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
