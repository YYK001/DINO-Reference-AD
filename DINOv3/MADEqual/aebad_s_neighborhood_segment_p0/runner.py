"""Default is a read-only preflight. Every computation requires an explicit stage."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import inputs as ip
from .prediction import bounded_images


def preflight(paths,workers=4,memory_budget_bytes=8<<30):
    started=time.perf_counter();missing=[]
    for key,path in paths.items():
        if not Path(path).exists(): missing.append(dict(input=key,path=str(path)))
    report=dict(tool=ip.TOOL,status='blocked',missing_assets=missing,errors=[],torch_imported=False,
                model_loaded=False,feature_extraction=False,GT_decoded=False,predictions_computed=False,
                required_model=False,required_Source_memory=False,required_support_features=False)
    if missing: return report
    try:
        ctx=ip.context(paths,check_data=True)
        def one(sample):
            with ip.Archive(paths['raw_archive'], index=ctx['raw_index']) as a:
                _,meta,_=ip.load_image(ctx,a,sample)
            return int(__import__('numpy').prod(meta['output_shape'])),len(ctx['segments'])
        sizes=[size for size,_ in bounded_images(ctx['samples'],one,workers)]
        from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.metrics import resource_bytes
        pixels=sum(sizes)
        report.update(status='filesystem_preflight_passed',identity=ctx['identity'],
                      images=297,normal_images=85,anomaly_images=212,supports=8,
                      capture_files_verified=297,segment_files_verified=1188,
                      endpoints_verified=True,query_support_geometry_verified=True,
                      CH1_image_MAD_sha256=ctx['state']['state_sha256'],
                      resource_estimate=dict(complete_evaluation_pixels=pixels,
                          new_distance_gate_output_uncompressed_bytes=300<<20,
                          CPU_pixel_scratch_bytes=pixels*9,exact_pixel_sort_budget_required_bytes=resource_bytes(pixels),
                          configured_sort_budget_bytes=memory_budget_bytes,CPU_workers=workers,
                          available_ram_bytes=ip.available_ram(),minimum_available_evaluation_ram_bytes=resource_bytes(pixels)+(512<<20),
                          interpretation='estimates not measured peaks; new predictions only, no copied original caches'),
                      AUPRO='pending_explicit_original_CUDA_ADEval_200_FPR_0.3')
        report['environment']=ip.environment()
        if 'torch' in sys.modules: raise RuntimeError('CPU preflight unexpectedly imported Torch')
        if ip.context(paths)['identity']!=ctx['identity']: raise ValueError('inputs changed during preflight')
    except Exception as error:
        report['status']='blocked';report['errors'].append(f'{type(error).__name__}: {error}')
    report['seconds']=time.perf_counter()-started
    return report


def parser():
    root=Path(__file__).resolve().parents[3]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight','predict','evaluate','evaluate-aupro','visualize'))
    for key,value in ip.defaults(root).items(): p.add_argument('--'+key.replace('_','-'),type=Path,default=value)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--prediction-dir',type=Path)
    p.add_argument('--cpu-analysis-dir',type=Path)
    p.add_argument('--cpu-workers',type=int,default=4)
    p.add_argument('--pixel-sort-budget-gib',type=float,default=8)
    p.add_argument('--device',default='cuda:0',help='used ONLY by explicit evaluate-aupro')
    p.add_argument('--no-write',action='store_true',help='preflight already never writes; invalid for other stages')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    paths={key:getattr(args,key) for key in ip.defaults(Path.cwd())}
    budget=int(args.pixel_sort_budget_gib*2**30)
    if not 1<=args.cpu_workers<=8 or budget<=0: raise ValueError('invalid CPU workers/budget')
    if args.stage=='preflight':
        report=preflight(paths,args.cpu_workers,budget)
        print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['status']=='filesystem_preflight_passed' else 2
    if args.no_write or args.output_dir is None: raise ValueError('explicit computation requires --output-dir and no --no-write')
    ctx=ip.context(paths,check_data=args.stage in ('evaluate','evaluate-aupro','visualize'))
    new_output=not args.output_dir.exists()
    try:
        return execute(args,ctx,budget)
    except Exception as error:
        status_file=args.output_dir/'status.json'
        if new_output and status_file.is_file():
            status=ip.json_file(status_file)
            if status.get('tool')==ip.TOOL and status.get('identity')==ctx['identity']:
                status.update(status='failed',error=f'{type(error).__name__}: {error}',all_metrics_complete=False)
                ip.write_json(status_file,status)
        raise


def execute(args,ctx,budget):
    if args.stage=='predict':
        from .prediction import predict
        out=predict(ctx,args.output_dir,args.cpu_workers)
    else:
        if args.prediction_dir is None: raise ValueError('--prediction-dir is required')
        if args.stage=='evaluate':
            from .analysis import evaluate
            out=evaluate(ctx,args.prediction_dir,args.output_dir,args.cpu_workers,budget)
        elif args.stage=='evaluate-aupro':
            if args.cpu_analysis_dir is None: raise ValueError('--cpu-analysis-dir is required')
            from .aupro import evaluate_aupro
            out=evaluate_aupro(ctx,args.prediction_dir,args.cpu_analysis_dir,args.output_dir,args.device)
        else:
            from .visualize import visualize
            out=visualize(ctx,args.prediction_dir,args.output_dir)
    if args.stage!='evaluate-aupro' and 'torch' in sys.modules: raise RuntimeError('CPU stage unexpectedly imported Torch')
    print('OUTPUT='+str(out));return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as error:
        # Preserve all intermediate data; only this invocation's new output may
        # receive a failed status. Existing results are never modified.
        print(json.dumps(dict(tool=ip.TOOL,status='failed',error=f'{type(error).__name__}: {error}'),ensure_ascii=False),file=sys.stderr)
        raise SystemExit(1)
