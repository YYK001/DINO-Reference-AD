"""Default read-only preflight; all computation requires an explicit stage."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from . import inputs


def workers(value):
    number=int(value)
    if not 1<=number<=15:raise argparse.ArgumentTypeError('cpu-workers must be between 1 and 15')
    return number


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('mode',choices=('preflight','prepare-target','infer','evaluate'),nargs='?',default='preflight')
    result.add_argument('--project-root',default=str(Path(__file__).resolve().parents[3]))
    result.add_argument('--sparc-base-dir');result.add_argument('--output-dir')
    result.add_argument('--cpu-workers',type=workers,default=min(8,os.cpu_count() or 1),
        help='bounded CPU correction workers; up to 8 support tasks or 4 query-layer tasks, no GPU algorithm change')
    result.add_argument('--device',default='cuda:0')
    result.add_argument('--no-write',action='store_true',help='preflight is always read-only; flag documents this explicitly')
    return result


def stage_cost(data,mode,state):
    from DINOv3.MADEqual.aebad_s_target_local_segment.runner import memory_observation
    root=Path(data['paths']['output'])
    cost=dict(tool=inputs.TOOL,stage=mode,status=state['status'],seconds=state['seconds'],timing=state.get('timing',{}),
        **memory_observation(cuda_evaluation_started='torch' in sys.modules),
        parent_identity=data['frozen']['parent_identity'],historical_adaptation_cost_repeated=False,
        new_output_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        Source_candidate_scratch_bytes=0,peak_scope='whole current stage; not per branch',
        verification_overhead='source_verification_retrieval',
        parallel_scope='CPU original Projection.correct: independent supports/layers; ordered results, GPU searches sequential',
        cpu_workers=data['frozen']['cpu_workers'])
    if 'torch' in sys.modules:
        torch=sys.modules['torch'];cost['precision']=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    inputs.replay.atomic_json(root/f'cost_{mode.replace("-","_")}.json',cost)


def main(argv=None):
    args=parser().parse_args(argv);paths=inputs.paths_for(args);started=time.perf_counter();may_write=False
    try:
        if args.no_write and args.mode!='preflight':raise ValueError('--no-write only applies to preflight')
        if args.mode in ('preflight','prepare-target'):
            check=inputs.preflight(args)
            report={k:check[k] for k in ('tool','status','errors','missing_assets','paths','model_loaded','feature_extraction','source_builder_called','MAD_fit_called')}
            report.update(resources=check.get('resources'),resource_shape_source=check['resource_shape_source'],parent_identity=check['parent']['identity'] if check['parent'] else None,
                          torch_imported='torch' in sys.modules)
            print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
            if check['errors']:return 2
            if args.mode=='preflight':return 0
            data=inputs.freeze(check);root=Path(data['paths']['output']);root.mkdir(parents=True,exist_ok=True)
            inputs.replay.atomic_json(root/'inputs.json',data);may_write=True
            from .runtime import prepare_target
            state=prepare_target(data)
        else:
            data,prepared=inputs.read_prepared(paths['output']);inputs.guard_output(data['paths'],prepared=True)
            if any(str(paths[k])!=data['paths'][k] for k in paths):raise ValueError('frozen project/parent/output path changed')
            if args.cpu_workers!=data['frozen']['cpu_workers']:raise ValueError('cpu-workers differs from frozen preparation')
            root=Path(data['paths']['output']);may_write=True
            if args.mode=='infer':
                from .runtime import infer
                data['device']=args.device;state=infer(data,prepared)
            else:
                from .analysis import evaluate
                from .runtime import evaluation_device
                evaluation_device(data,args.device)
                state=evaluate(data,prepared)
        stage_cost(data,args.mode,state);inputs.replay.atomic_json(root/'status.json',dict(state,stage=args.mode))
        print(f"SPARC_UNION_{args.mode.upper()}_STATUS={state['status']}\nOUTPUT={root}",flush=True)
        return 0 if state['status'] in ('target_prepared','predictions_frozen','analysis_complete') else 3
    except Exception as e:
        failure=dict(tool=inputs.TOOL,stage=args.mode,status='stage_failed',error=f'{type(e).__name__}: {e}',seconds=time.perf_counter()-started)
        if may_write:
            path=Path(paths['output'])/f'{args.mode.replace("-","_")}_failure.json'
            if not path.exists():inputs.replay.atomic_json(path,failure)
        print(json.dumps(failure,ensure_ascii=False),flush=True);return 1


if __name__=='__main__':raise SystemExit(main())
