"""Independent projection control; default read-only preflight."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from . import inputs


def workers(value):
    n=int(value)
    if not 1<=n<=4: raise argparse.ArgumentTypeError('cpu-workers must be 1..4, one independent problem per layer')
    return n


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',nargs='?',default='preflight',choices=('preflight','smoke','infer','evaluate'))
    project=Path(__file__).resolve().parents[3]
    p.add_argument('--parent-dir',default=str(project/'DINOv3/outputs/mad_equal'/inputs.PARENT_NAME))
    p.add_argument('--output-dir',default=str(project/'DINOv3/outputs/mad_equal/aebad_s_fastref_projection_target512_source_v1'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-workers',type=workers,default=4)
    p.add_argument('--no-write',action='store_true',help='preflight always writes nothing')
    return p


def cost(data,mode,state,started):
    from DINOv3.MADEqual.aebad_s_target_local_segment.runner import memory_observation
    root=Path(data['paths']['output'])
    value=dict(tool=inputs.TOOL,stage=mode,status=state['status'],seconds=state['seconds'],
               whole_stage_seconds=time.perf_counter()-started,timing=state.get('timing',{}),
               timing_scope='new stage; includes checks; not an algorithm speedup ratio',
               **memory_observation(cuda_evaluation_started='torch' in sys.modules),
               new_output_bytes=sum(f.stat().st_size for f in root.rglob('*') if f.is_file()),
               OT_called=False,Source_memory_rebuilt=False,MAD_fitted=False,prototypes_resampled=False,
               Q_or_A_persisted=False,cpu_workers=data['frozen']['cpu_workers'])
    inputs.replay.atomic_json(root/f'cost_{mode}.json',value)
    parent_costs={m:inputs.read_json(Path(data['paths']['parent'])/f'cost_{m}.json') for m in ('prepare','smoke','infer','evaluate')}
    new_costs={m:inputs.read_json(root/f'cost_{m}.json') for m in ('smoke','infer','evaluate') if (root/f'cost_{m}.json').is_file()}
    inputs.replay.atomic_json(root/'cost_comparison.json',dict(
        parent_source=data['paths']['parent'],parent_costs=parent_costs,new_costs=new_costs,
        note='Parent full_query_FastRef_CPU includes initial projection, OT, updates and diagnostics. It is not pure OT time. Avoiding historical repeat work is not an algorithm acceleration ratio.'))


def main(argv=None):
    args=parser().parse_args(argv);started=time.perf_counter();writable=False
    try:
        if args.no_write and args.mode!='preflight': raise ValueError('--no-write applies only to preflight')
        if args.mode in ('preflight','smoke'):
            check=inputs.preflight(args);print(json.dumps(check,ensure_ascii=False,indent=2),flush=True)
            if check['errors']: return 2
            if args.mode=='preflight': return 0
            p,prepared,parent_state=inputs.read_parent(args.parent_dir)
            data=inputs.freeze(args,p,prepared);root=Path(data['paths']['output']);root.mkdir(parents=True,exist_ok=True)
            inputs.replay.atomic_json(root/'inputs.json',data);writable=True
        else:
            data,p,prepared,parent_state=inputs.read_frozen(args.output_dir);root=Path(data['paths']['output'])
            if str(Path(args.parent_dir).resolve())!=data['paths']['parent'] or args.cpu_workers!=data['frozen']['cpu_workers']:
                raise ValueError('frozen parent/worker configuration changed')
            writable=True
        if args.mode=='evaluate':
            from .analysis import evaluate
            state=evaluate(data,p,prepared,parent_state,args.device)
        else:
            from .runtime import inference
            state=inference(data,p,prepared,parent_state,args.device,smoke=args.mode=='smoke')
        cost(data,args.mode,state,started);inputs.replay.atomic_json(root/'status.json',dict(state,stage=args.mode))
        print(f"PROJECTION_{args.mode.upper()}_STATUS={state['status']}\nOUTPUT={root}",flush=True)
        return 0 if state['status'] in ('smoke_verified','predictions_frozen','analysis_complete') else 3
    except Exception as error:
        failure=dict(tool=inputs.TOOL,stage=args.mode,status='stage_failed',error=f'{type(error).__name__}: {error}',seconds=time.perf_counter()-started)
        if writable:
            file=root/f'{args.mode}_failure.json'
            if not file.exists(): inputs.replay.atomic_json(file,failure)
        print(json.dumps(failure,ensure_ascii=False),flush=True);return 1


if __name__=='__main__': raise SystemExit(main())
