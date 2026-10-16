"""Default read-only preflight; computational stages must be explicit."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from . import inputs


def workers(value):
    count=int(value)
    if not 1<=count<=4: raise argparse.ArgumentTypeError('cpu-workers must be 1..4 (independent full-layer OT problems)')
    return count


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',nargs='?',default='preflight',choices=('preflight','prepare','smoke','infer','evaluate'))
    p.add_argument('--project-root',default=str(Path(__file__).resolve().parents[3]))
    for name in ('results-dir','capture-dir','support-cache-dir','mad','data','output-dir','memory','checkpoint','dinov3-root'): p.add_argument('--'+name)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--cpu-workers',type=workers,default=4)
    p.add_argument('--no-write',action='store_true',help='preflight always writes nothing; not valid for other modes')
    return p


def cost(data,mode,state,started):
    from DINOv3.MADEqual.aebad_s_target_local_segment.runner import memory_observation
    root=Path(data['paths']['output'])
    value=dict(tool=inputs.TOOL,stage=mode,status=state['status'],seconds=state['seconds'],
        whole_stage_seconds=time.perf_counter()-started,seconds_scope='stage computation function; whole_stage_seconds also includes admission/hash/loading overhead',timing=state.get('timing',{}),
        **memory_observation(cuda_evaluation_started='torch' in sys.modules),
        new_output_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        Source_memory_rebuilt=False,MAD_fitted=False,query_features_persisted=False,OT_matrices_persisted=False,
        refined_vectors_persisted=False,Source_candidate_scratch_bytes=0,
        cpu_workers=data['frozen']['config']['cpu_workers'],peak_scope='whole current stage, not per branch')
    inputs.replay.atomic_json(root/f'cost_{mode}.json',value)


def main(argv=None):
    args=parser().parse_args(argv); supplied,paths=inputs.paths_for(args); started=time.perf_counter(); writable=False
    try:
        if args.no_write and args.mode!='preflight': raise ValueError('--no-write applies only to preflight')
        if args.mode in ('preflight','prepare'):
            check=inputs.preflight(args)
            print(json.dumps({k:v for k,v in check.items() if k not in ('capture_inputs','capture_status','required_files','samples','supports','host_config')},ensure_ascii=False,indent=2),flush=True)
            if check['errors']: return 2
            if args.mode=='preflight': return 0
            data=inputs.freeze(check); root=Path(data['paths']['output']); root.mkdir(parents=True,exist_ok=True)
            inputs.replay.atomic_json(root/'inputs.json',data); writable=True
            from .runtime import prepare
            state=prepare(data,args.device)
        else:
            data,prepared=inputs.read_prepared(paths['output']); root=Path(data['paths']['output'])
            for key,path in paths.items():
                if key=='data': path=inputs.shared.data_root(path)
                if str(path)!=data['paths'][key]: raise ValueError('frozen input/output path changed: '+key)
            if args.cpu_workers!=data['frozen']['config']['cpu_workers']: raise ValueError('frozen cpu-workers changed')
            writable=True
            if args.mode=='evaluate':
                from .analysis import evaluate
                state=evaluate(data,prepared,args.device)
            else:
                from .runtime import inference
                state=inference(data,prepared,args.device,smoke=args.mode=='smoke')
        cost(data,args.mode,state,started); inputs.replay.atomic_json(root/'status.json',dict(state,stage=args.mode))
        print(f"FASTREF_{args.mode.upper()}_STATUS={state['status']}\nOUTPUT={root}",flush=True)
        return 0 if state['status'] in ('prototypes_prepared','smoke_verified','predictions_frozen','analysis_complete') else 3
    except Exception as error:
        failure=dict(tool=inputs.TOOL,stage=args.mode,status='stage_failed',error=f'{type(error).__name__}: {error}',seconds=time.perf_counter()-started)
        if hasattr(error,'diagnostics'): failure['numerical_diagnostics']=error.diagnostics
        if writable:
            file=Path(paths['output'])/f'{args.mode}_failure.json'
            if not file.exists(): inputs.replay.atomic_json(file,failure)
        print(json.dumps(failure,ensure_ascii=False),flush=True); return 1


if __name__=='__main__': raise SystemExit(main())
