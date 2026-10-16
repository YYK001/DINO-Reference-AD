"""Default filesystem preflight; prepare/infer/evaluate require explicit modes."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from .inputs import TOOL,replay,paths_for,guard_output,preflight,freeze_check,read_prepared


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('preflight','prepare','infer','evaluate'),nargs='?',default='preflight')
    parser.add_argument('--project-root',default=str(Path(__file__).resolve().parents[3]))
    for key in ('results-dir','capture-dir','support-cache-dir','official-root','checkpoint','output-dir'):
        parser.add_argument('--'+key)
    parser.add_argument('--dinov3-root',default='/root/autodl-tmp/dinov3')
    parser.add_argument('--data',default='/root/autodl-tmp/datasets/AeBAD/AeBAD_S')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--candidate-storage',choices=('mixed','disk'),default='mixed',
                        help='preflight/prepare: mixed uses L7/L15 RAM and L23/L31 disk; disk uses all four on disk')
    parser.add_argument('--no-write',action='store_true',help='preflight only; no files/directories created')
    return parser


def stage_cost(out,mode,state):
    from DINOv3.MADEqual.aebad_s_target_local_segment.runner import memory_observation
    versions={}
    for name in ('numpy','torch','torchvision','Pillow','adeval'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    files=[p for p in out.rglob('*') if p.is_file()]
    observation=memory_observation(cuda_evaluation_started='torch' in sys.modules)
    info=dict(tool=TOOL,stage=mode,status=state['status'],seconds=state.get('seconds'),
        timing=state.get('timing',{}),**observation,software=versions,
        cost_scope='one_time_adaptation' if mode=='prepare' else ('query_backbone_correction_retrieval' if mode=='infer' else 'offline_cached_evaluation'),
        memory_peak_scope='whole stage process, not an independently measured branch',
        on_disk_bytes=sum(p.stat().st_size for p in files),
        projection_bytes=sum(p.stat().st_size for p in files if 'projection' in p.relative_to(out).parts),
        raw_cache_bytes=sum(p.stat().st_size for p in files if any(x in p.relative_to(out).parts for x in ('sparc_raw','source_calibration'))),
        memory_bytes=(out/'sparc_source_memory.pt').stat().st_size if (out/'sparc_source_memory.pt').is_file() else None)
    if 'torch' in sys.modules:
        torch=sys.modules['torch']
        info['precision']=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    if mode=='prepare' and (out/'candidate_sources.json').is_file():
        candidates=json.loads((out/'candidate_sources.json').read_text(encoding='utf-8'))
        info['candidate_scratch_peak_bytes']=candidates['builder_metadata']['candidate_scratch_peak_bytes']
    replay.atomic_json(out/f'cost_{mode}.json',info)
    replay.write_csv(out/f'timing_{mode}.csv',[dict(stage=k,seconds=v,cost_scope=info['cost_scope']) for k,v in info['timing'].items()])


def main(argv=None):
    args=build_parser().parse_args(argv);paths=paths_for(args)
    if args.no_write and args.mode!='preflight': raise ValueError('--no-write applies only to preflight')
    started=time.perf_counter();may_write=False
    try:
        if args.mode in ('preflight','prepare'):
            check=preflight(args)
            print(json.dumps(dict(tool=TOOL,status=check['status'],errors=check['errors'],
                missing_assets=check['missing_assets'],source_images=len(check['source_rows']),
                evaluation_images=len(check['samples']),supports=len(check['supports']),
                existing_support_features=sum(r['exists'] for r in check['support_features']),
                resource_estimate=check['resource_estimate'],paths=check['paths'],
                torch_imported='torch' in sys.modules),indent=2,ensure_ascii=False),flush=True)
            if check['errors']: return 2
            if args.mode=='preflight':
                if not args.no_write:
                    paths['output'].mkdir(parents=True,exist_ok=True)
                    replay.atomic_json(paths['output']/'preflight.json',check)
                return 0
            paths={k:Path(v) for k,v in check['paths'].items()}
            inputs=freeze_check(check);out=paths['output'];out.mkdir(parents=True,exist_ok=True)
            replay.atomic_json(out/'inputs.json',inputs);may_write=True
            from .runtime import prepare
            state=prepare(inputs,args.device)
        else:
            guard_output(paths,allow_prepared=True)
            inputs,prepared=read_prepared(paths['output'])
            # Keep the initial data/model/capture identities; a later mode cannot
            # silently redirect any frozen path via different CLI arguments.
            for key in paths:
                if str(paths[key])!=inputs['paths'][key]: raise ValueError(f'frozen path changed: {key}')
            out=paths['output'];may_write=True
            if args.mode=='infer':
                from .runtime import infer
                state=infer(inputs,prepared,args.device)
            else:
                from .analysis import evaluate
                state=evaluate(inputs,prepared)
        stage_cost(out,args.mode,state)
        replay.atomic_json(out/'status.json',dict(state,stage=args.mode))
        print(f"SPARC_{args.mode.upper()}_STATUS={state['status']}\nOUTPUT={out}",flush=True)
        return 0 if state['status'] in ('prepare_complete','predictions_frozen','analysis_complete') else 3
    except Exception as e:
        error=dict(tool=TOOL,stage=args.mode,status='stage_failed',error=f'{type(e).__name__}: {e}',seconds=time.perf_counter()-started)
        if may_write:
            # Never reset a prior complete stage/status when refusing a rerun.
            path=paths['output']/f'{args.mode}_failure.json'
            if not path.exists(): replay.atomic_json(path,error)
        print(json.dumps(error,ensure_ascii=False),flush=True)
        return 1


if __name__=='__main__': raise SystemExit(main())
