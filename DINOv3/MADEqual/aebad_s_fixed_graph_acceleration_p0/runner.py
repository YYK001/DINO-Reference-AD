"""Default read-only preflight; explicit continuation, no model rebuild stages."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import tarfile
from . import continuation as c, search


STAGES = ('preflight', 'prepare', 'smoke', 'predict', 'evaluate', 'evaluate-aupro', 'summarize', 'pack', 'all')
REVIEW_SUFFIXES = frozenset(('.json', '.csv', '.tsv', '.md', '.txt', '.log', '.exitcode', '.png', '.svg', '.pdf'))


def scratch_directory(name):
    name = name.casefold()
    return name in ('scratch', '_scratch', 'pixel_scratch') or name.endswith('_scratch') or name.startswith('scratch_')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=STAGES, default='preflight', nargs='?')
    p.add_argument('--parent-run', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cpu-workers', type=int, default=8)
    p.add_argument('--no-import-completed', action='store_true')
    p.add_argument('--smoke-query-count', type=int, default=128)
    p.add_argument('--limit', type=int)
    return p


def preflight(args):
    missing = []
    try:
        parent, files = c.snapshot(args.parent_run)
        files += [parent.data, parent.checkpoint, parent.dinov3_root/'hubconf.py', parent.official_root/'config.json']
        missing = [str(p) for p in files if not p.exists()]
    except (KeyError, OSError, ValueError) as exc:
        missing.append(str(exc))
    value = dict(engine=search.ENGINE, status='missing_inputs' if missing else 'inputs_present_not_runtime_verified',
                 missing_inputs=missing, torch_imported='torch' in sys.modules,
                 output_created=False, sampling=False, Source_rebuild=False, graph_rebuild=False,
                 capacity_admission_checks=False, mixed_precision='FP32 screening + FP64 direct winners; no FP16 backbone',
                 parent=str(args.parent_run), output=str(args.output_root), cache=str(args.cache_root))
    print(json.dumps(value, indent=2), flush=True)
    return 2 if missing else 0


def pack(args):
    archive = args.output_root.with_name(args.output_root.name+'_review.tar.gz')
    # A failed allocation/interrupt can leave mmap payloads behind, including
    # before evaluate's finally block starts. Never descend into scratch, even
    # for JSON/log files there. Explicit file types also exclude raw arrays
    # accidentally left elsewhere in the output. No payload hash/size scan.
    with tarfile.open(archive, 'w:gz', compresslevel=1, dereference=False) as tar:
        for current, dirs, names in os.walk(args.output_root, topdown=True, followlinks=False):
            dirs[:] = sorted(name for name in dirs if not scratch_directory(name)
                             and not (Path(current)/name).is_symlink())
            for name in sorted(names):
                path = Path(current)/name
                if path.suffix.casefold() in REVIEW_SUFFIXES and not path.is_symlink() and path.is_file():
                    tar.add(path, arcname=(Path(args.output_root.name)/path.relative_to(args.output_root)).as_posix(),
                            recursive=False)
    print('REVIEW_ARCHIVE='+str(archive), flush=True)
    return dict(archive=str(archive), bytes=archive.stat().st_size)


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('parent_run', 'output_root', 'cache_root'):
        setattr(args, name, getattr(args, name).resolve())
    if not 1 <= args.cpu_workers <= 8 or args.smoke_query_count < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('1..8 CPU workers and positive smoke/limit required')
    if not args.device.startswith('cuda:') or not args.device[5:].isdigit():
        raise ValueError('one explicit cuda:N required; set CUDA_VISIBLE_DEVICES before launch')
    if args.stage == 'preflight':
        return preflight(args)
    c.check_outputs(args.parent_run, args.output_root, args.cache_root, args.device, args.cpu_workers)
    if args.stage == 'pack':
        if not (args.output_root/'acceleration_inputs.json').is_file():
            raise ValueError('no initialized acceleration output')
        pack(args)
        return 0
    if args.stage == 'all':
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.autodl import run
        common = ['--parent-run', str(args.parent_run), '--output-root', str(args.output_root),
                  '--cache-root', str(args.cache_root), '--device', args.device,
                  '--cpu-workers', str(args.cpu_workers), '--smoke-query-count', str(args.smoke_query_count)]
        if args.no_import_completed:
            common.append('--no-import-completed')
        if args.limit is not None:
            raise ValueError('all cannot use a partial prediction limit')
        status = dict(status='running', success=False)
        try:
            for stage in STAGES[1:-2]:
                status['stage'] = stage
                c.st.write_json(args.output_root/'pipeline_status.json', status)
                run(stage, [sys.executable, '-B', '-u', '-m', __package__+'.runner', stage, *common], args.output_root)
            status.update(status='all_requested_metrics_complete', success=True)
            return 0
        except BaseException as exc:
            status.update(status='failed_outputs_retained', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            c.st.write_json(args.output_root/'pipeline_status.json', status)
            pack(args)
    started = time.perf_counter()
    a = c.initialize(args.parent_run, args.output_root, args.cache_root, args.device, args.cpu_workers,
                     import_completed=not args.no_import_completed)
    if args.stage == 'prepare':
        result = dict(status='frozen_parent_assets_linked', identity=c.ip.identity(a), rebuild=False)
    elif args.stage == 'smoke':
        path = a.run_root/'mixed_smoke.json'
        if path.exists():
            c.require_smoke(a)
            result = dict(status='smoke_reused')
        else:
            result = c.smoke(a, query_count=args.smoke_query_count)
    elif args.stage == 'predict':
        c.require_smoke(a)
        result = c.predict(a, limit=args.limit)
    else:
        from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import evaluation as ev, reporting
        c.old.verified(a)  # GT is inaccessible until the WHOLE population freezes.
        if args.stage == 'evaluate':
            result = [ev.evaluate(a, d, global_checked=True) for d in c.protocol.DOMAINS]
        elif args.stage == 'evaluate-aupro':
            result = [ev.aupro(a, d, b) for d in c.protocol.DOMAINS for b in c.protocol.BRANCHES]
        else:
            result = reporting.summarize(a)
            if not result['all_metrics_complete']:
                raise ValueError('summary remains pending; not a completed experiment')
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.runner import resources
    observed = resources()
    calls = result.get('query_forward_calls', 0) if isinstance(result, dict) else 0
    observed.update(model_forward_calls=calls, model_inference=calls > 0, feature_extraction=calls > 0)
    record = dict(stage=args.stage, seconds=time.perf_counter()-started, result=result, resources=observed,
                  engine=search.ENGINE, parent_costs_reused_not_remeasured=True)
    c.st.write_json(a.run_root/'acceleration_costs'/f'{args.stage}_{time.time_ns()}.json', record)
    print(json.dumps(record), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
