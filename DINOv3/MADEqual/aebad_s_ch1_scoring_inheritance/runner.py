"""Default read-only preflight. Real observations require recover-calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inputs import preflight


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', nargs='?', default='preflight', choices=('preflight', 'recover-calibration', 'evaluate'))
    p.add_argument('--project-root', default=str(Path(__file__).resolve().parents[3]))
    p.add_argument('--results-dir', help='original formal directory, mapped explicitly on each machine')
    parent = p.add_mutually_exclusive_group()
    parent.add_argument('--parent-archive', help='approved local FastRef v3 results tar.gz')
    parent.add_argument('--parent-dir', help='read-only server parent experiment (recovery preflight only)')
    p.add_argument('--data', help='local AeBAD.zip or server AeBAD_S root')
    p.add_argument('--memory')
    p.add_argument('--checkpoint')
    p.add_argument('--official-root')
    p.add_argument('--dinov3-root')
    p.add_argument('--source-calibration-dir', help='new recovered result root (not its raw/ subdirectory)')
    p.add_argument('--output-dir')
    p.add_argument('--device', default='cuda:0', help='only used by explicit recovery')
    p.add_argument('--cpu-workers', type=int, default=4, help='independent CPU test-image reconstruction workers')
    p.add_argument('--for-recovery', action='store_true', help='preflight server recovery assets instead of CPU calibration input')
    p.add_argument('--no-write', action='store_true', help='explicitly read-only; only valid for preflight')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.cpu_workers <= 32:
        raise ValueError('--cpu-workers must be in [1,32]')
    if args.stage != 'preflight' and args.no_write:
        raise ValueError('--no-write is only valid for preflight')
    if args.for_recovery and args.stage == 'evaluate':
        raise ValueError('--for-recovery cannot bypass CPU evaluation admission checks')
    report, ctx = preflight(args)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if report['status'] != 'filesystem_preflight_passed':
        return 2
    if args.stage == 'preflight':
        return 0
    try:
        if args.stage == 'recover-calibration':
            from .recovery import recover
            state = recover(ctx, args.device)
            expected = 'calibration_recovered'
        else:
            from .evaluation import evaluate
            state = evaluate(ctx)
            expected = 'evaluation_complete'
        print(json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if state['status'] == expected else 2
    except Exception as error:
        # A resume/hash rejection must not overwrite the trusted prior ledger.
        print(json.dumps(dict(status='stage_failed', stage=args.stage,
                              error=f'{type(error).__name__}: {error}'), ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
