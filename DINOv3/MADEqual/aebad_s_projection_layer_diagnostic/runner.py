"""Default read-only local preflight; explicit full-domain CPU analysis after review."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from . import inputs


def workers(value):
    n=int(value)
    if not 1<=n<=8:raise argparse.ArgumentTypeError('cpu-workers must be 1..8; full-pixel sorts remain sequential')
    return n


def memory(value):
    n=float(value)
    if not math.isfinite(n) or n<=0:raise argparse.ArgumentTypeError('positive finite memory budget required')
    return n


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',nargs='?',default='preflight',choices=('preflight','analyze'))
    p.add_argument('--project-root',default=str(Path(__file__).resolve().parents[3]))
    for name in ('parent-archive','projection-archive','mad','data-zip','output-dir'):p.add_argument('--'+name)
    p.add_argument('--cpu-workers',type=workers,default=4)
    p.add_argument('--memory-budget-gib',type=memory,default=8)
    p.add_argument('--no-write',action='store_true',help='preflight always writes nothing; invalid for analyze')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.no_write and args.mode!='preflight':raise ValueError('--no-write applies only to preflight')
        check,context=inputs.preflight(args)
        print(json.dumps(check,ensure_ascii=False,indent=2),flush=True)
        if check['errors']:return 2
        if args.mode=='preflight':return 0
        from .analysis import analyze
        state=analyze(context,args)
        print(json.dumps(state,ensure_ascii=False,indent=2),flush=True)
        return 0 if state['status']=='analysis_complete' else 3
    except Exception as error:
        print(json.dumps(dict(tool=inputs.TOOL,status='blocked',error=f'{type(error).__name__}: {error}'),ensure_ascii=False),flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
