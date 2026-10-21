"""Original stages with capacity estimates disabled, identity checks unchanged."""
import sys
from . import inputs as ip
from .resource_policy import observation_only


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if argv and argv[0]=='diagnostic':
        from .runner import main as original
        argv=argv[1:]
    else:
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.runner import main as original
    from pathlib import Path
    target=None
    for flag in ('--output-dir','--cache-root'):
        if flag in argv:
            target=Path(argv[argv.index(flag)+1]);break
    existed=target is not None and target.exists()
    with observation_only() as audit:
        print('[resource policy] capacity estimates reported only; no RAM/VRAM/disk admission',flush=True)
        try:return original(argv)
        finally:
            # Preflight stays read-only. Never write into an old/rejected
            # output directory; record failures only in this invocation's output.
            if argv and argv[0]!='preflight' and target is not None and not existed and target.is_dir():
                ip.write_json(target/'capacity_policy.json',audit)


if __name__=='__main__':raise SystemExit(main())
