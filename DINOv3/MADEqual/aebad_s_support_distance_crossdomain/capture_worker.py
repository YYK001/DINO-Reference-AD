"""Run the unchanged capture with working-set RAM accounting and safe pause.

No model or math replacement. A requested pause occurs only after both the
cache manifest and its complete per-image verification CSV have been written.
"""
import os
from pathlib import Path
import sys
from unittest.mock import patch
from . import inputs as ip,resources
from .resource_policy import observation_only

PAUSED=75


class CapturePaused(BaseException):
    pass


def run(argv=None,invoke=None):
    if invoke is None:
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.runner import main
        invoke=main
    stop=os.environ.get('DINO_CAPTURE_PAUSE_FILE')
    original=ip.old.write_csv
    def checkpoint(path,rows,*args,**kwargs):
        original(path,rows,*args,**kwargs)
        if not stop or not Path(stop).is_file() or Path(path).name!='capture_verification.csv':return
        root=Path(path).parent;state=ip.old.read_json(root/'cache_status.json')
        checks=ip.old.csv_file(path)
        if any(str(r.get('verified')).lower() not in ('true','1') for r in checks):return
        verified={(r.get('domain'),r.get('sample_id'),r.get('check_kind')) for r in checks}
        if any((r['domain'],r['sample_id'],kind) not in verified
               for r in state['records'] for kind in ('SOURCE_CURRENT','UNION_CURRENT')):return
        if state['status']!='partial_cache_verified':return
        # A pause can land on the last image. A subsequent cache-only resume
        # will not construct a host or create capture_environment.json, so keep
        # this real worker's environment now rather than inventing it at merge.
        torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            ip.write_json(root/'capture_environment.json',dict(torch=torch.__version__,CUDA=torch.version.cuda,
                device=torch.cuda.get_device_name(),precision=ip.old.PRECISION,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                environment_scope='actual_worker_at_verified_RAM_pause; cache-only resume may follow'))
        # BaseException deliberately bypasses the original Exception failure
        # handler. No numerical error is erased or converted to a pause.
        ip.write_json(root/'worker_pause.json',dict(status='paused_after_verified_commit',
            identity=state['identity'],records=len(state['records']),reason='RAM scheduling only',
            memory=resources.memory_snapshot()))
        raise CapturePaused()
    try:
        with observation_only(),patch.object(ip.old,'write_csv',checkpoint):
            return invoke(sys.argv[1:] if argv is None else argv)
    except CapturePaused:
        print('[capture worker] paused after verified commit; resume will check original hashes',flush=True)
        return PAUSED


if __name__=='__main__':raise SystemExit(run())
