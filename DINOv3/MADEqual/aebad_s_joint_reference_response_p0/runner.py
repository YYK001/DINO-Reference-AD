"""Default inventory only. GPU/model work is explicit and never runs on import."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
from . import inputs as ip
from .inputs import st
from .protocol import *


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', nargs='?', default='preflight', choices=('preflight', 'capture-normals', 'capture', 'prepare', 'smoke', 'predict', 'evaluate', 'aupro', 'cases', 'summarize', 'pack', 'all', 'all-cpu'))
    p.add_argument('--parent-run', type=Path, default=ip.ROOT/'server_results/aebad_target_loo_two_K8_20261030_013433')
    p.add_argument('--sacp-run', type=Path, default=ip.ROOT/'server_results/aebad_s_sacp_seed42_20261028_011454')
    p.add_argument('--run-root', type=Path, default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_joint_reference_response_v1')
    p.add_argument('--cache-root', type=Path)
    for name in ('memory42-root', 'memory43-root', 'source-bank', 'support-feature-root', 'feature-manifest', 'raw-cache'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--data', type=Path, default=ip.ROOT/'datasets/AeBAD.zip')
    p.add_argument('--evaluation-scratch', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--official-root', type=Path, default=ip.ROOT/'external_baselines/SuperADD')
    p.add_argument('--dinov3-root', type=Path, default=ip.ROOT/'external_baselines/dinov3')
    p.add_argument('--device', default='cpu')
    p.add_argument('--cpu-workers', type=int, choices=range(1, 9), default=8)
    p.add_argument('--query-chunk', type=int, default=1024)
    p.add_argument('--reference-chunk', type=int, default=512)
    p.add_argument('--domains', nargs='+', choices=DOMAINS, default=list(DOMAINS))
    p.add_argument('--support-seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    for name in ('supplement-retrieval', 'extract-missing', 'benchmark-retrieval', 'benchmark-forward', 'benchmark-aupro', 'allow-over-12h'):
        p.add_argument('--'+name, action='store_true')
    return p


def overlap(a, b):return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def normalize(a):
    for k, v in vars(a).copy().items():
        if isinstance(v, Path):setattr(a, k, v.resolve())
    a.cache_root = a.cache_root or a.run_root.with_name(a.run_root.name+'_cache')
    a.evaluation_scratch = a.evaluation_scratch or a.run_root.with_name(a.run_root.name+'_pixels')
    protected = [v for k in ('parent_run', 'sacp_run', 'memory42_root', 'memory43_root', 'source_bank', 'support_feature_root', 'feature_manifest', 'raw_cache', 'data', 'checkpoint') if (v := getattr(a, k)) is not None]
    writable = (a.run_root, a.cache_root, a.evaluation_scratch)
    if any(overlap(x, y) for i, x in enumerate(writable) for y in writable[i+1:]) or any(overlap(x, y) for x in writable for y in protected):
        raise ValueError('new output/cache/scratch must be disjoint and separate from all parent assets')
    if a.query_chunk < 1 or a.reference_chunk < 1:raise ValueError('positive block sizes required')
    if len(set(a.domains)) != len(a.domains) or len(set(a.support_seeds)) != len(a.support_seeds):raise ValueError('duplicate settings')
    a._started = time.perf_counter()
    return a


def pack(a):
    dest = a.run_root.with_name(a.run_root.name+'_review.tar.gz')
    with tarfile.open(dest, 'w:gz', compresslevel=1) as archive:
        for root, dirs, files in os.walk(a.run_root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in ('smoke', 'scratch', 'pixel_scratch') and not (Path(root)/d).is_symlink()]
            for name in files:
                path = Path(root)/name
                if path.is_symlink() or path.suffix not in ('.json', '.csv', '.md', '.txt', '.log', '.png'):continue
                archive.add(path, arcname=a.run_root.name+'/'+path.relative_to(a.run_root).as_posix(), recursive=False)
    return dict(archive=str(dest), bytes=dest.stat().st_size, native_maps_distances_features_and_scratch_excluded=True)


def execute(a):
    if a.stage == 'pack':return pack(a)
    if a.stage == 'preflight':return ip.preflight(a)
    ip.freeze(a)
    if a.stage in ('capture', 'predict'):
        from .budget import launch_check
        launch_check(a)
    if a.stage in ('capture', 'capture-normals', 'prepare'):
        from . import runtime
        return runtime.build_references(a) if a.stage == 'prepare' else runtime.capture(a, only_normals=a.stage == 'capture-normals')
    if a.stage == 'smoke':
        from .smoke import run
        return run(a)
    if a.stage == 'predict':
        from .prediction import predict
        return predict(a)
    if a.stage in ('evaluate', 'aupro', 'cases'):
        from . import evaluation
        return getattr(evaluation, a.stage)(a)
    from .reporting import summarize
    return summarize(a)


def cost(a, tick, status, before):
    import psutil
    from importlib.metadata import version, PackageNotFoundError
    rss = psutil.Process().memory_info();peak = getattr(rss, 'peak_wset', rss.rss)
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
    except ImportError:pass
    r = dict(stage=a.stage, status=status, seconds=time.perf_counter()-tick, process_peak_rss_bytes=peak,
             feature_forwards=getattr(a, 'feature_forwards', 0), Exact_execution=getattr(a, 'exact_execution', {}), training=0, source_rebuilds=0)
    r['new_disk_bytes'] = sum(x.stat().st_size for root in (a.run_root, a.cache_root) for x in root.rglob('*') if x.is_file() and not x.is_symlink())-before
    r['software'] = {}
    for name in ('numpy', 'scipy', 'Pillow', 'torch', 'adeval'):
        try:r['software'][name] = version(name)
        except PackageNotFoundError:r['software'][name] = None
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        original_runtime = sys.modules.get('DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime')
        prior = getattr(original_runtime, 'PRIOR_GPU_PEAK', {})
        r.update(peak_cuda_allocated_bytes=max(torch.cuda.max_memory_allocated(), getattr(a, 'peak_cuda_allocated_bytes', 0),prior.get('allocated',0)),
                 peak_cuda_reserved_bytes=max(torch.cuda.max_memory_reserved(), getattr(a, 'peak_cuda_reserved_bytes', 0),prior.get('reserved',0)))
    st.write_json(a.run_root/'costs'/f'{a.stage}_{time.time_ns()}.json', r)
    st.write_csv(a.run_root/'costs.csv', [st.read_json(p) for p in sorted((a.run_root/'costs').glob('*.json'))])


def orchestrate(a, args):
    from .budget import launch_check
    ip.freeze(a);launch_check(a)
    stages = (['capture'] if a.supplement_retrieval else [])+['preflight', 'prepare', 'predict', 'evaluate']+(['aupro'] if a.stage == 'all' else [])+['cases', 'summarize']
    try:
        for stage in stages:
            log = a.run_root/'logs'/(stage+'.log');log.parent.mkdir(parents=True, exist_ok=True)
            st.write_json(a.run_root/'pipeline_status.json', dict(status='running', stage=stage))
            print(f'STAGE={stage} LOG={log}', flush=True)
            with log.open('a', encoding='utf-8') as out, subprocess.Popen([sys.executable, '-B', '-u', '-m', __package__+'.runner', stage, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
                for line in child.stdout:print(line, end='', flush=True);out.write(line);out.flush()
                rc = child.wait()
            if rc and not (stage == 'summarize' and a.stage == 'all-cpu' and rc == 2):
                st.write_json(a.run_root/'pipeline_status.json', dict(status='failed_or_pending', stage=stage, exit_code=rc));return rc
        st.write_json(a.run_root/'pipeline_status.json', dict(status='complete' if a.stage == 'all' else 'CPU_complete_AUPRO_pending', stage='summarize'))
        return 0
    finally:
        print(json.dumps(pack(a)), flush=True)


def main(argv=None):
    tokens = list(sys.argv[1:] if argv is None else argv);a = normalize(parser().parse_args(tokens))
    a.run_root.mkdir(parents=True, exist_ok=True)
    if a.stage in ('all', 'all-cpu'):return orchestrate(a, tokens[1:])
    stop = threading.Event();tick = time.perf_counter();status = 'failed'
    before = sum(x.stat().st_size for root in (a.run_root, a.cache_root) for x in root.rglob('*') if x.is_file() and not x.is_symlink())
    def heartbeat():
        while not stop.wait(30):print(f'[heartbeat] joint stage={a.stage} seconds={time.perf_counter()-tick:.0f}', flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True);thread.start()
    try:
        result = execute(a);print(json.dumps(result, ensure_ascii=False, indent=2), flush=True);status = result.get('status', 'complete')
        return 2 if status in ('missing_inputs', 'blocked', 'pending', 'pending_unmeasured_costs', 'partial_normal_model') else 0
    finally:
        stop.set();thread.join()
        if a.stage != 'pack':cost(a, tick, status, before)


if __name__ == '__main__':raise SystemExit(main())
