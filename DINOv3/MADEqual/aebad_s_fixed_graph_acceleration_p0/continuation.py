"""Read-only frozen parent assets; independently identified acceleration output.

The old module is not edited or monkey-patched. Function bindings replace only
the graph-search dependency; original feature extraction, NN, scoring and
seven-branch construction execute unchanged.
"""
from pathlib import Path
from types import FunctionType, SimpleNamespace
import importlib.metadata
import os
import shutil
import numpy as np
from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import inputs as ip, prediction as old
from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import runtime, protocol
from . import search

st = ip.st
PATHS = ('run_root', 'cache_root', 'data', 'checkpoint', 'official_root', 'dinov3_root',
         'graph_root', 'query_cache_root', 'prediction_root', 'evaluation_scratch', 'reuse_fresh_run')
SMALL_FILES = ('fresh_catalog.json', 'fresh_catalog.seal.json', 'image_calibration_state.json',
               'support_manifest.csv', 'evaluation_manifest.csv', 'graph_summary.csv',
               'witness_search_summary.csv', 'witness_summary.csv', 'support_loo_scores.csv')


def parent_args(parent):
    parent = Path(parent).resolve()
    config = st.read_json(parent/'setup.json')['configuration']
    values = {k: Path(v) if k in PATHS and v is not None else v for k, v in config.items()}
    values.update(run_root=parent, domains=list(protocol.DOMAINS), layers=list(protocol.LAYERS),
                  heldouts=list(range(9)), limit=None, branch=None)
    return SimpleNamespace(**values)


def own_context(parent, output, cache, device, workers):
    a = parent_args(parent)
    a.run_root, a.cache_root = Path(output).resolve(), Path(cache).resolve()
    a.graph_root = a.cache_root/'graphs'
    a.query_cache_root = a.cache_root/'queries'
    a.prediction_root = a.cache_root/'predictions'
    a.evaluation_scratch = a.run_root/'pixel_scratch'
    a.reuse_fresh_run = None
    a.device, a.cpu_workers = device, workers
    # No query persistence is added by acceleration.
    a.query_mode = 'stream'
    return a


def engine_files():
    return {p.name: st.sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def link_asset(a, source, destination, parent_identity):
    """Link immutable payload, with a separate new provenance record."""
    source, destination = Path(source), Path(destination)
    if st.restore(destination, ip.identity(a)):
        return
    record = st.restore(source, parent_identity)
    if record is None:
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source.resolve(), destination)
    except OSError:
        # Cross-filesystem hard links are not possible; POSIX symlink keeps
        # the same immutable bytes without copying gigabytes to the SSD.
        destination.symlink_to(source.resolve())
    st.write_json(destination.with_suffix('.json'), dict(identity=ip.identity(a), sha256=record['sha256'],
                  metadata=dict(record['metadata'], reuse_source=dict(path=str(source.resolve()),
                  identity=parent_identity, original_record_sha256=st.sha(source.with_suffix('.json'))))))


def snapshot(parent):
    a = parent_args(parent)
    required = [a.run_root/name for name in ('setup.json', 'run_config.json',
                'fresh_catalog.json', 'fresh_catalog.seal.json', 'image_calibration_state.json', 'frozen_assets.json')]
    required += [a.run_root/'selection'/f'{d}.json' for d in protocol.DOMAINS]
    required += [a.cache_root/name for name in ('source_bank.npz', 'source_patch_mad.npz')]
    required += [a.cache_root/'supports'/d/f'{i:03d}.npz' for d in protocol.DOMAINS for i in range(1, 9)]
    required += [runtime.graph_path(a, d, l, 0) for d in protocol.DOMAINS for l in protocol.LAYERS]
    for p in list(required):
        if p.suffix == '.npz':
            required.append(p.with_suffix('.json'))
    return a, required


def validate_implementation(a):
    """Small code/version checks also apply on resumed stages."""
    config = st.read_json(a.run_root/'setup.json')['configuration']
    for relative, sha in config['code'].items():
        if st.sha(ip.ROOT/relative) != sha:
            raise ValueError('original implementation changed: '+relative)
    external = {}
    for name, root in (('dinov3', a.dinov3_root), ('SuperADD', a.official_root)):
        for p in sorted(root.rglob('*.py')):
            external[name+'/'+p.relative_to(root).as_posix()] = st.sha(p)
    if external != config['external_source_code'] or st.read_json(a.official_root/'config.json') != config['host_config']:
        raise ValueError('parent host source/configuration differs')
    for name, version in config['software'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('parent software differs: '+name)


def validate_parent(parent):
    a, files = snapshot(parent)
    if missing := [str(p) for p in files if not p.is_file()]:
        raise FileNotFoundError('missing frozen parent assets: '+str(missing))
    setup = st.read_json(a.run_root/'setup.json')
    if setup['identity'] != st.digest({k: v for k, v in setup.items() if k != 'identity'}):
        raise ValueError('parent setup identity differs')
    config = setup['configuration']
    if st.read_json(a.run_root/'run_config.json') != config:
        raise ValueError('parent configuration differs')
    if st.sha(a.run_root/'fresh_catalog.json') != setup['catalog_sha256'] or st.read_json(
            a.run_root/'fresh_catalog.seal.json')['sha256'] != setup['catalog_sha256']:
        raise ValueError('parent population identity differs')
    validate_implementation(a)
    assets = st.read_json(a.run_root/'frozen_assets.json')
    if assets['identity'] != setup['identity'] or assets['seal'] != st.digest({k: v for k, v in assets.items() if k != 'seal'}):
        raise ValueError('parent frozen asset record differs')
    for p, record in assets['assets'].items():
        if st.read_json(Path(p).with_suffix('.json')) != record:
            raise ValueError('parent asset history changed: '+p)
    if assets['image_calibration_state_sha256'] != st.sha(a.run_root/'image_calibration_state.json'):
        raise ValueError('parent image MAD differs')
    ip.patch_state(a)
    for d in protocol.DOMAINS:
        old.selection(a, d)
        for l in protocol.LAYERS:
            runtime.load_graph(a, d, l, 0)
    for p in files:
        if p.suffix == '.npz':
            st.restore(p, setup['identity'])
    return a, {str(p): st.sha(p) for p in files if p.suffix != '.npz'}


def check_outputs(parent, output, cache, device, workers):
    """Read-only path check, including before a launcher creates its log."""
    parent = Path(parent).resolve()
    a = own_context(parent, output, cache, device, workers)
    previous = parent_args(parent)
    protected = (parent, previous.cache_root, previous.graph_root, previous.prediction_root,
                 previous.data, previous.checkpoint, ip.ROOT)
    # Repo is a protected input, but the user's normal run directory may be a
    # sibling of it. No output inside a parent artifact or the source tree.
    for dest in (a.run_root, a.cache_root):
        for p in protected:
            if dest == p or p in dest.parents or dest in p.parents:
                raise ValueError('output overlaps parent/input/code: '+str(dest))
    if a.run_root == a.cache_root or a.run_root in a.cache_root.parents or a.cache_root in a.run_root.parents:
        raise ValueError('use separate output metadata and mechanical-disk cache roots')
    return a


def initialize(parent, output, cache, device, workers, *, import_completed=True):
    parent = Path(parent).resolve()
    a = check_outputs(parent, output, cache, device, workers)
    manifest = a.run_root/'acceleration_inputs.json'
    if manifest.exists():
        saved = st.read_json(manifest)
        if (saved['seal'] != st.digest({k: v for k, v in saved.items() if k != 'seal'}) or
                saved['engine'] != search.ENGINE or saved['code'] != engine_files() or
                saved['parent'] != str(parent) or saved['device'] != device or saved['cpu_workers'] != workers or
                saved['cache'] != str(a.cache_root) or saved['import_completed'] != import_completed or
                saved['physical_GPU'] != os.environ.get('CUDA_VISIBLE_DEVICES')):
            raise ValueError('acceleration configuration changed; preserve old output')
        for p, sha in saved['parent_records'].items():
            if st.sha(p) != sha:
                raise ValueError('frozen parent input changed: '+p)
        for p, sha in saved['local_records'].items():
            if st.sha(a.run_root/p) != sha:
                raise ValueError('frozen acceleration input changed: '+p)
        if ip.identity(a) != saved['identity']:
            raise ValueError('acceleration identity differs')
        validate_implementation(parent_args(parent))
        return a
    previous, hashes = validate_parent(parent)
    config = dict(engine=search.ENGINE, code=engine_files(), parent=str(parent), parent_records=hashes,
                  cache=str(a.cache_root), device=device, cpu_workers=workers,
                  physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'), import_completed=import_completed)
    identity = st.digest(config)
    plan = dict(config, identity=identity)
    plan_path = a.run_root/'prepare_plan.json'
    if plan_path.exists() and st.read_json(plan_path) != plan:
        raise ValueError('interrupted preparation belongs to another configuration')
    st.write_json(plan_path, plan)
    # The identity is independent; no old setup/hash/status is rewritten.
    st.write_json(a.run_root/'setup.json', dict(identity=identity, configuration=config, cache=str(a.cache_root)))
    for name in SMALL_FILES:
        src = parent/name
        if src.is_file():
            shutil.copyfile(src, a.run_root/name)
    old_id = ip.identity(previous)
    for name in ('source_bank.npz', 'source_patch_mad.npz'):
        link_asset(a, previous.cache_root/name, a.cache_root/name, old_id)
    for d in protocol.DOMAINS:
        choice = st.read_json(parent/'selection'/f'{d}.json')
        choice.update(identity=identity, acceleration_parent_selection_sha256=st.sha(parent/'selection'/f'{d}.json'))
        choice['seal'] = st.digest({k: v for k, v in choice.items() if k != 'seal'})
        st.write_json(a.run_root/'selection'/f'{d}.json', choice)
        for i in range(1, 9):
            link_asset(a, previous.cache_root/'supports'/d/f'{i:03d}.npz', a.cache_root/'supports'/d/f'{i:03d}.npz', old_id)
        for l in protocol.LAYERS:
            link_asset(a, runtime.graph_path(previous, d, l, 0), runtime.graph_path(a, d, l, 0), old_id)
    if import_completed:
        for sample in ip.catalog(previous)['samples']:
            src = previous.prediction_root/sample['domain']/(sample['sample_id']+'.npz')
            if not src.is_file() or not src.with_suffix('.json').is_file():
                continue  # A still-running parent may be writing its next image.
            _, meta, record = ip.verify_prediction(previous, sample)
            for l in protocol.LAYERS:
                if meta['graph_sha256'][str(l)] != st.read_json(runtime.graph_path(previous, sample['domain'], l, 0).with_suffix('.json'))['sha256']:
                    raise ValueError('completed parent prediction references different graph')
            dst = a.prediction_root/sample['domain']/(sample['sample_id']+'.npz')
            if st.restore(dst, identity):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src.resolve(), dst)
            except OSError:
                dst.symlink_to(src.resolve())
            meta = dict(meta, selection_sha256=st.sha(a.run_root/'selection'/f'{sample["domain"]}.json'),
                        acceleration=dict(engine=search.ENGINE, origin='reused_original_FP64_prediction',
                                          parent_file=str(src), parent_record_sha256=st.sha(src.with_suffix('.json'))))
            st.write_json(dst.with_suffix('.json'), dict(identity=identity, sha256=record['sha256'], metadata=meta))
    st.write_json(a.run_root/'protocol.json', dict(protocol.PROTOCOL, acceleration=search.ENGINE,
                  geometry='FP32 screening intervals; FP64 direct global winners', parent=str(parent)))
    frozen = [a.run_root/name for name in SMALL_FILES if (a.run_root/name).is_file()]
    frozen += [a.run_root/'setup.json', a.run_root/'protocol.json']
    frozen += list((a.run_root/'selection').glob('*.json'))
    value = dict(config, identity=identity,
                 local_records={p.relative_to(a.run_root).as_posix(): st.sha(p) for p in frozen})
    value['seal'] = st.digest(value)
    st.write_json(manifest, value)
    return a


def bind(function, **dependencies):
    return FunctionType(function.__code__, dict(function.__globals__, **dependencies),
                        function.__name__, function.__defaults__, function.__closure__)


class PredictionStorage:
    def __getattr__(self, name):
        return getattr(st, name)

    @staticmethod
    def seal(path, identity, arrays, metadata, **kwargs):
        graph_seconds = sum(r.get('graph_numerics', {}).get('seconds', 0.) for r in metadata.get('audits', []))
        print(f'[mixed graph image] {metadata.get("domain")}/{metadata.get("sample_id")} '
              f'four_layer_graph_seconds={graph_seconds:.3f}', flush=True)
        return st.seal(path, identity, arrays, dict(metadata, acceleration=dict(
            engine=search.ENGINE, origin='new_mixed_precision_prediction')), **kwargs)


def predict(a, *, limit=None):
    a.limit = limit
    layer = bind(old.layer_branches, scan_segments=search.scan_segments)
    freeze = bind(old.freeze, PreparedSearch=search.PreparedSearch,
                  layer_branches=layer, st=PredictionStorage())
    return freeze(a)


def smoke(a, *, query_count=128):
    """Same original-host Q for old FP64 and accelerated graph search; no GT."""
    runtime.precision(a)
    model, _ = runtime.host(a)
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.numerics import scan_segments as reference
    checks = []
    import time
    for d in protocol.DOMAINS:
        sample = next(s for s in ip.catalog(a)['samples'] if s['domain'] == d)
        with ip.Data(Path(ip.catalog(a)['data_root'])) as data:
            features, geometry = old.extraction(model, data.image(sample), a.device)
        choice = old.selection(a, d)
        for l in protocol.LAYERS:
            banks = ip.load_banks(a, d, l)
            nodes, *_ = runtime.graph.layout(banks)
            pool, limits, _ = runtime.load_graph(a, d, l, 0)
            main, plain = choice['main']['selected'], choice['plain']['selected']
            configs = [tuple(choice['paired_config']) if main else None,
                       tuple(main['config']) if main else None, tuple(plain['config']) if plain else None]
            masks = np.asarray([runtime.graph.masks(pool, limits, [c])[0] if c else
                               np.zeros(len(pool['edges']), bool) for c in configs])
            q = features[f'layer_{l}'][:query_count]
            # A permissive anchor exposes the graph minimum, rather than
            # hiding graph discrepancies behind the original Source minimum.
            union = np.full(len(q), np.finfo(np.float32).max**.25, np.float32)
            started = time.perf_counter()
            exact = reference(q, nodes, pool['edges'], masks, union, device=a.device,
                              query_chunk=a.query_chunk, edge_chunk=a.edge_chunk, reference_chunk=a.reference_chunk)
            reference_seconds = time.perf_counter()-started
            started = time.perf_counter()
            mixed = search.scan_segments(q, nodes, pool['edges'], masks, union, device=a.device,
                         query_chunk=a.query_chunk, edge_chunk=a.edge_chunk, reference_chunk=a.reference_chunk)
            mixed_seconds = time.perf_counter()-started
            # Keep the existing distance tolerance; canonical winner IDs are
            # also checked because they determine representative provenance.
            verified = np.allclose(exact[0], mixed[0], atol=1e-7, rtol=1e-6) and np.array_equal(exact[1], mixed[1])
            row = dict(domain=d, layer=l, sample_id=sample['sample_id'], queries=len(q),
                       verified=verified, max_abs_error=float(np.abs(exact[0]-mixed[0]).max(initial=0)),
                       atol=1e-7, rtol=1e-6,
                       winner_mismatches=int((exact[1] != mixed[1]).sum()),
                       reference_seconds=reference_seconds, mixed_seconds=mixed_seconds,
                       scope='first query rows of one image/domain; preparation included; not full-image time/score validation',
                       diagnostics=mixed[2])
            checks.append(row)
            print('[mixed graph smoke] '+__import__('json').dumps(row), flush=True)
    value = dict(identity=ip.identity(a), engine=search.ENGINE, query_count=query_count, checks=checks,
                 verified=all(r['verified'] for r in checks), GT_used=False,
                 query_forward_calls=len(protocol.DOMAINS))
    value['seal'] = st.digest(value)
    st.write_json(a.run_root/'mixed_smoke.json', value)
    if not value['verified']:
        raise ValueError('mixed graph/reference disagreement; smoke evidence preserved')
    return value


def require_smoke(a):
    value = st.read_json(a.run_root/'mixed_smoke.json')
    if (value['seal'] != st.digest({k: v for k, v in value.items() if k != 'seal'}) or
            value['identity'] != ip.identity(a) or value['engine'] != search.ENGINE or not value['verified'] or
            {(r['domain'], r['layer']) for r in value['checks']} != set(__import__('itertools').product(protocol.DOMAINS, protocol.LAYERS))):
        raise ValueError('matching completed mixed/reference smoke required')
