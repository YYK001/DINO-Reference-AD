"""Durable recovery ledger and explicit, lazy-loaded original host observation."""
from __future__ import annotations

import io
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .inputs import TOOL, SourceData, digest, sha, replay, load_context, recovery_assets
from .scoring import LAYERS, DISTANCE_ATOL, DISTANCE_RTOL, statistics, check


def calibration_rows(ctx):
    return [r for r in ctx['source'] if r['role'] == 'calibration']


def read_cache(path, row, identity):
    with np.load(path, allow_pickle=False) as f:
        raw = f['raw_distances'].copy()
        grid = f['grid'].tolist()
        metadata = json.loads(str(f['metadata'].item()))
        if (str(f['protocol_id'].item()) != replay.PROTOCOL_ID
                or str(f['relative_path'].item()) != row['relative_path']
                or f['layers'].tolist() != list(LAYERS)
                or f['output_shape'].tolist() != row['output_shape']
                or metadata != dict(input_identity=identity, image=row, distance_divisor=1280,
                                    raw_already_divided=True, physical_row_major=True)):
            raise ValueError('calibration cache input/geometry identity mismatch: ' + str(path))
    if (len(grid) != 2 or min(grid) < 3 or raw.dtype != np.float32
            or raw.shape != (4, *grid) or not np.isfinite(raw).all() or np.any(raw < 0)):
        raise ValueError('invalid finite float32 [4,H,W] calibration raw: ' + str(path))
    return raw


def validate_recovery(root, ctx, *, complete):
    """Compare prior hashes BEFORE reading/reapproving numerical contents.

    Does not mutate status on failure. Orphan files are deliberately not adopted:
    without a prior ledger entry their provenance cannot be established on resume.
    Absolute paths in inputs.json are documentary, never followed on the CPU.
    """
    root = Path(root)
    state = json.loads((root / 'recovery_status.json').read_text(encoding='utf-8'))
    snapshot = json.loads((root / 'inputs.json').read_text(encoding='utf-8'))
    if (state['tool'] != TOOL or state['identity'] != ctx['identity']
            or snapshot['identity'] != ctx['identity'] or digest(snapshot['frozen']) != ctx['identity']
            or snapshot['frozen'] != ctx['frozen']
            or sha(root / 'inputs.json') != state['inputs_sha256']
            or sha(root / 'source_ordered_manifest.csv') != state['manifest_sha256']):
        raise ValueError('recovery inputs/code identity or snapshot hash mismatch')
    expected = {f"raw/{r['calibration_index']:03d}.npz": r for r in calibration_rows(ctx)}
    records = state['records']
    names = [r['file'] for r in records]
    actual = {p.relative_to(root).as_posix() for p in (root / 'raw').glob('*.npz')}
    if (len(names) != len(set(names)) or not set(names).issubset(expected)
            or set(names) != actual or (complete and set(names) != set(expected))):
        raise ValueError('missing, duplicate, orphan or out-of-scope recovery cache; preserving existing ledger')
    for record in records:
        name = record['file']
        if sha(root / name) != record['cache_sha256']:
            raise ValueError('recovery cache changed since previous record: ' + name)
        raw = read_cache(root / name, expected[name], ctx['identity'])
        if record['relative_path'] != expected[name]['relative_path']:
            raise ValueError('recovery record image mismatch')
        reference = np.asarray([record['peak'], record['local']], np.float32)
        if reference.shape != (2, 4) or not np.isfinite(reference).all():
            raise ValueError('original scoring reference missing/nonfinite')
        del raw
    if complete:
        if state['status'] != 'calibration_recovered' or state['images'] != 66 or state['failed_checks'] != 0:
            raise ValueError('Source calibration is computed but not verified complete')
        for name in ('source_ordered_manifest.csv', 'calibration_reference.csv', 'patch_mad_validation.csv', 'numerical_verification.csv'):
            if sha(root / name) != state['output_sha256'][name]:
                raise ValueError('recovery reference/report hash changed: ' + name)
    return state


def verify_patch_mad(raws, ctx, checks):
    old = replay.read_csv(ctx['paths']['formal'] / 'calibration_stats.csv')
    if [int(r['layer']) for r in old] != list(LAYERS):
        raise ValueError('calibration_stats layer order mismatch')
    result = []
    for offset, layer in enumerate(LAYERS):
        values = np.concatenate([a[offset].reshape(-1) for a in raws])
        # Original fit_calibrators uses a Python float multiplication followed by
        # float32 storage. This differs from image MAD's array multiplication.
        median = float(np.median(values))
        scale = float(np.float32(max(1.4826 * float(np.median(np.abs(values - median))), 1e-8)))
        for name, actual, expected in (
            ('median_vs_state', median, ctx['medians'][offset]),
            ('scale_vs_state', scale, ctx['scales'][offset]),
            ('median_vs_csv', median, float(old[offset]['median'])),
            ('scale_vs_csv', scale, float(old[offset]['mad_scale_1p4826'])),
        ):
            check(checks, 'source_calibration', name, actual, expected,
                  atol=DISTANCE_ATOL, rtol=DISTANCE_RTOL, layer=layer)
        check(checks, 'source_calibration', 'pooled_patch_count', values.size,
              int(old[offset]['patch_count']), atol=0, layer=layer)
        result.append(dict(layer=layer, patch_count=values.size, median=median, scale=scale,
                           frozen_median=float(ctx['medians'][offset]), frozen_scale=float(ctx['scales'][offset]),
                           median_abs_error=abs(median - float(ctx['medians'][offset])),
                           scale_abs_error=abs(scale - float(ctx['scales'][offset])),
                           atol=DISTANCE_ATOL, rtol=DISTANCE_RTOL,
                           old_per_image_raw_available=False, original_patch_state_modified=False))
    return result


def recover(ctx, device_name):
    start = time.perf_counter()
    p = ctx['paths']
    output = p['output']
    if output is None:
        raise ValueError('recover-calibration requires an independent --output-dir')
    errors = recovery_assets(ctx)
    if errors:
        raise ValueError('; '.join(errors))
    if output.exists():
        # Do this BEFORE resetting status or loading Torch/model.
        state = validate_recovery(output, ctx, complete=False)
        if state['status'] == 'calibration_recovered':
            validate_recovery(output, ctx, complete=True)
            return state
    else:
        output.mkdir(parents=True)
        replay.atomic_json(output / 'inputs.json', dict(tool=TOOL, identity=ctx['identity'],
                           frozen=ctx['frozen'], paths={k: str(v) if v else None for k, v in p.items()}))
        replay.write_csv(output / 'source_ordered_manifest.csv', ctx['source'])
        state = dict(tool=TOOL, stage='recover-calibration', identity=ctx['identity'],
                     inputs_sha256=sha(output / 'inputs.json'),
                     manifest_sha256=sha(output / 'source_ordered_manifest.csv'), status='recovering', records=[])
        replay.atomic_json(output / 'recovery_status.json', state)
    # Explicit recovery is the ONLY import boundary for real model dependencies.
    import torch
    from DINOv3.MADEqual.aebad_s_sparc_p0.runtime import load_host, save_npz
    from DINOv3.MADEqual.aebad_s_target_fewshot.runner import _load_memory
    from DINOv3.relation_reliability.superadd_host import observe_pil
    from DINOv3.MADEqual.image_granularity_calibration.scoring import interpolated_layer_maxima
    from DINOv3.MADEqual.connected_region.scoring import local_mean
    from DINOv3.relation_oracle_cache.extract import mad_calibrated

    model, device = load_host(dict(paths={k: str(v) for k, v in p.items() if v is not None}, frozen=ctx['frozen']), device_name)
    if not _load_memory(p['memory'], model, source_count=521, device=device):
        raise ValueError('frozen memory missing; rebuilding is forbidden')
    data = SourceData(p['data'])
    state.update(status='recovering', environment=dict(python=platform.python_version(), numpy=np.__version__,
                 torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(device)),
                 images=len(state['records']))
    replay.atomic_json(output / 'recovery_status.json', state)
    done = {r['file'] for r in state['records']}
    try:
        for row in calibration_rows(ctx):
            name = f"raw/{row['calibration_index']:03d}.npz"
            if name in done:
                continue
            blob = data.read(row['relative_path'])
            import hashlib
            if hashlib.sha256(blob).hexdigest() != row['image_sha256']:
                raise ValueError('Source image changed during recovery')
            started = time.perf_counter()
            with Image.open(io.BytesIO(blob)) as image:
                observed = observe_pil(model, image.convert('RGB'), device=device)
            raw = np.asarray(observed.raw_distances, np.float32)
            _, peak = interpolated_layer_maxima(raw, ctx['medians'], ctx['scales'], tuple(row['output_shape']))
            z = mad_calibrated(raw, ctx['medians'], ctx['scales'])
            local = np.asarray([local_mean(a) for a in z], np.float32)
            save_npz(output / name, protocol_id=np.asarray(replay.PROTOCOL_ID),
                     relative_path=np.asarray(row['relative_path']), raw_distances=raw,
                     layers=np.asarray(LAYERS, np.int32), grid=np.asarray(observed.grid, np.int32),
                     output_shape=np.asarray(row['output_shape'], np.int32),
                     metadata=np.asarray(json.dumps(dict(input_identity=ctx['identity'], image=row,
                         distance_divisor=1280, raw_already_divided=True, physical_row_major=True))))
            state['records'].append(dict(file=name, relative_path=row['relative_path'],
                                         cache_sha256=sha(output / name), peak=peak.tolist(), local=local.tolist(),
                                         seconds=time.perf_counter() - started))
            state['images'] = len(state['records'])
            replay.atomic_json(output / 'recovery_status.json', state)
            print(f"[CH1 calibration] {state['images']}/66 {row['relative_path']}", flush=True)
    finally:
        data.close()
    # Recheck every recorded input and cache; never replace prior cache hashes.
    validate_recovery(output, ctx, complete=False)
    current = load_context(p, recovery=True)
    if current['identity'] != ctx['identity'] or recovery_assets(current):
        raise ValueError('protected recovery inputs changed during execution')
    checks, raws, references = [], [], []
    records = {r['file']: r for r in state['records']}
    for row in calibration_rows(ctx):
        name = f"raw/{row['calibration_index']:03d}.npz"
        raw = read_cache(output / name, row, ctx['identity'])
        raws.append(raw)
        _, peak, local = statistics(raw.reshape(4, -1), np.arange(raw[0].size, dtype=np.int64),
                                    raw.shape[1:], row['output_shape'], ctx['medians'], ctx['scales'])
        record = records[name]
        for kind, actual in (('peak', peak), ('local', local)):
            check(checks, row['relative_path'], 'CPU_vs_original_' + kind, actual, record[kind])
            for j, layer in enumerate(LAYERS):
                references.append(dict(calibration_index=row['calibration_index'], relative_path=row['relative_path'],
                                       statistic=kind, layer=layer, original_value=record[kind][j],
                                       source='original_CH1_functions_on_recovered_raw'))
    patch_rows = verify_patch_mad(raws, ctx, checks)
    replay.write_csv(output / 'calibration_reference.csv', references)
    replay.write_csv(output / 'patch_mad_validation.csv', patch_rows)
    replay.write_csv(output / 'numerical_verification.csv', checks)
    state.update(status='calibration_recovered' if len(raws) == 66 and all(r['verified'] for r in checks) else 'partial_evidence',
                 images=len(raws), failed_checks=sum(not r['verified'] for r in checks),
                 seconds=time.perf_counter() - start, old_per_image_distances_reproduced=False,
                 memory_rebuilt=False, patch_MAD_refitted=False,
                 process_peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                 process_peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
    state['output_sha256'] = {name: sha(output / name) for name in
        ('source_ordered_manifest.csv', 'calibration_reference.csv', 'patch_mad_validation.csv', 'numerical_verification.csv')}
    replay.atomic_json(output / 'recovery_status.json', state)
    return state
