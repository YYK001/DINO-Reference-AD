"""Stream four native response maps, freeze all settings BEFORE GT can enter."""
import time
import numpy as np
from . import inputs as ip, runtime as rt, numerics as nm
from .inputs import st
from .protocol import *
from .budget import observe
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.evaluation import shared
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values, response
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.numerics import weighted


def baseline_checks(a, d, seed, row, x, meta):
    p = ip.parent(a);baseline, original, ref = ip.original_baseline(a, d, seed, row, (U, S, 'positions'))
    grid = meta['grid'];shape = [row['image_height']//4, row['image_width']//4]
    original_grid = original['metadata']['grid']
    if grid != original_grid:
        raise ValueError('baseline grid differs')
    du = np.minimum(x['Source_distance'], x['target_distance'].min(1))
    expected = nm.physical(baseline[U], baseline['positions'], grid)
    actual = nm.physical(z_values(du, p.inputs['mad']['patch']), x['positions'], grid)
    error = float(np.abs(actual.astype(np.float64)-expected).max())
    checks = [dict(check='same_bank_full_K8_UNION_z', max_abs_error=error, atol=1e-5, verified=error <= 1e-5)]
    if error > 1e-5:
        raise ValueError('same-bank full K8 UNION mismatch; preserve error, no tolerance relaxation or metric stitching')
    frozen = p.record(d, seed, row)
    for branch in (U, S, SOURCE):
        if branch == SOURCE:
            image, scores = weighted(baseline[U], baseline['positions'], grid, shape, p.weights[d, 'WS'])
        else:
            image, scores = response(baseline[branch], baseline['positions'], grid, shape, p.inputs['mad']['image'])
        map_ok = shared.array_sha(image) == frozen['map_hashes'][branch]
        score_error = max(abs(v-frozen['scores'][branch][k]) for k, v in scores.items())
        checks.append(dict(check='frozen_K8_map_and_score', branch=branch, map_sha_equal=map_ok,
                           max_abs_error=score_error, atol=1e-5, verified=map_ok and score_error <= 1e-5))
        if not checks[-1]['verified']:
            raise ValueError('full K8 reference map/score reconstruction differs')
    return checks, ref


def predict(a):
    p = ip.parent(a);frozen_model = st.sealed_json(a.run_root/'normal_model.json');new = 0;reused = 0
    for d in a.domains:
        for seed in a.support_seeds:
            references = rt.normal_model(a, d, seed);records = []
            for i, row in enumerate(p.samples(d), 1):
                path = ip.prediction_path(a, d, seed, row);record = ip.record_path(a, d, seed, row)
                if record.exists():
                    verify_record(a, d, seed, row);reused += 1
                else:
                    x, source = ip.inspect_split(a, d, seed, row);meta = source['metadata'];grid = meta['grid']
                    checks, baseline = baseline_checks(a, d, seed, row, x, meta)
                    zs, zt = nm.normalized(x['Source_distance'], x['target_distance'], p.inputs['mad']['patch'])
                    zs = nm.physical(zs, x['positions'], grid);zt = nm.physical(zt, x['positions'], grid)
                    tick = time.perf_counter()
                    native = nm.score_native(zs, zt, references, np.asarray(p.weights[d, 'WS'])[:, 0], device=a.device,
                                             query_chunk=a.query_chunk, reference_chunk=a.reference_chunk)
                    shape = [row['image_height']//4, row['image_width']//4];images = nm.maps(native, grid, shape)
                    dependencies = dict(**ip.signature(a, d, seed), image=ip.image_contract(row),
                                        split_seal=source['seal'], normal_model_seal=frozen_model['seal'])
                    saved = st.put(path, native, dict(**dependencies, grid=grid, output_shape=shape, physical_order=True, GT_used=False))
                    seg = p.record(d, seed, row)['scores'][S]['CH1']
                    scores = {b: dict(SEGMENT_CH1=seg, MAP_MAX=float(image.max())) for b, image in images.items()}
                    st.sealed_json(record, dict(**dependencies, sample_id=row['sample_id'], cache_seal=saved['seal'],
                         map_hashes={b: shared.array_sha(image) for b, image in images.items()}, scores=scores,
                         baseline_origin=baseline, checks=checks, GT_used=False))
                    new += 1
                    print(f'[joint tail predict] {d}/seed{seed} {i}/{len(p.samples(d))} {row["sample_id"]} four_maps seconds={time.perf_counter()-tick:.3f}', flush=True)
                r = st.sealed_json(record);records.append(dict(sample_id=row['sample_id'], seal=r['seal']))
                observe(a)
            st.sealed_json(ip.unit(a, d, seed)/'predictions_frozen.json', dict(**ip.signature(a, d, seed), records=records, GT_used=False))
    if all((ip.unit(a, d, s)/'predictions_frozen.json').exists() for d in DOMAINS for s in SEEDS):
        global_freeze(a)
    return dict(status='requested_predictions_complete', new_images=new, reused_images=reused, GT_used=False)


def verify_record(a, d, seed, row):
    r = st.sealed_json(ip.record_path(a, d, seed, row));frozen_model = st.sealed_json(a.run_root/'normal_model.json')
    if any(r.get(k) != v for k, v in ip.signature(a, d, seed).items()) or r['image'] != ip.image_contract(row) or r['normal_model_seal'] != frozen_model['seal'] or r['GT_used']:
        raise ValueError('frozen query input/reference identity differs')
    _, split = ip.inspect_split(a, d, seed, row, load=False)
    if split['seal'] != r['split_seal']:
        raise ValueError('query distances changed since prediction')
    native, saved = st.get(ip.prediction_path(a, d, seed, row))
    if saved['seal'] != r['cache_seal'] or saved['metadata']['split_seal'] != r['split_seal'] or set(native) != set(MAPS):
        raise ValueError('native prediction cache differs')
    n = int(np.prod(saved['metadata']['grid']))
    if any(v.shape != (n,) or v.dtype != np.float32 or not np.isfinite(v).all() for v in native.values()):
        raise ValueError('native FP32 map layout differs')
    if any(not c['verified'] for c in r['checks']):
        raise ValueError('prior baseline verification failed')
    return r, native, saved


def global_freeze(a):
    cached = getattr(a, '_global_prediction_freeze', None)
    if cached is not None:
        # Each subprocess validates the complete list once. Native map payloads
        # are still checked against their per-image freeze during every replay.
        current = st.sealed_json(a.run_root/'predictions_frozen.json')
        if current != cached:raise ValueError('global prediction freeze changed within stage')
        return current
    p = ip.parent(a);records = []
    for d in DOMAINS:
        for seed in SEEDS:
            r = st.sealed_json(ip.unit(a, d, seed)/'predictions_frozen.json')
            if any(r.get(k) != v for k, v in ip.signature(a, d, seed).items()) or len(r['records']) != len(p.samples(d)) or r['GT_used']:
                raise ValueError('incomplete prediction setting; GT refused')
            for row, ref in zip(p.samples(d), r['records']):
                value = st.sealed_json(ip.record_path(a, d, seed, row))
                if ref != dict(sample_id=row['sample_id'], seal=value['seal']):
                    raise ValueError('prediction list identity differs')
            records.append(dict(domain=d, support_seed=seed, seal=r['seal']))
    result = st.sealed_json(a.run_root/'predictions_frozen.json', dict(identity=st.sealed_json(a.run_root/'inputs.json')['identity'], records=records, images=1804, GT_used=False))
    a._global_prediction_freeze = result
    return result


def replay(a, d, seed, row, only=None):
    r, native, saved = verify_record(a, d, seed, row)
    chosen = native if only is None else {only: native[only]}
    images = nm.maps(chosen, saved['metadata']['grid'], saved['metadata']['output_shape'])
    for b, image in images.items():
        if shared.array_sha(image) != r['map_hashes'][b] or float(image.max()) != r['scores'][b]['MAP_MAX']:
            raise ValueError('native map replay differs from pre-GT freeze')
    return images, r['scores'], saved['metadata']
