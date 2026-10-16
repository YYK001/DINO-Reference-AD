"""Explicit label-free cache prediction and immutable hand-off to evaluation."""
from __future__ import annotations

import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import inputs as ip
from .gates import LAYERS, BRANCHES, compute, scatter


def bounded_images(samples, function, workers):
    """At most workers image results retained, including completed future results."""
    if not 1 <= workers <= 8: raise ValueError('CPU workers must be in 1..8')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        source = iter(samples); pending = deque()
        for _ in range(workers):
            sample = next(source, None)
            if sample is not None: pending.append(pool.submit(function, sample))
        while pending:
            yield pending.popleft().result()
            sample = next(source, None)
            if sample is not None: pending.append(pool.submit(function, sample))


def original_distances(raw, meta, segments):
    union = np.minimum(raw['source_distance'], raw['target_distance'].min(axis=0))
    segment = np.stack([scatter(segments[l]['segment_distance'], segments[l]['positions'],
                                tuple(meta['grid'])).ravel()[raw['positions']] for l in LAYERS])
    return union, segment


def fresh(ctx, output, stage):
    out = ip.protect_output(ctx, output); out.mkdir(parents=True)
    ip.write_json(out/'protocol.json', ip.PROTOCOL)
    ip.write_json(out/'input_manifest.json', dict(identity=ctx['identity'], frozen=ctx['manifest']))
    ip.write_json(out/'status.json', dict(tool=ip.TOOL, stage=stage, status='running', identity=ctx['identity']))
    return out


def protect_dependencies(output, *directories):
    p=Path(output).resolve()
    for directory in directories:
        d=Path(directory).resolve()
        if p==d or p in d.parents or d in p.parents:
            raise ValueError('new output overlaps a frozen prediction/analysis directory')


def predict(ctx, output, workers=4):
    started = time.perf_counter(); out = fresh(ctx, output, 'predict')
    (out/'raw_cache').mkdir()
    def one(sample):
        with ip.Archive(ctx['paths']['raw_archive'], index=ctx['raw_index']) as archive:
            raw, meta, segments = ip.load_image(ctx, archive, sample)
        arrays, stats = compute(raw, meta, segments)
        sid = sample['sample_id']
        metadata = dict(format=ip.TOOL, identity=ctx['identity'], sample_id=sid,
                        relative_path=meta['relative_path'], grid=meta['grid'], output_shape=meta['output_shape'],
                        capture_sha256=ctx['captures'][sid]['sha256'],
                        segment_sha256={str(l):ctx['segments'][(sid,l)]['sha256'] for l in LAYERS},
                        coordinate_layout='distances_compact_gates_physical', distance_unit=ip.PROTOCOL['distance_unit'])
        file = 'raw_cache/'+sid+'.npz'
        with (out/file).open('xb') as f:
            np.savez_compressed(f, metadata=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)
        return dict(sample_id=sid, file=file, sha256=ip.sha(out/file)), [dict(sample_id=sid, **r) for r in stats]
    records, rows = [], []
    for record, statistics in bounded_images(ctx['samples'], one, workers):
        records.append(record); rows.extend(statistics)
        print(f"[neighborhood predict] {len(records)}/{len(ctx['samples'])} {record['sample_id']}", flush=True)
    ip.write_csv(out/'gate_statistics.csv', rows)
    ip.write_csv(out/'numerical_verification.csv', [dict(sample_id=r['sample_id'], check_kind='endpoints_gates_distances',
                                                       verified=True, atol=1e-7, rtol=1e-6) for r in records])
    # Rehash protected input files, not merely the current prediction output.
    current = ip.context(ctx['paths'])
    if current['identity'] != ctx['identity']: raise ValueError('protected inputs/code changed during prediction')
    files = ['protocol.json', 'input_manifest.json', 'gate_statistics.csv', 'numerical_verification.csv']
    manifest = dict(tool=ip.TOOL, identity=ctx['identity'], records=records, images=len(records),
                    files={name:ip.sha(out/name) for name in files}, GT_used=False, labels_used=False)
    if len(records) != 297: raise ValueError('incomplete predictions cannot be frozen')
    ip.write_json(out/'prediction_manifest.json', manifest)
    ip.write_json(out/'timing.json', dict(predict_seconds=time.perf_counter()-started, cpu_workers=workers,
                                        environment=ip.environment(),
                                        model_inference=False, feature_extraction=False, torch_imported=False))
    ip.write_json(out/'status.json', dict(tool=ip.TOOL, stage='predict', status='predictions_frozen',
                                        identity=ctx['identity'], images=len(records),
                                        prediction_manifest_sha256=ip.sha(out/'prediction_manifest.json'),
                                        protected_inputs_unchanged=True, GT_used=False, labels_used=False))
    return out


def verified_predictions(ctx, directory):
    root = Path(directory).resolve(); status = ip.json_file(root/'status.json')
    if (status.get('status') != 'predictions_frozen' or status.get('images') != 297
            or status.get('identity') != ctx['identity'] or status.get('GT_used') is not False
            or status.get('prediction_manifest_sha256') != ip.sha(root/'prediction_manifest.json')):
        raise ValueError('predictions incomplete, modified or belong to different inputs/code')
    manifest = ip.json_file(root/'prediction_manifest.json')
    if (manifest['identity'] != ctx['identity'] or manifest['images'] != 297
            or manifest['GT_used'] is not False or manifest['labels_used'] is not False
            or [r['sample_id'] for r in manifest['records']] != [s['sample_id'] for s in ctx['samples']]):
        raise ValueError('prediction scope/order/identity mismatch')
    for name, checksum in manifest['files'].items():
        if ip.sha(root/ip.safe(name)) != checksum: raise ValueError('prediction sidecar modified: '+name)
    input_manifest = ip.json_file(root/'input_manifest.json')
    if input_manifest != dict(identity=ctx['identity'], frozen=ctx['manifest']):
        raise ValueError('frozen input manifest differs from current protected inputs')
    records = manifest['records']
    expected = {'raw_cache/'+s['sample_id']+'.npz' for s in ctx['samples']}
    if {r['file'] for r in records} != expected or {p.relative_to(root).as_posix() for p in (root/'raw_cache').glob('*.npz')} != expected:
        raise ValueError('prediction raw scope is incomplete or has unexpected files')
    for record in records:
        if ip.sha(root/ip.safe(record['file'])) != record['sha256']:
            raise ValueError('prediction raw modified after freezing: '+record['file'])
    return root, {r['sample_id']:r for r in records}


def load_prediction(ctx, directory, record, raw, meta, segments):
    file = Path(directory)/ip.safe(record['file'])
    if ip.sha(file) != record['sha256']: raise ValueError('prediction changed while reading')
    with np.load(file, allow_pickle=False) as f:
        m = json.loads(str(f['metadata'].item()))
        saved = {k:f[k].copy() for k in f.files if k != 'metadata'}
    if (m['identity'] != ctx['identity'] or m['sample_id'] != record['sample_id']
            or m['relative_path'] != meta['relative_path'] or m['grid'] != meta['grid']
            or m['output_shape'] != meta['output_shape'] or m['distance_unit'] != ip.PROTOCOL['distance_unit']
            or m['capture_sha256'] != ctx['captures'][record['sample_id']]['sha256']
            or m['segment_sha256'] != {str(l):ctx['segments'][(record['sample_id'],l)]['sha256'] for l in LAYERS}):
        raise ValueError('prediction metadata identity mismatch')
    # Recompute the inexpensive label-free kernel: checks gate provenance as well
    # as byte integrity; no features, retrieval or model operation is performed.
    expected, _ = compute(raw, meta, segments)
    if set(saved) != set(expected) or any(saved[k].dtype != a.dtype or not np.array_equal(saved[k], a) for k,a in expected.items()):
        raise ValueError('prediction is not the frozen gate computation')
    return saved
