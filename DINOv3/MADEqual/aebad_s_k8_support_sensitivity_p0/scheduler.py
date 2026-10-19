"""One query / Source search, many support images; unchanged eight-image kernels."""
from __future__ import annotations

import numpy as np
from .selection import SEEDS, support_pool
from .inputs import LAYERS
from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import predict, validate_geometry
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.inputs import (
    validate_arrays, FORMAT,
)

GEOMETRY_FIELDS = {'segment_distance':'K8_SEGMENT', 'alpha':'alpha', 'endpoint_length':'endpoint_length',
                   'endpoint_support':'endpoint_support', 'endpoint_index':'endpoint_index',
                   'squared_correction':'squared_correction'}


def extract_query(model, image, device, image_tensor, extract_processed):
    """Same full-image operations as observe_fewshot_pil, before support scanning.

    Injectable boundaries allow CPU tests to count calls; production passes the
    original image_tensor/_extract_processed_batch, with unchanged crop batching.
    """
    tensor = image_tensor(image).to(device=device)[None]
    processed = model.preprocessing(tensor)
    items = extract_processed(model,processed,device=device,query_masks=[None],include_identity_descriptors=False)
    if len(items)!=1: raise ValueError('expected exactly one full original query')
    return items[0], list(processed.shape[-2:])


def scan_layer(query, source, support_count, load_bank, exact):
    """exact returns raw Euclidean NumPy FP32 / int64; kernel is injected at boundary.

    Production passes _exact_top1_chunked, tests explicitly replace only that
    Torch boundary. No cache of Q or dense query x memory matrix is generated.
    """
    d, idx = exact(query, source)
    source_distance = d.astype(np.float32) / 1280.
    targets = []; indices = []
    for i in range(support_count):
        bank = load_bank(i)
        value, index = exact(query, bank)
        targets.append(value.astype(np.float32)/1280.); indices.append(index.astype(np.int64))
        del bank
    return source_distance, idx.astype(np.int64), np.stack(targets), np.stack(indices)


def attach_geometry(shared, mapping, load_feature):
    result = dict(shared); diagnostics = []
    # Features are mapped one layer at a time. They are full original patch rows.
    for seed in SEEDS:
        selected = mapping[str(seed)]; layers = []
        for off, layer in enumerate(LAYERS):
            features = [load_feature(i, layer) for i in selected]
            value, diag = geometry_layer(shared['target_distance'][selected, off],
                shared['target_index'][selected, off], shared['source_distance'][off], features)
            layers.append(value); diagnostics.append(dict(support_seed=seed, layer=layer, **diag))
            del features
        for name, old_name in GEOMETRY_FIELDS.items():
            result[f'seed{seed}_{name}'] = np.stack([v[old_name] for v in layers])
    return result, diagnostics


def group_raw(arrays, mapping, seed):
    selected = mapping[str(seed)]
    if len(selected) != 8 or len(set(selected)) != 8:
        raise ValueError('one group must map to eight distinct global support images')
    result = {k:arrays[k] for k in ('positions', 'source_distance', 'source_index')}
    result.update(target_distance=arrays['target_distance'][selected], target_index=arrays['target_index'][selected])
    result.update({name: arrays[f'seed{seed}_{name}'] for name in GEOMETRY_FIELDS})
    return result


def validate_shared(ctx, sample, arrays, meta):
    domain = sample['domain']; pool, mapping = support_pool(ctx['plan'], domain)
    if (meta.get('identity') != ctx['identity'] or meta.get('sample_id') != sample['sample_id']
            or meta.get('relative_path') != sample['relative_path'] or meta.get('domain') != domain
            or meta.get('image_sha256') != sample['image_sha256']
            or meta.get('distance_unit') != 'euclidean_over_1280_once'
            or meta.get('mapping') != mapping or len(meta['supports']) != len(pool)
            or meta.get('output_shape') != [sample['image_height']//4, sample['image_width']//4]):
        raise ValueError('shared capture identity/geometry mismatch')
    for actual, expected in zip(meta['supports'], pool):
        if any(actual.get(k) != expected[k] for k in ('relative_path', 'image_sha256')):
            raise ValueError('shared support pool identity/order changed')
    for seed in SEEDS:
        raw = group_raw(arrays, mapping, seed)
        group_meta = dict(meta, format=FORMAT, source_rows=[100000]*4,
            supports=[dict(meta['supports'][i], support_index=j+1) for j, i in enumerate(mapping[str(seed)])])
        validate_arrays(raw, group_meta)
        validate_geometry(raw)
        predict(raw, tuple(meta['grid']))  # validates containment and physical local pair IDs


def predictions(arrays, mapping, grid):
    return {seed: predict(group_raw(arrays, mapping, seed), tuple(grid)) for seed in SEEDS}
