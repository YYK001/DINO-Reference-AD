"""Label-free, per-image/per-layer gates in PHYSICAL native-grid coordinates."""
from __future__ import annotations

import numpy as np

LAYERS = (7, 15, 23, 31)
BRANCHES = ('UNION_K8_CH1', 'SEGMENT_K8_CH1',
            'SEGMENT_PAIR_GATE_K8_CH1', 'SEGMENT_NEIGHBOR_GATE_K8_CH1')
ATOL, RTOL = 1e-7, 1e-6


def scatter(a, positions, grid):
    a, p = np.asarray(a), np.asarray(positions)
    n = int(np.prod(grid))
    if p.dtype != np.int64 or p.shape != (n,) or not np.array_equal(np.sort(p), np.arange(n)) or a.shape[-1] != n:
        raise ValueError('complete physical bijection required')
    out = np.empty_like(a)
    out[..., p] = a
    return out.reshape(*a.shape[:-1], *grid)


def neighborhood(endpoint_support, endpoint_index, positions, grid, support_grids):
    """Return physical maps. Endpoint order is immaterial; support identities are not.

    Same-pair votes and BOTH endpoint displacement tests are evaluated directly
    against the original fields, never against previously accepted neighbors.
    """
    support = scatter(endpoint_support, positions, grid)
    index = scatter(endpoint_index, positions, grid)
    if support.shape != (2, *grid) or index.shape != support.shape:
        raise ValueError('two endpoints required')
    if support.dtype.kind not in 'iu' or index.dtype.kind not in 'iu':
        raise ValueError('integer support IDs / patch indices required')
    if len(support_grids) != 8 or any(tuple(g) != tuple(grid) for g in support_grids):
        raise ValueError('unsupported query/support grid scale; no coordinate transform')
    if np.any((support < 1) | (support > 8)) or np.any(support[0] == support[1]):
        raise ValueError('two different manifest support IDs in 1..8 required')
    order = np.argsort(support, axis=0, kind='stable')
    pair = np.take_along_axis(support, order, axis=0)
    idx = np.take_along_axis(index, order, axis=0)
    widths = np.asarray([g[1] for g in support_grids], np.int64)[pair - 1]
    sizes = np.asarray([np.prod(g) for g in support_grids], np.int64)[pair - 1]
    if np.any(idx < 0) or np.any(idx >= sizes):
        raise ValueError('support patch index out of bounds')
    coordinates = np.stack((idx // widths, idx % widths), axis=1)
    h, w = grid
    count = np.zeros(grid, np.uint8)
    pair_votes = count.copy(); geom_votes = count.copy()
    repeated_any_votes = count.copy(); repeated_both_votes = count.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            ys = slice(max(0, -dy), min(h, h-dy))
            xs = slice(max(0, -dx), min(w, w-dx))
            yj = slice(max(0, dy), min(h, h+dy))
            xj = slice(max(0, dx), min(w, w+dx))
            same = np.all(pair[:, ys, xs] == pair[:, yj, xj], axis=0)
            delta = coordinates[:, :, yj, xj] - coordinates[:, :, ys, xs]
            compatible = np.all(np.abs(delta - np.asarray([dy, dx])[None, :, None, None]) <= 1, axis=(0, 1))
            repeated = idx[:, ys, xs] == idx[:, yj, xj]
            count[ys, xs] += 1
            pair_votes[ys, xs] += same
            geom_votes[ys, xs] += same & compatible
            repeated_any_votes[ys, xs] += same & np.any(repeated, axis=0)
            repeated_both_votes[ys, xs] += same & np.all(repeated, axis=0)
    threshold = (count + 1) // 2
    pair_gate = (count > 0) & (pair_votes >= threshold)
    geom_gate = (count > 0) & (geom_votes >= threshold)
    if np.any(geom_gate & ~pair_gate):
        raise AssertionError('geometry acceptance must be a subset of pair acceptance')
    return dict(pair_gate=pair_gate, geom_gate=geom_gate, neighbor_count=count,
                pair_votes=pair_votes, geom_votes=geom_votes,
                repeated_any_votes=repeated_any_votes, repeated_both_votes=repeated_both_votes)


def bounded(union, segment, pair_gate, geom_gate):
    """Use saved SEGMENT or FULL union, with no clipping / rescaling."""
    u, s = np.asarray(union), np.asarray(segment)
    if u.dtype != np.float32 or s.dtype != np.float32 or u.shape != s.shape or not np.isfinite([u, s]).all() or np.any(s < 0):
        raise ValueError('finite nonnegative float32 saved distances required')
    if pair_gate.shape != u.shape or geom_gate.shape != u.shape or pair_gate.dtype != bool or geom_gate.dtype != bool:
        raise ValueError('aligned boolean gates required')
    if np.any(geom_gate & ~pair_gate):
        raise ValueError('invalid gate inclusion')
    p, g = np.where(pair_gate, s, u), np.where(geom_gate, s, u)
    for lower, upper in ((s, p), (p, g), (g, u)):
        if np.any(lower > upper + ATOL + RTOL * np.abs(upper)):
            raise ValueError('distance containment failed; refusing to clip')
    return p, g


def compute(arrays, metadata, segments):
    """Production prediction kernel. No label, RGB, GT or cross-image state input."""
    positions = arrays['positions']; grid = tuple(metadata['grid'])
    union = np.minimum(arrays['source_distance'], arrays['target_distance'].min(axis=0))
    target = arrays['target_distance']
    selected = np.argsort(target, axis=0, kind='stable')[:2]
    columns = np.arange(len(positions))
    distances, gates, rows = [], [], []
    for offset, layer in enumerate(LAYERS):
        seg = segments[layer]
        # Independent segment compact order is permitted; align through physical positions.
        segment = scatter(seg['segment_distance'], seg['positions'], grid).ravel()[positions]
        endpoint_support = scatter(seg['endpoint_support'], seg['positions'], grid).reshape(2, -1)[:, positions]
        endpoint_index = scatter(seg['endpoint_index'], seg['positions'], grid).reshape(2, -1)[:, positions]
        expected_support = selected[:, offset] + 1
        expected_index = arrays['target_index'][selected[:, offset], offset, columns]
        if not np.array_equal(endpoint_support, expected_support) or not np.array_equal(endpoint_index, expected_index):
            raise ValueError('saved SEGMENT endpoints differ from stable manifest-order Top2')
        fields = neighborhood(endpoint_support, endpoint_index, positions, grid, [s['grid'] for s in metadata['supports']])
        compact_gates = [fields[k].ravel()[positions] for k in ('pair_gate', 'geom_gate')]
        pair, geom = bounded(union[offset], segment, *compact_gates)
        distances.append((pair, geom)); gates.append(fields)
        reduced = union[offset] - segment > ATOL + RTOL * np.abs(union[offset])
        a, b = np.sort(target[:, offset], axis=0, kind='stable')[:2]
        _, inverse, multiplicity = np.unique(np.stack((endpoint_support.ravel(), endpoint_index.ravel()),axis=1),
                                             axis=0,return_inverse=True,return_counts=True)
        reused = (multiplicity[inverse] > 1).reshape(2,-1).any(axis=0)
        pair_count=int(compact_gates[0].sum());geom_count=int(compact_gates[1].sum())
        rows.append(dict(layer=layer, patches=len(positions), segment_reduced_patches=int(reduced.sum()),
                         pair_accept_ratio=float(fields['pair_gate'].mean()),
                         geom_accept_ratio=float(fields['geom_gate'].mean()),
                         pair_pass_geom_reject_ratio=float((fields['pair_gate'] & ~fields['geom_gate']).mean()),
                         pair_actual_reduction_ratio=float((reduced & compact_gates[0]).mean()),
                         geom_actual_reduction_ratio=float((reduced & compact_gates[1]).mean()),
                         pair_actual_reduction_given_accept=float(np.sum(reduced & compact_gates[0])/pair_count) if pair_count else None,
                         geom_actual_reduction_given_accept=float(np.sum(reduced & compact_gates[1])/geom_count) if geom_count else None,
                         distinct_selected_support_patch_pairs=int(len(multiplicity)),
                         repeated_selected_endpoint_occurrences=int(np.sum(multiplicity-1)),
                         query_positions_with_reused_selected_support_patch=int(reused.sum()),
                         near_tied_endpoint_candidates=int(np.sum(b-a <= ATOL+RTOL*np.abs(a))),
                         boundary_patches=int(np.sum(fields['neighbor_count'] < 8)),
                         repeated_any_neighbor_votes=int(fields['repeated_any_votes'].sum()),
                         repeated_both_neighbor_votes=int(fields['repeated_both_votes'].sum()),
                         **{f'{k}_{v}_patches': int(np.sum(fields[k] == v))
                            for k in ('neighbor_count', 'pair_votes', 'geom_votes') for v in range(9)}))
    output = dict(positions=positions.copy(),
                  pair_distance=np.stack([d[0] for d in distances]),
                  neighbor_distance=np.stack([d[1] for d in distances]))
    for name in gates[0]:
        output[name] = np.stack([g[name] for g in gates])
    return output, rows
