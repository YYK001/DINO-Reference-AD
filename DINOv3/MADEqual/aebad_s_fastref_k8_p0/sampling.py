"""Finite inverse-density sampling of original rows; no eager Torch import.

The local-density statistic follows SuperADD (raw Euclidean kNN and mean/10
starting radius). Exact-size exponential-race sampling is an explicit algorithm
adaptation, not the upstream fixed-uniform radius loop or a silent fallback.
"""
from __future__ import annotations
import hashlib
import numpy as np

CONFIG = dict(revision='target512_inverse_density_exact_budget_v3',
    entry='DINOv3.MADEqual.aebad_s_fastref_k8_p0.sampling.sample_indices',
    distance_entry='external_baselines.superadd_external.memory_variants._chunked_exact_nearest_neighbors',
    rounds=100, knn_neighbors=100, normalize=False, random_projection=False,
    distance='raw_float32_euclidean_torch_cdist_use_mm_for_euclid_dist',
    distance_matrix_max_bytes=1 << 30, quota_per_layer=512, seed=42,
    rng_policy='local_numpy_RandomState42_once_continuous_L7_L15_L23_L31',
    candidate_pool='manifest_order_8_supports_then_physical_row_major_all_patches_per_layer',
    grouping='one_random_permutation_balanced_disjoint_min_rounds_count_groups',
    small_pool_neighbors='min_100_group_size_including_self',
    radius_factor=.1, radius_policy='fixed_group_mean_knn_distance_times_radius_factor',
    weight='1_over_1_plus_count_knn_distance_strictly_less_than_radius',
    selection='global_exponential_race_without_replacement_exact_N',
    priority='negative_log1p_negative_uniform_over_weight',
    tie_rule='ascending_original_candidate_index',
    output='original_rows_in_ascending_candidate_index_order',
    adaptation='Source_inverse_density_preference_with_new_bounded_exact_budget_selection')


def admission(candidate_count, quota=512, dimension=1280):
    count, quota, dimension = int(candidate_count), int(quota), int(dimension)
    error = ''
    if not 0 < quota <= dimension:
        error = 'prototype budget must be positive and <= feature dimension for ordinary full-row-rank FastRef solve'
    elif count < quota:
        error = f'not enough distinct patch indices: candidates={count}, budget={quota}; no duplication or reduced budget'
    return dict(status='blocked' if error else 'static_budget_check_passed_runtime_unverified',
        candidate_count=count,quota=quota,feature_dimension=dimension,
        effective_rounds=min(CONFIG['rounds'],max(0,count)),error=error)


def torch_knn_distances(features, neighbors, device):
    """Original exact distance backend, imported only during explicit prepare."""
    import torch
    from external_baselines.superadd_external.memory_variants import _chunked_exact_nearest_neighbors
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(features)).to(device)
        distances, _ = _chunked_exact_nearest_neighbors(tensor,tensor,neighbors,False,
            max_distance_matrix_bytes=CONFIG['distance_matrix_max_bytes'])
        values = distances.cpu().numpy().copy()
    return values


def density_weights(distances, radius_factor=.1):
    """Actual production density statistic; zero radius retains strict < rule."""
    values = np.asarray(distances)
    if (values.ndim != 2 or not values.size or not np.isfinite(values).all()
        or np.any(values < 0) or not np.isfinite(radius_factor) or radius_factor <= 0):
        raise ValueError('finite nonnegative kNN distances and positive radius factor required')
    radius = float(np.mean(values,dtype=np.float64)*radius_factor)
    if not np.isfinite(radius): raise ValueError('nonfinite density radius; no sampler fallback')
    close = np.sum(values < radius,axis=1).astype(np.int64)
    weights = 1./(close.astype(np.float64)+1.)
    return weights,close,radius


def exact_weighted_indices(weights, uniforms, quota):
    """Exponential clocks implement weighted sampling without replacement.

    There are exactly M finite keys, one ordering, and N distinct indices.
    No while/retry or subset truncation is used. Ties have an explicit rule.
    """
    weights, uniforms = np.asarray(weights,np.float64),np.asarray(uniforms,np.float64)
    quota = int(quota)
    if (weights.ndim != 1 or uniforms.shape != weights.shape or not 0 < quota <= len(weights)
        or not np.isfinite(weights).all() or np.any(weights <= 0)
        or not np.isfinite(uniforms).all() or np.any(uniforms < 0) or np.any(uniforms >= 1)):
        raise ValueError('positive finite weights, uniforms in [0,1), and feasible exact budget required')
    keys = -np.log1p(-uniforms)/weights
    if not np.isfinite(keys).all(): raise ValueError('nonfinite weighted priority; no sampler fallback')
    order = np.lexsort((np.arange(len(weights),dtype=np.int64),keys))
    return np.sort(order[:quota]).astype(np.int64),keys


def sample_indices(features, quota, device, *, rng, rounds=100, neighbors=100, distance_backend=None):
    """Production control path; tests may replace only the missing kNN backend.

    Every candidate is assigned once to a balanced random group; all group
    weights compete for the ONE global quota. No per-group budget is rounded.
    RNG is caller-owned, never re-seeded, never global, and never retried.
    """
    values = np.asarray(features)
    quota,rounds,neighbors=int(quota),int(rounds),int(neighbors)
    if (values.ndim != 2 or values.dtype != np.float32 or not values.shape[1]
        or not np.isfinite(values).all() or not 0 < quota <= values.shape[0] or rounds <= 0 or neighbors <= 0):
        raise ValueError('finite raw float32 pool, feasible positive budget/rounds/neighbors required')
    count = len(values)
    if not isinstance(rng,np.random.RandomState): raise ValueError('caller-owned frozen RandomState required')
    backend = torch_knn_distances if distance_backend is None else distance_backend
    permutation = rng.permutation(count)
    groups = np.array_split(permutation,min(rounds,count))
    weights = np.empty(count,np.float64);densities=np.empty(count,np.int64)
    memberships=np.empty(count,np.int64);radii=np.empty(count,np.float64);summaries=[]
    for number,indices in enumerate(groups):
        k = min(neighbors,len(indices))
        distances = np.asarray(backend(values[indices],k,device))
        if distances.shape != (len(indices),k): raise ValueError('exact kNN backend returned invalid shape')
        w,close,radius = density_weights(distances,CONFIG['radius_factor'])
        weights[indices]=w;densities[indices]=close;memberships[indices]=number;radii[indices]=radius
        summaries.append(dict(group=number,candidates=len(indices),neighbors=k,radius=radius,
            weight_min=float(w.min()),weight_max=float(w.max()),density_count_min=int(close.min()),density_count_max=int(close.max())))
    selected,keys = exact_weighted_indices(weights,rng.random_sample(count),quota)
    if selected.shape != (quota,) or np.unique(selected).size != quota:
        raise ValueError('exact unique prototype budget violated')
    report = dict(candidate_count=count,quota=quota,requested_rounds=rounds,effective_rounds=len(groups),
        distance_backend='original_Torch_exact' if distance_backend is None else 'explicit_test_backend',
        all_candidates_visited_once=True,per_group_quotas=False,radius_iterations=0,retries=0,
        weighted_selection='exponential_race_without_replacement',selected_rows=quota,
        candidate_weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
        weight_min=float(weights.min()),weight_max=float(weights.max()),weight_mean=float(weights.mean()),
        density_count_mean=float(densities.mean()),uniform_weights=bool(np.all(weights==weights[0])),
        effective_weight_sample_size=float(weights.sum()**2/np.sum(weights*weights)),groups=summaries,
        selected=[dict(candidate_index=int(i),group=int(memberships[i]),neighbors=min(neighbors,len(groups[memberships[i]])),
            radius=float(radii[i]),density_count=int(densities[i]),weight=float(weights[i]),priority=float(keys[i])) for i in selected])
    return selected,report
