"""Float64 Euclidean FastRef equations; one balanced OT problem per full query."""
from __future__ import annotations
import numpy as np

CONFIG = dict(seed=42, dimension=1280, prototype_count=512,
    reconstruction='sum_squared_euclidean', lambda_value=.3, outer_iterations=2,
    feature_unit_divisor=1280, epsilon_support_fraction=.1,
    sinkhorn_max_iterations=1000, sinkhorn_check_every=5,
    marginal_relative_tolerance=1e-6, total_mass_tolerance=1e-8,
    maximum_gram_condition=1e12, solve_relative_tolerance=1e-10,
    update_relative_tolerance=1e-8, new_nn_matrix_bytes=64 << 20)


class NumericalFailure(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def squared_distances(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    aa = np.sum(a*a, axis=1)[:, None]; bb = np.sum(b*b, axis=1)[None, :]
    values = aa + bb - 2*(a @ b.T)
    if not np.isfinite(values).all() or np.any(values < -1e-10*(aa+bb+1)):
        raise NumericalFailure('invalid squared Euclidean cost', {})
    return np.maximum(values, 0.)


def make_constants(prototypes, config=CONFIG):
    raw = np.asarray(prototypes)
    if raw.ndim != 2 or not 0 < len(raw) <= raw.shape[1] or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise NumericalFailure('finite float32 full-row-rank prototype matrix required', {})
    scaled = raw.astype(np.float64)/config['feature_unit_divisor']
    gram = scaled @ scaled.T
    eigenvalues = np.linalg.eigvalsh(gram)
    threshold = max(scaled.shape)*np.finfo(np.float64).eps*float(eigenvalues[-1])
    rank = int(np.count_nonzero(eigenvalues > threshold))
    condition = float(eigenvalues[-1]/eigenvalues[0]) if eigenvalues[0] > 0 else None
    diag = dict(prototype_rows=len(raw), feature_dimension=raw.shape[1], gram_rank=rank,
        gram_condition=condition, rank_threshold=float(threshold), solver='ordinary_cholesky_solve_float64',
        fallback_used=False, feature_unit_divisor=config['feature_unit_divisor'])
    if rank != len(raw) or condition is None or condition > config['maximum_gram_condition']:
        raise NumericalFailure('prototype Gram is singular or ill-conditioned; no fallback', diag)
    try:
        chol = np.linalg.cholesky(gram)
        factor = np.linalg.solve(chol.T, np.linalg.solve(chol, scaled))
    except np.linalg.LinAlgError as error:
        raise NumericalFailure('ordinary Cholesky solve failed', diag) from error
    denominator = np.linalg.norm(gram)*np.linalg.norm(factor)+np.linalg.norm(scaled)
    residual = float(np.linalg.norm(gram@factor-scaled)/denominator)
    diag['solve_relative_residual'] = residual
    if not np.isfinite(factor).all() or residual > config['solve_relative_tolerance']:
        raise NumericalFailure('prototype solve residual failed', diag)
    costs = squared_distances(scaled, scaled)
    median = float(np.median(costs[~np.eye(len(raw), dtype=bool)])) if len(raw) > 1 else 0.
    epsilon = config['epsilon_support_fraction']*median
    diag.update(support_pair_cost_median=median, epsilon=float(epsilon), epsilon_origin='fixed_fraction_of_normal_prototype_pair_squared_cost')
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise NumericalFailure('normal support cost scale must be positive; no epsilon fallback', diag)
    return dict(prototypes=raw.copy(), scaled=scaled, factor=factor, epsilon=epsilon, diagnostics=diag)


def logsumexp(values, axis):
    maximum = values.max(axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(values-maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def balanced_transport(cost, epsilon, config=CONFIG):
    cost = np.asarray(cost, np.float64)
    if cost.ndim != 2 or not all(cost.shape) or not np.isfinite(cost).all() or np.any(cost < 0) or not np.isfinite(epsilon) or epsilon <= 0:
        raise NumericalFailure('invalid balanced OT inputs', {})
    m, n = cost.shape
    log_a = -np.log(float(m)); log_b = -np.log(float(n))
    kernel = -cost/epsilon
    if not np.isfinite(kernel).all(): raise NumericalFailure('nonfinite log OT kernel', {})
    u = np.zeros(m); v = np.zeros(n); diag = {}
    for step in range(1, config['sinkhorn_max_iterations']+1):
        u = log_a-logsumexp(kernel+v[None, :], 1)
        v = log_b-logsumexp(kernel+u[:, None], 0)
        if step % config['sinkhorn_check_every'] and step != config['sinkhorn_max_iterations']: continue
        transport = np.exp(kernel+u[:, None]+v[None, :])
        row = transport.sum(axis=1); column = transport.sum(axis=0)
        row_error = float(np.max(np.abs(row*m-1)))
        column_error = float(np.max(np.abs(column*n-1)))
        mass_error = float(abs(transport.sum()-1))
        diag = dict(sinkhorn_iterations=step, row_max_relative_error=row_error,
            column_max_relative_error=column_error, total_mass=float(transport.sum()), total_mass_error=mass_error,
            row_target=1/m, column_target=1/n, nonfinite_entries=int((~np.isfinite(transport)).sum()),
            underflow_zero_entries=int((transport == 0).sum()), epsilon=float(epsilon), global_query_rows=m)
        if not np.isfinite(transport).all() or not np.isfinite(row_error+column_error+mass_error):
            raise NumericalFailure('nonfinite balanced transport', diag)
        if max(row_error, column_error) <= config['marginal_relative_tolerance'] and mass_error <= config['total_mass_tolerance']:
            return transport, diag
    raise NumericalFailure('Sinkhorn did not converge; thresholds unchanged', diag)


def refine(query, constants, *, enabled=True, config=CONFIG):
    """No mutable per-query state is retained in constants. Disabled returns P."""
    q = np.asarray(query); p = constants['prototypes']
    if q.ndim != 2 or not len(q) or q.shape[1] != p.shape[1] or q.dtype != np.float32 or not np.isfinite(q).all():
        raise NumericalFailure('raw query/prototypes must share finite float32 feature space', {})
    if not enabled: return p.copy(), dict(refinement_enabled=False, refined_rows=len(p))
    x = q.astype(np.float64)/config['feature_unit_divisor']; m = constants['scaled']
    query_gram = x @ m.T
    projection = query_gram @ constants['factor']; refined = projection.copy(); diagnostics = []
    for outer in range(config['outer_iterations']):
        transport, diag = balanced_transport(squared_distances(refined, m), constants['epsilon'], config)
        mass = transport.sum(axis=1, keepdims=True)
        pull = transport @ m
        next_refined = (projection+config['lambda_value']*pull)/(1+config['lambda_value']*mass)
        gradient = ((1+config['lambda_value']*mass)*next_refined-x-config['lambda_value']*pull) @ m.T
        denominator = max(float(np.linalg.norm(query_gram)), np.finfo(np.float64).tiny)
        residual = float(np.linalg.norm(gradient)/denominator)
        diag.update(outer_iteration=outer+1, lambda_value=config['lambda_value'], transport_rescaled_by_m=False,
            update_normal_equation_relative_residual=residual,
            query_relative_reconstruction_error=float(np.linalg.norm(next_refined-x)/max(np.linalg.norm(x),np.finfo(np.float64).tiny)))
        diagnostics.append(diag)
        if not np.isfinite(next_refined).all() or not np.isfinite(residual) or residual > config['update_relative_tolerance']:
            raise NumericalFailure('refinement normal-equation residual failed', diag)
        refined = next_refined
    result = np.ascontiguousarray((refined*config['feature_unit_divisor']).astype(np.float32))
    if not np.isfinite(result).all(): raise NumericalFailure('nonfinite float32 refined vectors', diagnostics[-1])
    return result, dict(refinement_enabled=True, refined_rows=len(result), initial_prototype_rows=len(p),
        reconstruction='sum_squared_euclidean', refined_vectors_persisted=False, outer=diagnostics,
        gram_rank=constants['diagnostics']['gram_rank'], gram_condition=constants['diagnostics']['gram_condition'],
        solve_relative_residual=constants['diagnostics']['solve_relative_residual'])
