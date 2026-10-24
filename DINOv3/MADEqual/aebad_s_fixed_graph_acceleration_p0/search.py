"""FP32 endpoint screening, conservative intervals, FP64 direct winners.

No autocast of the backbone or original Source/Target search. Inactive edges
and unused endpoints are removed without changing canonical winning edge IDs.
Two passes find a global upper bound BEFORE refining any candidate, avoiding
the old refinement of provisional winners in every edge block.
"""
from contextlib import contextmanager
import threading
import time
import numpy as np

ENGINE = 'compact_fp32_screen_fp64_global_refine_v1'
_PRECISION_LOCK = threading.RLock()


@contextmanager
def strict_screen(torch, device):
    """The FP32 error interval assumes no TF32 truncation or outer autocast.

    Restore the process's previous settings, including on exceptions. This
    context encloses ONLY graph search, never backbone/Source NN calls.
    """
    with _PRECISION_LOCK:
        previous = torch.get_float32_matmul_precision()
        tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.autocast(device_type=device.type, enabled=False):
                yield
        finally:
            torch.set_float32_matmul_precision(previous)
            torch.backends.cuda.matmul.allow_tf32 = tf32


def gamma(count, epsilon):
    x = count * epsilon
    if not 0 <= x < 1:
        raise ValueError('unsupported dimension for the rounding interval')
    return x / (1 - x)


class PreparedSearch:
    def __init__(self, nodes, edges, masks, device, edge_chunk=2048):
        import torch
        self.torch = torch
        self.device = device
        self.dev = torch.device('cpu' if device == 'torch:cpu' else device)
        if self.dev.type not in ('cpu', 'cuda'):
            raise ValueError('explicit CPU test or CUDA device required')
        nodes = np.asarray(nodes)
        edges = np.asarray(edges, np.int64).reshape(-1, 2)
        masks = np.asarray(masks, bool)
        if nodes.ndim != 2 or nodes.dtype != np.float32 or not np.isfinite(nodes).all():
            raise ValueError('original finite FP32 node rows required')
        if masks.ndim != 2 or masks.shape[1] != len(edges) or edge_chunk < 1:
            raise ValueError('aligned masks and positive edge chunk required')
        if len(edges) and (edges.min() < 0 or edges.max() >= len(nodes)):
            raise ValueError('invalid endpoint index')
        self.original_nodes = nodes
        self.original_edges = edges.copy()
        self.original_masks = masks.copy()
        self.canonical = np.flatnonzero(masks.any(axis=0))
        self.node_ids, compact = np.unique(edges[self.canonical].ravel(), return_inverse=True)
        self.x = torch.as_tensor(nodes[self.node_ids].copy(), device=self.dev)
        self.edges = torch.as_tensor(compact.reshape(-1, 2), device=self.dev)
        self.active = torch.as_tensor(masks[:, self.canonical].copy(), device=self.dev)
        self.canonical_device = torch.as_tensor(self.canonical, device=self.dev)
        self.c2 = torch.empty(len(self.canonical), dtype=torch.float64, device=self.dev)
        self.norms = torch.empty(len(self.node_ids), dtype=torch.float64, device=self.dev)
        with torch.inference_mode():
            for r in range(0, len(self.x), edge_chunk):
                value = self.x[r:r+edge_chunk].double()
                self.norms[r:r+len(value)] = (value * value).sum(-1)
            for r in range(0, len(self.edges), edge_chunk):
                es = self.edges[r:r+edge_chunk]
                delta = self.x[es[:, 1]].double() - self.x[es[:, 0]].double()
                self.c2[r:r+len(es)] = (delta * delta).sum(-1)
            if (self.c2 <= 0).any().item():
                raise ValueError('active zero-length edges must be removed')
        self.dot_gamma = gamma(nodes.shape[1] + 8, np.finfo(np.float32).eps)
        self.sum_gamma = gamma(nodes.shape[1] + 8, np.finfo(np.float64).eps)

    def endpoint(self, q, qnorm, first, count):
        """FP32 GEMM with FP64 norms and an absolute forward-error interval.

        Cauchy bounds sum(abs(q*r)). FP32 overflow is not treated as a valid
        distance: its interval becomes [0,+inf], forcing direct FP64 checking.
        The tiny term covers underflow/flush-to-zero absolute dot error.
        """
        torch = self.torch
        norms = self.norms[first:first+count]
        cross = (q @ self.x[first:first+count].T).double()
        magnitude = qnorm[:, None] + norms[None, :]
        product = qnorm[:, None].sqrt() * norms[None, :].sqrt()
        distance = magnitude - 2 * cross
        error = (4 * self.dot_gamma * product + 8 * self.sum_gamma * magnitude
                 + 4 * q.shape[1] * np.finfo(np.float32).tiny)
        usable = torch.isfinite(distance) & torch.isfinite(error)
        distance = torch.where(usable, distance, 0.)
        error = torch.where(usable, error, torch.inf)
        return distance, error


def _segment_interval(torch, endpoints, bounds, edges, c2, sum_gamma):
    aa, bb = endpoints[:, edges[:, 0]], endpoints[:, edges[:, 1]]
    alpha = ((aa + c2 - bb) / (2 * c2)).clamp(0, 1)
    square = (1-alpha)*aa + alpha*bb - alpha*(1-alpha)*c2
    # The minimum over alpha of the two perturbed quadratics differs by at
    # most max(endpoint errors) + .25*length error. Rounding is inflated here;
    # only scalar FP64 arithmetic remains after the FP32 dot products.
    error = (torch.maximum(bounds[:, edges[:, 0]], bounds[:, edges[:, 1]])
             + 4 * sum_gamma * c2 + 64 * np.finfo(np.float64).eps *
             (aa.abs() + bb.abs() + c2))
    return (square-error).clamp_min(0), (square+error).clamp_min(0)


def scan_segments(query, nodes, edges, masks, union, *, device='cpu',
                  query_chunk=128, edge_chunk=2048, reference_chunk=2048,
                  progress=None, prepared=None, refine_chunk=2048):
    started = time.perf_counter()
    import torch
    query, nodes = np.asarray(query), np.asarray(nodes)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    masks = np.asarray(masks, bool)
    union = np.asarray(union)
    if (query.dtype != np.float32 or query.ndim != 2 or union.dtype != np.float32
            or union.shape != (len(query),) or query.shape[1] != nodes.shape[1]
            or not np.isfinite(query).all() or not np.isfinite(union).all() or (union < 0).any()):
        raise ValueError('original finite FP32 query and nonnegative union required')
    if masks.ndim != 2 or masks.shape[1] != len(edges):
        raise ValueError('mask shape differs')
    if min(query_chunk, edge_chunk, reference_chunk, refine_chunk) < 1:
        raise ValueError('positive numerical blocks required')
    p = prepared or PreparedSearch(nodes, edges, masks, device, edge_chunk)
    if (p.device != device or p.original_nodes is not nodes or
            not np.array_equal(p.original_edges, edges) or not np.array_equal(p.original_masks, masks)):
        raise ValueError('prepared search belongs to different graph/masks')
    result = np.broadcast_to(union, (len(masks), len(query))).copy()
    winners = np.full(result.shape, -1, np.int64)
    refined = 0
    diagnostics = dict(backend=ENGINE, original_edges=len(edges), active_edges=len(p.canonical),
                       original_nodes=len(nodes), active_nodes=len(p.node_ids),
                       endpoint_pairs=len(query)*len(p.node_ids), direct_pairs=0,
                       screen_passes=2, screen_dtype='FP32_no_TF32', final_dtype='FP64_to_original_FP32')
    if not len(p.canonical):
        diagnostics['seconds'] = time.perf_counter()-started
        return result, winners, diagnostics
    with torch.inference_mode(), strict_screen(torch, p.dev):
        for first in range(0, len(query), query_chunk):
            stop = min(first+query_chunk, len(query))
            q = torch.as_tensor(query[first:stop], device=p.dev)
            q64 = q.double()
            qnorm = (q64*q64).sum(-1)
            endpoint = torch.empty((len(q), len(p.x)), dtype=torch.float64, device=p.dev)
            bounds = torch.empty_like(endpoint)
            for r in range(0, len(p.x), reference_chunk):
                count = min(reference_chunk, len(p.x)-r)
                d, error = p.endpoint(q, qnorm, r, count)
                endpoint[:, r:r+count], bounds[:, r:r+count] = d, error
            initial = torch.as_tensor((union[first:stop].astype(np.float64)*1280)**2, device=p.dev)
            best = initial[None].expand(len(masks), -1).clone()
            upper = best.clone()
            chosen = torch.full(best.shape, -1, dtype=torch.int64, device=p.dev)
            # First pass: a global upper bound for each query/branch, rather
            # than refining a temporary winner in every edge block.
            for r in range(0, len(p.edges), edge_chunk):
                es, cc = p.edges[r:r+edge_chunk], p.c2[r:r+edge_chunk]
                _, hi = _segment_interval(torch, endpoint, bounds, es, cc, p.sum_gamma)
                for k in range(len(masks)):
                    bound = torch.where(p.active[k, r:r+len(es)], hi, torch.inf).min(1).values
                    upper[k] = torch.minimum(upper[k], bound)
            # Second pass: all possible global winners, no arbitrary Top-k.
            for r in range(0, len(p.edges), edge_chunk):
                es, cc = p.edges[r:r+edge_chunk], p.c2[r:r+edge_chunk]
                lo, _ = _segment_interval(torch, endpoint, bounds, es, cc, p.sum_gamma)
                needed = torch.zeros_like(lo, dtype=torch.bool)
                for k in range(len(masks)):
                    needed |= p.active[k, r:r+len(es)] & (lo <= upper[k, :, None])
                coords = torch.nonzero(needed)
                exact = torch.full_like(lo, torch.inf)
                for z in range(0, len(coords), refine_chunk):
                    qi, ei = coords[z:z+refine_chunk].unbind(1)
                    u = p.x[es[ei, 0]].double()
                    direction = p.x[es[ei, 1]].double() - u
                    delta = q64[qi] - u
                    fraction = ((delta*direction).sum(-1)/(direction*direction).sum(-1)).clamp(0, 1)
                    residual = delta-fraction[:, None]*direction
                    exact[qi, ei] = (residual*residual).sum(-1)
                refined += len(coords)
                for k in range(len(masks)):
                    distance, idx = torch.where(p.active[k, r:r+len(es)], exact, torch.inf).min(1)
                    improve = distance < best[k]
                    best[k] = torch.where(improve, distance, best[k])
                    chosen[k] = torch.where(improve, p.canonical_device[r+idx], chosen[k])
            result[:, first:stop] = np.minimum(result[:, first:stop], (best.sqrt()/1280).float().cpu().numpy())
            winners[:, first:stop] = chosen.cpu().numpy()
            if progress:
                progress(stop, len(query))
    diagnostics['direct_pairs'] = refined
    diagnostics['seconds'] = time.perf_counter()-started
    return result, winners, diagnostics
