"""Batched FP64 witness geometry and array proofs, without per-edge JSON/SQL.

NumPy CPU and lazy Torch CPU/CUDA execute the same candidate formula. Only
roundoff-ambiguous boundary edges use the original scalar NumPy calculation.
No query features, labels or GT are inputs to this module.
"""
from pathlib import Path
import time
import numpy as np
from . import storage as st
from .method import witness

FORMAT = 'batched_witness_array_proofs_v1'
POINT_REASONS = ('padding', 'nonfinite_witness', 'nonfinite_alpha',
                 'outside_internal_interval', 'zero_endpoint_denominator',
                 'nonfinite_denominator', 'nonfinite_rho', 'too_far_from_segment', 'accepted')
EDGE_REASONS = ('no_qualified_third_image', 'accepted', 'nonfinite_endpoint',
                'zero_length_connection', 'nonfinite_endpoint_geometry')


def canonical_edges(edges):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 4).copy()
    if len(edges):
        swap = edges[:, 0] > edges[:, 2]
        edges[swap] = edges[swap][:, [2, 3, 0, 1]]
        if np.any(edges[:, 0] == edges[:, 2]) or np.any(edges[:, [1, 3]] < 0):
            raise ValueError('distinct endpoint supports and nonnegative rows required')
    return edges


def _geometry(u, v, w, xp):
    """[edges, candidates, dimension]; no matrix identity or precision reduction."""
    direction = v-u
    wu = w-u[:, None, :]
    if xp is np:
        squared = lambda a: np.einsum('...d,...d->...',a,a)
        c2 = squared(direction)
        alpha = np.einsum('nkd,nd->nk',wu,direction)/c2[:, None]
    else:
        squared = lambda a: xp.sum(a*a,axis=-1)
        c2 = squared(direction)
        alpha = xp.sum(wu*direction[:, None, :], axis=-1)/c2[:, None]
    z = u[:, None, :]+alpha[:, :, None]*direction[:, None, :]
    base = xp.minimum(xp.sqrt(squared(wu)),xp.sqrt(squared(w-v[:, None, :])))
    rho = xp.sqrt(squared(w-z))/base
    finite_w = xp.isfinite(w).all(axis=-1)
    return alpha, rho, base, finite_w


class Engine:
    """One domain/layer per process. Banks reside on the selected device once.

    Unique edges are cached in RAM. Completed array-proof chunks also support
    restart, with one artifact hash per chunk instead of one JSON hash per edge.
    """
    def __init__(self, banks, neighbors, *, device='cpu', batch_size=256,
                 path=None, identity='', neighbor_identity=''):
        if batch_size < 1:
            raise ValueError('positive witness batch size required')
        self.banks, self.neighbors = banks, neighbors
        self.device, self.batch_size = device, batch_size
        self.seconds = 0.; self.new = 0; self.reused = 0; self.scalar_fallback_edges = 0
        self.geometry_seconds = 0.; self.proof_write_seconds = 0.; self.cache_read_seconds = 0.
        self.memory = {}; self.path = Path(path) if path is not None else None
        self.identity = identity
        self.config = dict(format=FORMAT, neighbors=neighbor_identity,
                           backend='numpy_fp64' if device == 'cpu' else 'torch_fp64',
                           tensor_device_type='cpu' if device in ('cpu', 'torch:cpu') else 'cuda',
                           batch_size=batch_size, point_reasons=POINT_REASONS,
                           edge_reasons=EDGE_REASONS, boundary_policy='scalar_original_8_dimension_eps_v1')
        self.torch = None
        self.offsets = {}; off = 0
        for c, bank in sorted(banks.items()):
            self.offsets[c] = off; off += len(bank)
        all_banks = np.concatenate([banks[c] for c in sorted(banks)])
        transfer_tick = time.perf_counter()
        if device == 'cpu':
            self.values = all_banks.astype(np.float64)
        else:
            import torch
            self.torch = torch
            dev = torch.device('cpu' if device == 'torch:cpu' else device)
            if dev.type == 'cuda':
                torch.cuda.set_device(dev)
            self.values = torch.as_tensor(all_banks, dtype=torch.float64, device=dev)
        self.bank_transfer_seconds = time.perf_counter()-transfer_tick
        self.sequence = 0
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            cfg = self.path/'config.json'
            # JSON round-trip normalizes tuples; no file hashes in the hot loop.
            config = {**self.config, 'point_reasons': list(POINT_REASONS), 'edge_reasons': list(EDGE_REASONS)}
            if cfg.exists() and st.read_json(cfg) != config:
                raise ValueError('witness execution configuration differs; use a fresh output')
            if not cfg.exists(): st.write_json(cfg, config)
            read_tick = time.perf_counter()
            for record_path in self.path.glob('batch_*.json'):
                if not record_path.with_suffix('.npz').is_file():
                    raise ValueError('incomplete witness array proof preserved')
            for p in sorted(self.path.glob('batch_*.npz')):
                record = st.restore(p, identity)
                if record['metadata'] != dict(format=FORMAT, neighbors=neighbor_identity):
                    raise ValueError('incompatible witness array proof')
                with np.load(p, allow_pickle=False) as z:
                    self._remember(z['edges'], tuple(z['excluded'].tolist()), z['source_mask'], z['edge_reason'])
                self.sequence = max(self.sequence, int(p.stem.split('_')[1])+1)
            self.cache_read_seconds = time.perf_counter()-read_tick

    def _remember(self, edges, excluded, masks, reasons):
        for e, mask, reason in zip(edges, masks, reasons):
            key = (excluded, tuple(map(int, e))); value = (int(mask), int(reason))
            if key in self.memory and self.memory[key] != value:
                raise ValueError('conflicting frozen witness proof')
            self.memory[key] = value

    def _to_numpy(self, x):
        return x if self.torch is None else x.cpu().numpy()

    def _index(self, ids):
        return self.values[ids] if self.torch is None else self.values[self.torch.as_tensor(ids, device=self.values.device)]

    def compute(self, edges, excluded=()):
        edges = canonical_edges(edges); excluded = tuple(sorted(set(map(int, excluded))))
        n = len(edges); ids = sorted(self.banks)
        if any(c < 1 or c > 8 for c in ids): raise ValueError('original support IDs 1..8 required')
        for c in ids:
            for side in (0, 2):
                take = edges[:, side] == c
                if np.any(edges[take, side+1] >= len(self.banks[c])):
                    raise ValueError('endpoint outside original support rows')
        if np.any(np.isin(edges[:, [0, 2]], excluded)) or not np.isin(edges[:, [0, 2]], ids).all():
            raise ValueError('held-out or unknown endpoint support')
        result = dict(edges=edges, excluded=np.asarray(excluded, np.int64),
                      candidates=np.full((n, 8, 10), -1, np.int32),
                      alpha=np.full((n, 8, 10), np.nan), rho=np.full((n, 8, 10), np.nan),
                      point_reason=np.zeros((n, 8, 10), np.uint8),
                      source_mask=np.zeros(n, np.uint8), edge_reason=np.zeros(n, np.uint8))
        if not n: return result
        global_ids = [np.asarray([self.offsets[int(c)] for c in edges[:, s]])+edges[:, s+1] for s in (0, 2)]
        u, v = (self._index(i) for i in global_ids)
        u_cpu = np.stack([self.banks[int(a)][i] for a, i in edges[:, :2]]).astype(np.float64)
        v_cpu = np.stack([self.banks[int(b)][j] for b, j in edges[:, 2:]]).astype(np.float64)
        finite_endpoints = np.isfinite(u_cpu).all(1) & np.isfinite(v_cpu).all(1)
        with np.errstate(all='ignore'):
            c2 = np.sum((v_cpu-u_cpu)**2, axis=1)
        result['edge_reason'][~finite_endpoints] = 2
        result['edge_reason'][finite_endpoints & (c2 == 0)] = 3
        result['edge_reason'][finite_endpoints & ~np.isfinite(c2)] = 4
        valid_edge = result['edge_reason'] == 0
        ambiguous = np.zeros(n, bool)
        for c in ids:
            take = np.flatnonzero(valid_edge & (edges[:, 0] != c) & (edges[:, 2] != c) & (c not in excluded))
            if not len(take): continue
            # Gather the two per-image k5 lists; sort/unique inside each edge.
            candidates = np.empty((len(take), 10), np.int64)
            for a in ids:
                group = edges[take, 0] == a
                if group.any(): candidates[group, :5] = self.neighbors[a, c][edges[take[group], 1]]
                group = edges[take, 2] == a
                if group.any(): candidates[group, 5:] = self.neighbors[a, c][edges[take[group], 3]]
            candidates.sort(axis=1)
            distinct = np.ones(candidates.shape, bool); distinct[:, 1:] = candidates[:, 1:] != candidates[:, :-1]
            result['candidates'][take, c-1] = np.where(distinct, candidates, -1)
            w = self._index(candidates+self.offsets[c])
            with np.errstate(all='ignore'):
                if self.torch is None:
                    values = _geometry(u[take], v[take], w, np)
                else:
                    with self.torch.inference_mode(): values = _geometry(u[take], v[take], w, self.torch)
            alpha, rho, base, finite_w = map(self._to_numpy, values)
            inside = np.isfinite(alpha) & (alpha >= .25) & (alpha <= .75)
            reason = np.where(~finite_w, 1, np.where(~np.isfinite(alpha), 2,
                np.where(~inside, 3, np.where(base == 0, 4, np.where(~np.isfinite(base), 5,
                np.where(~np.isfinite(rho), 6, np.where(rho > .5, 7, 8))))))).astype(np.uint8)
            reason[~distinct] = 0
            result['alpha'][take, c-1] = np.where(distinct & finite_w & np.isfinite(alpha), alpha, np.nan)
            result['rho'][take, c-1] = np.where(distinct & inside & (base > 0) & np.isfinite(rho), rho, np.nan)
            result['point_reason'][take, c-1] = reason
            result['source_mask'][take] |= ((reason == 8).any(1).astype(np.uint8) << (c-1))
            # This is NOT a relaxed acceptance interval. Close comparisons are
            # recalculated by the exact original scalar function below.
            guard = 8*u_cpu.shape[1]*np.finfo(np.float64).eps
            near_alpha = np.minimum(np.abs(alpha-.25), np.abs(alpha-.75)) <= guard*np.maximum(1, np.abs(alpha))
            near_rho = inside & (np.abs(rho-.5) <= guard*np.maximum(1, np.abs(rho)))
            ambiguous[take] |= ((near_alpha | near_rho) & distinct).any(1)
        result['edge_reason'][valid_edge & (result['source_mask'] != 0)] = 1
        for row in np.flatnonzero(ambiguous):
            self._scalar(result, row, excluded)
        self.scalar_fallback_edges += int(ambiguous.sum())
        return result

    def _scalar(self, result, row, excluded):
        edge = tuple(map(int, result['edges'][row])); proof = witness(self.banks, self.neighbors, edge, excluded)
        result['source_mask'][row] = sum(1 << (c-1) for c in proof['sources'])
        result['edge_reason'][row] = EDGE_REASONS.index(proof['reason'])
        a, i, b, j = edge; u = np.asarray(self.banks[a][i], np.float64); v = np.asarray(self.banks[b][j], np.float64)
        direction = v-u; c2 = float(direction@direction)
        for c in self.banks:
            for k, idx in enumerate(result['candidates'][row, c-1]):
                if idx < 0: continue
                w = np.asarray(self.banks[c][idx], np.float64); alpha = float((w-u)@direction/c2)
                rho = np.nan; reason = 3
                if not np.isfinite(w).all(): reason = 1
                elif not np.isfinite(alpha): reason = 2
                elif .25 <= alpha <= .75:
                    base = float(min(np.linalg.norm(w-u), np.linalg.norm(w-v)))
                    if base == 0: reason = 4
                    elif not np.isfinite(base): reason = 5
                    else:
                        rho = float(np.linalg.norm(w-(u+alpha*direction))/base)
                        reason = 6 if not np.isfinite(rho) else 8 if rho <= .5 else 7
                result['alpha'][row, c-1, k] = alpha
                result['rho'][row, c-1, k] = rho
                result['point_reason'][row, c-1, k] = reason

    def get_many(self, edges, excluded=()):
        tick = time.perf_counter(); edges = canonical_edges(edges)
        excluded = tuple(sorted(set(map(int, excluded))))
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        keys = [(excluded, tuple(map(int, e))) for e in unique]
        missing = [i for i, k in enumerate(keys) if k not in self.memory]
        self.reused += len(unique)-len(missing)
        for start in range(0, len(missing), self.batch_size):
            selected = unique[missing[start:start+self.batch_size]]
            compute_tick = time.perf_counter()
            proof = self.compute(selected, excluded)
            self.geometry_seconds += time.perf_counter()-compute_tick
            if self.path is not None:
                write_tick = time.perf_counter()
                p = self.path/f'batch_{self.sequence:08d}.npz'
                st.seal(p, self.identity, proof, dict(format=FORMAT, neighbors=self.config['neighbors']))
                self.sequence += 1
                self.proof_write_seconds += time.perf_counter()-write_tick
            self._remember(selected, excluded, proof['source_mask'], proof['edge_reason'])
            self.new += len(selected)
        masks, reasons = np.asarray([self.memory[k] for k in keys], np.uint8).reshape(-1, 2).T
        self.seconds += time.perf_counter()-tick
        return masks[inverse], reasons[inverse], dict(edges=unique, source_mask=masks, edge_reason=reasons)

    def close(self): pass
    def timing(self):
        return dict(bank_transfer_seconds=self.bank_transfer_seconds,cache_read_seconds=self.cache_read_seconds,
                    batched_geometry_seconds=self.geometry_seconds,proof_write_seconds=self.proof_write_seconds,
                    geometry_scope='CPU candidate preparation + device FP64 geometry + transfers + scalar boundary fallback')
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
