"""Position-wise mean-preserving projection; NumPy only, no dense projectors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

LAYERS = (7, 15, 23, 31)
DIMENSION = 1280
MAX_RANK = 7
POSITION_CHUNK = 64
BRANCHES = ('SOURCE', 'UNION_K8', 'SPARC_SOURCE_K8')
RANK_RULE = 's > max(8,dimension)*eps(float64)*s_max; rank<=7'


def physical_positions(positions, grid):
    p = np.asarray(positions)
    n = int(np.prod(grid))
    if (len(grid) != 2 or min(grid) <= 0 or p.dtype != np.int64 or p.shape != (n,)
            or not np.array_equal(np.sort(p), np.arange(n))):
        raise ValueError('a complete bijective int64 physical positions map is required')
    return p


def fit_cells(supports):
    """[positions,8,dimension] -> float64 mean, padded basis, empirical ranks."""
    x = np.asarray(supports, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != 8 or not np.isfinite(x).all():
        raise ValueError('fit requires eight finite same-position support vectors')
    mu = x.mean(axis=1)
    centered = x - mu[:, None, :]
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    tol = max(8, x.shape[2]) * np.finfo(np.float64).eps * singular[:, :1]
    rank = np.minimum((singular > tol).sum(axis=1), min(MAX_RANK, x.shape[2]))
    width = min(MAX_RANK, x.shape[2])
    basis = vh[:, :width].transpose(0, 2, 1).copy()
    basis *= (np.arange(width)[None, :] < rank[:, None])[:, None, :]
    return mu, basis, rank.astype(np.int8)


@dataclass
class Projection:
    grid: tuple[int, int]
    mu: np.ndarray
    basis: np.ndarray
    rank: np.ndarray

    def close(self):
        for value in (self.mu,self.basis,self.rank):
            if isinstance(value,np.memmap): value._mmap.close()

    def correct(self, features, positions, grid, *, chunk=POSITION_CHUNK):
        if tuple(grid) != tuple(self.grid):
            raise ValueError(f'incompatible native grid: {grid} != {self.grid}; no projector resizing')
        p = physical_positions(positions, grid)
        f = np.asarray(features)
        if (f.shape != self.mu.shape or f.dtype != np.float32 or not np.isfinite(f).all()
                or chunk <= 0):
            raise ValueError('correction requires finite float32 compact [full_grid,dimension] features')
        out = np.empty_like(f)
        for start in range(0, len(p), chunk):
            stop = min(start + chunk, len(p)); ids = p[start:stop]
            value = f[start:stop].astype(np.float64)
            v = self.basis[ids]
            coeff = np.einsum('ndr,nd->nr', v, value - self.mu[ids])
            value -= np.einsum('ndr,nr->nd', v, coeff)
            out[start:stop] = value.astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError('nonfinite corrected features')
        return out


def fit_layer(banks, grid, root: Path):
    """Persist only mu/V/rank, filling disk arrays a bounded position block at a time."""
    n = int(np.prod(grid))
    if (len(banks) != 8 or any(b.shape != (n, DIMENSION) or b.dtype != np.float32
                              or not np.isfinite(b).all() for b in banks)):
        raise ValueError('support feature shape/dtype/finiteness differs from fixed native grid')
    root.mkdir(parents=True, exist_ok=False)
    mu = np.lib.format.open_memmap(root/'mu.npy', mode='w+', dtype=np.float64, shape=(n,DIMENSION))
    v = np.lib.format.open_memmap(root/'basis.npy', mode='w+', dtype=np.float64, shape=(n,DIMENSION,MAX_RANK))
    ranks = np.lib.format.open_memmap(root/'rank.npy', mode='w+', dtype=np.int8, shape=(n,))
    ortho_error = 0.; collapse_error = 0.
    for start in range(0,n,POSITION_CHUNK):
        stop = min(start+POSITION_CHUNK,n)
        x = np.stack([b[start:stop] for b in banks],axis=1)
        means,basis,rank = fit_cells(x)
        mu[start:stop],v[start:stop],ranks[start:stop] = means,basis,rank
        gram = np.einsum('ndr,nds->nrs',basis,basis)
        expected = np.zeros_like(gram)
        for j in range(MAX_RANK): expected[:,j,j] = rank>j
        ortho_error = max(ortho_error,float(np.max(np.abs(gram-expected))))
        diff = x.astype(np.float64)-means[:,None,:]
        residual = diff-np.einsum('ndr,nkr->nkd',basis,np.einsum('ndr,nkd->nkr',basis,diff))
        collapse_error = max(collapse_error,float(np.max(np.abs(residual))))
        if not np.allclose(residual,0,atol=1e-7,rtol=0) or ortho_error>1e-10:
            raise ValueError('per-cell projection failed orthogonality/support-collapse audit')
    stats = dict(rank_rule=RANK_RULE,grid=list(grid),rank_counts=np.bincount(ranks,minlength=8).tolist(),
                 max_orthogonality_error=ortho_error,max_support_collapse_abs_error=collapse_error)
    mu.flush();v.flush();ranks.flush()
    del mu,v,ranks
    return stats


def load_projection(root: Path, grid):
    n = int(np.prod(grid))
    mu=np.load(root/'mu.npy',mmap_mode='r',allow_pickle=False)
    v=np.load(root/'basis.npy',mmap_mode='r',allow_pickle=False)
    rank=np.load(root/'rank.npy',mmap_mode='r',allow_pickle=False)
    if (mu.shape!=(n,DIMENSION) or mu.dtype!=np.float64 or v.shape!=(n,DIMENSION,MAX_RANK)
            or v.dtype!=np.float64 or rank.shape!=(n,) or rank.dtype!=np.int8
            or (rank<0).any() or (rank>MAX_RANK).any()):
        raise ValueError('invalid projection state shape/dtype/rank')
    return Projection(tuple(grid),mu,v,rank)
