"""The fixed probability model. NumPy CPU reference and optional stock Torch FP32.

No custom GPU kernels, clipping of negative z, probability epsilon, or GT access.
Arrays use physical position order. Blocks bound Q x R x 4 temporary storage.
"""
import hashlib
import json
import numpy as np
from scipy.special import logsumexp
from .protocol import JOINT, INDEPENDENT, UNION4, CAPACITY, heldout_indices
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import bilinear


def positions_check(positions, grid):
    p = np.asarray(positions)
    if p.dtype != np.int64 or p.shape != (int(np.prod(grid)),) or not np.array_equal(np.sort(p), np.arange(len(p))):
        raise ValueError('complete int64 physical mapping required')
    return p


def physical(value, positions, grid):
    positions_check(positions, grid)
    value = np.asarray(value)
    if value.shape[-1] != len(positions):
        raise ValueError('compact position count differs')
    result = np.empty_like(value)
    result[..., positions] = value
    return result


def sample_positions(relative_path, grid, count=512):
    """Uniform hash-rank sampling, independent of distances, labels, and order."""
    n = int(np.prod(grid))
    if n < 1 or count < 1:
        raise ValueError('nonempty grid and positive sample count required')
    def key(i):
        blob = json.dumps(['joint_normal_positions_v1', relative_path, i // grid[1], i % grid[1]],
                          ensure_ascii=False, separators=(',', ':')).encode()
        return hashlib.sha256(blob).digest(), i
    return np.asarray(sorted(sorted(range(n), key=key)[:min(n, count)]), np.int64)


def split_check(ds, dt, positions, grid, count=8):
    positions_check(positions, grid)
    ds, dt = np.asarray(ds), np.asarray(dt)
    n = len(positions)
    if ds.shape != (4, n) or dt.shape != (4, count, n) or ds.dtype != np.float32 or dt.dtype != np.float32:
        raise ValueError('original scaled FP32 Source[4,M], Target[4,K,M] required')
    if not np.isfinite(ds).all() or not np.isfinite(dt).all() or (ds < 0).any() or (dt < 0).any():
        raise ValueError('nonfinite/negative original distances')


def seven_targets(dt):
    if dt.ndim != 3 or dt.shape[:2] != (4, 8):
        raise ValueError('four layers and this group of eight supports required')
    return np.stack([dt[:, heldout_indices(j)].min(1) for j in range(8)])


def normalized(ds, dt, patch):
    # Distances already contain the ORIGINAL /1280. Do not divide again.
    return z_values(ds, patch), np.stack([z_values(x, patch) for x in seven_targets(dt)])


def score_native(zs, zt7, references, source_weights, *, device='cpu', query_chunk=256, reference_chunk=128):
    """references[j]=(Rs[Nj,4], Rt[Nj,4]), equal image weight even if Nj differ."""
    zs, zt7 = np.asarray(zs, np.float32), np.asarray(zt7, np.float32)
    if zs.ndim != 2 or zs.shape[0] != 4 or zt7.shape != (8, *zs.shape) or len(references) != 8:
        raise ValueError('four physical layers, eight K7 branches and eight normal references required')
    w = np.asarray(source_weights, np.float32)
    from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.numerics import WEIGHT_SUM_ATOL
    if w.shape != (4,) or not np.isfinite(w).all() or (w <= 0).any() or abs(float(w.sum())-1) > WEIGHT_SUM_ATOL:
        raise ValueError('original normalized Source u weights required')
    if query_chunk < 1 or reference_chunk < 1 or not np.isfinite(zs).all() or not np.isfinite(zt7).all():
        raise ValueError('invalid finite query or block size')
    refs = []
    for rs, rt in references:
        rs, rt = np.asarray(rs, np.float32), np.asarray(rt, np.float32)
        if rs.ndim != 2 or rs.shape[1] != 4 or rt.shape != rs.shape or not len(rs) or not np.isfinite(rs).all() or not np.isfinite(rt).all():
            raise ValueError('paired finite normal response vectors required')
        refs.append((rs, rt))
    if device != 'cpu':
        logs = _torch(zs, zt7, refs, device, query_chunk, reference_chunk)
    else:
        logs = _numpy(zs, zt7, refs, query_chunk, reference_chunk)
    result = {b: np.asarray(-logs[i] / np.float32(divisor), np.float32) for i, (b, divisor) in enumerate(((JOINT, 8), (INDEPENDENT, 8), (UNION4, 4)))}
    mean_union = np.minimum(zs[None], zt7).mean(0, dtype=np.float32)
    result[CAPACITY] = mean_union.mean(0, dtype=np.float32) if np.array_equal(w, np.full(4, .25, np.float32)) else np.sum(mean_union*w[:, None], axis=0, dtype=np.float32)
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError('nonfinite tail model output; no epsilon substitution')
    return result


def _numpy(zs, zt, refs, qc, rc):
    out = np.empty((3, zs.shape[1]), np.float32)
    for lo in range(0, zs.shape[1], qc):
        vs = zs[:, lo:lo+qc].T
        group = []
        for j, (rs, rt) in enumerate(refs):
            vt = zt[j, :, lo:lo+qc].T
            sums = np.full((4, len(vs)), -np.inf, np.float32)
            for r in range(0, len(rs), rc):
                # A and B are shared by joint and analytic independent controls.
                a = -np.logaddexp(np.float32(0), vs[:, None]-rs[None, r:r+rc]).sum(-1, dtype=np.float32)
                b = -np.logaddexp(np.float32(0), vt[:, None]-rt[None, r:r+rc]).sum(-1, dtype=np.float32)
                u = -np.logaddexp(np.float32(0), np.minimum(vs, vt)[:, None]-np.minimum(rs, rt)[None, r:r+rc]).sum(-1, dtype=np.float32)
                values = np.stack([logsumexp(a+b, axis=1), logsumexp(a, axis=1), logsumexp(b, axis=1), logsumexp(u, axis=1)])
                sums = np.logaddexp(sums, values)
            sums -= np.float32(np.log(len(rs)))
            group.append(np.stack([sums[0], sums[1]+sums[2], sums[3]]))
        out[:, lo:lo+qc] = logsumexp(np.stack(group), axis=0)-np.float32(np.log(8))
    return out


def _torch(zs, zt, refs, device, qc, rc):
    import torch
    import torch.nn.functional as F
    # Stock tensor kernels only; FP32, no autocast or matrix precision change.
    with torch.inference_mode():
        s, t = torch.tensor(zs.T, device=device), torch.tensor(zt.transpose(0, 2, 1), device=device)
        # Batch the eight reference images as an explicit axis, rather than
        # launching the same tiny pointwise chain eight times per query block.
        # Padding is masked in log probability space, preserving per-image Nj.
        counts = np.asarray([len(rs) for rs, _ in refs], np.int64)
        maximum = int(counts.max());rs_array = np.zeros((8, maximum, 4), np.float32);rt_array = np.zeros_like(rs_array)
        valid_array = np.arange(maximum)[None] < counts[:, None]
        for j, (rs, rt) in enumerate(refs):
            rs_array[j, :len(rs)] = rs;rt_array[j, :len(rt)] = rt
        rs = torch.tensor(rs_array, device=device);rt = torch.tensor(rt_array, device=device)
        ru = torch.minimum(rs, rt);valid = torch.tensor(valid_array, device=device)
        log_counts = torch.tensor(np.log(counts).astype(np.float32), device=device)[None, :, None]
        out = torch.empty((3, len(s)), device=device, dtype=torch.float32)
        for lo in range(0, len(s), qc):
            vs = s[lo:lo+qc];vt = t[:, lo:lo+qc];vu = torch.minimum(vs[None], vt)
            sums = torch.full((4, 8, len(vs)), -float('inf'), device=device, dtype=torch.float32)
            for r in range(0, maximum, rc):
                mask = valid[:, None, r:r+rc]
                a = F.logsigmoid(rs[:, None, r:r+rc]-vs[None, :, None]).sum(-1).masked_fill(~mask, -float('inf'))
                b = F.logsigmoid(rt[:, None, r:r+rc]-vt[:, :, None]).sum(-1).masked_fill(~mask, -float('inf'))
                u = F.logsigmoid(ru[:, None, r:r+rc]-vu[:, :, None]).sum(-1).masked_fill(~mask, -float('inf'))
                value = torch.stack([torch.logsumexp(a+b, 2), torch.logsumexp(a, 2), torch.logsumexp(b, 2), torch.logsumexp(u, 2)])
                sums = torch.logaddexp(sums, value)
            sums -= log_counts
            group = torch.stack([sums[0], sums[1]+sums[2], sums[3]])
            out[:, lo:lo+qc] = torch.logsumexp(group, 1)-float(np.log(8))
        return out.cpu().numpy()


def maps(native, grid, shape):
    return {b: bilinear(x.reshape(grid), shape) for b, x in native.items()}
