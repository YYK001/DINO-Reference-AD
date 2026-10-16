"""Deterministic vectorized priority selection for ShiftADD v2."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .protocol import FEATURE_DIM, SAMPLER_REVISION, SEED


UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
LAYER_CONSTANTS = {
    7: np.uint64(0x243F6A8885A308D3),
    15: np.uint64(0x13198A2E03707344),
    23: np.uint64(0xA4093822299F31D0),
    31: np.uint64(0x082EFA98EC4E6C89),
}
BRANCH_CONSTANTS = {
    "OFFICIAL_ANCHOR": np.uint64(0x452821E638D01377),
    "ILLUMINATION": np.uint64(0xBE5466CF34E90C6C),
    "POSE": np.uint64(0xC0AC29B7C97C50DD),
    "SCENE_VALID": np.uint64(0x3F84D5B5B5470917),
    "BRIGHTNESS2": np.uint64(0x9216D5D98979FB1B),
}


def first64_sha256(*parts: object) -> np.uint64:
    payload = "\n".join(str(part) for part in parts).encode("utf-8")
    return np.uint64(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def path_priority_seed(relative_path: str, *, seed: int = SEED) -> np.uint64:
    return first64_sha256(SAMPLER_REVISION, int(seed), str(relative_path))


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 with deliberate uint64 wraparound."""
    value = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        value = (value + np.uint64(0x9E3779B97F4A7C15)) & UINT64_MASK
        value = ((value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & UINT64_MASK
        value = ((value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & UINT64_MASK
    return value ^ (value >> np.uint64(31))


def patch_priorities(
    relative_path: str,
    layer: int,
    branch: str,
    patch_count: int,
    *,
    seed: int = SEED,
) -> np.ndarray:
    return patch_priorities_from_seed(
        path_priority_seed(relative_path, seed=seed), layer, branch, patch_count,
    )


def patch_priorities_from_seed(
    path_seed: np.uint64,
    layer: int,
    branch: str,
    patch_count: int,
) -> np.ndarray:
    """Vectorize layer/branch patch keys from one already-computed path SHA."""
    if layer not in LAYER_CONSTANTS or branch not in BRANCH_CONSTANTS:
        raise ValueError(f"unsupported priority identity: layer={layer}, branch={branch}")
    patch_index = np.arange(int(patch_count), dtype=np.uint64)
    base = (
        np.uint64(path_seed)
        ^ LAYER_CONSTANTS[layer]
        ^ BRANCH_CONSTANTS[branch]
    )
    return splitmix64(base ^ patch_index)


def anchor_priorities(layer: int, count: int = 100_000, *, seed: int = SEED) -> np.ndarray:
    if layer not in LAYER_CONSTANTS:
        raise ValueError(f"unsupported anchor layer: {layer}")
    base = first64_sha256(SAMPLER_REVISION, int(seed), "official_anchor", int(layer))
    indices = np.arange(int(count), dtype=np.uint64)
    return splitmix64(base ^ LAYER_CONSTANTS[layer] ^ BRANCH_CONSTANTS["OFFICIAL_ANCHOR"] ^ indices)


def priority_topk_indices(
    priorities: np.ndarray,
    source_indices: np.ndarray,
    patch_indices: np.ndarray,
    quota: int,
) -> np.ndarray:
    """Select stable smallest keys; collisions use source then patch index."""
    keys = np.asarray(priorities, dtype=np.uint64).reshape(-1)
    source = np.asarray(source_indices, dtype=np.int32).reshape(-1)
    patch = np.asarray(patch_indices, dtype=np.int64).reshape(-1)
    if not (keys.size == source.size == patch.size):
        raise ValueError("priority and tie-break arrays must have matching sizes")
    keep = int(quota)
    if keep <= 0 or keys.size < keep:
        raise ValueError(f"priority selection requires at least {keep} candidates")
    if keys.size == keep:
        chosen = np.arange(keys.size, dtype=np.int64)
    else:
        boundary = np.partition(keys, keep - 1)[keep - 1]
        lower = np.flatnonzero(keys < boundary)
        equal = np.flatnonzero(keys == boundary)
        missing = keep - int(lower.size)
        equal_order = np.lexsort((patch[equal], source[equal]))
        chosen = np.concatenate((lower, equal[equal_order[:missing]])).astype(np.int64)
    order = np.lexsort((patch[chosen], source[chosen], keys[chosen]))
    return chosen[order]


def anchor_indices(layer: int, *, count: int = 100_000, quota: int = 50_000) -> np.ndarray:
    bank_index = np.arange(int(count), dtype=np.int32)
    return priority_topk_indices(
        anchor_priorities(layer, count),
        bank_index,
        bank_index.astype(np.int64),
        quota,
    )


@dataclass(frozen=True)
class ReservoirResult:
    features: np.ndarray
    priorities: np.ndarray
    source_indices: np.ndarray
    patch_indices: np.ndarray
    subtype_codes: np.ndarray
    pose_attempts: np.ndarray
    scene_overlap: np.ndarray


class PriorityReservoir:
    """A bounded, traversal-order-independent block reservoir."""

    def __init__(self, quota: int, feature_dim: int = FEATURE_DIM) -> None:
        self.quota = int(quota)
        self.feature_dim = int(feature_dim)
        self._features = np.empty((0, self.feature_dim), dtype=np.float32)
        self._priorities = np.empty(0, dtype=np.uint64)
        self._source_indices = np.empty(0, dtype=np.int32)
        self._patch_indices = np.empty(0, dtype=np.int64)
        self._subtype_codes = np.empty(0, dtype=np.int16)
        self._pose_attempts = np.empty(0, dtype=np.int16)
        self._scene_overlap = np.empty(0, dtype=np.float32)
        self.candidate_count = 0

    def update(
        self,
        features: np.ndarray,
        priorities: np.ndarray,
        *,
        source_index: int,
        patch_indices: np.ndarray,
        subtype_code: int,
        pose_attempt: int = -1,
        scene_overlap: np.ndarray | None = None,
    ) -> None:
        values = np.asarray(features, dtype=np.float32)
        keys = np.asarray(priorities, dtype=np.uint64).reshape(-1)
        patches = np.asarray(patch_indices, dtype=np.int64).reshape(-1)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"factor feature block must be [N,{self.feature_dim}]")
        if values.shape[0] != keys.size or keys.size != patches.size:
            raise ValueError("factor feature and priority block sizes differ")
        if not np.isfinite(values).all():
            raise RuntimeError("non-finite factor feature encountered")
        overlap = (
            np.full(keys.size, np.nan, dtype=np.float32)
            if scene_overlap is None
            else np.asarray(scene_overlap, dtype=np.float32).reshape(-1)
        )
        if overlap.size != keys.size or np.isinf(overlap).any():
            raise ValueError("invalid scene-overlap block")
        self.candidate_count += int(keys.size)
        features_all = np.concatenate((self._features, values), axis=0)
        priorities_all = np.concatenate((self._priorities, keys))
        source_all = np.concatenate((
            self._source_indices,
            np.full(keys.size, int(source_index), dtype=np.int32),
        ))
        patch_all = np.concatenate((self._patch_indices, patches))
        subtype_all = np.concatenate((
            self._subtype_codes,
            np.full(keys.size, int(subtype_code), dtype=np.int16),
        ))
        attempt_all = np.concatenate((
            self._pose_attempts,
            np.full(keys.size, int(pose_attempt), dtype=np.int16),
        ))
        overlap_all = np.concatenate((self._scene_overlap, overlap))
        keep_count = min(self.quota, priorities_all.size)
        if priorities_all.size == keep_count:
            selected = np.lexsort((patch_all, source_all, priorities_all))
        else:
            selected = priority_topk_indices(
                priorities_all, source_all, patch_all, keep_count,
            )
        self._features = np.ascontiguousarray(features_all[selected], dtype=np.float32)
        self._priorities = priorities_all[selected]
        self._source_indices = source_all[selected]
        self._patch_indices = patch_all[selected]
        self._subtype_codes = subtype_all[selected]
        self._pose_attempts = attempt_all[selected]
        self._scene_overlap = overlap_all[selected]

    def finalize(self) -> ReservoirResult:
        if self._priorities.size != self.quota:
            raise RuntimeError(
                f"factor reservoir has {self._priorities.size} entries, expected {self.quota}"
            )
        return ReservoirResult(
            features=self._features,
            priorities=self._priorities,
            source_indices=self._source_indices,
            patch_indices=self._patch_indices,
            subtype_codes=self._subtype_codes,
            pose_attempts=self._pose_attempts,
            scene_overlap=self._scene_overlap,
        )
