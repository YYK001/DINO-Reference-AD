"""Frozen ShiftADD AeBAD-S P0 v2 identifiers and deterministic planning."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np


PROTOCOL_ID = "shiftadd_aebad_s_seed42_p0_v2_anchor_priority"
BRIGHTNESS2_CONTROL_PROTOCOL_ID = (
    "shiftadd_aebad_s_seed42_p0_brightness2_control_v2_anchor_priority"
)
TRANSFORM_REVISION = "shift_factors_v2_pose_contained_scene_safe_v1"
SAMPLER_REVISION = "factor_balanced_priority_reservoir_v1"
AUDIT_REVISION = "scene_overlap_v1"
CACHE_REVISION = "shiftadd_raw_top1_cache_v2_anchor_priority"

SEED = 42
LAYERS = (7, 15, 23, 31)
FEATURE_DIM = 1280
THRESHOLD_FRACTION = 8
MAX_DATABASE_SIZE = 100_000
ANCHOR_QUOTA = 50_000
PATCH_SIZE = 640
PATCH_OVERLAP = 128
CHECKPOINT_SHA256 = (
    "7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488"
)

SHIFT_BRANCHES = ("ILLUMINATION", "POSE", "SCENE")
FACTOR_QUOTAS = {
    "ILLUMINATION": 16_667,
    "POSE": 16_667,
    "SCENE_VALID": 16_666,
}
CONTROL_FACTOR_QUOTA = 50_000

BRANCH_CODES = {
    "OFFICIAL_ANCHOR": 0,
    "ILLUMINATION": 1,
    "POSE": 2,
    "SCENE_VALID": 3,
    "BRIGHTNESS2": 4,
}

MAIN_METHODS = (
    "SUPERADD_OFFICIAL",
    "MAD_EQUAL",
    "ANCHOR50_EQUAL",
    "ANCHOR50_MAD",
    "SHIFTADD_EQUAL",
    "SHIFTADD_MAD",
)
MAIN_NEW_METHODS = (
    "ANCHOR50_EQUAL",
    "ANCHOR50_MAD",
    "SHIFTADD_EQUAL",
    "SHIFTADD_MAD",
)
CONTROL_METHODS = (
    "SUPERADD_OFFICIAL",
    "MAD_EQUAL",
    "ANCHOR50_EQUAL",
    "ANCHOR50_MAD",
    "BRIGHTNESS2_EQUAL",
    "BRIGHTNESS2_MAD",
)
CONTROL_NEW_METHODS = ("BRIGHTNESS2_EQUAL", "BRIGHTNESS2_MAD")
METRIC_NAMES = (
    "image_AUROC",
    "image_AUPR",
    "pixel_AUROC",
    "pixel_AUPR",
    "AUPRO",
)

RNG_NAMESPACES = (
    "shift_transform",
    "pose_attempt",
    "brightness2_control",
)


def protocol_id_for_variant(variant: str) -> str:
    if variant == "shiftadd":
        return PROTOCOL_ID
    if variant == "brightness2_control":
        return BRIGHTNESS2_CONTROL_PROTOCOL_ID
    raise ValueError(f"unsupported ShiftADD v2 variant: {variant}")


def stable_seed(
    relative_path: str,
    namespace: str,
    seed: int = SEED,
    *,
    attempt: int | None = None,
) -> int:
    """Return a process/order-independent seed for transform parameters."""
    if namespace not in RNG_NAMESPACES:
        raise ValueError(f"unknown deterministic namespace: {namespace}")
    fields = [str(relative_path), str(int(seed)), namespace]
    if attempt is not None:
        fields.append(str(int(attempt)))
    payload = "\n".join(fields).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def stable_rng(
    relative_path: str,
    namespace: str,
    seed: int = SEED,
    *,
    attempt: int | None = None,
) -> np.random.Generator:
    return np.random.default_rng(
        stable_seed(relative_path, namespace, seed, attempt=attempt)
    )


def brightness2_factors(
    relative_paths: Iterable[str],
    *,
    seed: int = SEED,
) -> dict[str, float]:
    """Return the independent second legacy Official brightness draw."""
    paths = sorted(str(path) for path in relative_paths)
    generator = np.random.RandomState(int(seed))
    for _ in paths:
        generator.uniform(0.8, 1.2)
    values = [float(generator.uniform(0.8, 1.2)) for _ in paths]
    return dict(zip(paths, values))


@dataclass(frozen=True)
class CandidatePlan:
    relative_path: str
    planned_branch: str
    shift_seed: int


def make_candidate_plans(
    relative_paths: Iterable[str],
    *,
    seed: int = SEED,
) -> list[CandidatePlan]:
    """Assign 455 sorted paths to the frozen 152/152/151 factor plan."""
    paths = sorted(str(path) for path in relative_paths)
    if len(set(paths)) != len(paths):
        raise ValueError("memory-fit relative paths must be unique")
    ordering = np.asarray(paths, dtype=object)
    assignment_rng = stable_rng("__shift_assignment__", "shift_transform", seed)
    permutation = assignment_rng.permutation(len(ordering))
    branch_for_path: dict[str, str] = {}
    for position, path_index in enumerate(permutation.tolist()):
        branch_for_path[str(ordering[path_index])] = SHIFT_BRANCHES[
            position % len(SHIFT_BRANCHES)
        ]
    if len(paths) == 455:
        counts = {
            branch: sum(value == branch for value in branch_for_path.values())
            for branch in SHIFT_BRANCHES
        }
        expected = {"ILLUMINATION": 152, "POSE": 152, "SCENE": 151}
        if counts != expected:
            raise AssertionError(f"frozen ShiftADD branch plan mismatch: {counts}")
    return [
        CandidatePlan(
            relative_path=path,
            planned_branch=branch_for_path[path],
            shift_seed=stable_seed(path, "shift_transform", seed),
        )
        for path in paths
    ]


def revisions() -> dict[str, str]:
    return {
        "protocol_id": PROTOCOL_ID,
        "transform_revision": TRANSFORM_REVISION,
        "sampler_revision": SAMPLER_REVISION,
        "audit_revision": AUDIT_REVISION,
        "cache_revision": CACHE_REVISION,
    }
