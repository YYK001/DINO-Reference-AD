"""AeBAD-S support selection built on the existing few-shot feature cache."""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Mapping, Sequence

from DINOv3.MADEqual.robustad_target_fewshot_p0.support import (
    SHOT_COUNTS,
    SUPPORT_POOL_SIZE,
    SupportFeatures,
    extract_or_load_support,
    support_prefixes,
)
from DINOv3.relation_reliability.datasets import Record


DOMAINS = ("same", "background", "illumination", "view")
EXPECTED_HELDOUT = {
    "same": (222, 459, 681),
    "background": (85, 212, 297),
    "illumination": (67, 198, 265),
    "view": (84, 280, 364),
}


def relative_path(record: Record, dataset_root: Path) -> str:
    return Path(record.path).resolve().relative_to(dataset_root.resolve()).as_posix()


def stable_support_key(
    record: Record,
    *,
    dataset_root: Path,
    domain: str,
    seed: int = 42,
) -> tuple[int, str]:
    relative = relative_path(record, dataset_root)
    value = f"{seed}|{domain}|{relative}".encode("utf-8")
    return zlib.crc32(value) & 0xFFFFFFFF, relative


def select_target_support(
    records: Sequence[Record],
    *,
    dataset_root: Path,
) -> tuple[dict[str, list[Record]], dict[str, list[Record]]]:
    pools: dict[str, list[Record]] = {}
    heldout: dict[str, list[Record]] = {}
    for domain in DOMAINS:
        group = [record for record in records if record.domain == domain]
        normals = sorted(
            (record for record in group if int(record.label) == 0),
            key=lambda record: stable_support_key(
                record, dataset_root=dataset_root, domain=domain,
            ),
        )
        if len(normals) < SUPPORT_POOL_SIZE:
            raise RuntimeError(f"AeBAD-S/{domain} has fewer than eight target normals")
        pool = normals[:SUPPORT_POOL_SIZE]
        selected = {str(Path(record.path).resolve()) for record in pool}
        evaluation = [
            record for record in group
            if str(Path(record.path).resolve()) not in selected
        ]
        normal_count = sum(int(record.label) == 0 for record in evaluation)
        anomaly_count = sum(int(record.label) == 1 for record in evaluation)
        expected_normal, expected_anomaly, expected_total = EXPECTED_HELDOUT[domain]
        if (normal_count, anomaly_count, len(evaluation)) != (
            expected_normal, expected_anomaly, expected_total,
        ):
            raise RuntimeError(
                f"AeBAD-S/{domain} held-out count mismatch: "
                f"{normal_count}/{anomaly_count}/{len(evaluation)}"
            )
        if any(int(record.label) != 0 for record in pool):
            raise RuntimeError("target support contains an anomaly")
        pools[domain] = pool
        heldout[domain] = evaluation
    if sum(len(values) for values in heldout.values()) != 1607:
        raise RuntimeError("AeBAD-S common held-out set must contain 1607 images")
    return pools, heldout


def select_source_extra(
    calibration: Sequence[Record],
    *,
    dataset_root: Path,
) -> list[Record]:
    if len(calibration) != 66 or any(int(record.label) != 0 for record in calibration):
        raise RuntimeError("source-extra input must be the 66 source calibration normals")
    selected = sorted(
        calibration,
        key=lambda record: stable_support_key(
            record, dataset_root=dataset_root, domain="source_extra",
        ),
    )[:SUPPORT_POOL_SIZE]
    if len({str(Path(record.path).resolve()) for record in selected}) != SUPPORT_POOL_SIZE:
        raise RuntimeError("source-extra support images are not distinct")
    return selected


def support_manifest_rows(
    source_pool: Sequence[Record],
    target_pools: Mapping[str, Sequence[Record]],
    *,
    dataset_root: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups = [("source_extra", "source", source_pool)] + [
        ("target_matched", domain, target_pools[domain]) for domain in DOMAINS
    ]
    for branch, domain, pool in groups:
        if len(pool) != SUPPORT_POOL_SIZE:
            raise RuntimeError(f"support pool is not eight images: {branch}/{domain}")
        for index, record in enumerate(pool, start=1):
            rows.append({
                "branch": branch,
                "domain": domain,
                "support_index": index,
                "relative_path": relative_path(record, dataset_root),
                "label": int(record.label),
                "included_in_K1": int(index <= 1),
                "included_in_K2": int(index <= 2),
                "included_in_K4": int(index <= 4),
                "included_in_K8": int(index <= 8),
            })
    return rows


__all__ = [
    "DOMAINS", "EXPECTED_HELDOUT", "SHOT_COUNTS", "SupportFeatures",
    "extract_or_load_support", "relative_path", "select_source_extra",
    "select_target_support", "stable_support_key", "support_manifest_rows",
    "support_prefixes",
]
