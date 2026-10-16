"""Offline bookkeeping; distances, MAD and interpolation use the original paths."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from DINOv3.MADEqual.aebad_s_target_fewshot import replay12 as replay
from DINOv3.MADEqual.aebad_s_target_support_trace.analysis import comparison, patch_projection
from DINOv3.MADEqual.aebad_s_union_diagnostics.analysis import pair_error
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import (
    components, instance_regions, interpolation_stencil,
)

LAYERS = replay.LAYERS
BRANCHES = ("K8", *(f"minus_{j}" for j in range(1, 9)))
ALL_BRANCHES = ("Source", *BRANCHES)
FORMAT = "same_call_target_support_loo_v1"


class CaptureLog:
    """Consume copies emitted by the SAME original scoring calls, never re-search."""

    def __init__(self) -> None:
        self.events: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.grid: tuple[int, int] | None = None
        self.positions: np.ndarray | None = None
        self.processed_shape: tuple[int, int] | None = None

    def geometry(self, grid: tuple[int, int], positions: np.ndarray,
                 processed_shape: tuple[int, int]) -> None:
        if self.grid is not None:
            raise ValueError("one capture log belongs to one full query")
        self.grid = tuple(grid)
        self.positions = np.asarray(positions).copy()
        self.processed_shape = tuple(processed_shape)

    def nearest(self, pool: str, layer: int, index: int,
                distances: np.ndarray, indices: np.ndarray) -> None:
        key = (pool, layer, index)
        if key in self.events or pool not in ("source", "target"):
            raise ValueError("duplicate or unexpected nearest event")
        self.events[key] = (distances.copy(), indices.copy())

    def arrays(self) -> dict[str, np.ndarray]:
        expected = {("source", l, 0) for l in LAYERS} | {
            ("target", l, j) for l in LAYERS for j in range(1, 9)}
        if set(self.events) != expected or self.grid is None or self.positions is None:
            raise ValueError("incomplete same-call distance log")
        return {
            "positions": self.positions,
            "source_distance": np.stack([self.events[("source", l, 0)][0] for l in LAYERS]),
            "source_index": np.stack([self.events[("source", l, 0)][1] for l in LAYERS]),
            "target_distance": np.stack([np.stack([self.events[("target", l, j)][0]
                                                    for l in LAYERS]) for j in range(1, 9)]),
            "target_index": np.stack([np.stack([self.events[("target", l, j)][1]
                                                 for l in LAYERS]) for j in range(1, 9)]),
        }


def validate_arrays(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    grid = tuple(metadata["grid"])
    if len(grid) != 2 or min(grid) <= 0 or metadata["format"] != FORMAT:
        raise ValueError("invalid raw format/grid")
    count = int(np.prod(grid))
    positions = arrays["positions"]
    if (positions.dtype != np.int64 or positions.shape != (count,)
            or not np.array_equal(np.sort(positions), np.arange(count))):
        raise ValueError("capture requires a complete physical grid and an explicit bijective positions map")
    for name, shape in (("source_distance", (4, count)), ("target_distance", (8, 4, count))):
        a = arrays[name]
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError(f"invalid same-call distance: {name}")
    for name, shape in (("source_index", (4, count)), ("target_index", (8, 4, count))):
        a = arrays[name]
        if a.shape != shape or a.dtype != np.int64 or (a < 0).any():
            raise ValueError(f"invalid nearest index: {name}")
    if len(metadata["source_rows"]) != 4 or len(metadata["supports"]) != 8:
        raise ValueError("invalid bank row metadata")
    for offset in range(4):
        if np.any(arrays["source_index"][offset] >= metadata["source_rows"][offset]):
            raise ValueError("Source memory row index out of bounds")
    for j, support in enumerate(metadata["supports"]):
        if support["support_index"] != j + 1 or min(support["grid"]) <= 0:
            raise ValueError("support order/grid mismatch")
        if np.any(arrays["target_index"][j] >= np.prod(support["grid"])):
            raise ValueError("Target patch index out of bounds")
    if len(metadata["output_shape"]) != 2 or min(metadata["output_shape"]) <= 0:
        raise ValueError("invalid output shape")


def save_raw(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    validate_arrays(arrays, metadata)
    if path.exists() or path.with_suffix(".npz.tmp").exists():
        raise ValueError("refusing existing or half-written raw cache")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, metadata=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)
    temporary.replace(path)


def load_raw(path: Path, *, identity: str, sample: dict[str, Any], checksum: str) -> tuple[dict, dict]:
    if not path.is_file() or replay.sha256(path) != checksum:
        raise ValueError("raw cache missing or modified after capture; no re-approval")
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        arrays = {name: payload[name].copy() for name in
                  ("positions", "source_distance", "source_index", "target_distance", "target_index")}
    if (metadata["identity"] != identity or metadata["sample_id"] != sample["sample_id"]
            or metadata["relative_path"] != sample["relative_path"]):
        raise ValueError("raw cache input identity mismatch")
    validate_arrays(arrays, metadata)
    return arrays, metadata


def branch_raws(arrays: dict, grid: tuple[int, int]) -> dict[str, np.ndarray]:
    """Compact rows are scattered through positions before ORIGINAL map scoring."""
    source, target = arrays["source_distance"], arrays["target_distance"]
    values = {"Source": source, "K8": np.minimum(source, target.min(axis=0))}
    for j in range(8):
        values[f"minus_{j + 1}"] = np.minimum(source, np.delete(target, j, axis=0).min(axis=0))
        if not np.all(values[f"minus_{j + 1}"] >= values["K8"] - 1e-7):
            raise ValueError("removal decreased patch distance")
    result = {}
    for name, compact in values.items():
        physical = np.empty((4, int(np.prod(grid))), np.float32)
        physical[:, arrays["positions"]] = compact
        result[name] = physical.reshape(4, *grid)
    return result


def branch_maps(raws: dict, shape: tuple[int, int], backend: Any) -> dict[str, np.ndarray]:
    """Delegate layer MAD, mean and interpolation to the frozen original runner."""
    return {name: backend.runner._method_map({"official_raw": raw}, "MAD_EQUAL", backend.mad, shape)
            for name, raw in raws.items()}


def image_rows(sample: dict, maps: dict) -> list[dict]:
    rows = []
    baseline = float(maps["K8"].max())
    for name in ALL_BRANCHES:
        value = maps[name]
        y, x = np.unravel_index(np.argmax(value), value.shape)
        rows.append(dict(sample_id=sample["sample_id"], relative_path=sample["relative_path"], label=sample["label"],
                         branch=name, image_score=float(value[y, x]), score_change_vs_K8=float(value[y, x]) - baseline,
                         full_peak_y=int(y), full_peak_x=int(x), spatial_category="unreviewed"))
    return rows


def spatial_rows(sample: dict, maps: dict, mask: np.ndarray | None, grid: tuple[int, int],
                 backend: Any) -> tuple[list[dict], list[dict]]:
    """Both reference-position and re-peaked responses; preserve GT fragments."""
    peaks, instances = [], []
    regions: list[tuple[str, int | str, np.ndarray]] = [("full", "", np.ones(maps["K8"].shape, bool))]
    if sample["label"]:
        if mask is None:
            raise ValueError("anomaly spatial evidence requires GT")
        labels, descriptors = components(mask)
        if not descriptors:
            for name in ALL_BRANCHES:
                instances.append(dict(sample_id=sample["sample_id"], relative_path=sample["relative_path"],
                    instance_id="", branch=name, area_original_pixels=0, instance_area_output_pixels=0,
                    ring_area_output_pixels=0, patch_coverage=0, instance_max=None, ring_max=None, margin=None,
                    instance_max_change_vs_K8=None, ring_max_change_vs_K8=None, margin_change_vs_K8=None,
                    response_status="empty_GT_annotation", annotation_semantics="unreviewed_no_size_threshold"))
        for descriptor in descriptors:
            component = labels == descriptor["instance_id"]
            inside, ring = instance_regions(component, mask, maps["K8"].shape)
            # Original pooling raises if empty; an originally nonempty component
            # is checked with the original backend, never a new area threshold.
            patch_coverage = int(backend.patch_mask(Image.fromarray(component.astype(np.uint8) * 255), grid).sum())
            maxima = {name: (float(m[inside].max()) if inside.any() else None,
                             float(m[ring].max()) if ring.any() else None) for name, m in maps.items()}
            base_i, base_r = maxima["K8"]
            base_margin = base_i - base_r if base_i is not None and base_r is not None else None
            for name in ALL_BRANCHES:
                im, rm = maxima[name]
                margin = im - rm if im is not None and rm is not None else None
                instances.append(dict(sample_id=sample["sample_id"], relative_path=sample["relative_path"],
                    **descriptor, branch=name, patch_coverage=patch_coverage,
                    instance_area_output_pixels=int(inside.sum()), ring_area_output_pixels=int(ring.sum()),
                    ring_radius_original_pixels=16, ring_metric="Chebyshev", ring_excludes_all_GT=True,
                    instance_max=im, ring_max=rm, margin=margin,
                    instance_max_change_vs_K8=im-base_i if im is not None and base_i is not None else None,
                    ring_max_change_vs_K8=rm-base_r if rm is not None and base_r is not None else None,
                    margin_change_vs_K8=margin-base_margin if margin is not None and base_margin is not None else None,
                    response_status="mapped_instance_disappeared" if not inside.any() else
                                    ("empty_ring" if not ring.any() else "available"),
                    annotation_semantics="unreviewed_no_size_threshold"))
            regions.extend((("instance", descriptor["instance_id"], inside), ("ring", descriptor["instance_id"], ring)))
    for role, instance_id, region in regions:
        if not region.any():
            continue
        for origin in ALL_BRANCHES:
            flat = int(np.argmax(np.where(region, maps[origin], -np.inf)))
            y, x = divmod(flat, region.shape[1])
            for measured in ALL_BRANCHES:
                peaks.append(dict(sample_id=sample["sample_id"], relative_path=sample["relative_path"],
                    region=role, instance_id=instance_id, peak_of_branch=origin, measured_branch=measured,
                    output_y=y, output_x=x, score=float(maps[measured][y, x]),
                    score_change_vs_K8=float(maps[measured][y, x])-float(maps["K8"][y, x]),
                    fixed_K8_reference=origin == "K8", original_reference=origin == "Source",
                    peak_tie_rule="first_row_major_maximum", spatial_category="unreviewed"))
    return peaks, instances


def correspondence_rows(sample: dict, peaks: list[dict], arrays: dict, metadata: dict,
                        maps: dict, raws: dict, backend: Any) -> tuple[list[dict], list[dict]]:
    inverse = {int(flat): row for row, flat in enumerate(arrays["positions"])}
    rows, effects = [], []
    positions = sorted({(row["output_y"], row["output_x"]) for row in peaks})
    calibrated = {b: backend.calibrated(r, backend.mad.medians, backend.mad.scales) for b, r in raws.items()}
    for y, x in positions:
        stencil = interpolation_stencil((y, x), tuple(metadata["grid"]), tuple(metadata["output_shape"]))
        for branch in ALL_BRANCHES:
            response = sum(w * float(calibrated[branch][:, py, px].mean()) for py, px, w in stencil)
            effects.append(dict(sample_id=sample["sample_id"], output_y=y, output_x=x, branch=branch,
                                **comparison(response, maps[branch][y, x], score=True)))
            retained = list(range(8)) if branch in ("Source", "K8") else [j for j in range(8) if j+1 != int(branch[6:])]
            for py, px, weight in stencil:
                compact = inverse[py * metadata["grid"][1] + px]
                for offset, layer in enumerate(LAYERS):
                    source = float(arrays["source_distance"][offset, compact])
                    target = arrays["target_distance"][:, offset, compact]
                    minimum = min(float(target[j]) for j in retained)
                    sorted_distances = sorted(float(target[j]) for j in retained)
                    winning_support = min(j for j in retained if float(target[j]) == minimum) + 1
                    for j in range(8):
                        support = metadata["supports"][j]
                        flat = int(arrays["target_index"][j, offset, compact])
                        rows.append(dict(sample_id=sample["sample_id"], output_y=y, output_x=x, branch=branch,
                            layer=layer, query_compact_row=compact, query_flat_index=py*metadata["grid"][1]+px,
                            interpolation_weight=weight, source_distance=source,
                            source_memory_row=int(arrays["source_index"][offset, compact]), source_image="unavailable",
                            source_patch="unavailable", support_index=j+1, support_relative_path=support["relative_path"],
                            support_retained=branch != "Source" and j in retained,
                            target_distance=float(target[j]), target_patch_flat=flat,
                            target_patch_y=flat//support["grid"][1], target_patch_x=flat%support["grid"][1],
                            target_projection=json.dumps(patch_projection(flat, support["geometry"])),
                            query_projection=json.dumps(patch_projection(py*metadata["grid"][1]+px, metadata["geometry"])),
                            chosen_target_support=winning_support if branch != "Source" else "",
                            chosen_union_origin="source" if branch == "Source" or source < minimum else
                                                ("target" if minimum < source else "source_target_tie"),
                            cross_support_top2_gap=sorted_distances[1]-sorted_distances[0],
                            near_tie_numerical_flag=sorted_distances[1]-sorted_distances[0] <= 1e-7+1e-6*abs(minimum),
                            within_support_patch_uniqueness="unknown", query_region="unreviewed",
                            support_correspondence="unreviewed", coordinate_meaning="input_token_cell_not_full_receptive_field"))
    return rows, effects


def support_layer_rows(sample: dict, arrays: dict) -> list[dict]:
    source, target = arrays["source_distance"], arrays["target_distance"]
    minimum = target.min(axis=0)
    full = np.minimum(source, minimum)
    rows = []
    for j in range(8):
        removed = np.minimum(source, np.delete(target, j, axis=0).min(axis=0))
        for offset, layer in enumerate(LAYERS):
            own = target[j, offset]
            gap = np.partition(target[:, offset], 1, axis=0)[1] - minimum[offset]
            rows.append(dict(sample_id=sample["sample_id"], label=sample["label"], support_index=j+1, layer=layer,
                patch_count=source.shape[1], target_strict_win_fraction=float((minimum[offset] < source[offset]).mean()),
                source_strict_win_fraction=float((source[offset] < minimum[offset]).mean()),
                source_target_tie_fraction=float((source[offset] == minimum[offset]).mean()),
                own_target_min_fraction=float((own == minimum[offset]).mean()),
                own_unique_target_win_fraction=float(((own == minimum[offset]) & (gap > 0) & (own < source[offset])).mean()),
                lost_distance_drop_fraction=float((removed[offset] > full[offset]).mean()),
                lost_distance_drop_mean=float((removed[offset] - full[offset]).mean()),
                cross_support_gap_min=float(gap.min()), cross_support_gap_mean=float(gap.mean()),
                cross_support_near_tie_fraction=float((gap <= 1e-7+1e-6*np.abs(minimum[offset])).mean()),
                within_support_patch_uniqueness="unknown"))
    return rows


def ranking_rows(per_image: list[dict], fixed_top10: list[str]) -> tuple[list[dict], list[dict]]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    groups = defaultdict(dict)
    for row in per_image:
        if row["sample_id"] in groups[row["branch"]]:
            raise ValueError("duplicate image/branch")
        groups[row["branch"]][row["sample_id"]] = row
    ids = sorted(groups["K8"])
    if any(set(g) != set(ids) for g in groups.values()) or set(groups) != set(ALL_BRANCHES):
        raise ValueError("all branches must use the identical held-out set")
    labels = np.asarray([groups["K8"][i]["label"] for i in ids])
    normals = [i for i in ids if groups["K8"][i]["label"] == 0]
    anomalies = [i for i in ids if groups["K8"][i]["label"] == 1]
    summary, per_pairs = [], []
    for branch in ALL_BRANCHES:
        scores = [groups[branch][i]["image_score"] for i in ids]
        totals = {i: [0.0, 0, 0, 0] for i in ids}
        improved = worsened = unchanged = ties = 0
        errors = 0.0
        for n in normals:
            for a in anomalies:
                before = pair_error(groups["K8"][n]["image_score"], groups["K8"][a]["image_score"])
                after = pair_error(groups[branch][n]["image_score"], groups[branch][a]["image_score"])
                errors += after
                improved += after < before
                worsened += after > before
                unchanged += after == before
                ties += after == .5
                for i in (n, a):
                    totals[i][0] += after
                    totals[i][1] += after < before
                    totals[i][2] += after > before
                    totals[i][3] += after == before
        normal_scores = [groups[branch][i]["image_score"] for i in normals]
        summary.append(dict(branch=branch, image_count=len(ids), normal_count=len(normals), anomaly_count=len(anomalies),
            image_AUROC=float(roc_auc_score(labels, scores)), image_AUPR=float(average_precision_score(labels, scores)),
            pair_count=len(normals)*len(anomalies), weighted_pair_errors=errors, pair_ties=ties,
            improved_pairs_vs_K8=improved, worsened_pairs_vs_K8=worsened, unchanged_pairs_vs_K8=unchanged,
            tie_error_weight=.5, normal_score_p90=float(np.percentile(normal_scores, 90)),
            normal_score_p95=float(np.percentile(normal_scores, 95)), normal_score_max=float(max(normal_scores)),
            fixed_top10_mean_score_change=float(np.mean([groups[branch][i]["score_change_vs_K8"] for i in fixed_top10 if i in groups[branch]]))))
        for i in ids:
            per_pairs.append(dict(sample_id=i, branch=branch, weighted_pair_errors=totals[i][0],
                improved_pairs_vs_K8=totals[i][1], worsened_pairs_vs_K8=totals[i][2], unchanged_pairs_vs_K8=totals[i][3],
                fixed_K8_top10=i in fixed_top10))
    return summary, per_pairs
