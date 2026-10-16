"""Small spatial/index helpers; no model imports or alternate detector."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import (
    components, instance_regions, interpolation_stencil,
)

CASES = (
    ("test_0887", "test/good/background/IMG_7728.png"),
    ("test_0921", "test/good/background/IMG_7765.png"),
    ("test_0545", "test/fracture/background/IMG_9385.png"),
    ("test_0023", "test/ablation/background/IMG_8863.png"),
)
LAYERS = (7, 15, 23, 31)
METHODS = ("MAD_EQUAL", "TARGET_MATCHED_K8_MAD")
DIST_ATOL, DIST_RTOL, SCORE_ATOL = 1e-7, 1e-6, 1e-5
TIE_POLICY = "first_row_major_patch_within_support_then_lowest_manifest_support_index"


def comparison(actual: Any, expected: Any, *, score: bool = False) -> dict[str, Any]:
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError("verification requires aligned nonempty values")
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not finite:
        return {"verified": False, "max_abs_error": None, "max_rel_error": None,
                "reason": "nonfinite"}
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), 1e-12)
    bound = SCORE_ATOL if score else DIST_ATOL + DIST_RTOL * np.abs(expected)
    return {"verified": bool(np.all(absolute <= bound)),
            "max_abs_error": float(absolute.max()), "max_rel_error": float(relative.max()),
            "reason": "matched" if np.all(absolute <= bound) else "tolerance_exceeded"}


def target_check(source: Any, traced_target: Any, cached_target: Any,
                 cached_union: Any) -> dict[str, Any]:
    target = comparison(traced_target, cached_target)
    union = comparison(np.minimum(source, traced_target), cached_union)
    return {"verified": target["verified"] and union["verified"],
            "target_abs_error": target["max_abs_error"], "target_rel_error": target["max_rel_error"],
            "union_abs_error": union["max_abs_error"], "union_rel_error": union["max_rel_error"]}


def flat_coordinates(index: int, grid: tuple[int, int]) -> tuple[int, int]:
    if len(grid) != 2 or min(grid) <= 0 or not 0 <= int(index) < grid[0] * grid[1]:
        raise ValueError("patch flat index outside actual grid")
    return divmod(int(index), int(grid[1]))


def trace_positions(sample: dict[str, Any], maps: dict[str, np.ndarray],
                    mask: np.ndarray | None) -> list[dict[str, Any]]:
    source, union = (np.asarray(maps[key]) for key in METHODS)
    if source.shape != union.shape or source.ndim != 2 or not np.isfinite([source, union]).all():
        raise ValueError("invalid aligned original score maps")
    inside = ring = np.zeros(source.shape, bool)
    if sample["label"]:
        if mask is None:
            raise ValueError("anomaly trace requires original GT")
        labels, instances = components(mask)
        if len(instances) != 1:
            raise ValueError("each fixed anomaly must have exactly one GT instance")
        inside, ring = instance_regions(labels == instances[0]["instance_id"], mask, source.shape)
        if not inside.any() or not ring.any():
            raise ValueError("fixed GT instance or ring is empty at evaluation grid")
    unique: dict[tuple[int, int], dict[str, Any]] = {}
    for method, prefix in zip(METHODS, ("source", "union"), strict=True):
        roles = [("full_peak", np.ones(source.shape, bool))]
        if sample["label"]:
            roles += [("instance_peak", inside), ("ring_peak", ring)]
        for role, region in roles:
            # np.argmax returns the first flat index; excluded pixels are -inf.
            flat = int(np.argmax(np.where(region, maps[method], -np.inf)))
            y, x = divmod(flat, source.shape[1])
            row = unique.setdefault((y, x), {
                "sample_id": sample["sample_id"], "relative_path": sample["relative_path"],
                "position_id": f"{sample['sample_id']}_p{len(unique) + 1:02d}",
                "label": sample["label"], "output_y": y, "output_x": x,
                "output_height": source.shape[0], "output_width": source.shape[1],
                "inside_gt_instance": bool(inside[y, x]), "inside_nondefect_ring": bool(ring[y, x]),
                "source_score": float(source[y, x]), "union_score": float(union[y, x]),
                "roles": [], "peak_tie_rule": "first_row_major_maximum_within_fixed_region",
                "query_region": "unreviewed", "support_correspondence": "unreviewed",
            })
            row["roles"].append(prefix + "_" + role)
    for row in unique.values():
        row["roles"] = ";".join(row["roles"])
    return list(unique.values())


def position_patches(positions: list[dict[str, Any]], grid: tuple[int, int]) -> list[dict[str, Any]]:
    rows = []
    for position in positions:
        pixel = (position["output_y"], position["output_x"])
        stencil = interpolation_stencil(pixel, grid, (position["output_height"], position["output_width"]))
        dominant = min(stencil, key=lambda item: (-item[2], item[0] * grid[1] + item[1]))
        for y, x, weight in stencil:
            for layer in LAYERS:
                rows.append({"sample_id": position["sample_id"], "position_id": position["position_id"],
                             "roles": position["roles"], "layer": layer, "query_patch_y": y, "query_patch_x": x,
                             "query_flat_index": y * grid[1] + x, "interpolation_weight": weight,
                             "query_grid_height": grid[0], "query_grid_width": grid[1],
                             "display_patch": (y, x) == dominant[:2]})
    return rows


def per_support_nearest(query: Any, banks: list[Any], grids: list[tuple[int, int]],
                        top1: Callable[[Any, Any], tuple[Any, Any]]) -> list[list[dict[str, Any]]]:
    """Production injects the original torch kernel; tests inject a NumPy reference."""
    if len(banks) != 8 or len(grids) != 8 or int(query.shape[1]) != 1280:
        raise ValueError("trace requires exactly eight support banks and dimension 1280")
    results: list[list[dict[str, Any]]] = [[] for _ in range(int(query.shape[0]))]
    for support_index, (bank, grid) in enumerate(zip(banks, grids, strict=True), 1):
        if tuple(bank.shape) != (grid[0] * grid[1], 1280):
            raise ValueError("support bank does not match its stored grid")
        distances, indices = top1(query, bank)
        distances = np.asarray(distances, np.float32) / 1280.0
        indices = np.asarray(indices)
        if (distances.shape != (len(results),) or indices.shape != distances.shape
                or not np.issubdtype(indices.dtype, np.integer)
                or not np.isfinite(distances).all() or (distances < 0).any()):
            raise ValueError("invalid exact nearest-neighbor result")
        for offset, (distance, index) in enumerate(zip(distances, indices, strict=True)):
            y, x = flat_coordinates(int(index), grid)
            results[offset].append({"support_index": support_index, "support_flat_index": int(index),
                                    "support_patch_y": y, "support_patch_x": x,
                                    "support_grid_height": grid[0], "support_grid_width": grid[1],
                                    "distance_euclidean_over_1280": float(distance), "tie_policy": TIE_POLICY})
    for candidates in results:
        candidates.sort(key=lambda row: (row["distance_euclidean_over_1280"], row["support_index"]))
        for rank, row in enumerate(candidates, 1):
            row.update(support_rank=rank, winner=rank == 1)
    return results


def image_geometry(model: Any, height: int, width: int, grid: tuple[int, int],
                   processed_shape: tuple[int, int]) -> dict[str, Any]:
    patch = int(model.patch_exec.model_patch_size)
    if grid != tuple(size // patch for size in processed_shape):
        raise ValueError("feature grid differs from actual preprocessed image geometry")
    axes = {}
    for name, size, expected in zip(("y", "x"), processed_shape, grid, strict=True):
        inputs, predictions, results = model.patch_exec.axis_patch_split(size)
        cursor = 0
        for input_roi, prediction, result in zip(inputs, predictions, results, strict=True):
            if result[0] != cursor or result[1] - result[0] != prediction[1] - prediction[0]:
                raise ValueError("stitched grid is not a contiguous partition")
            if input_roi[0] + prediction[1] * patch > input_roi[1]:
                raise ValueError("prediction tokens outside input tile")
            cursor = result[1]
        if cursor != expected:
            raise ValueError("tile results do not cover stored feature grid")
        axes[name] = {"input": [list(pair) for pair in inputs], "prediction": [list(pair) for pair in predictions],
                      "result": [list(pair) for pair in results]}
    return {"original_height": height, "original_width": width,
            "processed_height": processed_shape[0], "processed_width": processed_shape[1],
            "grid": list(grid), "model_patch_size": patch, "axes": axes,
            "coordinate_meaning": "projected_input_token_cell_not_complete_receptive_field"}


def patch_projection(index: int, geometry: dict[str, Any]) -> dict[str, Any]:
    y, x = flat_coordinates(index, tuple(geometry["grid"]))
    values: dict[str, Any] = {}
    token = geometry["model_patch_size"]
    for axis, coordinate in (("y", y), ("x", x)):
        entries = geometry["axes"][axis]
        segment = next(i for i, pair in enumerate(entries["result"]) if pair[0] <= coordinate < pair[1])
        local = entries["prediction"][segment][0] + coordinate - entries["result"][segment][0]
        start = entries["input"][segment][0] + local * token
        original = geometry["original_height" if axis == "y" else "original_width"]
        processed = geometry["processed_height" if axis == "y" else "processed_width"]
        values.update({f"tile_{axis}_index": segment, f"tile_local_patch_{axis}": local,
                       f"original_{axis}_center": (start + token / 2) * original / processed - 0.5,
                       f"original_{axis}0": start * original / processed,
                       f"original_{axis}1_exclusive": (start + token) * original / processed})
    values["coordinate_meaning"] = geometry["coordinate_meaning"]
    return values


def weighted_layers(values: dict[str, np.ndarray], patches: list[dict[str, Any]],
                    calibrated: Callable, medians: np.ndarray, scales: np.ndarray) -> dict[str, list[float]]:
    results = {}
    for name in ("official_raw", "target_support_8", "target_raw_8"):
        response = calibrated(values[name], medians, scales)
        results[name] = [sum(row["interpolation_weight"] * float(response[offset, row["query_patch_y"], row["query_patch_x"]]) / 4
                             for row in patches if row["layer"] == layer)
                         for offset, layer in enumerate(LAYERS)]
    return results
