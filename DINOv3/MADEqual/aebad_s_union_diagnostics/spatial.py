"""Spatial diagnostics. The original map backend is imported only for existing caches."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from DINOv3.MADEqual.aebad_s_broad6_compose2 import mmr_compatible_eval as mmr
from DINOv3.relation_reliability.protocol import LAYERS
from .analysis import CACHE_REVISION, METHODS, PROTOCOL_ID

RING_RADIUS = 16  # Chebyshev dilation in original GT pixels, fixed before diagnosis.


def load_cache(path: Path, relative: str) -> tuple[dict[str, np.ndarray], tuple[int, int], float]:
    with np.load(path, allow_pickle=False) as payload:
        for key, expected in (("protocol_id", PROTOCOL_ID), ("cache_revision", CACHE_REVISION),
                              ("relative_path", relative)):
            if str(payload[key].item()) != expected:
                raise ValueError(f"cache {key} mismatch: {path}")
        shape_array = payload["output_shape"]
        if shape_array.shape != (2,) or not np.issubdtype(shape_array.dtype, np.integer) or np.any(shape_array <= 0):
            raise ValueError(f"invalid cache output_shape: {path}")
        shape = tuple(int(value) for value in shape_array)
        names = ["official_raw", *[f"{branch}_{kind}_{shot}" for branch in ("source", "target")
                                  for kind in ("raw", "support") for shot in (1, 2, 4, 8)]]
        values = {name: np.asarray(payload[name], np.float32) for name in names}
    base = values["official_raw"]
    if (base.ndim != 3 or base.shape[0] != 4 or min(base.shape[1:]) <= 0
            or any(value.shape != base.shape or not np.isfinite(value).all() or (value < 0).any()
                   for value in values.values())):
        raise ValueError(f"invalid distance arrays: {path}")
    error = float(np.max(np.abs(values["target_raw_8"] - np.minimum(base, values["target_support_8"]))))
    if not np.allclose(values["target_raw_8"], np.minimum(base, values["target_support_8"]), rtol=1e-6, atol=1e-7):
        raise ValueError(f"union != layer-wise min(d_S,d_T): {path}, error={error}")
    return values, shape, error


class MapBackend:
    """Reuse the cache loader and maps, never the original runner.run()."""

    def __init__(self, mad_path: Path):
        from DINOv3.MADEqual.aebad_s_target_fewshot import runner
        from DINOv3.relation_oracle_cache.extract import mad_calibrated, mad_equal
        from DINOv3.relation_reliability.superadd_host import interpolate_score, mask_at_patch_grid
        if runner.PROTOCOL_ID != PROTOCOL_ID or runner.CACHE_REVISION != CACHE_REVISION:
            raise ValueError("diagnostic protocol differs from current original runner")
        self.runner = runner
        self.mad = runner._load_mad(mad_path)
        if self.mad is None:
            raise FileNotFoundError(mad_path)
        self.calibrated = mad_calibrated
        self.equal = mad_equal
        self.interpolate = interpolate_score
        self.patch_mask = mask_at_patch_grid

    def maps(self, path: Path, relative: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[int, int]]:
        values, shape = self.runner._load_observation(path, relative=relative)
        maps = {method: self.runner._method_map(values, method, self.mad, shape) for method in METHODS}
        maps["TARGET_ONLY_DIAGNOSTIC"] = self.interpolate(
            self.equal(values["target_support_8"], self.mad.medians, self.mad.scales), shape,
        ).astype(np.float32)
        return values, maps, shape


def components(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())
    rows = []
    for index, slices in enumerate(ndimage.find_objects(labels), 1):
        if slices is None:
            continue
        ys, xs = slices
        rows.append({"instance_id": index, "area_original_pixels": int(areas[index]),
                     "bbox_y0": ys.start, "bbox_y1_exclusive": ys.stop,
                     "bbox_x0": xs.start, "bbox_x1_exclusive": xs.stop,
                     "connectivity": 8})
    assert len(rows) == count
    return labels, rows


def resized_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (shape[1], shape[0]), Image.Resampling.NEAREST)) > 0


def crop_diagnostics(value: np.ndarray, mask: np.ndarray | None, label: int) -> dict[str, Any]:
    """Use the very same resized float map for max and the original MMR crop."""
    resized = np.asarray(Image.fromarray(np.asarray(value, np.float32)).resize(
        (mmr.RESIZE_SIZE, mmr.RESIZE_SIZE), Image.Resampling.BILINEAR), dtype=np.float32)
    cropped = mmr.center_crop_224(resized)
    if not np.array_equal(cropped, mmr.resize_and_crop_map(value)):
        raise ValueError("resize/crop differs from original MMR postprocessing")
    maximum = float(resized.max())
    peak_positions = np.argwhere(resized == maximum)
    outside = ((peak_positions[:, 0] < mmr.CROP_START) | (peak_positions[:, 0] >= mmr.CROP_STOP)
               | (peak_positions[:, 1] < mmr.CROP_START) | (peak_positions[:, 1] >= mmr.CROP_STOP))
    result = {"original_max": float(value.max()), "resize_256_max": maximum,
              "crop_224_max": float(cropped.max()),
              "resize_drop": float(value.max()) - maximum,
              "boundary_removal_drop": maximum - float(cropped.max()),
              "total_drop": float(value.max()) - float(cropped.max()),
              "some_resize_peak_cropped": bool(outside.any()),
              "all_resize_peaks_cropped": bool(outside.all()), "original_label": label,
              "gt_status": "normal_no_defect" if label == 0 else "missing_gt"}
    if mask is not None:
        resized_gt = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (256, 256), Image.Resampling.NEAREST)) > 0
        cropped_gt = mmr.resize_and_crop_mask(mask)
        result.update({"gt_status": "verified", "gt_original_area": int(mask.sum()),
                       "gt_resize_area": int(resized_gt.sum()), "gt_crop_area": int(cropped_gt.sum()),
                       "gt_removed_by_crop": int(resized_gt.sum() - cropped_gt.sum()),
                       "gt_empty_after_resize": not bool(resized_gt.any()),
                       "gt_empty_after_crop": not bool(cropped_gt.any()),
                       "archived_mmr_label_rule": mmr.mmr_compatible_label(label, cropped_gt)})
    return result


def interpolation_stencil(pixel: tuple[int, int], grid: tuple[int, int],
                          output: tuple[int, int]) -> list[tuple[int, int, float]]:
    """align_corners=False half-pixel coordinates, edge clamping, row-major patches."""
    axes = []
    for position, size, outsize in zip(pixel, grid, output, strict=True):
        coordinate = max(0.0, min(size - 1.0, (position + 0.5) * size / outsize - 0.5))
        lower = math.floor(coordinate)
        fraction = coordinate - lower
        axes.append([(lower, 1.0 - fraction), (min(lower + 1, size - 1), fraction)])
    weights: dict[tuple[int, int], float] = {}
    for y, wy in axes[0]:
        for x, wx in axes[1]:
            weights[y, x] = weights.get((y, x), 0.0) + wy * wx
    return [(y, x, weight) for (y, x), weight in weights.items() if weight > 0]


def peak_rows(sample: dict[str, Any], values: dict[str, np.ndarray], maps: dict[str, np.ndarray],
              backend: MapBackend) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_map, union_map = (maps[method] for method in METHODS)
    source_peak = tuple(int(i) for i in np.unravel_index(np.argmax(source_map), source_map.shape))
    union_peak = tuple(int(i) for i in np.unravel_index(np.argmax(union_map), union_map.shape))
    grid = values["official_raw"].shape[1:]
    calibrated = {key: backend.calibrated(values[key], backend.mad.medians, backend.mad.scales)
                  for key in ("official_raw", "target_support_8", "target_raw_8")}
    summaries, layers = [], []
    for name, pixel in (("source_peak", source_peak), ("union_peak", union_peak)):
        stencil = interpolation_stencil(pixel, grid, source_map.shape)
        summaries.append({"sample_id": sample["sample_id"], "relative_path": sample["relative_path"],
                          "peak": name, "pixel_y": pixel[0], "pixel_x": pixel[1],
                          "output_height": source_map.shape[0], "output_width": source_map.shape[1],
                          "source_score_at_pixel": float(source_map[pixel]),
                          "target_only_score_at_pixel": float(maps["TARGET_ONLY_DIAGNOSTIC"][pixel]),
                          "union_score_at_pixel": float(union_map[pixel]),
                          "peak_shift_output_pixels": math.dist(source_peak, union_peak),
                          "original_image_y_center": (pixel[0] + 0.5) * sample["image_height"] / source_map.shape[0] - 0.5,
                          "original_image_x_center": (pixel[1] + 0.5) * sample["image_width"] / source_map.shape[1] - 0.5,
                          "spatial_category": "unreviewed",
                          "pixel_to_patch": "bilinear_align_corners_false_half_pixel_stencil"})
        for offset, layer in enumerate(LAYERS):
            row = {"sample_id": sample["sample_id"], "peak": name, "layer": layer,
                   "target_win_weight": sum(weight for y, x, weight in stencil
                                            if values["target_support_8"][offset, y, x] < values["official_raw"][offset, y, x]),
                   "target_tie_weight": sum(weight for y, x, weight in stencil
                                            if values["target_support_8"][offset, y, x] == values["official_raw"][offset, y, x]),
                   "stencil": ";".join(f"{y},{x},{weight:.9g}" for y, x, weight in stencil)}
            for key, prefix in (("official_raw", "source"), ("target_support_8", "target_only"), ("target_raw_8", "union")):
                row[prefix + "_distance_at_pixel"] = sum(weight * float(values[key][offset, y, x]) for y, x, weight in stencil)
                row[prefix + "_mad_contribution"] = sum(weight * float(calibrated[key][offset, y, x]) for y, x, weight in stencil) / 4
            # Interpolation of patch-wise min; NOT min of the two interpolated distances.
            row["signed_scaled_reduction"] = (row["source_distance_at_pixel"] - row["union_distance_at_pixel"]) / (float(backend.mad.scales[offset]) + 1e-8)
            layers.append(row)
    return summaries, layers


def instance_regions(component: np.ndarray, full_gt: np.ndarray,
                     shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    # A fixed 16-original-pixel Chebyshev ring; remove ALL other GT defects.
    ring = ndimage.maximum_filter(component, size=2 * RING_RADIUS + 1, mode="constant", cval=0) & ~full_gt
    inside, outside = resized_mask(component, shape), resized_mask(ring, shape)
    outside &= ~resized_mask(full_gt, shape)
    return inside, outside


def instance_response(component: np.ndarray, full_gt: np.ndarray, maps: dict[str, np.ndarray],
                      values: dict[str, np.ndarray], backend: MapBackend) -> dict[str, Any]:
    inside, outside = instance_regions(component, full_gt, maps[METHODS[0]].shape)
    patch = backend.patch_mask(Image.fromarray(component.astype(np.uint8) * 255), values["official_raw"].shape[1:])
    result = {"patch_coverage": int(patch.sum()), "patch_coverage_status": "adaptive_max_original_gt",
              "response_status": "available" if inside.any() else "instance_empty_at_evaluation_grid",
              "ring_radius_original_pixels": RING_RADIUS, "ring_metric": "Chebyshev", "ring_area_output_pixels": int(outside.sum()),
              "instance_area_output_pixels": int(inside.sum()),
              "margin_definition": "instance_max_minus_nondefect_ring_max"}
    if inside.any():
        for method, prefix in zip(METHODS, ("source", "union"), strict=True):
            result[prefix + "_instance_max"] = float(maps[method][inside].max())
            result[prefix + "_instance_mean"] = float(maps[method][inside].mean())
            result[prefix + "_margin"] = float(maps[method][inside].max() - maps[method][outside].max()) if outside.any() else ""
        result["instance_max_drop"] = result["source_instance_max"] - result["union_instance_max"]
        result["margin_change"] = result["union_margin"] - result["source_margin"] if outside.any() else ""
    reduction = (values["official_raw"] - values["target_raw_8"]) / (backend.mad.scales[:, None, None] + 1e-8)
    for offset, layer in enumerate(LAYERS):
        result[f"scaled_distance_drop_L{layer}"] = float(reduction[offset][patch].mean())
    return result


def montage(items: list[tuple[Image.Image, str]], path: Path, columns: int = 4) -> None:
    width, height = 350, 290
    canvas = Image.new("RGB", (columns * width, math.ceil(len(items) / columns) * height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (image, caption) in enumerate(items):
        image = image.copy()
        image.thumbnail((width - 12, height - 62))
        x, y = (index % columns) * width, (index // columns) * height
        canvas.paste(image, (x + (width - image.width) // 2, y + 4))
        draw.multiline_text((x + 8, y + height - 55), caption, fill="black", font=font, spacing=3)
    canvas.save(path)


def plot_maps(image: Image.Image, mask: np.ndarray | None, maps: dict[str, np.ndarray],
              values: dict[str, np.ndarray], backend: MapBackend, path: Path) -> dict[str, float]:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    names = [*METHODS, "TARGET_ONLY_DIAGNOSTIC"]
    lo, hi = min(float(maps[name].min()) for name in names), max(float(maps[name].max()) for name in names)
    # Raw Euclidean/1280 distance drop, averaged after layer-wise union; not a score.
    raw_drop = backend.interpolate((values["official_raw"] - values["target_raw_8"]).mean(axis=0), maps[METHODS[0]].shape)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original RGB")
    if mask is None:
        axes[0, 1].text(0.5, 0.5, "Normal: no defect GT", ha="center")
    else:
        axes[0, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Defect GT (not product foreground)")
    for ax, name in zip((axes[0, 2], axes[1, 0], axes[1, 1]), names, strict=True):
        im = ax.imshow(maps[name], vmin=lo, vmax=hi, cmap="viridis")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, label="Source MAD-Equal score")
    im = axes[1, 2].imshow(raw_drop, cmap="magma", vmin=0, vmax=max(float(raw_drop.max()), 1e-12))
    axes[1, 2].set_title("Mean layer distance: Source - union")
    fig.colorbar(im, ax=axes[1, 2], label="Euclidean distance / 1280 (independent scale)")
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"score_color_min": lo, "score_color_max": hi, "distance_drop_max": float(raw_drop.max())}
