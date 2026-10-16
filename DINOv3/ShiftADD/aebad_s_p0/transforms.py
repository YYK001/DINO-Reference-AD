"""Frozen source-normal RGB factors for ShiftADD AeBAD-S P0 v2."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .protocol import CandidatePlan, SEED, stable_rng


_BICUBIC = getattr(Image.Resampling, "BICUBIC", Image.BICUBIC)
_BILINEAR = getattr(Image.Resampling, "BILINEAR", Image.BILINEAR)
_NEAREST = getattr(Image.Resampling, "NEAREST", Image.NEAREST)
_AFFINE = getattr(Image.Transform, "AFFINE", Image.AFFINE)
_PERSPECTIVE = getattr(Image.Transform, "PERSPECTIVE", Image.PERSPECTIVE)


@dataclass(frozen=True)
class ColorSegmentation:
    background_connected: np.ndarray
    trusted_foreground_bbox: tuple[int, int, int, int] | None
    threshold: float


@dataclass(frozen=True)
class TransformResult:
    image: Image.Image
    actual_branch: str
    subtype: str
    params: dict[str, Any]
    candidate_active: bool
    pose_attempt: int = -1
    pose_containment_pass: bool = False
    scene_valid: bool = False
    scene_mask: np.ndarray | None = None


def _as_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _to_image(values: np.ndarray) -> Image.Image:
    return Image.fromarray(
        np.clip(np.rint(values), 0, 255).astype(np.uint8), mode="RGB"
    )


def _border_connected(mask: np.ndarray) -> np.ndarray:
    """Return true pixels connected to the boundary by four-connectivity."""
    value = np.asarray(mask, dtype=bool)
    height, width = value.shape
    output = np.zeros_like(value, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    boundary = (
        [(0, x) for x in range(width)]
        + [(height - 1, x) for x in range(width)]
        + [(y, 0) for y in range(1, max(1, height - 1))]
        + [(y, width - 1) for y in range(1, max(1, height - 1))]
    )
    for y, x in boundary:
        if value[y, x] and not output[y, x]:
            output[y, x] = True
            queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= next_y < height
                and 0 <= next_x < width
                and value[next_y, next_x]
                and not output[next_y, next_x]
            ):
                output[next_y, next_x] = True
                queue.append((next_y, next_x))
    return output


def _trusted_foreground_bbox(
    background_connected: np.ndarray,
) -> tuple[int, int, int, int] | None:
    foreground = ~np.asarray(background_connected, dtype=bool)
    height, width = foreground.shape
    visited = np.zeros_like(foreground, dtype=bool)
    minimum_area = int(math.ceil(0.02 * height * width))
    best: tuple[int, int, int, int, int] | None = None
    for start_y in range(height):
        for start_x in range(width):
            if not foreground[start_y, start_x] or visited[start_y, start_x]:
                continue
            visited[start_y, start_x] = True
            queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
            area = 0
            min_y = max_y = start_y
            min_x = max_x = start_x
            touches_boundary = False
            while queue:
                y, x = queue.popleft()
                area += 1
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_x, max_x = min(min_x, x), max(max_x, x)
                touches_boundary |= y in (0, height - 1) or x in (0, width - 1)
                for next_y, next_x in (
                    (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1),
                ):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and foreground[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))
            if area >= minimum_area and not touches_boundary:
                candidate = (area, min_x, min_y, max_x, max_y)
                if best is None or candidate > best:
                    best = candidate
    if best is None:
        return None
    _, min_x, min_y, max_x, max_y = best
    margin = max(4, int(math.ceil(0.01 * max(height, width))))
    return (
        max(0, min_x - margin),
        max(0, min_y - margin),
        min(width - 1, max_x + margin),
        min(height - 1, max_y + margin),
    )


def source_color_segmentation(image: Image.Image) -> ColorSegmentation:
    """Build the shared source-only border-color segmentation."""
    source = _as_rgb_array(image)
    height, width = source.shape[:2]
    scale = min(1.0, 256.0 / float(max(height, width)))
    small_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    small = np.asarray(
        Image.fromarray(source, mode="RGB").resize(small_size, _BILINEAR),
        dtype=np.float32,
    )
    border = np.concatenate(
        (small[0], small[-1], small[1:-1, 0], small[1:-1, -1]), axis=0
    )
    centre = np.median(border, axis=0)
    border_distance = np.linalg.norm(border - centre, axis=1)
    threshold = max(8.0, float(np.percentile(border_distance, 80.0)) + 4.0)
    candidate = np.linalg.norm(small - centre.reshape(1, 1, 3), axis=2) <= threshold
    connected_small = _border_connected(candidate)
    connected = np.asarray(
        Image.fromarray(connected_small.astype(np.uint8) * 255, mode="L").resize(
            (width, height), _NEAREST,
        ),
        dtype=np.uint8,
    ) > 0
    return ColorSegmentation(
        background_connected=connected,
        trusted_foreground_bbox=_trusted_foreground_bbox(connected),
        threshold=threshold,
    )


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(radius))):
        padded = np.pad(value, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(value, dtype=bool)
        for delta_y in range(3):
            for delta_x in range(3):
                eroded &= padded[
                    delta_y:delta_y + value.shape[0],
                    delta_x:delta_x + value.shape[1],
                ]
        value = eroded
    return value


def scene_safe_mask(image: Image.Image, segmentation: ColorSegmentation) -> np.ndarray:
    height, width = image.height, image.width
    radius = max(2, 2 * int(math.ceil(max(height, width) / 256.0)))
    return _erode(segmentation.background_connected, radius)


def _illumination(image: Image.Image, rng: np.random.Generator) -> TransformResult:
    source = _as_rgb_array(image)
    values = source.astype(np.float32) / 255.0
    subtype = ("brightness", "contrast", "gamma", "white_balance", "smooth_shadow")[
        int(rng.integers(0, 5))
    ]
    params: dict[str, Any]
    if subtype == "brightness":
        factor = float(rng.uniform(0.75, 1.25))
        values *= factor
        params = {"brightness_factor": factor}
    elif subtype == "contrast":
        factor = float(rng.uniform(0.8, 1.2))
        centre = values.mean(axis=(0, 1), keepdims=True)
        values = (values - centre) * factor + centre
        params = {"contrast_factor": factor}
    elif subtype == "gamma":
        gamma = float(rng.uniform(0.8, 1.25))
        values = np.power(np.clip(values, 0.0, 1.0), gamma)
        params = {"gamma": gamma}
    elif subtype == "white_balance":
        gains = rng.uniform(0.9, 1.1, size=3).astype(np.float32)
        gains /= float(gains.mean())
        values *= gains.reshape(1, 1, 3)
        params = {
            "gain_r": float(gains[0]),
            "gain_g": float(gains[1]),
            "gain_b": float(gains[2]),
        }
    else:
        height, width = values.shape[:2]
        factor = float(rng.uniform(0.75, 0.95))
        centre_x = float(rng.uniform(-0.15, 1.15)) * width
        centre_y = float(rng.uniform(-0.15, 1.15)) * height
        radius = float(rng.uniform(0.55, 1.10)) * max(height, width)
        y, x = np.ogrid[:height, :width]
        alpha = np.clip(
            1.0 - np.hypot(x - centre_x, y - centre_y) / radius, 0.0, 1.0
        )
        values *= (1.0 - (1.0 - factor) * alpha)[..., None]
        params = {
            "shadow_factor": factor,
            "shadow_center_x_fraction": centre_x / max(width, 1),
            "shadow_center_y_fraction": centre_y / max(height, 1),
            "shadow_radius_fraction": radius / max(height, width, 1),
        }
    return TransformResult(
        image=_to_image(np.clip(values, 0.0, 1.0) * 255.0),
        actual_branch="ILLUMINATION",
        subtype=subtype,
        params=params,
        candidate_active=True,
    )


def _pose_parameters(rng: np.random.Generator) -> dict[str, float]:
    """Draw in the exact v1 order; attempt zero therefore reproduces v1."""
    return {
        "rotation_degrees": float(rng.uniform(-12.0, 12.0)),
        "translate_x_fraction": float(rng.uniform(-0.08, 0.08)),
        "translate_y_fraction": float(rng.uniform(-0.08, 0.08)),
        "scale": float(rng.uniform(0.9, 1.1)),
        "perspective_x_fraction": float(rng.uniform(-0.03, 0.03)),
        "perspective_y_fraction": float(rng.uniform(-0.03, 0.03)),
    }


def pose_attempt_parameters(
    relative_path: str,
    attempt: int,
    *,
    seed: int = SEED,
) -> dict[str, float]:
    if not 0 <= int(attempt) < 8:
        raise ValueError("Pose attempt must be in [0,7]")
    rng = (
        stable_rng(relative_path, "shift_transform", seed)
        if int(attempt) == 0
        else stable_rng(relative_path, "pose_attempt", seed, attempt=int(attempt))
    )
    return _pose_parameters(rng)


def _pose_inverse_matrices(
    width: int,
    height: int,
    padding: int,
    params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    rotation = math.radians(float(params["rotation_degrees"]))
    scale = float(params["scale"])
    cosine = math.cos(rotation) / scale
    sine = math.sin(rotation) / scale
    centre_x, centre_y = width / 2.0, height / 2.0
    translate_x = float(params["translate_x_fraction"]) * width
    translate_y = float(params["translate_y_fraction"]) * height
    affine_inverse = np.asarray(
        (
            (cosine, sine, centre_x - cosine * centre_x - sine * centre_y - translate_x),
            (-sine, cosine, centre_y + sine * centre_x - cosine * centre_y - translate_y),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    g = float(params["perspective_x_fraction"]) / max(width, 1)
    h = float(params["perspective_y_fraction"]) / max(height, 1)
    perspective_inverse = np.asarray(
        (
            (1.0 - padding * g, -padding * h, 0.0),
            (-padding * g, 1.0 - padding * h, 0.0),
            (g, h, 1.0),
        ),
        dtype=np.float64,
    )
    return affine_inverse, perspective_inverse


def pose_forward_homography(
    width: int,
    height: int,
    params: dict[str, float],
) -> np.ndarray:
    padding = max(8, int(math.ceil(max(height, width) * 0.20)))
    affine_inverse, perspective_inverse = _pose_inverse_matrices(
        width, height, padding, params,
    )
    return np.linalg.inv(affine_inverse @ perspective_inverse)


def pose_contains_bbox(
    width: int,
    height: int,
    bbox: tuple[int, int, int, int],
    params: dict[str, float],
) -> bool:
    x0, y0, x1, y1 = bbox
    corners = np.asarray(
        ((x0, y0, 1.0), (x1, y0, 1.0), (x1, y1, 1.0), (x0, y1, 1.0)),
        dtype=np.float64,
    ).T
    try:
        projected = pose_forward_homography(width, height, params) @ corners
    except np.linalg.LinAlgError:
        return False
    denominator = projected[2]
    if not np.isfinite(projected).all() or not (
        np.all(denominator > 1e-9) or np.all(denominator < -1e-9)
    ):
        return False
    xy = projected[:2] / denominator
    return bool(
        np.isfinite(xy).all()
        and np.all(xy[0] >= 0.0)
        and np.all(xy[0] <= width - 1.0)
        and np.all(xy[1] >= 0.0)
        and np.all(xy[1] <= height - 1.0)
    )


def _reflect_pad(values: np.ndarray, padding: int) -> Image.Image:
    padded = np.pad(
        values, ((padding, padding), (padding, padding), (0, 0)), mode="reflect"
    )
    return Image.fromarray(padded.astype(np.uint8), mode="RGB")


def _apply_pose(image: Image.Image, params: dict[str, float]) -> Image.Image:
    source = _as_rgb_array(image)
    height, width = source.shape[:2]
    padding = max(8, int(math.ceil(max(height, width) * 0.20)))
    affine_inverse, _ = _pose_inverse_matrices(width, height, padding, params)
    first = _reflect_pad(source, padding).transform(
        (width, height),
        _AFFINE,
        (
            float(affine_inverse[0, 0]),
            float(affine_inverse[0, 1]),
            float(affine_inverse[0, 2] + padding),
            float(affine_inverse[1, 0]),
            float(affine_inverse[1, 1]),
            float(affine_inverse[1, 2] + padding),
        ),
        resample=_BICUBIC,
    )
    return _reflect_pad(_as_rgb_array(first), padding).transform(
        (width, height),
        _PERSPECTIVE,
        (
            1.0,
            0.0,
            float(padding),
            0.0,
            1.0,
            float(padding),
            float(params["perspective_x_fraction"]) / max(width, 1),
            float(params["perspective_y_fraction"]) / max(height, 1),
        ),
        resample=_BICUBIC,
    ).convert("RGB")


def _pose(
    image: Image.Image,
    relative_path: str,
    segmentation: ColorSegmentation,
    *,
    seed: int,
) -> TransformResult:
    bbox = segmentation.trusted_foreground_bbox
    if bbox is None:
        return TransformResult(
            image=image.copy(),
            actual_branch="POSE_NOOP",
            subtype="pose_containment_invalid",
            params={},
            candidate_active=False,
        )
    selected: tuple[int, dict[str, float]] | None = None
    for attempt in range(8):
        params = pose_attempt_parameters(relative_path, attempt, seed=seed)
        if pose_contains_bbox(image.width, image.height, bbox, params):
            selected = (attempt, params)
            break
    if selected is None:
        return TransformResult(
            image=image.copy(),
            actual_branch="POSE_NOOP",
            subtype="pose_containment_invalid",
            params={"trusted_bbox": list(bbox)},
            candidate_active=False,
        )
    attempt, params = selected
    manifest_params: dict[str, Any] = dict(params)
    manifest_params["trusted_bbox"] = list(bbox)
    return TransformResult(
        image=_apply_pose(image, params),
        actual_branch="POSE",
        subtype="affine_perspective_2d",
        params=manifest_params,
        candidate_active=True,
        pose_attempt=attempt,
        pose_containment_pass=True,
    )


def _scene(
    image: Image.Image,
    rng: np.random.Generator,
    segmentation: ColorSegmentation,
) -> TransformResult:
    source = _as_rgb_array(image)
    safe = scene_safe_mask(image, segmentation)
    if float(safe.mean()) < 0.02:
        return TransformResult(
            image=image.copy(),
            actual_branch="SCENE_NOOP",
            subtype="background_mask_invalid",
            params={"background_fraction": float(safe.mean())},
            candidate_active=False,
            scene_valid=False,
            scene_mask=safe,
        )
    gain = float(rng.uniform(0.85, 1.15))
    offset = rng.uniform(-12.0, 12.0, size=3).astype(np.float32)
    changed = source.astype(np.float32)
    changed[safe] = np.clip(
        changed[safe] * gain + offset.reshape(1, 3), 0.0, 255.0
    )
    return TransformResult(
        image=_to_image(changed),
        actual_branch="SCENE_VALID",
        subtype="border_connected_background",
        params={
            "background_gain": gain,
            "offset_r": float(offset[0]),
            "offset_g": float(offset[1]),
            "offset_b": float(offset[2]),
            "background_fraction": float(safe.mean()),
        },
        candidate_active=True,
        scene_valid=True,
        scene_mask=safe,
    )


def apply_shift_transform(
    image: Image.Image,
    plan: CandidatePlan,
    *,
    seed: int = SEED,
) -> TransformResult:
    """Apply one frozen v2 factor in raw RGB space."""
    if plan.planned_branch not in set(("ILLUMINATION", "POSE", "SCENE")):
        raise ValueError(f"plan is not a Shift candidate: {plan.planned_branch}")
    rng = stable_rng(plan.relative_path, "shift_transform", seed)
    if plan.planned_branch == "ILLUMINATION":
        return _illumination(image, rng)
    segmentation = source_color_segmentation(image)
    if plan.planned_branch == "POSE":
        return _pose(image, plan.relative_path, segmentation, seed=seed)
    return _scene(image, rng, segmentation)
