from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from DINOv3.MADEqual.aebad_s_broad6_compose2.augmentations import (
    SEED,
    normalize_relative_path,
    stable_seed,
)


POLICY_NAME = "Phys4_Compose2"
PHYS4 = (
    "Brightness",
    "Gamma",
    "WhiteBalance",
    "IlluminationField",
)


def choose_phys4_compose2(relative_path: str, *, seed: int = SEED) -> tuple[str, str]:
    path = normalize_relative_path(relative_path)
    rng = np.random.default_rng(
        stable_seed(f"ASSIGN::{POLICY_NAME}", path, seed=seed)
    )
    selected = rng.choice(len(PHYS4), size=2, replace=False)
    return PHYS4[int(selected[0])], PHYS4[int(selected[1])]


def atomic_params(op: str, relative_path: str, *, seed: int = SEED) -> dict[str, Any]:
    path = normalize_relative_path(relative_path)
    rng = np.random.default_rng(stable_seed(op, path, seed=seed))
    if op == "Brightness":
        return {"factor": float(rng.uniform(0.8, 1.2))}
    if op == "Gamma":
        return {"gamma": float(rng.uniform(0.8, 1.25))}
    if op == "WhiteBalance":
        gains = rng.uniform(0.9, 1.1, size=3).astype(np.float64)
        gains /= float(np.prod(gains) ** (1.0 / 3.0))
        return {"r": float(gains[0]), "g": float(gains[1]), "b": float(gains[2])}
    if op == "IlluminationField":
        return {
            "amplitude": float(rng.uniform(0.1, 0.2)),
            "angle": float(rng.uniform(0.0, 2.0 * np.pi)),
        }
    raise ValueError(f"unsupported Phys4 operation: {op}")


def apply_atomic_float(rgb: np.ndarray, op: str, params: dict[str, Any]) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("Phys4 input must be an HWC RGB array")
    if op == "Brightness":
        output = value * np.float32(params["factor"])
    elif op == "Gamma":
        output = np.power(value, np.float32(params["gamma"]), dtype=np.float32)
    elif op == "WhiteBalance":
        gains = np.asarray([params["r"], params["g"], params["b"]], dtype=np.float32)
        output = value * gains[None, None, :]
    elif op == "IlluminationField":
        height, width = value.shape[:2]
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, height, dtype=np.float32),
            np.linspace(-1.0, 1.0, width, dtype=np.float32),
            indexing="ij",
        )
        angle = float(params["angle"])
        field = np.cos(angle) * xx + np.sin(angle) * yy
        denominator = float(np.max(np.abs(field)))
        if denominator > 0:
            field /= denominator
        gain = 1.0 + np.float32(params["amplitude"]) * field
        output = value * gain[:, :, None]
    else:
        raise ValueError(f"unsupported Phys4 operation: {op}")
    return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class Phys4Result:
    tensor: torch.Tensor
    op1: str
    op2: str
    params1: dict[str, Any]
    params2: dict[str, Any]

    def manifest_row(self, relative_path: str) -> dict[str, Any]:
        return {
            "relative_path": normalize_relative_path(relative_path),
            "op1": self.op1,
            "op2": self.op2,
            "recipe": f"{self.op1}>{self.op2}",
            "op1_params": json.dumps(self.params1, sort_keys=True, separators=(",", ":")),
            "op2_params": json.dumps(self.params2, sort_keys=True, separators=(",", ":")),
        }


def phys4_compose2(image: Image.Image, relative_path: str, *, seed: int = SEED) -> Phys4Result:
    path = normalize_relative_path(relative_path)
    op1, op2 = choose_phys4_compose2(path, seed=seed)
    params1 = atomic_params(op1, path, seed=seed)
    params2 = atomic_params(op2, path, seed=seed)
    value = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    value = apply_atomic_float(value, op1, params1)
    value = apply_atomic_float(value, op2, params2)
    tensor = torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1)))
    return Phys4Result(tensor, op1, op2, params1, params2)


def phys4_manifest_row(relative_path: str, *, seed: int = SEED) -> dict[str, Any]:
    path = normalize_relative_path(relative_path)
    op1, op2 = choose_phys4_compose2(path, seed=seed)
    return {
        "relative_path": path,
        "op1": op1,
        "op2": op2,
        "recipe": f"{op1}>{op2}",
        "op1_params": json.dumps(
            atomic_params(op1, path, seed=seed), sort_keys=True, separators=(",", ":")
        ),
        "op2_params": json.dumps(
            atomic_params(op2, path, seed=seed), sort_keys=True, separators=(",", ":")
        ),
    }
