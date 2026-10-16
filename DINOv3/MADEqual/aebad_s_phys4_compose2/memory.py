from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from DINOv3.MADEqual.aebad_s_broad6_compose2.memory import _Broad6Preprocessing
from DINOv3.relation_reliability.datasets import Record
from DINOv3.relation_reliability.protocol import LAYERS
from DINOv3.relation_reliability.superadd_host import _official_candidate_resource_estimate
from external_baselines.superadd_external.memory_variants import build_official_memory_spooled

from .augmentations import SEED, phys4_compose2


PROTOCOL_ID = "mad_equal_aebad_s_phys4_compose2_seed42_v1"


def build_phys4_memory(
    model: Any,
    train_records: Sequence[Record],
    *,
    dataset_root: Path,
    device: torch.device,
    scratch_dir: Path,
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    tensors: list[torch.Tensor] = []
    manifest: list[dict[str, Any]] = []
    for index, record in enumerate(train_records):
        if index % int(model.threshold_fraction) == 0:
            tensors.append(torch.empty(0))
            continue
        relative = Path(record.path).resolve().relative_to(dataset_root).as_posix()
        with Image.open(record.path) as source:
            result = phys4_compose2(source.convert("RGB"), relative, seed=SEED)
        tensors.append(result.tensor)
        manifest.append(result.manifest_row(relative))
    if len(manifest) != 455:
        raise RuntimeError(f"Phys4 memory-fit count must be 455, got {len(manifest)}")

    expected_rows, estimated_bytes = _official_candidate_resource_estimate(model, tensors)
    original = model.augmented_preprocessing
    try:
        model.augmented_preprocessing = _Broad6Preprocessing(model, original)
        np.random.seed(SEED)
        build = build_official_memory_spooled(
            model,
            tensors,
            device=device,
            scratch_dir=scratch_dir,
            storage_backend="auto",
            expected_candidate_rows=expected_rows,
            estimated_peak_bytes=estimated_bytes,
        )
    finally:
        model.augmented_preprocessing = original
    for layer in LAYERS:
        if tuple(build.banks[layer].shape) != (100_000, 1280):
            raise RuntimeError(
                f"Phys4 L{layer} bank shape is {tuple(build.banks[layer].shape)}"
            )
    model.prototype_embeddings = build.banks
    return build.banks, manifest, build.metadata


def save_memory(path: Path, banks: dict[int, torch.Tensor], *, source_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"protocol_id": PROTOCOL_ID, "source_count": int(source_count)}
    for layer in LAYERS:
        state[f"bank_{layer}"] = banks[layer].detach().cpu()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_memory(path: Path, model: Any, *, source_count: int, device: torch.device) -> bool:
    if not path.is_file():
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("protocol_id") != PROTOCOL_ID or int(state.get("source_count", -1)) != source_count:
        raise RuntimeError(f"incompatible Phys4 memory cache: {path}")
    banks: dict[int, torch.Tensor] = {}
    for layer in LAYERS:
        value = state.get(f"bank_{layer}")
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (100_000, 1280):
            raise RuntimeError(f"invalid Phys4 memory cache L{layer}: {path}")
        banks[layer] = value.to(device=device, dtype=torch.float32)
    model.prototype_embeddings = banks
    return True
