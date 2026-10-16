"""Twelve-case maps only; no detector, interpolation or semantic classifier."""
from __future__ import annotations

from pathlib import Path
import json
import textwrap
from typing import Any

import numpy as np


def plot_case(image: Any, mask: np.ndarray | None, maps: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    names = list(maps)
    lo = min(float(m.min()) for m in maps.values())
    hi = max(float(m.max()) for m in maps.values())
    differences = {n: maps[n]-maps["K8"] for n in names if n != "K8"}
    bound = max(1e-12, max(float(np.abs(m).max()) for m in differences.values()))
    fig, axes = plt.subplots(4, 6, figsize=(22, 14), constrained_layout=True)
    for ax in axes.ravel():
        ax.set_axis_off()
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Query RGB; semantics unreviewed")
    if mask is not None:
        axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Defect GT; complement is not background")
    score_plot = None
    for ax, name in zip(axes.ravel()[2:12], names, strict=True):
        score_plot = ax.imshow(maps[name], cmap="viridis", vmin=lo, vmax=hi)
        ax.set_title(name)
    difference_plot = None
    for ax, (name, difference) in zip(axes.ravel()[12:21], differences.items(), strict=True):
        difference_plot = ax.imshow(difference, cmap="coolwarm", vmin=-bound, vmax=bound)
        ax.set_title(name+" minus K8")
    fig.colorbar(score_plot, ax=list(axes.ravel()[2:12]), shrink=.6, label="Shared Source MAD-Equal response scale")
    fig.colorbar(difference_plot, ax=list(axes.ravel()[12:21]), shrink=.6, label="Independent difference scale")
    fig.suptitle("Frozen union support LOO; input token projections are not complete receptive fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_correspondences(image: Any, sample: dict, peaks: list[dict], rows: list[dict],
                         index: Any, metadata: dict, output: Path) -> None:
    """Show each branch's full peak; every stencil participant is in the CSV.

    Use existing projection/marking panels. The displayed cell is explicitly the
    largest-weight participant, not a replacement for the bilinear stencil.
    """
    from PIL import Image, ImageDraw, ImageFont
    from DINOv3.MADEqual.aebad_s_target_support_trace.visualization import _marked, _place, _neighborhood
    from DINOv3.MADEqual.aebad_s_target_support_trace.runner import trace_image
    names = list(dict.fromkeys(r["peak_of_branch"] for r in peaks if r["region"] == "full"))
    for branch in names:
        peak = next(r for r in peaks if r["region"] == "full" and r["peak_of_branch"] == branch and r["measured_branch"] == branch)
        y, x = peak["output_y"], peak["output_x"]
        candidates = [r for r in rows if r["branch"] == branch and r["output_y"] == y and r["output_x"] == x]
        if not candidates:
            continue
        dominant = min(candidates, key=lambda r: (-r["interpolation_weight"], r["query_flat_index"]))["query_flat_index"]
        candidates = [r for r in candidates if r["query_flat_index"] == dominant]
        canvas = Image.new("RGB", (1500, 860), "white")
        draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
        draw.text((10, 8), f"{sample['sample_id']} {branch}: current capture NN log; stencil verification in CSV; historical unique indices unknown", fill="black", font=font)
        draw.text((10, 28), "Largest-weight stencil token shown; all participants in CSV. Token cell is NOT the full receptive field.", fill="black", font=font)
        draw.text((10, 48), "Semantic labels unreviewed. Source image/patch unavailable; Source memory rows recorded below.", fill="black", font=font)
        query_cell = json.loads(candidates[0]["query_projection"])
        marked = _marked(image, (query_cell["original_x_center"], query_cell["original_y_center"]), query_cell)
        panels = [(f"Query peak xy=({x},{y}); score={peak['score']:.7g}", image, None),
                  (f"Query token flat={dominant}; weight={candidates[0]['interpolation_weight']:.7g}", marked,
                   _neighborhood(marked, (query_cell["original_x_center"], query_cell["original_y_center"])))]
        for layer in (7, 15, 23, 31):
            group = [r for r in candidates if r["layer"] == layer]
            winner = next(r for r in group if r["support_index"] == r["chosen_target_support"]) if branch != "Source" else group[0]
            caption = f"L{layer}: union origin={winner['chosen_union_origin']}; Source row={winner['source_memory_row']} dS={winner['source_distance']:.7g}"
            if branch == "Source":
                panels.append((caption+"\nSource provenance unavailable", Image.new("RGB", (200, 100), "white"), None))
                continue
            support = trace_image(index, winner["support_relative_path"], "RGB")
            cell = json.loads(winner["target_projection"])
            center = (cell["original_x_center"], cell["original_y_center"])
            marked = _marked(support, center, cell)
            caption += (f"\nTarget #{winner['support_index']} {Path(winner['support_relative_path']).name}"
                        f" flat={winner['target_patch_flat']} grid=({winner['target_patch_y']},{winner['target_patch_x']})"
                        f"\ndT={winner['target_distance']:.7g}; center=({center[0]:.2f},{center[1]:.2f}); patch uniqueness unknown")
            panels.append((caption, marked, _neighborhood(marked, center)))
        for number, (caption, full, zoom) in enumerate(panels):
            bx, by = (number % 3)*500, 80+(number//3)*385
            caption = "\n".join("\n".join(textwrap.wrap(line, width=72)) for line in caption.splitlines())
            draw.multiline_text((bx+10, by+4), caption, fill="black", font=font, spacing=4)
            _place(canvas, full, (bx+10, by+100, bx+250 if zoom is not None else bx+490, by+375))
            if zoom is not None:
                _place(canvas, zoom, (bx+260, by+100, bx+490, by+375))
        output.mkdir(parents=True, exist_ok=True)
        canvas.save(output / f"{sample['sample_id']}_{branch}_full_peak_correspondence.png")
