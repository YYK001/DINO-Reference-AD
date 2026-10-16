"""Fixed correspondence display only; crops are never passed to a backbone."""
from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

from PIL import Image, ImageDraw, ImageFont

DISPLAY_RADIUS = 64  # Original pixels, fixed for display, unrelated to the 16px GT ring.
PANEL_WIDTH, PANEL_HEIGHT = 500, 385


def _marked(image: Image.Image, center: tuple[float, float],
            cell: dict[str, Any] | None = None) -> Image.Image:
    value = image.copy().convert("RGB")
    draw = ImageDraw.Draw(value)
    x, y = center
    if cell is not None:
        draw.rectangle((cell["original_x0"], cell["original_y0"],
                        cell["original_x1_exclusive"] - 1, cell["original_y1_exclusive"] - 1),
                       outline="orange", width=5)
    draw.line((x - 15, y, x + 15, y), fill="red", width=4)
    draw.line((x, y - 15, x, y + 15), fill="red", width=4)
    return value


def _place(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    image = image.copy()
    image.thumbnail((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    canvas.paste(image, (box[0] + (box[2] - box[0] - image.width) // 2,
                        box[1] + (box[3] - box[1] - image.height) // 2))


def _neighborhood(image: Image.Image, center: tuple[float, float]) -> Image.Image:
    x, y = (round(value) for value in center)
    return image.crop((max(0, x - DISPLAY_RADIUS), max(0, y - DISPLAY_RADIUS),
                       min(image.width, x + DISPLAY_RADIUS + 1), min(image.height, y + DISPLAY_RADIUS + 1)))


def correspondence_figure(query: Image.Image, position: dict[str, Any], query_cell: dict[str, Any],
                          winners: list[dict[str, Any]], data_root: Path, path: Path) -> None:
    verified = position["correspondence_verified"]
    canvas = Image.new("RGB", (3 * PANEL_WIDTH, 2 * PANEL_HEIGHT + 80), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    state = "VERIFIED selected correspondence" if verified else "UNVERIFIED - not exact original correspondence evidence"
    draw.text((12, 8), f"{position['position_id']} {state}", fill="darkgreen" if verified else "red", font=font)
    draw.text((12, 28), "Red cross: reported center. Orange box: projected input token cell; NOT full receptive field.", fill="black", font=font)
    draw.text((12, 48), "Display crops: fixed +/-64 original pixels, no feature extraction. Semantic labels: unreviewed.", fill="black", font=font)
    output_center = (position["original_x_center"], position["original_y_center"])
    patch_center = (query_cell["original_x_center"], query_cell["original_y_center"])
    full = _marked(query, output_center, query_cell)
    token = _marked(query, patch_center, query_cell)
    panels = [
        ("Query output location\n" + position["roles"] + f"\nS={position['source_score']:.6g} ST={position['union_score']:.6g}",
         full, None),
        (f"Largest interpolation-weight query patch: flat={position['display_query_flat_index']}\n"
         f"Token center xy=({patch_center[0]:.2f}, {patch_center[1]:.2f})\nProjection only; query-region label=unreviewed",
         token, _neighborhood(token, patch_center)),
    ]
    for winner in winners:
        with Image.open(data_root / winner["support_relative_path"]) as handle:
            support = handle.convert("RGB")
        center = (winner["original_x_center"], winner["original_y_center"])
        marked = _marked(support, center, winner)
        caption = (f"L{winner['layer']} Target support #{winner['support_index']} {Path(winner['support_relative_path']).name}\n"
                   f"flat={winner['support_flat_index']} grid=({winner['support_patch_y']},{winner['support_patch_x']})\n"
                   f"dT={winner['distance_euclidean_over_1280']:.8g}; cached dT={winner['cached_d_T']:.8g}\n"
                   f"center xy=({center[0]:.2f}, {center[1]:.2f}); "
                   f"patch-check={'PASS' if winner['correspondence_verified'] else 'FAIL'}")
        panels.append((caption, marked, _neighborhood(marked, center)))
    for index, (caption, full, zoom) in enumerate(panels):
        x, y = (index % 3) * PANEL_WIDTH, 80 + (index // 3) * PANEL_HEIGHT
        caption = "\n".join("\n".join(textwrap.wrap(line, width=72)) for line in caption.splitlines())
        draw.multiline_text((x + 10, y + 4), caption, fill="black", font=font, spacing=3)
        if zoom is None:
            _place(canvas, full, (x + 10, y + 90, x + PANEL_WIDTH - 10, y + PANEL_HEIGHT - 10))
        else:
            _place(canvas, full, (x + 10, y + 110, x + 250, y + PANEL_HEIGHT - 10))
            _place(canvas, zoom, (x + 260, y + 110, x + PANEL_WIDTH - 10, y + PANEL_HEIGHT - 10))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
