"""Draw numbered tear boxes on a photo (Pillow only)."""
from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from .schema import Detection, Finding

TEAR_COLOR = (230, 30, 30)
KIND_COLORS = {"tear": (230, 30, 30), "scratch": (240, 140, 20), "stain": (150, 60, 200),
               "stitching": (0, 150, 150), "hole": (170, 0, 80), "damage": (230, 30, 30)}
BAG_COLOR = (150, 150, 150)
HANDLE_COLOR = (40, 120, 230)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def annotate(image: Image.Image, findings: Sequence[Finding],
             context: Sequence[Detection] = ()) -> Image.Image:
    """Return a copy of `image` with numbered tear boxes (and optional bag/handle boxes)."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    scale = max(img.size) / 1000
    line = max(2, round(3 * scale))
    font = _font(max(14, round(22 * scale)))

    for d in context:
        color = BAG_COLOR if d.label == "bag" else HANDLE_COLOR
        b = d.box
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=max(1, line // 2))

    for f in findings:
        b = f.box
        color = KIND_COLORS.get(f.kind, TEAR_COLOR)
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=line)
        label = f"#{f.id} {f.kind} {f.conf:.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        pad = max(2, line)
        ty = b.y1 - th - 2 * pad
        if ty < 0:  # label would fall off the top: put it inside the box
            ty = b.y1
        draw.rectangle([b.x1, ty, b.x1 + tw + 2 * pad, ty + th + 2 * pad], fill=color)
        draw.text((b.x1 + pad - left, ty + pad - top), label, fill=(255, 255, 255), font=font)
    return img
