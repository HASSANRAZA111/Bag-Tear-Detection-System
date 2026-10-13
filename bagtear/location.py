"""Say WHERE on the bag a tear is. Pure geometry, no ML.

Left/right/top/bottom are as seen in the photo.
"seam" wording is an approximation: it means "close to the outer edge of the bag".
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from .schema import Box, Detection, Location

EDGE_FRAC = 0.12          # outer 12% of the bag box counts as "near the seam"
HANDLE_NEAR_FRAC = 0.15   # within 15% of the bag diagonal counts as "near the handle"


def choose_bag(bags: Sequence[Detection], tear: Box):
    """Pick the bag box that contains (or is closest to) the tear. None if no bag."""
    if not bags:
        return None
    best = min(bags, key=lambda d: (d.box.gap_to(tear), -d.conf))
    return best.box


def _band(v: float) -> str:
    return "top" if v < 1 / 3 else "bottom" if v > 2 / 3 else "middle"


def _side(v: float) -> str:
    return "left" if v < 1 / 3 else "right" if v > 2 / 3 else "center"


def locate(view: str, tear: Box, bag, image_size, handles: Iterable[Box] = ()) -> Location:
    iw, ih = image_size
    bag_found = bag is not None
    ref = bag if bag is not None else Box(0, 0, iw, ih)

    tx = min(max((tear.cx - ref.x1) / max(ref.w, 1e-6), 0.0), 1.0)
    ty = min(max((tear.cy - ref.y1) / max(ref.h, 1e-6), 0.0), 1.0)
    v, h = _band(ty), _side(tx)

    edge_h = "left" if tx < EDGE_FRAC else "right" if tx > 1 - EDGE_FRAC else None
    edge_v = "top" if ty < EDGE_FRAC else "bottom" if ty > 1 - EDGE_FRAC else None

    near = []
    if edge_h and edge_v:
        where_parts = [f"near the {edge_v}-{edge_h} seam"]
        near.append(f"{edge_v}-{edge_h} seam")
    else:
        grid = "center" if (v, h) == ("middle", "center") else f"{v}-{h}"
        where_parts = [grid]
        if edge_h:
            where_parts.append(f"near the {edge_h} seam")
            near.append(f"{edge_h} seam")
        elif edge_v:
            where_parts.append("near the top edge" if edge_v == "top" else "near the bottom seam")
            near.append(f"{edge_v} edge")

    diag = math.hypot(ref.w, ref.h)
    if any(tear.gap_to(hb) <= HANDLE_NEAR_FRAC * diag for hb in handles):
        where_parts.append("near the handle")
        near.append("handle")

    where = ", ".join(where_parts)
    panel = f"{view} panel"
    return Location(panel=panel, where=where, text=f"{panel}, {where}",
                    vertical=v, horizontal=h, near=tuple(near), bag_found=bag_found)
