"""Severity = damage area / bag area.

Boxes are used for the area, so a thin diagonal tear is overestimated a little.
If the bag outline is not found, the whole photo is used as the reference.
The two limits below are sensible starting values, not an industry standard: tune them.
"""
from __future__ import annotations

from .schema import Box

MINOR_MAX = 0.01      # below 1% of the bag area  -> Minor
MODERATE_MAX = 0.05   # 1% to below 5%            -> Moderate, 5% or more -> Severe
LEVELS = ("Minor", "Moderate", "Severe")


def area_ratio(damage: Box, ref: Box) -> float:
    ref_area = ref.w * ref.h
    if ref_area <= 0:
        return 0.0
    return min((damage.w * damage.h) / ref_area, 1.0)


def classify(ratio: float) -> str:
    return "Minor" if ratio < MINOR_MAX else "Moderate" if ratio < MODERATE_MAX else "Severe"


def overall(levels) -> str:
    """Worst level among the findings; 'None' if there is no damage."""
    levels = list(levels)
    return max(levels, key=LEVELS.index) if levels else "None"
