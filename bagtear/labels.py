"""Map whatever class names a model/dataset uses to our classes.

Damage types: tear, scratch, stain, stitching, hole.
"damage" = damage whose type the detector does not say (a model trained with one generic class).
Other classes: bag, handle. Anything else is ignored (None).
"""
from __future__ import annotations

DAMAGE_TYPES = ("tear", "scratch", "stain", "stitching", "hole")
DAMAGE_KINDS = DAMAGE_TYPES + ("damage",)

_SYNONYMS = {
    "tear": {"tear", "tears", "torn", "rip", "rips", "ripped", "cut", "cuts", "slit"},
    "scratch": {"scratch", "scratches", "scuff", "scuffs", "scrape", "scrapes", "abrasion"},
    "stain": {"stain", "stains", "dirt", "dirty", "mark", "marks", "spot", "spots",
              "discoloration", "discolouration", "mold", "mould", "ink"},
    "stitching": {"stitching", "stitch", "stitches", "broken stitch", "broken stitching",
                  "loose thread", "thread", "seam", "seam damage", "stitch defect",
                  "stitching defect"},
    "hole": {"hole", "holes", "puncture", "punctures", "pinhole"},
    "damage": {"damage", "damaged", "defect", "defects"},
    "bag": {"bag", "bags", "handbag", "purse"},
    "handle": {"handle", "handles"},
}
_LOOKUP = {name: kind for kind, names in _SYNONYMS.items() for name in names}

_DISPLAY = {
    "tear": ("tear", "tears"),
    "scratch": ("scratch", "scratches"),
    "stain": ("stain", "stains"),
    "stitching": ("stitching defect", "stitching defects"),
    "hole": ("hole", "holes"),
    "damage": ("damage (type not classified)", "damage (type not classified)"),
}


def normalize_label(name: str):
    """Return one of tear/scratch/stain/stitching/hole/damage/bag/handle, or None to ignore."""
    n = name.strip().lower().replace("_", " ").replace("-", " ")
    return _LOOKUP.get(n)


def is_damage(label: str) -> bool:
    return label in DAMAGE_KINDS


def type_name(kind: str, n: int = 1) -> str:
    singular, plural = _DISPLAY.get(kind, (kind, kind + "s"))
    return singular if n == 1 else plural
