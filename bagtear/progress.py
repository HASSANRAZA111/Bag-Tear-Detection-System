"""Text for the 'processing status' checklist shown while analysing."""
from __future__ import annotations

from .schema import VIEWS


def _views_text(views) -> str:
    seen = set(views)
    return ", ".join([v for v in VIEWS if v in seen] + sorted(seen - set(VIEWS)))


def steps_markdown(n_photos: int, views, stage: str, done: int = 0, total: int = 0) -> str:
    """stage: 'detect' (working on photo done+1 of total) | 'report' | 'done'."""
    photos = f"{n_photos} photo{'s' if n_photos != 1 else ''}"
    lines = [f"\u2713 Photos uploaded ({photos})",
             f"\u2713 Bag views identified ({_views_text(views)})"]
    if stage == "detect":
        lines += [f"\u27f3 Detecting damage... (photo {min(done + 1, max(total, 1))} of {total})",
                  "\u25cb Generating inspection report"]
    elif stage == "report":
        lines += ["\u2713 Damage detection done", "\u27f3 Generating inspection report..."]
    else:
        lines += ["\u2713 Damage detection done", "\u2713 Inspection report generated"]
    return "\n\n".join(lines)
