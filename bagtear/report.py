"""Combine per-image findings into ONE report per bag, with a plain-language summary."""
from __future__ import annotations

import json
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from typing import List

from .labels import DAMAGE_KINDS, type_name
from .schema import VIEWS, Finding, ImageSummary
from .severity import LEVELS, overall

MAX_LISTED = 5  # list at most this many items per panel in the sentence; the table has all

_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
          6: "six", 7: "seven", 8: "eight", 9: "nine"}


def _num(n: int) -> str:
    return _WORDS.get(n, str(n))


def _ordered_views(views) -> list:
    views = set(views)
    return [v for v in VIEWS if v in views] + sorted(views - set(VIEWS))


def _counts(findings) -> str:
    c = Counter(f.kind for f in findings)
    order = [k for k in DAMAGE_KINDS if c[k]]
    return ", ".join(f"{c[k]} {type_name(k, c[k])}" for k in order)


def _item(f: Finding) -> str:
    return f"{type_name(f.kind, 1)}, {f.location.where} ({f.severity})"


def summarize(findings: List[Finding], images: List[ImageSummary]) -> str:
    if not images:
        return "No photos were provided."
    views_seen = {i.view for i in images}
    total = len(findings)

    if total == 0:
        names = ", ".join(_ordered_views(views_seen))
        return (f"No damage was detected in the {len(images)} photo(s) provided ({names}). "
                "Small or low-contrast damage can be missed, so check the images yourself.")

    by_view = {}
    for f in findings:
        by_view.setdefault(f.view, []).append(f)

    torn_views = sorted(by_view, key=lambda v: (-len(by_view[v]),
                        VIEWS.index(v) if v in VIEWS else 99))
    parts = [f"{total} damage area{'s' if total != 1 else ''} found on this bag "
             f"({_counts(findings)}). Overall severity: {overall(f.severity for f in findings)}."]
    for v in torn_views:
        fs = by_view[v]
        listed = "; ".join(_item(f) for f in fs[:MAX_LISTED])
        if len(fs) > MAX_LISTED:
            listed += f"; and {len(fs) - MAX_LISTED} more (see the table)"
        noun = "damage area" if len(fs) == 1 else "damage areas"
        parts.append(f"The {v} panel has {_num(len(fs))} {noun}: {listed}.")

    clean = [v for v in _ordered_views(views_seen) if v not in by_view]
    if clean:
        panels = " and ".join(clean)
        parts.append(f"The {panels} panel{'s' if len(clean) > 1 else ''} "
                     f"look{'s' if len(clean) == 1 else ''} intact.")

    if len(torn_views) > 1:
        parts.append("Note: each photo is checked separately, so damage visible in two "
                     "photos may be counted twice.")
    if any(f.kind == "damage" for f in findings):
        parts.append("The detector in use does not name the damage type for some items.")
    return " ".join(parts)


@dataclass
class BagReport:
    findings: List[Finding] = field(default_factory=list)
    images: List[ImageSummary] = field(default_factory=list)
    summary: str = ""

    @property
    def total(self) -> int:
        return len(self.findings)

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def avg_confidence(self):
        if not self.findings:
            return None
        return sum(f.conf for f in self.findings) / len(self.findings)

    @property
    def overall_severity(self) -> str:
        return overall(f.severity for f in self.findings)

    @property
    def status_title(self) -> str:
        return "Damage detected - inspection required" if self.total else "No damage detected"

    def stats_line(self, sep: str = " · ") -> str:
        """e.g. '3 photos analyzed · 2 damage areas detected · 94% avg. confidence'"""
        parts = [f"{self.n_images} photo{'s' if self.n_images != 1 else ''} analyzed",
                 f"{self.total} damage area{'s' if self.total != 1 else ''} detected"]
        if self.avg_confidence is not None:
            parts.append(f"{self.avg_confidence:.0%} avg. confidence")
        return sep.join(parts)

    def severity_line(self) -> str:
        if not self.findings:
            return "Severity: None"
        worst = max(self.findings, key=lambda f: (LEVELS.index(f.severity), f.area_pct))
        return (f"Severity: {self.overall_severity} "
                f"(largest damage covers {worst.area_pct:.1f}% of the bag)")

    def types_line(self) -> str:
        return f"Types: {_counts(self.findings)}" if self.findings else ""

    def table_rows(self) -> list:
        rows = [{"#": f.id, "View": f.view.capitalize(), "Image": f.image,
                 "Type": type_name(f.kind, 1).capitalize(),
                 "Part of bag": f.location.panel.capitalize(),
                 "Position": f.location.where, "Severity": f.severity,
                 "Area of bag": f"{f.area_pct:.1f}%", "Confidence": f"{f.conf:.0%}"}
                for f in self.findings]
        for img in self.images:
            if img.n_tears == 0:
                rows.append({"#": "-", "View": img.view.capitalize(), "Image": img.name,
                             "Type": "-", "Part of bag": f"{img.view.capitalize()} panel",
                             "Position": "No damage detected", "Severity": "-",
                             "Area of bag": "-", "Confidence": "-"})
        return rows

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "total_damage": self.total,
            "overall_severity": self.overall_severity,
            "types": dict(Counter(f.kind for f in self.findings)),
            "damage": [{"id": f.id, "type": f.kind, "view": f.view, "image": f.image,
                        "part": f.location.panel, "position": f.location.where,
                        "severity": f.severity, "area_pct_of_bag": f.area_pct,
                        "confidence": round(f.conf, 3), "box_xyxy": f.box.as_list()}
                       for f in self.findings],
            "images": [{"name": i.name, "view": i.view, "damage_count": i.n_tears,
                        "bag_outline_found": i.bag_found} for i in self.images],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_text(self, generated: str | None = None) -> str:
        """Plain-text inspection report (what the 'Download report' button saves)."""
        bar = "=" * 60
        out = ["BAG INSPECTION REPORT", bar]
        if generated:
            out.append(f"Generated: {generated}")
        out += ["", f"RESULT: {self.status_title.upper()}", self.stats_line(" | "),
                self.severity_line().upper()]
        if self.findings:
            out.append(self.types_line().upper())
        out += ["", "SUMMARY", "-" * 7, textwrap.fill(self.summary, 78), "",
                "DAMAGE LOCATIONS", "-" * 16]
        if self.findings:
            for f in self.findings:
                out.append(f"#{f.id}  {type_name(f.kind, 1).capitalize()} | "
                           f"{f.location.panel.capitalize()} | {f.location.where} | "
                           f"{f.severity} ({f.area_pct:.1f}% of bag) | "
                           f"{f.conf:.0%} confidence | {f.image}")
        else:
            out.append("No damage detected.")
        out += ["", "PHOTOS ANALYZED", "-" * 15]
        for i in self.images:
            n = f"{i.n_tears} damage area{'s' if i.n_tears != 1 else ''}" if i.n_tears else "no damage"
            out.append(f"- {i.name} ({i.view}): {n}")
        out += ["", textwrap.fill(
            "Note: positions are approximate (left/right as seen in the photo; 'seam' means "
            "close to the bag's outer edge). Severity = damage box area / bag box area "
            "(Minor under 1%, Moderate 1-5%, Severe 5% or more).", 78), ""]
        return "\n".join(out)

    def to_markdown(self) -> str:
        lines = ["# Bag inspection report", "", f"**{self.status_title}**  ",
                 self.stats_line(), "", f"{self.severity_line()}", "",
                 "## Summary", "", self.summary, "", "## Damage locations", ""]
        rows = self.table_rows()
        if rows:
            cols = list(rows[0].keys())
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "|".join("---" for _ in cols) + "|")
            for r in rows:
                lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        lines += ["", "_Positions are approximate (left/right as seen in the photo; "
                  "\"seam\" means close to the bag's outer edge)._", ""]
        return "\n".join(lines)


def build_report(findings: List[Finding], images: List[ImageSummary]) -> BagReport:
    return BagReport(findings=findings, images=images, summary=summarize(findings, images))