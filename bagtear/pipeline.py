"""Glue: images + view labels -> detector -> locations -> one report + annotated images."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from PIL import Image, ImageOps

from .annotate import annotate
from .labels import is_damage
from .location import choose_bag, locate
from .report import BagReport, build_report
from .schema import VIEWS, Box, Finding, ImageSummary
from .severity import area_ratio, classify


@dataclass
class ImageInput:
    name: str
    view: str          # front | back | side
    image: Image.Image


@dataclass
class AnnotatedImage:
    name: str
    view: str
    image: Image.Image
    n_tears: int


@dataclass
class AnalysisResult:
    report: BagReport
    annotated: List[AnnotatedImage]


def guess_view(filename: str) -> str:
    n = filename.lower()
    for v in VIEWS:
        if v in n:
            return v
    return "front"


def prepare_image(img: Image.Image, max_side: int = 1280) -> Image.Image:
    """Fix phone-photo rotation (EXIF), convert to RGB, shrink big photos."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.Resampling.LANCZOS)
    return img


def analyze(detector, inputs: Sequence[ImageInput], conf: float = 0.25,
            max_side: int = 1280, show_context: bool = False,
            on_progress=None) -> AnalysisResult:
    """Run the whole pipeline for ONE bag (one or more photos).

    on_progress(stage, done, total) is optional: called with ("detect", i, n) before
    each photo and ("report", n, n) once all photos are done.
    """
    if not inputs:
        raise ValueError("Provide at least one image.")
    order = sorted(range(len(inputs)),
                   key=lambda i: (VIEWS.index(inputs[i].view) if inputs[i].view in VIEWS else 99, i))

    findings: List[Finding] = []
    summaries: List[ImageSummary] = []
    annotated: List[AnnotatedImage] = []
    next_id = 1

    for n_done, idx in enumerate(order):
        if on_progress:
            on_progress("detect", n_done, len(order))
        item = inputs[idx]
        img = prepare_image(item.image, max_side)
        dets = detector.detect(img, conf)
        tears = sorted((d for d in dets if is_damage(d.label) and d.conf >= conf),
                       key=lambda d: -d.conf)
        bags = [d for d in dets if d.label == "bag"]
        handles = [d.box for d in dets if d.label == "handle"]

        img_findings, bag_found = [], True
        for t in tears:
            bag_box = choose_bag(bags, t.box)
            bag_found = bag_found and bag_box is not None
            loc = locate(item.view, t.box, bag_box, img.size, handles)
            ref = bag_box if bag_box is not None else Box(0, 0, img.width, img.height)
            ratio = area_ratio(t.box, ref)
            img_findings.append(Finding(next_id, item.view, item.name, t.conf, t.box, loc,
                                        kind=t.label, severity=classify(ratio),
                                        area_pct=round(ratio * 100, 2)))
            next_id += 1

        findings.extend(img_findings)
        summaries.append(ImageSummary(item.name, item.view, len(img_findings), bag_found))
        context = [d for d in dets if d.label in ("bag", "handle")] if show_context else []
        annotated.append(AnnotatedImage(item.name, item.view,
                                        annotate(img, img_findings, context), len(img_findings)))

    if on_progress:
        on_progress("report", len(order), len(order))
    return AnalysisResult(build_report(findings, summaries), annotated)
