"""Command line:  python -m bagtear.cli front.jpg:front back.jpg:back --out results/"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from .detectors import DetectorUnavailable, build_detector
from .pipeline import ImageInput, analyze, guess_view
from .schema import VIEWS
from .settings import Settings


def parse_item(arg: str) -> ImageInput:
    path, view = arg, None
    if ":" in arg:
        head, tail = arg.rsplit(":", 1)
        if tail.lower() in VIEWS:  # keeps Windows paths like C:\x.jpg working
            path, view = head, tail.lower()
    p = Path(path)
    return ImageInput(p.name, view or guess_view(p.name), Image.open(p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Detect damage on bag photos.")
    ap.add_argument("images", nargs="+", help="image paths, optionally 'path:front|back|side'")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--context", action="store_true", help="also draw bag/handle boxes")
    args = ap.parse_args(argv)

    try:
        detector = build_detector(Settings.from_env())
    except DetectorUnavailable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    result = analyze(detector, [parse_item(a) for a in args.images],
                     conf=args.conf, show_context=args.context)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(result.annotated, 1):
        a.image.save(out / f"{i:02d}_{a.view}_{Path(a.name).stem}.jpg", quality=92)
    (out / "report.md").write_text(result.report.to_markdown(), encoding="utf-8")
    (out / "report.json").write_text(result.report.to_json(), encoding="utf-8")
    print(result.report.summary)
    print(f"\nSaved annotated images and report to: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
