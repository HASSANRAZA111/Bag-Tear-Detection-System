"""Merge one or more YOLO-format datasets into ONE clean dataset for training.

What it does
  1. Reads every source folder (images + YOLO .txt labels, boxes OR polygons).
  2. Renames classes with configs/class_map.yaml (Damage -> tear, Loose thread -> stitching ...)
     and ignores the rest (it tells you which classes it ignored).
  3. Converts polygon labels to boxes.
  4. Splits train/val/test BY ORIGINAL PHOTO, so augmented copies of one photo never end up
     in both train and val (that would inflate your scores).
  5. Writes <out>/data.yaml ready for ultralytics, and warns about damage types that have
     no (or too few) boxes, because a model cannot learn a type it never sees.

Usage
  # 1) see what classes your downloaded datasets contain and how they map:
  python training/prepare_dataset.py --src "data/raw/*" --list
  # 2) build the merged dataset:
  python training/prepare_dataset.py --src "data/raw/*" --out data/bag_tear
"""
from __future__ import annotations

import argparse
import glob
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WEAK_BELOW = 100  # fewer boxes than this for a type -> warning


# ----------------------------------------------------------------- helpers
def expand_sources(patterns) -> list:
    """Accept folders and glob patterns (so 'data/raw/*' also works in PowerShell)."""
    out = []
    for p in patterns:
        s = str(p)
        found = sorted(Path(x) for x in glob.glob(s)) if any(c in s for c in "*?[") else [Path(s)]
        out += [f for f in found if f.is_dir()]
    seen, uniq = set(), []
    for f in out:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            uniq.append(f)
    return uniq


def _names_from_yaml(path: Path):
    try:
        names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("names")
    except Exception:
        return None
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    if isinstance(names, list):
        return [str(n) for n in names]
    return None


def load_names(src: Path) -> list:
    """Class names of a source dataset (data.yaml anywhere inside, else classes.txt)."""
    yamls = [src / "data.yaml", *sorted(src.rglob("data.yaml")),
             *sorted(src.glob("*.yaml")), *sorted(src.glob("*.yml"))]
    for y in yamls:
        if y.exists():
            names = _names_from_yaml(y)
            if names:
                return names
    for t in ("classes.txt", "obj.names", "labels.txt"):
        for f in [src / t, *sorted(src.rglob(t))]:
            if f.exists():
                return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    raise SystemExit(
        f"[{src}] cannot find class names. Put a data.yaml (with 'names:') or a "
        f"classes.txt (one name per line, in class-id order) in that folder.")


def find_label(img: Path):
    parts = list(img.parts)
    if "images" in parts:  # .../images/x.jpg -> .../labels/x.txt
        i = len(parts) - 1 - parts[::-1].index("images")
        cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
        if cand.exists():
            return cand
    same = img.with_suffix(".txt")
    return same if same.exists() else None


def parse_label_line(line: str):
    """-> (class_id, cx, cy, w, h) normalised, or None. Handles boxes and polygons."""
    t = line.split()
    if len(t) < 5:
        return None
    try:
        cid, vals = int(float(t[0])), [float(v) for v in t[1:]]
    except ValueError:
        return None
    if len(vals) == 4:
        return (cid, *vals)
    if len(vals) >= 6 and len(vals) % 2 == 0:  # polygon -> bounding box
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return (cid, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    return None


_SUFFIXES = [r"\.rf\.[0-9a-fA-F]+$", r"_(jpe?g|png)$", r"[-_](aug|flip|rot|noise|copy)\w*$"]


def group_key(src_name: str, stem: str) -> str:
    """Photos that are augmented copies of each other share a key."""
    s = stem
    for _ in range(3):
        for pat in _SUFFIXES:
            s = re.sub(pat, "", s)
    return f"{src_name}::{s}"


def list_sources(sources, cmap) -> None:
    """--list: show each dataset's classes, box counts and how they will be mapped."""
    for src in sources:
        names = load_names(src)
        boxes, images = Counter(), 0
        for img in (p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT):
            lab = find_label(img)
            if lab is None:
                continue
            images += 1
            for raw in lab.read_text(encoding="utf-8").splitlines():
                p = parse_label_line(raw)
                if p and p[0] < len(names):
                    boxes[names[p[0]]] += 1
        print(f"\n[{src.name}]  {images} labelled images")
        for n in names:
            tgt = cmap.get(n.lower())
            arrow = f"-> {tgt}" if tgt else "-> (ignored: not in configs/class_map.yaml)"
            print(f"   {n!r:32} {boxes[n]:6d} boxes   {arrow}")


# -------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True,
                    help="dataset folders; globs like 'data/raw/*' are allowed")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--map", default="configs/class_map.yaml", type=Path)
    ap.add_argument("--list", action="store_true",
                    help="only print each dataset's classes and how they map, then exit")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-unlabeled", action="store_true",
                    help="keep images with no label file as empty negatives (intact bags)")
    args = ap.parse_args(argv)

    sources = expand_sources(args.src)
    if not sources:
        raise SystemExit(f"No dataset folders found for: {args.src}")
    cfg = yaml.safe_load(args.map.read_text(encoding="utf-8"))
    targets: list = cfg["target_classes"]
    cmap = {str(k).lower(): v for k, v in cfg["map"].items() if v in targets}
    if args.list:
        list_sources(sources, cmap)
        return
    if args.out is None:
        raise SystemExit("--out is required (unless you use --list)")
    tid = {n: i for i, n in enumerate(targets)}

    samples = []  # (img_path, [label lines], group)
    stats, ignored = Counter(), Counter()
    for src in sources:
        names = load_names(src)
        print(f"[{src.name}] source classes: {names}")
        for img in sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT):
            lab = find_label(img)
            if lab is None:
                if args.keep_unlabeled:
                    samples.append((img, [], group_key(src.name, img.stem)))
                else:
                    stats["skipped_no_label_file"] += 1
                continue
            lines = []
            for raw in lab.read_text(encoding="utf-8").splitlines():
                p = parse_label_line(raw)
                if not p or p[0] >= len(names):
                    continue
                tgt = cmap.get(names[p[0]].lower())
                if tgt is None:
                    ignored[names[p[0]]] += 1
                    continue
                cx, cy, w, h = (min(max(v, 0.0), 1.0) for v in p[1:])
                lines.append(f"{tid[tgt]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if lines or args.keep_unlabeled:
                samples.append((img, lines, group_key(src.name, img.stem)))
            else:
                stats["skipped_no_usable_boxes"] += 1

    if not samples:
        raise SystemExit("No usable images found. Check --src folders and class_map.yaml.")

    groups = defaultdict(list)
    for s in samples:
        groups[s[2]].append(s)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    n = len(keys)
    n_test, n_val = round(n * args.test), round(n * args.val)
    split_of = {}
    for i, k in enumerate(keys):
        split_of[k] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    if args.out.exists():
        shutil.rmtree(args.out)
    per_split, per_class = Counter(), defaultdict(Counter)
    for k, items in groups.items():
        sp = split_of[k]
        for j, (img, lines, _) in enumerate(items):
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{k.replace('::', '_')}_{j}")
            (args.out / "images" / sp).mkdir(parents=True, exist_ok=True)
            (args.out / "labels" / sp).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, args.out / "images" / sp / f"{stem}{img.suffix.lower()}")
            (args.out / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines))
            per_split[sp] += 1
            for l in lines:
                per_class[sp][targets[int(l.split()[0])]] += 1

    data = {"path": str(args.out.resolve()), "train": "images/train", "val": "images/val",
            "names": {i: n for i, n in enumerate(targets)}}
    if per_split["test"]:
        data["test"] = "images/test"
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    print(f"\nPhoto groups: {n} | images written: {sum(per_split.values())}")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s}: {per_split[sp]:5d} images, boxes: {dict(per_class[sp])}")
    for k, v in stats.items():
        print(f"  note: {k} = {v}")
    if ignored:
        print("\nIgnored classes (not in configs/class_map.yaml; add them if you want them):")
        for name, c in ignored.most_common():
            print(f"   {name!r}: {c} boxes")

    totals = Counter()
    for sp in per_class.values():
        totals.update(sp)
    print("\nBoxes per class (all splits): " + ", ".join(f"{t} {totals[t]}" for t in targets))
    for name in targets:
        if name == "bag":
            continue
        if totals[name] == 0:
            print(f"  WARNING: no '{name}' boxes found in any source, so the model cannot "
                  f"learn '{name}'. Add a dataset that labels it (RUN_GUIDE.md, Part C).")
        elif totals[name] < WEAK_BELOW:
            print(f"  WARNING: only {totals[name]} '{name}' boxes (aim for {WEAK_BELOW}+). "
                  f"'{name}' will be weak and unreliable.")
    print(f"\nReady: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
