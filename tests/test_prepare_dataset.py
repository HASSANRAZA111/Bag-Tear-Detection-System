import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import prepare_dataset as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def make_src(root: Path, name: str, names, items):
    src = root / name
    (src / "train" / "images").mkdir(parents=True)
    (src / "train" / "labels").mkdir(parents=True)
    (src / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    for stem, lines in items:
        Image.new("RGB", (64, 64)).save(src / "train" / "images" / f"{stem}.jpg")
        if lines is not None:
            (src / "train" / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    return src


def test_parse_label_line_box_and_polygon():
    assert pd.parse_label_line("1 0.5 0.5 0.2 0.4") == (1, 0.5, 0.5, 0.2, 0.4)
    cid, cx, cy, w, h = pd.parse_label_line("0 0.1 0.1 0.3 0.1 0.3 0.5 0.1 0.5")
    assert (cid, round(cx, 2), round(cy, 2), round(w, 2), round(h, 2)) == (0, 0.2, 0.3, 0.2, 0.4)
    assert pd.parse_label_line("bad") is None


def test_group_key_strips_roboflow_and_aug():
    a = pd.group_key("s", "bag1_jpg.rf.abc123")
    b = pd.group_key("s", "bag1_jpg.rf.def456")
    assert a == b


def test_end_to_end_merge(tmp_path):
    # Source 1: Roboflow-like, names Damage/Bag/Handle/Straps; 3 augmented copies of one photo
    s1 = make_src(tmp_path, "rf", ["Bag", "Damage", "Handle", "Straps"], [
        ("a_jpg.rf.111", ["0 0.5 0.5 0.9 0.9", "1 0.3 0.3 0.1 0.1", "2 0.5 0.05 0.2 0.1"]),
        ("a_jpg.rf.222", ["0 0.5 0.5 0.9 0.9", "1 0.3 0.3 0.1 0.1"]),
        ("a_jpg.rf.333", ["1 0.3 0.3 0.1 0.1"]),
        ("b_jpg.rf.444", ["3 0.5 0.5 0.1 0.1"]),      # only an unmapped class -> skipped
        ("c_jpg.rf.555", None),                        # no label file -> skipped
    ])
    # Source 2: polygon labels, names damage/bag
    s2 = make_src(tmp_path, "mend", ["damage", "bag"], [
        ("m1", ["0 0.1 0.1 0.3 0.1 0.3 0.5 0.1 0.5", "1 0.5 0.5 0.8 0.8"]),
        ("m2", ["0 0.4 0.4 0.1 0.1"]),
    ])
    out = tmp_path / "out"
    pd.main(["--src", str(s1), str(s2), "--out", str(out), "--map",
             str(ROOT / "configs" / "class_map.yaml"), "--val", "0.25", "--test", "0.25"])

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "tear", 1: "scratch", 2: "stain", 3: "stitching", 4: "hole", 5: "bag"}

    # All 3 augmented copies of photo 'a' must sit in the same split (no leakage)
    where = {}
    for sp in ("train", "val", "test"):
        d = out / "images" / sp
        if d.exists():
            for f in d.iterdir():
                where.setdefault("a" if "rf_a" in f.name else f.name, set()).add(sp)
    assert len(where["a"]) == 1

    # Labels only contain class ids 0/1 and are 5 columns
    for lab in (out / "labels").rglob("*.txt"):
        for line in lab.read_text().splitlines():
            parts = line.split()
            assert len(parts) == 5 and parts[0] in ("0", "5")  # tear=0, bag=5
    total = sum(1 for _ in (out / "images").rglob("*.jpg"))
    assert total == 5  # 3 (a) + 2 (m1, m2); b and c skipped


def _typed_src(tmp_path, name, names, items):
    """items: list of (stem, [label lines]); data.yaml is placed one folder DEEPER on purpose."""
    root = tmp_path / name
    inner = root / "export"
    (inner / "train" / "images").mkdir(parents=True)
    (inner / "train" / "labels").mkdir(parents=True)
    (inner / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    for stem, lines in items:
        Image.new("RGB", (32, 32)).save(inner / "train" / "images" / f"{stem}.jpg")
        (inner / "train" / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    return root


def test_glob_sources_and_nested_yaml_and_ignored_report(tmp_path, capsys):
    raw = tmp_path / "raw"
    _typed_src(raw, "tdi", ["hole", "stain", "Loose thread", "tear", "clothes"],
               [(f"a{i}_jpg.rf.{i:08x}", ["0 0.5 0.5 0.1 0.1", "2 0.3 0.3 0.1 0.1",
                                          "4 0.5 0.5 0.9 0.9"]) for i in range(6)])
    _typed_src(raw, "leather", ["Pin hole", "Leather cut", "vein mark"],
               [(f"b{i}_jpg.rf.{i + 50:08x}", ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1",
                                               "2 0.4 0.4 0.1 0.1"]) for i in range(6)])
    out = tmp_path / "out"
    pd.main(["--src", str(raw / "*"), "--out", str(out), "--map",
             str(ROOT / "configs" / "class_map.yaml")])
    text = capsys.readouterr().out
    assert "Ignored classes" in text and "'clothes'" in text and "'vein mark'" in text
    assert "Boxes per class" in text
    assert "no 'scratch' boxes" in text          # nothing labelled scratch in these sources
    assert "only 6 'stitching' boxes" in text    # present but far below 100
    ids = {line.split()[0] for f in (out / "labels").rglob("*.txt")
           for line in f.read_text().splitlines()}
    assert ids <= {"0", "2", "3", "4"}           # tear, stain, stitching, hole only (no bag here)
    assert "1" not in ids and "5" not in ids     # never scratch / bag


def test_list_mode_prints_mapping_and_writes_nothing(tmp_path, capsys):
    raw = tmp_path / "raw"
    _typed_src(raw, "x", ["Damage", "Clothes"], [("a_jpg.rf.00000001", ["0 0.5 0.5 0.1 0.1",
                                                                         "1 0.5 0.5 0.9 0.9"])])
    pd.main(["--src", str(raw / "x"), "--list", "--map", str(ROOT / "configs" / "class_map.yaml")])
    text = capsys.readouterr().out
    assert "'Damage'" in text and "-> tear" in text
    assert "'Clothes'" in text and "ignored" in text
    assert not (tmp_path / "out").exists()


def test_class_map_file_covers_checked_dataset_names():
    cfg = yaml.safe_load((ROOT / "configs" / "class_map.yaml").read_text())
    m = {str(k).lower(): v for k, v in cfg["map"].items()}
    assert m["loose thread"] == "stitching" and m["pin hole"] == "hole"
    assert m["leather cut"] == "tear" and m["cut"] == "tear" and m["scratch"] == "scratch"
    assert m["damage"] == "tear" and m["stain"] == "stain" and m["hole"] == "hole"
    assert set(cfg["target_classes"]) == {"tear", "scratch", "stain", "stitching", "hole", "bag"}
