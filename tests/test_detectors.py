import sys
import types

import numpy as np
import pytest
from PIL import Image

from bagtear.cli import parse_item
from bagtear.detectors import DetectorUnavailable, YoloDetector, build_detector, parse_yolo_result
from bagtear.labels import normalize_label
from bagtear.settings import Settings


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = np.array(xyxy), np.array(conf), np.array(cls)

    def __len__(self):
        return len(self.cls)


class FakeResult:
    def __init__(self, names, boxes):
        self.names, self.boxes = names, boxes


def test_parse_yolo_result_ignores_unknown_classes():
    res = FakeResult({0: "tear", 1: "bag", 2: "straps"},
                     FakeBoxes([[10, 20, 50, 60], [0, 0, 100, 200], [5, 5, 9, 9]],
                               [0.9, 0.95, 0.8], [0, 1, 2]))
    dets = parse_yolo_result(res)
    assert [d.label for d in dets] == ["tear", "bag"]
    assert dets[0].box.x2 == 50


def test_parse_yolo_empty():
    assert parse_yolo_result(FakeResult({}, None)) == []


def test_parse_yolo_typed_classes():
    names = {0: "tear", 1: "scratch", 2: "stain", 3: "stitching", 4: "hole", 5: "bag"}
    boxes = FakeBoxes([[0, 0, 100, 200], [10, 10, 30, 30], [40, 40, 60, 60], [5, 5, 9, 9],
                       [70, 70, 90, 90], [1, 1, 3, 3]], [0.95, 0.9, 0.8, 0.7, 0.6, 0.5],
                      [5, 0, 1, 2, 3, 4])
    labels = [d.label for d in parse_yolo_result(FakeResult(names, boxes))]
    assert labels == ["bag", "tear", "scratch", "stain", "stitching", "hole"]


def test_normalize_label():
    assert normalize_label("Damage") == "damage"
    assert normalize_label("Handle") == "handle"
    assert normalize_label("Straps") is None


def _install_stub_ultralytics(monkeypatch, names):
    class StubYOLO:
        def __init__(self, path):
            self.path, self.names = path, names

        def predict(self, image, conf, imgsz, verbose):
            boxes = FakeBoxes([[0, 0, 80, 100], [10, 10, 30, 30]], [0.9, 0.8], [0, 1])
            return [FakeResult(self.names, boxes)]

    mod = types.ModuleType("ultralytics")
    mod.YOLO = StubYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)


def test_yolo_detector_reports_known_types(monkeypatch, tmp_path):
    _install_stub_ultralytics(monkeypatch, {0: "bag", 1: "Loose thread", 2: "stain", 3: "Tear"})
    det = YoloDetector(tmp_path / "w.pt")
    assert det.damage_types == ["tear", "stain", "stitching"]  # our order, not the model's
    out = det.detect(Image.new("RGB", (100, 100)), 0.25)
    assert [d.label for d in out] == ["bag", "stitching"]


def test_yolo_detector_names_as_list(monkeypatch, tmp_path):
    _install_stub_ultralytics(monkeypatch, ["bag", "tear"])
    assert YoloDetector(tmp_path / "w.pt").damage_types == ["tear"]


def test_build_detector_missing_model(tmp_path):
    with pytest.raises(DetectorUnavailable) as e:
        build_detector(Settings(model_path=tmp_path / "nope.pt"))
    assert "No trained model found" in str(e.value) and "RUN_GUIDE" in str(e.value)


def test_build_detector_with_model(monkeypatch, tmp_path):
    _install_stub_ultralytics(monkeypatch, {0: "bag", 1: "tear"})
    w = tmp_path / "best.pt"
    w.write_bytes(b"x")
    assert build_detector(Settings(model_path=w)).name == "yolo"


def test_settings_from_env():
    s = Settings.from_env({"BAGTEAR_MODEL": "x/y.pt", "BAGTEAR_IMGSZ": "320"})
    assert str(s.model_path) == "x/y.pt" and s.imgsz == 320


def test_cli_parse_item_windows_path(tmp_path):
    p = tmp_path / "x_side.jpg"
    Image.new("RGB", (10, 10)).save(p)
    assert parse_item(f"{p}:back").view == "back"
    assert parse_item(str(p)).view == "side"  # guessed from file name
