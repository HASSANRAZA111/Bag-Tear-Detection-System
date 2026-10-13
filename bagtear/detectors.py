"""Detector: turns a PIL image into a list of Detection.

Labels are normalised to: tear, scratch, stain, stitching, hole (damage types), damage
(damage with no type), bag, handle. The model decides which types it knows: it can only
report a type that its training data labelled.
"""
from __future__ import annotations

from typing import List

from PIL import Image

from .labels import DAMAGE_TYPES, normalize_label
from .schema import Box, Detection
from .settings import Settings


class DetectorUnavailable(RuntimeError):
    """Raised with a human-friendly message when the detector cannot be built."""


def _to_list(x):
    x = getattr(x, "cpu", lambda: x)()
    return x.tolist()


def parse_yolo_result(result) -> List[Detection]:
    """Convert one ultralytics Results object into Detections."""
    names = result.names
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy, confs, clss = _to_list(boxes.xyxy), _to_list(boxes.conf), _to_list(boxes.cls)
    out = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
        label = normalize_label(str(names[int(k)]))
        if label:
            out.append(Detection(label, float(c), Box(x1, y1, x2, y2)))
    return out


class YoloDetector:
    name = "yolo"

    def __init__(self, weights, imgsz: int = 640):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise DetectorUnavailable("The 'ultralytics' package is not installed. "
                                      "Run: pip install -r requirements.txt") from e
        self.model = YOLO(str(weights))
        self.imgsz = imgsz

    @property
    def class_names(self) -> list:
        names = self.model.names
        return [str(v) for v in (names.values() if isinstance(names, dict) else names)]

    @property
    def damage_types(self) -> list:
        """Damage types this model was trained to report, e.g. ['tear', 'stain']."""
        found = {normalize_label(n) for n in self.class_names}
        return [t for t in DAMAGE_TYPES if t in found]

    def detect(self, image: Image.Image, conf: float = 0.25) -> List[Detection]:
        res = self.model.predict(image.convert("RGB"), conf=conf, imgsz=self.imgsz,
                                 verbose=False)
        return parse_yolo_result(res[0])


def build_detector(settings: Settings):
    if not settings.model_path.exists():
        raise DetectorUnavailable(
            f"No trained model found at '{settings.model_path}'.\n"
            "Train one (RUN_GUIDE.md, Part C) and put the file at models/best.pt.")
    return YoloDetector(settings.model_path, settings.imgsz)
