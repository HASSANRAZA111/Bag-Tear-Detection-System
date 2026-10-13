"""Plain data classes shared by all modules (no heavy dependencies)."""
from __future__ import annotations

import math
from dataclasses import dataclass

VIEWS = ("front", "back", "side")


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(self.x2 - self.x1, 0.0)

    @property
    def h(self) -> float:
        return max(self.y2 - self.y1, 0.0)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def gap_to(self, other: "Box") -> float:
        """Shortest distance between the two boxes (0 if they touch or overlap)."""
        dx = max(other.x1 - self.x2, self.x1 - other.x2, 0.0)
        dy = max(other.y1 - self.y2, self.y1 - other.y2, 0.0)
        return math.hypot(dx, dy)

    def as_list(self) -> list[float]:
        return [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)]


@dataclass(frozen=True)
class Detection:
    label: str  # normalised: "tear" | "bag" | "handle"
    conf: float
    box: Box


@dataclass(frozen=True)
class Location:
    panel: str  # "front panel" | "back panel" | "side panel"
    where: str  # e.g. "bottom-left, near the left seam"
    text: str  # e.g. "back panel, bottom-left, near the left seam"
    vertical: str  # top | middle | bottom
    horizontal: str  # left | center | right
    near: tuple = ()
    bag_found: bool = True


@dataclass
class Finding:
    id: int
    view: str
    image: str
    conf: float
    box: Box
    location: Location
    kind: str = "tear"        # tear | scratch | stain | stitching | hole | damage
    severity: str = "Minor"   # Minor | Moderate | Severe
    area_pct: float = 0.0     # damage area as % of bag area


@dataclass
class ImageSummary:
    name: str
    view: str
    n_tears: int = 0
    bag_found: bool = True
