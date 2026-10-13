"""Configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "best.pt"


@dataclass
class Settings:
    model_path: Path = DEFAULT_MODEL
    imgsz: int = 640

    @classmethod
    def from_env(cls, extra: dict | None = None) -> "Settings":
        env = dict(os.environ)
        env.update(extra or {})
        return cls(model_path=Path(env.get("BAGTEAR_MODEL", str(DEFAULT_MODEL))),
                   imgsz=int(env.get("BAGTEAR_IMGSZ", "640")))
