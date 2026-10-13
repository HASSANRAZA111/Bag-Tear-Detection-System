"""Bundle results (annotated images + report) into one downloadable ZIP."""
from __future__ import annotations

import io
import re
import zipfile

from .pipeline import AnalysisResult


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def result_to_zip(result: AnalysisResult) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, a in enumerate(result.annotated, 1):
            img_buf = io.BytesIO()
            a.image.save(img_buf, "JPEG", quality=92)
            z.writestr(f"annotated/{i:02d}_{a.view}_{_safe(a.name).rsplit('.', 1)[0]}.jpg",
                       img_buf.getvalue())
        z.writestr("report.md", result.report.to_markdown())
        z.writestr("report.json", result.report.to_json())
    return buf.getvalue()
