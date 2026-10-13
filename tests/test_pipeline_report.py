import io
import json
import zipfile

from PIL import Image

from bagtear.export import result_to_zip
from bagtear.pipeline import ImageInput, analyze, guess_view, prepare_image
from bagtear.schema import Box, Detection

BAG = Detection("bag", 0.95, Box(100, 100, 700, 900))  # 600 x 800 = 480,000 px


class FakeDetector:
    name = "fake"

    def __init__(self, per_image):
        self.per_image = list(per_image)

    def detect(self, image, conf=0.25):
        return self.per_image.pop(0)


def img(w=800, h=1000):
    return Image.new("RGB", (w, h), (200, 200, 200))


def run(per_image, views):
    inputs = [ImageInput(f"{v}.jpg", v, img()) for v in views]
    return analyze(FakeDetector(per_image), inputs)  # detector is called in front/back/side order


def tear(conf=0.9, box=(300, 400, 340, 440)):
    return Detection("tear", conf, Box(*box))


def test_multi_view_report():
    front = [BAG]
    back = [BAG, Detection("tear", 0.91, Box(110, 860, 150, 900)),
            Detection("stain", 0.84, Box(380, 480, 420, 520))]
    side = [BAG, Detection("hole", 0.78, Box(380, 150, 420, 190)),
            Detection("handle", 0.8, Box(350, 40, 450, 140))]
    res = run([front, back, side], ["side", "front", "back"])  # upload order scrambled
    rep = res.report
    assert rep.total == 3
    assert [f.view for f in rep.findings] == ["back", "back", "side"]
    assert [f.kind for f in rep.findings] == ["tear", "stain", "hole"]
    assert [f.id for f in rep.findings] == [1, 2, 3]
    assert "3 damage areas found" in rep.summary
    assert "(1 tear, 1 stain, 1 hole)" in rep.summary
    assert "The back panel has two damage areas" in rep.summary
    assert "tear, near the bottom-left seam (Minor)" in rep.summary
    assert "hole" in rep.summary and "near the handle" in rep.summary
    assert "front panel looks intact" in rep.summary
    assert "counted twice" in rep.summary
    assert len(res.annotated) == 3


def test_severity_levels_from_area_ratio():
    # bag area 480,000: 40x40=1,600 (0.33%) Minor; 120x120=14,400 (3%) Moderate;
    # 200x150=30,000 (6.25%) Severe
    dets = [BAG, tear(0.9, (300, 400, 340, 440)),
            Detection("scratch", 0.8, Box(300, 200, 420, 320)),
            Detection("stain", 0.7, Box(200, 500, 400, 650))]
    rep = run([dets], ["front"]).report
    by_kind = {f.kind: f for f in rep.findings}
    assert by_kind["tear"].severity == "Minor"
    assert by_kind["scratch"].severity == "Moderate"
    assert by_kind["stain"].severity == "Severe"
    assert by_kind["stain"].area_pct == 6.25
    assert rep.overall_severity == "Severe"
    assert rep.severity_line() == "Severity: Severe (largest damage covers 6.2% of the bag)" \
        or rep.severity_line() == "Severity: Severe (largest damage covers 6.3% of the bag)"
    assert rep.types_line() == "Types: 1 tear, 1 scratch, 1 stain"


def test_no_damage():
    res = run([[BAG], [BAG]], ["front", "back"])
    rep = res.report
    assert rep.total == 0
    assert rep.summary.startswith("No damage was detected")
    assert rep.overall_severity == "None"
    assert rep.severity_line() == "Severity: None"
    assert rep.types_line() == ""
    assert any(r["Position"] == "No damage detected" for r in rep.table_rows())


def test_low_confidence_filtered():
    res = analyze(FakeDetector([[BAG, Detection("tear", 0.1, Box(1, 1, 5, 5))]]),
                  [ImageInput("a.jpg", "front", img())], conf=0.25)
    assert res.report.total == 0


def test_no_bag_uses_whole_photo_for_severity():
    # whole photo = 800x1000 = 800,000; box 200x200 = 40,000 = 5% -> Severe
    res = run([[Detection("tear", 0.9, Box(10, 700, 210, 900))]], ["front"])
    f = res.report.findings[0]
    assert f.location.bag_found is False
    assert f.severity == "Severe" and f.area_pct == 5.0
    assert "bag outline was not found" in res.report.summary


def test_untyped_damage_is_flagged():
    rep = run([[BAG, Detection("damage", 0.9, Box(300, 400, 340, 440))]], ["front"]).report
    assert rep.findings[0].kind == "damage"
    assert "type not classified" in rep.types_line()
    assert "does not name the damage type" in rep.summary


def test_exports():
    res = run([[BAG, tear(0.9)]], ["front"])
    data = json.loads(res.report.to_json())
    assert data["total_damage"] == 1 and data["damage"][0]["view"] == "front"
    assert data["damage"][0]["type"] == "tear" and data["damage"][0]["severity"] == "Minor"
    assert data["overall_severity"] == "Minor" and data["types"] == {"tear": 1}
    md = res.report.to_markdown()
    assert "| # |" in md and "Front panel" in md and "Severity" in md
    z = zipfile.ZipFile(io.BytesIO(result_to_zip(res)))
    names = z.namelist()
    assert "report.md" in names and "report.json" in names
    assert any(n.startswith("annotated/01_front") for n in names)


def test_prepare_image_resizes():
    assert max(prepare_image(img(3000, 2000), max_side=1280).size) == 1280


def test_guess_view():
    assert guess_view("IMG_back_01.JPG") == "back"
    assert guess_view("photo.png") == "front"


def test_many_damage_summary_is_capped():
    items = [Detection("tear", 0.9 - i * 0.01, Box(120 + i * 30, 400, 140 + i * 30, 420))
             for i in range(8)]
    res = run([[BAG] + items], ["front"])
    assert res.report.total == 8
    assert "and 3 more (see the table)" in res.report.summary
    assert len(res.report.table_rows()) == 8


def test_stats_line_and_status():
    res = run([[BAG, tear(0.9)], [BAG, tear(0.8)], [BAG]], ["front", "back", "side"])
    rep = res.report
    assert rep.total == 2 and rep.n_images == 3
    assert round(rep.avg_confidence, 2) == 0.85
    assert rep.stats_line() == ("3 photos analyzed · 2 damage areas detected "
                                "· 85% avg. confidence")
    assert rep.status_title.startswith("Damage detected")


def test_status_no_damage_has_no_confidence_part():
    rep = run([[BAG]], ["front"]).report
    assert rep.avg_confidence is None
    assert rep.stats_line() == "1 photo analyzed · 0 damage areas detected"
    assert rep.status_title == "No damage detected"


def test_to_text_report():
    rep = run([[BAG], [BAG, Detection("tear", 0.91, Box(110, 860, 150, 900))]],
              ["front", "back"]).report
    txt = rep.to_text(generated="2026-10-12 12:00")
    assert txt.startswith("BAG INSPECTION REPORT")
    assert "Generated: 2026-10-12 12:00" in txt
    assert "RESULT: DAMAGE DETECTED - INSPECTION REQUIRED" in txt
    assert "2 photos analyzed | 1 damage area detected | 91% avg. confidence" in txt
    assert "SEVERITY: MINOR" in txt and "TYPES: 1 TEAR" in txt
    assert "#1  Tear | Back panel | near the bottom-left seam | Minor" in txt
    assert "- front.jpg (front): no damage" in txt
    clean = run([[BAG]], ["front"]).report.to_text()
    assert "RESULT: NO DAMAGE DETECTED" in clean and "SEVERITY: NONE" in clean
    txt.encode("ascii")


def test_progress_callback_order():
    events = []
    inputs = [ImageInput(f"{v}.jpg", v, img()) for v in ("side", "front")]
    analyze(FakeDetector([[BAG], [BAG]]), inputs,
            on_progress=lambda stage, done, total: events.append((stage, done, total)))
    assert events == [("detect", 0, 2), ("detect", 1, 2), ("report", 2, 2)]
