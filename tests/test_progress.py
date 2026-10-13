from bagtear.progress import steps_markdown


def test_detect_stage():
    md = steps_markdown(3, ["side", "front", "back"], "detect", 1, 3)
    assert "\u2713 Photos uploaded (3 photos)" in md
    assert "\u2713 Bag views identified (front, back, side)" in md
    assert "\u27f3 Detecting damage... (photo 2 of 3)" in md
    assert "\u25cb Generating inspection report" in md


def test_report_and_done_stages():
    assert "\u27f3 Generating inspection report..." in steps_markdown(1, ["front"], "report", 1, 1)
    done = steps_markdown(1, ["front"], "done")
    assert "\u2713 Inspection report generated" in done and "\u27f3" not in done
    assert "(1 photo)" in done
