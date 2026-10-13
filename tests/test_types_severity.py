import pytest

from bagtear.labels import is_damage, normalize_label, type_name
from bagtear.schema import Box
from bagtear.severity import area_ratio, classify, overall


@pytest.mark.parametrize("name,kind", [
    ("Tear", "tear"), ("rip", "tear"), ("Scratch", "scratch"), ("scuff", "scratch"),
    ("Stain", "stain"), ("dirt", "stain"), ("Loose_Thread", "stitching"),
    ("broken-stitch", "stitching"), ("Hole", "hole"), ("puncture", "hole"),
    ("Damage", "damage"), ("Bag", "bag"), ("Handle", "handle"), ("Straps", None),
])
def test_normalize_label(name, kind):
    assert normalize_label(name) == kind


def test_is_damage_and_names():
    assert is_damage("stain") and is_damage("damage")
    assert not is_damage("bag") and not is_damage("handle")
    assert type_name("scratch", 2) == "scratches" and type_name("stitching") == "stitching defect"


def test_classify_boundaries():
    assert classify(0.0) == "Minor" and classify(0.0099) == "Minor"
    assert classify(0.01) == "Moderate" and classify(0.0499) == "Moderate"
    assert classify(0.05) == "Severe" and classify(1.0) == "Severe"


def test_overall_and_area_ratio():
    assert overall([]) == "None"
    assert overall(["Minor", "Severe", "Moderate"]) == "Severe"
    assert area_ratio(Box(0, 0, 10, 10), Box(0, 0, 100, 100)) == 0.01
    assert area_ratio(Box(0, 0, 500, 500), Box(0, 0, 100, 100)) == 1.0  # capped
    assert area_ratio(Box(0, 0, 10, 10), Box(0, 0, 0, 0)) == 0.0
