from bagtear.location import choose_bag, locate
from bagtear.schema import Box, Detection

BAG = Box(100, 100, 700, 900)  # 600 x 800
SIZE = (800, 1000)


def tear_at(cx, cy, s=20):
    return Box(cx - s, cy - s, cx + s, cy + s)


def test_center():
    loc = locate("front", tear_at(400, 500), BAG, SIZE)
    assert loc.where == "center"
    assert loc.text == "front panel, center"


def test_bottom_left_corner_uses_seam_wording():
    loc = locate("back", tear_at(120, 880), BAG, SIZE)
    assert loc.where == "near the bottom-left seam"
    assert loc.text == "back panel, near the bottom-left seam"


def test_left_edge_only():
    loc = locate("back", tear_at(110, 500), BAG, SIZE)
    assert loc.where == "middle-left, near the left seam"


def test_top_edge_only():
    loc = locate("front", tear_at(400, 110), BAG, SIZE)
    assert loc.where == "top-center, near the top edge"


def test_grid_cell_without_edge():
    loc = locate("side", tear_at(600, 750), BAG, SIZE)
    assert loc.where == "bottom-right"


def test_near_handle():
    handle = Box(350, 20, 450, 120)
    loc = locate("front", tear_at(400, 160), BAG, SIZE, handles=[handle])
    assert "near the handle" in loc.where


def test_far_from_handle():
    handle = Box(350, 20, 450, 120)
    loc = locate("front", tear_at(400, 800), BAG, SIZE, handles=[handle])
    assert "handle" not in loc.where


def test_no_bag_falls_back_to_whole_image():
    loc = locate("front", tear_at(40, 960), None, SIZE)
    assert loc.bag_found is False
    assert loc.where == "near the bottom-left seam"


def test_tear_center_outside_bag_is_clamped():
    loc = locate("front", tear_at(50, 500), BAG, SIZE)
    assert loc.horizontal == "left"


def test_choose_bag_prefers_containing_box():
    a = Detection("bag", 0.9, Box(0, 0, 100, 100))
    b = Detection("bag", 0.5, Box(200, 200, 600, 600))
    assert choose_bag([a, b], Box(300, 300, 320, 320)) == b.box
    assert choose_bag([], Box(0, 0, 1, 1)) is None
