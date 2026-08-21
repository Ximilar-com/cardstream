"""core.quad: mask -> four corners -> deskewed rectangle."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cardstream.core.models import BoundingBox
from cardstream.core.quad import (
    bbox_quad,
    expand_quad,
    map_quad,
    mask_to_quad,
    order_quad,
    paid_quad,
    quad_bbox,
    warp_quad,
)

# TL, TR, BR, BL of a portrait rectangle — the canonical order everything here
# returns, and the shape a card actually is.
PORTRAIT = np.float32([[10, 20], [60, 20], [60, 120], [10, 120]])


def _rect_mask(h, w, centre, size, angle) -> np.ndarray:
    """A filled rotated rectangle — a card's mask, without needing a model."""
    mask = np.zeros((h, w), dtype=np.uint8)
    box = cv2.boxPoints((centre, size, angle))
    cv2.fillPoly(mask, [box.astype(np.int32)], 1)
    return mask.astype(bool)


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_order_quad_is_canonical_from_any_winding(shift, reverse):
    """Contours arrive at whatever corner and in whichever direction OpenCV
    felt like — the ordering must collapse all eight of those to one."""
    pts = np.roll(PORTRAIT, shift, axis=0)
    if reverse:
        pts = pts[::-1]
    assert order_quad(pts).tolist() == PORTRAIT.tolist()


def test_order_quad_survives_a_45_degree_quad():
    """The case the usual x+y / y-x rule breaks on: at 45 degrees two corners
    tie on the sum, and a hand-held card sits near exactly that tilt."""
    diamond = np.float32([[50, 0], [100, 50], [50, 100], [0, 50]])
    ordered = order_quad(diamond)
    # Clockwise on screen, starting top-most-left-most, with no corner repeated.
    assert ordered.tolist() == diamond.tolist()
    assert cv2.contourArea(ordered, True) > 0
    assert len({tuple(p) for p in ordered.tolist()}) == 4


def test_quad_bbox_is_the_axis_aligned_hull():
    """The box every stage after the locator still speaks."""
    bbox = quad_bbox(PORTRAIT)
    assert bbox.as_list() == [10, 20, 50, 100]


def test_quad_bbox_rounds_outward():
    """Same reason FramePair.crop rounds outward — never eat the card's edge."""
    bbox = quad_bbox(
        np.float32([[10.6, 20.6], [60.4, 20.6], [60.4, 120.4], [10.6, 120.4]])
    )
    assert bbox.as_list() == [10, 20, 51, 101]


def test_mask_to_quad_recovers_a_rotated_cards_corners():
    mask = _rect_mask(400, 400, (200, 200), (120, 180), 25)
    quad = mask_to_quad(mask)
    assert quad is not None
    expected = order_quad(cv2.boxPoints(((200, 200), (120, 180), 25)))
    assert np.allclose(quad, expected, atol=3)


def test_mask_to_quad_gives_a_deskewable_quad_for_a_tilted_card():
    """The point of the whole module: the quad's own edges are the card's
    dimensions, NOT the inflated axis-aligned hull a tilt produces."""
    mask = _rect_mask(400, 400, (200, 200), (120, 180), 30)
    quad = mask_to_quad(mask)
    warped = warp_quad(np.zeros((400, 400, 3), np.uint8), quad)
    h, w = warped.shape[:2]
    assert (w, h) == pytest.approx((120, 180), abs=4)
    # The hull is markedly squarer than the card — which is exactly what makes
    # the box path's crop include background wedges.
    hull = quad_bbox(quad)
    assert hull.w > w + 20 and hull.h > h + 10


def test_mask_to_quad_falls_back_to_the_min_area_rect_on_a_ragged_mask():
    """A blob never simplifies to four vertices, but a card is still in there
    somewhere — a rotated rect beats losing the detection."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 60, 1, -1)
    quad = mask_to_quad(mask.astype(bool))
    assert quad is not None and quad.shape == (4, 2)


def test_mask_to_quad_is_none_for_an_empty_mask():
    assert mask_to_quad(np.zeros((50, 50), dtype=bool)) is None


def test_warp_quad_deskews_to_a_square_on_rectangle():
    """A rotated card painted into a frame comes back upright and tight."""
    frame = np.zeros((400, 400, 3), np.uint8)
    box = cv2.boxPoints(((200, 200), (100, 160), 20))
    cv2.fillPoly(frame, [box.astype(np.int32)], (255, 255, 255))
    warped = warp_quad(frame, box)
    h, w = warped.shape[:2]
    assert (w, h) == pytest.approx((100, 160), abs=3)
    # Tight cut: the card fills the result, instead of the ~85% an axis-aligned
    # box of the same card manages.
    assert (warped > 128).mean() > 0.97


def test_warp_quad_output_is_owned():
    """It goes to the identify thread, which must not alias the capture frame."""
    frame = np.zeros((200, 200, 3), np.uint8)
    warped = warp_quad(frame, PORTRAIT)
    assert warped.base is not frame
    frame[:] = 255
    assert warped.max() == 0


def test_warp_quad_is_none_for_a_degenerate_quad():
    frame = np.zeros((200, 200, 3), np.uint8)
    assert (
        warp_quad(frame, np.float32([[10, 10], [12, 10], [12, 12], [10, 12]])) is None
    )


def test_warp_quad_is_none_for_an_implausible_quad():
    """Guards the allocation: a garbage quad costs a skipped frame, not GBs."""
    frame = np.zeros((200, 200, 3), np.uint8)
    huge = np.float32([[0, 0], [9000, 0], [9000, 9000], [0, 9000]])
    assert warp_quad(frame, huge) is None


# --- expand_quad: --detection-expansion, the corners half ----------------------


def test_expand_quad_matches_the_bbox_convention():
    """A given --detection-expansion must mean the same thing whichever locator
    found the card, or the flag would be a different knob per backend."""
    quad = np.float32([[100, 200], [400, 200], [400, 600], [100, 600]])
    grown = quad_bbox(expand_quad(quad, 0.1))
    boxed = quad_bbox(quad).expanded(0.1)
    assert grown.w == pytest.approx(boxed.w, abs=1)
    assert grown.h == pytest.approx(boxed.h, abs=1)
    assert grown.x == pytest.approx(boxed.x, abs=1)
    assert grown.y == pytest.approx(boxed.y, abs=1)


def test_expand_quad_keeps_the_tilt():
    """It scales about the centroid, so a deskewed crop of the grown quad is
    still square-on — just with a margin of context around the card."""
    quad = order_quad(cv2.boxPoints(((200, 200), (120, 180), 30)))
    grown = expand_quad(quad, 0.25)
    assert np.allclose(grown.mean(axis=0), quad.mean(axis=0), atol=0.5)
    warped = warp_quad(np.zeros((600, 600, 3), np.uint8), grown)
    h, w = warped.shape[:2]
    assert (w, h) == pytest.approx((120 * 1.5, 180 * 1.5), abs=4)


def test_expand_quad_is_a_no_op_at_zero():
    quad = np.float32([[10, 20], [60, 20], [60, 120], [10, 120]])
    assert expand_quad(quad, 0.0) is quad


def test_expand_quad_preserves_the_corner_order():
    """The corners stay TL, TR, BR, BL. Reordering them would rotate or mirror
    the deskewed crop, which is the kind of bug that only shows up as a worse
    match rate rather than an error."""
    quad = order_quad(cv2.boxPoints(((200, 200), (120, 180), 20)))
    grown = expand_quad(quad, 0.3)
    assert np.allclose(grown, order_quad(grown), atol=0.5)
    # Each corner moved OUTWARD along its own diagonal, not to a new position
    # in the cycle: the direction from the centre is unchanged.
    centre = quad.mean(axis=0)
    for before, after in zip(quad - centre, grown - centre, strict=False):
        assert np.dot(before, after) > 0
        assert np.linalg.norm(after) > np.linalg.norm(before)


def test_expand_quad_scales_a_tilted_card_the_same_as_an_upright_one():
    """--detection-expansion must not quietly depend on how the card is held."""
    upright = np.float32([[100, 100], [220, 100], [220, 280], [100, 280]])
    tilted = order_quad(cv2.boxPoints(((160, 190), (120, 180), 35)))
    for quad in (upright, tilted):
        grown = expand_quad(quad, 0.2)
        side = np.linalg.norm(quad[1] - quad[0])
        grown_side = np.linalg.norm(grown[1] - grown[0])
        assert grown_side == pytest.approx(side * 1.4, rel=1e-3)


def test_a_quad_expanded_past_the_frame_still_warps():
    """Corners are deliberately unclamped, so growing a card at the frame edge
    border-fills instead of shearing the rectangle or returning None."""
    frame = np.zeros((200, 200, 3), np.uint8)
    edge = np.float32([[0, 0], [80, 0], [80, 120], [0, 120]])
    warped = warp_quad(frame, expand_quad(edge, 0.5))
    assert warped is not None
    h, w = warped.shape[:2]
    assert (w, h) == pytest.approx((160, 240), abs=2)


def test_bbox_quad_round_trips_through_quad_bbox():
    """A box described as corners and reduced back must be the same box —
    the page draws one shape kind, and this is the conversion that allows it."""
    box = BoundingBox(100, 80, 50, 70)
    assert quad_bbox(bbox_quad(box)).as_list() == box.as_list()


def test_bbox_quad_is_in_canonical_corner_order():
    assert bbox_quad(BoundingBox(10, 20, 50, 100)).tolist() == [
        [10, 20],
        [60, 20],
        [60, 120],
        [10, 120],
    ]


# --- map_quad: carrying corners along with a tracker's box --------------------


def test_map_quad_reproduces_a_pure_translation_exactly():
    """The common case: the card moved, its shape did not."""
    quad = np.float32([[10, 20], [60, 22], [58, 120], [8, 118]])
    moved = map_quad(quad, BoundingBox(8, 20, 52, 100), BoundingBox(108, 220, 52, 100))
    assert np.allclose(moved, quad + np.float32([100, 200]))


def test_map_quad_scales_when_the_box_grew():
    """A card coming towards the camera: the tracker's box grows, and the
    corners have to grow with it or the outline sits inside the card."""
    quad = np.float32([[0, 0], [10, 0], [10, 20], [0, 20]])
    moved = map_quad(quad, BoundingBox(0, 0, 10, 20), BoundingBox(0, 0, 20, 40))
    assert np.allclose(moved, np.float32([[0, 0], [20, 0], [20, 40], [0, 40]]))


def test_map_quad_keeps_the_tilt():
    """The whole point of carrying corners: an axis-aligned result would be
    the hull, which is what the page already had."""
    quad = np.float32([[10, 20], [60, 22], [58, 120], [8, 118]])
    moved = map_quad(quad, BoundingBox(8, 20, 52, 100), BoundingBox(28, 40, 52, 100))
    assert moved[0][1] != moved[1][1]  # top edge still slopes
    assert moved[0][0] != moved[3][0]  # left edge still leans


def test_map_quad_is_none_for_a_degenerate_source_box():
    quad = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert map_quad(quad, BoundingBox(0, 0, 0, 10), BoundingBox(5, 5, 10, 10)) is None


def test_map_quad_does_not_alias_its_input():
    quad = np.float32([[0, 0], [10, 0], [10, 20], [0, 20]])
    map_quad(quad, BoundingBox(0, 0, 10, 20), BoundingBox(99, 99, 10, 20))
    assert quad.tolist() == [[0, 0], [10, 0], [10, 20], [0, 20]]


# --- paid_quad: one statement of how --detection-expansion applies ------------


def test_paid_quad_prefers_the_corners_when_a_segmentor_found_them():
    quad = np.float32([[10, 20], [60, 22], [58, 120], [8, 118]])
    assert np.allclose(
        paid_quad(BoundingBox(8, 20, 52, 100), quad, 0.1), expand_quad(quad, 0.1)
    )


def test_paid_quad_falls_back_to_the_box_as_corners():
    box = BoundingBox(10, 20, 50, 100)
    assert np.allclose(paid_quad(box, None, 0.1), bbox_quad(box.expanded(0.1)))


def test_paid_quad_is_none_when_nothing_was_located():
    assert paid_quad(None, None, 0.1) is None
