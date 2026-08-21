"""Detection filters: pure rules, tested without an analyzer or a camera."""

from __future__ import annotations

import pytest

from cardstream.core.detection_filters import (
    MinAspectFilter,
    MinSizeFilter,
    make_detection_filters,
)
from cardstream.core.models import BoundingBox

FRAME = (1920, 1080)


def box(w, h, x=0, y=0):
    return BoundingBox(x=x, y=y, w=w, h=h)


# --- MinSizeFilter -----------------------------------------------------------


def test_a_card_that_fills_enough_of_the_frame_passes():
    assert MinSizeFilter(0.2).reject(box(550, 702), FRAME) is None


def test_either_dimension_can_reject():
    """Narrow OR short — a card at the edge of shot fails on one axis only."""
    narrow = MinSizeFilter(0.2).reject(box(214, 702), FRAME)
    short = MinSizeFilter(0.2).reject(box(700, 100), FRAME)
    assert narrow is not None and short is not None


def test_the_reason_names_the_numbers_that_decided_it():
    """The reason IS the debug line, so it has to be readable on its own."""
    reason = MinSizeFilter(0.2).reject(box(214, 702), FRAME)
    assert "214x702" in reason and "0.20" in reason and "1920x1080" in reason


def test_the_same_box_is_judged_against_the_frame_it_came_from():
    """Frame-relative: a box that is a sliver at 1920 is a card at 640."""
    rule = MinSizeFilter(0.2)
    assert rule.reject(box(214, 702), (1920, 1080)) is not None
    assert rule.reject(box(214, 702), (640, 720)) is None


# --- MinAspectFilter ---------------------------------------------------------


def test_a_card_shape_passes_either_way_up():
    """0.71 portrait and landscape are the SAME card — which is the whole point
    of short-side-over-long rather than width-over-height."""
    rule = MinAspectFilter(0.4)
    assert rule.reject(box(550, 702), FRAME) is None
    assert rule.reject(box(702, 550), FRAME) is None


@pytest.mark.parametrize("w,h", [(214, 702), (702, 214)])
def test_a_sliver_is_rejected_on_either_axis(w, h):
    assert MinAspectFilter(0.4).reject(box(w, h), FRAME) is not None


def test_the_real_sliver_reports_its_ratio():
    reason = MinAspectFilter(0.4).reject(box(214, 702), FRAME)
    assert "0.30" in reason and "0.40" in reason


def test_a_degenerate_box_is_rejected_rather_than_dividing_by_zero():
    assert MinAspectFilter(0.4).reject(box(0, 0), FRAME) is not None


def test_aspect_ignores_the_frame_entirely():
    """Holding a card nearer the lens must not change the verdict — the reason
    to prefer this rule over a size one."""
    rule = MinAspectFilter(0.4)
    assert rule.reject(box(550, 702), (1920, 1080)) is None
    assert rule.reject(box(55, 70), (1920, 1080)) is None


# --- make_detection_filters --------------------------------------------------


def test_a_zero_threshold_builds_no_rule_at_all():
    """Disabled costs nothing per detection, rather than a branch that always
    falls through."""
    assert make_detection_filters(0.0, 0.0) == ()
    assert [type(f) for f in make_detection_filters(0.2, 0)] == [MinSizeFilter]
    assert [type(f) for f in make_detection_filters(0, 0.4)] == [MinAspectFilter]


def test_both_enabled_keeps_size_first():
    assert [type(f) for f in make_detection_filters(0.2, 0.4)] == [
        MinSizeFilter,
        MinAspectFilter,
    ]
