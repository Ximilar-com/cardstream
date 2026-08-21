"""Serialization tests for the shared data models."""

from __future__ import annotations

from _helpers import make_identification
from cardstream.core.models import (
    AnalysisResult,
    BoundingBox,
    CardState,
    ConfidenceTier,
)


def test_bounding_box_as_list():
    assert BoundingBox(1, 2, 3, 4).as_list() == [1, 2, 3, 4]


def test_identification_to_dict_serializes_enum():
    d = make_identification().to_dict()
    assert d["confidence_tier"] == "high"  # enum -> value
    assert d["full_name"] == "Charizard"
    assert d["links"] == {"ximilar": "https://example.com/card"}


def test_analysis_result_empty_to_dict():
    d = AnalysisResult(state=CardState.EMPTY, frame_id=7).to_dict()
    assert d == {
        "state": "empty",
        "bbox": None,
        "identification": None,
        "frame_id": 7,
        "quad": None,
        "crop_quad": None,
    }


def test_analysis_result_identified_to_dict():
    res = AnalysisResult(
        state=CardState.IDENTIFIED,
        bbox=BoundingBox(10, 20, 30, 40),
        identification=make_identification(),
        frame_id=42,
    )
    d = res.to_dict()
    assert d["state"] == "identified"
    assert d["bbox"] == [10, 20, 30, 40]
    assert d["identification"]["confidence_tier"] == "high"
    assert d["frame_id"] == 42


def test_confidence_tier_is_str_enum():
    assert ConfidenceTier.MEDIUM.value == "medium"
    assert ConfidenceTier.MEDIUM == "medium"  # str-Enum equality


def test_analysis_result_serializes_the_quad_as_plain_ints():
    """It crosses a WebSocket as JSON — a numpy array does not survive that."""
    import json

    import numpy as np

    res = AnalysisResult(
        state=CardState.SETTLED,
        bbox=BoundingBox(8, 20, 52, 100),
        quad=np.float32([[10.4, 20.6], [60, 22], [58, 120], [8, 118]]),
    )
    d = res.to_dict()
    assert d["quad"] == [[10, 20], [60, 22], [58, 120], [8, 118]]
    json.dumps(d)  # must not raise


def test_analysis_result_quad_defaults_to_none():
    """Every box locator leaves it unset; the page then draws the bbox."""
    res = AnalysisResult(state=CardState.SETTLED, bbox=BoundingBox(1, 2, 3, 4))
    assert res.to_dict()["quad"] is None


# --- BoundingBox.expanded: --detection-expansion, the box half ----------------


def test_expanded_grows_every_side_by_the_fraction():
    """0.1 pushes all four edges out by a tenth, so each dimension gains 20%."""
    assert BoundingBox(100, 200, 300, 400).expanded(0.1).as_list() == [
        70,
        160,
        360,
        480,
    ]


def test_expanded_is_a_no_op_at_zero():
    """The default must cost nothing and change nothing — same object back."""
    box = BoundingBox(10, 20, 30, 40)
    assert box.expanded(0.0) is box


def test_expanded_does_not_clamp_to_any_frame():
    """It has no frame to clamp to; FramePair.crop clamps when it cuts, and
    clamping here would quietly change the box's shape."""
    assert BoundingBox(0, 0, 100, 100).expanded(0.5).as_list() == [-50, -50, 200, 200]
