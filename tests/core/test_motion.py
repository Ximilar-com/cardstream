"""Tests for the cheap local gates: MotionGate + pHash + Hamming."""

from __future__ import annotations

from _helpers import make_frame, textured_crop
from cardstream.core.motion import MotionGate, hamming, phash


def test_first_frame_is_never_settled():
    gate = MotionGate(motion_threshold=4.0, still_frames_required=2)
    settled, score = gate.update(make_frame())
    assert settled is False
    assert score == 255.0  # sentinel for "no previous frame"


def test_identical_frames_settle_after_required_count():
    gate = MotionGate(motion_threshold=4.0, still_frames_required=2)
    frame = make_frame(fill=120)
    gate.update(frame)  # establishes the baseline
    assert gate.update(frame)[0] is False  # still_count = 1
    assert gate.update(frame)[0] is True  # still_count = 2 -> settled


def test_motion_resets_still_count():
    gate = MotionGate(motion_threshold=4.0, still_frames_required=2)
    frame = make_frame(fill=0)
    gate.update(frame)
    gate.update(frame)  # still_count = 1
    moved = gate.update(make_frame(fill=255))  # big diff -> reset
    assert moved[0] is False
    assert moved[1] > 4.0  # score above threshold


def test_reset_clears_state():
    gate = MotionGate(motion_threshold=4.0, still_frames_required=1)
    frame = make_frame(fill=50)
    gate.update(frame)
    gate.reset()
    # after reset the next frame is treated as the first again
    assert gate.update(frame) == (False, 255.0)


def test_phash_is_stable_for_same_image():
    crop = textured_crop(seed=1)
    assert phash(crop) == phash(crop.copy())


def test_phash_differs_for_different_images():
    a, b = textured_crop(seed=1), textured_crop(seed=2)
    assert phash(a) != phash(b)
    assert hamming(phash(a), phash(b)) > 10  # clearly distinct "cards"


def test_hamming_properties():
    a, b = phash(textured_crop(1)), phash(textured_crop(2))
    assert hamming(a, a) == 0
    assert hamming(a, b) == hamming(b, a)  # symmetric
    assert 0 <= hamming(a, b) <= 64
