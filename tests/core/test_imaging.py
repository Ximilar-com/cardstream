"""Shared image helpers — JPEG roundtrip, the upscale guard, and the
analysis/full-resolution frame pair identification crops are cut from."""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import make_frame, textured_crop
from cardstream.core.imaging import (
    FramePair,
    decode_jpeg,
    downscale,
    encode_jpeg,
    encode_jpeg_b64,
    upscale_small,
)
from cardstream.core.models import BoundingBox


def test_encode_decode_roundtrip():
    frame = make_frame(w=64, h=48, fill=90)
    data = encode_jpeg(frame)
    assert data is not None and data[:3] == b"\xff\xd8\xff"
    decoded = decode_jpeg(data)
    assert decoded is not None and decoded.shape == (48, 64, 3)


def test_decode_garbage_and_empty_return_none():
    assert decode_jpeg(b"not a jpeg") is None
    assert decode_jpeg(b"") is None


def test_encode_b64_is_ascii_base64():
    import base64

    b64 = encode_jpeg_b64(textured_crop())
    assert b64 is not None
    assert base64.b64decode(b64)[:3] == b"\xff\xd8\xff"


def test_upscale_small_only_upscales_below_threshold():
    small = make_frame(w=100, h=64)
    up = upscale_small(small, min_long_edge=500)
    assert up.shape[1] == 500  # long edge hit the target
    assert up.shape[0] == 320  # aspect kept (64 * 5)

    big = make_frame(w=800, h=600)
    assert upscale_small(big, min_long_edge=500) is big  # untouched


def test_upscale_small_ignores_degenerate_input():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert upscale_small(empty) is empty


# --- analysis downscale --------------------------------------------------------


def frame(w: int, h: int) -> np.ndarray:
    return np.random.RandomState(0).randint(0, 255, (h, w, 3), dtype=np.uint8)


def test_downscale_preserves_aspect():
    out = downscale(frame(1920, 1080), 960)
    assert out.shape[:2] == (540, 960)


@pytest.mark.parametrize("width", [0, -1])
def test_downscale_off_returns_the_same_object(width):
    src = frame(1920, 1080)
    assert downscale(src, width) is src  # identity, not a copy


def test_downscale_never_upscales():
    """A 640 px webcam under a 960 px target is left alone — interpolating up
    would hand the detector invented pixels."""
    src = frame(640, 480)
    assert downscale(src, 960) is src


# --- FramePair ----------------------------------------------------------------


def test_pair_splits_analysis_from_full():
    pair = FramePair.from_frame(frame(1920, 1080), 960)
    assert pair.size == (960, 540)
    assert pair.full_size == (1920, 1080)
    assert pair.scale == 2.0


def test_pair_is_identity_when_no_downscale_happens():
    src = frame(320, 240)
    pair = FramePair.from_frame(src, 960)
    assert pair.analysis is pair.full is src
    assert pair.scale == 1.0
    # ...and the crop then behaves exactly as a plain slice would.
    assert pair.crop(BoundingBox(10, 10, 50, 70)).shape == (70, 50, 3)


def test_crop_rescales_to_full_resolution():
    pair = FramePair.from_frame(frame(1920, 1080), 480)  # scale 4
    crop = pair.crop(BoundingBox(x=10, y=20, w=100, h=140))
    assert crop.shape == (560, 400, 3)  # 4x the analysis box


def test_crop_rounds_outward():
    """Rounding must never eat the card's border: the box grows, not shrinks."""
    pair = FramePair.from_frame(frame(1000, 1000), 300)  # scale 3.33...
    crop = pair.crop(BoundingBox(x=10, y=10, w=100, h=100))
    assert crop.shape[0] >= 333 and crop.shape[1] >= 333


def test_crop_clamps_to_the_frame():
    pair = FramePair.from_frame(frame(1920, 1080), 960)
    crop = pair.crop(BoundingBox(x=900, y=500, w=200, h=200))  # runs off the edge
    assert crop.shape[1] == 1920 - 1800 and crop.shape[0] == 1080 - 1000


def test_crop_is_owned_not_a_view():
    """The identify thread outlives the frame — a view would be overwritten by
    the next capture."""
    src = frame(1920, 1080)
    pair = FramePair.from_frame(src, 960)
    crop = pair.crop(BoundingBox(10, 10, 100, 100))
    assert not np.shares_memory(crop, src)
    before = crop.copy()
    src[:] = 0
    assert np.array_equal(crop, before)


@pytest.mark.parametrize(
    "bbox", [BoundingBox(5, 5, 0, 0), BoundingBox(2000, 5, 10, 10)]
)
def test_crop_rejects_degenerate_boxes(bbox):
    assert FramePair.from_frame(frame(1920, 1080), 960).crop(bbox) is None


# --- FramePair.warp: the segmentation counterpart of crop ----------------------


def _quad(x, y, w, h):
    """TL, TR, BR, BL of an axis-aligned box — warp's simplest input."""
    return np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


def test_warp_rescales_analysis_coords_to_full_resolution():
    """Same contract as crop: geometry in analysis space, pixels from full."""
    pair = FramePair.from_frame(frame(1920, 1080), 480)  # scale 4
    warped = pair.warp(_quad(10, 20, 100, 140))
    assert warped.shape == (560, 400, 3)


def test_warp_deskews_a_tilted_card():
    """The reason the segmentor exists: a card at an angle comes back upright
    and the size of the CARD, not of the box that contains it."""
    import cv2

    src = np.zeros((1080, 1920, 3), np.uint8)
    corners = cv2.boxPoints(((960, 540), (400, 600), 25))
    cv2.fillPoly(src, [corners.astype(np.int32)], (255, 255, 255))
    pair = FramePair.from_frame(src, 960)  # scale 2
    warped = pair.warp(corners / 2.0)  # quad in analysis coords
    h, w = warped.shape[:2]
    assert (w, h) == pytest.approx((400, 600), abs=6)
    assert (warped > 128).mean() > 0.97  # tight: card, nothing else


def test_warp_is_owned_not_a_view():
    """It goes to the identify thread, same as crop's result."""
    src = frame(1920, 1080)
    pair = FramePair.from_frame(src, 960)
    warped = pair.warp(_quad(10, 10, 100, 100))
    assert not np.shares_memory(warped, src)
    before = warped.copy()
    src[:] = 0
    assert np.array_equal(warped, before)


def test_warp_needs_no_downscale_to_work():
    pair = FramePair.from_frame(frame(640, 480), 960)  # already small
    assert pair.warp(_quad(10, 10, 50, 70)).shape == (70, 50, 3)


def test_warp_rejects_a_degenerate_quad():
    assert FramePair.from_frame(frame(1920, 1080), 960).warp(_quad(5, 5, 0, 0)) is None
