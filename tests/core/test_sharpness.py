"""core.sharpness — the score behind --min-sharpness.

The properties that make ONE threshold usable: blur lowers it, and nothing
else a camera does to a card moves it much — not exposure, not how large the
crop is. The absolute numbers on real crops are in the module docstring; these
are synthetic, with perfectly crisp edges, so they score a little higher.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from _helpers import printed_card
from cardstream.core.sharpness import sharpness


def out_of_focus(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigma)


def smeared(image, length, vertical=False):
    """Motion blur along one axis only — a hand still moving."""
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(image, -1, kernel.T if vertical else kernel)


def test_a_card_in_focus_scores_high_and_a_blurred_one_low():
    card = printed_card()
    assert sharpness(card) > 0.8
    assert sharpness(out_of_focus(card, 6)) < 0.5


def test_more_blur_is_always_a_lower_score():
    card = printed_card()
    scores = [sharpness(card)]
    scores += [sharpness(out_of_focus(card, sigma)) for sigma in (1, 2, 4, 8, 12)]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)


@pytest.mark.parametrize("vertical", [False, True])
def test_blur_along_one_axis_only_is_still_caught(vertical):
    """A card smeared sideways is crisp top to bottom. The score is the WORSE
    axis, not the average, or half the detail would vouch for the other half."""
    card = printed_card()
    assert sharpness(smeared(card, 25, vertical)) < 0.75 * sharpness(card)


@pytest.mark.parametrize("alpha, beta", [(0.4, 0), (0.25, 0), (1.0, 40)])
def test_exposure_does_not_move_it(alpha, beta):
    """A ratio of the crop against itself: a dim room is not a blurred card."""
    card = printed_card()
    dimmed = cv2.convertScaleAbs(card, alpha=alpha, beta=beta)
    assert sharpness(dimmed) == pytest.approx(sharpness(card), abs=0.02)


@pytest.mark.parametrize("scale", [0.5, 2.0, 3.0])
def test_the_size_of_the_crop_does_not_move_it(scale):
    """Judged at a fixed size, so the threshold means the same at 720p and 4K
    and when a card is held nearer the lens."""
    card = printed_card()
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(card, None, fx=scale, fy=scale, interpolation=interp)
    assert sharpness(resized) == pytest.approx(sharpness(card), abs=0.03)


def test_a_featureless_crop_scores_zero():
    """Nothing to blur is not "sharp" — a blank is not worth a call either."""
    assert sharpness(np.full((70, 50, 3), 128, dtype=np.uint8)) == 0.0


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 50, 3), (50, 1, 3), (2, 2, 3)])
def test_a_degenerate_crop_is_a_score_not_a_crash(shape):
    crop = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
    assert 0.0 <= sharpness(crop) <= 1.0


def test_dark_to_light_steps_count_as_much_as_light_to_dark():
    """uint8 differences wrap instead of going negative; measured that way a
    card and its mirror image would score differently."""
    card = printed_card()
    assert sharpness(card[:, ::-1]) == pytest.approx(sharpness(card), abs=1e-3)
    assert sharpness(card[::-1]) == pytest.approx(sharpness(card), abs=1e-3)
