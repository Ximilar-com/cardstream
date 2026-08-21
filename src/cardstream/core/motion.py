"""Cheap, fully-local change detection: the motion gate + identity gate.

These two signals are what let us call the paid tcg_id endpoint only once per
distinct card presentation, regardless of frame rate:

* **Motion gate** (`MotionGate`) — mean absolute difference between consecutive
  downscaled gray frames. While the scene is moving (card being placed/swapped)
  the diff is high; once it stays low for N frames the card is "settled".

* **Identity gate** (`phash` + `hamming`) — a 64-bit DCT perceptual hash of the
  settled card crop. Comparing the Hamming distance against the last *identified*
  card's hash tells us whether a settled card is the SAME one (skip) or a NEW one
  (trigger tcg_id).
"""

from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    """Tracks frame-to-frame motion and reports when the scene has settled."""

    def __init__(self, motion_threshold: float, still_frames_required: int) -> None:
        self._threshold = motion_threshold
        self._required = still_frames_required
        self._prev_small: np.ndarray | None = None
        self._still_count = 0

    def update(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        """Feed a frame. Returns (is_settled, motion_score).

        ``is_settled`` becomes True once motion has stayed below the threshold
        for ``still_frames_required`` consecutive frames.
        """
        small = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(small, (160, 90), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)

        if self._prev_small is None:
            self._prev_small = small
            return False, 255.0

        score = float(np.mean(cv2.absdiff(small, self._prev_small)))
        self._prev_small = small

        if score < self._threshold:
            self._still_count += 1
        else:
            self._still_count = 0

        return self._still_count >= self._required, score

    def reset(self) -> None:
        self._prev_small = None
        self._still_count = 0


def phash(image_bgr: np.ndarray) -> np.uint64:
    """64-bit DCT perceptual hash (pHash). No external dependency.

    Resize to 32x32 gray, take the top-left 8x8 of the DCT (low frequencies),
    threshold each coefficient against the median.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(gray)
    low = dct[:8, :8]
    med = np.median(low[1:].flatten())  # exclude DC term from the median
    bits = (low > med).flatten()
    value = np.uint64(0)
    for bit in bits:
        value = np.uint64(value << np.uint64(1)) | np.uint64(1 if bit else 0)
    return value


def hamming(a: np.uint64, b: np.uint64) -> int:
    """Hamming distance between two 64-bit hashes (0-64)."""
    return int(bin(int(a) ^ int(b)).count("1"))
