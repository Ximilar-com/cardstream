"""Visual single-object tracking between detections.

A tracker carries the card bbox on the frames where detection is throttled
away: the driver re-inits it from every successful detection (the detector box
is ground truth), calls ``update()`` once per frame while
``DecisionCore.tracking`` is set, and feeds the outcome back via
``DecisionCore.on_track``. A lost track (score below threshold) drops the
core back to the motion tiers and forces an immediate re-detect.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from cardstream.core.models import BoundingBox


class ObjectTracker(ABC):
    """Single-object tracker: seeded with a detector bbox, then updated per
    frame. Implementations may be re-``init``-ed any number of times."""

    name = "base"

    @abstractmethod
    def init(self, frame_bgr: np.ndarray, bbox: BoundingBox) -> None:
        """(Re)seed the tracker with the card at ``bbox`` in ``frame_bgr``."""

    @abstractmethod
    def update(self, frame_bgr: np.ndarray) -> tuple[bool, BoundingBox | None]:
        """Advance one frame. Returns (locked, bbox); ``locked=False`` means
        the target is lost and the caller should fall back to detection."""


class VitTracker(ObjectTracker):
    """OpenCV TrackerVit (~800 KB ONNX, runs on CPU via cv2.dnn).

    Exposes a real tracking score, so "lost the card" is a score drop —
    unlike e.g. NanoTrack, which reports ~0.9 no matter what.
    """

    name = "vit"

    def __init__(self, model_path: str, score_threshold: float = 0.3) -> None:
        if not hasattr(cv2, "TrackerVit_create"):
            raise RuntimeError(
                "cv2.TrackerVit is unavailable — TrackerVit needs "
                "opencv-python-headless >= 4.9"
            )
        params = cv2.TrackerVit_Params()  # type: ignore[attr-defined]
        params.net = model_path
        self._tracker = cv2.TrackerVit_create(params)
        self._score_threshold = score_threshold

    def init(self, frame_bgr: np.ndarray, bbox: BoundingBox) -> None:
        self._tracker.init(frame_bgr, (bbox.x, bbox.y, bbox.w, bbox.h))

    def update(self, frame_bgr: np.ndarray) -> tuple[bool, BoundingBox | None]:
        ok, (x, y, w, h) = self._tracker.update(frame_bgr)
        score = self._tracker.getTrackingScore()
        if not ok or score < self._score_threshold or w <= 0 or h <= 0:
            return False, None
        return True, BoundingBox(int(x), int(y), int(w), int(h))


def make_tracker(
    model: str | None, score_threshold: float = 0.3
) -> ObjectTracker | None:
    """Build a tracker from a model path; no path = tracking disabled."""
    if not model:
        return None
    return VitTracker(model, score_threshold)
