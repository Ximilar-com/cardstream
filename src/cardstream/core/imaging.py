"""Shared image helpers — JPEG encode/decode, rescaling, and the analysis /
full-resolution frame pair. One copy for every call site.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import cv2
import numpy as np

from cardstream.core.models import BoundingBox
from cardstream.core.quad import warp_quad

# Quality for crops/frames sent to identification endpoints (fidelity matters).
JPEG_QUALITY_ID = 90
# Quality for frames re-encoded for browser viewers (bandwidth matters).
JPEG_QUALITY_STREAM = 70

# Every ImageNet-pretrained backbone here normalizes with these — the RF-DETR
# detector blob and the MobileNetV2 embedder alike. One copy: they were written
# out in two modules, in two layouts, with no way to tell they had to agree.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def encode_jpeg(image_bgr: np.ndarray, quality: int = JPEG_QUALITY_ID) -> bytes | None:
    """Encode a BGR image to JPEG bytes, or None if encoding fails."""
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


def encode_jpeg_b64(
    image_bgr: np.ndarray, quality: int = JPEG_QUALITY_ID
) -> str | None:
    """Encode a BGR image to a base64 JPEG string (Ximilar ``_base64`` records)."""
    data = encode_jpeg(image_bgr, quality)
    return base64.b64encode(data).decode("ascii") if data is not None else None


def decode_jpeg(data: bytes) -> np.ndarray | None:
    """Decode JPEG (or any cv2-supported format) bytes to a BGR frame.

    Returns None for undecodable (or empty) input instead of raising.
    """
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def upscale_small(image_bgr: np.ndarray, min_long_edge: int = 500) -> np.ndarray:
    """Upscale images whose long edge is below ``min_long_edge`` (cubic).

    Tiny low-res card crops make the id endpoints error more (the intermittent
    500 "'_tags'") and match worse; upscaling before sending helps both.
    """
    h, w = image_bgr.shape[:2]
    long_edge = max(h, w)
    if not (0 < long_edge < min_long_edge):
        return image_bgr
    scale = float(min_long_edge) / long_edge
    return cv2.resize(
        image_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC
    )


def downscale(frame_bgr: np.ndarray, width: int) -> np.ndarray:
    """Shrink a frame to ``width`` px wide, preserving aspect.

    Returns the SAME object when nothing needs doing — ``width=0`` (full
    resolution) or a frame already narrower than the target. Never upscales:
    a 640 px webcam under a 960 px target is left alone.
    """
    h, w = frame_bgr.shape[:2]
    if width > 0 and w > width:
        scale = width / w
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    return frame_bgr


@dataclass(frozen=True)
class FramePair:
    """One captured frame in the two resolutions the pipeline needs.

    Detection, motion, tracking and the identity gate run on ``analysis``
    (cheap, and a fixed width keeps gate embeddings comparable across frames).
    The crop sent to the paid id endpoint is cut from ``full`` — that is the
    only stage where the extra pixels are worth anything.

    When no downscale was needed ``analysis is full``, so every consumer works
    unchanged on sources that were already small.
    """

    analysis: np.ndarray
    full: np.ndarray

    @classmethod
    def from_frame(cls, frame_bgr: np.ndarray, width: int) -> FramePair:
        return cls(analysis=downscale(frame_bgr, width), full=frame_bgr)

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the analysed frame — the bbox coordinate space."""
        h, w = self.analysis.shape[:2]
        return w, h

    @property
    def full_size(self) -> tuple[int, int]:
        h, w = self.full.shape[:2]
        return w, h

    @property
    def scale(self) -> float:
        """Full-resolution pixels per analysis pixel (1.0 when not downscaled)."""
        return self._scales[0]

    @property
    def _scales(self) -> tuple[float, float]:
        """(sx, sy) from analysis space to the full frame.

        MEASURED, not the configured width: a frame narrower than the target is
        not resized at all, and then the scale is exactly 1.
        """
        return (
            self.full.shape[1] / self.analysis.shape[1],
            self.full.shape[0] / self.analysis.shape[0],
        )

    def crop(self, bbox: BoundingBox) -> np.ndarray | None:
        """Cut ``bbox`` (analysis coords) out of the FULL-resolution frame.

        Rounds outward so a rescale never eats the card's border, clamps to the
        frame, and returns an OWNED array — callers hand it to a background
        identify thread, which must not alias a frame the capture loop is about
        to overwrite. None for a degenerate box.
        """
        sx, sy = self._scales
        fh, fw = self.full.shape[:2]
        x1 = max(0, math.floor(bbox.x * sx))
        y1 = max(0, math.floor(bbox.y * sy))
        x2 = min(fw, math.ceil((bbox.x + bbox.w) * sx))
        y2 = min(fh, math.ceil((bbox.y + bbox.h) * sy))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        return np.ascontiguousarray(self.full[y1:y2, x1:x2])

    def warp(self, quad: np.ndarray) -> np.ndarray | None:
        """Deskew ``quad`` (analysis coords) out of the FULL-resolution frame.

        The segmentation counterpart to :meth:`crop`: same contract in and out
        — analysis-space geometry in, an OWNED full-resolution array the
        identify thread can outlive the frame with, None when degenerate — but
        the card comes back square-on and tight, with none of the background
        wedges an axis-aligned box drags in at an angle.
        """
        scaled = np.asarray(quad, dtype=np.float32) * np.array(
            self._scales, dtype=np.float32
        )
        return warp_quad(self.full, scaled)
