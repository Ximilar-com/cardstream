"""Card corners: mask -> quadrilateral -> deskewed rectangle.

The geometry half of the ``--segmentor`` locator, kept apart from
``detectors.py`` so it is numpy + cv2 only and tests without a model. A
segmentation model knows where the card's EDGES are, not just its box, and
these four functions are what turns that into pixels worth paying to identify:

``mask_to_quad`` finds the four corners, ``quad_bbox`` reduces them back to the
axis-aligned box every other stage still consumes (filters, tracker, overlay),
and ``warp_quad`` cuts the card out square-on.

Coordinates are plain pixels in whatever frame the mask came from — the caller
owns the analysis/full-resolution distinction (see ``FramePair.warp``).
"""

from __future__ import annotations

import cv2
import numpy as np

from cardstream.core.models import BoundingBox

# Below this a crop is not worth identifying, whichever locator produced it —
# imported by ``detectors._crop_with_margin`` so the two paths cannot drift.
MIN_CROP_SIDE = 5
# A quad derived from a mask cannot exceed the frame's diagonal, so anything
# past this multiple of the source is garbage, not a card. Guards the
# allocation in warp_quad rather than trusting the numbers. The headroom is for
# --detection-expansion: at its 1.0 maximum a card that already fills the frame
# triples, so a tighter bound would silently drop the very crops it was asked
# to grow.
_MAX_SIDE_FACTOR = 4.0


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Four points as (4, 2) float32 in TL, TR, BR, BL order.

    Sorted by angle around the centroid rather than by the usual x+y / y-x
    rule: angle-sorting always yields a non-self-intersecting cycle whatever
    order the contour arrived in, while the sum/diff rule degenerates near 45°
    — exactly the tilt a hand-held card sits at. The cycle is then rotated so
    the top-left-most corner leads, and forced clockwise in image coords (y
    down), so the caller gets one canonical winding.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    centre = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
    cycle = pts[order]
    # Clockwise on screen, which in image coords (y down) is the POSITIVE
    # oriented area — so a negative one is the cycle running the other way.
    if cv2.contourArea(cycle, True) < 0:
        cycle = cycle[::-1]
    start = int(np.argmin(cycle.sum(axis=1)))
    return np.roll(cycle, -start, axis=0).astype(np.float32)


def mask_to_quad(mask: np.ndarray) -> np.ndarray | None:
    """The card's four corners from a binary mask, or None if there is no card.

    Largest external contour (same rule as the training repo's
    ``mask_to_polygon``), then an epsilon sweep for a contour that simplifies
    to exactly four vertices — a true perspective quad, which is what makes
    deskewing worth doing. A ragged mask that never reduces to four falls back
    to the minimum-area rotated rect, which is still square-on and still beats
    an axis-aligned box.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None

    perimeter = cv2.arcLength(contour, True)
    for eps in np.arange(0.01, 0.10, 0.005):
        approx = cv2.approxPolyDP(contour, float(eps) * perimeter, True)
        if len(approx) == 4:
            return order_quad(approx.reshape(4, 2))
    return order_quad(cv2.boxPoints(cv2.minAreaRect(contour)))


def expand_quad(quad: np.ndarray, fraction: float) -> np.ndarray:
    """Push a quad's corners outward from its centroid by ``fraction``.

    The quad counterpart of :meth:`BoundingBox.expanded`, using the same
    convention so a given ``--detection-expansion`` means the same thing
    whichever locator found the card: scaling about the centroid by
    ``1 + 2 * fraction`` moves each EDGE out by ``fraction`` of the
    corresponding dimension, so 0.1 is a tenth more card on every side.

    Not clamped to the frame — ``warp_quad`` border-fills what falls outside,
    which costs a dark sliver, where clamping would shear the rectangle.
    """
    if fraction <= 0:
        return quad
    pts = np.asarray(quad, dtype=np.float32)
    centre = pts.mean(axis=0)
    return ((pts - centre) * (1.0 + 2.0 * fraction) + centre).astype(np.float32)


def quad_bbox(quad: np.ndarray) -> BoundingBox:
    """The axis-aligned hull of a quad — the box the rest of the pipeline sees.

    Detection filters, the tracker seed and the browser overlay all speak
    boxes; reducing here means none of them has to learn about polygons.
    """
    xs, ys = quad[:, 0], quad[:, 1]
    x1, y1 = int(np.floor(xs.min())), int(np.floor(ys.min()))
    x2, y2 = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
    return BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def bbox_quad(bbox: BoundingBox) -> np.ndarray:
    """A box as four corners — the inverse of :func:`quad_bbox`.

    Lets a box locator describe its crop outline in the same shape a segmentor
    does, so the page draws corners whichever locator is running.
    """
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def map_quad(
    quad: np.ndarray, from_bbox: BoundingBox, to_bbox: BoundingBox
) -> np.ndarray | None:
    """Carry corners along with the box that contains them.

    The affine that maps ``from_bbox`` onto ``to_bbox``, applied to each
    corner. What a visual tracker reports is a MOVED BOX, and this is the most
    the box can honestly say about the corners inside it: exact for a
    translation or a uniform scale, and no worse than the box itself for
    anything else — which is the standard the outline has to meet, since the
    box is what the page would otherwise fall back to drawing.

    None when the source box has no area to scale from.
    """
    if from_bbox.w <= 0 or from_bbox.h <= 0:
        return None
    sx = to_bbox.w / from_bbox.w
    sy = to_bbox.h / from_bbox.h
    moved = np.asarray(quad, dtype=np.float32).reshape(4, 2).copy()
    moved[:, 0] = to_bbox.x + (moved[:, 0] - from_bbox.x) * sx
    moved[:, 1] = to_bbox.y + (moved[:, 1] - from_bbox.y) * sy
    return moved


def paid_quad(
    bbox: BoundingBox | None, quad: np.ndarray | None, fraction: float
) -> np.ndarray | None:
    """The four corners the PAID crop is cut from, or None if nothing located.

    The one statement of how ``--detection-expansion`` applies to each kind of
    locator: corners grow about their centroid, a box about its centre, and
    both conventions move every EDGE out by ``fraction`` of that dimension. A
    caller that cuts pixels (``SmartAnalyzer._identify_crop``) picks warp or
    crop from the same ``quad is not None`` test and applies the same two
    primitives, so the outline drawn on the page and the crop that costs money
    describe the same region.
    """
    if quad is not None:
        return expand_quad(quad, fraction)
    if bbox is not None:
        return bbox_quad(bbox.expanded(fraction))
    return None


def warp_quad(image_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Deskew ``quad`` out of ``image_bgr`` into a square-on rectangle.

    The output is sized from the quad's own edges, so a card photographed at an
    angle comes back at roughly the resolution its longest edges had. Cut TIGHT
    — the ``_CROP_MARGIN`` the box detectors add exists to survive the
    background wedges an axis-aligned box drags in, and there are none here.

    Corners are deliberately not clamped to the frame: ``warpPerspective``
    border-fills whatever falls outside, which costs a dark sliver, whereas
    clamping would shear the rectangle. Returns None for a degenerate or
    implausible quad, and always a freshly allocated (owned) array otherwise.
    """
    src = order_quad(quad)
    tl, tr, br, bl = src
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    if not (np.isfinite(width) and np.isfinite(height)):
        return None
    w, h = round(width), round(height)
    fh, fw = image_bgr.shape[:2]
    if w < MIN_CROP_SIDE or h < MIN_CROP_SIDE:
        return None
    if w > fw * _MAX_SIDE_FACTOR or h > fh * _MAX_SIDE_FACTOR:
        return None

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, matrix, (w, h))
