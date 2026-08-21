"""Shared data types for the streaming card-analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


class CardState(str, Enum):
    """State of the per-card tracking machine."""

    EMPTY = "empty"  # no card in frame
    MOVING = "moving"  # card present but scene still changing
    SETTLED = "settled"  # card steady, not yet identified
    IDENTIFYING = "identifying"  # tcg_id call in flight
    IDENTIFIED = "identified"  # result available, card being held


class ConfidenceTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class BoundingBox:
    """Card region in pixel coords of the analysed frame."""

    x: int
    y: int
    w: int
    h: int

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    def expanded(self, fraction: float) -> BoundingBox:
        """Grow by ``fraction`` of each dimension on EVERY side.

        0.1 pushes all four edges out by a tenth of the box, so the result is
        1.2x in each dimension — the same convention the detectors' own
        ``_CROP_MARGIN`` uses. Deliberately not clamped to any frame:
        ``FramePair.crop`` already clamps when it cuts, and clamping here would
        quietly turn the box into a different shape.
        """
        if fraction <= 0:
            return self
        dx = round(self.w * fraction)
        dy = round(self.h * fraction)
        return BoundingBox(
            x=self.x - dx, y=self.y - dy, w=self.w + 2 * dx, h=self.h + 2 * dy
        )


@dataclass
class DetectionResult:
    """What a card locator found: where the card is, and the crop to compare.

    ``prob`` is the locator's confidence in the box — every backend sets it.

    ``quad`` is the card's four corners (TL, TR, BR, BL) in the same
    analysis-frame coords as ``bbox``, and is the one signal that separates the
    two kinds of locator: a segmentor knows the card's BOUNDARY and fills it
    in, every box detector leaves it ``None``. Set, it means the identify crop
    can be deskewed (``FramePair.warp``) instead of cut square (
    ``FramePair.crop``); ``bbox`` stays the axis-aligned hull either way, so
    everything downstream of the locator is unaffected.
    """

    bbox: BoundingBox
    crop: np.ndarray
    prob: float | None = None
    quad: np.ndarray | None = None


@dataclass
class Identification:
    """A single card match from tcg_id, flattened for the client."""

    name: str
    full_name: str
    set: str
    set_code: str
    card_number: str
    series: str
    year: str
    subcategory: str
    distance: float
    confidence_tier: ConfidenceTier
    links: dict[str, str] = field(default_factory=dict)
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["confidence_tier"] = self.confidence_tier.value
        return d


# What an identification looks like to code that only passes it along: the
# server keeps the parsed Identification, the smart client the flattened dict
# its identify targets return. One name for the union both carry.
IdentificationLike = Identification | dict[str, Any]


def _quad_json(quad: np.ndarray | None) -> list[list[int]] | None:
    """Four corners as plain nested ints — this crosses a WebSocket as JSON,
    and a numpy array does not serialize."""
    return None if quad is None else [[int(x), int(y)] for x, y in quad]


@dataclass
class AnalysisResult:
    """Event emitted per processed frame. ``identification`` is only populated
    on a new-card transition; otherwise the client just gets state + bbox so it
    can keep the overlay aligned.

    ``identification`` is either an :class:`Identification` or the
    already-flattened dict the identify clients return — ``to_dict`` accepts
    both, which is why the annotation is deliberately loose.
    """

    state: CardState
    bbox: BoundingBox | None = None
    identification: IdentificationLike | None = None
    # echoed back so the client can compute round-trip latency
    frame_id: int | None = None
    # The card's four corners (TL, TR, BR, BL) in the same analysed-frame space
    # as ``bbox``, when a segmentor found them — the overlay draws this outline
    # in preference to the box. None from every box locator.
    quad: np.ndarray | None = None
    # The outline the PAID crop will be cut along, as four corners whichever
    # locator found the card — so --detection-expansion is visible on the page
    # next to what was located. None when the expansion is 0 and the two would
    # be the same shape drawn twice.
    crop_quad: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        ident = self.identification
        if isinstance(ident, Identification):
            ident = ident.to_dict()
        return {
            "state": self.state.value,
            "bbox": self.bbox.as_list() if self.bbox else None,
            "identification": ident,
            "frame_id": self.frame_id,
            "quad": _quad_json(self.quad),
            "crop_quad": _quad_json(self.crop_quad),
        }
