"""Why a confident detection still is not a card worth paying to identify.

The detector answers "is there a card-like object here". These answer "is it
enough of one" — a corner clipped by the frame edge, a sleeve lip and a card
caught edge-on mid-swap all come back at 0.9 confidence and identify at
nothing, and each one costs a paid call if nothing stops it.

One class per rule. ``reject`` returns the REASON it rejected — which doubles
as the debug line, so the rule owns its own wording and the driver owns none of
it — or None to let the box through. Adding a rule is a class plus one line in
:func:`make_detection_filters`; neither the analyzer nor its ``process`` loop
grows a branch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cardstream.core.models import BoundingBox


class DetectionFilter(ABC):
    """One reason a detection is not worth identifying."""

    @abstractmethod
    def reject(self, bbox: BoundingBox, frame_size: tuple[int, int]) -> str | None:
        """Why ``bbox`` is not a card, or None to accept it.

        ``frame_size`` is (width, height) of the frame the box was detected in
        — the ANALYSIS frame, so a rule expressed against it keeps its meaning
        when ``--width`` or the camera resolution changes.
        """


class MinSizeFilter(DetectionFilter):
    """Too small: under ``fraction`` of the frame in either dimension.

    Catches a card too far from the lens to read. Frame-relative rather than a
    pixel count, but it does move when someone holds a card nearer the camera —
    which is what :class:`MinAspectFilter` does not.
    """

    def __init__(self, fraction: float) -> None:
        self._fraction = fraction

    def reject(self, bbox: BoundingBox, frame_size: tuple[int, int]) -> str | None:
        width, height = frame_size
        if bbox.w >= self._fraction * width and bbox.h >= self._fraction * height:
            return None
        return (
            f"[size] box {bbox.w}x{bbox.h} under {self._fraction:.2f} of "
            f"{width}x{height} — ignored"
        )


class MinAspectFilter(DetectionFilter):
    """Wrong shape: shortest side over longest, under ``floor``.

    Orientation-blind by construction — a card is ~0.71 held portrait OR
    landscape, so one threshold covers both, and a sliver is rejected whichever
    axis it is thin on. Unlike a size rule this does not move with how close
    the card is held, which makes it the sharper test for a FRAGMENT of a card.
    """

    def __init__(self, floor: float) -> None:
        self._floor = floor

    def reject(self, bbox: BoundingBox, frame_size: tuple[int, int]) -> str | None:
        longest = max(bbox.w, bbox.h)
        if longest <= 0:
            return f"[aspect] box {bbox.w}x{bbox.h} is degenerate — ignored"
        aspect = min(bbox.w, bbox.h) / longest
        if aspect >= self._floor:
            return None
        return (
            f"[aspect] box {bbox.w}x{bbox.h} is {aspect:.2f} "
            f"(under {self._floor:.2f}) — ignored"
        )


def make_detection_filters(
    min_fraction: float, min_aspect: float
) -> tuple[DetectionFilter, ...]:
    """The ENABLED rules, in the order they are cheapest to fail.

    A threshold of 0 means the rule is not built at all, so "disabled" costs
    nothing per detection instead of a branch that always falls through.
    """
    filters: list[DetectionFilter] = []
    if min_fraction > 0:
        filters.append(MinSizeFilter(min_fraction))
    if min_aspect > 0:
        filters.append(MinAspectFilter(min_aspect))
    return tuple(filters)
