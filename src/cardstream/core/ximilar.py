"""Shared Ximilar API surface: confidence tiers, HTTP POST helper and
tolerant response parsing.

The parsing of the nested ``_objects -> _identification -> best_match`` shape
and the tier cutoffs live here, apart from the call that produces them, so they
can be tested against a recorded response with no HTTP in the way. What varies per id type (endpoint
URL, category attributes, games) lives in :mod:`cardstream.core.id_types`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

from cardstream.core.models import ConfidenceTier, Identification

logger = logging.getLogger("cardstream.ximilar")


@dataclass(frozen=True)
class TierThresholds:
    """Confidence tier cutoffs on best-match distance (lower = better)."""

    high_max_distance: float = 0.18
    medium_max_distance: float = 0.30


DEFAULT_TIERS = TierThresholds()


def distance_to_tier(
    distance: float, tiers: TierThresholds = DEFAULT_TIERS
) -> ConfidenceTier:
    """Map a best-match distance to High / Medium / Low."""
    if distance <= tiers.high_max_distance:
        return ConfidenceTier.HIGH
    if distance <= tiers.medium_max_distance:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.LOW


def full_image_card_object(width: int, height: int) -> dict[str, Any]:
    """Synthetic ``_objects`` entry covering the whole sent image (prob 1.0).

    Included with pre-cropped records so the id endpoints (tcg_id, sport_id, …)
    reuse our box instead of re-running their own detection.
    """
    return {"prob": 1.0, "name": "Card", "bound_box": [0, 0, int(width), int(height)]}


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    tag: str,
) -> dict[str, Any] | None:
    """POST JSON, return the decoded JSON body, or None on ANY failure.

    Handles the three failure modes uniformly (connection error, non-2xx, and a
    2xx with a non-JSON body) and logs each with the given tag. Bare
    ``requests.post`` on purpose: Ximilar drops idle keep-alive sockets, so a
    Session buys nothing.
    """
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        logger.warning("[%s] connection error: %s", tag, exc)
        return None
    if not r.ok:
        logger.warning("[%s] HTTP %s: %s", tag, r.status_code, r.text[:200])
        return None
    try:
        body = r.json()
    except ValueError:
        logger.warning("[%s] non-JSON response body: %s", tag, r.text[:200])
        return None
    return body if isinstance(body, dict) else None


def _unwrap(response: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Unwrap an optional envelope, then validate ``records[0]`` is a dict."""
    if isinstance(response.get(key), dict):
        response = response[key]
    records = response.get("records") or []
    if not records or not isinstance(records[0], dict):
        return None
    return records[0]


def all_objects(response: dict[str, Any]) -> list[tuple[str, float]]:
    """Every detected object as (name, prob) — for DEBUG visibility."""
    rec = _unwrap(response, "data")
    if rec is None:
        return []
    out: list[tuple[str, float]] = []
    for obj in rec.get("_objects") or []:
        if isinstance(obj, dict):
            out.append((str(obj.get("name", "?")), float(obj.get("prob", 0.0))))
    return out


def best_card_object(
    response: dict[str, Any], min_prob: float
) -> tuple[int, int, int, int, float] | None:
    """Return (x1, y1, x2, y2, prob) of the highest-prob ``Card`` object, or None.

    The detection response nests detections under ``records[0]._objects``; some
    proxies wrap the whole thing under ``data``. ``bound_box`` is [x1,y1,x2,y2].
    """
    rec = _unwrap(response, "data")
    if rec is None:
        return None
    best: tuple[int, int, int, int, float] | None = None
    for obj in rec.get("_objects") or []:
        if not isinstance(obj, dict) or obj.get("name") != "Card":
            continue
        prob = float(obj.get("prob", 0.0))
        if prob < min_prob:
            continue
        box = obj.get("bound_box") or []
        if len(box) != 4:
            continue
        if best is None or prob > best[4]:
            best = (int(box[0]), int(box[1]), int(box[2]), int(box[3]), prob)
    return best


def parse_best_match(
    response: dict[str, Any], tiers: TierThresholds = DEFAULT_TIERS
) -> Identification | None:
    """Walk the id-endpoint response to the best match and flatten it.

    Tolerant of the three shapes upstream handles: an account-task wrapper under
    ``response``, ``_identification`` on the record, or (the common case)
    ``_identification`` inside a detected object in ``_objects``.
    """
    rec = _unwrap(response, "response")
    if rec is None:
        return None

    ident = rec.get("_identification")
    if ident is None:
        for obj in rec.get("_objects") or []:
            if isinstance(obj, dict) and obj.get("_identification"):
                ident = obj["_identification"]
                break
    if not isinstance(ident, dict):
        return None

    best = ident.get("best_match") or {}
    distances = ident.get("distances") or []
    distance = float(distances[0]) if distances else 1.0

    return Identification(
        name=best.get("name", ""),
        full_name=best.get("full_name", ""),
        set=best.get("set", ""),
        set_code=best.get("set_code", ""),
        card_number=best.get("card_number", ""),
        series=best.get("series", ""),
        year=str(best.get("year", "")),
        subcategory=best.get("subcategory", ""),
        distance=distance,
        confidence_tier=distance_to_tier(distance, tiers),
        links=best.get("links", {}) or {},
        alternatives=[
            {
                "full_name": a.get("full_name", ""),
                "set": a.get("set", ""),
                "links": a.get("links", {}),
            }
            for a in (ident.get("alternatives") or [])[:4]
        ],
    )
