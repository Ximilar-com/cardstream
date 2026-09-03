"""Market price statistics: parse the endpoint's ``price_stats``, summarise it.

The id endpoints (``tcg_id``, ``sport_id``, ``comics_id``) return aggregated
market prices for the best match when the request carries the top-level
``price_stats`` flag: one entry per ``stats_type`` (``ungraded``, ``graded``,
``overall``), each with the distribution (min / quartiles / median / max), the
latest and oldest sale and a trend block. Amounts are USD; the API names no
currency.

This module owns the whole thing once for the Python side: the tolerant parse
into flat dicts (the shape that crosses the WebSocket), the rule for WHICH
entries to show and in what order, and the one-line summary the headless
client prints. ``webui/shared/overlay.js`` carries a line-for-line JS twin of
the selection and the formatting, so the page and the terminal agree.

Stdlib only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

# Display order. ``overall`` is a fallback only: when a card has ungraded or
# graded sales, the blend of the two says less than either on its own.
PREFERRED_TYPES = ("ungraded", "graded")
FALLBACK_TYPE = "overall"

_NUMBER_KEYS = ("min", "max", "mean", "median", "q1", "q3", "latest", "oldest")
_DATE_KEYS = ("latest_date", "oldest_date")


def _number(value: object) -> float | None:
    """A finite float, or None for anything that is not a usable amount."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def parse_price_stats(raw: object) -> list[dict[str, Any]]:
    """Flatten the endpoint's ``price_stats`` list; anything unusable is skipped.

    Each surviving entry is ``{stats_type, interval, min, max, mean, median,
    q1, q3, latest, latest_date, oldest, oldest_date}`` — amounts as floats
    (None when absent), dates as the ISO strings the API sends. ``std``,
    ``range`` and ``trend`` are dropped: nothing here shows them. A missing or
    malformed block yields ``[]`` rather than an error, because the price is
    an extra on top of the match, never a reason to lose it.
    """
    if not isinstance(raw, list):
        return []
    entries: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        stats_type = _text(item.get("stats_type"))
        value = item.get("value")
        if stats_type is None or not isinstance(value, dict):
            continue
        if _number(value.get("median")) is None:
            continue
        entry: dict[str, Any] = {
            "stats_type": stats_type,
            "interval": _text(item.get("interval")),
        }
        for key in _NUMBER_KEYS:
            entry[key] = _number(value.get(key))
        for key in _DATE_KEYS:
            entry[key] = _text(value.get(key))
        entries.append(entry)
    return entries


def select_price_stats(
    entries: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """The entries worth showing, in display order: ungraded, then graded;
    ``overall`` only when the card has neither."""
    by_type: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        stats_type = _text(entry.get("stats_type"))
        if stats_type is not None and stats_type not in by_type:
            by_type[stats_type] = entry
    chosen = [by_type[t] for t in PREFERRED_TYPES if t in by_type]
    if not chosen and FALLBACK_TYPE in by_type:
        chosen = [by_type[FALLBACK_TYPE]]
    return chosen


def format_number(amount: float) -> str:
    """Two decimals with a whole-dollar ``.00`` dropped: 15, 24.99, 32.50."""
    text = f"{amount:.2f}"
    return text[:-3] if text.endswith(".00") else text


def money(amount: float) -> str:
    """``$`` + :func:`format_number` — the one place the currency is assumed."""
    return "$" + format_number(amount)


def price_summary(entries: Iterable[Mapping[str, Any]]) -> str:
    """One line for a history row or the terminal: per shown entry the type,
    the median and the min-max range in parentheses (left out when either
    bound is missing), joined by a middle dot — ``ungraded $24.99 (15-60)``
    then ``graded $45.00 (30-80)``. Empty when there is nothing to show.
    """
    parts: list[str] = []
    for entry in select_price_stats(entries):
        median = _number(entry.get("median"))
        if median is None:
            continue
        text = f"{entry['stats_type']} {money(median)}"
        low, high = _number(entry.get("min")), _number(entry.get("max"))
        if low is not None and high is not None:
            text += f" ({format_number(low)}\u2013{format_number(high)})"
        parts.append(text)
    return " \u00b7 ".join(parts)
