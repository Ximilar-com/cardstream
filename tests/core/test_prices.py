"""core.prices — the endpoint's ``price_stats``, parsed once and summarised once.

``webui/shared/overlay.js`` carries the JS twin of the selection and the
formatting, and ``tests/webui/price-stats.test.js`` runs the same cases against
it; a rule that changes here changes there.
"""

from __future__ import annotations

import pytest

from cardstream.core.prices import (
    format_number,
    money,
    parse_price_stats,
    price_summary,
    select_price_stats,
)


def _entry(stats_type, **value):
    """One raw entry as the endpoint sends it."""
    return {
        "stats_type": stats_type,
        "interval": "overall",
        "start_date": None,
        "value": value,
    }


_GRADED = _entry(
    "graded",
    min=15.0,
    max=60.0,
    mean=29.12,
    median=24.99,
    q1=18.93,
    q3=32.5,
    std=14.52,
    range=45.0,
    latest=55,
    oldest=17.5,
    latest_date="2026-02-22",
    oldest_date="2024-12-12",
    trend={"slope_per_day": 0.0101, "forecast_30d": 55.3},
)


# ---------------------------------------------------------------------- parse


def test_parse_flattens_one_entry_per_stats_type():
    (entry,) = parse_price_stats([_GRADED])
    assert entry == {
        "stats_type": "graded",
        "interval": "overall",
        "min": 15.0,
        "max": 60.0,
        "mean": 29.12,
        "median": 24.99,
        "q1": 18.93,
        "q3": 32.5,
        "latest": 55.0,
        "oldest": 17.5,
        "latest_date": "2026-02-22",
        "oldest_date": "2024-12-12",
    }
    assert isinstance(entry["latest"], float)  # the int the API sent, coerced


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        "nope",
        [],
        [None, 3, "x"],
        [{"stats_type": "graded"}],  # no value block
        [{"value": {"median": 1}}],  # no stats type
        [_entry("", median=1)],
        [_entry("graded", min=1, max=2)],  # no median -> nothing to show
        [_entry("graded", median="24.99")],
        [_entry("graded", median=True)],
        [_entry("graded", median=float("nan"))],
    ],
)
def test_parse_skips_anything_unusable(raw):
    assert parse_price_stats(raw) == []


def test_parse_tolerates_missing_fields():
    (entry,) = parse_price_stats([{"stats_type": "ungraded", "value": {"median": 4}}])
    assert entry["median"] == 4.0
    assert entry["min"] is None and entry["latest_date"] is None
    assert entry["interval"] is None


# --------------------------------------------------------------------- select


def test_select_prefers_ungraded_then_graded():
    entries = parse_price_stats(
        [_GRADED, _entry("overall", median=30), _entry("ungraded", median=20)]
    )
    assert [e["stats_type"] for e in select_price_stats(entries)] == [
        "ungraded",
        "graded",
    ]


def test_select_falls_back_to_overall_only_when_alone():
    only = parse_price_stats([_entry("overall", median=30)])
    assert [e["stats_type"] for e in select_price_stats(only)] == ["overall"]
    assert select_price_stats([]) == []


# --------------------------------------------------------------------- format


@pytest.mark.parametrize(
    ("amount", "text"),
    [(15.0, "15"), (24.99, "24.99"), (32.5, "32.50"), (0.5, "0.50"), (1234.0, "1234")],
)
def test_format_number_drops_only_a_whole_dollar_fraction(amount, text):
    assert format_number(amount) == text
    assert money(amount) == "$" + text


def test_summary_is_median_and_range_per_type():
    entries = parse_price_stats(
        [_GRADED, _entry("ungraded", min=3, max=9.5, median=4.5)]
    )
    assert (
        price_summary(entries)
        == "ungraded $4.50 (3\u20139.50) \u00b7 graded $24.99 (15\u201360)"
    )


def test_summary_without_both_bounds_has_no_range():
    assert price_summary([{"stats_type": "graded", "median": 24.99, "min": 15.0}]) == (
        "graded $24.99"
    )


def test_summary_overall_only_as_fallback_and_empty_when_nothing():
    assert price_summary(parse_price_stats([_entry("overall", median=10)])) == (
        "overall $10"
    )
    assert price_summary([]) == ""
