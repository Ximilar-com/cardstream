"""Shared Ximilar response parsing + tier mapping (no network).

One test module for the ONE copy of the parsing code in
:mod:`cardstream.core.ximilar`, which every identify path builds on — so these
behaviours are asserted exactly once, against recorded responses.
"""

from __future__ import annotations

import pytest

from cardstream.core.models import ConfidenceTier
from cardstream.core.ximilar import (
    TierThresholds,
    all_objects,
    best_card_object,
    distance_to_tier,
    parse_best_match,
)

# --- distance_to_tier -------------------------------------------------------


@pytest.mark.parametrize(
    "distance,tier",
    [
        (0.10, ConfidenceTier.HIGH),
        (0.18, ConfidenceTier.HIGH),  # inclusive upper bound
        (0.25, ConfidenceTier.MEDIUM),
        (0.30, ConfidenceTier.MEDIUM),  # inclusive upper bound
        (0.40, ConfidenceTier.LOW),
    ],
)
def test_distance_to_tier_boundaries(distance, tier):
    assert distance_to_tier(distance) is tier


def test_distance_to_tier_custom_thresholds():
    tight = TierThresholds(high_max_distance=0.05, medium_max_distance=0.10)
    assert distance_to_tier(0.06, tight) is ConfidenceTier.MEDIUM
    assert distance_to_tier(0.11, tight) is ConfidenceTier.LOW


# --- parse_best_match shapes ------------------------------------------------

_IDENT = {
    "best_match": {
        "name": "Charizard",
        "full_name": "Charizard - Base Set 4",
        "set": "Base",
        "set_code": "BS",
        "card_number": "4",
        "series": "Base",
        "year": 1999,
        "subcategory": "Pokemon",
        "links": {"ximilar": "https://example.com/4"},
        # Only with the top-level price_stats request flag; the trend block
        # and std/range are dropped on the way in.
        "price_stats": [
            {
                "stats_type": "ungraded",
                "interval": "overall",
                "start_date": None,
                "value": {
                    "min": 15.0,
                    "max": 60.0,
                    "median": 24.99,
                    "std": 14.52,
                    "range": 45.0,
                    "latest": 55.0,
                    "latest_date": "2026-02-22",
                    "trend": {"forecast_30d": 55.3},
                },
            }
        ],
    },
    "distances": [0.05],
    "alternatives": [
        {"full_name": "Charizard - other", "set": "Jungle", "links": {}},
    ],
}


def test_parse_identification_on_record():
    body = {"records": [{"_identification": _IDENT}]}
    ident = parse_best_match(body)
    assert ident is not None
    assert ident.full_name == "Charizard - Base Set 4"
    assert ident.year == "1999"  # coerced to str
    assert ident.distance == pytest.approx(0.05)
    assert ident.confidence_tier is ConfidenceTier.HIGH
    assert len(ident.alternatives) == 1
    (prices,) = ident.price_stats
    assert prices["stats_type"] == "ungraded" and prices["median"] == 24.99
    assert prices["latest_date"] == "2026-02-22"
    assert "trend" not in prices and "std" not in prices


def test_parse_identification_in_objects():
    body = {"records": [{"_objects": [{"_identification": _IDENT}]}]}
    ident = parse_best_match(body)
    assert ident is not None and ident.name == "Charizard"


def test_parse_response_wrapper():
    body = {"response": {"records": [{"_identification": _IDENT}]}}
    ident = parse_best_match(body)
    assert ident is not None and ident.set == "Base"


def test_parse_empty_returns_none():
    assert parse_best_match({"records": []}) is None
    assert parse_best_match({}) is None
    assert parse_best_match({"records": [{"_objects": [{"name": "Card"}]}]}) is None


def test_parse_missing_distances_defaults_to_low():
    body = {"records": [{"_identification": {"best_match": {"name": "X"}}}]}
    ident = parse_best_match(body)
    assert ident is not None
    assert ident.distance == 1.0
    assert ident.confidence_tier is ConfidenceTier.LOW
    assert ident.price_stats == []  # not asked for -> not there, not an error


# --- detection object helpers ----------------------------------------------

_DETECT_BODY = {
    "records": [
        {
            "_objects": [
                {"name": "Card", "prob": 0.6, "bound_box": [0, 0, 50, 70]},
                {"name": "Card", "prob": 0.9, "bound_box": [5, 5, 60, 80]},
                {"name": "Background", "prob": 0.99, "bound_box": [0, 0, 99, 99]},
                {
                    "name": "Card",
                    "prob": 0.2,
                    "bound_box": [1, 1, 2, 2],
                },  # below min_prob
            ]
        }
    ]
}


def test_best_card_object_picks_highest_prob_card():
    assert best_card_object(_DETECT_BODY, min_prob=0.5) == (5, 5, 60, 80, 0.9)


def test_best_card_object_respects_min_prob_and_name():
    assert best_card_object(_DETECT_BODY, min_prob=0.95) is None  # no Card >= 0.95
    assert best_card_object({"records": []}, min_prob=0.5) is None


def test_all_objects_lists_names_and_probs():
    objs = all_objects(_DETECT_BODY)
    assert ("Background", 0.99) in objs
    assert len(objs) == 4


def test_data_wrapper_is_unwrapped():
    wrapped = {"data": _DETECT_BODY}
    assert best_card_object(wrapped, min_prob=0.5) == (5, 5, 60, 80, 0.9)
