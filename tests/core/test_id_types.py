"""The id-type registry: games, alphabets and set codes.

What the registry FEEDS (the Ximilar record, the query params) is covered
in test_identify_options.py, alongside the object that builds both.
"""

from __future__ import annotations

import pytest

from cardstream.core.id_types import (
    ALPHABETS,
    ID_TYPES,
    NOT_SPECIFIED,
    normalize_alphabet,
    normalize_set_code,
    resolve_id_type,
)

TCG = ID_TYPES["tcg"]


def normalize_game(value, id_type="tcg"):
    return resolve_id_type(id_type).normalize_game(value)


def subcategory_for(value, id_type="tcg"):
    return resolve_id_type(id_type).subcategory_for(value)


def test_mapping_covers_the_offered_games():
    assert TCG.subcategories["Pokémon"] == "Pokemon"
    assert TCG.subcategories["Magic The Gathering"] == "Magic The Gathering"
    assert TCG.subcategories["One Piece"] == "One Piece"
    assert TCG.game_choices[0] == NOT_SPECIFIED


@pytest.mark.parametrize("value", [None, "", "  ", NOT_SPECIFIED, "not specified"])
def test_normalize_unset_values(value):
    assert normalize_game(value) is None
    assert subcategory_for(value) is None


@pytest.mark.parametrize("value", ["Pokémon", "pokémon", "Pokemon", "POKEMON"])
def test_normalize_accepts_display_and_api_spellings(value):
    assert normalize_game(value) == "Pokémon"
    assert subcategory_for(value) == "Pokemon"


def test_normalize_rejects_unknown_game():
    with pytest.raises(ValueError, match="unknown game"):
        normalize_game("Uno")


@pytest.mark.parametrize("value", [None, "", "   "])
def test_normalize_set_code_treats_blank_as_unset(value):
    assert normalize_set_code(value) is None


def test_normalize_set_code_strips_but_keeps_case():
    assert normalize_set_code("  PBL ") == "PBL"
    assert normalize_set_code("sv3pt5") == "sv3pt5"  # not all codes are upper-case


def test_subcategories_are_per_category():
    assert subcategory_for("Baseball", "sport") == "Baseball"
    assert subcategory_for("Pokémon", "tcg") == "Pokemon"
    with pytest.raises(ValueError):
        subcategory_for("Pokémon", "sport")  # tcg game, sport endpoint
    with pytest.raises(ValueError):
        subcategory_for("Baseball", "comics")  # comics_id takes none
    assert subcategory_for(None, "sport") is None
    assert ID_TYPES["comics"].subcategories == {}


def test_every_id_type_is_self_consistent():
    """The registry is the only place a category is declared — so a new entry
    that forgets a field fails here rather than at the endpoint."""
    for key, t in ID_TYPES.items():
        assert t.key == key
        assert t.label and t.url.endswith(f"/{key}_id")
        assert t.category_attrs.get("Top Category")
        assert t.game_choices[0] == NOT_SPECIFIED
        # Every offered game round-trips through its own normalizer.
        for display in t.subcategories:
            assert t.normalize_game(display) == display
            assert t.subcategory_for(display) == t.subcategories[display]


def test_registry_is_frozen():
    with pytest.raises(TypeError):
        ID_TYPES["tcg"].subcategories["Uno"] = "Uno"  # type: ignore[index]


def test_resolve_id_type_wording_follows_the_caller():
    with pytest.raises(ValueError, match="unknown id type"):
        resolve_id_type("nope")
    with pytest.raises(ValueError, match="unknown category"):
        resolve_id_type("nope", noun="category")
    assert resolve_id_type(" TCG ") is TCG  # tolerant, like the query param
    assert resolve_id_type(TCG) is TCG  # already-resolved passes through


@pytest.mark.parametrize("value", [None, "", "  ", NOT_SPECIFIED])
def test_unspecified_means_omit_for_every_optional_field(value):
    assert normalize_game(value) is None
    assert normalize_set_code(value) is None
    assert normalize_alphabet(value) is None


def test_normalize_alphabet_rejects_unknown_values():
    # The API echoes an unknown alphabet back and quietly mismatches, so the
    # whitelist here is the only thing that can catch a typo.
    assert normalize_alphabet("Japanese") == "japanese"
    assert set(ALPHABETS) >= {"latin", "japanese"}
    with pytest.raises(ValueError, match="unknown alphabet"):
        normalize_alphabet("cyrillic")
