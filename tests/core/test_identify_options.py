"""IdentifyOptions — the one place the identify prefills are normalized.

The record tests are the point of the file: every prefill the CLI, the web
settings dialog and the analyzer can set has to land in the right slot of the
Ximilar record, and switching category has to drop a game the new one cannot
know.
"""

from __future__ import annotations

import pytest

from cardstream.core.id_types import ID_TYPES, NOT_SPECIFIED
from cardstream.core.identify_options import IdentifyOptions


def _card_object():
    return {"name": "Card", "bound_box": [0, 0, 10, 10], "prob": 1.0}


# --------------------------------------------------------------- construction


def test_normalizes_every_field_once():
    opts = IdentifyOptions(
        "tcg", game="POKEMON", set_code="  PBL ", alphabet="Japanese"
    )
    assert opts.id_type is ID_TYPES["tcg"]
    assert opts.game == "Pokémon"  # canonical display name
    assert opts.subcategory == "Pokemon"  # the Ximilar value
    assert opts.set_code == "PBL"
    assert opts.alphabet == "japanese"


def test_defaults_send_nothing_but_the_type():
    opts = IdentifyOptions()
    assert opts.id_type.key == "tcg" and opts.known_attrs is True
    assert opts.game is None and opts.set_code is None and opts.alphabet is None
    assert opts.price_stats is False  # the extra data is not documented as free


@pytest.mark.parametrize("value", [None, "", "   ", NOT_SPECIFIED])
def test_unspecified_is_the_same_word_for_every_optional_field(value):
    opts = IdentifyOptions("tcg", game=value, set_code=value, alphabet=value)
    assert (opts.game, opts.set_code, opts.alphabet) == (None, None, None)


def test_rejects_a_game_the_category_does_not_know():
    with pytest.raises(ValueError, match="unknown game"):
        IdentifyOptions("sport", game="Pokémon")


# ---------------------------------------------------------------------- with_


def test_switching_category_drops_a_game_the_new_one_does_not_know():
    opts = IdentifyOptions("tcg", game="Pokémon", set_code="PBL", alphabet="japanese")
    moved = opts.with_(id_type="sport")
    assert moved.game is None  # dropped, not sent as a bad hint
    assert moved.set_code == "PBL"  # unrelated fields survive
    assert moved.alphabet == "japanese"


def test_switching_category_keeps_a_game_the_new_one_knows():
    # Nothing shares a game across categories today; the rule is still that a
    # known game survives, so assert it where it does hold: same category.
    opts = IdentifyOptions("tcg", game="One Piece")
    assert opts.with_(set_code="OP01").game == "One Piece"


def test_an_explicitly_patched_bad_game_raises_instead_of_being_dropped():
    # A leftover is the machine's problem; a typed-in game is the user's.
    opts = IdentifyOptions("tcg", game="Pokémon")
    with pytest.raises(ValueError, match="unknown game"):
        opts.with_(id_type="sport", game="Pokémon")


def test_patch_distinguishes_absent_from_cleared():
    opts = IdentifyOptions("tcg", game="Pokémon", set_code="PBL")
    assert opts.with_(known_attrs=False).set_code == "PBL"  # absent = unchanged
    assert opts.with_(set_code=None).set_code is None  # present = cleared
    assert opts.with_(game=NOT_SPECIFIED).game is None


def test_patch_rejects_unknown_options():
    with pytest.raises(ValueError, match="unknown option"):
        IdentifyOptions().with_(colour="red")


def test_price_stats_round_trips_through_with_and_coerces():
    assert IdentifyOptions().with_(price_stats=True).price_stats is True
    assert IdentifyOptions(price_stats=1).price_stats is True
    assert (
        IdentifyOptions(price_stats=True).with_(price_stats=False).price_stats is False
    )


# ------------------------------------------------------------------ wire form


def test_record_puts_attributes_on_the_object_and_set_code_on_the_record():
    objects = [_card_object()]
    record = IdentifyOptions("tcg", game="Pokémon", set_code="PBL").record(
        "b64", objects
    )
    assert record["_base64"] == "b64"
    assert record["set_code"] == "PBL"  # record level
    (obj,) = record["_objects"]
    assert "set_code" not in obj
    assert obj == {
        **_card_object(),
        "Top Category": "Card",
        "Category": "Card/Trading Card Game",
        "Side": "front",
        "Rotation": "rotation_ok",
        "Subcategory": "Pokemon",
    }
    assert objects[0] == _card_object()  # caller's object not mutated


def test_record_category_attrs_follow_the_id_type():
    def obj(id_type, **kw):
        (o,) = IdentifyOptions(id_type, **kw).record("b64", [_card_object()])[
            "_objects"
        ]
        return o

    assert obj("sport")["Category"] == "Card/Sport Card"
    # slab/comics: only the top category is unambiguous — no Category guessed.
    assert obj("slab")["Top Category"] == "Slab Label" and "Category" not in obj("slab")
    assert obj("comics")["Top Category"] == "Comics" and "Category" not in obj("comics")


def test_known_attrs_false_lets_the_endpoint_decide_side_and_rotation():
    (obj,) = IdentifyOptions("tcg", known_attrs=False).record("b64", [_card_object()])[
        "_objects"
    ]
    assert "Side" not in obj and "Rotation" not in obj
    assert obj["Top Category"] == "Card"  # the category pair still rides along


def test_record_carries_the_alphabet_when_set():
    (obj,) = IdentifyOptions("tcg", alphabet="japanese").record(
        "b64", [_card_object()]
    )["_objects"]
    assert obj["Alphabet"] == "japanese"


def test_price_stats_is_a_preference_the_id_type_gates():
    """The flag rides on the POST body, not the record — and only where the
    endpoint documents it. slab_id does not, so the preference survives a
    category switch while the wire stays clean."""
    record = {"_base64": "b64", "_objects": []}
    assert IdentifyOptions().payload(record) == {"records": [record]}
    on = IdentifyOptions("tcg", price_stats=True)
    assert on.payload(record) == {"records": [record], "price_stats": True}
    for key in ("sport", "comics"):
        assert IdentifyOptions(key, price_stats=True).payload(record)["price_stats"]
    slab = on.with_(id_type="slab")
    assert slab.price_stats is True and "price_stats" not in slab.payload(record)
    assert slab.with_(id_type="tcg").payload(record)["price_stats"] is True


def test_record_tolerates_no_objects():
    assert IdentifyOptions().record("b64", [])["_objects"] == []


# -------------------------------------------------------------------- boolean
