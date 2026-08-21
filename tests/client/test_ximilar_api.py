"""Direct Ximilar client: response parsing, tier mapping, constructor guards."""

from __future__ import annotations

import pytest

from cardstream.client.identify_target import IdentifyTarget
from cardstream.client.ximilar_api import DirectXimilarClient
from cardstream.core.identify_options import IdentifyOptions
from cardstream.core.ximilar import distance_to_tier, parse_best_match


def _response(distance=0.1, wrap_in_response=False, in_objects=True):
    best = {
        "name": "Charizard",
        "full_name": "Charizard (Base Set 4/102)",
        "set": "Base Set",
        "set_code": "BS",
        "card_number": "4",
        "series": "Base",
        "year": 1999,
        "subcategory": "Pokemon",
        "links": {"tcgplayer": "https://example.com"},
    }
    ident = {
        "best_match": best,
        "distances": [distance],
        "alternatives": [
            {"full_name": "Charizard (alt)", "set": "Base 2", "links": {}}
        ],
    }
    rec = (
        {"_objects": [{"name": "Card", "_identification": ident}]}
        if in_objects
        else {"_identification": ident}
    )
    body = {"records": [rec]}
    return {"response": body} if wrap_in_response else body


def test_parse_flattens_best_match():
    ident = parse_best_match(_response(distance=0.12)).to_dict()
    assert ident["full_name"] == "Charizard (Base Set 4/102)"
    assert ident["set"] == "Base Set"
    assert ident["year"] == "1999"  # stringified like the server
    assert ident["distance"] == 0.12
    assert ident["confidence_tier"] == "high"
    assert ident["links"] == {"tcgplayer": "https://example.com"}
    assert ident["alternatives"][0]["full_name"] == "Charizard (alt)"


def test_parse_handles_all_three_shapes():
    for kwargs in (
        {"in_objects": True},
        {"in_objects": False},
        {"in_objects": True, "wrap_in_response": True},
    ):
        assert parse_best_match(_response(**kwargs)) is not None


def test_parse_returns_none_without_identification():
    assert parse_best_match({}) is None
    assert parse_best_match({"records": []}) is None
    assert parse_best_match({"records": [{"_objects": [{"name": "Card"}]}]}) is None


def test_distance_to_tier_cutoffs():
    assert distance_to_tier(0.10) == "high"
    assert distance_to_tier(0.18) == "high"
    assert distance_to_tier(0.25) == "medium"
    assert distance_to_tier(0.30) == "medium"
    assert distance_to_tier(0.45) == "low"


def test_constructor_requires_key_and_valid_type():
    with pytest.raises(ValueError, match="API key"):
        DirectXimilarClient("", IdentifyOptions("tcg"))
    with pytest.raises(ValueError, match="unknown id type"):
        DirectXimilarClient("key", IdentifyOptions("magic"))
    with pytest.raises(ValueError, match="unknown game"):
        DirectXimilarClient("key", IdentifyOptions(game="Uno"))
    # valid combination constructs fine (no network involved)
    DirectXimilarClient("key", IdentifyOptions("sport"))


# --- identify() HTTP paths (monkeypatched requests) --------------------------


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.ok = 200 <= status < 300
        self.text = str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _crop():
    import numpy as np

    return np.random.RandomState(0).randint(0, 256, size=(64, 64, 3), dtype=np.uint8)


def test_identify_posts_b64_and_flattens(monkeypatch):
    from cardstream.core import ximilar as core_ximilar

    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(url=url, json=json, headers=headers)
        return _FakeResponse(_response(distance=0.1))

    monkeypatch.setattr(core_ximilar.requests, "post", fake_post)
    ident = DirectXimilarClient("key", IdentifyOptions("tcg")).identify(_crop())
    assert ident is not None and ident["full_name"] == "Charizard (Base Set 4/102)"
    assert captured["url"].endswith("/tcg_id")
    assert captured["headers"]["Authorization"] == "Token key"
    record = captured["json"]["records"][0]
    assert "_base64" in record
    # A full-image Card object always rides along so the id endpoint skips
    # its own detection (the 64x64 crop is upscaled to a 500px long edge);
    # the known attributes ride on it — identification runs per object.
    (obj,) = record["_objects"]
    assert obj == {
        "prob": 1.0,
        "name": "Card",
        "bound_box": [0, 0, 500, 500],
        "Side": "front",
        "Rotation": "rotation_ok",
        "Top Category": "Card",
        "Category": "Card/Trading Card Game",
    }
    # No --alphabet -> the field is absent and the endpoint classifies it.
    assert "Alphabet" not in obj
    assert "Subcategory" not in obj  # no --game -> no prefill
    assert "set_code" not in record  # no --set-code -> no prefill


def test_identify_sends_game_as_subcategory(monkeypatch):
    from cardstream.core import ximilar as core_ximilar

    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["json"] = json
        return _FakeResponse(_response())

    monkeypatch.setattr(core_ximilar.requests, "post", fake_post)

    def sent_object():
        return captured["json"]["records"][0]["_objects"][0]

    client = DirectXimilarClient("key", IdentifyOptions(game="Pokémon"))
    client.identify(_crop())
    assert sent_object()["Subcategory"] == "Pokemon"  # mapped display -> Ximilar value
    assert sent_object()["Side"] == "front"

    client.options = client.options.with_(game="Magic The Gathering")  # live edit
    client.identify(_crop())
    assert sent_object()["Subcategory"] == "Magic The Gathering"

    client.options = client.options.with_(game=None)
    client.identify(_crop())
    assert "Subcategory" not in sent_object()


def test_identify_crop_objects_box_matches_crop_dims(monkeypatch):
    """A crop already >= 500 px on the long edge is sent unscaled, and the
    synthetic ``_objects`` box is exactly [0, 0, crop_width, crop_height]."""
    import numpy as np

    from cardstream.core import ximilar as core_ximilar

    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["json"] = json
        return _FakeResponse(_response())

    monkeypatch.setattr(core_ximilar.requests, "post", fake_post)
    crop = np.random.RandomState(1).randint(0, 256, size=(600, 420, 3), dtype=np.uint8)
    DirectXimilarClient("key").identify(crop)
    (obj,) = captured["json"]["records"][0]["_objects"]
    assert obj["bound_box"] == [0, 0, 420, 600]
    assert obj["name"] == "Card" and obj["prob"] == 1.0


@pytest.mark.parametrize(
    "response",
    [
        _FakeResponse({"detail": "boom"}, status=500),  # HTTP error
        _FakeResponse(ValueError("not json")),  # malformed 200
        _FakeResponse({}),  # no identification
    ],
)
def test_identify_returns_none_on_failure(monkeypatch, response):
    from cardstream.core import ximilar as core_ximilar

    monkeypatch.setattr(core_ximilar.requests, "post", lambda *a, **k: response)
    assert DirectXimilarClient("key").identify(_crop()) is None


def test_identify_none_on_connection_error(monkeypatch):
    import requests as requests_mod

    from cardstream.core import ximilar as core_ximilar

    def fake_post(*a, **k):
        raise requests_mod.ConnectionError("refused")

    monkeypatch.setattr(core_ximilar.requests, "post", fake_post)
    assert DirectXimilarClient("key").identify(_crop()) is None


def test_identify_sends_set_code_on_the_record(monkeypatch):
    from cardstream.core import ximilar as core_ximilar

    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["json"] = json
        return _FakeResponse(_response())

    monkeypatch.setattr(core_ximilar.requests, "post", fake_post)

    def sent_record():
        return captured["json"]["records"][0]

    client = DirectXimilarClient("key", IdentifyOptions(set_code=" PBL "))
    client.identify(_crop())
    assert sent_record()["set_code"] == "PBL"
    assert "set_code" not in sent_record()["_objects"][0]

    client.options = client.options.with_(set_code=None)  # same swap as game
    client.identify(_crop())
    assert "set_code" not in sent_record()


def test_identify_known_attrs_false_drops_side_and_rotation(monkeypatch):
    from cardstream.core import ximilar as core_ximilar

    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["json"] = json
        return _FakeResponse(_response())

    monkeypatch.setattr(core_ximilar.requests, "post", fake_post)
    client = DirectXimilarClient("key", IdentifyOptions(known_attrs=False))
    client.identify(_crop())
    (obj,) = captured["json"]["records"][0]["_objects"]
    assert "Side" not in obj and "Rotation" not in obj
    assert obj["Category"] == "Card/Trading Card Game"  # category still asserted


def test_category_switch_changes_endpoint_and_clears_a_foreign_game(monkeypatch):
    from cardstream.core import ximilar as core_ximilar

    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"], captured["json"] = url, json
        return _FakeResponse(_response())

    monkeypatch.setattr(core_ximilar.requests, "post", fake_post)
    client = DirectXimilarClient("key", IdentifyOptions(game="Pokémon"))

    client.options = client.options.with_(id_type="sport")  # the settings dialog
    assert client.options.game is None  # Pokémon means nothing to sport_id
    client.options = client.options.with_(game="Baseball")
    client.identify(_crop())
    assert captured["url"].endswith("/sport_id")
    (obj,) = captured["json"]["records"][0]["_objects"]
    assert obj["Category"] == "Card/Sport Card" and obj["Subcategory"] == "Baseball"

    with pytest.raises(ValueError):
        client.options = client.options.with_(id_type="nope")
    assert client.options.id_type.key == "sport"  # unchanged on bad input


def test_alphabet_rides_with_the_game_prefill(monkeypatch):
    """A game prefill switches the endpoint's alphabet classifier off, so the
    two must travel together — sending only the game is what matched Japanese
    cards to their English print."""
    from cardstream.core import ximilar as core_ximilar

    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["json"] = json
        return _FakeResponse(_response())

    monkeypatch.setattr(core_ximilar.requests, "post", fake_post)
    client = DirectXimilarClient(
        "key", IdentifyOptions(game="Pokémon", alphabet="japanese")
    )
    client.identify(_crop())
    (obj,) = captured["json"]["records"][0]["_objects"]
    assert obj["Subcategory"] == "Pokemon" and obj["Alphabet"] == "japanese"

    with pytest.raises(ValueError):  # silently wrong upstream
        DirectXimilarClient("key", IdentifyOptions(alphabet="klingon"))


def test_alphabet_is_omitted_unless_asked_for(monkeypatch):
    """Default: no Alphabet in the record, so the endpoint classifies it."""
    from cardstream.core import ximilar as core_ximilar

    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["json"] = json
        return _FakeResponse(_response())

    monkeypatch.setattr(core_ximilar.requests, "post", fake_post)
    client = DirectXimilarClient(
        "key", IdentifyOptions(game="Pokémon")
    )  # game, no alphabet
    client.identify(_crop())
    (obj,) = captured["json"]["records"][0]["_objects"]
    assert obj["Subcategory"] == "Pokemon" and "Alphabet" not in obj

    client.options = client.options.with_(alphabet="latin")  # only when asked
    client.identify(_crop())
    assert captured["json"]["records"][0]["_objects"][0]["Alphabet"] == "latin"


def test_the_identify_target_contract_holds():
    """SmartAnalyzer is annotated with IdentifyTarget, not the concrete class:
    the options bundle and its normalization live on the ABC, so a second
    implementation cannot drift from this one."""
    target = DirectXimilarClient("key", IdentifyOptions("sport", game="Hockey"))
    assert isinstance(target, IdentifyTarget)
    assert target.options.subcategory == "Hockey"
    assert target.options.id_type.key == "sport"


# --- --store-images ----------------------------------------------------------


def test_store_images_writes_exactly_the_bytes_that_were_posted(monkeypatch, tmp_path):
    """The file on disk must decode to the record's own _base64 — not a
    re-encode of the crop, which would answer a different question."""
    import base64

    from cardstream.core import ximilar as core_ximilar
    from cardstream.core.image_store import ImageStore

    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(json=json)
        return _FakeResponse(_response())

    monkeypatch.setattr(core_ximilar.requests, "post", fake_post)
    client = DirectXimilarClient(
        "key", IdentifyOptions("tcg"), store=ImageStore(tmp_path)
    )
    client.identify(_crop())

    (written,) = list(tmp_path.iterdir())
    sent = captured["json"]["records"][0]["_base64"]
    assert written.read_bytes() == base64.b64decode(sent)


def test_store_images_keeps_the_crop_when_the_call_fails(monkeypatch, tmp_path):
    """An unmatched card is the one you most want to look at afterwards, so the
    image is written before the POST, not after a successful parse."""
    from cardstream.core import ximilar as core_ximilar
    from cardstream.core.image_store import ImageStore

    monkeypatch.setattr(
        core_ximilar.requests, "post", lambda *a, **k: _FakeResponse("nope", status=500)
    )
    client = DirectXimilarClient(
        "key", IdentifyOptions("tcg"), store=ImageStore(tmp_path)
    )
    assert client.identify(_crop()) is None
    assert len(list(tmp_path.iterdir())) == 1


def test_no_store_no_files(monkeypatch, tmp_path):
    from cardstream.core import ximilar as core_ximilar

    monkeypatch.setattr(
        core_ximilar.requests, "post", lambda *a, **k: _FakeResponse(_response())
    )
    DirectXimilarClient("key", IdentifyOptions("tcg")).identify(_crop())
    assert not list(tmp_path.iterdir())


# --- request headers ---------------------------------------------------------


def test_every_call_identifies_itself_as_cardstream(monkeypatch):
    """The User-Agent rides on the shared header dict, so it reaches whichever
    endpoint the id type points at — not just tcg_id."""
    from cardstream.core import ximilar as core_ximilar

    seen = []

    def fake_post(url, json, headers, timeout):
        seen.append(headers)
        return _FakeResponse(_response())

    monkeypatch.setattr(core_ximilar.requests, "post", fake_post)
    for id_type in ("tcg", "sport", "slab", "comics"):
        DirectXimilarClient("key", IdentifyOptions(id_type)).identify(_crop())

    assert len(seen) == 4
    for headers in seen:
        assert headers["User-Agent"] == "CardStream"
        assert headers["Authorization"] == "Token key"
        assert headers["Content-Type"] == "application/json"
