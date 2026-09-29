"""--ximilar-stream: the upload item, the session API calls and the recorder."""

from __future__ import annotations

import threading
from datetime import datetime

import pytest
import requests

from _helpers import SESSION_ID, FakeSessionApi, make_identification, wait_until
from cardstream.core import ximilar_session
from cardstream.core.ximilar_session import (
    MAX_BATCH,
    NEW_SESSION,
    Outcome,
    SessionApi,
    SessionError,
    SessionRecorder,
    UploadReply,
    open_session,
    parse_session_spec,
    session_item,
)

SEEN = 1_790_000_000.0  # a fixed wall-clock moment, epoch seconds


def _ident(**overrides) -> dict:
    """What the analyzer hands on: the flattened identification + elapsed_ms."""
    ident = make_identification().to_dict()
    ident["elapsed_ms"] = 420
    ident.update(overrides)
    return ident


# --- the upload item -----------------------------------------------------------


def test_item_uses_the_api_field_names():
    item = session_item(_ident(), "tcg", SEEN, event_id="evt-1")
    assert item["event_id"] == "evt-1"
    assert item["id_type"] == "tcg"
    assert datetime.fromisoformat(item["seen"]).timestamp() == SEEN
    assert item["set_name"] == "Base"  # the client calls it `set`
    assert item["confidence"] == "high"  # ... and this `confidence_tier`
    assert item["full_name"] == "Charizard"
    assert item["card_number"] == "4"
    assert item["distance"] == 0.1
    assert item["links"] == {"ximilar": "https://example.com/card"}
    # Nothing the API does not declare rides along.
    assert "set" not in item and "confidence_tier" not in item
    assert "elapsed_ms" not in item


def test_items_get_distinct_random_event_ids():
    ids = {session_item(_ident(), "tcg", SEEN)["event_id"] for _ in range(50)}
    assert len(ids) == 50
    assert all(len(event_id) <= 64 for event_id in ids)


def test_item_clips_text_to_the_api_limits():
    item = session_item(_ident(full_name="x" * 900, set_code="y" * 80), "tcg", SEEN)
    assert len(item["full_name"]) == 500
    assert len(item["set_code"]) == 50


def test_item_drops_what_the_api_would_reject():
    item = session_item(
        _ident(
            distance=float("nan"),
            confidence_tier="certain",
            links={"tcgplayer": "https://t.example/1", "broken": None, "n": 3},
            alternatives=[{"full_name": "a"}, "not a dict"] + [{}] * 20,
            price_stats=None,
            year=None,
        ),
        "tcg",
        SEEN,
    )
    assert "distance" not in item
    assert "confidence" not in item
    assert item["links"] == {"tcgplayer": "https://t.example/1"}
    assert item["alternatives"][0] == {"full_name": "a"}
    assert len(item["alternatives"]) == 10
    assert "price_stats" not in item
    assert item["year"] == ""


def test_item_keeps_price_statistics():
    stats = [{"stats_type": "ungraded", "median": 12.5, "latest_date": "2026-09-01"}]
    assert session_item(_ident(price_stats=stats), "tcg", SEEN)["price_stats"] == stats


# --- the --ximilar-stream value -----------------------------------------------


@pytest.mark.parametrize("value", ["NEW", "new", " New "])
def test_new_in_any_case_starts_a_session(value):
    assert parse_session_spec(value) == NEW_SESSION


def test_a_session_id_is_canonicalised():
    assert parse_session_spec(SESSION_ID.upper()) == SESSION_ID


@pytest.mark.parametrize("value", ["", "latest", "1234", SESSION_ID + "0"])
def test_anything_else_is_refused(value):
    with pytest.raises(ValueError, match="neither NEW nor a session id"):
        parse_session_spec(value)


# --- starting or resuming ---------------------------------------------------------


def test_new_creates_a_session():
    api = FakeSessionApi()
    session, created = open_session(
        api,
        "NEW",
        name="Friday show",
        game="Pokémon",
        platform="whatnot",
        client={"version": "0.3.0"},
    )
    assert created and session["id"] == SESSION_ID
    assert api.created == [
        {
            "name": "Friday show",
            "game": "Pokémon",
            "platform": "whatnot",
            "client": {"version": "0.3.0"},
        }
    ]


def test_an_id_resumes_a_live_session():
    api = FakeSessionApi()
    session, created = open_session(api, SESSION_ID)
    assert not created and session["id"] == SESSION_ID
    assert api.fetched == [SESSION_ID] and api.created == []


def test_a_closed_session_cannot_be_resumed():
    with pytest.raises(SessionError, match=r"is closed .* --ximilar-stream NEW"):
        open_session(FakeSessionApi(status="closed"), SESSION_ID)


# --- the HTTP calls ---------------------------------------------------------------


class _Response:
    def __init__(self, status: int, payload: object = None) -> None:
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload
        self.text = str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture
def http(monkeypatch):
    """Replies to requests.request in order; records what was sent."""
    calls: list[dict] = []
    replies: list[object] = []

    def fake_request(method, url, json=None, headers=None, timeout=None):
        calls.append({"method": method, "url": url, "json": json, "headers": headers})
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(ximilar_session.requests, "request", fake_request)
    return calls, replies


def test_the_api_needs_a_key():
    with pytest.raises(SessionError, match="API key"):
        SessionApi("")


def test_create_posts_to_the_session_endpoint_with_the_identify_headers(http):
    calls, replies = http
    replies.append(_Response(201, {"id": SESSION_ID, "status": "live"}))
    api = SessionApi("key", "http://localhost:8000/api/cardstream/v2/")
    assert api.create({"name": "Show"})["id"] == SESSION_ID
    call = calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "http://localhost:8000/api/cardstream/v2/session/"
    assert call["headers"]["Authorization"] == "Token key"
    assert call["headers"]["User-Agent"] == "CardStream"


def test_a_refused_start_says_why(http):
    _, replies = http
    replies.append(
        _Response(403, {"detail": "not allowed to access service Cardstream"})
    )
    with pytest.raises(SessionError, match="HTTP 403: not allowed to access service"):
        SessionApi("key").get(SESSION_ID)


def test_an_unreachable_api_fails_the_start(http):
    _, replies = http
    replies.append(requests.ConnectionError("no route"))
    with pytest.raises(SessionError, match="unreachable"):
        SessionApi("key").create({})


@pytest.mark.parametrize(
    ("reply", "outcome"),
    [
        (_Response(201, {"created": 2, "duplicates": 0}), Outcome.STORED),
        (_Response(200, {"created": 0, "duplicates": 2}), Outcome.STORED),
        (_Response(500, "boom"), Outcome.RETRY),
        (_Response(429, {"detail": "slow down"}), Outcome.RETRY),
        (requests.Timeout("timed out"), Outcome.RETRY),
        (_Response(400, {"identifications": ["bad"]}), Outcome.REJECTED),
        (_Response(409, {"detail": "The session is closed"}), Outcome.STOPPED),
        (_Response(403, {"detail": "forbidden"}), Outcome.STOPPED),
        (_Response(404, {"detail": "Not found."}), Outcome.STOPPED),
    ],
)
def test_upload_replies_are_sorted_by_what_happens_next(http, reply, outcome):
    calls, replies = http
    replies.append(reply)
    items = [session_item(_ident(), "tcg", SEEN) for _ in range(2)]
    result = SessionApi("key").upload(SESSION_ID, items)
    assert result.outcome is outcome
    assert calls[0]["url"].endswith(f"/session/{SESSION_ID}/identifications/")
    assert calls[0]["json"] == {"identifications": items}


def test_close_reports_a_failure_instead_of_raising(http):
    calls, replies = http
    replies.extend([_Response(200, {}), _Response(404, {"detail": "Not found."})])
    api = SessionApi("key")
    assert api.close(SESSION_ID) is None
    assert api.close(SESSION_ID) == "HTTP 404: Not found."
    assert calls[0]["url"].endswith(f"/session/{SESSION_ID}/close/")


# --- the recorder -----------------------------------------------------------------


def _recorder(api, logs=None, **kwargs) -> SessionRecorder:
    return SessionRecorder(
        api,
        SESSION_ID,
        log=(logs.append if logs is not None else lambda _: None),
        start=False,
        **kwargs,
    )


def test_recorded_identifications_are_uploaded_in_order():
    api = FakeSessionApi()
    recorder = _recorder(api)
    recorder.record(_ident(full_name="A"), "tcg", SEEN)
    recorder.record(_ident(full_name="B"), "sport", SEEN + 5)
    assert recorder.flush()
    assert [[i["full_name"] for i in batch] for batch in api.uploads] == [["A", "B"]]
    assert api.uploads[0][1]["id_type"] == "sport"
    assert recorder.pending == 0 and recorder.stored == 2


def test_large_queues_go_up_in_api_sized_batches():
    api = FakeSessionApi()
    recorder = _recorder(api)
    for _ in range(MAX_BATCH + 3):
        recorder.record(_ident(), "tcg", SEEN)
    assert recorder.flush()
    assert [len(batch) for batch in api.uploads] == [MAX_BATCH, 3]


def test_a_transient_failure_keeps_the_items_and_resends_the_same_ids():
    api = FakeSessionApi(replies=[UploadReply(Outcome.RETRY, "HTTP 502")])
    logs: list[str] = []
    recorder = _recorder(api, logs)
    recorder.record(_ident(), "tcg", SEEN)
    assert not recorder.flush()
    assert recorder.pending == 1
    assert "retrying" in logs[-1]
    assert recorder.flush()
    first, second = api.uploads
    # Same event ids: the API skips the ones a lost reply already stored.
    assert [i["event_id"] for i in first] == [i["event_id"] for i in second]
    assert recorder.pending == 0 and recorder.stored == 1


def test_a_rejected_batch_is_dropped_and_the_rest_continue():
    api = FakeSessionApi(replies=[UploadReply(Outcome.REJECTED, "HTTP 400")])
    logs: list[str] = []
    recorder = _recorder(api, logs)
    recorder.record(_ident(), "tcg", SEEN)
    assert recorder.flush()
    assert recorder.dropped == 1 and recorder.pending == 0
    assert "rejected" in logs[-1]
    recorder.record(_ident(), "tcg", SEEN)
    assert recorder.flush() and recorder.stored == 1


def test_a_closed_session_stops_uploads_for_good():
    api = FakeSessionApi(replies=[UploadReply(Outcome.STOPPED, "HTTP 409: closed")])
    logs: list[str] = []
    recorder = _recorder(api, logs)
    recorder.record(_ident(), "tcg", SEEN)
    assert recorder.flush()
    assert recorder.stopped == "HTTP 409: closed"
    assert "uploads stopped" in logs[-1]
    recorder.record(_ident(), "tcg", SEEN)  # ignored from now on
    assert recorder.pending == 0 and len(api.uploads) == 1
    recorder.close()
    assert api.closed == []  # nothing left to close


def test_a_full_queue_drops_the_oldest_and_says_so_once():
    logs: list[str] = []
    recorder = _recorder(FakeSessionApi(), logs, max_pending=3)
    for name in "ABCDE":
        recorder.record(_ident(full_name=name), "tcg", SEEN)
    assert recorder.pending == 3 and recorder.dropped == 2
    assert sum("queue full" in line for line in logs) == 1


def test_a_batch_in_flight_survives_the_queue_overflowing():
    """record() drops from the head while a batch is on the wire; settling it
    must remove exactly the items that were sent, not the first N."""
    recorder = _recorder(FakeSessionApi(), max_pending=2)
    recorder.record(_ident(full_name="A"), "tcg", SEEN)
    recorder.record(_ident(full_name="B"), "tcg", SEEN)
    batch = [recorder._pending[0], recorder._pending[1]]
    recorder.record(_ident(full_name="C"), "tcg", SEEN)  # drops A
    recorder._settle(batch, UploadReply(Outcome.STORED))
    assert [i["full_name"] for i in recorder._pending] == ["C"]


def test_close_uploads_what_is_left_and_closes_the_session():
    api = FakeSessionApi()
    logs: list[str] = []
    recorder = _recorder(api, logs)
    recorder.record(_ident(), "tcg", SEEN)
    recorder.close()
    assert len(api.uploads) == 1 and api.closed == [SESSION_ID]
    assert "session closed" in logs[-1]
    recorder.record(_ident(), "tcg", SEEN)  # after close: ignored
    recorder.close()  # idempotent
    assert len(api.uploads) == 1 and api.closed == [SESSION_ID]


def test_keep_open_leaves_the_session_resumable():
    api = FakeSessionApi()
    logs: list[str] = []
    recorder = _recorder(api, logs, close_session=False)
    recorder.close()
    assert api.closed == []
    assert f"--ximilar-stream {SESSION_ID}" in logs[-1]


def test_close_gives_up_on_an_unreachable_api_after_the_timeout():
    api = FakeSessionApi(replies=[UploadReply(Outcome.RETRY, "down")] * 100)
    logs: list[str] = []
    recorder = _recorder(api, logs)
    recorder.record(_ident(), "tcg", SEEN)
    recorder.close(timeout=0)
    assert any("could not be uploaded" in line for line in logs)


def test_a_failed_close_is_reported():
    api = FakeSessionApi()
    api.close_failure = "HTTP 500: boom"
    logs: list[str] = []
    _recorder(api, logs).close()
    assert "could not close it (HTTP 500: boom)" in logs[-1]


async def test_the_background_thread_uploads_without_being_asked():
    api = FakeSessionApi()
    recorder = SessionRecorder(api, SESSION_ID, flush_seconds=0.01, log=lambda _: None)
    try:
        recorder.record(_ident(), "tcg", SEEN)
        assert await wait_until(lambda: recorder.stored == 1)
    finally:
        recorder.close()
    assert api.closed == [SESSION_ID]


def test_record_is_safe_from_many_identify_threads():
    api = FakeSessionApi()
    recorder = _recorder(api)
    threads = [
        threading.Thread(
            target=lambda: [recorder.record(_ident(), "tcg", SEEN) for _ in range(50)]
        )
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert recorder.recorded == 400
    assert recorder.flush()
    assert sum(len(batch) for batch in api.uploads) == 400
