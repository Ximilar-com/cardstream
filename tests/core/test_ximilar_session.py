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


def test_item_carries_the_rows_duration_and_calls():
    item = session_item(_ident(), "tcg", SEEN, duration=8.1234, calls=3)
    assert (item["duration"], item["calls"]) == (8.123, 3)
    bare = session_item(_ident(), "tcg", SEEN)
    assert "duration" not in bare and "calls" not in bare
    assert "duration" not in session_item(_ident(), "tcg", SEEN, duration=float("nan"))


def test_the_thumbnail_never_leaves_the_machine():
    item = session_item(_ident(thumbnail="data:image/jpeg;base64,AAAA"), "tcg", SEEN)
    assert "thumbnail" not in item


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


def test_paid_calls_are_reported_per_run(http):
    calls, replies = http
    replies.extend(
        [_Response(200, {"paid_calls": 7}), _Response(409, {"detail": "closed"})]
    )
    api = SessionApi("key")
    assert api.report_calls(SESSION_ID, "run-1", 7).outcome is Outcome.STORED
    assert calls[0]["url"].endswith(f"/session/{SESSION_ID}/calls/")
    assert calls[0]["json"] == {"run": "run-1", "calls": 7}
    assert api.report_calls(SESSION_ID, "run-1", 8).outcome is Outcome.STOPPED


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


def _row(event_id: str, name: str = "Charizard", duration: float = 1.0, calls: int = 1):
    return session_item(
        _ident(full_name=name),
        "tcg",
        SEEN,
        event_id=event_id,
        duration=duration,
        calls=calls,
    )


def test_queued_rows_are_uploaded_in_order():
    api = FakeSessionApi()
    recorder = _recorder(api)
    recorder.queue(_row("a", "A"))
    recorder.queue(_row("b", "B"))
    assert recorder.flush()
    assert [[i["full_name"] for i in batch] for batch in api.uploads] == [["A", "B"]]
    assert recorder.pending == 0 and recorder.rows_saved == 2


def test_a_row_changed_before_it_went_out_is_sent_once_in_its_latest_state():
    api = FakeSessionApi()
    recorder = _recorder(api)
    recorder.queue(_row("a", duration=1.0, calls=1))
    recorder.queue(_row("b"))
    recorder.queue(_row("a", duration=8.1, calls=3))
    assert recorder.flush()
    (batch,) = api.uploads
    assert [(i["event_id"], i["duration"], i["calls"]) for i in batch] == [
        ("a", 8.1, 3),  # keeps its place in the queue
        ("b", 1.0, 1),
    ]


def test_a_row_changed_while_its_batch_was_on_the_wire_goes_out_again():
    api = FakeSessionApi()
    recorder = _recorder(api)
    recorder.queue(_row("a", duration=1.0))

    def upload_then_change(session_id, items):
        if not api.uploads:
            recorder.queue(_row("a", duration=5.0))  # the card is still on stream
        return FakeSessionApi.upload(api, session_id, items)

    api.upload = upload_then_change
    assert recorder.flush()
    assert recorder.pending == 1  # the newer version waits for the next flush
    assert recorder.flush()
    assert recorder.pending == 0
    assert [batch[0]["duration"] for batch in api.uploads] == [1.0, 5.0]


def test_large_queues_go_up_in_api_sized_batches():
    api = FakeSessionApi()
    recorder = _recorder(api)
    for index in range(MAX_BATCH + 3):
        recorder.queue(_row(f"row-{index}"))
    assert recorder.flush()
    assert [len(batch) for batch in api.uploads] == [MAX_BATCH, 3]


def test_a_transient_failure_keeps_the_rows_and_resends_them():
    api = FakeSessionApi(replies=[UploadReply(Outcome.RETRY, "HTTP 502")])
    logs: list[str] = []
    recorder = _recorder(api, logs)
    recorder.queue(_row("a"))
    assert not recorder.flush()
    assert recorder.pending == 1
    assert "retrying" in logs[-1]
    assert recorder.flush()
    first, second = api.uploads
    assert [i["event_id"] for i in first] == [i["event_id"] for i in second] == ["a"]
    assert recorder.pending == 0 and recorder.rows_saved == 1


def test_a_rejected_batch_is_dropped_and_the_rest_continue():
    api = FakeSessionApi(replies=[UploadReply(Outcome.REJECTED, "HTTP 400")])
    logs: list[str] = []
    recorder = _recorder(api, logs)
    recorder.queue(_row("a"))
    assert recorder.flush()
    assert recorder.dropped == 1 and recorder.pending == 0
    assert "rejected" in logs[-1]
    recorder.queue(_row("b"))
    assert recorder.flush() and recorder.rows_saved == 1


def test_a_closed_session_stops_uploads_for_good():
    api = FakeSessionApi(replies=[UploadReply(Outcome.STOPPED, "HTTP 409: closed")])
    logs: list[str] = []
    recorder = _recorder(api, logs)
    recorder.queue(_row("a"))
    assert recorder.flush()
    assert recorder.stopped == "HTTP 409: closed"
    assert "uploads stopped" in logs[-1]
    recorder.queue(_row("b"))  # ignored from now on
    recorder.count_call()
    assert recorder.pending == 0 and len(api.uploads) == 1
    recorder.close()
    assert api.closed == [] and api.reported == []


def test_a_full_queue_drops_the_oldest_row_and_says_so_once():
    logs: list[str] = []
    recorder = _recorder(FakeSessionApi(), logs, max_pending=3)
    for name in "ABCDE":
        recorder.queue(_row(name, name))
    assert recorder.pending == 3 and recorder.dropped == 2
    assert sum("queue full" in line for line in logs) == 1
    recorder.queue(
        _row("E", "E", duration=9.0)
    )  # an update of a queued row drops nothing
    assert recorder.pending == 3 and recorder.dropped == 2


def test_paid_calls_are_reported_after_the_rows():
    api = FakeSessionApi()
    recorder = _recorder(api)
    for _ in range(3):
        recorder.count_call()
    recorder.queue(_row("a"))
    assert recorder.flush()
    assert len(api.uploads) == 1
    assert api.reported == [(recorder.run_id, 3)]
    assert recorder.flush()  # nothing new: no second report
    assert len(api.reported) == 1
    recorder.count_call()
    assert recorder.flush()
    assert api.reported[-1] == (recorder.run_id, 4)


def test_a_failed_call_report_is_retried():
    api = FakeSessionApi(call_replies=[UploadReply(Outcome.RETRY, "down")])
    recorder = _recorder(api)
    recorder.count_call()
    assert not recorder.flush()
    assert recorder.flush()
    assert api.reported == [(recorder.run_id, 1), (recorder.run_id, 1)]


def test_every_run_reports_under_its_own_id():
    first, second = _recorder(FakeSessionApi()), _recorder(FakeSessionApi())
    assert first.run_id != second.run_id


def test_close_finishes_the_rows_uploads_and_closes_the_session():
    api = FakeSessionApi()
    logs: list[str] = []
    recorder = _recorder(api, logs, min_card_time=0)
    history = recorder.history(lambda: "tcg")
    history.observe(
        "identified", {"full_name": "Charizard", "name": "Charizard"}, now=0.0
    )
    history.observe(
        "identified", {"full_name": "Charizard", "name": "Charizard"}, now=2.0
    )
    recorder.count_call()
    recorder.close()
    rows = [item for batch in api.uploads for item in batch]
    assert rows and rows[-1]["duration"] >= 2.0  # the open row was closed by close()
    assert api.reported == [(recorder.run_id, 1)]
    assert api.closed == [SESSION_ID]
    assert "1 card(s) and 1 paid call(s)" in logs[-1] and "session closed" in logs[-1]
    recorder.queue(_row("late"))  # after close: ignored
    recorder.close()  # idempotent
    assert api.closed == [SESSION_ID]


def test_histories_follow_the_recorders_page_rules():
    recorder = _recorder(FakeSessionApi(), min_card_time=2.5, split_results=True)
    history = recorder.history(lambda: "sport")
    assert history._min_card_time == 2.5 and history._split is True


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
    recorder.queue(_row("a"))
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
        recorder.queue(_row("a"))
        assert await wait_until(lambda: recorder.rows_saved == 1)
    finally:
        recorder.close()
    assert api.closed == [SESSION_ID]


def test_queue_and_count_are_safe_from_many_threads():
    api = FakeSessionApi()
    recorder = _recorder(api)

    def work(worker: int) -> None:
        for index in range(50):
            recorder.queue(_row(f"{worker}-{index}"))
            recorder.count_call()

    threads = [threading.Thread(target=work, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert recorder.paid_calls == 400
    assert recorder.flush()
    assert sum(len(batch) for batch in api.uploads) == 400
    assert api.reported == [(recorder.run_id, 400)]
