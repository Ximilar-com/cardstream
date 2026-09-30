"""ShowHistory: the page's history rows, in Python, for the session."""

from __future__ import annotations

import pytest

from cardstream.core.show_history import ShowHistory, card_key

WALL_START = 1_790_000_000.0


class Clock:
    """A manual clock; the tracker's monotonic and wall clocks move together."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def wall(self) -> float:
        return WALL_START + self.now


def card(name: str = "Mega Greninja ex", number: str = "22") -> dict:
    """A fresh identification dict, as one paid call returns it."""
    return {
        "full_name": f"{name} Chaos Rising (CRI) #{number}",
        "name": name,
        "set": "Chaos Rising",
        "card_number": number,
    }


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def emitted() -> list:
    return []


def make_history(clock, emitted, **kwargs) -> ShowHistory:
    return ShowHistory(
        lambda row, duration: emitted.append((row.event_id, row.calls, duration, row)),
        id_type=lambda: "tcg",
        clock=clock,
        wall_clock=clock.wall,
        **kwargs,
    )


def step(history, clock, state, ident=None, until=None, every=0.1):
    """Feed frames of one state from now until ``until`` (inclusive)."""
    history.observe(state, ident)
    while until is not None and clock.now + every <= until + 1e-9:
        clock.now = round(clock.now + every, 6)
        history.observe(state, ident)


def test_three_calls_on_one_card_are_one_row_with_the_total_time(clock, emitted):
    """The reported case: the same card identified three times, 8.1 s on stream."""
    history = make_history(clock, emitted)
    first, second, third = card(), card(), card()  # three paid calls, same card
    step(history, clock, "moving", until=0.4)  # appears before any name lands
    step(history, clock, "identified", first, until=3.0)
    step(history, clock, "empty", until=3.5)  # leaves ...
    step(history, clock, "identified", second, until=6.0)  # ... and comes back
    step(history, clock, "identified", third, until=8.1)
    history.finish()

    rows = {event_id for event_id, *_ in emitted}
    assert len(rows) == 1
    _, calls, duration, row = emitted[-1]
    assert calls == 3
    # In frame from its first frame at 0.0 to 3.0, then from 3.5 to 8.1: 7.6 s.
    assert duration == pytest.approx(7.6, abs=0.01)
    assert row.seen == pytest.approx(WALL_START)  # when it appeared, not when named
    assert row.id_type == "tcg"
    assert row.identification is first  # the row keeps its first match


def test_a_card_under_min_card_time_is_never_sent(clock, emitted):
    history = make_history(clock, emitted, min_card_time=1.0)
    step(history, clock, "identified", card(), until=0.8)
    step(history, clock, "empty", until=1.5)
    history.finish()
    assert emitted == []


def test_a_row_is_sent_when_it_earns_its_place_and_again_when_the_card_leaves(
    clock, emitted
):
    history = make_history(clock, emitted, min_card_time=1.0)
    step(history, clock, "identified", card(), until=2.5)
    assert [round(duration, 1) for *_, duration, _ in emitted] == [1.0]
    step(history, clock, "empty")
    assert [round(duration, 1) for *_, duration, _ in emitted] == [1.0, 2.5]
    assert len({event_id for event_id, *_ in emitted}) == 1


def test_a_card_held_for_long_is_refreshed(clock, emitted):
    history = make_history(clock, emitted, min_card_time=0, refresh_seconds=10)
    step(history, clock, "identified", card(), until=25.0, every=0.5)
    assert [round(duration) for *_, duration, _ in emitted] == [0, 10, 20]


def test_another_card_in_between_starts_a_new_row(clock, emitted):
    history = make_history(clock, emitted, min_card_time=0)
    step(history, clock, "identified", card("A", "1"), until=2.0)
    step(history, clock, "empty", until=2.5)
    step(history, clock, "identified", card("B", "2"), until=4.0)
    step(history, clock, "empty", until=4.5)
    step(history, clock, "identified", card("A", "1"), until=6.0)
    history.finish()
    names = []
    for event_id, _, _, row in emitted:
        if (event_id, row.identification["name"]) not in names:
            names.append((event_id, row.identification["name"]))
    assert [name for _, name in names] == ["A", "B", "A"]


def test_a_card_swapped_in_place_starts_its_own_clock(clock, emitted):
    history = make_history(clock, emitted, min_card_time=0)
    step(history, clock, "identified", card("A", "1"), until=3.0)
    clock.now = 3.2
    step(history, clock, "identified", card("B", "2"), until=5.0)
    history.finish()
    final = {row.identification["name"]: duration for _, _, duration, row in emitted}
    assert final["A"] == pytest.approx(3.2)
    assert final["B"] == pytest.approx(1.8)


def test_split_results_makes_every_appearance_its_own_row(clock, emitted):
    history = make_history(clock, emitted, min_card_time=0, split_results=True)
    ident = card()
    step(history, clock, "identified", ident, until=2.0)
    step(history, clock, "empty", until=2.5)
    # The same card back from the analyzer's memory: no new paid call.
    step(history, clock, "identified", ident, until=4.0)
    history.finish()
    rows = {}
    for event_id, calls, duration, _ in emitted:
        rows[event_id] = (calls, duration)
    assert [calls for calls, _ in rows.values()] == [1, 0]
    assert [round(duration, 1) for _, duration in rows.values()] == [2.0, 1.5]


def test_a_repeated_identification_is_not_a_new_call(clock, emitted):
    history = make_history(clock, emitted, min_card_time=0)
    ident = card()
    step(history, clock, "identified", ident, until=3.0)  # every frame shows it
    history.finish()
    assert emitted[-1][1] == 1


def test_finish_closes_the_card_in_frame(clock, emitted):
    history = make_history(clock, emitted, min_card_time=0)
    step(history, clock, "identified", card(), until=4.0)
    history.finish()
    assert emitted[-1][2] == pytest.approx(4.0)
    clock.now = 10.0
    history.finish()  # idempotent: nothing is on stream any more
    assert emitted[-1][2] == pytest.approx(4.0)


def test_the_key_is_the_pages():
    assert card_key(card()) == card_key(card())
    assert card_key(card(number="23")) != card_key(card())
