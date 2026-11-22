"""`bentrix.Events`: a function bound to a stream receives each item once, in order, on the caller's thread."""

import threading
import time

import pytest

from bentrix import BentrixError, Events
from stub_server import TOKEN, Stub

BENCH = "bentrix://lab/1/bench/2"
CLOSED = f"{BENCH}/bars/EURUSD/M5/closed"
TICKS = f"{BENCH}/ticks/EURUSD/live"
BAR_COLUMNS = ["time_ms", "open", "high", "low", "close", "tick_volume", "spread_points"]


def connect(fx, stub):
    return fx.Bentrix(stub.url, TOKEN)


def run_briefly(events):
    """Run until the handlers stop it, or five seconds at most, so a lost event fails rather than hangs."""
    deadline = time.monotonic() + 5
    events.run(until=lambda: time.monotonic() > deadline)


def test_each_closed_bar_reaches_its_handler_once_keyed_by_the_columns_and_the_cursor_carries(fx):
    with Stub() as stub:
        stub.streams[CLOSED] = [
            {"columns": BAR_COLUMNS, "items": [[300000, 1, 2, 0.5, 1.5, 10, 2]], "next": "300000"},
            {"columns": BAR_COLUMNS, "items": [[600000, 1.5, 2, 1, 1.8, 7, 2], [900000, 1.8, 1.9, 1.7, 1.75, 3, 2]], "next": "900000"},
        ]
        events = Events(connect(fx, stub), timeout_ms=10)
        got = []

        @events.on_bar_close(BENCH, "EURUSD", "M5")
        def closed(bar):
            got.append(bar)
            if len(got) == 3:
                events.stop()

        run_briefly(events)
        assert [b["time_ms"] for b in got] == [300000, 600000, 900000], "a bar lost, repeated or out of order"
        assert got[0]["close"] == 1.5, "an event is not keyed by the stream's columns"
        assert stub.stream_after[CLOSED][:2] == ["c0", "300000"], "the cursor does not start at the present and carry on"


def test_a_rewind_calls_on_rewind_with_the_new_cursor_and_the_bars_come_again(fx):
    with Stub() as stub:
        stub.streams[CLOSED] = [
            {"columns": BAR_COLUMNS, "items": [[600000, 1, 1, 1, 1, 1, 0]], "next": "600000"},
            {"columns": BAR_COLUMNS, "items": [], "next": "300000", "rewound": True},
            {"columns": BAR_COLUMNS, "items": [[600000, 2, 2, 2, 2, 1, 0]], "next": "600000"},
        ]
        events = Events(connect(fx, stub), timeout_ms=10)
        seen = []
        events.on_bar_close(BENCH, "EURUSD", "M5", lambda bar: seen.append(("bar", bar["close"])) or (len(seen) == 3 and events.stop()), on_rewind=lambda r: seen.append(("rewind", r["next"])))
        run_briefly(events)
        assert seen == [("bar", 1), ("rewind", "300000"), ("bar", 2)], f"a rewind was not said between the two closes: {seen}"


def test_handlers_run_on_the_callers_thread_one_at_a_time_across_bindings(fx):
    with Stub() as stub:
        stub.streams[CLOSED] = [{"columns": BAR_COLUMNS, "items": [[300000, 1, 1, 1, 1, 1, 0]], "next": "300000"}]
        stub.streams[TICKS] = [{"columns": ["time_ms", "bid", "ask", "last", "volume", "flags"], "items": [[1, 1.1, 1.2, 0, 0, 6], [2, 1.1, 1.3, 0, 0, 4]], "next": "d:2"}]
        events = Events(connect(fx, stub), timeout_ms=10)
        threads, kinds = set(), []

        def note(kind):
            def h(e):
                threads.add(threading.get_ident())
                kinds.append(kind)
                if len(kinds) == 3:
                    events.stop()
            return h

        events.on_bar_close(BENCH, "EURUSD", "M5", note("bar"))
        events.on_tick(BENCH, "EURUSD", note("tick"))
        run_briefly(events)
        assert sorted(kinds) == ["bar", "tick", "tick"], kinds
        assert threads == {threading.get_ident()}, "a handler ran off the caller's thread"


def test_a_handler_that_raises_stops_run_with_its_exception(fx):
    with Stub() as stub:
        stub.streams[TICKS] = [{"columns": ["time_ms"], "items": [[1]], "next": "d:1"}]
        events = Events(connect(fx, stub), timeout_ms=10)

        @events.on_tick(BENCH, "EURUSD")
        def boom(t):
            raise ValueError("the handler's own")

        with pytest.raises(ValueError, match="the handler's own"):
            run_briefly(events)


def test_a_bench_that_is_not_a_uri_and_a_run_with_nothing_bound_are_refused(fx):
    with Stub() as stub:
        events = Events(connect(fx, stub))
        with pytest.raises(BentrixError, match="not a workbench"):
            events.on_tick("lab 1 bench 2", "EURUSD", print)
        with pytest.raises(BentrixError, match="nothing is bound"):
            events.run()
