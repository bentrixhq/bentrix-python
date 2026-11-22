import warnings

import pytest

from bentrix import _runtime as rt
from stub_server import TOKEN, Stub


def connect(fx, stub, **kw):
    return fx.Bentrix(stub.url, TOKEN, **kw)


def test_a_typed_call_goes_over_the_2026_wire_and_the_stub_finds_nothing_to_object_to(fx):
    with Stub() as stub:
        b = connect(fx, stub)
        out = b.fx.bench_create("lab1", name="n")
        assert out == fx.BenchCreateOut(bench="b1", uri="bentrix://bench/b1")
        assert stub.violations == []
        method, params = stub.calls[-1]
        assert method == "tools/call" and params["name"] == "fx_bench_create"
        assert params["arguments"] == {"lab": "lab1", "name": "n"}, "None arguments are not sent"


def test_a_keyword_named_argument_is_renamed_for_python_and_not_on_the_wire(fx):
    with Stub() as stub:
        connect(fx, stub).fx.bench_create("lab1", from_="2026-01-02")
        assert stub.calls[-1][1]["arguments"] == {"lab": "lab1", "from": "2026-01-02"}


def test_an_enum_argument_reaches_the_wire_as_its_token(fx):
    with Stub() as stub:
        connect(fx, stub).fx.bench_create("lab1", tick_model="ohlc_m1")
        assert stub.calls[-1][1]["arguments"]["tick_model"] == "ohlc_m1"


def test_an_answer_is_typed_nested_and_forward_compatible(fx):
    with Stub() as stub:
        job = connect(fx, stub).fx.replay_step("b1", "m1", 1)
        out = job.wait()
        assert isinstance(out, fx.ReplayStepOut) and isinstance(out.undone, fx.Undone)
        assert out.quotes == {"EURUSD": 1.1}
        assert out.refused is None, "an absent optional is None"
        # the stub's answer carried a key this SDK has never heard of: ignored, not an error


def test_a_missing_required_key_names_itself(fx):
    with pytest.raises(KeyError, match="moment"):
        fx.ReplayStepOut.from_dict({"moment_ms": 1, "quotes": {}, "undone": {"deals": 0, "requests": 0}})


def test_a_job_is_polled_until_it_completes_and_waiting_for_the_person_is_not_a_failure(fx):
    with Stub() as stub:
        job = connect(fx, stub).fx.replay_step("b1", "tick", 999)
        assert job.handle == "j1"
        out = job.wait(poll=0.01)
        assert out.moment == "later" and out.undone.requests == 2
        assert stub.job_polls == 3


def test_a_job_that_never_finishes_times_out(fx):
    with Stub() as stub:
        stub.job_polls = -10_000  # always "working"
        job = connect(fx, stub).fx.replay_step("b1", "tick", 999)
        with pytest.raises(TimeoutError):
            job.wait(timeout=0.05, poll=0.01)


def test_a_tool_error_is_the_products_sentence(fx):
    with Stub() as stub:
        b = connect(fx, stub)
        with pytest.raises(rt.ToolError, match="bench b7 is not in lab3"):
            b.call("fx_fail", {})


def test_a_question_this_client_cannot_answer_is_said_not_swallowed(fx):
    with Stub() as stub:
        with pytest.raises(rt.InputRequired):
            connect(fx, stub).call("fx_ask", {})


def test_a_refused_request_is_an_rpc_error_with_its_code(fx):
    with Stub() as stub:
        with pytest.raises(rt.RpcError) as e:
            connect(fx, stub).call("no_such_tool", {})
        assert e.value.code == -32602 and "Unknown tool" in e.value.message


def test_a_busy_answer_is_waited_out_and_the_request_sent_again_bounded(fx, monkeypatch):
    """**Bentrix's rate limit is waited out, never surfaced while it passes**: a request answered *busy* (HTTP 429, -40001) ran nothing, so
    it is sent again after the wait the answer names; past ``BUSY_RETRIES`` the refusal is raised.

    Mutations (2026-11-06), each red: the retry removed (the first busy answer was raised); the bound
    removed (the eleventh busy answer was not raised); the named wait ignored (it slept a second)."""
    slept = []
    monkeypatch.setattr(rt.time, "sleep", lambda s: slept.append(s))
    with Stub() as stub:
        b = connect(fx, stub)
        stub.busy = 3
        out = b.fx.bench_create("plain")
        assert out.bench == "b1", "the first busy answer was raised"
        assert [m for m, _ in stub.calls].count("busy") == 3
        assert slept == [0.02, 0.02, 0.02], f"the named wait ignored: {slept}"
        stub.busy = rt.BUSY_RETRIES + 1
        with pytest.raises(rt.RpcError) as e:
            b.fx.bench_create("plain")
        assert e.value.code == rt.BUSY, "the eleventh busy answer was not raised"
    assert rt.busy_wait_s({"retryAfterMs": 60_000}) == rt.BUSY_WAIT_MAX_S
    assert rt.busy_wait_s(None) == 1.0


def test_a_response_on_a_request_scoped_event_stream_is_found_among_notifications(fx):
    with Stub() as stub:
        out = connect(fx, stub).fx.bench_create("sse")
        assert out.bench == "b1"


def test_a_cursor_paged_resource_is_followed_to_its_last_page(fx):
    with Stub() as stub:
        b = connect(fx, stub)
        first = b.fx.bench_journal("b1")
        assert first.next and first.lines[0].text == "a"
        pages = list(b.fx.bench_journal_pages("b1"))
        assert [p.lines[0].text for p in pages] == ["a", "b"]
        assert pages[-1].next is None
        assert stub.violations == [], "Mcp-Name carries the URI on resources/read"


def test_a_stream_yields_items_and_carries_its_cursor_forward(fx):
    with Stub() as stub:
        it = connect(fx, stub).fx.bench_quotes("b1", timeout_ms=10)
        got = [next(it) for _ in range(3)]
        assert [q.symbol for q in got] == ["EURUSD", "GBPUSD", "USDJPY"]
        assert stub.after_seen[:2] == [None, "c1"], "the first wait has no cursor; the next carries the page's"
        wait = [p for m, p in stub.calls if m == "tools/call" and p["name"] == "wait_for"][0]
        assert wait["arguments"] == {"uri": "bentrix://bench/b1/quotes", "timeout_ms": 10}


def test_a_stream_asks_again_when_busy_and_ends_on_a_rewind_with_its_cursor(fx):
    """**A passing refusal does not end a stream, and a rewind is said** (the strict audit of
    2026-11-04, S3/S4): ``busy`` and ``expired`` are asked again; a page saying ``rewound`` ends the
    iteration with ``Rewound`` carrying the cursor to read on from, after the items before it.

    Mutations (2026-11-04), each red: the retry removed (the first ``busy`` ended the stream); the
    rewound check removed (the iteration went on past the rewind)."""
    uri = "bentrix://bench/b1/quotes"
    with Stub() as stub:
        stub.streams[uri] = [
            {"__error__": "busy"},
            {"items": [{"symbol": "EURUSD", "bid": 1.1, "ask": 1.2}], "next": "c1"},
            {"__error__": "expired"},
            {"items": [], "next": "c0", "rewound": True},
            # Read on past a rewind, this would be handed as new.
            {"items": [{"symbol": "GBPUSD", "bid": 1.3, "ask": 1.4}], "next": "c1"},
        ]
        it = connect(fx, stub).fx.bench_quotes("b1", timeout_ms=10)
        first = next(it)
        assert first.symbol == "EURUSD", "the busy answer ended the stream"
        with pytest.raises(rt.Rewound) as caught:
            next(it)
        assert caught.value.cursor == "c0", caught.value.cursor


def test_the_sdk_refuses_a_server_on_another_major_line(fx):
    with Stub(catalogue="2.0.0") as stub:
        with pytest.raises(rt.IncompatibleCatalogue, match="major"):
            connect(fx, stub)


def test_the_sdk_warns_about_a_server_a_minor_ahead_and_not_about_a_patch(fx):
    with Stub(catalogue="1.3.0") as stub:
        with pytest.warns(RuntimeWarning, match="newer"):
            connect(fx, stub)
    with Stub(catalogue="1.0.7") as stub, warnings.catch_warnings():
        warnings.simplefilter("error")
        connect(fx, stub)


def test_the_version_falls_back_to_the_whoami_tool_when_discover_carries_none(fx):
    with Stub(discover_meta=False) as stub:
        connect(fx, stub)
        assert any(p.get("name") == "agent_whoami" for _, p in stub.calls)


def test_a_server_that_reports_no_catalogue_at_all_is_refused(fx):
    with Stub(discover_meta=False, catalogue=None) as stub:
        with pytest.raises(rt.IncompatibleCatalogue):
            connect(fx, stub)


def test_the_version_check_can_be_turned_off_for_a_deliberate_mismatch(fx):
    with Stub(catalogue="9.0.0") as stub:
        connect(fx, stub, check_version=False)


def test_a_document_round_trips_through_the_wire_shape(fx):
    doc = fx.StrategyDocument(schema_version=1, name="s", markets=["EURUSD"])
    assert doc.to_dict() == {"schema_version": 1, "name": "s", "markets": ["EURUSD"]}, "None optionals are not written"
    assert fx.StrategyDocument.from_dict({**doc.to_dict(), "max_open": 2}).max_open == 2


def test_a_template_expands_paths_and_queries_and_drops_what_is_not_given():
    t = "bentrix://bench/{bench}/journal{?after,limit}"
    assert rt.expand_template(t, {"bench": "a b"}, {"after": None}) == "bentrix://bench/a%20b/journal"
    assert (
        rt.expand_template(t, {"bench": "b"}, {"after": "x/y", "limit": 5})
        == "bentrix://bench/b/journal?after=x%2Fy&limit=5"
    )
    with pytest.raises(ValueError):
        rt.expand_template(t, {})


def test_a_header_value_that_plain_ascii_cannot_carry_is_base64_sentinel_encoded():
    assert rt._header_value("fx_bench_create") == "fx_bench_create"
    assert rt._header_value("héllo") == "=?base64?aMOpbGxv?="
    assert rt._header_value(" padded ") == "=?base64?IHBhZGRlZCA=?="
    literal = "=?base64?literal?="
    assert rt._header_value(literal) != literal, "a plain value that looks like the sentinel is encoded too"


def test_the_token_is_never_shown(fx):
    with Stub() as stub:
        b = connect(fx, stub)
        assert TOKEN not in repr(b)


def test_a_long_poll_longer_than_the_clients_timeout_is_waited_for(fx):
    """**A call that asks the server to wait is given that long** (found 2026-11-02, LiveEnv on a live
    paper workbench: ``wait_for`` at 90 s against a 60 s client ended as a socket timeout). A client
    timed at 0.5 s asks for a 1.5 s wait, and the stub holds 1 s: the answer arrives.

    Mutation (2026-11-02): the request timed by the client's own timeout alone — red, ``timed out``."""
    with Stub() as stub:
        stub.wait_for_delay = 1.0
        b = connect(fx, stub, timeout=0.5)
        page = b.call("wait_for", {"uri": "bentrix://quotes", "timeout_ms": 1500})
        assert page["items"], page
