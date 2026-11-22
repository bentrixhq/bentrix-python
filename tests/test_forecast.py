"""**What a forecasting model reads**: ``bentrix.forecast.window`` and ``windows`` walked on the CI
tape, embedded, through the same tools and stream an agent reaches over MCP."""

import pytest

np = pytest.importorskip("numpy")

from bentrix.forecast import COLUMNS, window, windows  # noqa: E402

H1 = 3_600_000
M15 = 900_000


def _stand(local, bench):
    local.arm(bench)
    at = local.call("replay_step", bench=bench, unit="tick", n=1)["cursor_ms"]
    return local.advance(bench, (at // H1 + 40) * H1 + 20 * 60_000)


def test_a_window_is_cut_at_one_instant_and_aligned_to_an_interval(tape):
    """**Every frame ends at the cut, and aligned to H1 the M15 frame ends where the H1 bar ended.**

    Mutation (2026-11-07): ``aligned_to`` dropped from the call's arguments — red, the aligned cut
    stood at the cursor (``a.at_ms < w.at_ms`` failed)."""
    local, bench = tape
    cursor = _stand(local, bench)
    w = window(local, bench, ["M15", "H1"], symbol="XAUUSD", limit=40)
    assert w.at_ms == w.cursor_ms and w.cursor_ms <= cursor + H1
    m15 = w.bars("XAUUSD", "M15")
    assert m15.shape[1] == len(COLUMNS) and 0 < len(m15) <= 40
    assert (w.times("XAUUSD", "M15") + M15 <= w.at_ms).all(), "an M15 bar still forming at the cursor is a row"
    a = window(local, bench, ["M15"], symbol="XAUUSD", aligned_to="H1", limit=8)
    assert a.at_ms % H1 == 0 and a.at_ms < w.at_ms, (a.at_ms, w.at_ms)
    t = a.times("XAUUSD", "M15")
    assert (t + M15 <= a.at_ms).all(), "the M15 frame runs past the H1 close"
    assert t[-1] + M15 == a.at_ms, "the M15 frame does not end on the H1 close"


def test_a_model_reads_one_window_at_each_close(tape):
    """**Each H1 close once, oldest first, each with its window cut where that bar ended** — however far the
    market moved between two reads.

    Mutation (2026-11-07): ``next_page`` re-reading from the first cursor (``cursor`` not
    advanced) — red, *a close came twice*."""
    local, bench = tape
    at = [_stand(local, bench)]

    def market():
        at[0] = local.advance(bench, (at[0] // H1 + 3) * H1 + 60_000)

    ws = windows(local, bench, "XAUUSD", "H1", frames=["M15", "H1"], limit=12, timeout_ms=0, idle_limit=2, before_wait=market)
    seen = []
    for c in ws:
        seen.append(c)
        if len(seen) >= 7:
            break
    assert len(seen) == 7, len(seen)
    opens = [c.bar[0] for c in seen]
    assert opens == sorted(set(opens)), f"a close came twice or out of order: {opens}"
    assert all(b - a == H1 for a, b in zip(opens, opens[1:])), f"a close was skipped: {opens}"
    for c in seen:
        assert c.close_ms == c.bar[0] + H1
        assert c.window.at_ms == c.close_ms, "the window is not cut at its bar's close"
        assert c.window.times("XAUUSD", "H1")[-1] == c.bar[0], "the H1 frame does not end on its bar"
        assert (c.window.times("XAUUSD", "M15") + M15 <= c.close_ms).all(), "the M15 frame runs past the close"
    assert ws.rewound == 0


def test_a_model_puts_a_forecast_at_each_close_and_each_is_read_by_arrival(tape):
    """**The loop a forecasting model runs**: at each H1 close a window, a forecast anchored at that close put back;
    each lands stamped at the workbench's clock, and a reader just before an arrival sees the forecast before it —
    never the one that had not arrived.

    Mutation (2026-11-07): ``Forecast.after`` anchoring at the bar after the close (``bar[0] + H1``) —
    red, *the forecast was refused: its anchor has not closed*."""
    from bentrix.forecast import Forecast, clear, get

    local, bench = tape
    at = [_stand(local, bench)]

    def market():
        at[0] = local.advance(bench, (at[0] // H1 + 1) * H1 + 60_000)

    arrivals = []
    for c in windows(local, bench, "XAUUSD", "H1", frames=["H1"], limit=4, timeout_ms=0, idle_limit=2, before_wait=market):
        last = c.window.close("XAUUSD", "H1")[-1]
        f = Forecast.after(c, "naive", "H1", epoch=len(arrivals)).step(last * 1.001, open=last, high=last * 1.002, low=last * 0.999, bands=[(last * 0.995, last * 1.005, 0.8)])
        put = f.put(local, bench)
        assert put["anchor_ms"] == c.bar[0] and put["state"] == "current", put
        arrivals.append(put["put_ms"])
        if len(arrivals) == 3:
            break
    assert arrivals == sorted(arrivals) and len(set(arrivals)) == 3, arrivals
    seen = get(local, bench, model="naive", at_ms=arrivals[2] - 1)["forecasts"]
    assert [s["put_ms"] for s in seen] == [arrivals[1]], "a reader before an arrival saw the forecast that had not arrived"
    assert get(local, bench, model="naive")["arrivals"][0]["put_ms"] == arrivals
    assert clear(local, bench, model="naive") == 1
    with pytest.raises(ValueError):
        Forecast.from_dict({**f.spec(), "put_ms": arrivals[0]})


def test_an_element_forecast_names_no_interval_and_carries_its_rule():
    """**A forecast of a chart built by price**: ``of_elements`` names no interval, carries its rule in
    ``form.elements`` and a step's ``expected_ms``; read back, it is the same document.

    Mutation (2026-11-08): ``spec`` writing ``timeframe`` whatever it is — red, the element form named an interval."""
    from bentrix.forecast import Forecast
    f = Forecast.of_elements("bricks", "EURUSD", "renko", 1_700_000_000_000, elements={"renko_box": 0.0005, "renko_reversal": 2})
    f.step(1.1005, expected_ms=1_700_000_600_000).step(1.1010, expected_ms=1_700_001_200_000)
    d = f.spec()
    assert d["form"] == {"chart_type": "renko", "elements": {"renko_box": 0.0005, "renko_reversal": 2}}, d["form"]
    assert [s["expected_ms"] for s in d["steps"]] == [1_700_000_600_000, 1_700_001_200_000]
    assert Forecast.from_dict(d).spec() == d
    t = Forecast("m", "EURUSD", "H1", 0).step(1.0).spec()
    assert t["form"] == {"chart_type": "candlestick", "timeframe": "H1"}


class _Sheet:
    """A door whose ``bentrix://forecasts/eval`` event carries scripted moves, a page at a time."""

    def __init__(self, pages):
        self.pages, self.asked = list(pages), []

    def call(self, tool, **args):
        assert tool == "wait_for" and args["uri"] == "bentrix://forecasts/eval", (tool, args)
        self.asked.append(args["after"])
        items = self.pages.pop(0) if self.pages else []
        return {"items": items, "next": str(int(args["after"]) + len(items))}


def test_an_evals_sheet_is_waited_on_and_a_decline_is_not_a_failure():
    """**The run, the decline and the failure are three answers**: ``eval_started`` answers a run at
    once; over an ask it watches the event from its start, skipping another ask's moves, and answers the run the
    person's *Make it and run* started, raises ``DeclinedByThePerson`` for *No* and ``EvalFailed`` with the why.

    Mutation (2026-11-08): ``declined`` read as a failure — red, a decline raised EvalFailed."""
    from bentrix.forecast import DeclinedByThePerson, EvalFailed, eval_started
    assert eval_started(_Sheet([]), {"eval": 7}) == 7
    ask = {"ask": "bentrix://ask/2", "asks": "make_laboratory", "watch": "bentrix://forecasts/eval{?after}"}
    other = {"ask": "bentrix://ask/1", "state": "declined"}
    door = _Sheet([[other, {"ask": "bentrix://ask/2", "state": "asked"}], [{"ask": "bentrix://ask/2", "state": "confirmed"}, {"ask": "bentrix://ask/2", "state": "made_laboratory"}], [{"ask": "bentrix://ask/2", "state": "started", "eval": 11}]])
    assert eval_started(door, ask, idle_limit=3) == 11
    assert door.asked == ["0", "2", "4"], door.asked
    with pytest.raises(DeclinedByThePerson):
        eval_started(_Sheet([[{"ask": "bentrix://ask/2", "state": "declined", "by": "the person"}]]), ask, idle_limit=3)
    with pytest.raises(EvalFailed) as e:
        eval_started(_Sheet([[{"ask": "bentrix://ask/2", "state": "failed", "why": "the laboratory was not made"}]]), ask, idle_limit=3)
    assert e.value.why == "the laboratory was not made"
    with pytest.raises(TimeoutError):
        eval_started(_Sheet([]), ask, idle_limit=2)


def test_an_onnx_model_file_is_installed_checked_and_a_policy_run_by_bentrix():
    """**A model file for Bentrix's engines**: ``export_manifest`` writes the manifest the server
    reads, refusing a transform it does not apply; ``ModelFile.from_onnx(...).install`` sends the file as base64 with
    it, ``check`` asks ``model_check``; the Gym's ``Model(engine="bentrix")`` runs a policy through ``model_infer`` and
    reads scores as their largest.

    Mutation (2026-11-08): the policy's argmax as argmin — red, the action was not the largest score's."""
    import base64
    import importlib
    import os
    import tempfile
    m = importlib.import_module("bentrix.model")
    with pytest.raises(ValueError):
        m.export_manifest("x", "M1", 2, transform="log_returns")
    man = m.export_manifest("x", "M15", 3, frames=["M15", "H1"], fields=("close", "high"), transform="returns", output="delta")
    assert man["input"] == {"fields": ["close", "high"], "transform": "returns"} and man["output"] == {"kind": "delta", "pick": "first"}

    class Door:
        def __init__(self):
            self.calls = []

        def call(self, tool, **a):
            self.calls.append((tool, a))
            return {"model_infer": {"output": [0.1, 0.7, 0.2]}}.get(tool, {"ok": True})

    d = Door()
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "m.onnx")
        open(p, "wb").write(b"\x08\x07")
        f = m.Model.from_onnx(p, man)
        f.install(d, "x")
        f.check(d)
    assert d.calls[0][0] == "model_install" and base64.b64decode(d.calls[0][1]["onnx_base64"]) == b"\x08\x07"
    assert d.calls[1] == ("model_check", {"file": "x"})
    live = importlib.import_module("bentrix.live")
    pol = live.Model("x", engine="bentrix", door=d)
    assert pol([1.0, 2.0]) == 1, "the action was not the largest score's"
    assert d.calls[-1] == ("model_infer", {"name": "x", "input": [1.0, 2.0]})
