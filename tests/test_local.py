"""**BENTRIX's engine in this Python process**: the extension module's own walks.

Skipped where the wheel with ``bentrix._native`` is not installed (the pure client). The backtest's
walk runs on the CI tape (``conftest.tape``).
"""

import pytest

native = pytest.importorskip("bentrix._native")

from bentrix.backtest import Strategy, run  # noqa: E402
from bentrix.local import Local  # noqa: E402


def test_an_empty_workspace_opens_answers_and_is_held(tmp_path):
    """An empty workspace opens with no window; a tool answers in the catalogue's shape, an unknown one
    is refused as an error, and a second embedding of the same workspace is refused by name."""
    local = Local(workspace=str(tmp_path / "ws"), store=str(tmp_path / "store"))
    state = local.call("get_state")
    assert "laboratories" in state and "accounts" in state
    with pytest.raises(RuntimeError):
        local.call("no_such_tool")
    with pytest.raises(RuntimeError):
        Local(workspace=str(tmp_path / "ws"), store=str(tmp_path / "store"))
    assert native.interval_ms("H1") == 3_600_000
    assert native.interval_ms("MN1") is None


class _BuyThenClose(Strategy):
    """Buys at the third closed candle, closes everything at the tenth."""

    def __init__(self):
        self.bars = []
        self.answers = []

    def on_bar(self, bar, ctx):
        self.bars.append(bar)
        if len(self.bars) == 3:
            self.answers.append(ctx.buy(0.01))
        elif len(self.bars) == 10:
            self.answers.append(ctx.close_all())


def test_a_strategy_runs_on_a_recording_a_closed_candle_at_a_time(tape):
    """``backtest.run`` hands the strategy each closed hour in order, its candles a whole hour apart and
    never one still forming; the buy sent at the third fills at the next tick, and the book after the
    tenth's close holds nothing — one deal in, one out.

    Mutation (2026-11-01): the candle still forming handed instead of the one that closed
    (``rows[-1]``) — red, a handed candle is not the closed candle the chart holds."""
    local, bench = tape
    s = _BuyThenClose()
    out = run(s, bench, interval="H1", local=local, max_bars=12)
    assert out["bars"] == 12, out
    times = [b.time_ms for b in s.bars]
    assert all(t % 3_600_000 == 0 for t in times), times
    assert times == sorted(times) and len(set(times)) == len(times), times
    assert "accepted" in s.answers[0]["last_answer"], s.answers
    assert out["book"]["positions"] == [], out["book"]
    # The buy filled and was closed: one deal in, one out, the in after the third candle's close.
    home, owner, _, name = bench.removeprefix("bentrix://").split("/")[:4]
    deals = local.read("trade_deals", home=home, owner=owner, bench=name)["items"]
    assert len(deals) == 2, deals
    # Each candle handed is the closed candle the chart holds, never one still forming.
    final = {r[0]: r[4] for r in local.call("bars_get", bench=bench, symbol=deals[0]["symbol"], timeframe="H1", limit=5000)["bars"]}
    assert all(final.get(b.time_ms) == b.close for b in s.bars), [(b.time_ms, b.close, final.get(b.time_ms)) for b in s.bars]


EMA_CROSS = {
    "schema": 1, "name": "ema sdk", "version": 1, "made_ms": 0, "made_by": "person", "markets": ["XAUUSD"], "interval": "H1",
    "studies": {
        "fast": {"kind": "ema", "convention": "talib", "applied": "close", "settings": {"shape": "period", "period": 8}},
        "slow": {"kind": "ema", "convention": "talib", "applied": "close", "settings": {"shape": "period", "period": 21}},
    },
    "rules": {
        "buy": {"compare": {"cross_up": ["fast", "slow"]}}, "sell": {"compare": {"cross_down": ["fast", "slow"]}},
        "close_buys": {"compare": {"cross_down": ["fast", "slow"]}}, "close_sells": {"compare": {"cross_up": ["fast", "slow"]}},
    },
    "entry": {"kind": "market", "offset": None, "from": None, "limit_offset": None, "expires_candles": None, "cancel_on_opposite": False},
    "exits": {"stop": {"points": 3000.0}, "target": {"points": 6000.0}, "first_target": None, "breakeven": None, "trail": None},
    "sizing": {"mode": "lots", "value": 0.1},
    "sessions": {"window": None, "desks": [], "flat_at": None},
    "limits": {"one_position_per_market": True, "max_open": None, "per_day": None, "max_bars": None},
    "notes": "",
}


def test_a_json_strategy_runs_through_the_sdk_and_its_account_adds_up(tape):
    """**A strategy document run through the SDK** — the same door an agent's MCP client takes: saved
    (version 1), its bench armed, run at machine speed from the bench's cursor and read until it ends;
    then the bench's account adds up (the opening balance and every deal's profit, swap and
    commission), every deal carries the run's magic, every entry fills within the minute after an
    hour's close, and the report keeps one closed trade per exit.

    Mutation (2026-11-04), red: a market order sent without the run's magic."""
    import time

    local, bench = tape
    local.arm(bench)
    local.advance(bench, 1_709_640_000_000)  # 2024-03-05 12:00 UTC: a day and a half of history behind it
    home, owner, _, name = bench.removeprefix("bentrix://").split("/")[:4]
    opening = local.read("trade_account", home=home, owner=owner, bench=name)["balance"]
    saved = local.call("strategy_save", document=EMA_CROSS)
    assert saved["version"] == 1, saved
    local.call("bench_set_automation", bench=bench, armed=True)
    started = local.call("strategy_run", strategy="ema sdk", version=1, bench=bench)
    deadline = time.monotonic() + 120
    while True:
        got = local.call("run_get", run=started["uri"])
        if isinstance(got["state"], dict) and "ended" in got["state"]:
            break
        assert time.monotonic() < deadline, got
        local.settle(20)
        time.sleep(0.01)
    manifest = got["manifest"]
    assert manifest and manifest.get("report"), got
    magic = manifest["magic"]
    deals = local.read("trade_deals", home=home, owner=owner, bench=name)["items"]
    assert len(deals) >= 2, deals
    assert all(d.get("magic") == magic for d in deals), [d.get("magic") for d in deals]
    moved = sum(d["profit"] + d["swap"] + d["commission"] for d in deals)
    balance = local.read("trade_account", home=home, owner=owner, bench=name)["balance"]
    assert abs(balance - (opening + moved)) < 0.011, (balance, opening, moved)
    ins = [d for d in deals if d["entry"] == "in"]
    assert ins and all(d["time_ms"] % 3_600_000 < 60_000 for d in ins), [d["time_ms"] % 3_600_000 for d in ins]
    exits = [d for d in deals if d["entry"] != "in"]
    assert len(manifest["deals"]) == len(exits), (len(manifest["deals"]), len(exits))


def test_a_strict_gate_reads_each_days_kind_and_keeps_the_runs_it_can_stand_on(tape):
    """``bentrix.data.quality`` answers the store's own report of each day — the CI tape writes candles
    and no tick file, so every traded day is modelled — and ``segments`` keeps no run unless modelled
    days are allowed, then one from Monday to Friday; ``bars(strict=True)`` refuses the candles by their
    days and takes them when modelled days are allowed.

    Mutations (2026-11-04), each red: modelled days never refused; strict never refusing;
    allow_modelled ignored."""
    np = pytest.importorskip("numpy")
    from bentrix import data

    local, bench = tape
    local.arm(bench)
    local.advance(bench, 1_709_640_000_000)  # 2024-03-05 12:00 UTC
    report = data.quality(local, bench, "XAUUSD")
    days = ["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"]
    assert report["traded"] == days, report
    assert report["modelled"] == days and report["holes"] == [], report
    assert report["per_year"] == [{"year": 2024, "ticks": 0, "traded_days": 5, "tick_days": 0, "modelled_days": 5, "hole_days": 0}], report
    assert data.segments(report) == []
    assert data.segments(report, allow_modelled=True) == [("2024-03-04", "2024-03-08")]
    plain = data.bars(local, bench, "XAUUSD", "H1")
    assert len(plain["time_ms"]) > 0
    with pytest.raises(ValueError, match="2024-03-04"):
        data.bars(local, bench, "XAUUSD", "H1", strict=True)
    kept = data.bars(local, bench, "XAUUSD", "H1", strict=True, allow_modelled=True)
    assert np.array_equal(kept["time_ms"], plain["time_ms"])


class _TwoMarkets(Strategy):
    """Gold and silver on one account: at the third close a buy of 0.02 gold (the context's own
    market), a sell of silver and a silver buy limit far below; at the sixth half the gold closed; at
    the eighth silver's positions closed and its pending order cancelled."""

    def __init__(self):
        self.closes = []
        self.seen = {}

    def on_bars(self, bars, ctx):
        self.closes.append(bars)
        n = len(self.closes)
        if n == 3:
            ctx.buy(0.02)
            ctx.sell(0.01, symbol="XAGUSD")
            ctx.pending("buy_limit", 20.0, 0.01, symbol="XAGUSD")
        elif n == 6:
            self.seen["before"] = {p["symbol"]: p["volume"] for p in ctx.positions()}
            self.seen["orders"] = [o["symbol"] for o in ctx.orders()]
            ctx.close(ctx.positions("XAUUSD")[0]["ticket"], volume=0.01)
        elif n == 8:
            ctx.close_all(symbol="XAGUSD")
            for o in ctx.orders("XAGUSD"):
                ctx.close(o["ticket"])


def test_a_strategy_trades_two_markets_on_one_clock(tape2):
    """``run(symbols=[...])`` walks gold and silver on one clock: each close hands both markets' candle
    of the same hour, silver's its own (the formula at 0.0125 of gold's level), never gold's twice; an
    order names its market or takes the context's first; a pending order, a partial close and a close
    of one market's positions land on that market alone; and a market the workbench does not hold is
    refused before the walk.

    Mutations (2026-11-03), each red: one market walked; a sell, a pending order, a
    partial close or a market's close_all ignoring its market or volume; an unheld market walked; the
    tape's second market written at the first's level."""
    local, bench = tape2
    if "XAGUSD" not in local.call("replay_step", bench=bench, unit="tick", n=1).get("markets", []):
        local.call("bench_add_market", bench=bench, symbol="XAGUSD")
    with pytest.raises(ValueError):
        run(Strategy(), bench, interval="H1", local=local, max_bars=1, symbols=["XAUUSD", "EURUSD"])
    s = _TwoMarkets()
    out = run(s, bench, interval="H1", local=local, max_bars=10, symbols=["XAUUSD", "XAGUSD"])
    assert out["bars"] == 10, out
    for bars in s.closes:
        assert set(bars) == {"XAUUSD", "XAGUSD"}, bars
        gold, silver = bars["XAUUSD"], bars["XAGUSD"]
        assert gold.time_ms == silver.time_ms, (gold, silver)
        assert abs(silver.close - gold.close * 0.0125) < 0.002, (gold.close, silver.close)
    times = [b["XAUUSD"].time_ms for b in s.closes]
    assert times == sorted(times) and len(set(times)) == len(times), times
    assert s.seen["before"] == {"XAUUSD": 0.02, "XAGUSD": 0.01}, s.seen
    assert s.seen["orders"] == ["XAGUSD"], s.seen
    book = out["book"]
    assert [(p["symbol"], p["volume"]) for p in book["positions"]] == [("XAUUSD", 0.01)], book
    assert book["orders"] == [], book
    home, owner, _, name = bench.removeprefix("bentrix://").split("/")[:4]
    deals = local.read("trade_deals", home=home, owner=owner, bench=name)["items"]
    assert sorted(d["symbol"] for d in deals) == ["XAGUSD", "XAGUSD", "XAUUSD", "XAUUSD"], deals


class _Weeks(Strategy):
    """Notes each closed week and where the workbench's clock stands when it is handed."""

    def __init__(self):
        self.seen = []

    def on_bar(self, bar, ctx):
        self.seen.append((bar.time_ms, ctx._local.state(ctx.bench)["cursor_ms"]))


def test_a_run_reads_where_the_bench_stands_and_closes_each_candle_where_its_chart_does(tape3w):
    """**A run of no candles leaves the clock where it stood**, and **a week is handed at its own close**:
    each W1 candle opens on a Monday and reaches the strategy with the clock at the next Monday — the
    chart's cut, asked of the engine — two whole weeks of the three-week tape.

    The strict audit of 2026-11-04: S1, the run began with a step of one tick to read the state, so even
    ``max_bars=0`` moved the clock; S2, closes were ``t // length`` on the epoch, so a W1 closed on
    Thursdays (1970-01-01 was one) and its candle was handed days after its close — as a zone's D1
    was handed at UTC midnight.

    Mutations (2026-11-04), each red: the state read as a step of one tick again (the
    clock moved on a run of none); the closes epoch-aligned again (each week handed on a Thursday,
    three days after its close)."""
    local, bench = tape3w
    before = local.state(bench)["cursor_ms"]
    out = run(Strategy(), bench, interval="W1", local=local, max_bars=0)
    assert out["bars"] == 0 and local.state(bench)["cursor_ms"] == before, (before, local.state(bench)["cursor_ms"])
    s = _Weeks()
    out = run(s, bench, interval="W1", local=local)
    day, week = 86_400_000, 7 * 86_400_000
    assert out["bars"] == 2 and len(s.seen) == 2, (out, s.seen)
    for opened, at in s.seen:
        assert (opened // day + 4) % 7 == 1, f"a week opened on weekday {(opened // day + 4) % 7}, not a Monday"
        assert at == opened + week, f"the week of {opened} was handed at {at}, not at its close {opened + week}"


def test_a_closed_session_is_written_let_go_and_refuses_calls(tape, tmp_path):
    """**``close()`` writes the workspace and lets it go; ``with`` closes it** (the strict audit of
    2026-11-04, section 2.3): a closed session refuses a call, and the workspace it held opens again at
    once in another - the hold went with it - and that one, used under ``with``, is let go at its end.

    Mutation (2026-11-04), red: ``close`` doing nothing (the workspace, still held, was
    refused to the second opening)."""
    local, bench = tape
    ws, store = str(tmp_path / "tape" / "ws"), str(tmp_path / "tape" / "store")
    local.close()
    with pytest.raises(RuntimeError, match="closed"):
        local.call("get_state")
    local.close()  # a second close does nothing
    with Local(workspace=ws, store=store) as again:
        assert bench in str(again.call("get_state")), "the workspace reopened without its workbench"
    third = Local(workspace=ws, store=store)
    third.close()


def test_run_until_hands_the_closes_in_order_as_the_chart_closes_them(tape):
    """**``run_until`` hands ``n`` closes in one call** (the strict audit of 2026-11-04, section 2.4):
    five hourly closes in order, each the candle that closed there — opened an hour before the clock
    stood after it, and the very candle ``bars_get`` holds once closed — and the workbench's clock at
    the last of them.

    Mutations (2026-11-05), each red: the candle read without its open matched (the
    forming one handed); the weekend's ask-again removed (the three-week W1 run above ends at the first
    weekend)."""
    local, bench = tape
    local.arm(bench)
    symbol = local.state(bench)["markets"][0]
    closes = local.run_until(bench, "H1", 5, [symbol])
    assert len(closes) == 5, closes
    for reached, opened, rows in closes:
        assert reached - opened == 3_600_000, (reached, opened)
        assert rows and rows[0][0] == symbol and int(rows[0][1][0]) == opened, rows
    assert [c[1] for c in closes] == sorted(c[1] for c in closes)
    held = {r[0]: r for r in local.call("bars_get", bench=bench, symbol=symbol, timeframe="H1", limit=50)["bars"]}
    for _, opened, rows in closes:
        assert held[opened][1:5] == rows[0][1][1:5], (opened, held[opened], rows[0][1])
    assert local.state(bench)["cursor_ms"] == closes[-1][0]


def test_a_snapshot_brings_a_workbench_back_and_a_sweep_branches_from_it(tape):
    """**``snapshot``/``restore`` branch a sweep from one state** (the strict audit of 2026-11-04,
    section 2.4): a position bought, then a snapshot; one branch closes it and runs on, another runs on
    holding it; restored, the workbench stands at the snapshot's moment with the position open again
    and no deal of either branch, and runs to the same end as the first branch did.

    Mutations (2026-11-05), each red: the record not rebuilt on restore (the clock
    only moved) - *a branch's deal survived the restore*; the snapshot's moment not used (rebuilt at
    the clock's present) - *the workbench did not stand at the snapshot's moment*."""
    local, bench = tape
    local.arm(bench)
    symbol = local.state(bench)["markets"][0]
    local.run_until(bench, "H1", 3, [symbol])
    local.call("order_market", bench=bench, symbol=symbol, side="buy", volume=0.01)
    local.run_until(bench, "M1", 1, [symbol])
    held = local.book(bench)["positions"]
    assert len(held) == 1, held
    deals = lambda: local.read("trade_deals", **dict(zip(("home", "owner", "bench"), [bench.removeprefix("bentrix://").split("/")[i] for i in (0, 1, 3)])))["items"]
    at, before = local.state(bench)["cursor_ms"], len(deals())
    snap = local.snapshot(bench)
    # Branch one: close it and run on two hours.
    local.call("order_close", bench=bench, ticket=held[0]["ticket"])
    local.run_until(bench, "H1", 2, [symbol])
    first_end = local.state(bench)["cursor_ms"]
    assert local.book(bench)["positions"] == [] and len(deals()) > before
    # Back to the snapshot: the position open again, no deal of the branch.
    stood = local.restore(bench, snap)
    assert stood == at and local.state(bench)["cursor_ms"] == at, "the workbench did not stand at the snapshot's moment"
    assert [p["ticket"] for p in local.book(bench)["positions"]] == [held[0]["ticket"]], local.book(bench)
    assert len(deals()) == before, "a branch's deal survived the restore"
    # Branch two: hold it, run the same two hours.
    local.run_until(bench, "H1", 2, [symbol])
    assert local.state(bench)["cursor_ms"] == first_end
    assert len(local.book(bench)["positions"]) == 1
