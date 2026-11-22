"""**Vectorised backtests, reconciled with the engine**."""

import pytest

np = pytest.importorskip("numpy")

from bentrix.vector import PRICE_TOLERANCE_POINTS, from_signals, reconcile  # noqa: E402


def test_a_signal_acts_at_the_next_open_on_the_right_side():
    """A long entered at the next open's ask and left at a later open's bid; a reversal closes and opens at
    one open; a position still open at the end is not counted.

    Mutation (2026-11-01): a long left at the ask — red, its exit is not the bid; and the
    walk below red too."""
    cols = {
        "time_ms": np.arange(6, dtype=np.int64) * 3_600_000,
        "open": np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0]),
        "spread_points": np.full(6, 10.0),
    }
    trades = from_signals(cols, np.array([1, 1, -1, -1, 0, 0]), point=0.01, contract_size=100.0, lots=0.1)
    assert len(trades) == 2
    long, short = trades
    assert (long.side, long.entry, long.exit) == (1, 101.10, 103.0), long
    assert abs(long.gross - (103.0 - 101.10) * 0.1 * 100) < 1e-9
    # Signalled short at candle 2, it acts at candle 3's open; flat at candle 4, it leaves at candle 5's ask.
    assert (short.side, short.entry, short.exit) == (-1, 103.0, 105.10), short
    assert from_signals(cols, np.array([0, 1, 1, 1, 1, 1]), 0.01, 100.0, 0.1) == []


def test_the_vectorised_pass_reconciles_with_the_engine_on_the_ci_tape(tape):
    """**The same signals, vectorised and event-driven**: on the CI tape's first two days, a close above its
    five-candle average is long and below it short; the vectorised trades and the workbench's own deals
    are the same count and every bid-side price within the published tolerance; the ask side's gap is the
    venue's own spread against the bar's recorded one, measured and reported (on this tape, 20 points
    against 10).

    Mutation (2026-11-01): the vectorised entry taken at the candle that signalled rather
    than the next — red, the bid-side prices part by more than the tolerance."""
    pytest.importorskip("bentrix._native")
    from bentrix.backtest import Strategy, run
    from bentrix.data import bars

    local, bench = tape
    local.arm(bench)
    start = local.call("replay_step", bench=bench, unit="tick", n=1)["cursor_ms"]
    local.advance(bench, start + 48 * 3_600_000)
    cols = bars(local, bench, "XAUUSD", "H1")
    c = cols["close"]
    sma = np.convolve(c, np.ones(5) / 5, mode="full")[: len(c)]
    signals = np.where(np.arange(len(c)) >= 4, np.sign(c - sma), 0).astype(np.int8)
    by_time = dict(zip(cols["time_ms"].tolist(), signals.tolist()))
    local.call("replay_seek", bench=bench, to_ms=start, undo_deals=True)

    class Follow(Strategy):
        held = 0

        def on_bar(self, bar, ctx):
            want = by_time.get(bar.time_ms, 0)
            if want == self.held:
                return
            if self.held:
                ctx.close_all()
            if want > 0:
                ctx.buy(0.1)
            elif want < 0:
                ctx.sell(0.1)
            self.held = want

    run(Follow(), bench, interval="H1", local=local, max_bars=len(c))
    home, owner, _, name = bench.removeprefix("bentrix://").split("/")[:4]
    deals = local.read("trade_deals", home=home, owner=owner, bench=name)["items"]
    info = local.call("symbol_info", bench=bench, symbol="XAUUSD")
    vec = from_signals(cols, signals, info["point"], info["contract_size"], 0.1)
    rec = reconcile(vec, deals, info["point"])
    assert rec.trades >= 3, rec
    assert rec.within, rec
    assert rec.bid_worst_points <= PRICE_TOLERANCE_POINTS + 1e-9
    assert rec.ask_mean_points > 0, "the venue's spread was the bar's: the measurement this reports is gone"
