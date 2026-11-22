"""**Datasets from a workbench**: the bars as columns up to the cursor and never past it;
triple-barrier labels on bid and ask; walk-forward and purged splits."""

import pytest

np = pytest.importorskip("numpy")

from bentrix.data import purged_kfold, triple_barrier, walk_forward  # noqa: E402


def _cols(o, h, l, spread=0.0):
    n = len(o)
    return {"open": np.array(o, float), "high": np.array(h, float), "low": np.array(l, float), "spread_points": np.full(n, spread)}


def test_a_long_is_labelled_by_the_bid_after_entering_at_the_next_ask():
    """A long enters at the next open's ask: a target met by the bid is ``+1``, a stop ``-1``, a candle
    touching both is the stop's and flagged, a horizon passed is ``0``.

    Mutation (2026-11-01): the entry taken at the next open's bid (the spread dropped) — red,
    the first candle's target is met where it should not be."""
    point = 0.01
    # Candle 0's entry is candle 1's open + spread (100.00 + 0.10 = 100.10); target +50 points, stop -50.
    c = _cols(o=[100.0, 100.0, 100.0, 100.0, 100.0], h=[100.0, 100.55, 100.62, 100.70, 100.0], l=[100.0, 99.70, 99.80, 99.50, 100.0], spread=10)
    labels, ambiguous = triple_barrier(c, point, target_points=50, stop_points=50, horizon=1)
    # Candle 1: high 100.55 < 100.60, low 99.70 > 99.60 → neither within one candle.
    assert labels[0] == 0
    # Candle 2's entry 100.10: high 100.62 ≥ 100.60 → the target.
    assert labels[1] == 1, labels
    # Candle 3: both 100.70 ≥ 100.60 and 99.50 ≤ 99.60 → the stop, flagged.
    assert labels[2] == -1 and ambiguous[2]
    assert labels[4] == 0, "the last candle has no next candle to enter at"


def test_a_short_is_labelled_by_the_ask():
    """A short enters at the next open's bid and its barriers are met by the ask (bid + spread)."""
    point = 0.01
    c = _cols(o=[100.0, 100.0, 100.0], h=[100.0, 100.0, 100.45], l=[100.0, 99.40, 99.0], spread=10)
    labels, _ = triple_barrier(c, point, target_points=50, stop_points=50, horizon=1, side=-1)
    # Candle 1's ask low is 99.50 ≤ 99.50: the short's target.
    assert labels[0] == 1, labels
    # Candle 2's ask high is 100.55 ≥ 100.50: its stop (both touched: the stop's).
    assert labels[1] == -1, labels


def test_splits_never_let_a_test_row_train():
    """Walk-forward tests follow their train and never overlap; purged folds drop the horizon before each
    test fold and the embargo after it.

    Mutation (2026-11-01): the purge's horizon not subtracted — red, a train row's label reaches the
    test fold."""
    for train, test in walk_forward(100, 30, 10):
        assert train.max() < test.min() and len(test) == 10
    anchored = list(walk_forward(100, 30, 10, anchored=True))
    assert all(t[0][0] == 0 for t in anchored) and len(anchored) == 7
    for train, test in purged_kfold(100, 5, horizon=3, embargo=2):
        t0, t1 = test.min(), test.max()
        assert not set(train) & set(range(t0 - 3, t1 + 3)), "a train row's label reaches the test fold"
        assert len(train) + len(test) + 3 + 2 >= 100 - 5


def test_the_bars_are_the_workbenchs_up_to_its_cursor(tape):
    """Read off the CI tape, the columns are every closed candle up to the cursor: oldest first, each
    closed before the cursor, the one still forming left out; an Arrow table where pyarrow is installed."""
    pytest.importorskip("bentrix._native")
    from bentrix.data import bars, to_arrow

    local, bench = tape
    local.arm(bench)
    start = local.call("replay_step", bench=bench, unit="tick", n=1)["cursor_ms"]
    cursor = local.advance(bench, start + 30 * 3_600_000)
    c = bars(local, bench, "XAUUSD", "H1")
    t = c["time_ms"]
    assert len(t) >= 20 and (np.diff(t) > 0).all()
    assert (t + 3_600_000 <= cursor).all(), "a candle still forming, or past the cursor, was read"
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return
    assert to_arrow(c).num_rows == len(t)


ASSAY_SMA = """//@version=5
indicator("Data SMA")
len = input.int(5, "Length")
plot(ta.sma(close, len), "SMA")
"""


def test_an_assay_indicator_is_read_through_the_sdk_with_no_chart(tape):
    """**An Assay indicator through the SDK**: a script of the indicators folder
    computed by BENTRIX over the workbench's candles to the cursor, no chart open — its default length
    and a length given — each equal to a moving average computed here over the same candles; an
    unknown id is refused by name.

    Mutation (2026-11-01): the inputs given not handed to the script — red, the length
    of 9 answers the length of 5."""
    import pathlib
    import time

    from bentrix.data import bars, indicator

    local, bench = tape
    local.arm(bench)
    start = local.call("replay_step", bench=bench, unit="tick", n=1)["cursor_ms"]
    local.advance(bench, start + 30 * 3_600_000)
    folder = pathlib.Path(local.call("resource_read", uri="bentrix://indicators")["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data_sma.assay").write_text(ASSAY_SMA, encoding="utf-8")
    local.call("indicator_rescan")
    for _ in range(200):
        local.settle(20)
        scripts = local.call("resource_read", uri="bentrix://indicators")["scripts"]
        row = next((s for s in scripts if s["file"] == "data_sma.assay"), None)
        if row and row["state"] == "ready":
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"the Assay script never described itself: {row}")
    cols = bars(local, bench, "XAUUSD", "H1")
    # An input goes by its spec name too, its title made an id: input.int(5, "Length") is "length".
    for n, given in ((5, None), (9, {"length": 9})):
        got = indicator(local, bench, "XAUUSD", "H1", "custom:data_sma", inputs=given)
        # A line goes by its spec's name, the plot's title made an id (`derive::slug`): "SMA" is "sma".
        assert list(got) == ["time_ms", "sma"], list(got)
        closed = {t: i for i, t in enumerate(cols["time_ms"].tolist())}
        rows = [(i, v) for t, v in zip(got["time_ms"].tolist(), got["sma"].tolist()) if t in closed for i in [closed[t]] if i >= n - 1]
        assert len(rows) >= 20, len(rows)
        for i, v in rows:
            want = float(np.mean(cols["close"][i - n + 1 : i + 1]))
            assert abs(v - want) <= 1e-9 * want, (n, i, v, want)
    with pytest.raises(RuntimeError, match="No Assay indicator"):
        indicator(local, bench, "XAUUSD", "H1", "custom:nothing_here")
    with pytest.raises(RuntimeError, match="it declares"):
        indicator(local, bench, "XAUUSD", "H1", "custom:data_sma", inputs={"period": 3})
