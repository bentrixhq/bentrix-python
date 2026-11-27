"""**Vectorised backtests, reconciled with the engine**.

``from_signals`` turns a signal per closed candle (``+1`` long, ``-1`` short, ``0`` flat) into trades in
one pass over the columns ``bentrix.data.bars`` reads: a change of signal at candle ``i`` acts at candle
``i + 1``'s open — a long enters at the ask (the bid + the bar's own spread) and leaves at the bid, a
short the reverse — and a trade's gross is its price move × lots × contract size. Fast enough to sweep
thousands of signal sets.

**It is not the engine**, and ``reconcile`` says by how much: the same signals run event-driven through
the workbench's venue (``bentrix.backtest.run``) and the two trade lists set side by side — each trade's
bid-side price (a long's exit, a short's entry) within ``PRICE_TOLERANCE_POINTS`` of the venue's fill and
the same count of trades. **The ask side is measured and reported, not toleranced**: the venue quotes
its own spread (on the CI tape the Bentrix standard's 20 points where the bars
record 10), so a long's entry and a short's exit part by the difference, and the grosses with them.
Beside them, what the vectorised pass leaves out: the venue's commission and swap, read off its deals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

#: **The published tolerance on the bid side, in points — measured, on the CI tape.** The event-driven run
#: decides at the close and its order fills at the tick AFTER the one the decision saw (the venue's market
#: at the next tick): the next candle's second tick, where the vectorised pass takes its open. On the CI
#: tape's one-minute OHLC ticks that move is at most 5 points (measured 2026-11-01, ten trades); on
#: another recording the gap is the reconciliation's to report, and this figure is the CI tape's.
PRICE_TOLERANCE_POINTS = 5.0


@dataclass(frozen=True)
class Trade:
    side: int
    entry_ms: int
    entry: float
    exit_ms: int
    exit: float
    gross: float


def from_signals(columns: Dict[str, np.ndarray], signals: np.ndarray, point: float, contract_size: float, lots: float) -> List[Trade]:
    """**The trades ``signals`` make on ``columns``** (module doc). A position still open at the last
    candle is left open and not counted."""
    t, o = columns["time_ms"], columns["open"]
    spread = columns["spread_points"] * point
    signals = np.asarray(signals, dtype=np.int8)
    trades: List[Trade] = []
    held, entry, entry_ms = 0, 0.0, 0
    for i in range(len(o) - 1):
        want = int(signals[i])
        if want == held:
            continue
        j = i + 1
        bid, ask = o[j], o[j] + spread[j]
        if held != 0:
            exit_price = bid if held > 0 else ask
            gross = (exit_price - entry) * held * lots * contract_size
            trades.append(Trade(held, entry_ms, entry, int(t[j]), float(exit_price), float(gross)))
        if want != 0:
            entry = ask if want > 0 else bid
            entry_ms = int(t[j])
        held = want
    return trades


@dataclass(frozen=True)
class Reconciliation:
    trades: int
    engine_trades: int
    #: The largest gap, in points, between a vectorised fill and the venue's on the BID side (a long's exit,
    #: a short's entry): the open against the first tick after the close.
    bid_worst_points: float
    #: The largest and the mean gap on the ASK side — the venue's own spread (its cost provenance)
    #: against the bar's recorded one the vectorised pass used: measured, never assumed away.
    ask_worst_points: float
    ask_mean_points: float
    vector_gross: float
    #: The engine's realised profit, before its commission and swap.
    engine_gross: float
    #: The commission and swap the venue booked, which the vectorised pass leaves out.
    engine_costs: float
    within: bool


def reconcile(vector: List[Trade], deals: List[dict], point: float) -> Reconciliation:
    """**The vectorised trades set beside the engine's deals** (module doc): each engine position's
    entry and exit deal against the vectorised trade in the same order, the bid side and the ask side
    apart. **Within the tolerance** when the counts are equal and every bid-side price is within
    ``PRICE_TOLERANCE_POINTS``; the ask side's gap is the spread the two read, reported."""
    by_pos: Dict[int, List[dict]] = {}
    for d in deals:
        by_pos.setdefault(d["position"], []).append(d)
    engine = []
    for ds in by_pos.values():
        ins = [d for d in ds if d["entry"] == "in"]
        outs = [d for d in ds if d["entry"] == "out"]
        if ins and outs:
            engine.append((ins[0], outs[-1], sum(d["profit"] for d in ds), sum(d["commission"] + d["swap"] for d in ds)))
    engine.sort(key=lambda e: e[0]["time_ms"])
    bid_gaps, ask_gaps = [0.0], []
    for v, (i, o, _, _) in zip(vector, engine):
        entry_gap, exit_gap = abs(v.entry - i["price"]) / point, abs(v.exit - o["price"]) / point
        if v.side > 0:  # a long enters at the ask and leaves at the bid
            ask_gaps.append(entry_gap)
            bid_gaps.append(exit_gap)
        else:  # a short enters at the bid and leaves at the ask
            bid_gaps.append(entry_gap)
            ask_gaps.append(exit_gap)
    vg = sum(v.gross for v in vector[: len(engine)])
    profit = sum(e[2] for e in engine)
    costs = sum(e[3] for e in engine)
    bid_worst = float(max(bid_gaps))
    within = len(vector) == len(engine) and bid_worst <= PRICE_TOLERANCE_POINTS + 1e-9
    return Reconciliation(
        len(vector), len(engine), bid_worst, float(max(ask_gaps, default=0.0)), float(np.mean(ask_gaps)) if ask_gaps else 0.0, vg, profit, costs, bool(within)
    )
