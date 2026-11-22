"""**Datasets from a workbench**: its bars as columns (NumPy, or an Arrow table where
``pyarrow`` is installed), triple-barrier labels on bid and ask with a stop's and a target's own
semantics, and walk-forward and purged splits.

**A strict gate of your own** (``quality``, ``segments``, ``bars(strict=True)``): the store's own report
of each day — held as ticks, modelled from candles, or a tick file with hours missing — so a study can
refuse what it will not stand on and keep the runs of days it will. The thresholds are yours; the report
gives each year's counts to set them from.

**Nothing past the cursor.** Bars are read through the door a chart reads (``bars_get``), which never
answers past the workbench's cursor — so a dataset is what the workbench has seen; advance it first
(``Local.advance``) to take more. The candle still forming at the cursor is left out.

**Bid and ask.** The recorded bars are the bid's; the ask is the bid plus the bar's own spread
(``spread_points`` × the market's point). A long enters at the next open's ask and leaves at a bid; a
short enters at the next open's bid and leaves at an ask.
"""

from __future__ import annotations

import time
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from bentrix.local import Local

COLUMNS = ("time_ms", "open", "high", "low", "close", "tick_volume", "spread_points")


def quality(local: Local, bench: str, symbol: str, timeout_s: float = 600.0) -> dict:
    """**The store's report of ``symbol`` over ``bench``'s days** (``data_quality``), waited for: its
    figures, and ``traded`` (every day the store holds candles for), ``modelled`` (traded days with no
    tick file), ``holes`` (``{"day", "hours"}``, a tick file's UTC hours graded missing) and ``per_year``
    (each year's ticks, traded days, tick days, modelled days and days with a hole)."""
    deadline = time.monotonic() + timeout_s
    while True:
        got = local.call("data_quality", bench=bench, symbol=symbol)
        if not got.get("measuring"):
            return got
        if time.monotonic() > deadline:
            raise TimeoutError(f"the data-quality measure of {symbol} did not land in {timeout_s} s")
        time.sleep(0.05)


def _refused_days(report: dict, allow_modelled: bool, allow_holes: bool) -> set:
    bad = set()
    if not allow_modelled:
        bad |= set(report.get("modelled", []))
    if not allow_holes:
        bad |= {h["day"] for h in report.get("holes", [])}
    return bad


def segments(report: dict, allow_modelled: bool = False, allow_holes: bool = False) -> List[Tuple[str, str]]:
    """**The runs of days a study can stand on** — ``(first, last)``, ISO dates, from a ``quality``
    report: consecutive traded days (in the market's own trading order, so a weekend does not break a
    run) none of which is modelled or holed, unless that kind is allowed. Oldest first."""
    bad = _refused_days(report, allow_modelled, allow_holes)
    runs: List[Tuple[str, str]] = []
    start: Optional[str] = None
    prev: Optional[str] = None
    for d in report.get("traded", []):
        if d in bad:
            if start is not None:
                runs.append((start, prev))
                start = None
            continue
        if start is None:
            start = d
        prev = d
    if start is not None:
        runs.append((start, prev))
    return runs


def bars(local: Local, bench: str, symbol: str, interval: str = "H1", strict: bool = False, allow_modelled: bool = False, allow_holes: bool = False) -> Dict[str, np.ndarray]:
    """**Every closed candle of ``symbol`` on ``bench`` up to its cursor**, oldest first, as columns.

    ``strict``: refused (``ValueError``, naming the days) when a candle stands on a day the store holds
    modelled from candles or with a tick file's hours missing — unless that kind is allowed. A day is
    the candle's open time's UTC date, as the store files its days."""
    pages: List[List[list]] = []
    before: Optional[int] = None
    last_forming = None
    while True:
        page = local.call("bars_get", bench=bench, symbol=symbol, timeframe=interval, limit=5000, before_ms=before)
        got = page.get("bars", [])
        if last_forming is None:
            last_forming = bool(page.get("last_forming"))
        # Kept newest page first and joined once at the end: prepending page by page copied the whole
        # history each time (the strict audit of 2026-11-04, section 2.2).
        pages.append(got)
        before = page.get("earlier_before_ms")
        if not before or not got:
            break
    rows: List[list] = [r for p in reversed(pages) for r in p]
    if last_forming and rows:
        rows = rows[:-1]
    arr = np.array(rows, dtype=np.float64).reshape(-1, len(COLUMNS))
    out = {c: arr[:, k] for k, c in enumerate(COLUMNS)}
    out["time_ms"] = out["time_ms"].astype(np.int64)
    if strict:
        bad = _refused_days(quality(local, bench, symbol), allow_modelled, allow_holes)
        days = np.array([time.strftime("%Y-%m-%d", time.gmtime(t // 1000)) for t in out["time_ms"]])
        hit = sorted(set(days[np.isin(days, list(bad))])) if bad and len(days) else []
        if hit:
            raise ValueError(f"{len(hit)} of the days these candles stand on are not held as whole ticks ({', '.join(hit[:5])}{', ...' if len(hit) > 5 else ''}); see bentrix.data.segments")
    return out


def to_arrow(columns: Dict[str, np.ndarray]):
    """The columns as an Arrow table (``pip install pyarrow``)."""
    import pyarrow as pa

    return pa.table({k: pa.array(v) for k, v in columns.items()})


def triple_barrier(
    columns: Dict[str, np.ndarray],
    point: float,
    target_points: float,
    stop_points: float,
    horizon: int,
    side: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """**Triple-barrier labels with a stop's and a target's own semantics** (López de Prado): for each
    candle, a position entered at the NEXT candle's open (a long at its ask, a short at its bid), a target
    ``target_points`` and a stop ``stop_points`` away, and ``horizon`` candles to live. ``+1`` the target is
    met first, ``-1`` the stop, ``0`` neither by the horizon (or no next candle). A long's barriers are met
    by the bid (its exit side), a short's by the ask. **A candle that touches both is the stop's** — inside
    one OHLC bar the order is unknown, and the conservative reading is the one that does not flatter —
    and it is flagged in the second array.

    ``horizon`` counts the entry candle itself: the position lives from that candle's open, so its own
    high and low may meet a barrier, and ``horizon=1`` looks at that candle alone.
    """
    o, h, l = columns["open"], columns["high"], columns["low"]
    spread = columns["spread_points"] * point
    n = len(o)
    labels = np.zeros(n, dtype=np.int8)
    ambiguous = np.zeros(n, dtype=bool)
    for i in range(n - 1):
        j0 = i + 1
        if side > 0:
            entry = o[j0] + spread[j0]
            up, down = entry + target_points * point, entry - stop_points * point
        else:
            entry = o[j0]
            up, down = entry + stop_points * point, entry - target_points * point
        for j in range(j0, min(n, j0 + horizon)):
            if side > 0:
                hi, lo = h[j], l[j]  # the bid's: a long leaves at the bid
            else:
                hi, lo = h[j] + spread[j], l[j] + spread[j]  # the ask's: a short leaves at the ask
            hit_up, hit_down = hi >= up, lo <= down
            if hit_up and hit_down:
                labels[i] = -1
                ambiguous[i] = True
                break
            if side > 0 and hit_up or side < 0 and hit_down:
                labels[i] = 1
                break
            if side > 0 and hit_down or side < 0 and hit_up:
                labels[i] = -1
                break
    return labels, ambiguous


def walk_forward(n: int, train: int, test: int, anchored: bool = False) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """**Walk-forward splits** over ``n`` rows: ``train`` rows then the next ``test``, rolling (or
    anchored at the start), each test after its train and never overlapping the next test."""
    start = 0
    while start + train + test <= n:
        lo = 0 if anchored else start
        yield np.arange(lo, start + train), np.arange(start + train, start + train + test)
        start += test


def purged_kfold(n: int, k: int, horizon: int, embargo: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """**Purged k-fold** (López de Prado): ``k`` contiguous test folds; from each fold's train set the rows
    whose label's horizon reaches into the test fold are purged (``horizon`` rows before it), and an
    ``embargo`` of rows after it is dropped too."""
    edges = np.linspace(0, n, k + 1, dtype=int)
    for a, b in zip(edges[:-1], edges[1:]):
        test = np.arange(a, b)
        keep = np.ones(n, dtype=bool)
        keep[max(0, a - horizon) : min(n, b + embargo)] = False
        yield np.flatnonzero(keep), test


def indicator(local: Local, bench: str, symbol: str, interval: str, study: str, inputs: Optional[Dict[str, object]] = None, limit: int = 5000) -> Dict[str, np.ndarray]:
    """**An indicator's values over the workbench's candles up to its cursor**, with no chart open
    (``study_values``): ``study`` is ``"custom:<id>"`` for an Assay indicator of your indicators folder,
    ``inputs`` its inputs by name (the rest its defaults). Answers ``time_ms`` and one column per line,
    by the line's declared name, the newest ``limit`` values; NaN where a line has no value yet."""
    got = local.call("study_values", bench=bench, symbol=symbol, timeframe=interval, study=study, inputs=inputs, limit=limit)
    out = {"time_ms": np.asarray(got["times"], dtype=np.int64)}
    for line in got["lines"]:
        out[line["line"]] = np.array([np.nan if v is None else v for v in line["values"]], dtype=np.float64)
    return out
