"""**Metrics of an equity series and its trades**, in your own process: the figures of the ``stats_metrics`` tool, by
the same definitions in the same order, so a number computed here is the number the tool answers
(``tests/test_metrics.py`` holds the two equal). A calculator, not a judgement: it reads and keeps nothing.

``equity`` is rows ``(time_ms, equity)``, ``(time_ms, equity, low, high)`` or ``bench_equity``'s
``(time_ms, equity, low, high, gross, net, gross_max)``, oldest first - the low and high the lowest and highest equity
inside the period, so a drawdown is read on its tick, and the book's face value for the exposure and the leverage.
``deals`` (``trade_deals``' items) gives a row a closed trade, whose results are the trades when none are given.
"""

from __future__ import annotations

import decimal
import math
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["metrics", "trade_rows"]


def _fig(value: Optional[float], definition: str) -> Dict[str, Any]:
    if value is not None and not math.isfinite(value):
        value = None
    return {"value": value, "definition": definition}


_MOST_DECIMALS = 8


def _decimals_of(x: float) -> int:
    """A figure's decimals as its shortest text writes them, at most eight - ``stats_metrics``' own count."""
    exponent = decimal.Decimal(repr(float(x))).normalize().as_tuple().exponent
    return min(max(0, -int(exponent)), _MOST_DECIMALS)


def _at_decimals(x: float, decimals: int) -> float:
    """``x`` at ``decimals``, a half away from zero as the tool rounds it."""
    scale = 10.0 ** decimals
    y = math.copysign(math.floor(abs(x) * scale + 0.5), x) / scale
    return y if math.isfinite(y) else x


def _quantile(srt: Sequence[float], q: float) -> Optional[float]:
    """The nearest-rank quantile of an ascending list: its ceil(n x q)-th value."""
    if not srt:
        return None
    return srt[min(max(int(math.ceil(len(srt) * q)), 1), len(srt)) - 1]


def trade_rows(deals: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """**A row a closed trade**, from ``trade_deals``' items - ``stats_metrics``' ``trade_rows``. A trade is a round
    from flat to flat: a position's deals walked in order with the volume they leave open, an entry from flat opening
    one, an exit bringing the volume to nothing closing it, a reversal (``in_out``, a netting account's) closing it with
    the volume that was open and opening the next with the rest - its commission shared by volume, its profit, swap and
    excursions the closing trade's. A position still open is no row. Each row: entry and exit times and volume-weighted
    prices, duration, volume, result (its deals' profit, commission and swap), costs, and the worst and best its
    closing deals carry; in order of exit."""
    by: Dict[int, List[Dict[str, Any]]] = {}
    order: List[int] = []
    for k, d in enumerate(deals):
        if d.get("position") is None:
            raise ValueError(f"Deal {k} names no position.")
        if d.get("entry") not in ("in", "out", "in_out", "out_by"):
            raise ValueError(f"Deal {k}'s entry is not in, out, in_out or out_by.")
        p = int(d["position"])
        if p not in by:
            by[p] = []
            order.append(p)
        by[p].append(d)
    num = lambda d, k: float(d.get(k) or 0.0)  # noqa: E731

    def opened(p: int, d: Dict[str, Any]) -> Dict[str, Any]:
        t = int(d.get("time_ms") or 0)
        return {"position": p, "symbol": d.get("symbol") or "", "side": d.get("side") or "", "entry_ms": t, "exit_ms": t,
                "in_v": 0.0, "in_pv": 0.0, "out_v": 0.0, "out_pv": 0.0, "profit": 0.0, "costs": 0.0, "worst": None, "best": None}

    def row(t: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "position": t["position"], "symbol": t["symbol"], "side": t["side"],
            "entry_ms": t["entry_ms"], "exit_ms": t["exit_ms"], "duration_ms": t["exit_ms"] - t["entry_ms"],
            "volume": t["in_v"], "entry_price": t["in_pv"] / t["in_v"] if t["in_v"] > 0 else 0.0,
            "exit_price": t["out_pv"] / t["out_v"] if t["out_v"] > 0 else 0.0,
            "result": t["profit"] + t["costs"], "costs": t["costs"], "worst": t["worst"], "best": t["best"],
        }

    out = []
    for p in order:
        open_v = 0.0
        cur: Optional[Dict[str, Any]] = None
        for d in by[p]:
            volume, price = num(d, "volume"), num(d, "price")
            if d["entry"] == "in":
                if cur is None:
                    cur = opened(p, d)
                cur["in_v"] += volume
                cur["in_pv"] += volume * price
                cur["costs"] += num(d, "commission") + num(d, "swap")
                open_v += volume
                continue
            closing = min(open_v, volume) if d["entry"] == "in_out" else volume
            share = closing / volume if volume > 0 else 1.0
            if cur is not None:
                cur["out_v"] += closing
                cur["out_pv"] += closing * price
                cur["exit_ms"] = max(cur["exit_ms"], int(d.get("time_ms") or 0))
                cur["profit"] += num(d, "profit")
                cur["costs"] += num(d, "commission") * share + num(d, "swap")
                if d.get("worst") is not None:
                    cur["worst"] = d["worst"] if cur["worst"] is None else min(cur["worst"], d["worst"])
                if d.get("best") is not None:
                    cur["best"] = d["best"] if cur["best"] is None else max(cur["best"], d["best"])
            open_v -= closing
            if open_v <= 1e-9:
                open_v = 0.0
                if cur is not None:
                    out.append(row(cur))
                    cur = None
            rest = volume - closing
            if d["entry"] == "in_out" and rest > 1e-9:
                cur = opened(p, d)
                cur["in_v"], cur["in_pv"] = rest, rest * price
                cur["costs"] = num(d, "commission") * (1.0 - share)
                open_v = rest
    out.sort(key=lambda t: (t["exit_ms"], t["position"]))
    return out


def metrics(equity: Sequence[Sequence[float]], trades: Sequence[float] = (), periods_per_year: Optional[float] = None,
            level: float = 0.95, deals: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """**The figures**, each ``{"value", "definition"}`` - ``stats_metrics``'s answer, computed here."""
    if len(equity) < 2:
        raise ValueError("An equity series is at least 2 rows.")
    if not 0.5 <= level <= 0.9999:
        raise ValueError("level is 0.5 to 0.9999.")
    rows = []
    risk = []
    width = len(equity[0])
    for k, r in enumerate(equity):
        if len(r) not in (2, 4, 7) or len(r) != width:
            raise ValueError(f"Row {k} is not (time_ms, equity), (time_ms, equity, low, high) or (time_ms, equity, low, high, gross, net, gross_max), as long as the first row.")
        t, e = float(r[0]), float(r[1])
        low, high = (min(float(r[2]), e), max(float(r[3]), e)) if len(r) >= 4 else (e, e)
        if e <= 0:
            raise ValueError(f"Row {k}'s equity is not above zero.")
        rows.append((t, e, low, high))
        if len(r) == 7:
            g, nt, top = float(r[4]), float(r[5]), max(float(r[6]), float(r[4]))
            if g < 0 or top < 0:
                raise ValueError(f"Row {k}'s gross face value is below zero.")
            risk.append((g, nt, top, top / e))
    tr = trade_rows(deals)
    if not trades:
        trades = [t["result"] for t in tr]
    if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("The rows' times do not rise.")
    ppy = periods_per_year
    r = [b[1] / a[1] - 1.0 for a, b in zip(rows, rows[1:])]
    n = float(len(r))
    mean = sum(r) / n
    var = sum((x - mean) ** 2 for x in r) / (n - 1.0) if len(r) > 1 else None
    sd = math.sqrt(var) if var is not None else None
    m2 = sum((x - mean) ** 2 for x in r) / n
    m3 = sum((x - mean) ** 3 for x in r) / n
    m4 = sum((x - mean) ** 4 for x in r) / n
    skew = m3 / m2 ** 1.5 if m2 > 0 else None
    kurt = m4 / (m2 * m2) - 3.0 if m2 > 0 else None
    down = math.sqrt(sum(min(x, 0.0) ** 2 for x in r) / n)
    sharpe = mean / sd * math.sqrt(ppy) if (sd is not None and sd > 0 and ppy is not None) else None
    sortino = mean / down * math.sqrt(ppy) if (ppy is not None and down > 0) else None
    peak, peak_t = max(rows[0][3], rows[0][1]), rows[0][0]
    dd, dd_peak, dd_peak_t, dd_trough_t = 0.0, peak, rows[0][0], rows[0][0]
    under = 0
    for t, e, low, high in rows:
        fall = peak - low
        if fall > dd:
            dd, dd_peak, dd_peak_t, dd_trough_t = fall, peak, peak_t, t
        if e < peak:
            under += 1
        if high > peak or e > peak:
            peak, peak_t = max(high, e), t
    # The drawdown is a difference of two of the series' figures, so it is one of them: at the most decimals they carry.
    dd = _at_decimals(dd, max(_decimals_of(x) for _, e, low, high in rows for x in (e, low, high)))
    dd_pct = dd / dd_peak if dd_peak > 0 else 0.0
    dd_peak_value = max(max(h, e) for t, e, _, h in rows if t <= dd_peak_t)
    recovered = None
    if dd > 0:
        recovered = next((t for t, e, _, _ in rows if t > dd_trough_t and e >= dd_peak_value), None)
    duration = ((recovered if recovered is not None else rows[-1][0]) - dd_peak_t) if dd > 0 else None
    years = n / ppy if ppy is not None else None
    growth = rows[-1][1] / rows[0][1]
    annual = growth ** (1.0 / years) - 1.0 if (years is not None and years > 0) else None
    calmar = annual / dd_pct if (annual is not None and dd_pct > 0) else None
    srt = sorted(r)
    k = min(max(int(math.ceil(n * (1.0 - level))), 1), len(srt))
    var_at = -srt[k - 1]
    es = -(sum(srt[:k]) / k)
    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x < 0]
    gains = sum(wins)
    lost = abs(sum(losses))
    gross = sorted(x[0] for x in risk)
    net = sorted(x[1] for x in risk)
    lev = sorted(x[3] for x in risk)
    lev_mean = sum(lev) / len(lev) if lev else None
    return {
        "periods": len(r),
        "mean": _fig(mean, "the mean of the period returns, each equity over the one before less one"),
        "deviation": _fig(sd, "their sample standard deviation (n - 1)"),
        "skew": _fig(skew, "their skewness, the third central moment over the second to the power 1.5"),
        "excess_kurtosis": _fig(kurt, "their kurtosis less 3, the fourth central moment over the second squared"),
        "sharpe": _fig(sharpe, "mean over deviation, times the square root of periods_per_year; no risk-free rate"),
        "sortino": _fig(sortino, "mean over the downside deviation (the root mean square of the returns below zero, over every period), times the square root of periods_per_year"),
        "annual_return": _fig(annual, "the growth of the series raised to one over its years, less one; years are periods over periods_per_year"),
        "calmar": _fig(calmar, "the annual return over the largest drawdown as a share of its peak"),
        "max_drawdown": _fig(dd, "the largest fall from a running peak (each period's high or equity) to a later period's low, in money"),
        "max_drawdown_share": _fig(dd_pct, "that fall as a share of its peak"),
        "drawdown_duration_ms": _fig(float(duration) if duration is not None else None, "from the peak of the largest drawdown to its recovery, or to the series' end where it never recovered"),
        "recovery_ms": _fig(recovered - dd_trough_t if recovered is not None else None, "from the largest drawdown's trough to the first period whose equity is back at its peak; null if it never was"),
        "time_under_water": _fig(under / len(rows), "the share of periods whose equity is below the running peak"),
        "value_at_risk": _fig(var_at, "the loss of the period return at the tail level's quantile (the ceil(n x (1 - level))-th worst), as a positive share"),
        "expected_shortfall": _fig(es, "the mean loss of the returns at and below that quantile, as a positive share"),
        "worst_period": _fig(srt[0] if srt else None, "the lowest period return"),
        "trades": len(trades),
        "win_rate": _fig(len(wins) / len(trades) if trades else None, "the share of trades with a result above zero"),
        "average_win": _fig(gains / len(wins) if wins else None, "the mean result of the winning trades"),
        "average_loss": _fig(-lost / len(losses) if losses else None, "the mean result of the losing trades"),
        "profit_factor": _fig(gains / lost if lost > 0 else None, "the sum of the wins over the sum of the losses' size"),
        "exposure_mean": _fig(sum(gross) / len(gross) if gross else None, "the mean of the book's gross face value at each period's end, long and short added, in the account's currency"),
        "exposure_median": _fig(_quantile(gross, 0.5), "its nearest-rank median"),
        "exposure_p95": _fig(_quantile(gross, 0.95), "its nearest-rank 95th percentile"),
        "exposure_max": _fig(gross[-1] if gross else None, "the largest gross face value at a period's end"),
        "net_exposure_mean": _fig(sum(net) / len(net) if net else None, "the mean of the book's net face value at each period's end, long less short"),
        "net_exposure_min": _fig(net[0] if net else None, "the lowest net face value at a period's end"),
        "net_exposure_max": _fig(net[-1] if net else None, "the highest net face value at a period's end"),
        "leverage_mean": _fig(lev_mean, "the mean of each period's largest gross face value over its equity"),
        "leverage_median": _fig(_quantile(lev, 0.5), "its nearest-rank median"),
        "leverage_p95": _fig(_quantile(lev, 0.95), "its nearest-rank 95th percentile"),
        "leverage_max": _fig(lev[-1] if lev else None, "the largest of each period's largest gross over its equity"),
        "time_under_risk": _fig(sum(1 for x in risk if x[2] > 0) / len(risk) if risk else None, "the share of periods in which the book held a position"),
        "return_per_leverage": _fig(mean / lev_mean if (lev_mean is not None and lev_mean > 0) else None, "the mean period return over the mean leverage: a period's return for each unit of leverage held"),
        "level": level,
        "trade_rows": tr,
    }
