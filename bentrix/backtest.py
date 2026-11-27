"""**Bentrix's engine in your own Python process**.

``bentrix.backtest.run`` drives a ``Strategy`` over a laboratory workbench: at each closed candle of
the interval it calls ``on_bar`` with that candle and a ``Context`` that sends orders through the
workbench's own venue — bid/ask fills, the broker's stops level, margin, swap and commission, the
clock that cannot run ahead of the candle. Nothing is simulated twice: the trades are the
workbench's, and its trade list is the one the Run window and ``run_parity`` read.

**Several markets at once**: ``run(..., symbols=[...])`` walks every market named on the one clock and
hands ``on_bars`` each market's candle that closed, by symbol; an order names its market
(``ctx.buy(0.1, symbol="XAGUSD")``) or takes the context's first. One account, one venue: margin,
stop-out and the book are the workbench's across all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bentrix.local import Local


@dataclass(frozen=True)
class Bar:
    """One closed candle: its open time (ms, on the workbench's clock) and prices."""

    time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class Context:
    """What a strategy acts through on its workbench. ``symbol`` is the market an order takes when it
    names none; ``symbols`` every market the run walks (``[symbol]`` for one)."""

    def __init__(self, local: Local, bench: str, symbol: str, symbols: Optional[List[str]] = None):
        self._local, self.bench, self.symbol = local, bench, symbol
        self.symbols = list(symbols) if symbols else [symbol]

    def buy(self, volume: float, sl: Optional[float] = None, tp: Optional[float] = None, symbol: Optional[str] = None) -> Any:
        return self._local.call("order_market", bench=self.bench, symbol=symbol or self.symbol, side="buy", volume=volume, sl=sl, tp=tp)

    def sell(self, volume: float, sl: Optional[float] = None, tp: Optional[float] = None, symbol: Optional[str] = None) -> Any:
        return self._local.call("order_market", bench=self.bench, symbol=symbol or self.symbol, side="sell", volume=volume, sl=sl, tp=tp)

    def pending(self, kind: str, price: float, volume: float, sl: Optional[float] = None, tp: Optional[float] = None, stop_limit: Optional[float] = None, symbol: Optional[str] = None) -> Any:
        """A pending order: ``kind`` one of buy_limit, sell_limit, buy_stop, sell_stop, buy_stop_limit,
        sell_stop_limit (``stop_limit`` the limit price a stop-limit places when its stop is met)."""
        return self._local.call("order_pending", bench=self.bench, symbol=symbol or self.symbol, kind=kind, price=price, volume=volume, sl=sl, tp=tp, stop_limit=stop_limit)

    def close(self, ticket: int, volume: Optional[float] = None) -> Any:
        """Close position ``ticket``, whole or ``volume`` of it; a pending order's ticket cancels it."""
        return self._local.call("order_close", bench=self.bench, ticket=ticket, volume=volume)

    def modify(self, ticket: int, sl: float, tp: float, price: Optional[float] = None) -> Any:
        """A position's or a pending order's stops (0 for none), and a pending order's price."""
        return self._local.call("order_modify", bench=self.bench, ticket=ticket, sl=sl, tp=tp, price=price)

    def close_all(self, symbol: Optional[str] = None) -> Any:
        """Close every position, or every position on ``symbol``."""
        return self._local.call("order_close_all", bench=self.bench, symbol=symbol)

    def book(self) -> Any:
        """The workbench's open positions and pending orders, as its Toolbox's Trade tab holds them."""
        return self._local.book(self.bench)

    def positions(self, symbol: Optional[str] = None) -> List[dict]:
        """The open positions, every market's or ``symbol``'s."""
        return [p for p in self.book().get("positions", []) if symbol is None or p.get("symbol") == symbol]

    def orders(self, symbol: Optional[str] = None) -> List[dict]:
        """The pending orders, every market's or ``symbol``'s."""
        return [o for o in self.book().get("orders", []) if symbol is None or o.get("symbol") == symbol]


class Strategy:
    """Override the callbacks you need; each is called on the workbench's clock, never ahead of it."""

    def on_start(self, ctx: Context) -> None:
        pass

    def on_bar(self, bar: Bar, ctx: Context) -> None:
        pass

    def on_bars(self, bars: Dict[str, Bar], ctx: Context) -> None:
        """A run over several markets: each market's candle that closed at this close, by symbol (a
        market with no candle there, its session shut, is absent). By default ``on_bar`` with the
        context's own market's candle."""
        if ctx.symbol in bars:
            self.on_bar(bars[ctx.symbol], ctx)

    def on_end(self, ctx: Context) -> None:
        pass

    def on_order_event(self, event: dict, ctx: Context) -> None:
        """An order's outcome on a live workbench (``bentrix.live``): ``sent``, ``awaiting_person`` (a
        real account's proposal, waiting for the person's Send) or ``refused`` with its words."""


def run(strategy: Strategy, bench: Any, symbol: Optional[str] = None, interval: str = "H1", local: Optional[Local] = None, max_bars: Optional[int] = None, symbols: Optional[List[str]] = None, to: Any = None) -> dict:
    """**Run ``strategy`` on ``bench`` (``bentrix://lab/<lab>/bench/<bench>``) from where it stands**,
    to the end of its days (or ``max_bars``, or the moment ``to``): the workbench advanced a tick at a time to each close of
    ``interval`` as its chart cuts it — a D1 at the bench's zone's midnight, a W1 from Monday — every
    tick fed to its venue, so a stop or a target between two closes is met where
    the tape meets it — and ``on_bar`` handed the candle that closed. An order sent there fills at the
    next tick. Answers how many closes it handed and the workbench's book at the end.

    With ``symbols``, every one of those markets (each one the workbench holds) is walked on the one
    clock and ``on_bars`` is handed each one's closed candle by symbol; ``symbol`` (else the first of
    ``symbols``) is the market an order takes when it names none.

    **A ``bentrix.Bench`` is the strategy's context**: given one (``bentrix.open(...).bench()``), its hooks are handed
    that ``Bench`` - ``bench.buy(0.1, sl=...)``, ``bench.positions`` - and the run steps its engine; given an address,
    a ``Context`` over ``local`` as before. The run steps the engine in this process, never the application."""
    from .core import Bench as _Bench, Error, ms

    acting = None
    if isinstance(bench, _Bench):
        door = bench.bx._door
        if not hasattr(door, "local"):
            raise Error("backtest.run steps the engine in this process: open the workspace with bentrix.open, not bentrix.connect")
        acting, local, bench = bench, door.local, bench.uri
    local = local or Local()
    local.arm(bench)
    # Read, never stepped: a run with max_bars=0 leaves the clock where it stood (the strict audit of
    # 2026-11-04, S1 — this read was a step of one tick).
    state = local.state(bench)
    held = state.get("markets") or []
    if symbols:
        missing = [s for s in symbols if s not in held]
        if missing:
            raise ValueError(f"{bench} does not hold {', '.join(missing)}; it holds {', '.join(held) or 'no market'}")
        symbol = symbol or symbols[0]
    if symbol is None:
        if not held:
            raise ValueError(f"{bench} holds no market")
        symbol = held[0]
    walked = list(symbols) if symbols else [symbol]
    end = (state.get("bounds") or {}).get("to_ms")
    if to is not None:
        end = ms(to) if end is None else min(end, ms(to))
    ctx: Any = acting if acting is not None else Context(local, bench, symbol, walked)
    strategy.on_start(ctx)
    seen = 0
    # Each close is the engine's own (S2) and one native call (section 2.4): the advance, the venue's
    # answers and every market's closed candle together, no frame drawn between. A tick interval, or a
    # month on a broker's clock, is refused here by its words.
    while max_bars is None or seen < max_bars:
        step = local.next_close(bench, interval, walked, end)
        if step is None:
            break
        _reached, _opened, rows = step
        closed: Dict[str, Bar] = {s: Bar(int(r[0]), r[1], r[2], r[3], r[4], r[5]) for s, r in rows}
        if symbols:
            if closed:
                strategy.on_bars(closed, ctx)
                seen += 1
        elif symbol in closed:
            strategy.on_bar(closed[symbol], ctx)
            seen += 1
    strategy.on_end(ctx)
    return {"bars": seen, "book": ctx.book if acting is not None else ctx.book()}
