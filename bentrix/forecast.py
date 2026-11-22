"""**A model's forecast on the chart**: the input side first — what a model reads.

- ``window(door, bench, frames, symbols=...)`` — ``bars_window``: several markets and frames (intervals, or
  ``{"chart": uri}`` for a chart's elements) cut at ONE instant, the cursor or ``aligned_to`` an interval's last
  close. A row is a bar that closed at or before the cut; nothing past the cursor is ever read.
- ``Forecast(model, symbol, timeframe, anchor_ms)`` — a forecast built step by step and put (`forecast_put`), with
  ``put_many``, ``get`` (by arrival) and ``clear``; it never names its arrival, which Bentrix stamps.
  ``Forecast.of_elements(model, symbol, "renko", anchor_ms, elements={...})`` — a forecast whose form is a chart
  built by price (Renko, Kagi, Line break, Point and figure, Range bars): it names no interval, and its rule is
  ``elements`` where it is not the chart's own.
- ``eval_started(door, made)`` — the run ``forecast_eval_make`` started, or, when it answered an ``ask`` (days the
  workbench's laboratory does not hold), the person's answer on the sheet in Bentrix's window, watched on the
  ``bentrix://forecasts/eval`` event: the run's id, or ``DeclinedByThePerson``, or ``EvalFailed`` with its why.
- ``windows(door, bench, symbol, interval, frames=...)`` — the ``data_closed_window`` stream: each bar of
  ``interval`` as it closes, each with the window cut where that bar ended, so a model reading it runs exactly
  once per close and sees what it would have seen at that close, never later.

A *door* is anything with ``call(tool, **args)``: ``bentrix.local.Local`` (a laboratory in this process),
``bentrix.live.Remote`` (a running Bentrix over MCP).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

#: The columns of an interval frame's rows.
COLUMNS = ("time_ms", "open", "high", "low", "close", "tick_volume", "spread_points")
#: The columns of a chart frame's rows (an element read as a bar).
CHART_COLUMNS = ("time_ms", "ordinal", "open", "high", "low", "close", "volume")

Frame = Union[str, Dict[str, str]]


class Window:
    """**One answer of ``bars_window``** (or one item of the stream), as arrays.

    ``at_ms`` is where every frame is cut, ``cursor_ms`` the workbench's clock when it was read."""

    def __init__(self, answer: Dict[str, Any]):
        self.raw = answer
        self.at_ms: int = answer["at_ms"]
        self.cursor_ms: int = answer["cursor_ms"]
        self.aligned_to: str = answer.get("aligned_to", "cursor")
        self.not_ready: List[Dict[str, Any]] = answer.get("not_ready") or []

    def symbols(self) -> List[str]:
        return list((self.raw.get("markets") or {}).keys())

    def intervals(self, symbol: str) -> List[str]:
        return list(((self.raw.get("markets") or {}).get(symbol) or {}).keys())

    def _frame(self, symbol: str, interval: str) -> Dict[str, Any]:
        try:
            return self.raw["markets"][symbol][interval]
        except KeyError:
            raise KeyError(f"the window holds no {interval} frame of {symbol}") from None

    def bars(self, symbol: str, interval: str) -> np.ndarray:
        """The closed bars, oldest first, an ``(n, 7)`` float64 array in ``COLUMNS`` order."""
        rows = self._frame(symbol, interval)["rows"]
        return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(COLUMNS))

    def times(self, symbol: str, interval: str) -> np.ndarray:
        """The bars' open times (UTC ms), int64 — exact, where ``bars()[:, 0]`` is a float."""
        return np.asarray([r[0] for r in self._frame(symbol, interval)["rows"]], dtype=np.int64)

    def close(self, symbol: str, interval: str) -> np.ndarray:
        return self.bars(symbol, interval)[:, 4]

    def forming(self, symbol: str, interval: str) -> Optional[list]:
        """The bar still forming at the cut, when ``include_forming`` asked for it; never a row."""
        return self._frame(symbol, interval).get("forming")

    def chart(self, uri: str) -> np.ndarray:
        """A chart's closed elements, oldest first, an ``(n, 7)`` float64 array in ``CHART_COLUMNS`` order."""
        try:
            rows = self.raw["charts"][uri]["rows"]
        except KeyError:
            raise KeyError(f"the window holds no frame of {uri}") from None
        return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(CHART_COLUMNS))


def window(
    door: Any,
    bench: str,
    frames: Sequence[Frame],
    symbols: Optional[Sequence[str]] = None,
    symbol: Optional[str] = None,
    limit: Optional[int] = None,
    aligned_to: Optional[str] = None,
    include_forming: Optional[bool] = None,
    partial: Optional[bool] = None,
) -> Window:
    """``bars_window`` through ``door``: the frames of the markets, cut at one instant."""
    args: Dict[str, Any] = {"bench": bench, "frames": list(frames)}
    for k, v in (("symbols", list(symbols) if symbols is not None else None), ("symbol", symbol), ("limit", limit), ("aligned_to", aligned_to), ("include_forming", include_forming), ("partial", partial)):
        if v is not None:
            args[k] = v
    return Window(door.call("bars_window", **args))


class Close:
    """One close of the stream: the bar of ``interval`` that closed (``bar``, its row), where it ended
    (``close_ms``), and the window cut there."""

    __slots__ = ("bar", "close_ms", "window")

    def __init__(self, item: Dict[str, Any]):
        self.bar: list = item["bar"]
        self.close_ms: int = item["close_ms"]
        self.window = Window(item["window"])


class Windows:
    """**A window at each close of one market and interval** — ``wait_for`` on the workbench's
    ``data_closed_window`` stream, from the present on. Iterate it; it ends when ``idle_limit`` waits in a
    row answered nothing (never, when ``idle_limit`` is None). ``rewound`` counts the times the workbench's
    clock went back behind the stream's cursor."""

    def __init__(
        self,
        door: Any,
        bench: str,
        symbol: str,
        interval: str,
        frames: Optional[Sequence[str]] = None,
        symbols: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        timeout_ms: int = 60_000,
        idle_limit: Optional[int] = None,
        before_wait: Optional[Callable[[], Any]] = None,
    ):
        home, owner, _, name = bench.removeprefix("bentrix://").split("/")[:4]
        query = []
        if frames:
            query.append("frames=" + ",".join(frames))
        if symbols:
            query.append("symbols=" + ",".join(symbols))
        if limit is not None:
            query.append(f"limit={int(limit)}")
        self.uri = f"bentrix://{home}/{owner}/bench/{name}/bars/{symbol}/{interval}/closed/window"
        self.door, self.timeout_ms, self.idle_limit, self.before_wait = door, timeout_ms, idle_limit, before_wait
        self.uri = self.uri + ("?" + "&".join(query) if query else "")
        first = door.call("wait_for", uri=self.uri, timeout_ms=0)
        self.cursor: str = first.get("next") or "0"
        self.rewound = 0

    def next_page(self) -> List[Close]:
        """The closes since the last read, oldest first, at most 8 — waiting for at least one; ``[]`` once
        ``idle_limit`` waits in a row answered none."""
        idle = 0
        while True:
            if self.before_wait is not None:
                self.before_wait()
            page = self.door.call("wait_for", uri=self.uri, after=self.cursor, timeout_ms=self.timeout_ms)
            if page.get("rewound"):
                self.rewound += 1
            if page.get("next") is not None:
                self.cursor = page["next"]
            items = page.get("items") or []
            if items:
                return [Close(i) for i in items]
            idle += 1
            if self.idle_limit is not None and idle >= self.idle_limit:
                return []

    def __iter__(self) -> Iterator[Close]:
        while True:
            page = self.next_page()
            if not page:
                return
            yield from page


def windows(door: Any, bench: str, symbol: str, interval: str, **kw: Any) -> Windows:
    """``Windows(door, bench, symbol, interval, ...)``: a window at each close of ``interval``."""
    return Windows(door, bench, symbol, interval, **kw)


# ---- the forecast itself --------------------------------------------------------------

class Forecast:
    """**A forecast to put on a workbench** — the ``forecast_spec`` document, built step by step.

    ``anchor_ms`` is the open of the last CLOSED bar of ``timeframe`` the forecast is made at; step 1 is the bar after
    it. **It carries no arrival**: Bentrix stamps ``put_ms`` on the workbench's own clock when it lands, and no
    reader sees the forecast before then — so a document naming ``put_ms`` is refused here as it is there."""

    def __init__(self, model: str, symbol: str, timeframe: Optional[str], anchor_ms: int, chart_type: str = "candlestick",
                 epoch: int = 0, note: Optional[str] = None, look: Optional[Dict[str, str]] = None,
                 elements: Optional[Dict[str, Any]] = None):
        self.model, self.symbol, self.timeframe, self.anchor_ms = model, symbol, timeframe, int(anchor_ms)
        self.chart_type, self.epoch, self.note, self.look = chart_type, int(epoch), note, look
        self.elements = elements
        self.steps: List[Dict[str, Any]] = []

    @classmethod
    def of_elements(cls, model: str, symbol: str, chart_type: str, anchor_ms: int,
                    elements: Optional[Dict[str, Any]] = None, **kw: Any) -> "Forecast":
        """A forecast of a chart built by price — ``"renko"``, ``"kagi"``, ``"line_break"``, ``"point_figure"``,
        ``"range_bars"``: no interval (its elements close on price, never on the clock); ``elements`` its rule
        (``{"renko_box": 0.0005, "renko_reversal": 2}``) where it is not the chart's own. ``anchor_ms`` is the
        moment of the last closed element; a step's ``expected_ms``, given on every step or none, says when it is
        expected — what places it on a chart by time."""
        return cls(model, symbol, None, anchor_ms, chart_type, elements=elements, **kw)

    @classmethod
    def after(cls, close: "Close", model: str, timeframe: str, **kw: Any) -> "Forecast":
        """A forecast anchored at a close of the ``windows`` stream: its bar, its market and its interval."""
        symbol = kw.pop("symbol", None) or close.window.symbols()[0]
        return cls(model, symbol, timeframe, int(close.bar[0]), **kw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Forecast":
        """A ``forecast_spec`` read back; one naming ``put_ms`` is refused — the arrival is Bentrix's to stamp."""
        if "put_ms" in d:
            raise ValueError("a forecast names no put_ms: Bentrix stamps the arrival")
        f = cls(d["model"], d["symbol"], d["form"].get("timeframe"), d["anchor"]["time_ms"], d["form"].get("chart_type", "candlestick"),
                d.get("epoch", 0), d.get("note"), d.get("look"), d["form"].get("elements"))
        f.steps = [dict(s) for s in d["steps"]]
        return f

    def step(self, close: float, open: Optional[float] = None, high: Optional[float] = None, low: Optional[float] = None,
             volume: Optional[float] = None, bands: Optional[Sequence[Any]] = None, quantiles: Optional[Dict[Any, float]] = None,
             samples: Optional[Sequence[float]] = None, expected_ms: Optional[int] = None) -> "Forecast":
        """One more bar. ``bands`` as ``(lo, hi, p)`` tuples or dicts; ``quantiles`` by level (0.05 or "0.05")."""
        s: Dict[str, Any] = {"close": float(close)}
        for k, v in (("open", open), ("high", high), ("low", low), ("volume", volume), ("expected_ms", expected_ms)):
            if v is not None:
                s[k] = v
        if bands:
            s["bands"] = [b if isinstance(b, dict) else {"lo": b[0], "hi": b[1], **({"p": b[2]} if len(b) > 2 else {})} for b in bands]
        if quantiles:
            s["quantiles"] = {str(k): float(v) for k, v in quantiles.items()}
        if samples is not None and len(samples):
            s["samples"] = [float(x) for x in samples]
        self.steps.append(s)
        return self

    def spec(self) -> Dict[str, Any]:
        """The ``forecast_spec`` document."""
        form: Dict[str, Any] = {"chart_type": self.chart_type}
        if self.timeframe is not None:
            form["timeframe"] = self.timeframe
        if self.elements is not None:
            form["elements"] = dict(self.elements)
        d: Dict[str, Any] = {"schema_version": 1, "model": self.model, "symbol": self.symbol, "form": form,
                             "anchor": {"time_ms": self.anchor_ms}, "steps": self.steps, "epoch": self.epoch}
        if self.note is not None:
            d["note"] = self.note
        if self.look is not None:
            d["look"] = self.look
        return d

    def put(self, door: Any, bench: str) -> Dict[str, Any]:
        """``forecast_put``: answers the arrival (``put_ms``), the state and where it stands on each chart."""
        return door.call("forecast_put", bench=bench, forecast=self.spec())


def put_many(door: Any, bench: str, forecasts: Sequence[Forecast], all_or_nothing: bool = False) -> Dict[str, Any]:
    """``forecast_put_many``: several markets and models at one instant."""
    return door.call("forecast_put_many", bench=bench, forecasts=[f.spec() for f in forecasts], all_or_nothing=all_or_nothing)


def get(door: Any, bench: str, symbol: Optional[str] = None, model: Optional[str] = None, at_ms: Optional[int] = None,
        steps: bool = False) -> Dict[str, Any]:
    """``forecast_get``: what a reader at ``at_ms`` (the workbench's clock when omitted) sees, by arrival."""
    args: Dict[str, Any] = {"bench": bench, "steps": steps}
    for k, v in (("symbol", symbol), ("model", model), ("at_ms", at_ms)):
        if v is not None:
            args[k] = v
    return door.call("forecast_get", **args)


def clear(door: Any, bench: str, symbol: Optional[str] = None, model: Optional[str] = None) -> int:
    """``forecast_clear``: answers how many (market, model) pairs went."""
    args: Dict[str, Any] = {"bench": bench}
    for k, v in (("symbol", symbol), ("model", model)):
        if v is not None:
            args[k] = v
    return door.call("forecast_clear", **args)["cleared"]


# ---- an evaluation's sheet ------------------------------------------------------------

#: The event an ask's moves are carried on.
EVAL_EVENT = "bentrix://forecasts/eval"


class DeclinedByThePerson(Exception):
    """The person pressed *No* on the evaluation's sheet. Not a failure: nothing was made or fetched."""

    def __init__(self, ask: str):
        super().__init__(f"the person declined {ask}")
        self.ask = ask


class EvalFailed(Exception):
    """The laboratory over the span was not made, or the run did not start on it — with Bentrix's why."""

    def __init__(self, ask: str, why: str):
        super().__init__(why)
        self.ask, self.why = ask, why


def eval_started(door: Any, made: Dict[str, Any], timeout_ms: int = 60_000, idle_limit: Optional[int] = None) -> int:
    """**The run to drive**, from ``forecast_eval_make``'s answer: its ``eval`` at once, or — when it answered an
    ``ask`` over days the workbench's laboratory does not hold — the person's answer on the sheet in Bentrix's window,
    watched on ``bentrix://forecasts/eval``: ``started`` answers the run's id; ``declined`` raises
    ``DeclinedByThePerson``; ``failed`` raises ``EvalFailed``. No tool answers the sheet; only the person does.
    ``idle_limit`` waits in a row with no move of this ask end the watch with ``TimeoutError`` (never, when None)."""
    if made.get("eval") is not None:
        return int(made["eval"])
    ask = made["ask"]
    uri = made.get("watch") or EVAL_EVENT
    after = uri.split("after=", 1)[1].split("&", 1)[0] if "after=" in uri else "0"
    uri = uri.split("{", 1)[0].split("?", 1)[0]
    cursor, idle = after or "0", 0
    while True:
        page = door.call("wait_for", uri=uri, after=cursor, timeout_ms=timeout_ms)
        if page.get("next") is not None:
            cursor = page["next"]
        moved = False
        for item in page.get("items") or []:
            if item.get("ask") != ask:
                continue
            moved = True
            state = item.get("state")
            if state == "started":
                return int(item["eval"])
            if state == "declined":
                raise DeclinedByThePerson(ask)
            if state == "failed":
                raise EvalFailed(ask, item.get("why") or "the evaluation did not start")
        idle = 0 if moved else idle + 1
        if idle_limit is not None and idle >= idle_limit:
            raise TimeoutError(f"{ask}: no answer on the sheet")
