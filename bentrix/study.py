"""A person's own study, run by Bentrix in a study host (``python -I -m bentrix.host <script>``).

    import numpy as np
    from bentrix.study import study, Line, Input, Price

    @study(id="my_sma", title="My SMA", family="overlay",
           lines=[Line("sma")],
           inputs=[Input("period", int, 20, min=1, max=5000), Input("source", Price, "close")],
           lookback=lambda p: p["period"])
    def my_sma(bars, p):
        return {"sma": rolling_mean(bars[p["source"]], p["period"])}

The function is handed ``bars`` — the chart's bars up to the replay's cursor and never past it, oldest
first — and the inputs, every declared one resolved. ``bars["close"]`` is a read-only ``float64`` array;
``time``, ``open``, ``high``, ``low``, ``close`` and ``volume`` are the columns, and ``median``,
``typical`` and ``weighted`` are the applied prices a built-in study takes. It answers a dict with one
array per declared line, each as long as ``bars``; NaN where a line has no value yet.

**Colours, bar by bar.** It may answer a pair instead — ``(lines, colors)`` — where ``colors`` holds an
array per *channel* it colours, each as long as ``bars``: a declared line's name, ``"fill:<k>"``,
``"background:<k>"`` or ``"bar_color:<k>"`` for the ``k``-th of ``fills``, ``backgrounds`` and
``bar_colors``. A colour is ``rgba(r, g, b, a=255)`` (``0xRRGGBBAA`` as a number) and ``NONE`` where the
bar takes none; a channel left out takes none on any bar::

    up = bars["close"] >= bars["open"]
    return {"sma": sma}, {"sma": np.where(up, rgba(38, 166, 154), rgba(239, 83, 80))}

The declaration is the ``custom_study_spec`` document (schema 1); Bentrix validates it and refuses a
spec by name — an id that is not ``[a-z0-9_]{1,32}``, more than 64 lines, a level outside a fixed scale,
an input default outside its bounds, a repeated name, an ``about`` that advises.

The helpers keep a script fast without a per-bar Python loop: ``wilder``, ``ema``, ``rolling_mean``,
``rolling_std``, ``rolling_max`` and ``rolling_min``, each over numpy, each returning an array as long
as its input with NaN before its first full window.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:  # The decorator needs no numpy; the helpers and the host do.
    import numpy as np
except ImportError:  # pragma: no cover - exercised only where numpy is absent
    np = None

SCHEMA_VERSION = 1

#: The applied prices, in MT5's order — the tokens of a ``Price`` input. Pinned against the
#: application's own enumeration by ``bentrix-pystudy``'s ``the_sdk_prices_are_the_applied_prices``.
PRICES = ("close", "open", "high", "low", "median", "typical", "weighted")

#: The columns of the file the host maps, in file order.
COLUMNS = ("time", "open", "high", "low", "close", "volume")

PLOTS = ("line", "step", "histogram", "marks", "area", "candle", "candle_part")

#: A bar that takes no colour.
NONE = float("nan")


def rgba(r, g, b, a=255):
    """``0xRRGGBBAA`` as a number — each part 0..255, a number or a numpy array of them."""
    if np is not None:
        r, g, b, a = (np.asarray(x, dtype=np.float64) for x in (r, g, b, a))
        for x in (r, g, b, a):
            if np.any((x < 0) | (x > 255) | (x != np.floor(x))):
                raise ValueError("a colour's parts are whole numbers from 0 to 255")
    elif not all(isinstance(x, int) and 0 <= x <= 255 for x in (r, g, b, a)):
        raise ValueError("a colour's parts are whole numbers from 0 to 255")
    return ((r * 256.0 + g) * 256.0 + b) * 256.0 + a


def channels(spec: Dict[str, Any]) -> List[str]:
    """The channels a study may colour, in the application's order: its lines, then its fills,
    backgrounds and bar colours by position."""
    out = [l["name"] for l in spec["lines"]]
    out += ["fill:%d" % k for k in range(len(spec.get("fills", ())))]
    out += ["background:%d" % k for k in range(len(spec.get("backgrounds", ())))]
    out += ["bar_color:%d" % k for k in range(len(spec.get("bar_colors", ())))]
    return out


class _PriceKind:
    """The kind of an input that names an applied price."""

    def __repr__(self) -> str:
        return "Price"


Price = _PriceKind()


class Choice:
    """The kind of an input that is one of a few names: ``Input("mode", Choice("fast", "slow"), "fast")``."""

    def __init__(self, *choices: str) -> None:
        if not choices:
            raise ValueError("a Choice needs at least one choice")
        self.choices = [str(c) for c in choices]

    def __repr__(self) -> str:
        return "Choice(%s)" % ", ".join(repr(c) for c in self.choices)


class Line:
    """A declared line: its name (the key the function answers it under) and how it is plotted."""

    def __init__(self, name: str, plot: str = "line") -> None:
        if plot not in PLOTS:
            raise ValueError("plot %r is not one of %s" % (plot, ", ".join(PLOTS)))
        self.name = name
        self.plot = plot

    def to_spec(self) -> Dict[str, Any]:
        return {"name": self.name, "plot": self.plot}


class Input:
    """A declared input: ``Input(name, kind, default, min=None, max=None)`` with ``kind`` one of
    ``int``, ``float``, ``bool``, ``Price`` or a ``Choice``."""

    def __init__(self, name: str, kind: Any, default: Any, *, min: Optional[float] = None, max: Optional[float] = None) -> None:
        self.name = name
        self.kind = kind
        self.default = default
        self.min = min
        self.max = max

    def kind_token(self) -> str:
        if self.kind is bool:
            return "bool"
        if self.kind is int:
            return "int"
        if self.kind is float:
            return "float"
        if self.kind is Price:
            return "price"
        if isinstance(self.kind, Choice):
            return "choice"
        raise TypeError("input %r: kind %r is not int, float, bool, Price or a Choice" % (self.name, self.kind))

    def to_spec(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "kind": self.kind_token(), "default": self.default}
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if isinstance(self.kind, Choice):
            out["choices"] = list(self.kind.choices)
        return out


class Study:
    """What ``@study`` makes: the function and its declaration."""

    def __init__(self, func: Callable, spec: Dict[str, Any], lookback: Union[None, int, Callable]) -> None:
        self.func = func
        self.spec = spec
        self._lookback = lookback

    def defaults(self) -> Dict[str, Any]:
        return {i["name"]: i["default"] for i in self.spec.get("inputs", [])}

    def resolve(self, given: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Every declared input: the given value where one is given, the default otherwise."""
        given = dict(given or {})
        unknown = sorted(set(given) - set(self.defaults()))
        if unknown:
            raise ValueError("inputs not declared by %s: %s" % (self.spec["id"], ", ".join(unknown)))
        out = self.defaults()
        out.update(given)
        # Each value as its declaration's kind: an int input given as a whole float (14.0, as a
        # chart's number travels) is the int the script declared — a period used as a slice bound
        # or a count must be one.
        for i in self.spec.get("inputs", []):
            v = out.get(i["name"])
            if i["kind"] == "int" and isinstance(v, float) and v.is_integer():
                out[i["name"]] = int(v)
            elif i["kind"] == "float" and isinstance(v, int) and not isinstance(v, bool):
                out[i["name"]] = float(v)
        return out

    def lookback(self, inputs: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """Bars re-handed per tick for these inputs; ``None`` updates at bar close only."""
        if self._lookback is None:
            return None
        value = self._lookback(self.resolve(inputs)) if callable(self._lookback) else self._lookback
        return int(value)

    def __call__(self, bars: Any, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.func(bars, self.resolve(inputs))


#: Every ``@study`` made in this process, in order. The host requires exactly one per script.
REGISTERED: List[Study] = []


def study(
    *,
    id: str,
    title: str,
    family: str,
    lines: Sequence[Line],
    inputs: Sequence[Input] = (),
    about: str = "",
    levels: Sequence[float] = (),
    scale: Union[None, str, Tuple[float, float]] = None,
    lookback: Union[None, int, Callable[[Dict[str, Any]], int]] = None,
    conditions: Sequence[Dict[str, Any]] = (),
    fills: Sequence[Dict[str, Any]] = (),
    backgrounds: Sequence[Dict[str, Any]] = (),
    bar_colors: Sequence[Dict[str, Any]] = (),
) -> Callable[[Callable], Study]:
    """Declare a study. ``family`` is ``"overlay"`` (drawn in prices on the price pane) or
    ``"oscillator"`` (its own pane); ``scale`` is ``None``/``"auto"`` or ``(low, high)``; ``lookback`` is
    the bars re-handed per tick, a number or a function of the inputs. ``fills`` (``{"between": [a, b]}``),
    ``backgrounds`` and ``bar_colors`` (``{}``, or ``{"color": "#rrggbb"}`` for bars the answer leaves
    uncoloured) are the spec's own, each coloured bar by bar through its channel."""

    if family not in ("overlay", "oscillator"):
        raise ValueError("family %r is not overlay or oscillator" % family)
    spec: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": id,
        "title": title,
        "about": about,
        "family": family,
        "lines": [l.to_spec() for l in lines],
        "scale": "auto" if scale in (None, "auto") else {"fixed": [float(scale[0]), float(scale[1])]},
    }
    if levels:
        spec["levels"] = [float(x) for x in levels]
    if inputs:
        spec["inputs"] = [i.to_spec() for i in inputs]
    if conditions:
        spec["conditions"] = [dict(c) for c in conditions]
    for key, given in (("fills", fills), ("backgrounds", backgrounds), ("bar_colors", bar_colors)):
        if given:
            spec[key] = [dict(x) for x in given]

    def wrap(func: Callable) -> Study:
        s = Study(func, spec, lookback)
        at_defaults = s.lookback()
        if at_defaults is not None:
            spec["lookback"] = at_defaults
        REGISTERED.append(s)
        return s

    return wrap


class Bars:
    """The bars a study function is handed: a column by name, oldest first, as ``float64``
    (``time`` as ``int64``). The applied prices are computed on first use."""

    def __init__(self, columns: Dict[str, Any]) -> None:
        self._cols = dict(columns)

    def __len__(self) -> int:
        return len(self._cols["time"])

    def __contains__(self, name: str) -> bool:
        return name in self._cols or name in PRICES

    def keys(self):
        return list(COLUMNS) + [p for p in PRICES if p not in COLUMNS]

    def __getitem__(self, name: str):
        if name not in self._cols:
            h, l, c = self._cols["high"], self._cols["low"], self._cols["close"]
            # The grouping is the built-in's (`AppliedPrice::at`): (h + l + c) / 3, never h + (l + c).
            if name == "median":
                v = (h + l) / 2.0
            elif name == "typical":
                v = (h + l + c) / 3.0
            elif name == "weighted":
                v = (h + l + 2.0 * c) / 4.0
            else:
                raise KeyError("no column %r; there are %s" % (name, ", ".join(self.keys())))
            v.flags.writeable = False
            self._cols[name] = v
        return self._cols[name]


# --------------------------------------------------------------------------------------------- helpers


def _need_numpy():
    if np is None:
        raise ImportError("bentrix.study's helpers need numpy (pip install numpy)")


def _first_full(x, n: int) -> Optional[int]:
    """The index where the first window of ``n`` finite values ends, after any leading NaNs."""
    finite = np.flatnonzero(np.isfinite(x))
    if finite.size == 0:
        return None
    start = int(finite[0])
    end = start + n - 1
    return end if end < len(x) else None


def _recursive(x, a: float, n: int):
    """``s[t] = s[t-1] + a * (x[t] - s[t-1])``, seeded with the mean of the first ``n`` values.

    Closed form per block, so there is no per-bar Python loop: within a block of ``m`` values after a
    known ``s``, ``s[k] = r**(k+1) * (s + a * sum_{j<=k} x[j] / r**(j+1))`` with ``r = 1 - a``. The block
    is kept short enough that ``r**-m`` stays far from overflow (at most 4,096). The result agrees
    with the per-bar recursion to about 1e-13 of the values' size, not bit for bit.
    """
    _need_numpy()
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan)
    if n < 1:
        raise ValueError("a period is at least 1")
    seed_at = _first_full(x, n)
    if seed_at is None:
        return out
    out[seed_at] = x[seed_at - n + 1 : seed_at + 1].mean()
    r = 1.0 - a
    if r <= 0.0:  # n == 1 for Wilder's: the average of one value is the value
        out[seed_at:] = x[seed_at:]
        return out
    block = int(min(4096, max(1, math.floor(250.0 / -math.log10(r)))))
    prev = out[seed_at]
    i = seed_at + 1
    total = len(x)
    while i < total:
        j = min(total, i + block)
        k = np.arange(1, j - i + 1, dtype=np.float64)
        rk = r ** k
        out[i:j] = rk * (prev + a * np.cumsum(x[i:j] / rk))
        prev = out[j - 1]
        i = j
    return out


def wilder(x, n: int):
    """Wilder's smoothing over ``n``: the mean of the first ``n`` values, then
    ``s[t] = (s[t-1] * (n - 1) + x[t]) / n``. RSI's and ATR's average."""
    return _recursive(x, 1.0 / n, n)


def ema(x, n: int):
    """The exponential moving average over ``n`` (``2 / (n + 1)``), seeded with the mean of the first
    ``n`` values — TA-Lib's seeding."""
    return _recursive(x, 2.0 / (n + 1.0), n)


def _blocks(total: int, n: int, block: Optional[int] = None):
    """Output blocks ``[i, j)`` for a trailing window of ``n``: each is computed from its own slice,
    so a running sum never grows past ``n + block`` values and precision does not drift."""
    block = block or max(4 * n, 4096)
    i = n - 1
    while i < total:
        j = min(total, i + block)
        yield i, j
        i = j


def rolling_mean(x, n: int):
    """The mean of the trailing ``n`` values (MT5's SMA)."""
    _need_numpy()
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan)
    if n < 1:
        raise ValueError("a period is at least 1")
    for i, j in _blocks(len(x), n):
        seg = x[i - n + 1 : j]
        c = np.concatenate(([0.0], np.cumsum(seg)))
        out[i:j] = (c[n:] - c[:-n]) / n
    return out


def rolling_std(x, n: int, ddof: int = 0):
    """The standard deviation of the trailing ``n`` values, population by default (``ddof=0``, MT5's
    StdDev).

    Up to 64 values a window is taken two-pass, mean then deviations, as MT5 does. Longer windows use
    running sums over blocks of ``4 n``, each centred on its own mean first so a price far from zero
    loses nothing to cancellation; measured within 1e-9 of the two-pass figure."""
    _need_numpy()
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan)
    if n < 1 or n - ddof < 1:
        raise ValueError("a period is at least 1 and above ddof")
    if n <= 64:
        from numpy.lib.stride_tricks import sliding_window_view

        for i, j in _blocks(len(x), n):
            out[i:j] = sliding_window_view(x[i - n + 1 : j], n).std(axis=1, ddof=ddof)
        return out
    for i, j in _blocks(len(x), n, 4 * n):
        seg = x[i - n + 1 : j]
        seg = seg - seg.mean()
        c1 = np.concatenate(([0.0], np.cumsum(seg)))
        c2 = np.concatenate(([0.0], np.cumsum(seg * seg)))
        s1 = c1[n:] - c1[:-n]
        s2 = c2[n:] - c2[:-n]
        var = (s2 - s1 * s1 / n) / (n - ddof)
        out[i:j] = np.sqrt(np.maximum(var, 0.0))
    return out


def _rolling_extreme(x, n: int, ufunc, fill: float):
    """The trailing extreme of ``n`` values in linear time (van Herk / Gil-Werman): a forward
    accumulation within each block of ``n`` and a backward one, combined."""
    _need_numpy()
    x = np.asarray(x, dtype=np.float64)
    total = len(x)
    out = np.full(total, np.nan)
    if n < 1:
        raise ValueError("a period is at least 1")
    if total < n:
        return out
    pad = (-total) % n
    xp = np.concatenate((x, np.full(pad, fill))).reshape(-1, n)
    fwd = ufunc.accumulate(xp, axis=1).ravel()
    bwd = ufunc.accumulate(xp[:, ::-1], axis=1)[:, ::-1].ravel()
    # The window ending at i is [i-n+1, i]: the backward run from its start and the forward run to its end.
    i = np.arange(n - 1, total)
    out[n - 1 :] = ufunc(bwd[i - n + 1], fwd[i])
    return out


def rolling_max(x, n: int):
    """The highest of the trailing ``n`` values."""
    return _rolling_extreme(x, n, np.maximum, -np.inf)


def rolling_min(x, n: int):
    """The lowest of the trailing ``n`` values."""
    return _rolling_extreme(x, n, np.minimum, np.inf)


__all__ = [
    "study", "Line", "Input", "Price", "Choice", "Study", "Bars", "REGISTERED", "PRICES", "COLUMNS",
    "SCHEMA_VERSION", "NONE", "rgba", "channels", "wilder", "ema", "rolling_mean", "rolling_std", "rolling_max", "rolling_min",
]
