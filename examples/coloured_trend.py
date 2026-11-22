"""A hosted overlay coloured bar by bar: an EMA drawn green where it rises and red where it falls, and
the candles behind a strong rise tinted.

The function answers ``(lines, colors)``: the colours by channel — the line's own name, and
``bar_color:0`` for the first of ``bar_colors`` — each ``rgba(...)`` or ``NONE`` where the bar takes
none.

Copy it into BENTRIX's indicators folder (Settings › Indicators shows where) and tick it there.
"""

import numpy as np

from bentrix.study import NONE, Input, Line, ema, rgba, study

UP = rgba(38, 166, 154)
DOWN = rgba(239, 83, 80)


@study(id="coloured_trend", title="Coloured trend", family="overlay",
       about="An exponential moving average coloured by the direction it moved on each bar.",
       lines=[Line("ema")],
       inputs=[Input("period", int, 21, min=2, max=500)],
       bar_colors=[{}],
       lookback=lambda p: 10 * p["period"])
def coloured_trend(bars, p):
    line = ema(bars["close"], p["period"])
    rising = np.concatenate(([False], np.diff(line) > 0))
    strong = np.concatenate(([False], np.diff(line) > np.nanstd(np.diff(line))))
    colours = np.where(np.isnan(line), NONE, np.where(rising, UP, DOWN))
    candles = np.where(strong, rgba(38, 166, 154, 96), NONE)
    return {"ema": line}, {"ema": colours, "bar_color:0": candles}
