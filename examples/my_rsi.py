"""A hosted oscillator: Wilder's relative strength index, in its own pane with levels at 30 and 70.

``bentrix.study.wilder`` does the smoothing without a per-bar Python loop; ``bentrix-pystudy``'s header
measures it beside the same RSI written as a loop.
"""

import numpy as np

from bentrix.study import Input, Line, study, wilder


@study(id="my_rsi", title="My RSI", family="oscillator",
       about="Wilder's ratio of the average gain to the average loss over a period, scaled to 0-100.",
       levels=[30, 70], scale=(0, 100),
       lines=[Line("rsi", plot="line")],
       inputs=[Input("period", int, 14, min=1, max=5000)],
       lookback=lambda p: 10 * p["period"])
def my_rsi(bars, p):
    close = bars["close"]
    n = p["period"]
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return {"rsi": out}
    d = np.diff(close)                               # change i is close[i+1] - close[i]
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    au, ad = wilder(up, n), wilder(dn, n)            # the first average at change n-1, which is bar n
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = np.where(ad == 0.0, 100.0, 100.0 - 100.0 / (1.0 + au / ad))
    out[1:] = np.where(np.isnan(au), np.nan, rsi)
    return {"rsi": out}
