"""A hosted overlay: the simple moving average of an applied price.

Put it in BENTRIX's indicators folder; it appears under *My indicators*. BENTRIX runs it with
``python -I -m bentrix.host my_sma.py`` and hands it the chart's bars up to the cursor.
"""

import numpy as np

from bentrix.study import Input, Line, Price, study


@study(id="my_sma", title="My SMA", family="overlay",
       about="The mean of the last period values of an applied price.",
       lines=[Line("sma", plot="line")],
       inputs=[Input("period", int, 20, min=1, max=5000), Input("source", Price, "close")],
       lookback=lambda p: p["period"])            # bars re-handed per tick
def my_sma(bars, p):
    src = bars[p["source"]]                        # np.ndarray, float64, oldest first, up to the cursor
    n = p["period"]
    out = np.full(len(src), np.nan)
    if len(src) >= n:
        c = np.cumsum(np.insert(src, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return {"sma": out}
