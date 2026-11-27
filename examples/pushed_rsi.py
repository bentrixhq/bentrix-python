"""A pushed study: the agent's own RSI, computed here and drawn on a Bentrix chart.

The values are this script's: Bentrix draws what it is handed and never computes them. The bars are
the workbench's own, read up to its cursor and never past it, so a pushed study cannot see the future
either: a time past the cursor is refused by name.

    python examples/pushed_rsi.py            # the first chart of the first workbench

Standard library only.
"""

import bentrix

SPEC = {
    "schema_version": 1,
    "id": "agent_rsi",
    "title": "Agent RSI",
    "about": "Wilder's ratio of the average gain to the average loss over 14 bars, scaled to 0-100.",
    "family": "oscillator",
    "levels": [30, 70],
    "scale": {"fixed": [0, 100]},
    "lines": [{"name": "rsi", "plot": "line"}],
}

PERIOD = 14


def rsi(closes, n=PERIOD):
    """Wilder's RSI over ``closes``, ``None`` before its first value."""
    out = [None] * len(closes)
    if len(closes) <= n:
        return out
    gains = [max(b - a, 0.0) for a, b in zip(closes, closes[1:])]
    losses = [max(a - b, 0.0) for a, b in zip(closes, closes[1:])]
    up, down = sum(gains[:n]) / n, sum(losses[:n]) / n
    for i in range(n, len(closes)):
        if i > n:
            up = (up * (n - 1) + gains[i - 1]) / n
            down = (down * (n - 1) + losses[i - 1]) / n
        out[i] = 100.0 if down == 0 else 100.0 - 100.0 / (1.0 + up / down)
    return out


def first_chart(b):
    """The first chart of the first workbench, and the workbench's address."""
    state = b.workspace.get_state(depth="charts")
    for lab in state.get("laboratories", []):
        for bench in lab.get("benches", []):
            for chart in bench.get("charts", []):
                return bench, chart
    raise SystemExit("No chart is open: open one in Bentrix first.")


def main():
    b = bentrix.connect()
    bench, chart = first_chart(b)
    b.study.define(SPEC)
    attached = b.study.attach(chart["uri"], "custom:agent_rsi")
    # study_attach answers the chart; a custom study is filed as kind "custom" with its spec's id, and
    # the one just attached is the newest.
    study = max(s["id"] for s in attached["setup"]["studies"]
                if s["kind"] == "custom" and s["settings"].get("spec") == "agent_rsi")
    page = b.data.bars_get(bench["uri"], chart["market"], chart["timeframe"], limit=5000)
    col = {name: k for k, name in enumerate(page["columns"])}
    rows = page["bars"]
    times = [r[col["time_ms"]] for r in rows]
    closes = [r[col["close"]] for r in rows]
    b.study.series_put(chart["uri"], study, epoch=1, times=times, lines={"rsi": rsi(closes)})
    print("Drew %d values of Agent RSI on %s %s." % (len(times), chart["market"], chart["timeframe"]))


if __name__ == "__main__":
    main()
