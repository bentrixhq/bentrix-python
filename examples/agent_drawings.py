"""An agent's drawings: a trendline across the chart's recent swing, then every drawing it placed found
again by its author.

A drawing an agent places carries the agent's name; the person sees it on the drawing's hover and can
remove all of one agent's drawings at once. An agent may change the person's drawings as a person
could, and the Journal says which.

    python examples/agent_drawings.py

Standard library only.
"""

import bentrix

from pushed_rsi import first_chart


def main():
    b = bentrix.connect()
    bench, chart = first_chart(b)
    page = b.data.bars_get(bench["uri"], chart["market"], chart["timeframe"], limit=200)
    col = {name: k for k, name in enumerate(page["columns"])}
    rows = page["bars"]
    if len(rows) < 2:
        raise SystemExit("The chart has fewer than two bars.")
    low = min(rows, key=lambda r: r[col["low"]])
    high = max(rows, key=lambda r: r[col["high"]])
    first, second = sorted([low, high], key=lambda r: r[col["time_ms"]])
    price = lambda r: r[col["low"]] if r is low else r[col["high"]]
    placed = b.drawing.object_place(
        chart["uri"],
        "trendline",
        anchors=[{"t_ms": first[col["time_ms"]], "price": price(first)}, {"t_ms": second[col["time_ms"]], "price": price(second)}],
        text="swing",
    )
    # The drawing carries this agent as its author — `agent:<pairing>` — which finds them all again.
    author = placed["object"]["drawing"]["author"]
    mine = b.drawing.object_find(chart["uri"], author=author)["objects"]
    print("Placed a trendline; this agent's drawings on the chart: %d." % len(mine))


if __name__ == "__main__":
    main()
