"""**A labelled dataset over MCP is the Python one**: ``extract_dataset``, called through
the embedded BENTRIX (the tool an agent calls), held to ``bentrix.data``. The environment's own tools
are held to the Python implementation they replaced by ``test_gym_reference.py``.

Skipped without the extension module; run on the CI tape (``conftest.tape``).
"""

import pytest

pytest.importorskip("bentrix._native")
np = pytest.importorskip("numpy")

H1 = 3_600_000

def test_an_extracted_dataset_is_the_python_ones(tape):
    """``extract_dataset`` writes the closed candles to the cursor with the labels
    ``bentrix.data.triple_barrier`` gives them — long and short, the same rows and the same labels.

    Mutation (2026-11-01): the Rust labeller giving a candle touching both barriers to
    the target — red, the ambiguous candles' labels differ."""
    import csv

    from bentrix.data import bars, triple_barrier

    local, bench = tape
    local.arm(bench)
    start = local.call("replay_step", bench=bench, unit="tick", n=1)["cursor_ms"]
    local.advance(bench, start + 60 * H1)
    cols = bars(local, bench, "XAUUSD", "H1")
    point = local.call("symbol_info", bench=bench, symbol="XAUUSD")["point"]
    for side in ("long", "short"):
        out = local.call("extract_dataset", bench=bench, symbol="XAUUSD", timeframe="H1", target_points=60.0, stop_points=40.0, horizon=6, side=side)
        rows = list(csv.DictReader(open(out["path"], encoding="utf-8")))
        labels, ambiguous = triple_barrier(cols, point, 60.0, 40.0, 6, side=1 if side == "long" else -1)
        assert [int(r["time_ms"]) for r in rows] == cols["time_ms"].tolist()
        assert [int(r["label"]) for r in rows] == labels.tolist(), side
        assert [r["ambiguous"] == "1" for r in rows] == ambiguous.tolist(), side
        assert ambiguous.any() and set(labels.tolist()) == {-1, 0, 1}, "the walk proves little without every label"
