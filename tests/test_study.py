"""The hosted side: ``bentrix.study`` (the decorator and the helpers) and ``bentrix.host`` over its pipes.

The decorator's tests need nothing; the helpers and the host need numpy and are skipped without it.
"""

import json
import math
import os
import struct
import subprocess
import sys
import textwrap

import pytest

from bentrix import study as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
BOOTSTRAP = "import sys; sys.path.insert(0, sys.argv[1]); from bentrix.host import main; sys.exit(main(sys.argv[2:]))"


# ------------------------------------------------------------------------------------------ the decorator


def test_the_decorator_writes_the_custom_study_spec():
    before = len(S.REGISTERED)

    @S.study(id="t_rsi", title="T RSI", family="oscillator", levels=[30, 70], scale=(0, 100),
             lines=[S.Line("rsi"), S.Line("hist", plot="histogram")],
             inputs=[S.Input("period", int, 14, min=1, max=5000), S.Input("source", S.Price, "close"),
                     S.Input("mode", S.Choice("fast", "slow"), "fast"), S.Input("k", float, 2.0), S.Input("on", bool, True)],
             lookback=lambda p: 10 * p["period"])
    def t_rsi(bars, p):
        return {}

    assert len(S.REGISTERED) == before + 1 and S.REGISTERED[-1] is t_rsi
    spec = t_rsi.spec
    assert spec["schema_version"] == 1 and spec["id"] == "t_rsi" and spec["family"] == "oscillator"
    assert spec["scale"] == {"fixed": [0.0, 100.0]} and spec["levels"] == [30.0, 70.0]
    assert spec["lines"] == [{"name": "rsi", "plot": "line"}, {"name": "hist", "plot": "histogram"}]
    kinds = [(i["name"], i["kind"]) for i in spec["inputs"]]
    assert kinds == [("period", "int"), ("source", "price"), ("mode", "choice"), ("k", "float"), ("on", "bool")]
    assert spec["inputs"][2]["choices"] == ["fast", "slow"]
    assert spec["lookback"] == 140, "the lookback at the defaults"
    assert t_rsi.lookback({"period": 21}) == 210, "and for other inputs"
    json.dumps(spec)  # the spec is plain JSON


def test_an_input_not_declared_is_refused():
    @S.study(id="t_one", title="One", family="overlay", lines=[S.Line("x")], inputs=[S.Input("n", int, 3)])
    def one(bars, p):
        return {}

    assert one.resolve({"n": 5}) == {"n": 5}
    with pytest.raises(ValueError, match="colour"):
        one.resolve({"colour": "red"})


def test_the_prices_are_the_seven_applied_prices():
    # Pinned against the application's own list by `bentrix-pystudy`; here the order and the count.
    assert S.PRICES == ("close", "open", "high", "low", "median", "typical", "weighted")


# ------------------------------------------------------------------------------------------ the helpers


def _loop_recursive(x, a, n):
    out = [math.nan] * len(x)
    if len(x) < n:
        return out
    s = sum(x[:n]) / n
    out[n - 1] = s
    for t in range(n, len(x)):
        s = s + a * (x[t] - s)
        out[t] = s
    return out


def _close(got, want, rel):
    np = pytest.importorskip("numpy")
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert got.shape == want.shape
    assert np.array_equal(np.isnan(got), np.isnan(want)), "NaN in the same places"
    m = ~np.isnan(want)
    assert np.all(np.abs(got[m] - want[m]) <= rel * np.maximum(np.abs(want[m]), 1.0)), float(np.max(np.abs(got[m] - want[m])))


def _walk(n, seed=7):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    return 2000.0 + np.cumsum(rng.normal(0, 0.5, n))


@pytest.mark.parametrize("n", [1, 2, 14, 50, 200])
def test_wilder_and_ema_agree_with_the_per_bar_recursion(n):
    x = _walk(30_000)
    _close(S.wilder(x, n), _loop_recursive(x.tolist(), 1.0 / n, n), 1e-11)
    _close(S.ema(x, n), _loop_recursive(x.tolist(), 2.0 / (n + 1.0), n), 1e-11)


def test_a_recursive_helper_starts_after_leading_nans():
    np = pytest.importorskip("numpy")
    x = np.concatenate(([np.nan] * 5, _walk(100)))
    w = S.wilder(x, 10)
    assert np.all(np.isnan(w[:14])) and not np.isnan(w[14])
    assert w[14] == pytest.approx(np.mean(x[5:15]))


@pytest.mark.parametrize("n", [1, 3, 20, 64, 65, 777])
def test_the_rolling_helpers_agree_with_a_window_at_a_time(n):
    np = pytest.importorskip("numpy")
    x = _walk(12_000)
    windows = [x[i - n + 1 : i + 1] for i in range(n - 1, len(x))]
    pad = [math.nan] * (n - 1)
    _close(S.rolling_mean(x, n), pad + [w.mean() for w in windows], 1e-12)
    _close(S.rolling_std(x, n), pad + [w.std() for w in windows], 1e-9)
    assert np.array_equal(S.rolling_max(x, n)[n - 1 :], [w.max() for w in windows])
    assert np.array_equal(S.rolling_min(x, n)[n - 1 :], [w.min() for w in windows])
    assert np.all(np.isnan(S.rolling_max(x, n)[: n - 1]))


def test_a_series_shorter_than_its_window_is_all_nan():
    np = pytest.importorskip("numpy")
    x = _walk(5)
    for f in (S.wilder, S.ema, S.rolling_mean, S.rolling_std, S.rolling_max, S.rolling_min):
        assert np.all(np.isnan(f(x, 6))), f.__name__


def test_the_applied_prices_are_grouped_as_the_built_in_groups_them():
    np = pytest.importorskip("numpy")
    h, l, c = np.array([3.3]), np.array([1.1]), np.array([2.2])
    b = S.Bars({"time": np.array([0]), "open": c, "high": h, "low": l, "close": c, "volume": c})
    assert b["typical"][0] == (3.3 + 1.1 + 2.2) / 3.0
    assert b["weighted"][0] == (3.3 + 1.1 + 2.0 * 2.2) / 4.0
    assert b["median"][0] == (3.3 + 1.1) / 2.0
    with pytest.raises(KeyError):
        b["hl2"]


# ------------------------------------------------------------------------------------------ the host


def _columns(path, closes):
    n = len(closes)
    with open(path, "wb") as f:
        f.write(struct.pack("<%dq" % n, *[60_000 * i for i in range(n)]))
        for col in ("open", "high", "low", "close", "volume"):
            f.write(struct.pack("<%dd" % n, *closes))


def _columns_ohlc(path, opens, closes):
    n = len(closes)
    with open(path, "wb") as f:
        f.write(struct.pack("<%dq" % n, *[60_000 * i for i in range(n)]))
        for col in (opens, [max(o, c) for o, c in zip(opens, closes)], [min(o, c) for o, c in zip(opens, closes)], closes, [1.0] * n):
            f.write(struct.pack("<%dd" % n, *col))


class Pipe:
    def __init__(self, script):
        self.p = subprocess.Popen([sys.executable, "-I", "-c", BOOTSTRAP, ROOT, script],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def ask(self, req):
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        return json.loads(line) if line else None

    def finish(self):
        try:
            self.p.stdin.write(json.dumps({"op": "close"}) + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        out, err = self.p.communicate(timeout=20)
        return self.p.returncode, out, err


def test_the_host_walks_describe_open_compute_advance_close(tmp_path):
    pytest.importorskip("numpy")
    closes = [1.0 + 0.01 * ((i * 37) % 11) for i in range(300)]
    cols = tmp_path / "columns-1.bin"
    _columns(cols, closes)
    h = Pipe(os.path.join(EXAMPLES, "my_sma.py"))
    spec = h.ask({"op": "describe"})["spec"]
    assert spec["id"] == "my_sma" and spec["lookback"] == 20
    opened = h.ask({"op": "open", "epoch": 1, "columns": str(cols), "len": 300, "point": 0.01, "tf": "M1", "inputs": {"period": 5}})
    assert opened == {"op": "opened", "epoch": 1, "len": 300, "last_time": 299 * 60_000, "lookback": 5}
    values = h.ask({"op": "compute", "epoch": 1, "len": 300})
    sma = values["lines"]["sma"]["inline"]
    assert values["len"] == 300 and len(sma) == 300
    assert sma[:4] == [None] * 4, "no value before the first full window travels as null"
    assert sma[10] == pytest.approx(sum(closes[6:11]) / 5)
    tail = [[60_000 * i, c, c, c, c, 1.0] for i, c in enumerate(closes[-6:], start=294)]
    t = h.ask({"op": "advance", "epoch": 1, "len": 300, "tail": tail})
    assert t["op"] == "tail" and len(t["tail"]["sma"]) == 6
    assert t["tail"]["sma"][-1] == pytest.approx(sma[-1])
    code, out, err = h.finish()
    assert code == 0 and out == ""


def test_a_long_series_is_answered_through_a_file(tmp_path):
    np = pytest.importorskip("numpy")
    n = 60_000
    closes = list(_walk(n))
    cols = tmp_path / "columns-3.bin"
    _columns(cols, closes)
    h = Pipe(os.path.join(EXAMPLES, "my_sma.py"))
    h.ask({"op": "describe"})
    h.ask({"op": "open", "epoch": 3, "columns": str(cols), "len": n, "point": 0.01, "tf": "M1", "inputs": {}})
    v = h.ask({"op": "compute", "epoch": 3, "len": n})
    path = v["lines"]["sma"]["file"]
    assert os.path.dirname(path) == str(tmp_path)
    got = np.fromfile(path, dtype="<f8")
    assert len(got) == n and got[-1] == pytest.approx(np.mean(closes[-20:]))
    h.finish()


def test_a_print_goes_to_stderr_and_an_error_is_a_traceback_and_an_exit(tmp_path):
    pytest.importorskip("numpy")
    script = tmp_path / "loud.py"
    script.write_text(textwrap.dedent("""
        from bentrix.study import study, Line
        print("loading")

        @study(id="loud", title="Loud", family="overlay", lines=[Line("x")])
        def loud(bars, p):
            raise ZeroDivisionError("on purpose")
    """))
    cols = tmp_path / "columns-1.bin"
    _columns(cols, [1.0, 2.0])
    h = Pipe(str(script))
    assert h.ask({"op": "describe"})["spec"]["id"] == "loud", "the print did not reach the protocol"
    h.ask({"op": "open", "epoch": 1, "columns": str(cols), "len": 2, "point": 0.01, "tf": "M1", "inputs": {}})
    assert h.ask({"op": "compute", "epoch": 1, "len": 2}) is None, "an error is not an answer"
    code, out, err = h.finish()
    assert code == 1
    assert "loading" in err
    assert err.strip().splitlines()[-1] == "ZeroDivisionError: on purpose"


def test_a_script_declaring_two_studies_is_refused(tmp_path):
    script = tmp_path / "two.py"
    script.write_text(textwrap.dedent("""
        from bentrix.study import study, Line
        @study(id="a", title="A", family="overlay", lines=[Line("x")])
        def a(bars, p): return {}
        @study(id="b", title="B", family="overlay", lines=[Line("x")])
        def b(bars, p): return {}
    """))
    h = Pipe(str(script))
    assert h.ask({"op": "describe"}) is None
    code, _, err = h.finish()
    assert code == 1 and "exactly one @study" in err


def test_an_answer_with_an_undeclared_line_is_refused_in_the_host(tmp_path):
    pytest.importorskip("numpy")
    script = tmp_path / "peek.py"
    script.write_text(textwrap.dedent("""
        from bentrix.study import study, Line
        @study(id="peek", title="Peek", family="overlay", lines=[Line("x")])
        def peek(bars, p):
            return {"x": bars["close"], "y": bars["close"]}
    """))
    cols = tmp_path / "columns-1.bin"
    _columns(cols, [1.0, 2.0])
    h = Pipe(str(script))
    h.ask({"op": "describe"})
    h.ask({"op": "open", "epoch": 1, "columns": str(cols), "len": 2, "point": 0.01, "tf": "M1", "inputs": {}})
    assert h.ask({"op": "compute", "epoch": 1, "len": 2}) is None
    code, _, err = h.finish()
    assert code == 1 and "undeclared" in err



_COLOURED = """
import numpy as np
from bentrix.study import study, Line, rgba, NONE

@study(id="dir", title="Direction", family="overlay", lines=[Line("x")], bar_colors=[{}])
def direction(bars, p):
    up = bars["close"] >= bars["open"]
    return {"x": bars["close"]}, {"x": np.where(up, rgba(38, 166, 154), rgba(239, 83, 80)),
                                  "bar_color:0": np.where(up, NONE, rgba(0, 0, 255, 128))}
"""


def test_rgba_is_the_colour_as_a_number_and_refuses_a_part_out_of_range():
    assert S.rgba(38, 166, 154) == 0x26A69AFF
    assert S.rgba(0, 0, 0, 0) == 0 and S.rgba(255, 255, 255, 255) == 0xFFFFFFFF
    for bad in ((256, 0, 0), (-1, 0, 0), (1.5, 0, 0)):
        with pytest.raises(ValueError):
            S.rgba(*bad)


def test_the_channels_are_the_lines_then_fills_backgrounds_and_bar_colours():
    spec = {"lines": [{"name": "a"}, {"name": "b"}], "fills": [{"between": ["a", "b"]}], "bar_colors": [{}]}
    assert S.channels(spec) == ["a", "b", "fill:0", "bar_color:0"]


def test_a_studys_colours_travel_beside_its_values_bar_by_bar(tmp_path):
    """Mutation (2026-10-29): the host dropping the answer's colours — red, *no colours answered*."""
    pytest.importorskip("numpy")
    script = tmp_path / "dir.py"
    script.write_text(_COLOURED)
    opens = [1.0, 2.0, 3.0, 2.0]
    closes = [2.0, 1.0, 3.0, 2.5]
    cols = tmp_path / "columns-1.bin"
    _columns_ohlc(cols, opens, closes)
    h = Pipe(str(script))
    spec = h.ask({"op": "describe"})["spec"]
    assert spec["bar_colors"] == [{}]
    h.ask({"op": "open", "epoch": 1, "columns": str(cols), "len": 4, "point": 0.01, "tf": "M1", "inputs": {}})
    v = h.ask({"op": "compute", "epoch": 1, "len": 4})
    assert "colors" in v, "no colours answered"
    up, down, blue = 0x26A69AFF, 0xEF5350FF, 0x0000FF80
    assert v["colors"]["x"]["inline"] == [up, down, up, up]
    assert v["colors"]["bar_color:0"]["inline"] == [None, blue, None, None], "NONE travels as null"
    tail = [[0, 3.0, 3.0, 1.0, 1.0, 1.0]]
    t = h.ask({"op": "advance", "epoch": 1, "len": 4, "tail": tail})
    assert t["colors"] == {"x": [down], "bar_color:0": [blue]}
    code, _, _ = h.finish()
    assert code == 0


def test_colours_for_an_undeclared_channel_are_an_error(tmp_path):
    pytest.importorskip("numpy")
    script = tmp_path / "peek.py"
    script.write_text(textwrap.dedent("""
        from bentrix.study import study, Line, rgba
        @study(id="peek", title="Peek", family="overlay", lines=[Line("x")])
        def peek(bars, p):
            return {"x": bars["close"]}, {"background:0": [rgba(1, 2, 3)] * len(bars)}
    """))
    cols = tmp_path / "columns-1.bin"
    _columns(cols, [1.0, 2.0])
    h = Pipe(str(script))
    h.ask({"op": "describe"})
    h.ask({"op": "open", "epoch": 1, "columns": str(cols), "len": 2, "point": 0.01, "tf": "M1", "inputs": {}})
    assert h.ask({"op": "compute", "epoch": 1, "len": 2}) is None
    code, _, err = h.finish()
    assert code == 1 and "colours for channels not declared" in err


def test_an_int_input_given_as_a_whole_float_reaches_the_script_as_an_int(tmp_path):
    """**A chart's 14.0 is the script's 14**: the RSI example, whose period is a slice bound in
    ``wilder``, opened with ``{"period": 14.0}`` as a chart's number travels, computes — the release
    smoke of 2026-11-01 found the study failing on the chart with *slice indices must be integers*.

    Mutation (2026-11-01): ``Study.resolve`` handing the values on as given — red, the
    host ends during compute."""
    pytest.importorskip("numpy")
    closes = [1.0 + 0.01 * ((i * 37) % 11) for i in range(300)]
    cols = tmp_path / "columns-14.bin"
    _columns(cols, closes)
    h = Pipe(os.path.join(EXAMPLES, "my_rsi.py"))
    h.ask({"op": "describe"})
    h.ask({"op": "open", "epoch": 1, "columns": str(cols), "len": 300, "point": 0.01, "tf": "M1", "inputs": {"period": 14.0}})
    values = h.ask({"op": "compute", "epoch": 1, "len": 300})
    rsi = values["lines"]["rsi"]["inline"]
    assert len(rsi) == 300 and rsi[13] is None and rsi[14] is not None, rsi[:16]
    h.finish()
