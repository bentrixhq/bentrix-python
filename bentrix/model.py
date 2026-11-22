"""**A model Bentrix runs for you**: one Python file in the models folder declaring one ``@model``.

Bentrix starts the person's own interpreter on the file (``python -I``, nothing on the path but this package), and at
each close of the model's interval on the workbench it was started on hands it the window cut at that close; what the
function answers is put on the chart as the model's forecast, **stamped with the instant it came back** — a slow
model's forecast arrives late, exactly as a live program's would.

::

    import bentrix

    @bentrix.model(name="drift", interval="H1", frames=["H1"], lookback=48, steps=3)
    def drift(w):
        closes = w.close(w.symbol, "H1")
        step = (closes[-1] - closes[-12]) / 12
        return [{"close": closes[-1] + step * (k + 1)} for k in range(3)]

The function takes a :class:`Window` and answers a list of steps — each a number (the close) or a dict as
``forecast_spec``'s steps (``open``, ``high``, ``low``, ``close``, ``bands``, ``quantiles``, ``samples``) — or
``None`` to say nothing at this close.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

_DECLARED: List[Dict[str, Any]] = []


class Window:
    """**What a model reads at one close**: the frames of its markets cut there, as lists (no numpy needed).

    ``at_ms`` is the close; ``symbol`` the market it forecasts; ``anchor_ms`` the open of the bar that closed."""

    def __init__(self, answer: Dict[str, Any], symbol: str, anchor_ms: int):
        self.raw = answer
        self.at_ms: int = answer.get("at_ms")
        self.symbol = symbol
        self.anchor_ms = anchor_ms

    def bars(self, symbol: str, interval: str) -> List[List[float]]:
        """The closed bars, oldest first: ``[time_ms, open, high, low, close, tick_volume, spread_points]``."""
        return self.raw["markets"][symbol][interval]["rows"]

    def close(self, symbol: str, interval: str) -> List[float]:
        return [r[4] for r in self.bars(symbol, interval)]

    def np(self, symbol: str, interval: str):
        """The bars as a numpy array, where numpy is installed."""
        import numpy

        return numpy.asarray(self.bars(symbol, interval), dtype=float)


def model(name: str, interval: str, frames: Optional[Sequence[str]] = None, markets: Optional[Sequence[str]] = None,
          lookback: int = 200, steps: int = 1, form: str = "line",
          outputs: Optional[Sequence[Dict[str, Any]]] = None) -> Callable[[Callable], Callable]:
    """Declare the file's one model: its label ([a-z0-9_]{1,32}), the interval whose closes it runs at, the frames and
    markets of its window (the bench's market alone when omitted), the bars each frame holds, the steps it answers,
    and the chart type its forecast is drawn as.

    ``outputs`` declares outputs of any of the twelve kinds — each :meth:`Output.declare` or its dict — and
    the function then answers a dict by output name (``{"direction": 0.73, "regime": {"up": 0.6, ...}}``), each value
    in its kind's shape or its plain form; a name the declaration lacks is an error at the first call that returns it.
    A list is still a path's steps, and ``None`` says nothing."""
    declared = None if outputs is None else _declarations(outputs, "python")
    if declared is not None and not _label_ok(name):
        raise OutputRefused(f"the model label {_q(name)} is not [a-z0-9_]{{1,32}}; its outputs carry it")

    def wrap(fn: Callable) -> Callable:
        manifest = {"name": name, "interval": interval, "frames": list(frames) if frames else [interval],
                    "markets": list(markets) if markets else None, "lookback": int(lookback), "steps": int(steps),
                    "form": form}
        if declared is not None:
            manifest["outputs"] = declared
        _DECLARED.append({"fn": fn, "manifest": manifest})
        return fn

    return wrap


# ---- an ONNX model file for Bentrix's own engines -----------------------------------------------

#: The transforms a manifest may name — the ones Bentrix applies identically when it serves the model.
TRANSFORMS = ("raw", "returns", "zscore")


def export_manifest(name: str, interval: str, steps: Optional[int] = None, frames: Optional[Sequence[str]] = None,
                    markets: Optional[Sequence[str]] = None, lookback: int = 200, form: str = "line",
                    fields: Sequence[str] = ("close",), transform: str = "raw", output: str = "close",
                    pick: str = "first", outputs: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """**The manifest an ONNX model is served by** (``model_manifest``): what its window is — the frames, the markets,
    the closed bars each holds, the fields read in order and one transform — and how its output becomes a forecast's
    closes (``output`` ``close``, ``delta`` or ``return``; ``pick`` ``first`` or ``last``). Train on the windows
    ``extract_windows`` writes with the same fields and transform, and the model sees at serving what it saw training.
    It names no library, path or operator domain: a file is never a way to make Bentrix load code.

    **With ``outputs``** the manifest is schema 2: the model's answer is read as outputs
    of any of the twelve kinds, each :meth:`Output.declare` naming its kind, its shape, where in the answer it is
    (``slice`` ``[from, to)``; ``tensor``) and a suggested meaning; a ``path`` output carries its own ``steps``,
    ``form``, ``as_`` and ``pick``, so ``steps``, ``form``, ``output`` and ``pick`` are not given beside it. Without
    ``outputs`` the manifest is schema 1, exactly as before."""
    if transform not in TRANSFORMS:
        raise ValueError(f"{transform} is not a transform; one of {', '.join(TRANSFORMS)}")
    if outputs is not None:
        given = [k for k, v, default in (("steps", steps, None), ("form", form, "line"), ("output", output, "close"),
                                         ("pick", pick, "first")) if v != default]
        if given:
            raise OutputRefused(f"{', '.join(given)} belong to a schema-1 manifest; with outputs, a path output carries "
                                "its own (Output.declare(..., 'path', steps=..., form=..., as_=..., pick=...))")
        if not _label_ok(name):
            raise OutputRefused(f"the model label {_q(name)} is not [a-z0-9_]{{1,32}}")
        return {"schema_version": 2, "name": name, "interval": interval,
                "frames": list(frames) if frames else [interval], "markets": list(markets) if markets else None,
                "lookback": int(lookback), "input": {"fields": list(fields), "transform": transform},
                "outputs": _declarations(outputs, "onnx")}
    if steps is None:
        raise ValueError("a manifest names the steps its model answers (or declares outputs=[...])")
    if output not in ("close", "delta", "return") or pick not in ("first", "last"):
        raise ValueError("output is close, delta or return; pick is first or last")
    return {"schema_version": 1, "name": name, "interval": interval, "frames": list(frames) if frames else [interval],
            "markets": list(markets) if markets else None, "lookback": int(lookback), "steps": int(steps), "form": form,
            "input": {"fields": list(fields), "transform": transform}, "output": {"kind": output, "pick": pick}}


class OnnxModel:
    """**An ONNX model file, to be run by Bentrix itself.** ``Model.from_onnx(path, manifest)`` then ``install(door,
    name)`` — written into the person's models folder, where it waits for the person's tick on these bytes — and
    ``check(door)``, Bentrix's own reading of it: its manifest, its operator census, and whether it runs."""

    def __init__(self, path: str, manifest: Optional[Dict[str, Any]] = None):
        self.path, self.manifest, self.name = path, manifest, None

    def install(self, door: Any, name: str) -> Dict[str, Any]:
        """``model_install``: the file and its manifest written as ``name``; it waits for the person's tick."""
        import base64

        if self.manifest is None:
            raise ValueError("an ONNX model is installed with its manifest (export_manifest)")
        with open(self.path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        self.name = name
        return door.call("model_install", name=name, onnx_base64=data, manifest=self.manifest)

    def check(self, door: Any, name: Optional[str] = None) -> Dict[str, Any]:
        """``model_check`` on the installed model (or ``name``): the manifest, the census, the engine — or the refusal's
        sentence and offers."""
        n = name or self.name
        if n is None:
            raise ValueError("install it first, or name a model already in the folder")
        return door.call("model_check", file=n)

    def info(self, door: Any) -> Dict[str, Any]:
        """``model_info``: its consent, engine and operator census."""
        return door.call("model_info", name=self.name)


class Model:
    """Model files Bentrix runs. ``Model.from_onnx(path, manifest)`` — see :class:`OnnxModel`."""

    @staticmethod
    def from_onnx(path: str, manifest: Optional[Dict[str, Any]] = None) -> OnnxModel:
        return OnnxModel(path, manifest)


# ---- a model's outputs of any kind --------------------------------------------------------------------------
#
# The contract is Rust's
# (`crates/bentrix-forecast/src/output.rs`); what follows checks the same rules before a document leaves Python, so a
# person reads the refusal where they wrote the value. It lives in this module because the model host's package is this
# file and `modelhost.py` alone (`crates/bentrix-shell/src/modelhost.rs`, `write_package`): stdlib only.

import math as _math
import numbers as _numbers

#: **The twelve kinds**, in the order the documents list them (`OutputKind::ALL`).
KINDS = ("path", "scalar", "classes", "quantiles", "rank", "position", "weights", "event", "levels", "action", "text",
         "series")
#: The `output_spec` document's schema version (`OUTPUT_SCHEMA_VERSION`).
OUTPUT_SCHEMA_VERSION = 1
#: A `text` output's longest text, in characters.
TEXT_MAX = 2_000
#: A person's label for a meaning's word, longest.
LABEL_MAX = 24
#: Names in one output's object of numbers, most.
NAMES_MAX = 256
#: A model's or an output's label, longest.
MODEL_MAX = 32
#: The longest note, in bytes.
NOTE_MAX = 2_000
#: **The words a meaning may use** — the product's own, closed.
MEANING_WORDS = ("buy", "sell", "nothing", "up", "down", "flat", "in", "out", "close")
#: What a value answers.
READS = ("direction", "strength", "state", "size", "level", "event", "none")
#: An `action` value's intents.
INTENTS = ("buy", "sell", "close_buys", "close_sells", "nothing")

_U32, _U64, _I64 = 2 ** 32 - 1, 2 ** 64 - 1, 2 ** 63
_SPEC_FIELDS = ("schema_version", "model", "output", "symbol", "kind", "range", "units", "anchor", "horizon", "values",
                "names", "exclusive", "softmax", "persistent", "levels", "universe", "date_ms", "markets", "long_only",
                "gross", "epoch", "suggested_meaning", "look", "note")
#: The kind's shape fields a builder takes, in the order the document writes them after ``values``.
_SHAPE_FIELDS = ("names", "exclusive", "softmax", "persistent", "levels", "universe", "date_ms", "markets", "long_only",
                 "gross")
_MEANING_FIELDS = ("schema_version", "reads", "high_means", "low_means", "act_above", "act_below", "hysteresis",
                   "labels", "classes", "size", "gates", "step")
_STEP_FIELDS = ("open", "high", "low", "close", "volume", "bands", "quantiles", "samples", "expected_ms")


class OutputRefused(ValueError):
    """**An output, a declaration or a meaning the contract refuses** — the sentence names the field and why."""


def _q(x: Any) -> str:
    """A name quoted as the contract quotes it in a sentence: ``"profit"``."""
    import json

    return json.dumps(x, ensure_ascii=False) if isinstance(x, str) else repr(x)


def _bad(field: str, why: str) -> OutputRefused:
    return OutputRefused(f"{field} {why}")


def _is_num(x: Any) -> bool:
    return isinstance(x, _numbers.Real) and not isinstance(x, bool)


def _is_int(x: Any, lo: int, hi: int) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and lo <= x <= hi


def _fmt(v: float) -> str:
    """A number as the contract writes it in a sentence: ``1`` and ``0.5``, never ``1.0``."""
    v = float(v)
    return str(int(v)) if v.is_integer() and abs(v) < 1e16 else repr(v)


def _label_ok(s: Any) -> bool:
    return isinstance(s, str) and 0 < len(s) <= MODEL_MAX and all(c in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in s)


def _plain(x: Any) -> Any:
    """A value as JSON carries it: numpy's arrays and numbers become lists and floats, tuples lists."""
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        x = x.tolist()
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if x is None or isinstance(x, (bool, str, int)):
        return x
    if isinstance(x, _numbers.Integral):
        return int(x)
    if isinstance(x, _numbers.Real):
        return float(x)
    return x


def _in_range(field: str, v: Any, rng: Optional[Sequence[float]]) -> float:
    """A number checked: finite, and inside the declared range when there is one — **refused in words, never
    clamped** (`in_range`)."""
    if not _is_num(v):
        raise _bad(field, "is not a number")
    v = float(v)
    if not _math.isfinite(v):
        raise _bad(field, f"{v} is not a number")
    if rng is not None:
        lo, hi = rng
        if v < lo or v > hi:
            raise _bad(field, f"the output answers {_fmt(lo)} to {_fmt(hi)}; {_fmt(v)} is outside it")
    return v


def _numbers_of(field: str, v: Any, rng: Optional[Sequence[float]]) -> Dict[str, float]:
    if not isinstance(v, dict):
        raise _bad(field, "is not an object of numbers")
    if len(v) > NAMES_MAX:
        raise _bad(field, f"names {len(v)} entries; at most {NAMES_MAX}")
    return {k: _in_range(f"{field}.{k}", v[k], rng) for k in sorted(v)}


def _only(field: str, v: Any, allowed: Sequence[str], required: Sequence[str] = ()) -> Dict[str, Any]:
    """An object with only these fields (`deny_unknown_fields`), the required ones present."""
    if not isinstance(v, dict):
        raise _bad(field, "is not an object")
    for k in v:
        if k not in allowed:
            raise _bad(field, f"carries {_q(k)}, which is not one of its fields: {', '.join(allowed)}")
    for k in required:
        if k not in v or v[k] is None:
            raise _bad(field, f"has no {k}")
    return v


def _whole(field: str, v: Any, lo: int, hi: int, required: bool = False) -> None:
    if v is None and not required:
        return
    if not _is_int(v, lo, hi):
        raise _bad(field, f"is not a whole number from {lo} to {hi}")


def _typed(field: str, v: Any, t: type, what: str) -> None:
    if v is not None and (not isinstance(v, t) or (t is not bool and isinstance(v, bool))):
        raise _bad(field, f"is not {what}")


def _texts(field: str, v: Any) -> None:
    if v is not None and not (isinstance(v, list) and all(isinstance(s, str) for s in v)):
        raise _bad(field, "is not a list of texts")


def _range_of(field: str, r: Any) -> Optional[List[float]]:
    if r is None:
        return None
    if not (isinstance(r, (list, tuple)) and len(r) == 2 and all(_is_num(x) for x in r)):
        raise _bad(field, "is [low, high]")
    lo, hi = float(r[0]), float(r[1])
    if not (_math.isfinite(lo) and _math.isfinite(hi) and lo < hi):
        raise _bad(field, f"[{_fmt(lo)}, {_fmt(hi)}] is not a range")
    return [lo, hi]


def _horizon_of(field: str, h: Any) -> None:
    """A horizon: ``{steps[, interval]}``, ``{business_days}`` or ``{until: levels | next_output}``."""
    if h is None:
        return
    if isinstance(h, dict) and "steps" in h:
        _only(field, h, ("steps", "interval"))
        _whole(f"{field}.steps", h["steps"], 0, _U32, True)
        _typed(f"{field}.interval", h.get("interval"), str, "an interval's label")
        if h["steps"] == 0:
            raise _bad(f"{field}.steps", "is at least 1")
    elif isinstance(h, dict) and "business_days" in h:
        _only(field, h, ("business_days",))
        _whole(f"{field}.business_days", h["business_days"], 0, _U32, True)
        if h["business_days"] == 0:
            raise _bad(f"{field}.business_days", "is at least 1")
    elif isinstance(h, dict) and "until" in h:
        _only(field, h, ("until",))
        if h["until"] not in ("levels", "next_output"):
            raise _bad(f"{field}.until", f"{_q(h['until'])} is not levels or next_output")
    else:
        raise _bad(field, "is {steps}, {business_days} or {until}")


def _horizon_arg(h: Any) -> Any:
    """A builder's horizon: a number of steps, ``"next_output"`` / ``"levels"``, or the document's own form."""
    if h is None or isinstance(h, dict):
        return h
    if isinstance(h, str):
        return {"until": h}
    if _is_int(h, 0, _U32):
        return {"steps": h}
    raise OutputRefused("horizon is a number of steps, \"next_output\", \"levels\" or {steps}/{business_days}/{until}")


def check_meaning(m: Any, kind: str, rng: Optional[Sequence[float]]) -> None:
    """**A meaning checked against the output it reads** (`OutputMeaning::check`): its words the product's, its
    thresholds inside the range (refused in words naming the range), its labels short."""
    _only("the meaning", m, _MEANING_FIELDS)
    reads = m.get("reads", "none")
    if reads not in READS:
        raise OutputRefused(f"reads {_q(reads)} is not one of {', '.join(READS)}")
    if kind in ("text", "series", "path") and reads != "none":
        raise OutputRefused(f"a {kind} output carries no meaning")
    for w in (m.get("high_means"), m.get("low_means")):
        if w is not None and w not in MEANING_WORDS:
            raise OutputRefused(f"{_q(w)} is not one of the meaning's words: {', '.join(MEANING_WORDS)}")
    classes = m.get("classes") or {}
    if not isinstance(classes, dict):
        raise OutputRefused("the meaning's classes is an object of class: word")
    for c, w in classes.items():
        if w not in MEANING_WORDS:
            raise OutputRefused(f"the class {c}'s {_q(w)} is not one of the meaning's words: {', '.join(MEANING_WORDS)}")
    labels = m.get("labels") or {}
    if not isinstance(labels, dict):
        raise OutputRefused("the meaning's labels is an object of word: label")
    for k, label in labels.items():
        if k not in MEANING_WORDS:
            raise OutputRefused(f"a label for {_q(k)}, which is not one of the meaning's words")
        if not isinstance(label, str) or len(label) > LABEL_MAX or not label.strip():
            raise OutputRefused(f"the label {_q(label)} is empty or longer than {LABEL_MAX} characters")
    for name in ("act_above", "act_below"):
        t = m.get(name)
        if t is None:
            continue
        if not _is_num(t) or not _math.isfinite(float(t)):
            raise OutputRefused(f"{name} is not a number")
        if rng is not None and (t < rng[0] or t > rng[1]):
            raise OutputRefused(f"the output answers {_fmt(rng[0])} to {_fmt(rng[1])}; {_fmt(t)} is outside it")
    a, b = m.get("act_above"), m.get("act_below")
    if a is not None and b is not None and b > a:
        raise OutputRefused(f"act_below {_fmt(b)} is above act_above {_fmt(a)}")
    h = m.get("hysteresis", 0.0)
    if not (_is_num(h) and _math.isfinite(float(h)) and h >= 0):
        raise OutputRefused("the hysteresis is 0 or more")
    s = m.get("size")
    if s is not None:
        _only("the meaning's size", s, ("rule", "max", "floor"), ("rule",))
        if s["rule"] not in ("none", "linear", "stepped", "kelly_capped"):
            raise OutputRefused(f"the size rule {_q(s['rule'])} is not none, linear, stepped or kelly_capped")
        mx, fl = s.get("max", 1.0), s.get("floor", 0.0)
        if not (_is_num(mx) and _is_num(fl) and 0 < mx <= 1 and 0 <= fl <= 1):
            raise OutputRefused("a size's max is in (0, 1] and its floor in [0, 1]")
    g = m.get("gates")
    if g is not None:
        rest = g[len("output:"):] if isinstance(g, str) and g.startswith("output:") else None
        ok = g == "strategy" or (rest is not None and "." in rest and _label_ok(rest.split(".", 1)[0])
                                 and _label_ok(rest.split(".", 1)[1]))
        if not ok:
            raise OutputRefused(f"gates {_q(g)} is not strategy or output:<model>.<name>")
    step = m.get("step", 1)
    if not _is_int(step, 0, _U32):
        raise OutputRefused("the step that acts is a whole number")
    if step == 0:
        raise OutputRefused("the step that acts is 1 or more")


def _step_doc(i: int, v: Any) -> None:
    f = f"values[{i}]"
    _only(f, v, _STEP_FIELDS, ("close",))
    for k in ("open", "high", "low", "close", "volume"):
        if v.get(k) is not None and not _is_num(v[k]):
            raise _bad(f"{f}.{k}", "is not a number")
    for j, band in enumerate(v.get("bands") or []):
        _only(f"{f}.bands[{j}]", band, ("lo", "hi", "p"), ("lo", "hi"))
        if not (_is_num(band["lo"]) and _is_num(band["hi"])) or (band.get("p") is not None and not _is_num(band["p"])):
            raise _bad(f"{f}.bands[{j}]", "is lo, hi and p, numbers")
    q = v.get("quantiles")
    if q is not None and not (isinstance(q, dict) and all(_is_num(x) for x in q.values())):
        raise _bad(f"{f}.quantiles", "is not an object of numbers")
    if any(not _is_num(x) for x in v.get("samples") or []):
        raise _bad(f"{f}.samples", "is not a list of numbers")
    _whole(f"{f}.expected_ms", v.get("expected_ms"), -_I64, _I64 - 1)


def validate_output(spec: Any) -> Dict[str, Any]:
    """**An ``output_spec`` checked** by the rules ``output.rs``'s ``validate_output`` applies: the document's fields
    and no other (a put carrying ``put_ms`` is malformed — Bentrix stamps arrival), its labels, its kind's required
    fields, every value in its declared range (refused in words, never clamped), its horizon, its suggested meaning.
    Answers the document unchanged, or raises :class:`OutputRefused`."""
    if not isinstance(spec, dict):
        raise OutputRefused("an output_spec is an object")
    if "put_ms" in spec:
        raise OutputRefused("put_ms is Bentrix's to stamp, the instant the output arrives; a put never carries it")
    _only("the output_spec", spec, _SPEC_FIELDS,
          ("schema_version", "model", "output", "symbol", "kind", "anchor", "values"))
    if spec["schema_version"] != OUTPUT_SCHEMA_VERSION:
        raise _bad("schema_version", f"{spec['schema_version']} is not read; this release reads {OUTPUT_SCHEMA_VERSION}")
    if not _label_ok(spec["model"]):
        raise _bad("model", f"{_q(spec['model'])} is not [a-z0-9_]{{1,32}}")
    if not _label_ok(spec["output"]):
        raise _bad("output", f"{_q(spec['output'])} is not [a-z0-9_]{{1,32}}")
    _typed("symbol", spec["symbol"], str, "a text")
    kind = spec["kind"]
    if kind not in KINDS:
        raise _bad("kind", f"{_q(kind)} is not one of the twelve kinds: {', '.join(KINDS)}")
    anchor = _only("anchor", spec["anchor"], ("time_ms", "ordinal"), ("time_ms",))
    _whole("anchor.time_ms", anchor["time_ms"], -_I64, _I64 - 1, True)
    _whole("anchor.ordinal", anchor.get("ordinal"), 0, _U32)
    note = spec.get("note")
    _typed("note", note, str, "a text")
    if note is not None and len(note.encode("utf-8")) > NOTE_MAX:
        raise _bad("note", f"is longer than {NOTE_MAX}")
    _typed("units", spec.get("units"), str, "a text")
    for k in ("exclusive", "softmax", "persistent", "long_only"):
        _typed(k, spec.get(k), bool, "true or false")
    for k in ("names", "markets"):
        _texts(k, spec.get(k))
    _whole("universe", spec.get("universe"), 0, _U32)
    _whole("date_ms", spec.get("date_ms"), -_I64, _I64 - 1)
    _whole("epoch", spec.get("epoch"), 0, _U64)
    if spec.get("gross") is not None and not _is_num(spec["gross"]):
        raise _bad("gross", "is not a number")
    if spec.get("levels") is not None and not (isinstance(spec["levels"], list)
                                               and all(_is_num(x) for x in spec["levels"])):
        raise _bad("levels", "is not a list of numbers")
    look = spec.get("look")
    if look is not None:
        _only("look", look, ("rising", "falling", "band", "label"))
        for k, x in look.items():
            _typed(f"look.{k}", x, str, "a text")
    r = _range_of("range", spec.get("range"))
    vs = spec["values"]
    if not isinstance(vs, list):
        raise _bad("values", "is not a list")
    if not vs:
        raise _bad("values", "holds no value")
    if spec.get("suggested_meaning") is not None:
        try:
            check_meaning(spec["suggested_meaning"], kind, r)
        except OutputRefused as e:
            raise _bad("suggested_meaning", str(e)) from None
    _check_values(kind, vs, r, spec)
    _horizon_of("horizon", spec.get("horizon"))
    return spec


def _field(v: Any, k: str) -> Any:
    return v.get(k) if isinstance(v, dict) else None


def _check_values(kind: str, vs: List[Any], r: Optional[Sequence[float]], spec: Dict[str, Any]) -> None:
    """Each value in its kind's shape (the module doc of ``output.rs``), as ``validate_output`` reads it."""
    if kind == "path":
        for i, v in enumerate(vs):
            _step_doc(i, v)
    elif kind == "scalar":
        for i, v in enumerate(vs):
            _in_range(f"values[{i}].value", _field(v, "value"), r)
    elif kind == "classes":
        names = spec.get("names")
        if not names:
            raise _bad("names", "a classes output names its classes")
        for i, v in enumerate(vs):
            p = _numbers_of(f"values[{i}].p", _field(v, "p"), None if spec.get("softmax") is True else (0.0, 1.0))
            for k in p:
                if k not in names:
                    raise _bad(f"values[{i}].p", f"{k} is not one of the classes {', '.join(names)}")
    elif kind == "quantiles":
        levels = spec.get("levels")
        if not levels:
            raise _bad("levels", "a quantiles output names its levels")
        if any(not (0.0 < lv < 1.0) for lv in levels):
            raise _bad("levels", "each level is between 0 and 1")
        for i, v in enumerate(vs):
            _numbers_of(f"values[{i}].q", _field(v, "q"), r)
    elif kind == "rank":
        if spec.get("universe") == 0:
            raise _bad("universe", "a rank's universe holds at least one symbol")
        _in_range("values[0].value", _field(vs[0], "value"), r if r is not None else (0.0, 1.0))
    elif kind == "position":
        v = vs[0]
        if isinstance(v, dict) and "targets" in v:
            m = _numbers_of("values[0].targets", v["targets"], r)
            markets = spec.get("markets")
            if markets is not None:
                for k in m:
                    if k not in markets:
                        raise _bad("values[0].targets", f"{k} is not one of the markets named")
        else:
            _in_range("values[0].target", _field(v, "target"), r)
    elif kind == "weights":
        long_only = spec.get("long_only") is True
        w = _numbers_of("values[0].w", _field(vs[0], "w"), (0.0, 1.0) if long_only else None)
        if long_only:
            s = sum(w.values())
            if abs(s - 1.0) > 1e-6:
                raise _bad("values[0].w", f"long-only weights sum to 1; these sum to {_fmt(s)}")
        elif spec.get("gross") is not None:
            g, s = float(spec["gross"]), sum(abs(x) for x in w.values())
            if s > g + 1e-9:
                raise _bad("values[0].w", f"the gross bound is {_fmt(g)}; these hold {_fmt(s)}")
    elif kind == "event":
        for i, v in enumerate(vs):
            f = f"values[{i}]"
            _only(f, v, ("score", "within_steps", "expected_ms", "name"), ("score",))
            _whole(f"{f}.within_steps", v.get("within_steps"), 0, _U32)
            _whole(f"{f}.expected_ms", v.get("expected_ms"), -_I64, _I64 - 1)
            _typed(f"{f}.name", v.get("name"), str, "a text")
            _in_range(f"{f}.score", v["score"], (0.0, 1.0))
    elif kind == "levels":
        for i, v in enumerate(vs):
            f = f"values[{i}]"
            _only(f, v, ("levels", "p", "until_ms"), ("levels",))
            if not isinstance(v["levels"], dict):
                raise _bad(f"{f}.levels", "is not an object of numbers")
            if not v["levels"]:
                raise _bad(f"{f}.levels", "names no level")
            for k in sorted(v["levels"]):
                _in_range(f"{f}.levels.{k}", v["levels"][k], None)
            if v.get("p") is not None:
                if not isinstance(v["p"], dict):
                    raise _bad(f"{f}.p", "is not an object of numbers")
                for k in sorted(v["p"]):
                    _in_range(f"{f}.p.{k}", v["p"][k], (0.0, 1.0))
            _whole(f"{f}.until_ms", v.get("until_ms"), -_I64, _I64 - 1)
    elif kind == "action":
        for i, v in enumerate(vs):
            f = f"values[{i}]"
            _only(f, v, ("intent", "size", "size_units"), ("intent",))
            if v["intent"] not in INTENTS:
                raise _bad(f"{f}.intent", f"{_q(v['intent'])} is not {', '.join(INTENTS)}")
            _typed(f"{f}.size_units", v.get("size_units"), str, "a text")
            if v.get("size") is not None:
                s = _in_range(f"{f}.size", v["size"], None)
                if s < 0:
                    raise _bad(f"{f}.size", f"{_fmt(s)} is below 0; the intent says the side")
    elif kind == "text":
        t = _field(vs[0], "text")
        if not isinstance(t, str):
            raise _bad("values[0].text", "is not a text")
        if len(t) > TEXT_MAX:
            raise _bad("values[0].text", f"is longer than {TEXT_MAX} characters")
    elif kind == "series":
        names = spec.get("names")
        if not names:
            raise _bad("names", "a series output names its lines")
        for i, v in enumerate(vs):
            lines = _numbers_of(f"values[{i}].lines", _field(v, "lines"), r)
            for k in lines:
                if k not in names:
                    raise _bad(f"values[{i}].lines", f"{k} is not one of the lines {', '.join(names)}")


def _qkey(k: Any) -> str:
    """A quantile's key as the document writes it: ``"0.05"``."""
    return repr(float(k)) if _is_num(k) else str(k)


def _value_of(kind: str, x: Any) -> Any:
    """**One step's value in its kind's shape**, from the shape itself or from its plain form: a number for a
    ``scalar``, ``rank``, ``event`` or single-market ``position``; ``{name: number}`` for ``classes``, ``quantiles``,
    ``weights``, ``levels``, ``series`` and a ``position`` of several markets; an intent for an ``action``; a text for
    a ``text``."""
    if kind == "path":
        if _is_num(x):
            return {"close": float(x)}
        if isinstance(x, dict):
            return x
        raise OutputRefused(f"a path's step is a number or a dict, not {type(x).__name__}")
    if kind in ("scalar", "rank"):
        return {"value": x} if _is_num(x) else x
    if kind == "event":
        return {"score": x} if _is_num(x) else x
    if kind == "position":
        if _is_num(x):
            return {"target": x}
        return x if isinstance(x, dict) and ("target" in x or "targets" in x) else {"targets": x}
    if kind == "action":
        return {"intent": x} if isinstance(x, str) else x
    if kind == "text":
        return {"text": x} if isinstance(x, str) else x
    if kind == "quantiles":
        if isinstance(x, dict) and isinstance(x.get("q"), dict):
            return {**x, "q": {_qkey(k): v for k, v in x["q"].items()}}
        return {"q": {_qkey(k): v for k, v in x.items()}} if isinstance(x, dict) else x
    if kind == "levels":
        return x if isinstance(x, dict) and isinstance(x.get("levels"), dict) else {"levels": x}
    key = {"classes": "p", "weights": "w", "series": "lines"}[kind]
    return x if isinstance(x, dict) and set(x) == {key} and isinstance(x[key], dict) else {key: x}


def output_values(kind: str, got: Any) -> List[Any]:
    """**What a model answered for one output, as ``output_spec.values``**: a list is one value per step, anything
    else one value — each in the kind's shape or its plain form (:func:`_value_of`)."""
    if kind not in KINDS:
        raise OutputRefused(f"kind {_q(kind)} is not one of the twelve kinds: {', '.join(KINDS)}")
    got = _plain(got)
    return [_value_of(kind, x) for x in (got if isinstance(got, list) else [got])]


def _first(v: Any, key: str) -> Any:
    """The first step of a builder's values, unwrapped from its ``{key: ...}`` form."""
    first = v[0] if isinstance(v, (list, tuple)) and v else v
    return first[key] if isinstance(first, dict) and isinstance(first.get(key), dict) else first


class Output:
    """**An output of any of the twelve kinds**: each builder answers an ``output_spec``
    document, checked here by the contract's own rules — a value outside its declared range is refused, never clamped;
    a kind's names are present; ``put_ms`` is never set (Bentrix stamps the arrival).

    ``output`` names the output within the model ([a-z0-9_]{1,32}); omitted, the kind's own token. ``horizon`` is a
    number of steps, ``"next_output"``, ``"levels"`` or ``{"business_days": n}``.

    ::

        spec = bentrix.Output.scalar("dir_v4", "EURUSD", anchor_ms, value=0.73, range=(0, 1), units="probability",
                                     horizon=4, output="direction",
                                     suggested_meaning={"reads": "direction", "high_means": "buy",
                                                        "low_means": "sell", "act_above": 0.6, "act_below": 0.4})
    """

    @staticmethod
    def spec(kind: str, model: str, symbol: str, anchor_ms: int, values: Any, *, output: Optional[str] = None,
             ordinal: int = 0, range: Optional[Sequence[float]] = None, units: Optional[str] = None,
             horizon: Any = None, epoch: int = 0, suggested_meaning: Optional[Dict[str, Any]] = None,
             look: Optional[Dict[str, Any]] = None, note: Optional[str] = None, **shape: Any) -> Dict[str, Any]:
        """**Any kind's document**: ``values`` as :func:`output_values` reads them, ``shape`` the kind's fields
        (``names``, ``exclusive``, ``softmax``, ``persistent``, ``levels``, ``universe``, ``date_ms``, ``markets``,
        ``long_only``, ``gross``)."""
        unknown = sorted(k for k in shape if k not in _SHAPE_FIELDS)
        if unknown:
            raise OutputRefused(f"{', '.join(unknown)} is not a field of an output_spec")
        doc: Dict[str, Any] = {"schema_version": OUTPUT_SCHEMA_VERSION, "model": model,
                               "output": output if output is not None else kind, "symbol": symbol, "kind": kind}
        if range is not None:
            doc["range"] = _plain(range)
        if units is not None:
            doc["units"] = units
        doc["anchor"] = {"time_ms": anchor_ms, "ordinal": ordinal} if ordinal else {"time_ms": anchor_ms}
        if horizon is not None:
            doc["horizon"] = _horizon_arg(horizon)
        doc["values"] = output_values(kind, values)
        for k in _SHAPE_FIELDS:
            if shape.get(k) is not None:
                doc[k] = _plain(shape[k])
        if epoch:
            doc["epoch"] = epoch
        if suggested_meaning is not None:
            doc["suggested_meaning"] = _plain(suggested_meaning)
        if look is not None:
            doc["look"] = look
        if note is not None:
            doc["note"] = note
        return validate_output(doc)

    @staticmethod
    def path(model: str, symbol: str, anchor_ms: int, steps: Sequence[Any], **kw: Any) -> Dict[str, Any]:
        """Today's candle forecast as an output: each step a number (the close) or a ``forecast_spec`` step."""
        return Output.spec("path", model, symbol, anchor_ms, list(steps), **kw)

    @staticmethod
    def scalar(model: str, symbol: str, anchor_ms: int, value: Any, **kw: Any) -> Dict[str, Any]:
        """One number per step — a probability, a strength, a return, a price; ``value`` a number or one per step;
        ``range=(lo, hi)`` and ``units`` its declaration."""
        return Output.spec("scalar", model, symbol, anchor_ms, value, **kw)

    @staticmethod
    def classes(model: str, symbol: str, anchor_ms: int, p: Any, *, exclusive: bool,
                names: Optional[Sequence[str]] = None, **kw: Any) -> Dict[str, Any]:
        """Probabilities over named classes: ``p`` ``{class: probability}`` (or one per step); ``names`` defaults to
        the first step's classes, in their order; ``softmax=True`` when the model answers scores, ``persistent=True``
        for a state that holds until replaced."""
        first = _first(p, "p")
        names = list(names) if names is not None else (list(first) if isinstance(first, dict) else None)
        return Output.spec("classes", model, symbol, anchor_ms, p, names=names, exclusive=exclusive, **kw)

    @staticmethod
    def quantiles(model: str, symbol: str, anchor_ms: int, q: Any, *, levels: Optional[Sequence[float]] = None,
                  **kw: Any) -> Dict[str, Any]:
        """Quantiles of a value per step: ``q`` ``{level: value}`` (or one per step); ``levels`` defaults to the first
        step's, ascending."""
        first = _first(q, "q")
        if levels is None and isinstance(first, dict):
            levels = sorted(float(k) for k in first)
        return Output.spec("quantiles", model, symbol, anchor_ms, q, levels=levels, **kw)

    @staticmethod
    def rank(model: str, symbol: str, anchor_ms: int, value: Any, *, universe: int, **kw: Any) -> Dict[str, Any]:
        """A symbol's rank against a universe of ``universe`` symbols (on ``date_ms``); ``[0, 1]`` unless ``range``
        says otherwise."""
        return Output.spec("rank", model, symbol, anchor_ms, value, universe=universe, **kw)

    @staticmethod
    def position(model: str, symbol: str, anchor_ms: int, target: Any, **kw: Any) -> Dict[str, Any]:
        """A target exposure: a number (this market's) or ``{market: target}`` with ``markets``; ``units`` fraction,
        lots or risk."""
        return Output.spec("position", model, symbol, anchor_ms, target, **kw)

    @staticmethod
    def weights(model: str, symbol: str, anchor_ms: int, w: Dict[str, float], **kw: Any) -> Dict[str, Any]:
        """Portfolio weights ``{market: weight}``: ``long_only=True`` (each in [0, 1], summing to 1) or under a
        ``gross`` bound."""
        return Output.spec("weights", model, symbol, anchor_ms, w, **kw)

    @staticmethod
    def event(model: str, symbol: str, anchor_ms: int, score: Any, *, within_steps: Optional[int] = None,
              expected_ms: Optional[int] = None, name: Optional[str] = None, **kw: Any) -> Dict[str, Any]:
        """An event's score in [0, 1]: something is about to happen — within ``within_steps``, or at
        ``expected_ms``."""
        extra = {k: v for k, v in (("within_steps", within_steps), ("expected_ms", expected_ms), ("name", name))
                 if v is not None}
        values = [_value_of("event", s) for s in (score if isinstance(score, (list, tuple)) else [score])]
        return Output.spec("event", model, symbol, anchor_ms,
                           [{**v, **extra} if isinstance(v, dict) else v for v in values], **kw)

    @staticmethod
    def levels(model: str, symbol: str, anchor_ms: int, levels: Optional[Dict[str, float]] = None, *,
               p: Optional[Dict[str, float]] = None, until_ms: Optional[int] = None, output: Optional[str] = None,
               horizon: Any = None, epoch: int = 0, suggested_meaning: Optional[Dict[str, Any]] = None,
               look: Optional[Dict[str, Any]] = None, note: Optional[str] = None, ordinal: int = 0,
               **named: float) -> Dict[str, Any]:
        """Named price levels — ``levels(..., stop=1.09, target=1.12)`` or ``levels={...}`` — with their
        probabilities ``p`` and the instant they stand until."""
        value: Dict[str, Any] = {"levels": {**(levels or {}), **named}}
        if p is not None:
            value["p"] = p
        if until_ms is not None:
            value["until_ms"] = until_ms
        return Output.spec("levels", model, symbol, anchor_ms, value, output=output, horizon=horizon, epoch=epoch,
                           suggested_meaning=suggested_meaning, look=look, note=note, ordinal=ordinal)

    @staticmethod
    def action(model: str, symbol: str, anchor_ms: int, intent: str, *, size: Optional[float] = None,
               size_units: Optional[str] = None, **kw: Any) -> Dict[str, Any]:
        """A policy's decision — buy, sell, close_buys, close_sells or nothing — with its size."""
        value: Dict[str, Any] = {"intent": intent}
        if size is not None:
            value["size"] = size
        if size_units is not None:
            value["size_units"] = size_units
        return Output.spec("action", model, symbol, anchor_ms, value, **kw)

    @staticmethod
    def text(model: str, symbol: str, anchor_ms: int, text: str, **kw: Any) -> Dict[str, Any]:
        """A text of at most 2000 characters — on hover and in the data window, never on the canvas."""
        return Output.spec("text", model, symbol, anchor_ms, text, **kw)

    @staticmethod
    def series(model: str, symbol: str, anchor_ms: int, lines: Any, *, names: Optional[Sequence[str]] = None,
               **kw: Any) -> Dict[str, Any]:
        """Named lines with no meaning of their own: ``{line: value}`` (or one per step); ``names`` defaults to the
        first step's lines."""
        first = _first(lines, "lines")
        names = list(names) if names is not None else (list(first) if isinstance(first, dict) else None)
        return Output.spec("series", model, symbol, anchor_ms, lines, names=names, **kw)

    @staticmethod
    def declare(name: str, kind: str, **fields: Any) -> Dict[str, Any]:
        """**One output a model declares** (a manifest's ``outputs[]``): its name and kind, and as
        the kind needs ``range``, ``units``, ``horizon``, ``names``, ``exclusive``, ``softmax``, ``persistent``,
        ``levels``, ``universe``, ``markets``, ``long_only``, ``gross`` and a ``suggested_meaning`` (the person's own
        replaces it). A ``path`` names ``steps`` and ``form``; an ONNX model's outputs also say where in its answer
        they are (``tensor``, ``slice`` ``[from, to)``) and a path what its values are (``as_`` close, delta or return;
        ``pick`` first or last)."""
        if "as_" in fields:
            fields["as"] = fields.pop("as_")
        return _declaration({"name": name, "kind": kind, **fields}, None)

    #: The document checked by the contract's rules (raises :class:`OutputRefused`).
    check = staticmethod(validate_output)


#: **A declaration's fields** — `ManifestOutput`'s, in the order a manifest writes them.
_DECL_FIELDS = ("name", "kind", "form", "steps", "as", "pick", "range", "units", "horizon", "names", "exclusive",
                "softmax", "persistent", "levels", "universe", "markets", "long_only", "gross", "tensor", "slice",
                "suggested_meaning")
#: The declaration's fields an output_spec carries as they are.
_CARRIED = ("range", "units", "names", "exclusive", "softmax", "persistent", "levels", "universe", "markets",
            "long_only", "gross")


def _declaration(d: Any, where: Optional[str]) -> Dict[str, Any]:
    """**One declared output, checked and written in a manifest's order.** ``where`` is ``"python"`` (a model run by
    the Python host answers by name, so it carries no ``tensor``, ``slice``, ``as`` or ``pick``), ``"onnx"`` (a model
    Bentrix's engines run: what reading its numbers needs, as ``bentrix_model``'s ``check_output`` asks, and a path's
    ``form``, ``as`` and ``pick`` written out) or ``None`` (decided where it is used)."""
    if not isinstance(d, dict):
        raise OutputRefused("a declared output is an object (bentrix.Output.declare)")
    d = {k: v for k, v in d.items() if v is not None}
    for k in d:
        if k not in _DECL_FIELDS:
            raise OutputRefused(f"a declared output carries {_q(k)}, which is not one of its fields: "
                                f"{', '.join(_DECL_FIELDS)}")
    name, kind = d.get("name"), d.get("kind")
    if not _label_ok(name):
        raise OutputRefused(f"the output name {_q(name)} is not [a-z0-9_]{{1,32}}")
    if kind not in KINDS:
        raise OutputRefused(f"the output {name}'s kind {_q(kind)} is not one of the twelve kinds: {', '.join(KINDS)}")
    if "horizon" in d:
        d["horizon"] = _horizon_arg(d["horizon"])
    if where == "onnx" and kind == "path":
        for k, v in (("form", "line"), ("as", "close"), ("pick", "first")):
            d.setdefault(k, v)
    d = {k: _plain(d[k]) for k in _DECL_FIELDS if k in d}
    try:
        _check_declaration(d, kind, where)
    except OutputRefused as e:
        raise OutputRefused(f"the output {name}: {e}") from None
    return d


def _check_declaration(d: Dict[str, Any], kind: str, where: Optional[str]) -> None:
    r = _range_of("range", d.get("range"))
    _typed("units", d.get("units"), str, "a text")
    _horizon_of("horizon", d.get("horizon"))
    for k in ("exclusive", "softmax", "persistent", "long_only"):
        _typed(k, d.get(k), bool, "true or false")
    for k in ("names", "markets"):
        _texts(k, d.get(k))
    if d.get("levels") is not None:
        if not (isinstance(d["levels"], list) and all(_is_num(x) for x in d["levels"])):
            raise _bad("levels", "is not a list of numbers")
        if any(not (0.0 < x < 1.0) for x in d["levels"]):
            raise _bad("levels", "each level is between 0 and 1")
    _whole("universe", d.get("universe"), 1, _U32)
    if d.get("gross") is not None and not _is_num(d["gross"]):
        raise _bad("gross", "is not a number")
    _typed("tensor", d.get("tensor"), str, "a text")
    sl = d.get("slice")
    if sl is not None and not (isinstance(sl, list) and len(sl) == 2 and all(_is_int(x, 0, _U32) for x in sl)
                               and sl[0] < sl[1]):
        raise _bad("slice", f"{sl} is not [from, to) with from below to")
    if d.get("suggested_meaning") is not None:
        try:
            check_meaning(d["suggested_meaning"], kind, r)
        except OutputRefused as e:
            raise _bad("suggested_meaning", str(e)) from None
    if kind == "path":
        _whole("steps", d.get("steps"), 1, _U32)
        _typed("form", d.get("form"), str, "a chart type's token")
        if d.get("as", "close") not in ("close", "delta", "return") or d.get("pick", "first") not in ("first", "last"):
            raise _bad("as", "is close, delta or return, and pick first or last")
    else:
        extra = [k for k in ("form", "steps", "as", "pick") if k in d]
        if extra:
            raise _bad(extra[0], "belongs to a path output only")
    if kind == "classes" and not d.get("names"):
        raise _bad("names", "a classes output names its classes")
    if kind == "series" and not d.get("names"):
        raise _bad("names", "a series output names its lines")
    if kind == "quantiles" and not d.get("levels"):
        raise _bad("levels", "a quantiles output names its levels")
    if where == "python":
        extra = [k for k in ("tensor", "slice", "as", "pick") if k in d]
        if extra:
            raise _bad(extra[0], "belongs to an ONNX model's manifest; a Python model answers its outputs by name")
    if where == "onnx":
        if kind == "text":
            raise _bad("kind", "text comes from a program (output_put), never from a model's numbers")
        if kind == "path" and "steps" not in d:
            raise _bad("steps", "a path output names its steps")
        if kind == "levels" and not d.get("names"):
            raise _bad("names", "a levels output names its levels, in the answer's order")
        if kind == "action":
            if not d.get("names"):
                raise _bad("names", "an action output names its intents, in the answer's order")
            if any(n not in INTENTS for n in d["names"]):
                raise _bad("names", f"an action output's names are intents: {', '.join(INTENTS)}")
        if kind == "weights" and not d.get("markets"):
            raise _bad("markets", "a weights output names its markets, in the answer's order")


def _declarations(outputs: Any, where: str) -> List[Dict[str, Any]]:
    """Every declared output checked, its name unique, at least one."""
    if isinstance(outputs, (dict, str)) or not isinstance(outputs, (list, tuple)):
        raise OutputRefused("outputs is a list of declared outputs (bentrix.Output.declare)")
    out = [_declaration(o, where) for o in outputs]
    if not out:
        raise OutputRefused("outputs declares no output; leave it out for a model that answers a path's steps")
    seen = set()
    for o in out:
        if o["name"] in seen:
            raise OutputRefused(f"the output {o['name']} is declared twice")
        seen.add(o["name"])
    return out


def answer_outputs(manifest: Dict[str, Any], got: Dict[str, Any], symbol: str, anchor_ms: int) -> Dict[str, Any]:
    """**A model's dict answer, as the host's ``outputs``**: ``{name: {"kind": kind, "values":
    [...]}}``, each output's values in its kind's shape, checked by the contract against its declaration — a name the
    declaration lacks is refused in words, a value outside its declared range refused, never clamped. An output
    answered ``None`` says nothing at this close."""
    declared = {o["name"]: o for o in manifest.get("outputs") or []}
    for name in got:
        if name not in declared:
            has = ", ".join(declared) if declared else "no outputs (return a list of steps, or declare outputs=[...])"
            raise OutputRefused(f"the model answered an output named {_q(name)}, which its @bentrix.model does not "
                                f"declare; it declares {has}")
    out: Dict[str, Any] = {}
    for name, v in got.items():
        if v is None:
            continue
        o = declared[name]
        values = output_values(o["kind"], v)
        spec = {"schema_version": OUTPUT_SCHEMA_VERSION, "model": manifest["name"], "output": name, "symbol": symbol,
                "kind": o["kind"], "anchor": {"time_ms": anchor_ms}, "values": values,
                **{k: o[k] for k in _CARRIED if k in o}}
        try:
            validate_output(spec)
        except OutputRefused as e:
            raise OutputRefused(f"the output {name}: {e}") from None
        out[name] = {"kind": o["kind"], "values": values}
    return out
