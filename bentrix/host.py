"""The study host: ``python -I -m bentrix.host <script.py>``, started by Bentrix and nothing else.

It loads the script, which declares exactly one ``@study``, and answers Bentrix on stdin/stdout, one
JSON object per line (the protocol is ``bentrix-pystudy``'s ``protocol.rs``):

* ``describe`` -> ``{"op": "spec", "spec": ...}`` — the decorator's declaration.
* ``open`` -> ``{"op": "opened", "epoch", "len", "last_time", "lookback"}`` — the column file is mapped
  (``time i64, open, high, low, close, volume f64``, little-endian, column-major) and what was read is
  said back, with the lookback for these inputs.
* ``compute`` -> ``{"op": "values", "epoch", "len", "lines": {name: {"inline": [...]} | {"file": path}}}``
  — the function over the whole mapped series; a line of 50,000 values or more is written to a file
  beside the columns instead of inline. A study answering ``(lines, colors)`` adds ``"colors"``, the
  same shape by channel.
* ``advance`` -> ``{"op": "tail", "epoch", "len", "tail": {name: [...]}}`` — the function over the tail
  it was handed, answered for that tail only, with ``"colors": {channel: [...]}`` where it colours.
* ``close`` -> the host exits.

A value that is not a number travels as ``null``. **An error is not an answer**: the traceback goes to
stderr and the host exits non-zero, and Bentrix shows the exception's line.

``print()`` in a script goes to stderr: stdout is the protocol's alone, and a stray line there would
read as a broken answer.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from . import study as _study

INLINE_MAX = 50_000


def _wire(values) -> List[Optional[float]]:
    return [float(v) if math.isfinite(v) else None for v in (float(x) for x in values)]


def load(script: str) -> _study.Study:
    """Import ``script`` under a name of its own and answer the one study it declares."""
    before = len(_study.REGISTERED)
    # The script's own folder, at the END of the path: a helper module beside it imports, and never
    # shadows the standard library (`-I` keeps the folder off the path's front).
    folder = os.path.dirname(os.path.abspath(script))
    if folder not in sys.path:
        sys.path.append(folder)
    name = "bentrix_user_study_" + "".join(c if c.isalnum() else "_" for c in os.path.splitext(os.path.basename(script))[0])
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise ImportError("%s is not a Python file" % script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    made = _study.REGISTERED[before:]
    if len(made) != 1:
        raise RuntimeError("%s declares %d studies; a script declares exactly one @study" % (script, len(made)))
    return made[0]


class Session:
    """One script's state across requests."""

    def __init__(self, s: _study.Study) -> None:
        self.study = s
        self.epoch: Optional[int] = None
        self.inputs: Dict[str, Any] = {}
        self.columns: Optional[Dict[str, Any]] = None
        self.path: Optional[str] = None
        self.len = 0
        self.point = 0.0
        self.tf = ""

    def _answer(self, result: Any, length: int):
        """The function's answer split into its lines and its colours — a dict of lines, or the pair
        ``(lines, colors)`` — each held to the declaration."""
        colors: Dict[str, Any] = {}
        if isinstance(result, tuple):
            if len(result) != 2:
                raise TypeError("%s answered a tuple of %d; a study answers lines, or (lines, colors)" % (self.study.spec["id"], len(result)))
            result, colors = result
        return self._lines(result, length), self._colors(colors, length)

    def _colors(self, colors: Any, length: int) -> Dict[str, Any]:
        """Colours by channel: each a declared channel, exactly ``length`` long, each value
        ``0xRRGGBBAA`` or ``NONE``."""
        if not isinstance(colors, dict):
            raise TypeError("%s answered colours as %s; colours are a dict by channel" % (self.study.spec["id"], type(colors).__name__))
        declared = _study.channels(self.study.spec)
        extra = sorted(set(colors) - set(declared))
        if extra:
            raise ValueError("colours for channels not declared: %s (declared: %s)" % (extra, declared))
        np = _study.np
        out = {}
        for name in declared:
            if name not in colors:
                continue
            v = np.asarray(colors[name], dtype=np.float64).ravel()
            if len(v) != length:
                raise ValueError("channel %r has %d colours for %d bars" % (name, len(v), length))
            known = v[~np.isnan(v)]
            if np.any((known < 0) | (known > 0xFFFFFFFF) | (known != np.floor(known))):
                raise ValueError("channel %r carries a value that is not a colour; use rgba(r, g, b, a)" % name)
            out[name] = v
        return out

    def _lines(self, result: Any, length: int) -> Dict[str, Any]:
        """The function's answer, held to the declaration: every declared line, nothing else, each
        exactly ``length`` long."""
        if not isinstance(result, dict):
            raise TypeError("%s answered %s; a study answers a dict of lines" % (self.study.spec["id"], type(result).__name__))
        declared = [l["name"] for l in self.study.spec["lines"]]
        extra = sorted(set(result) - set(declared))
        missing = [d for d in declared if d not in result]
        if extra or missing:
            raise ValueError("lines answered do not match those declared (undeclared: %s; missing: %s)" % (extra, missing))
        np = _study.np
        out = {}
        for name in declared:
            v = np.asarray(result[name], dtype=np.float64).ravel()
            if len(v) != length:
                raise ValueError("line %r has %d values for %d bars" % (name, len(v), length))
            out[name] = v
        return out

    def open(self, req: Dict[str, Any]) -> Dict[str, Any]:
        np = _study.np
        if np is None:
            raise ImportError("the study host needs numpy (pip install numpy)")
        n = int(req["len"])
        path = req["columns"]
        size = os.path.getsize(path)
        if size != 48 * n:
            raise ValueError("%s holds %d bytes; %d bars are %d" % (path, size, n, 48 * n))
        self.columns = None  # let the previous epoch's mapping go before the next is made
        cols: Dict[str, Any] = {}
        if n > 0:
            cols["time"] = np.memmap(path, dtype="<i8", mode="r", offset=0, shape=(n,))
            for i, name in enumerate(_study.COLUMNS[1:], start=1):
                cols[name] = np.memmap(path, dtype="<f8", mode="r", offset=8 * n * i, shape=(n,))
        else:
            cols["time"] = np.zeros(0, dtype="<i8")
            for name in _study.COLUMNS[1:]:
                cols[name] = np.zeros(0, dtype="<f8")
        self.columns = cols
        self.path = path
        self.epoch = int(req["epoch"])
        self.len = n
        self.point = float(req.get("point", 0.0))
        self.tf = str(req.get("tf", ""))
        self.inputs = self.study.resolve(req.get("inputs") or {})
        last = int(cols["time"][-1]) if n > 0 else None
        return {"op": "opened", "epoch": self.epoch, "len": n, "last_time": last, "lookback": self.study.lookback(self.inputs)}

    def compute(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if self.columns is None:
            raise RuntimeError("compute before open")
        n = min(int(req["len"]), self.len)
        bars = _study.Bars({k: v[:n] for k, v in self.columns.items()})
        lines, colors = self._answer(self.study.func(bars, dict(self.inputs)), n)

        def series(prefix, named):
            out = {}
            for i, (name, v) in enumerate(named.items()):
                if n >= INLINE_MAX:
                    file = os.path.join(os.path.dirname(self.path), "%s-%d-%d.bin" % (prefix, self.epoch, i))
                    v.astype("<f8").tofile(file)
                    out[name] = {"file": file}
                else:
                    out[name] = {"inline": _wire(v)}
            return out

        answer = {"op": "values", "epoch": self.epoch, "len": n, "lines": series("values", lines)}
        if colors:
            answer["colors"] = series("colors", colors)
        return answer

    def advance(self, req: Dict[str, Any]) -> Dict[str, Any]:
        np = _study.np
        if self.epoch is None:
            raise RuntimeError("advance before open")
        tail = req["tail"]
        cols = {
            "time": np.array([int(b[0]) for b in tail], dtype=np.int64),
        }
        for i, name in enumerate(_study.COLUMNS[1:], start=1):
            cols[name] = np.array([float(b[i]) for b in tail], dtype=np.float64)
        for v in cols.values():
            v.flags.writeable = False
        lines, colors = self._answer(self.study.func(_study.Bars(cols), dict(self.inputs)), len(tail))
        answer = {"op": "tail", "epoch": self.epoch, "len": int(req["len"]), "tail": {k: _wire(v) for k, v in lines.items()}}
        if colors:
            answer["colors"] = {k: _wire(v) for k, v in colors.items()}
        return answer


def serve(script: str, stdin, proto) -> int:
    s = load(script)
    session = Session(s)
    while True:
        line = stdin.readline()
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        op = req.get("op")
        if op == "close":
            return 0
        if op == "describe":
            answer = {"op": "spec", "spec": s.spec}
        elif op == "open":
            answer = session.open(req)
        elif op == "compute":
            answer = session.compute(req)
        elif op == "advance":
            answer = session.advance(req)
        else:
            raise ValueError("unknown request %r" % op)
        proto.write(json.dumps(answer, allow_nan=False) + "\n")
        proto.flush()


def main(argv: List[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("usage: python -I -m bentrix.host <script.py>\n")
        return 2
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", newline="\n" if stream is not sys.stdin else None, errors="backslashreplace" if stream is sys.stderr else "strict")
        except (AttributeError, ValueError):
            pass
    proto = sys.stdout
    sys.stdout = sys.stderr  # a script's print() never reaches the protocol
    try:
        return serve(argv[0], sys.stdin, proto)
    except Exception:  # noqa: BLE001 - every failure is said the same way: a traceback and an exit
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
