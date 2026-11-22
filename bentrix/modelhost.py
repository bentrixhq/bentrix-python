"""The model host: ``python -I`` on one model file, started by Bentrix and nothing else.

One JSON object per line on stdin and stdout:

* ``{"op": "describe"}`` -> ``{"op": "manifest", "manifest": {...}}`` — the file's one ``@model``.
* ``{"op": "predict", "window": {...}, "symbol": s, "anchor_ms": t}`` -> ``{"op": "steps", "steps": [...]}`` — the
  function over the window; ``"steps": null`` when it said nothing at this close. A function returning a **dict by
  output name** (a model declaring ``outputs=[...]``) answers ``{"op": "outputs", "outputs": {name: {"kind":
  k, "values": [...]}}}``, each output's values as ``output_spec.values`` carries them; a name its declaration lacks
  is an error, in words.
* ``{"op": "close"}`` -> the host exits.

**An error is not an answer**: the traceback goes to stderr and the host exits non-zero. ``print()`` goes to stderr.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import traceback
from typing import Any, List

from importlib import import_module

# The module itself: the package's `model` attribute is the decorator, the module's own name stays in sys.modules.
_model = import_module(__package__ + ".model")


def _step(s: Any) -> Any:
    if isinstance(s, (int, float)):
        if not math.isfinite(float(s)):
            raise ValueError(f"a step's close is not a number: {s!r}")
        return {"close": float(s)}
    if isinstance(s, dict):
        return s
    raise TypeError(f"a step is a number or a dict, not {type(s).__name__}")


def main(argv: List[str]) -> int:
    if len(argv) != 1:
        print("usage: modelhost <model.py>", file=sys.stderr)
        return 2
    out = sys.stdout
    sys.stdout = sys.stderr
    try:
        spec = importlib.util.spec_from_file_location("bentrix_model_file", argv[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if len(_model._DECLARED) != 1:
            print(f"the file declares {len(_model._DECLARED)} @model, not one", file=sys.stderr)
            return 3
        declared = _model._DECLARED[0]
        for line in sys.stdin:
            if not line.strip():
                continue
            req = json.loads(line)
            op = req.get("op")
            if op == "describe":
                answer = {"op": "manifest", "manifest": declared["manifest"]}
            elif op == "predict":
                w = _model.Window(req["window"], req["symbol"], req["anchor_ms"])
                got = declared["fn"](w)
                if isinstance(got, dict):
                    answer = {"op": "outputs", "outputs": _model.answer_outputs(declared["manifest"], got, req["symbol"],
                                                                                req["anchor_ms"])}
                else:
                    answer = {"op": "steps", "steps": None if got is None else [_step(s) for s in got]}
            elif op == "close":
                return 0
            else:
                print(f"an unknown request: {op!r}", file=sys.stderr)
                return 4
            out.write(json.dumps(answer) + "\n")
            out.flush()
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
