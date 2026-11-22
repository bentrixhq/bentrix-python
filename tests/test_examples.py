"""The examples: what they call exists in the generated client, and the hosted ones run.

**A census over the examples' own calls**: every ``b.<group>.<method>(...)`` names a method the client
generated from the catalogue has, with keywords it takes — so an example cannot drift from the
catalogue unseen. The hosted examples load in the study host and declare their spec; the coloured one
answers its colours beside its values.
"""

import ast
import inspect
import json
import os

import pytest

from bentrix import _generated as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
PUSHED = ["pushed_rsi.py", "agent_drawings.py"]
HOSTED = ["my_sma.py", "my_rsi.py", "coloured_trend.py"]


def calls(path):
    """Every ``b.<group>.<method>(…)`` in the file: ``(group, method, keywords, positional count)``."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        group = node.func.value
        if isinstance(group, ast.Attribute) and isinstance(group.value, ast.Name) and group.value.id == "b":
            out.append((group.attr, node.func.attr, [k.arg for k in node.keywords], len(node.args)))
    return out


def test_every_call_an_example_makes_is_in_the_generated_client():
    """Mutation (2026-10-29): `bars_get` renamed `bars_read` in an example — red, *names no method*."""
    seen = 0
    for name in PUSHED:
        for group, method, keywords, positional in calls(os.path.join(EXAMPLES, name)):
            cls = getattr(G, "_" + group.capitalize() + "Api", None)
            assert cls is not None, "%s: b.%s is no group of the client" % (name, group)
            fn = getattr(cls, method, None)
            assert fn is not None, "%s: b.%s.%s names no method" % (name, group, method)
            params = inspect.signature(fn).parameters
            for k in keywords:
                assert k in params, "%s: b.%s.%s takes no %s" % (name, group, method, k)
            required = [p for p in list(params.values())[1:] if p.default is inspect.Parameter.empty and p.kind == p.POSITIONAL_OR_KEYWORD]
            assert positional + len([k for k in keywords if k in [p.name for p in required]]) >= len(required), "%s: b.%s.%s misses an argument" % (name, group, method)
            seen += 1
    assert seen >= 8, "the census read too few calls: %d" % seen


def test_the_hosted_examples_load_and_declare_their_spec():
    """Each hosted example declares exactly one study with an id, a family and its lines."""
    pytest.importorskip("numpy")
    from bentrix import host, study

    for name in HOSTED:
        s = host.load(os.path.join(EXAMPLES, name))
        assert s.spec["id"] and s.spec["family"] in ("overlay", "oscillator") and s.spec["lines"], name
    assert study.REGISTERED[-1].spec["bar_colors"] == [{}], "the coloured example declares no bar colours"


def test_the_coloured_example_answers_its_colours_through_the_host(tmp_path):
    """The coloured example's values and colours travel over the host's pipe, bar by bar."""
    pytest.importorskip("numpy")
    from test_study import Pipe, _columns

    closes = [1.0 + 0.01 * i + 0.05 * ((i * 7) % 5) for i in range(200)]
    cols = tmp_path / "columns-1.bin"
    _columns(cols, closes)
    h = Pipe(os.path.join(EXAMPLES, "coloured_trend.py"))
    assert h.ask({"op": "describe"})["spec"]["id"] == "coloured_trend"
    h.ask({"op": "open", "epoch": 1, "columns": str(cols), "len": 200, "point": 0.01, "tf": "M1", "inputs": {"period": 5}})
    v = h.ask({"op": "compute", "epoch": 1, "len": 200})
    up = 0x26A69AFF
    colours = v["colors"]["ema"]["inline"]
    assert len(colours) == 200 and up in colours, json.dumps(colours[:12])
    assert set(v["colors"]) == {"ema", "bar_color:0"}
    code, _, _ = h.finish()
    assert code == 0
