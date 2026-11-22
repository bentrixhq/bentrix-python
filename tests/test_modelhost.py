"""**The Python model host answers outputs of any kind**: a model function returning a
list is a ``path`` — today's ``steps`` answer, byte for byte — one returning a dict by output name answers
``{"op": "outputs", ...}``, and ``None`` says nothing.

Every walk here starts the host exactly as Bentrix does: the bootstrap and the package's ``__init__`` are read from
``crates/bentrix-shell/src/modelhost.rs`` (``BOOTSTRAP``, ``write_package``), the package's ``model.py`` and
``modelhost.py`` are this SDK's, and the interpreter runs isolated (``-I``) on a model file in a folder of its own.
"""

import json
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

from bentrix import OutputRefused
from bentrix.model import model as declare_model

SDK = pathlib.Path(__file__).resolve().parents[1]
SHELL_HOST = SDK.parents[1] / "crates" / "bentrix-shell" / "src" / "modelhost.rs"

WINDOW = {"at_ms": 7_200_000, "markets": {"EURUSD": {"H1": {"rows": [[0, 1, 1, 1, 1.25, 10, 2],
                                                                    [3_600_000, 1, 1, 1, 1.5, 10, 2]]}}}}
FALLING = {"at_ms": 7_200_000, "markets": {"EURUSD": {"H1": {"rows": [[3_600_000, 1, 1, 1, -1, 10, 2]]}}}}


def _predict(window):
    return {"op": "predict", "window": window, "symbol": "EURUSD", "anchor_ms": 3_600_000}


def _shell_package_and_bootstrap():
    """The bootstrap and the package ``__init__`` the shell writes, read from its source rather than retyped."""
    src = SHELL_HOST.read_text(encoding="utf-8")
    boot = re.search(r'pub const BOOTSTRAP: &str = "((?:[^"\\]|\\.)*)";', src)
    init = re.search(r'\("__init__\.py", "((?:[^"\\]|\\.)*)"\)', src)
    assert boot and init, "the shell's BOOTSTRAP or its package __init__ was not found in modelhost.rs"
    return json.loads(f'"{init.group(1)}"'), json.loads(f'"{boot.group(1)}"')


def _host(tmp_path, source, requests):
    """**Run the host as Bentrix does**: answers (exit code, the answer lines as written, stderr)."""
    init, boot = _shell_package_and_bootstrap()
    pkg = tmp_path / "model-host" / "bentrix"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(init, encoding="utf-8")
    for f in ("model.py", "modelhost.py"):
        shutil.copyfile(SDK / "bentrix" / f, pkg / f)
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.py").write_text(source, encoding="utf-8")
    stdin = "".join(json.dumps(r) + "\n" for r in requests)
    p = subprocess.run([sys.executable, "-I", "-c", boot, str(pkg.parent), str(models / "m.py")], input=stdin,
                       capture_output=True, text=True, timeout=60, cwd=str(models))
    return p.returncode, p.stdout.splitlines(), p.stderr


LIST_MODEL = '''
import bentrix

@bentrix.model(name="drift", interval="H1", frames=["H1"], lookback=2, steps=2)
def drift(w):
    c = w.close(w.symbol, "H1")
    if c[-1] < 0:
        return None
    return [c[-1] + 0.25, {"close": c[-1] + 0.5, "high": c[-1] + 0.75}]
'''

#: **Today's answers, byte for byte** — captured 2026-11-11 from the host at `4ccdf822` (before the dict form)
#: running ``LIST_MODEL`` over these requests.
TODAY = [
    '{"op": "manifest", "manifest": {"name": "drift", "interval": "H1", "frames": ["H1"], "markets": null, '
    '"lookback": 2, "steps": 2, "form": "line"}}',
    '{"op": "steps", "steps": [{"close": 1.75}, {"close": 2.0, "high": 2.25}]}',
    '{"op": "steps", "steps": null}',
]


def test_a_list_returning_model_answers_todays_steps_byte_for_byte(tmp_path):
    """**Nothing a person made stops working**: a model returning a list answers
    ``steps`` and its describe carries no ``outputs``, byte-equal to the host before the dict form; ``None`` says
    nothing.

    Mutation (2026-11-11): every answer routed through ``answer_outputs`` (``isinstance(got, (dict, list))``) — red,
    the host ended on the list; the manifest always carrying ``"outputs": []`` — red, the describe line differs."""
    code, lines, err = _host(tmp_path, LIST_MODEL, [{"op": "describe"}, _predict(WINDOW), _predict(FALLING),
                                                    {"op": "close"}])
    assert code == 0, err
    assert lines == TODAY


DICT_MODEL = '''
import bentrix
from bentrix.model import Output

@bentrix.model(name="dir_v4", interval="H1", lookback=2, outputs=[
    Output.declare("direction", "scalar", range=(0, 1), units="probability", horizon=4,
                   suggested_meaning={"reads": "direction", "high_means": "buy", "low_means": "sell",
                                      "act_above": 0.6, "act_below": 0.4}),
    {"name": "regime", "kind": "classes", "names": ["up", "flat", "down"], "exclusive": True, "persistent": True},
    {"name": "path", "kind": "path", "steps": 1},
    {"name": "act", "kind": "action"},
    {"name": "why", "kind": "text"},
    {"name": "fan", "kind": "quantiles", "levels": [0.1, 0.9], "units": "price"},
    {"name": "size", "kind": "scalar", "range": [0, 1], "horizon": "next_output"},
])
def dir_v4(w):
    c = w.close(w.symbol, "H1")
    return {"direction": [0.73, 0.61], "regime": {"up": 0.6, "flat": 0.3, "down": 0.1}, "path": [c[-1] + 0.25],
            "act": "buy", "why": "the close rose", "fan": {0.1: 1.4, 0.9: 1.6}, "size": None}
'''


def test_a_dict_returning_model_answers_outputs_in_each_kinds_shape(tmp_path):
    """**A dict by output name answers ``outputs``** — each output ``{"kind", "values"}`` with its values as
    ``output_spec.values`` carries them, a plain number, a ``{class: p}``, a list of steps, an intent and a text each
    put into their kind's shape; an output answered ``None`` is absent. The describe carries the declaration.

    Mutation (2026-11-11): ``_value_of`` wrapping a ``classes`` dict as ``{"w": ...}`` — red, the regime's values
    differ; the dict branch removed from the host — red, the host ended (a dict iterated as steps)."""
    code, lines, err = _host(tmp_path, DICT_MODEL, [{"op": "describe"}, _predict(WINDOW), {"op": "close"}])
    assert code == 0, err
    described = json.loads(lines[0])["manifest"]
    assert [o["name"] for o in described["outputs"]] == ["direction", "regime", "path", "act", "why", "fan", "size"]
    assert described["outputs"][0] == {"name": "direction", "kind": "scalar", "range": [0, 1], "units": "probability",
                                       "horizon": {"steps": 4},
                                       "suggested_meaning": {"reads": "direction", "high_means": "buy",
                                                             "low_means": "sell", "act_above": 0.6, "act_below": 0.4}}
    assert described["outputs"][6]["horizon"] == {"until": "next_output"}
    assert json.loads(lines[1]) == {"op": "outputs", "outputs": {
        "direction": {"kind": "scalar", "values": [{"value": 0.73}, {"value": 0.61}]},
        "regime": {"kind": "classes", "values": [{"p": {"up": 0.6, "flat": 0.3, "down": 0.1}}]},
        "path": {"kind": "path", "values": [{"close": 1.75}]},
        "act": {"kind": "action", "values": [{"intent": "buy"}]},
        "why": {"kind": "text", "values": [{"text": "the close rose"}]},
        "fan": {"kind": "quantiles", "values": [{"q": {"0.1": 1.4, "0.9": 1.6}}]},
    }}


def test_an_undeclared_name_is_an_error_at_the_first_call_in_words(tmp_path):
    """**A returned name the declaration lacks ends the host at the first call that returns it**, its sentence naming
    the name and what is declared — never an answer that drops it silently; a dict from a model declaring no outputs
    says so.

    Mutation (2026-11-11): the undeclared-name check removed — red, the host ended on a ``KeyError`` with no
    sentence."""
    src = DICT_MODEL.replace('"direction": [0.73, 0.61]', '"directoin": 0.5')
    code, lines, err = _host(tmp_path, src, [{"op": "describe"}, _predict(WINDOW), {"op": "close"}])
    assert code == 1 and len(lines) == 1, (code, lines)
    assert ('the model answered an output named "directoin", which its @bentrix.model does not declare; it declares '
            'direction, regime, path, act, why, fan, size') in err, err
    bare = LIST_MODEL.replace('return [c[-1] + 0.25, {"close": c[-1] + 0.5, "high": c[-1] + 0.75}]',
                              'return {"direction": 0.5}')
    code, lines, err = _host(tmp_path / "bare", bare, [{"op": "describe"}, _predict(WINDOW), {"op": "close"}])
    assert code == 1 and "it declares no outputs (return a list of steps, or declare outputs=[...])" in err, err


def test_a_value_outside_its_declared_range_ends_the_host_in_words_never_clamped(tmp_path):
    """**The declaration is the model's own, so a value outside it is the model's error**: refused in words naming
    the output, the value and the range, never clamped into it.

    Mutation (2026-11-11): ``answer_outputs`` not calling ``validate_output`` — red, the host answered 1.3."""
    src = DICT_MODEL.replace('"direction": [0.73, 0.61]', '"direction": 1.3')
    code, lines, err = _host(tmp_path, src, [{"op": "describe"}, _predict(WINDOW), {"op": "close"}])
    assert code == 1 and len(lines) == 1, (code, lines)
    assert "the output direction: values[0].value the output answers 0 to 1; 1.3 is outside it" in err, err


@pytest.mark.parametrize("outputs, words", [
    ([{"name": "p", "kind": "scalar", "tensor": "prob"}], "tensor belongs to an ONNX model's manifest"),
    ([{"name": "r", "kind": "classes"}], "the output r: names a classes output names its classes"),
    ([{"name": "a", "kind": "scalar"}, {"name": "a", "kind": "event"}], "the output a is declared twice"),
    ([], "outputs declares no output"),
    ([{"name": "a", "kind": "probability"}], "is not one of the twelve kinds"),
    ([{"name": "a", "kind": "scalar", "steps": 3}], "steps belongs to a path output only"),
    ([{"name": "a", "kind": "scalar", "range": [0, 1], "suggested_meaning": {"reads": "direction", "act_above": 2}}],
     "the output answers 0 to 1; 2 is outside it"),
    ([{"name": "a", "kind": "scalar", "colour": "red"}], 'carries "colour"'),
])
def test_a_declaration_is_refused_in_words_when_the_decorator_runs(outputs, words):
    """**A declaration the contract cannot read is refused when the file is loaded**, before any close — a Python
    model answers by name, so it names no tensor; a kind's names are present; a name is declared once.

    Mutation (2026-11-11): ``_declarations``' uniqueness check removed — red on the twice-declared case."""
    with pytest.raises(OutputRefused) as e:
        declare_model(name="m", interval="H1", outputs=outputs)
    assert words in str(e.value), str(e.value)
