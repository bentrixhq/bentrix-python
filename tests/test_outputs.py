"""**A model's outputs of any kind** (the Python half): ``bentrix.Output``'s builders and the contract's rules checked in Python
before a document leaves it.

The documents the builders make are pinned in ``fixtures/outputs/accepted.json`` (one per kind) and the documents the
contract refuses in ``fixtures/outputs/refused.json`` — the Rust side (``bentrix_forecast::output::validate_output``)
reads the same files, so the two checkers cannot drift apart unseen. ``python tests/test_outputs.py --write`` rewrites
them from the builders; the guard below goes red when the files and the builders disagree.
"""

import copy
import json
import pathlib
import sys

import pytest

import bentrix
from bentrix import Output, OutputRefused
from bentrix.model import KINDS, export_manifest, validate_output

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "outputs"
T0 = 1_704_110_400_000

#: **One valid document of each kind, made by its builder** — keyed by the kind, so a kind without a builder row is
#: missing from the census below.
BUILT = {
    "path": lambda: Output.path("dir_v4", "EURUSD", T0, [1.1, {"close": 1.2, "high": 1.21, "low": 1.19}], horizon=2,
                                output="path"),
    "scalar": lambda: Output.scalar("dir_v4", "EURUSD", T0, [0.73, 0.61, 0.55, 0.52], range=(0, 1),
                                    units="probability", horizon=4, output="direction", epoch=17,
                                    suggested_meaning={"reads": "direction", "high_means": "buy", "low_means": "sell",
                                                       "act_above": 0.6, "act_below": 0.4},
                                    look={"rising": "#4C9BFFFF", "label": "Direction v4"},
                                    note="trained 2019-2024, H1, features: 48 closes + 12 ATRs"),
    "classes": lambda: Output.classes("regime_v1", "EURUSD", T0, {"up": 0.5, "flat": 0.3, "down": 0.2},
                                      exclusive=True, persistent=True,
                                      suggested_meaning={"reads": "state",
                                                         "classes": {"up": "buy", "down": "sell", "flat": "nothing"}}),
    "quantiles": lambda: Output.quantiles("fan_v2", "EURUSD", T0, [{0.1: 1.09, 0.5: 1.1, 0.9: 1.11},
                                                                   {0.1: 1.088, 0.5: 1.1, 0.9: 1.112}],
                                          units="price", horizon=2),
    "rank": lambda: Output.rank("xs_rank", "AAPL", T0, 0.62, universe=4871, date_ms=T0,
                                horizon={"business_days": 20}),
    "position": lambda: Output.position("alloc_v1", "EURUSD", T0, {"EURUSD": -0.5, "GBPUSD": 0.25},
                                        markets=["EURUSD", "GBPUSD"], units="fraction", range=(-1, 1),
                                        horizon="next_output"),
    "weights": lambda: Output.weights("alloc_v1", "EURUSD", T0, {"EURUSD": 0.6, "GBPUSD": 0.4}, long_only=True,
                                      markets=["EURUSD", "GBPUSD"]),
    "event": lambda: Output.event("breakout", "XAUUSD", T0, 0.8, within_steps=3, name="breakout"),
    "levels": lambda: Output.levels("barrier_v1", "EURUSD", T0, stop=1.09, target=1.12, p={"target": 0.55},
                                    until_ms=T0 + 86_400_000, horizon="levels"),
    "action": lambda: Output.action("ppo_v3", "EURUSD", T0, "buy", size=0.5, size_units="fraction"),
    "text": lambda: Output.text("llm_note", "EURUSD", T0, "the trend held above the weekly open"),
    "series": lambda: Output.series("pushed_rsi", "EURUSD", T0, [{"rsi": 61.2, "signal": 58.0},
                                                                 {"rsi": 63.0, "signal": 59.1}], range=(0, 100)),
}


def _refused():
    """**Documents the contract refuses**, each made from an accepted one by one change, with Python's sentence."""
    acc = {k: f() for k, f in BUILT.items()}
    cases = []

    def case(name, kind, change):
        d = copy.deepcopy(acc[kind])
        change(d)
        with pytest.raises(OutputRefused) as e:
            validate_output(d)
        cases.append({"case": name, "why": str(e.value), "spec": d})

    for k in KINDS:
        case(f"{k}_carrying_put_ms", k, lambda d: d.__setitem__("put_ms", T0 + 1))
    case("scalar_above_its_range", "scalar", lambda d: d["values"].__setitem__(1, {"value": 1.3}))
    case("rank_above_0_to_1", "rank", lambda d: d["values"].__setitem__(0, {"value": 1.5}))
    case("position_outside_its_range", "position", lambda d: d["values"][0]["targets"].__setitem__("EURUSD", 2.0))
    case("position_naming_a_market_not_named", "position", lambda d: d["values"][0]["targets"].__setitem__("USDJPY", 0.1))
    case("event_score_above_1", "event", lambda d: d["values"][0].__setitem__("score", 1.2))
    case("classes_probability_above_1", "classes", lambda d: d["values"][0]["p"].__setitem__("up", 1.4))
    case("classes_naming_a_class_not_named", "classes", lambda d: d["values"][0]["p"].__setitem__("sideways", 0.1))
    case("classes_without_names", "classes", lambda d: d.pop("names"))
    case("series_without_names", "series", lambda d: d.pop("names"))
    case("quantiles_level_of_1", "quantiles", lambda d: d.__setitem__("levels", [0.1, 0.5, 1.0]))
    case("weights_long_only_not_summing_to_1", "weights", lambda d: d["values"][0]["w"].__setitem__("EURUSD", 0.7))
    case("action_of_an_unknown_intent", "action", lambda d: d["values"][0].__setitem__("intent", "hold"))
    case("action_of_a_negative_size", "action", lambda d: d["values"][0].__setitem__("size", -0.5))
    case("text_longer_than_2000", "text", lambda d: d["values"][0].__setitem__("text", "x" * 2001))
    case("levels_naming_none", "levels", lambda d: d["values"][0].__setitem__("levels", {}))
    case("meaning_word_not_the_products", "scalar", lambda d: d["suggested_meaning"].__setitem__("high_means", "profit"))
    case("meaning_threshold_outside_the_range", "scalar", lambda d: d["suggested_meaning"].__setitem__("act_above", 1.5))
    case("meaning_on_a_text", "text", lambda d: d.__setitem__("suggested_meaning", {"reads": "direction"}))
    case("horizon_of_0_steps", "scalar", lambda d: d.__setitem__("horizon", {"steps": 0}))
    case("horizon_until_an_unknown", "position", lambda d: d.__setitem__("horizon", {"until": "tomorrow"}))
    case("model_label_not_a_label", "scalar", lambda d: d.__setitem__("model", "Dir V4"))
    case("range_not_a_range", "scalar", lambda d: d.__setitem__("range", [1, 0]))
    case("no_value", "scalar", lambda d: d.__setitem__("values", []))
    return cases


def test_the_kinds_are_twelve_and_each_has_a_builder_whose_document_the_contract_takes():
    """**The census**: ``KINDS`` is pinned at twelve, every kind has its builder row, and each builder's document is
    taken by the contract, its kind the one asked for.

    Mutation (2026-11-11): the ``weights`` row removed from ``BUILT`` — red, *weights has no builder*; ``KINDS`` cut
    to eleven — red on the pin."""
    assert len(KINDS) == 12 and len(set(KINDS)) == 12, "the kinds are pinned at twelve"
    for k in KINDS:
        assert k in BUILT, f"{k} has no builder"
        d = BUILT[k]()
        assert d["kind"] == k and validate_output(d) is d
        assert "put_ms" not in d, f"{k}'s builder set put_ms"
    assert set(BUILT) == set(KINDS)


def test_a_value_outside_its_declared_range_is_refused_in_words_never_clamped():
    """**Refused in words, never clamped** — in every kind that has a range; inside it the value is the model's own,
    to the bit.

    Mutation (2026-11-11): ``_in_range`` answering ``min(max(v, lo), hi)`` instead of raising — red, *1.3 was
    clamped*."""
    cases = [
        lambda: Output.scalar("m", "EURUSD", T0, 1.3, range=(0, 1)),
        lambda: Output.rank("m", "EURUSD", T0, 1.5, universe=10),
        lambda: Output.position("m", "EURUSD", T0, 2.0, range=(-1, 1)),
        lambda: Output.event("m", "EURUSD", T0, 1.2),
        lambda: Output.classes("m", "EURUSD", T0, {"up": 1.4}, exclusive=True),
    ]
    for make in cases:
        with pytest.raises(OutputRefused, match="is outside it"):
            make()
    with pytest.raises(OutputRefused) as e:
        Output.scalar("m", "EURUSD", T0, [0.5, 1.3], range=(0, 1))
    assert str(e.value) == "values[1].value the output answers 0 to 1; 1.3 is outside it", str(e.value)
    v = 0.1 + 0.2
    assert Output.scalar("m", "EURUSD", T0, v, range=(0, 1))["values"][0]["value"] == v


def test_a_put_carrying_put_ms_is_malformed_for_every_kind_and_no_builder_sets_it():
    """**Bentrix stamps the arrival**: a document carrying ``put_ms`` is refused for every kind, and
    no builder takes it.

    Mutation (2026-11-11): the ``put_ms`` check alone removed — GREEN, the unknown-field refusal still holds it (two
    layers, as ``deny_unknown_fields`` is Rust's); the check removed and ``put_ms`` added to the document's fields —
    red."""
    for k in KINDS:
        d = BUILT[k]()
        d["put_ms"] = T0 + 1
        with pytest.raises(OutputRefused, match="put_ms"):
            validate_output(d)
    with pytest.raises(OutputRefused, match="put_ms is not a field"):
        Output.scalar("m", "EURUSD", T0, 0.5, put_ms=T0)


def test_a_kinds_names_are_present_and_a_meanings_words_are_the_products():
    """**What output.rs requires, required here**: a ``classes`` or ``series`` output names its names (a builder
    takes them from the first step), a ``quantiles`` output its levels; a meaning's words are the closed nine and its
    thresholds inside the range; three examples are three meanings of one scalar.

    Mutation (2026-11-11): the ``MEANING_WORDS`` check removed — red, *"profit" was taken*."""
    assert Output.classes("m", "EURUSD", T0, [{"a": 0.2, "b": 0.8}], exclusive=False)["names"] == ["a", "b"]
    assert Output.series("m", "EURUSD", T0, {"x": 1.0})["names"] == ["x"]
    assert Output.quantiles("m", "EURUSD", T0, {0.9: 2.0, 0.1: 1.0})["levels"] == [0.1, 0.9]
    with pytest.raises(OutputRefused, match="a classes output names its classes"):
        Output.spec("classes", "m", "EURUSD", T0, {"p": {"a": 0.5}})
    for m in ({"reads": "direction", "high_means": "buy", "low_means": "sell", "act_above": 0.6, "act_below": 0.4},
              {"reads": "direction", "high_means": "buy", "low_means": "nothing", "act_above": 0.6},
              {"reads": "direction", "high_means": "sell", "low_means": "nothing", "act_above": 0.6}):
        Output.scalar("m", "EURUSD", T0, 0.7, range=(0, 1), suggested_meaning=m)
    with pytest.raises(OutputRefused, match='"profit" is not one of the meaning'):
        Output.scalar("m", "EURUSD", T0, 0.7, range=(0, 1), suggested_meaning={"reads": "direction", "high_means": "profit"})
    with pytest.raises(OutputRefused, match="0 to 1; 1.5 is outside it"):
        Output.scalar("m", "EURUSD", T0, 0.7, range=(0, 1), suggested_meaning={"reads": "direction", "act_above": 1.5})


def test_the_fixtures_the_rust_side_reads_are_what_the_builders_make_and_what_the_contract_refuses():
    """**The fixtures are not a second source**: ``accepted.json`` is exactly the builders' documents, one per kind,
    and every case of ``refused.json`` is refused here with the sentence it records.

    Mutation (2026-11-11): the builder writing a ``scalar``'s range as ``[0, 2]`` — red, the fixture's document
    differs; the quantile levels' (0, 1) check removed — red, the recorded case was taken."""
    accepted = json.loads((FIXTURES / "accepted.json").read_text(encoding="utf-8"))
    assert [d["kind"] for d in accepted] == list(KINDS)
    assert accepted == [json.loads(json.dumps(BUILT[k]())) for k in KINDS]
    refused = json.loads((FIXTURES / "refused.json").read_text(encoding="utf-8"))
    assert len(refused) == len(_refused()) == 12 + 23
    for c in refused:
        with pytest.raises(OutputRefused) as e:
            validate_output(c["spec"])
        assert str(e.value) == c["why"], c["case"]


# ---- export_manifest(outputs=...): schema 2 beside today's schema 1 ----------------------------------------------------

#: **Today's schema-1 manifests, byte for byte** — captured 2026-11-11 from ``export_manifest`` at `4ccdf822`, before
#: ``outputs`` existed: (the call's arguments, the JSON it wrote).
SCHEMA_1 = [
    ((("x", "M15", 3), {"frames": ["M15", "H1"], "fields": ("close", "high"), "transform": "returns", "output": "delta"}),
     '{"schema_version": 1, "name": "x", "interval": "M15", "frames": ["M15", "H1"], "markets": null, "lookback": 200, '
     '"steps": 3, "form": "line", "input": {"fields": ["close", "high"], "transform": "returns"}, "output": {"kind": '
     '"delta", "pick": "first"}}'),
    ((("dir_v1", "H1", 4), {}),
     '{"schema_version": 1, "name": "dir_v1", "interval": "H1", "frames": ["H1"], "markets": null, "lookback": 200, '
     '"steps": 4, "form": "line", "input": {"fields": ["close"], "transform": "raw"}, "output": {"kind": "close", '
     '"pick": "first"}}'),
    ((("dir_v1", "H1", 4), {"markets": ["EURUSD", "XAUUSD"], "lookback": 48, "form": "candlestick",
                            "fields": ("open", "high", "low", "close", "tick_volume"), "transform": "zscore",
                            "output": "return", "pick": "last"}),
     '{"schema_version": 1, "name": "dir_v1", "interval": "H1", "frames": ["H1"], "markets": ["EURUSD", "XAUUSD"], '
     '"lookback": 48, "steps": 4, "form": "candlestick", "input": {"fields": ["open", "high", "low", "close", '
     '"tick_volume"], "transform": "zscore"}, "output": {"kind": "return", "pick": "last"}}'),
]


def _schema_2():
    """A two-output example, its heads as slices of one answer (``bentrix_model`` reads no second tensor yet)."""
    return export_manifest("dir_lgbm_v4", "H1", frames=["H1", "H4"], lookback=48,
                           fields=("open", "high", "low", "close", "tick_volume"), transform="zscore", outputs=[
        Output.declare("direction", "scalar", range=(0, 1), units="probability", horizon=4, slice=[0, 4],
                       suggested_meaning={"reads": "direction", "high_means": "buy", "low_means": "sell",
                                          "act_above": 0.6, "act_below": 0.4}),
        Output.declare("size", "scalar", range=(0, 1), units="ratio", horizon="next_output", slice=[4, 5],
                       suggested_meaning={"reads": "size", "gates": "output:dir_lgbm_v4.direction"}),
        Output.declare("path", "path", steps=3, as_="return", slice=[5, 8]),
        {"name": "regime", "kind": "classes", "names": ["up", "flat", "down"], "exclusive": True, "softmax": True,
         "slice": [8, 11]},
    ])


def test_a_schema_1_export_is_byte_for_byte_todays():
    """**Without ``outputs``, nothing changes**: every schema-1 manifest is byte-equal to the one
    ``export_manifest`` wrote before ``outputs`` existed.

    Mutation (2026-11-11): the schema-1 branch writing ``"schema_version": 2`` — red, the bytes differ."""
    for (args, kw), today in SCHEMA_1:
        assert json.dumps(export_manifest(*args, **kw)) == today
    with pytest.raises(ValueError, match="names the steps"):
        export_manifest("x", "H1")


def test_a_schema_2_export_declares_its_outputs_as_designed():
    """**With ``outputs``, schema 2**: no top-level steps, form or output; each output in
    ``ManifestOutput``'s fields, a path's ``form``, ``as`` and ``pick`` written out; what reading an ONNX model's
    numbers needs is refused in words when it is missing.

    Mutation (2026-11-11): the path defaults not filled — red, the path's form, as and pick are absent; the ``text``
    refusal removed — red, a text output was exported."""
    m = _schema_2()
    assert m == {
        "schema_version": 2, "name": "dir_lgbm_v4", "interval": "H1", "frames": ["H1", "H4"], "markets": None,
        "lookback": 48, "input": {"fields": ["open", "high", "low", "close", "tick_volume"], "transform": "zscore"},
        "outputs": [
            {"name": "direction", "kind": "scalar", "range": [0, 1], "units": "probability", "horizon": {"steps": 4},
             "slice": [0, 4], "suggested_meaning": {"reads": "direction", "high_means": "buy", "low_means": "sell",
                                                    "act_above": 0.6, "act_below": 0.4}},
            {"name": "size", "kind": "scalar", "range": [0, 1], "units": "ratio", "horizon": {"until": "next_output"},
             "slice": [4, 5], "suggested_meaning": {"reads": "size", "gates": "output:dir_lgbm_v4.direction"}},
            {"name": "path", "kind": "path", "form": "line", "steps": 3, "as": "return", "pick": "first",
             "slice": [5, 8]},
            {"name": "regime", "kind": "classes", "names": ["up", "flat", "down"], "exclusive": True, "softmax": True,
             "slice": [8, 11]},
        ]}
    assert json.loads((FIXTURES / "manifest_schema2.json").read_text(encoding="utf-8")) == m
    assert json.loads((FIXTURES / "manifest_schema1.json").read_text(encoding="utf-8")) == json.loads(SCHEMA_1[2][1])
    assert export_manifest("m", "H1", outputs=[Output.declare("p", "scalar", tensor="prob")])["outputs"][0]["tensor"] == "prob"
    refused = [
        (dict(steps=3, outputs=[{"name": "a", "kind": "scalar"}]), "steps belong to a schema-1 manifest"),
        (dict(output="delta", outputs=[{"name": "a", "kind": "scalar"}]), "output belong to a schema-1 manifest"),
        (dict(outputs=[{"name": "t", "kind": "text"}]), "text comes from a program"),
        (dict(outputs=[{"name": "p", "kind": "path"}]), "a path output names its steps"),
        (dict(outputs=[{"name": "a", "kind": "action", "names": ["buy", "hold"]}]), "names are intents"),
        (dict(outputs=[{"name": "w", "kind": "weights"}]), "a weights output names its markets"),
        (dict(outputs=[{"name": "l", "kind": "levels"}]), "a levels output names its levels"),
        (dict(outputs=[{"name": "a", "kind": "scalar", "slice": [4, 4]}]), "is not [from, to)"),
        (dict(outputs=[]), "outputs declares no output"),
    ]
    for kw, words in refused:
        with pytest.raises(OutputRefused) as e:
            export_manifest("m", "H1", **kw)
        assert words in str(e.value), (kw, str(e.value))


def _write():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "accepted.json").write_text(json.dumps([BUILT[k]() for k in KINDS], indent=1) + "\n", encoding="utf-8")
    (FIXTURES / "refused.json").write_text(json.dumps(_refused(), indent=1) + "\n", encoding="utf-8")
    (FIXTURES / "manifest_schema1.json").write_text(SCHEMA_1[2][1] + "\n", encoding="utf-8")
    (FIXTURES / "manifest_schema2.json").write_text(json.dumps(_schema_2(), indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__" and "--write" in sys.argv:
    _write()
