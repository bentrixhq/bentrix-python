"""**Bentrix's measures against the reference libraries**.

`crates/bentrix-forecast/fixtures/measures_reference.json` holds fixed inputs and what scikit-learn and SciPy answer
for them, with the libraries' versions; `bentrix-forecast`'s
`the_measures_match_the_reference_library` asserts Bentrix's own measures equal those answers. This guard keeps the
fixture honest: where scikit-learn and SciPy are installed it recomputes every answer from the fixture's inputs and
requires the file to say the same - so the fixture cannot drift from the libraries it names. It is skipped, with its
reason, where they are not installed (it runs in `python:3.12-slim` with
both installed).

    python -m pytest sdk/python/tests/test_measures_reference.py

Writing the fixture (the inputs are a fixed-seed draw, so the file is the same each time):

    python sdk/python/tests/test_measures_reference.py --write

Mutation (2026-11-11): the fixture's AUC moved by 1e-9 - red here and in the Rust guard (a change inside the 1e-12
tolerance stays green, by design).
"""
import json
import pathlib
import random
import sys

import pytest

FIXTURE = pathlib.Path(__file__).resolve().parents[3] / "crates" / "bentrix-forecast" / "fixtures" / "measures_reference.json"


def inputs():
    """The fixed inputs: 240 probabilities (two decimals, so ties occur) with outcomes, 240 values against realised
    numbers, three classes, and a quantile's predictions."""
    rng = random.Random(191)
    p = [round(rng.uniform(0.02, 0.98), 2) for _ in range(240)]
    y = [rng.random() < q for q in p]
    v = [rng.gauss(0, 1) for _ in range(240)]
    a = [0.4 * x + rng.gauss(0, 1) for x in v]
    names = ["down", "flat", "up"]
    rows, labels = [], []
    for _ in range(120):
        raw = [rng.uniform(0.05, 1.0) for _ in names]
        s = sum(raw)
        row = {n: r / s for n, r in zip(names, raw)}
        rows.append(row)
        labels.append(rng.choices(names, weights=[row[n] for n in names])[0])
    q_pred = [rng.gauss(0, 1) for _ in range(120)]
    q_real = [x + rng.gauss(0, 0.5) for x in q_pred]
    return {"p": p, "y": y, "v": v, "a": a, "names": names, "rows": rows, "labels": labels, "q_level": 0.9, "q_pred": q_pred,
            "q_real": q_real, "threshold": 0.5}


def answers(i):
    """What scikit-learn and SciPy answer for the inputs."""
    from scipy import stats
    from sklearn import metrics

    predicted = [q > i["threshold"] for q in i["p"]]
    probs = [[row[n] for n in i["names"]] for row in i["rows"]]
    return {
        "auc": float(metrics.roc_auc_score(i["y"], i["p"])),
        "brier": float(metrics.brier_score_loss(i["y"], i["p"])),
        "log_loss": float(metrics.log_loss(i["y"], i["p"])),
        "matthews": float(metrics.matthews_corrcoef(i["y"], predicted)),
        "pearson": float(stats.pearsonr(i["v"], i["a"])[0]),
        "spearman": float(stats.spearmanr(i["v"], i["a"])[0]),
        "log_loss_classes": float(metrics.log_loss(i["labels"], probs, labels=i["names"])),
        "pinball": float(metrics.mean_pinball_loss(i["q_real"], i["q_pred"], alpha=i["q_level"])),
    }


def versions():
    import scipy
    import sklearn

    return {"scikit-learn": sklearn.__version__, "scipy": scipy.__version__}


def test_the_fixture_is_what_the_reference_libraries_answer():
    try:
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        pytest.skip("scikit-learn and SciPy are not installed here; the fixture's answers are checked where they are")
    f = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert f["inputs"] == inputs(), "the fixture's inputs are not the fixed-seed draw"
    got = answers(f["inputs"])
    for k, want in f["answers"].items():
        assert got[k] == pytest.approx(want, rel=0, abs=1e-12), f"{k}: the libraries answer {got[k]}, the fixture says {want}"
    assert set(got) == set(f["answers"])


if __name__ == "__main__" and "--write" in sys.argv:
    i = inputs()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({"libraries": versions(), "inputs": i, "answers": answers(i)}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote", FIXTURE)
