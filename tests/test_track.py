"""**Experiment tracking hooks**: a run's identity reaches whichever tracker is installed."""

import sys
import types

import pytest

pytest.importorskip("bentrix._native")

from bentrix.track import ENV_VERSION, identity, log_run  # noqa: E402


def test_a_runs_identity_reaches_mlflow_and_wandb_where_installed(monkeypatch):
    """With stand-ins for MLflow and W&B the tags, parameters and metrics reach both; with neither the
    hooks reach nothing and raise nothing.

    Mutation (2026-11-01): the tags not set on MLflow — red, the ledger's hash did not
    reach it."""
    seen = {}
    mlflow = types.SimpleNamespace(set_tags=lambda t: seen.setdefault("tags", t), log_params=lambda p: seen.setdefault("params", p), log_metrics=lambda m: seen.setdefault("metrics", m))
    config = types.SimpleNamespace(update=lambda d, allow_val_change: seen.setdefault("config", d))
    wandb = types.SimpleNamespace(run=types.SimpleNamespace(config=config), log=lambda m: seen.setdefault("wandb", m))
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    tags = identity(ledger_hash="abc", reward="dsr@1", seed=3, bench="bentrix://lab/1/bench/2")
    assert tags["bentrix.env"] == ENV_VERSION and tags["bentrix.ledger_hash"] == "abc"
    assert log_run(tags, {"algo": "PPO"}, {"eval": 1.5}) == ["mlflow", "wandb"]
    assert seen["tags"]["bentrix.ledger_hash"] == "abc", "the ledger's hash did not reach MLflow"
    assert seen["metrics"] == {"eval": 1.5} and seen["wandb"] == {"eval": 1.5}
    monkeypatch.setitem(sys.modules, "mlflow", None)
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert log_run(tags) == []


def test_the_sb3_example_trains_and_evaluates_on_the_ci_tape(tape):
    """**The SB3 example, end to end**: PPO trained 128 steps on the CI tape's workbench and one episode
    evaluated, its reward finite and its ledger hash a sha256."""
    pytest.importorskip("stable_baselines3")
    import math
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "examples"))
    import rl_sb3_ppo

    local, bench = tape
    out = rl_sb3_ppo.main(bench, steps=128, local=local)
    assert math.isfinite(out["eval_reward"]) and len(out["ledger_hash"]) == 64, out
