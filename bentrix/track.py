"""**Experiment tracking hooks**: a run's identity logged to MLflow and/or Weights & Biases,
whichever is installed — the ledger's id, the environment's version, the reward plugin, the seed — so a
result found there leads back to the trial BENTRIX counted.

Nothing here is required: without either library ``log_run`` answers which it could not reach and logs
nothing. No figure goes anywhere a person did not send it: these are the person's own trackers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

ENV_VERSION = "bentrix/TradingEnv-v0"


def identity(ledger_hash: Optional[str] = None, reward: Optional[str] = None, seed: Optional[int] = None, bench: Optional[str] = None, **more: Any) -> Dict[str, Any]:
    """**What a tracked run names** — the tags every hook writes."""
    from bentrix import _native

    tags = {"bentrix.env": ENV_VERSION, "bentrix.build": _native.build(), "bentrix.reward": reward, "bentrix.seed": seed, "bentrix.ledger_hash": ledger_hash, "bentrix.bench": bench}
    tags.update({f"bentrix.{k}": v for k, v in more.items()})
    return {k: v for k, v in tags.items() if v is not None}


def log_run(tags: Dict[str, Any], params: Optional[Dict[str, Any]] = None, metrics: Optional[Dict[str, float]] = None) -> List[str]:
    """**Log a run to every tracker installed** (MLflow's active run, W&B's ``wandb.run``): its tags, its
    parameters and its metrics. Answers the trackers it reached."""
    reached = []
    try:
        import mlflow

        mlflow.set_tags({k: str(v) for k, v in tags.items()})
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        reached.append("mlflow")
    except ImportError:
        pass
    try:
        import wandb

        if getattr(wandb, "run", None) is not None:
            wandb.run.config.update({**tags, **(params or {})}, allow_val_change=True)
            if metrics:
                wandb.log(metrics)
            reached.append("wandb")
    except ImportError:
        pass
    return reached
