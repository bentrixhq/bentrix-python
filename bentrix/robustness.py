"""**Robustness**: one policy over a grid of workbenches on other conditions.

``run`` asks Bentrix for the grid's workbenches in one call (``bench_matrix``: every combination of
tick models, spread multipliers, slippages, latencies and seeds, each a workbench alike the one
given), then runs the policy over each with ``gym.evaluate`` and answers the table. Bentrix writes
the matrix to the trial ledger as one trial of as many candidates as it has cells **before** it
answers, so nothing read off a cell escapes the count; each cell's result is then written as that
trial's ``result`` before ``run`` moves to the next.

Each row says, beside ``evaluate``'s report:

- ``sign_flip`` - the cell's IQM has the other sign from the workbench's own (a result that turns
  from a gain to a loss, or back, under its conditions);
- ``first_refused_step`` - the first step of any episode at which the venue refused the policy's
  order (``info["refused"]``), with the episode's seed; ``None`` when nothing was refused.

``env_factory`` takes a workbench and answers a ``TradingEnv`` over it, made as the caller makes theirs (its reward,
its episodes): the same factory makes the workbench's own and every cell's. Given a ``bentrix.Bench``, the factory is
handed each cell as a ``Bench`` too, through the same door; given a workbench's uri, each cell's uri, as before.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from bentrix.gym import evaluate


class _Watched:
    """An environment whose steps are counted, and its first refused order noted."""

    def __init__(self, env: Any):
        self._env = env
        self.first_refused: Optional[Dict[str, Any]] = None
        self._seed: Optional[int] = None
        self._step = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Any:
        self._seed, self._step = seed, 0
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        out = self._env.step(action)
        self._step += 1
        info = out[-1]
        if self.first_refused is None and isinstance(info, dict) and info.get("refused"):
            self.first_refused = {"step": self._step, "episode_seed": self._seed, "refused": info["refused"]}
        return out


def _sign(x: float) -> int:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0
    return (x > 0) - (x < 0)


def _report(env: Any, policy: Callable[[Any], Any], episodes: Sequence[int]) -> Dict[str, Any]:
    watched = _Watched(env)
    report = evaluate(policy, watched, seeds=episodes)
    report["first_refused_step"] = watched.first_refused
    return report


def run(
    env_factory: Callable[[str], Any],
    policy: Callable[[Any], Any],
    bench: Any,
    grid: Mapping[str, Sequence[Any]],
    seeds: Optional[Sequence[int]] = None,
    *,
    episodes: Sequence[int] = range(10),
    family: Optional[str] = None,
) -> Dict[str, Any]:
    """**The policy over every cell of ``grid``** (module doc): ``grid`` holds any of ``tick_model``,
    ``spread_multiplier``, ``slippage``, ``latency_ms``, ``commission`` and ``execution`` (``bench_matrix``'s grid), ``seeds`` the
    seeds the cells' random draws are made from, ``episodes`` the seeds ``evaluate`` runs each cell's
    episodes on. Answers the workbench's own report, one row a cell, and the trial's sequence number."""
    from bentrix.core import Bench as _Bench

    own = env_factory(bench)
    if isinstance(bench, _Bench):
        client: Any = bench.bx
        cell_of: Callable[[str], Any] = lambda uri: _Bench(client, uri)  # noqa: E731
        bench = bench.uri
    else:
        client = own.local
        cell_of = lambda uri: uri  # noqa: E731
    matrix = client.call("bench_matrix", bench=bench, grid=dict(grid), seeds=list(seeds) if seeds else None, family=family)
    trial = matrix["trial"]["seq"]
    baseline = _report(own, policy, episodes)
    rows: List[Dict[str, Any]] = []
    for cell in matrix["cells"]:
        report = _report(env_factory(cell_of(cell["bench"])), policy, episodes)
        flip = _sign(report["iqm"]) != 0 and _sign(baseline["iqm"]) != 0 and _sign(report["iqm"]) != _sign(baseline["iqm"])
        result = {
            "bench": cell["bench"],
            "conditions": cell["conditions"],
            "iqm": report["iqm"],
            "interval": report["interval"],
            "sign_flip": flip,
            "first_refused_step": report["first_refused_step"],
        }
        # The cell's result is the trial's before the next cell runs.
        line = client.call("ledger_record", kind="result", of=trial, result={k: v for k, v in result.items() if k != "interval"} | {"interval": [_finite(x) for x in report["interval"]], "iqm": _finite(report["iqm"])})
        rows.append({**result, "report": report, "ledger_seq": line["seq"]})
    return {
        "bench": bench,
        "baseline": baseline,
        "cells": rows,
        "sign_flips": sum(1 for r in rows if r["sign_flip"]),
        "trial": trial,
        "union": matrix["trial"]["union"],
    }


def _finite(x: Any) -> Any:
    """A figure the ledger's JSON can hold: a non-finite one written as ``None``."""
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None
