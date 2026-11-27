"""**Causal features**: helpers whose value at a step reads nothing after it.

After reading tomorrow's candle, the commonest leak in a learned strategy is normalisation over the
whole history: a mean and a deviation taken over every day, the test days included, and applied to
the training days. Everything here is causal by construction:

- ``log_returns(close, window)`` - ``log(close[t] / close[t - window])``; the first ``window`` are NaN.
- ``rolling_zscore(x, n)`` - ``(x[t] - mean) / deviation`` over the ``n`` values ending at ``t``
  (the population deviation); NaN while fewer than ``n`` values exist or the deviation is zero.
- ``Normalizer.fit(x, bench, local)`` - a mean and a deviation per column, taken from ``x`` and
  **refused unless** ``bench`` is a part of a split whose role is ``train`` (``bench_split``): the
  figures a model sees on validation or test days are then the training days' alone.
  ``Normalizer.apply(any)`` uses them on any part's features.

The property every function here keeps, and the test holds it to: the value at ``t`` computed from
``x[: t + 1]`` is the value at ``t`` computed from the whole of ``x``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def log_returns(close: Any, window: int = 1) -> np.ndarray:
    """``log(close[t] / close[t - window])`` for each ``t``; NaN for the first ``window``."""
    if window < 1:
        raise ValueError(f"window {window} is not 1 or more")
    c = np.asarray(close, dtype=np.float64)
    out = np.full(c.shape, np.nan)
    if len(c) > window:
        out[window:] = np.log(c[window:] / c[:-window])
    return out


def rolling_zscore(x: Any, n: int) -> np.ndarray:
    """``(x[t] - mean) / deviation`` over the ``n`` values ending at ``t`` - never a later one; NaN
    while fewer than ``n`` exist or the deviation is zero."""
    if n < 2:
        raise ValueError(f"n {n} is not 2 or more")
    v = np.asarray(x, dtype=np.float64)
    out = np.full(v.shape, np.nan)
    for t in range(n - 1, len(v)):
        w = v[t - n + 1 : t + 1]
        sd = w.std()
        if sd > 0:
            out[t] = (v[t] - w.mean()) / sd
    return out


class Normalizer:
    """A mean and a deviation per column, fitted on a training part only (module doc)."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, fitted_on: Optional[Dict[str, Any]] = None):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.fitted_on = fitted_on or {}

    @classmethod
    def fit(cls, x: Any, bench: Any, local: Any = None) -> "Normalizer":
        """**Fitted on ``x``, the features of ``bench``**, which must be a split's ``train`` part as
        ``bench_manifest`` reads it; any other workbench - a validation or test part, or one never
        split - is refused, since its figures would carry the days a model is judged on. ``bench`` is the
        workbench object (``Normalizer.fit(x, parts["train"])``), through either door, or its address with
        ``local``, the engine, as before."""
        from .core import Bench

        if isinstance(bench, Bench):
            bench, local = bench.uri, bench.bx
        elif local is None:
            raise ValueError("name the engine (local) with a workbench's address, or pass a bentrix.Bench")
        doc = local.call("bench_manifest", bench=bench)["document"]
        role = doc.get("role")
        if role != "train":
            said = f"its role is {role}" if role else "it is not a part of a split"
            raise ValueError(
                f"A normaliser is fitted on a training part only, and {bench} is not one: {said}. "
                "Split the workbench (bench_split) and fit on its train part."
            )
        v = np.asarray(x, dtype=np.float64)
        if v.ndim == 1:
            v = v[:, None]
        mean = np.nanmean(v, axis=0)
        sd = np.nanstd(v, axis=0)
        scale = np.where(sd > 0, sd, 1.0)
        return cls(mean, scale, {"bench": bench, "days": doc.get("days"), "rows": int(v.shape[0])})

    def apply(self, x: Any) -> np.ndarray:
        """``(x - mean) / deviation`` with the training part's figures, on any part's features."""
        v = np.asarray(x, dtype=np.float64)
        flat = v.ndim == 1
        out = (v[:, None] if flat else v) - self.mean
        out = out / self.scale
        return out[:, 0] if flat else out

    def to_dict(self) -> Dict[str, Any]:
        """The figures and where they were fitted, for a file beside a model."""
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "fitted_on": self.fitted_on}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Normalizer":
        return cls(np.asarray(d["mean"]), np.asarray(d["scale"]), d.get("fitted_on"))
