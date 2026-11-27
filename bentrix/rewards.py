"""**The Gym's eight rewards as plain Python**, for a reward of your own that starts from one of ours.

Each is what ``env_step`` computes in Rust for the reward of the same name, operation for operation, so
a copy you change differs from ours only by what you changed. ``tests/test_gym_reference.py`` holds
them to the Rust environment step for step: exactly, but the volatility-scaled one, whose standard
deviation NumPy sums pairwise (within 1e-12).

Every reward is called once a step with the account's equity ``before`` the action and ``after`` the
clock reached the next close, whether the venue ``refused`` the action's order, and the ``exposure``
(margin over equity, after). ``r`` below is the step's simple return, ``(after - before) / before``
(0 when ``before`` is not above zero).

- ``pnl@1`` - ``after - before``, in the account's money.
- ``log_return@1`` - ``log(after / before)``; -1 when either is not above zero.
- ``dsr@1`` - Moody and Saffell's differential Sharpe ratio with ``eta = 0.01``: with ``A`` and ``B``
  the exponentially weighted first and second moments of ``r`` before this step,
  ``(B * (r - A) - A * (r^2 - B) / 2) / (B - A^2)^1.5``, 0 while ``B`` is 0 or the variance is at or
  under ``1e-12 * B``; then ``A += eta * (r - A)``, ``B += eta * (r^2 - B)``.
- ``dsr_dd_penalty@1`` (the default) - ``dsr@1`` less half of what the drawdown from the episode's
  peak equity grew by this step (``max(0, dd - dd_before)``, ``dd = (peak - after) / peak``).
- ``cvar_lagrangian@1`` - ``r`` less half of the expected shortfall of the episode's returns so far:
  the mean loss of their worst 5 % (at least one), when it is a loss.
- ``cost_aware_vol_scaled@1`` - ``r`` over the standard deviation of the last 50 returns (NumPy's
  population deviation); 0 while it is at or under ``1e-12`` or there is one return.
- ``rule_adherence@1`` - -1 when the venue refused the action's order, else 0. A product hypothesis.
- ``risk_budget@1`` - ``-max(0, exposure - 1)``: a margin larger than the equity costs. A product
  hypothesis.

A participation cap's impact is charged by the environment after the reward (``reward - impact``);
it is not part of any function here.

    from bentrix import rewards
    mine = rewards.make("dsr_dd_penalty@1")
    r = mine(before=10_000.0, after=10_012.5, refused=False, exposure=0.2)
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np

#: The rewards ``env_make`` offers, in its order.
REWARDS = (
    "pnl@1",
    "log_return@1",
    "dsr@1",
    "dsr_dd_penalty@1",
    "cvar_lagrangian@1",
    "cost_aware_vol_scaled@1",
    "rule_adherence@1",
    "risk_budget@1",
)

#: The two that measure a hypothesis about the product rather than a trading objective.
PRODUCT_HYPOTHESES = ("rule_adherence@1", "risk_budget@1")


def simple_return(before: float, after: float) -> float:
    """``(after - before) / before``; 0 when ``before`` is not above zero."""
    return (after - before) / before if before > 0 else 0.0


def pnl(before: float, after: float, refused: bool = False, exposure: float = 0.0) -> float:
    """``pnl@1``: the step's change of equity, in the account's money."""
    return after - before


def log_return(before: float, after: float, refused: bool = False, exposure: float = 0.0) -> float:
    """``log_return@1``: ``log(after / before)``; -1 when either is not above zero."""
    return math.log(after / before) if before > 0 and after > 0 else -1.0


def rule_adherence(before: float, after: float, refused: bool = False, exposure: float = 0.0) -> float:
    """``rule_adherence@1``: -1 for an order the venue refused, else 0."""
    return -1.0 if refused else 0.0


def risk_budget(before: float, after: float, refused: bool = False, exposure: float = 0.0) -> float:
    """``risk_budget@1``: ``-max(0, exposure - 1)``, the margin past the equity."""
    return -max(0.0, exposure - 1.0)


class Dsr:
    """``dsr@1``: the differential Sharpe ratio (module doc), its moments kept across an episode."""

    def __init__(self, eta: float = 0.01):
        self.eta = eta
        self.a = 0.0
        self.b = 0.0

    def __call__(self, before: float, after: float, refused: bool = False, exposure: float = 0.0) -> float:
        return self.step(simple_return(before, after))

    def step(self, r: float) -> float:
        """The ratio for one return, then the moments moved by it."""
        da, db = r - self.a, r * r - self.b
        # The variance's floor is RELATIVE to the second moment: an absolute one answered 0 for most
        # hourly steps of a 0.01-lot position on 10,000, whose returns are near 1e-5.
        var = self.b - self.a * self.a
        dsr = (self.b * da - 0.5 * self.a * db) / var**1.5 if self.b > 0.0 and var > 1e-12 * self.b else 0.0
        self.a += self.eta * da
        self.b += self.eta * db
        return dsr


class DsrDrawdownPenalty(Dsr):
    """``dsr_dd_penalty@1``: ``dsr@1`` less half of the drawdown's growth this step (module doc)."""

    def __init__(self, eta: float = 0.01, penalty: float = 0.5):
        super().__init__(eta)
        self.penalty = penalty
        self.peak: Optional[float] = None
        self.drawdown = 0.0

    def __call__(self, before: float, after: float, refused: bool = False, exposure: float = 0.0) -> float:
        dsr = self.step(simple_return(before, after))
        self.peak = after if self.peak is None else max(self.peak, after)
        dd = (self.peak - after) / self.peak if self.peak > 0 else 0.0
        grew = max(0.0, dd - self.drawdown)
        self.drawdown = dd
        return dsr - self.penalty * grew


class CvarLagrangian:
    """``cvar_lagrangian@1``: the return less half of the episode's expected shortfall at 5 %."""

    def __init__(self, tail: int = 20, weight: float = 0.5):
        self.tail, self.weight = tail, weight
        self.returns: List[float] = []

    def __call__(self, before: float, after: float, refused: bool = False, exposure: float = 0.0) -> float:
        r = simple_return(before, after)
        self.returns.append(r)
        worst = sorted(self.returns)[: max(1, len(self.returns) // self.tail)]
        cvar = -sum(worst) / len(worst)
        return r - self.weight * max(0.0, cvar)


class CostAwareVolScaled:
    """``cost_aware_vol_scaled@1``: the return over the deviation of the last 50 returns."""

    def __init__(self, window: int = 50):
        self.window = window
        self.returns: List[float] = []

    def __call__(self, before: float, after: float, refused: bool = False, exposure: float = 0.0) -> float:
        r = simple_return(before, after)
        self.returns.append(r)
        recent = self.returns[-self.window :]
        sd = float(np.std(recent)) if len(recent) > 1 else 0.0
        return r / sd if sd > 1e-12 else 0.0


_MAKERS: Dict[str, Callable[[], Callable[..., float]]] = {
    "pnl@1": lambda: pnl,
    "log_return@1": lambda: log_return,
    "dsr@1": Dsr,
    "dsr_dd_penalty@1": DsrDrawdownPenalty,
    "cvar_lagrangian@1": CvarLagrangian,
    "cost_aware_vol_scaled@1": CostAwareVolScaled,
    "rule_adherence@1": lambda: rule_adherence,
    "risk_budget@1": lambda: risk_budget,
}


def make(name: str) -> Callable[..., float]:
    """**A reward for one episode**, fresh: ``(before, after, refused, exposure) -> float``. Make a new
    one at each reset, since four of them keep the episode's history."""
    if name not in _MAKERS:
        raise ValueError(f"{name} is not a reward; one of {', '.join(REWARDS)}")
    return _MAKERS[name]()
