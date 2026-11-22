"""**The Gym**: a Gymnasium environment over a BENTRIX workbench.

**The environment is BENTRIX's, in Rust** (``env_make``, ``env_reset``, ``env_step`` …), so every language reaches the same one, and ``TradingEnv``
here is its client: through your own process's engine (``bentrix.local``) or through a running
BENTRIX (``bentrix.live.Remote``), whichever ``local`` is. What it computes is the Rust module's
(``crates/bentrix-shell/src/agenttools/env.rs``); its doc is the reference, in brief:

- **Contract** — Gymnasium's: ``reset(seed, options) -> (obs, info)``; ``step(a) -> (obs, reward,
  terminated, truncated, info)``. ``terminated`` only for ruin (equity at or below ``ruin`` of the
  opening equity); ``truncated`` for the end of the episode or of the tape.
- **Observation** — a ``Dict``: ``market`` (the last ``window`` closed candles' open, high, low and
  close as log ratios to the last close) and ``account`` (the position's side, the equity's log ratio
  to the episode's start, the open profit over the opening equity). ``gymnasium.wrappers.
  FlattenObservation`` makes it a vector for SB3 or CleanRL. **Leakage is impossible by the cursor**:
  only candles closed at or before it are read.
- **Actions** — ``discrete`` (0 flat, 1 long, 2 short), ``continuous`` (a target in [-1, 1]: its sign
  the side, its size the share of ``lots``; under ``deadband`` flat) or ``order`` (a ``Dict`` of
  ``side``, ``size``, and ``stop`` and ``target`` in points from the entry quote, 0 for none). An
  order the venue refuses is **refused with its reason in ``info["refused"]``, never clipped into
  another**.
- **Participation** (optional, and a model) — ``participation`` caps an order's units at that share of
  the last closed candle's volume column (a tick count where a recording has no traded units, which
  makes the cap arithmetic, not a market fact); ``impact`` charges the reward, never the ledger.
- **Rewards** — named, versioned plugins (``REWARDS``): ``pnl@1``, ``log_return@1``, ``dsr@1``,
  ``dsr_dd_penalty@1`` (the default), ``cvar_lagrangian@1``, ``cost_aware_vol_scaled@1``, and two
  **product hypotheses**, ``rule_adherence@1`` and ``risk_budget@1``. Never on a human surface.
- **Episodes** — ``tape``, ``window`` (``episode_bars`` candles from a start the seed draws, a pure
  function of ``(seed, episode index)``) and ``session`` (one day); ``growing(first, last, over)`` a
  curriculum of lengthening window episodes. A ``sampler`` of your own (a callable of the episode
  index, a ``Draw`` and the environment, answering a start and a length) is run here and its choice
  handed over.
- **Determinism** — equal seeds and actions give an equal ``info["ledger_hash"]``; a reset undoes the
  previous episode's deals.
- **Evaluation** — ``evaluate`` runs a policy over ten seeds by default and reports the interquartile
  mean with a bootstrap interval (``iqm``). It runs your policy, so it lives with you.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from bentrix.local import Local

try:  # Gymnasium is optional: without it the class keeps the same contract.
    import gymnasium as _gym
    from gymnasium import spaces as _spaces

    _Base = _gym.Env
except ImportError:  # pragma: no cover - exercised where Gymnasium is absent
    _gym = None
    _spaces = None
    _Base = object

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
PRODUCT_HYPOTHESES = ("rule_adherence@1", "risk_budget@1")
EPISODES = ("tape", "window", "session")
ACTIONS = ("discrete", "continuous", "order")
DAY_MS = 86_400_000
_MASK = (1 << 64) - 1


def _mix(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


class Draw:
    """**An episode's draws, a pure function of ``(seed, episode index)``** — SplitMix64, the Rust
    environment's own (``env.rs``'s ``Draw``), for a sampler of your own. ``randrange(n)`` as
    ``random.Random``'s."""

    def __init__(self, seed: int, index: int):
        self.state = _mix((seed & _MASK) ^ ((index * 0x9E3779B97F4A7C15) & _MASK))

    def randrange(self, n: int) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        return _mix(self.state) % n


_UNIT_MS = {"S": 1_000, "M": 60_000, "H": 3_600_000, "D": DAY_MS, "W": 7 * DAY_MS}


def interval_ms(label: str) -> int:
    """An interval's length in ms (``M1``, ``H4``, ``D1``, ``W1``, a seconds interval ``S30``) — the
    engine's own reading where the extension module is present, the same table without it; ``0`` for
    one with no fixed length (``MN1``)."""
    try:
        from bentrix import _native

        return int(_native.interval_ms(label) or 0)
    except ImportError:
        unit, n = label[:1], label[1:]
        return _UNIT_MS[unit] * int(n) if unit in _UNIT_MS and n.isdigit() and not label.startswith("MN") else 0


def growing(first: int, last: int, over: int) -> Dict[str, Dict[str, int]]:
    """**A curriculum of lengthening episodes**: the ``i``-th window episode lasts ``first`` candles
    growing evenly to ``last`` over ``over`` episodes, from a start the seed draws — computed by the
    environment (``env_make``'s ``growing``)."""
    return {"growing": {"first": int(first), "last": int(last), "over": int(over)}}


def _plain(a: Any) -> Any:
    """An action as JSON: a NumPy value or array as numbers, a ``Dict`` action field by field."""
    if isinstance(a, dict):
        return {k: _plain(v) for k, v in a.items()}
    return np.asarray(a).tolist()


def _obs(o: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {"market": np.asarray(o["market"], dtype=np.float32), "account": np.asarray(o["account"], dtype=np.float32)}


class TradingEnv(_Base):
    """A workbench as a Gymnasium environment (module doc) — a client of Bentrix's own."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        bench: str,
        symbol: Optional[str] = None,
        interval: str = "H1",
        window: int = 32,
        action: str = "discrete",
        reward: str = "dsr_dd_penalty@1",
        lots: float = 0.01,
        episode: str = "window",
        episode_bars: int = 200,
        deadband: float = 0.1,
        ruin: float = 0.5,
        local: Any = None,
        sampler: Union[None, Dict[str, Any], Callable[[int, Draw, "TradingEnv"], Tuple[int, int]]] = None,
        participation: Optional[float] = None,
        impact: float = 0.0,
        max_points: float = 10_000.0,
        follow: bool = False,
        max_steps: Optional[int] = None,
    ):
        if action not in ACTIONS:
            raise ValueError(f"{action} is not an action kind; one of {', '.join(ACTIONS)}")
        if episode not in EPISODES:
            raise ValueError(f"{episode} is not an episode kind; one of {', '.join(EPISODES)}")
        if reward not in REWARDS:
            raise ValueError(f"{reward} is not a reward plugin; one of {', '.join(REWARDS)}")
        if participation is not None and not 0.0 < participation <= 1.0:
            raise ValueError(f"participation {participation} is not a share in (0, 1]")
        self.local = local if local is not None else Local()
        if hasattr(self.local, "arm") and not follow:
            try:
                self.local.arm(bench)
            except RuntimeError:
                pass  # a live workbench has no replay to arm; the environment says what it is
        self.sampler = sampler if callable(sampler) else None
        curriculum = sampler.get("growing") if isinstance(sampler, dict) else None
        made = self.local.call(
            "env_make",
            bench=bench,
            symbol=symbol,
            interval=interval,
            window=window,
            action=action,
            reward=reward,
            lots=lots,
            episode=episode,
            episode_bars=episode_bars,
            growing=curriculum,
            deadband=deadband,
            ruin=ruin,
            participation=participation,
            impact=impact if participation is not None else None,
            max_points=max_points,
            follow=follow or None,
            max_steps=max_steps,
        )
        self.env_id = made["env"]
        self.bench, self.symbol, self.interval, self.window = bench, made["symbol"], interval, window
        self.action_kind, self.reward_name, self.episode, self.episode_bars = action, reward, episode, episode_bars
        self.live = bool(made.get("live"))
        self.first, self.last, self.length = made.get("first_ms"), made.get("last_ms"), made["length_ms"]
        self.episode_index = 0
        self.cursor = self.end = None
        self.opening = 1.0
        if _spaces is not None:
            self.observation_space = _spaces.Dict(
                {
                    "market": _spaces.Box(-np.inf, np.inf, shape=(window, 4), dtype=np.float32),
                    "account": _spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                }
            )
            if action == "discrete":
                self.action_space = _spaces.Discrete(3)
            elif action == "continuous":
                self.action_space = _spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            else:
                points = _spaces.Box(0.0, max_points, shape=(1,), dtype=np.float32)
                self.action_space = _spaces.Dict({"side": _spaces.Discrete(3), "size": _spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32), "stop": points, "target": points})

    def close(self) -> None:
        try:
            self.local.call("env_close", env=self.env_id)
        except Exception:
            pass

    def _started(self, got: Dict[str, Any]):
        info = got["info"]
        self.cursor = info.get("start_ms")
        self.end = info.get("end_ms")
        self.opening = info.get("opening", 1.0)
        return _obs(got["obs"]), info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if _gym is not None:
            super().reset(seed=seed)
        if seed is not None:
            self.seed_given, self.episode_index = seed, 0
        start_ms = bars = None
        if self.sampler is not None and self.episode == "window":
            rng = Draw(getattr(self, "seed_given", None) or 0, self.episode_index)
            start_ms, bars = self.sampler(self.episode_index, rng, self)
        self.episode_index += 1
        return self._started(self.local.call("env_reset", env=self.env_id, seed=seed, start_ms=start_ms, bars=bars))

    def step(self, action):
        got = self.local.call("env_step", env=self.env_id, action=_plain(action))
        self.cursor = got["info"].get("cursor_ms", self.cursor)
        return _obs(got["obs"]), float(got["reward"]), bool(got["terminated"]), bool(got["truncated"]), got["info"]

    def observe_at(self, at_ms: int) -> Dict[str, np.ndarray]:
        """**The observation a step would answer at the close ``at_ms``** — the workbench standing
        there (``env_observe``); what ``bentrix.live.parity`` sets beside a live decision."""
        self.cursor = at_ms
        return _obs(self.local.call("env_observe", env=self.env_id, at_ms=at_ms)["obs"])


def register() -> None:
    """Register ``bentrix/TradingEnv-v0`` with Gymnasium (``gymnasium.make("bentrix/TradingEnv-v0",
    bench=...)``)."""
    if _gym is None:
        raise RuntimeError("Gymnasium is not installed: pip install gymnasium")
    if "bentrix/TradingEnv-v0" not in _gym.registry:
        _gym.register(id="bentrix/TradingEnv-v0", entry_point="bentrix.gym:TradingEnv")


def make_vec(benches: list, local: Optional[Local] = None, **kwargs):
    """**One environment per workbench, stepped together** — a ``SyncVectorEnv`` over one embedded
    BENTRIX, each environment on its own workbench (benches made alike are the same conditions)."""
    if _gym is None:
        raise RuntimeError("Gymnasium is not installed: pip install gymnasium")
    local = local or Local()
    return _gym.vector.SyncVectorEnv([(lambda b=b: TradingEnv(b, local=local, **kwargs)) for b in benches])


def iqm(scores: Sequence[float]) -> float:
    """**The interquartile mean**: the mean of the middle half, a quarter cut from each end (Agarwal et
    al., 2021) — robust to one lucky or unlucky seed where the mean is not, and less wasteful than the
    median. Fewer than four scores have no quarter to cut: their mean."""
    x = np.sort(np.asarray(scores, dtype=np.float64))
    k = len(x) // 4
    return float(x[k : len(x) - k].mean()) if len(x) else float("nan")


def evaluate(policy: Callable[[Any], Any], env: TradingEnv, seeds: Sequence[int] = range(10), confidence: float = 0.95, resamples: int = 2000) -> Dict[str, Any]:
    """**A policy over seeds, reported by IQM** (module doc): one episode per seed, its score the log
    return of the equity and its reward's sum beside it; the IQM of the scores with a percentile
    bootstrap interval (``resamples`` draws, its own generator seeded 0, so a report is reproducible).
    Fewer than ten seeds is reported and said: its interval is wide."""
    scores: List[float] = []
    rewards: List[float] = []
    hashes: List[str] = []
    for seed in seeds:
        obs, info = env.reset(seed=int(seed))
        total, done = 0.0, False
        while not done:
            obs, r, terminated, truncated, info = env.step(policy(obs))
            total += r
            done = terminated or truncated
        scores.append(math.log(info["equity"] / env.opening) if info["equity"] > 0 and env.opening > 0 else float("-inf"))
        rewards.append(total)
        hashes.append(info["ledger_hash"])
    x = np.asarray(scores, dtype=np.float64)
    draw = np.random.default_rng(0)
    boot = [iqm(x[draw.integers(0, len(x), len(x))]) for _ in range(resamples)] if len(x) else []
    lo, hi = np.quantile(boot, [(1 - confidence) / 2, (1 + confidence) / 2]).tolist() if boot else (float("nan"), float("nan"))
    return {
        "seeds": [int(s) for s in seeds],
        "scores": scores,
        "rewards": rewards,
        "ledger_hashes": hashes,
        "iqm": iqm(x),
        "interval": [lo, hi],
        "confidence": confidence,
        "mean": float(x.mean()) if len(x) else float("nan"),
        "median": float(np.median(x)) if len(x) else float("nan"),
        "too_few_seeds": len(x) < 10,
    }
