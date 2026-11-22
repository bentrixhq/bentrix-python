"""**The Gym as it was written in Python, kept as the walks' reference** (frozen
2026-11-01 when the environment moved whole into Rust, `crates/bentrix-shell/src/agenttools/env.rs`).

Not the product: ``bentrix.gym.TradingEnv`` is a client of the Rust environment. This file is the
implementation that environment replaced, unchanged but for this docstring, so
``test_gym_reference.py`` can hold the Rust environment to it — the same observations, rewards,
refusals and ledger hash, step for step, for every action kind, reward, the curriculum and the
participation cap. A difference is a defect in the port, or a decision written down beside it.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    """**An episode's draws, a pure function of ``(seed, episode index)``** — SplitMix64, written the
    same way in the environment an agent reaches over MCP (``env_reset``), so one seed draws one
    episode in both. ``randrange(n)`` as ``random.Random``'s, for a sampler."""

    def __init__(self, seed: int, index: int):
        self.state = _mix((seed & _MASK) ^ ((index * 0x9E3779B97F4A7C15) & _MASK))

    def randrange(self, n: int) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        return _mix(self.state) % n


_UNIT_MS = {"S": 1_000, "M": 60_000, "H": 3_600_000, "D": DAY_MS, "W": 7 * DAY_MS}


def interval_ms(label: str) -> int:
    """An interval's length in ms (``M1``, ``H4``, ``D1``, ``W1``, a seconds interval ``S30``) — the
    engine's own reading where the extension module is present, the same table without it (a script
    on a running BENTRIX needs no engine in its process); ``0`` for one with no fixed length (``MN1``)."""
    try:
        from bentrix import _native

        return int(_native.interval_ms(label) or 0)
    except ImportError:
        unit, n = label[:1], label[1:]
        return _UNIT_MS[unit] * int(n) if unit in _UNIT_MS and n.isdigit() and not label.startswith("MN") else 0


class _Reward:
    """Every plugin's state, kept across one episode's steps."""

    def __init__(self, name: str):
        if name not in REWARDS:
            raise ValueError(f"{name} is not a reward plugin; one of {', '.join(REWARDS)}")
        self.name = name
        self.a = 0.0  # DSR's first moment
        self.b = 0.0  # DSR's second moment
        self.eta = 0.01
        self.peak = None
        self.drawdown = 0.0
        self.returns: list = []

    def __call__(self, before: float, after: float, refused: bool, exposure: float) -> float:
        r = (after - before) / before if before > 0 else 0.0
        self.returns.append(r)
        if self.name == "pnl@1":
            return after - before
        if self.name == "log_return@1":
            return math.log(after / before) if before > 0 and after > 0 else -1.0
        if self.name in ("dsr@1", "dsr_dd_penalty@1"):
            da, db = r - self.a, r * r - self.b
            # The variance's floor is RELATIVE to the second moment: an absolute one (it was 1e-12 on
            # the variance to the 1.5) answered 0 for most hourly steps of a 0.01-lot position on
            # 10,000, whose returns are near 1e-5 — the DSR scaled out of existence by the lot size.
            var = self.b - self.a * self.a
            dsr = (self.b * da - 0.5 * self.a * db) / var**1.5 if self.b > 0.0 and var > 1e-12 * self.b else 0.0
            self.a += self.eta * da
            self.b += self.eta * db
            if self.name == "dsr@1":
                return dsr
            self.peak = after if self.peak is None else max(self.peak, after)
            dd = (self.peak - after) / self.peak if self.peak > 0 else 0.0
            grew = max(0.0, dd - self.drawdown)
            self.drawdown = dd
            return dsr - 0.5 * grew
        if self.name == "cvar_lagrangian@1":
            tail = sorted(self.returns)[: max(1, len(self.returns) // 20)]
            cvar = -sum(tail) / len(tail)
            return r - 0.5 * max(0.0, cvar)
        if self.name == "cost_aware_vol_scaled@1":
            recent = self.returns[-50:]
            sd = float(np.std(recent)) if len(recent) > 1 else 0.0
            return r / sd if sd > 1e-12 else 0.0
        if self.name == "rule_adherence@1":
            return -1.0 if refused else 0.0
        if self.name == "risk_budget@1":
            return -max(0.0, exposure - 1.0)
        raise AssertionError(self.name)


class TradingEnv(_Base):
    """A laboratory workbench as a Gymnasium environment (module doc)."""

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
        local: Optional[Local] = None,
        sampler: Optional[Callable[[int, Draw, "TradingEnv"], Tuple[int, int]]] = None,
        participation: Optional[float] = None,
        impact: float = 0.0,
        max_points: float = 10_000.0,
    ):
        self._configure(interval, window, action, reward, lots, deadband, ruin, participation, impact, max_points)
        self.sampler = sampler
        if episode not in EPISODES:
            raise ValueError(f"{episode} is not an episode kind; one of {', '.join(EPISODES)}")
        self.local = local or Local()
        self.bench, self.episode, self.episode_bars = bench, episode, episode_bars
        self.local.arm(bench)
        state = self.local.call("replay_step", bench=bench, unit="tick", n=1)
        markets = state.get("markets") or []
        if symbol is None:
            if not markets:
                raise ValueError(f"{bench} holds no market")
            symbol = markets[0]
        self.symbol = symbol
        bounds = state.get("bounds") or {}
        self.first, self.last = bounds.get("from_ms"), bounds.get("to_ms")
        home, owner, _, name = bench.removeprefix("bentrix://").split("/")[:4]
        self._vars = {"home": home, "owner": owner, "bench": name}
        self.episode_index = 0

    def _configure(self, interval: str, window: int, action: str, reward: str, lots: float, deadband: float, ruin: float, participation: Optional[float] = None, impact: float = 0.0, max_points: float = 10_000.0) -> None:
        """What a laboratory environment and a live one share: the interval, the observation, the action
        and the reward, refused by name at construction rather than at the first step."""
        if action not in ACTIONS:
            raise ValueError(f"{action} is not an action kind; one of {', '.join(ACTIONS)}")
        _Reward(reward)
        self.interval, self.window, self.action_kind = interval, window, action
        self.reward_name, self.lots, self.deadband, self.ruin = reward, lots, deadband, ruin
        if participation is not None and not 0.0 < participation <= 1.0:
            raise ValueError(f"participation {participation} is not a share in (0, 1]")
        self.participation, self.impact, self.max_points = participation, impact, max_points
        self.participated: Optional[dict] = None
        self._spec_cache: Optional[dict] = None
        self.length = interval_ms(interval)
        if not self.length:
            raise ValueError(f"{interval} has no fixed length")
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

    # ---- the workbench -------------------------------------------------------------------------

    def _account(self) -> Dict[str, Any]:
        return self.local.read("trade_account", **self._vars)

    def _positions(self) -> list:
        """This market's open positions on the workbench — the only ones the environment acts on."""
        return [p for p in self.local.read("trade_book", **self._vars).get("positions", []) if p.get("symbol", self.symbol) == self.symbol]

    def _deals(self) -> list:
        return self.local.read("trade_deals", **self._vars).get("items", [])

    def _closed(self, reached: int) -> list:
        rows = self.local.call("bars_get", bench=self.bench, symbol=self.symbol, timeframe=self.interval, limit=self.window + 2).get("bars", [])
        return [r for r in rows if r[0] + self.length <= reached][-self.window :]

    def _market(self) -> np.ndarray:
        """**The market half of an observation** — the last ``window`` candles closed at the cursor, as
        log ratios to the last close. One function for the laboratory and the live environment, so the
        features a policy was trained on are the features it is served (``bentrix.live.parity``)."""
        rows = self._closed(self.cursor)
        market = np.zeros((self.window, 4), dtype=np.float32)
        if rows:
            last = rows[-1][4]
            arr = np.array([[math.log(r[k] / last) if r[k] > 0 and last > 0 else 0.0 for k in (1, 2, 3, 4)] for r in rows], dtype=np.float32)
            market[-len(arr) :] = arr
        return market

    def _obs(self) -> Dict[str, np.ndarray]:
        market = self._market()
        pos = self._positions()
        side = 0.0 if not pos else (1.0 if pos[0]["side"] == "buy" else -1.0)
        acct = self._account()
        eq = acct.get("equity") or self.opening
        open_pnl = sum(p.get("profit", 0.0) for p in pos)
        account = np.array([side, math.log(eq / self.opening) if eq > 0 else -10.0, open_pnl / self.opening], dtype=np.float32)
        return {"market": market, "account": account}

    def _spec(self) -> dict:
        """The market's contract (``symbol_info``): its point, contract size and volume limits."""
        if self._spec_cache is None:
            self._spec_cache = self.local.call("symbol_info", bench=self.bench, symbol=self.symbol)
        return self._spec_cache

    def _quote(self) -> Tuple[Optional[float], Optional[float]]:
        """The market's bid and ask at the cursor (``data_quotes``)."""
        home, owner, _, name = self.bench.removeprefix("bentrix://").split("/")[:4]
        page = self.local.call("resource_read", uri=f"bentrix://{home}/{owner}/bench/{name}/quotes")
        q = next((i for i in page.get("items", []) if i.get("symbol") == self.symbol), {})
        return q.get("bid"), q.get("ask")

    def _target(self, a) -> tuple:
        self._stops = (0.0, 0.0)
        if self.action_kind == "order":
            side = {0: 0, 1: 1, 2: -1}[int(np.asarray(a["side"]).reshape(-1)[0])]
            size = float(np.clip(np.asarray(a["size"]).reshape(-1)[0], 0.0, 1.0))
            self._stops = tuple(float(np.clip(np.asarray(a[k]).reshape(-1)[0], 0.0, self.max_points)) for k in ("stop", "target"))
            if side == 0 or size <= 0.0:
                return (0, 0.0)
            spec = self._spec()
            step, least = spec.get("volume_step") or 0.01, spec.get("volume_min") or 0.01
            return (side, max(least, round(math.floor(self.lots * size / step + 1e-9) * step, 8)))
        if self.action_kind == "discrete":
            return {0: (0, 0.0), 1: (1, self.lots), 2: (-1, self.lots)}[int(a)]
        x = float(np.asarray(a).reshape(-1)[0])
        if abs(x) < self.deadband:
            return (0, 0.0)
        return (1 if x > 0 else -1, round(self.lots * min(1.0, abs(x)), 2) or self.lots)

    def _hash(self) -> str:
        lines = [f"{d['time_ms']}|{d['side']}|{d['entry']}|{d['price']:.10g}|{d['volume']:.4g}" for d in self._deals()[self.deals_before :]]
        return hashlib.sha256("\n".join(lines).encode()).hexdigest()

    # ---- the contract ----------------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if _gym is not None:
            super().reset(seed=seed)
        # A seed given starts the sampler over (Gymnasium's contract: the same seed, the same episode);
        # a reset with none takes the next episode of the last seed.
        if seed is not None:
            self.seed_given, self.episode_index = seed, 0
        rng = Draw(getattr(self, "seed_given", None) or 0, self.episode_index)
        self.episode_index += 1
        span = self.last - self.first
        warm = (self.window + 1) * self.length
        if self.episode == "tape":
            start = self.first + warm
        elif self.episode == "session":
            days = max(1, (span - warm) // DAY_MS)
            start = self.first + warm + rng.randrange(days) * DAY_MS
        elif self.sampler is not None:
            start, self.bars = self.sampler(self.episode_index - 1, rng, self)
        else:
            room = max(0, span - warm - self.episode_bars * self.length)
            start = self.first + warm + (rng.randrange(room // self.length + 1) * self.length if room else 0)
        start = (start // self.length) * self.length
        self.local.call("replay_seek", bench=self.bench, to_ms=self.first, undo_deals=True)
        self.cursor = self.local.advance(self.bench, start)
        self.deals_before = len(self._deals())
        self.opening = self._account().get("equity") or 1.0
        bars = self.bars if self.sampler is not None and self.episode == "window" else self.episode_bars
        self.end = {"tape": self.last, "session": start + DAY_MS, "window": start + bars * self.length}[self.episode]
        self.reward = _Reward(self.reward_name)
        return self._obs(), {"reward": self.reward_name, "ledger_hash": self._hash(), "start_ms": start}

    def _participate(self, volume: float) -> Tuple[float, Optional[str]]:
        """``volume`` cut to the participation cap (module doc), said in ``self.participated``; the
        refusal when the cut leaves less than the market's least volume."""
        self.participated = None
        if self.participation is None:
            return volume, None
        spec = self._spec()
        rows = self._closed(self.cursor)
        traded = float(rows[-1][5]) if rows and len(rows[-1]) > 5 else 0.0
        units = volume * spec["contract_size"]
        allowed = self.participation * traded
        self.participated = {"asked": units, "allowed": allowed, "candle_volume": traded, "modelled": True}
        if units <= allowed:
            self.participated["impact"] = self.impact * units / traded if traded > 0 else 0.0
            return volume, None
        step = spec.get("volume_step") or 0.01
        cut = math.floor(allowed / spec["contract_size"] / step + 1e-9) * step
        if cut < (spec.get("volume_min") or step):
            self.participated["impact"] = 0.0
            return 0.0, f"participation: {units:g} units is more than {self.participation:g} of the last candle's volume ({traded:g}); nothing was sent"
        self.participated["impact"] = self.impact * cut * spec["contract_size"] / traded
        return round(cut, 8), None

    def _stops_for(self, side: int) -> Tuple[Optional[float], Optional[float]]:
        """An order action's stop and target, in price, from the quote it will be filled against."""
        stop, target = getattr(self, "_stops", (0.0, 0.0))
        if not stop and not target:
            return None, None
        bid, ask = self._quote()
        entry, point = (ask if side > 0 else bid), self._spec()["point"]
        if entry is None:
            return None, None
        sl = entry - side * stop * point if stop else None
        tp = entry + side * target * point if target else None
        return sl, tp

    def _act(self, side: int, volume: float) -> Optional[str]:
        """Bring the position to ``side`` at ``volume``; the venue's refusal in its words, or ``None``."""
        pos = self._positions()
        held = 0 if not pos else (1 if pos[0]["side"] == "buy" else -1)
        self.participated = None
        if side == held:
            return None
        if pos:
            self.local.call("order_close_all", bench=self.bench, symbol=self.symbol)
        if side != 0:
            volume, refused = self._participate(volume)
            if refused:
                return refused
            sl, tp = self._stops_for(side)
            answer = self.local.call("order_market", bench=self.bench, symbol=self.symbol, side="buy" if side > 0 else "sell", volume=volume, sl=sl, tp=tp)
            last = answer.get("last_answer") or {}
            return last.get("reason") if "refused" in last else None
        return None

    def _next_close(self) -> int:
        """The laboratory's clock moved to the next close of the interval; where it stands."""
        return self.local.advance(self.bench, (self.cursor // self.length + 1) * self.length)

    def step(self, action):
        side, volume = self._target(action)
        before = self._account().get("equity") or self.opening
        refused = self._act(side, volume)
        close = (self.cursor // self.length + 1) * self.length
        self.cursor = self._next_close()
        acct = self._account()
        after = acct.get("equity") or before
        exposure = (acct.get("margin") or 0.0) / after if after > 0 else 0.0
        reward = self.reward(before, after, refused is not None, exposure)
        if self.participated is not None:
            reward -= self.participated.get("impact", 0.0)
        terminated = after <= self.ruin * self.opening
        truncated = self.cursor >= self.end or self.cursor < close
        info = {"reward": self.reward_name, "ledger_hash": self._hash(), "refused": refused, "equity": after, "cursor_ms": self.cursor}
        if self.participated is not None:
            info["participation"] = self.participated
        if self.reward_name in PRODUCT_HYPOTHESES:
            info["product_hypothesis"] = True
        return self._obs(), float(reward), bool(terminated), bool(truncated), info


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


def growing(first: int, last: int, over: int) -> Callable[[int, Draw, TradingEnv], Tuple[int, int]]:
    """**A curriculum of lengthening episodes**: the ``i``-th window episode lasts ``first`` candles
    growing evenly to ``last`` over ``over`` episodes, from a start the seed draws where it fits — a
    pure function of ``(seed, episode index)`` like every sampler."""

    def sampler(i: int, rng: Draw, env: TradingEnv) -> Tuple[int, int]:
        bars = int(round(first + (last - first) * min(1.0, i / max(1, over))))
        warm = (env.window + 1) * env.length
        room = max(0, env.last - env.first - warm - bars * env.length)
        return env.first + warm + (rng.randrange(room // env.length + 1) * env.length if room else 0), bars

    return sampler


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
