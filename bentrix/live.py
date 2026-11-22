"""**The same code on paper and REAL**: a ``Strategy`` and a Gymnasium environment on a
workbench that runs on the market's own clock, through the running BENTRIX.

A paper or REAL workbench lives in the application — its ticks come from the person's MetaTrader
(or the venue the account is on), its clock is the market's — so this module talks to it through the
agents' own door (``bentrix.connect()``, MCP), never through an engine of its own. That has one
consequence that is the point: **an order to a real account goes where every agent's goes**:
proposed in BENTRIX's trade panel for the person's Send (``outcome: awaiting_person``), or, under the
person's agents' One-click, sent within the account's caps (``outcome: sent``); refused ``expired``
past its validity and ``refused_by_cap`` past the caps. Nothing here can reach an account the
agent's pairing was not given.

- ``Remote`` — a running BENTRIX as a *door*: ``call(tool, **args)``, as ``bentrix.local.Local``
  answers it. Everything below needs only that one method, so a laboratory workbench in this
  process drives the same code (which is how it is tested).
- ``run(strategy, bench, symbol, interval)`` — ``on_bar`` once for each candle as it closes (the
  ``data_closed_bars`` stream), never the one still forming; an order's outcome handed to
  ``on_order_event`` rather than raised; every decision written to a journal.
- ``LiveEnv`` — ``bentrix.gym.TradingEnv``'s contract, observation, actions and rewards on the live
  clock: ``reset`` waits for the next close, ``step`` acts and waits for the one after. It acts on its
  market's position on that workbench and closes only that market's positions — give it a workbench
  of its own. ``terminated`` for ruin; ``truncated`` at ``max_steps`` or when no candle closed within
  ``idle_limit`` waits. A candle that closed while the policy was still thinking is said
  (``info["late"]``), never silently skipped. With ``draw="<model>"`` each decision is put on the
  workbench's chart as that model's output ``policy``, anchored at the close it was made at:
  a ``discrete`` or ``continuous`` action as a ``position`` (its target in [-1, 1]), an ``order`` as an
  ``action`` (buy, sell or nothing, with its size). A put the workbench refuses never stops the step;
  it is counted in ``draw_refused``.
- ``Model`` — a policy off the hot path: a callable (or an ONNX file, `UNVERIFIED`) run in a worker
  thread at each close, its answer used only within ``timeout_s``; past it, or on an error, or while
  the previous call still runs, the ``fallback`` action (flat by default), counted by reason.
- ``Retrain`` — retraining on the session calendar: the person's ``fit`` at each new session day or
  week, in a worker thread, the policy swapped when it is done (``serve(..., retrain=)``).
- ``parity`` — the parity report: every decision in a live journal set beside a laboratory workbench
  over the recording the live one filled: the market features the policy was served
  against the ones the laboratory computes at the same close, by the same function
  (``TradingEnv._market``), and with a model, the action it would have taken there.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from bentrix.backtest import Bar, Context, Strategy
from bentrix.gym import TradingEnv, _obs, _plain

#: The outcomes an order on a real account answers: ``sent`` under the person's agents'
#: One-click, ``awaiting_person`` proposed in the trade panel. A laboratory or paper workbench answers
#: none, and its order is ``sent``.
OUTCOMES = ("sent", "awaiting_person")
CADENCES = ("bar_close",)


class Remote:
    """**A running BENTRIX as a door**: ``call(tool, **args)`` over the agents' connection
    (``bentrix.connect()`` when none is given)."""

    def __init__(self, client: Any = None):
        if client is None:
            import bentrix

            client = bentrix.connect()
        self.client = client

    def call(self, tool: str, **args: Any) -> Any:
        return self.client.call(tool, {k: v for k, v in args.items() if v is not None})


def _uri(bench: str, tail: str) -> str:
    home, owner, _, name = bench.removeprefix("bentrix://").split("/")[:4]
    return f"bentrix://{home}/{owner}/bench/{name}/{tail}"


def _read(door: Any, uri: str) -> Any:
    return door.call("resource_read", uri=uri)


def _digest(market: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(market, dtype=np.float32).tobytes()).hexdigest()


class _Closes:
    """**The candles of one market and interval as they close** — ``wait_for`` on the workbench's
    ``data_closed_bars`` stream, from the present on."""

    def __init__(self, door: Any, bench: str, symbol: str, interval: str, timeout_ms: int, idle_limit: Optional[int], before_wait: Optional[Callable[[], Any]]):
        self.door, self.timeout_ms, self.idle_limit, self.before_wait = door, timeout_ms, idle_limit, before_wait
        self.uri = _uri(bench, f"bars/{symbol}/{interval}/closed")
        first = door.call("wait_for", uri=self.uri, timeout_ms=0)
        # No candle has closed yet: every one that closes is after the epoch.
        self.cursor = first.get("next") or "0"
        self.rewound = 0

    def next(self) -> List[list]:
        """The candles closed since the last read, oldest first — waiting for at least one; ``[]``
        once ``idle_limit`` waits answered none."""
        idle = 0
        while True:
            if self.before_wait is not None:
                self.before_wait()
            page = self.door.call("wait_for", uri=self.uri, after=self.cursor, timeout_ms=self.timeout_ms)
            if page.get("rewound"):
                self.rewound += 1
            if page.get("next") is not None:
                self.cursor = page["next"]
            items = page.get("items") or []
            if items:
                return items
            idle += 1
            if self.idle_limit is not None and idle >= self.idle_limit:
                return []


def _outcome(answer: Any) -> str:
    return (answer or {}).get("outcome") or "sent"


class LiveContext(Context):
    """``bentrix.backtest.Context`` on a live workbench: an order's outcome is recorded and handed to
    the strategy's ``on_order_event``, and a refusal is an event in its own words, never an exception
    that ends the run."""

    def __init__(self, door: Any, bench: str, symbol: str, strategy: Strategy):
        super().__init__(door, bench, symbol)
        self._strategy = strategy
        self.events: List[Dict[str, Any]] = []
        self.late = False
        self.bar: Optional[Bar] = None

    def _send(self, what: str, tool: str, **args: Any) -> Dict[str, Any]:
        event: Dict[str, Any] = {"what": what, "time_ms": self.bar.time_ms if self.bar else None, "late": self.late}
        try:
            answer = self._local.call(tool, **args)
            event["outcome"] = _outcome(answer)
            last = (answer or {}).get("last_answer") or {}
            if "refused" in last:
                event["outcome"], event["refused"] = "refused", last.get("reason")
        except Exception as e:  # the tool's refusal, in its own words
            event["outcome"], event["refused"] = "refused", str(e)
        self.events.append(event)
        hook = getattr(self._strategy, "on_order_event", None)
        if hook is not None:
            hook(event, self)
        return event

    def buy(self, volume: float, sl: Optional[float] = None, tp: Optional[float] = None) -> Any:
        return self._send("buy", "order_market", bench=self.bench, symbol=self.symbol, side="buy", volume=volume, sl=sl, tp=tp)

    def sell(self, volume: float, sl: Optional[float] = None, tp: Optional[float] = None) -> Any:
        return self._send("sell", "order_market", bench=self.bench, symbol=self.symbol, side="sell", volume=volume, sl=sl, tp=tp)

    def close_all(self) -> Any:
        return self._send("close_all", "order_close_all", bench=self.bench, symbol=self.symbol)

    def book(self) -> Any:
        return _read(self._local, _uri(self.bench, "book"))


def run(
    strategy: Strategy,
    bench: str,
    symbol: str,
    interval: str = "H1",
    door: Any = None,
    max_bars: Optional[int] = None,
    timeout_ms: int = 30_000,
    idle_limit: Optional[int] = None,
    before_wait: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """**Run ``strategy`` on a live workbench** (module doc) from the next candle that closes, for
    ``max_bars`` candles (forever when ``None``) or until ``idle_limit`` waits answer none. Answers
    how many candles it saw, the journal (each candle's open time and prices, and whether a newer one
    had already closed), the order events and the book."""
    door = door or Remote()
    closes = _Closes(door, bench, symbol, interval, timeout_ms, idle_limit, before_wait)
    ctx = LiveContext(door, bench, symbol, strategy)
    strategy.on_start(ctx)
    journal: List[Dict[str, Any]] = []
    seen = 0
    while max_bars is None or seen < max_bars:
        rows = closes.next()
        if not rows:
            break
        for k, r in enumerate(rows):
            if max_bars is not None and seen >= max_bars:
                break
            ctx.late = k < len(rows) - 1
            ctx.bar = Bar(r[0], r[1], r[2], r[3], r[4], float(r[5]) if len(r) > 5 else 0.0)
            journal.append({"time_ms": r[0], "bar": [r[1], r[2], r[3], r[4]], "late": ctx.late})
            strategy.on_bar(ctx.bar, ctx)
            seen += 1
    strategy.on_end(ctx)
    return {"bars": seen, "journal": journal, "events": ctx.events, "rewound": closes.rewound, "book": ctx.book()}


class LiveEnv(TradingEnv):
    """``bentrix.gym.TradingEnv`` on a live workbench (module doc) — a client of the Rust environment's
    live mode: ``env_reset`` at the close the market reached, ``env_act`` at each close, ``env_observe``
    at the next; the waiting for the close is this side's. A flip on a real account closes first and
    opens only once the close was sent (the Rust environment's own rule)."""

    def __init__(
        self,
        bench: str,
        symbol: str,
        interval: str = "H1",
        window: int = 32,
        action: str = "discrete",
        reward: str = "dsr_dd_penalty@1",
        lots: float = 0.01,
        deadband: float = 0.1,
        ruin: float = 0.5,
        max_steps: Optional[int] = None,
        door: Any = None,
        timeout_ms: int = 30_000,
        idle_limit: Optional[int] = None,
        before_wait: Optional[Callable[[], Any]] = None,
        participation: Optional[float] = None,
        impact: float = 0.0,
        max_points: float = 10_000.0,
        draw: Optional[str] = None,
    ):
        door = door or Remote()
        super().__init__(
            bench, symbol=symbol, interval=interval, window=window, action=action, reward=reward, lots=lots, deadband=deadband, ruin=ruin,
            local=door, participation=participation, impact=impact, max_points=max_points, follow=True, max_steps=max_steps,
        )
        self._closes = _Closes(door, bench, self.symbol, interval, timeout_ms, idle_limit, before_wait)
        self.journal: List[Dict[str, Any]] = []
        self.journal_mark = 0
        self.episode = "live"
        self.late = False
        self.outcome = None
        self.draw = draw
        self.draw_refused: Counter = Counter()

    def _draw_spec(self, action: Any) -> Optional[Dict[str, Any]]:
        """The ``output_spec`` the decision is drawn as, anchored at the open of the candle that just closed."""
        from bentrix.model import Output

        anchor = int(self.cursor - self.length)
        a = _plain(action)
        if self.action_kind == "discrete":
            target = {0: 0.0, 1: 1.0, 2: -1.0}.get(int(a))
            if target is None:
                return None
            return Output.position(self.draw, self.symbol, anchor, target, output="policy", units="fraction", range=(-1, 1), horizon="next_output")
        if self.action_kind == "continuous":
            v = float(np.asarray(a, dtype=np.float64).ravel()[0])
            return Output.position(self.draw, self.symbol, anchor, v, output="policy", units="fraction", range=(-1, 1), horizon="next_output")
        side = int(a.get("side", 0)) if isinstance(a, dict) else 0
        size = float(np.asarray(a.get("size", 0.0)).ravel()[0]) if isinstance(a, dict) else 0.0
        intent = {1: "buy", 2: "sell"}.get(side, "nothing")
        return Output.action(self.draw, self.symbol, anchor, intent, size=size if intent != "nothing" else None, output="policy", horizon="next_output")

    def _draw(self, action: Any) -> None:
        if not self.draw:
            return
        try:
            spec = self._draw_spec(action)
            if spec is None:
                self.draw_refused["not_an_action"] += 1
                return
            self.local.call("output_put", bench=self.bench, output=spec)
        except Exception as e:  # a drawing never stops a decision
            self.draw_refused[type(e).__name__] += 1

    def _hash(self) -> str:
        """The sha256 of this episode's decisions — what this side served and did; a live workbench's
        deals are the market's and the person's too, so the journal is the run's own record."""
        lines = [json.dumps(d, sort_keys=True) for d in self.journal[self.journal_mark :]]
        return hashlib.sha256("\n".join(lines).encode()).hexdigest()

    def _wait(self) -> bool:
        rows = self._closes.next()
        if not rows:
            return False
        self.late = len(rows) > 1
        self.cursor = rows[-1][0] + self.length
        return True

    def _serve(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self._served = (self.cursor, _digest(obs["market"]), obs["account"].tolist())
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        try:
            import gymnasium  # noqa: F401

            super(TradingEnv, self).reset(seed=seed)
        except ImportError:
            pass
        self.late = False
        if not self._wait():
            raise RuntimeError(f"no candle of {self.symbol} {self.interval} closed on {self.bench}")
        self.journal_mark = len(self.journal)
        got = self.local.call("env_reset", env=self.env_id, at_ms=self.cursor)
        self.opening = got["info"].get("opening", 1.0)
        info = dict(got["info"], ledger_hash=self._hash(), late=self.late)
        return self._serve(_obs(got["obs"])), info

    def step(self, action):
        acted = self.local.call("env_act", env=self.env_id, action=_plain(action))
        self.outcome, refused = acted.get("outcome"), acted.get("refused")
        at, features, account = self._served
        self.journal.append({"time_ms": at, "features": features, "account": account, "action": _plain(action), "outcome": self.outcome, "refused": refused})
        self._draw(action)
        moved = self._wait()
        got = self.local.call("env_observe", env=self.env_id, at_ms=self.cursor, moved=moved)
        info = dict(got["info"], ledger_hash=self._hash(), late=self.late, outcome=self.outcome, refused=refused)
        return self._serve(_obs(got["obs"])), float(got["reward"]), bool(got["terminated"]), bool(got["truncated"]), info


class Model:
    """**A policy off the hot path** (module doc): ``Model(policy)(obs)`` answers the policy's action
    within ``timeout_s``, else ``fallback``. ``policy`` is a callable of the observation, or the path
    of an ONNX file (``pip install onnxruntime``; its one input the flattened observation, an output of
    more than one value read as scores and its largest taken). With ``engine="bentrix"`` and a ``door``,
    ``policy`` is an ONNX model's name in the person's models folder, run by Bentrix's own engine in its
    model host (``model_infer``) — no onnxruntime in this process, and the person's tick on the file."""

    def __init__(self, policy: Any, timeout_s: float = 0.25, fallback: Any = 0, cadence: str = "bar_close",
                 engine: str = "own", door: Any = None):
        if cadence not in CADENCES:
            raise ValueError(f"{cadence} is not a cadence; one of {', '.join(CADENCES)}")
        if engine not in ("own", "bentrix"):
            raise ValueError("engine is own or bentrix")
        if engine == "bentrix":
            if door is None or not isinstance(policy, str):
                raise ValueError('engine="bentrix" runs a model of the folder by its name, through a door')
            policy = _bentrix_policy(door, policy)
        self.engine = engine
        self.policy = _onnx(policy) if isinstance(policy, str) else policy
        self.timeout_s, self.fallback, self.cadence = timeout_s, fallback, cadence
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bentrix-model")
        self._pending = None
        self.calls = 0
        self.fallbacks: Counter = Counter()

    def _fall(self, why: str) -> Any:
        self.fallbacks[why] += 1
        return self.fallback

    def __call__(self, obs: Any) -> Any:
        self.calls += 1
        if self._pending is not None and not self._pending.done():
            return self._fall("busy")
        self._pending = self._pool.submit(self.policy, obs)
        try:
            return self._pending.result(timeout=self.timeout_s)
        except _FutureTimeout:
            return self._fall("timeout")
        except Exception as e:
            return self._fall(f"error: {type(e).__name__}")


def _flat(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in sorted(obs)])
    return np.asarray(obs, dtype=np.float32).ravel()


def _bentrix_policy(door: Any, name: str) -> Callable[[Any], Any]:
    """A policy run by Bentrix's engine: ``model_infer`` on the flattened observation."""

    def policy(obs: Any) -> Any:
        out = np.asarray(door.call("model_infer", name=name, input=[float(x) for x in _flat(obs)])["output"], dtype=np.float64)
        return int(np.argmax(out)) if out.size > 1 else float(out[0])

    return policy


def _onnx(path: str) -> Callable[[Any], Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(path)
    name = session.get_inputs()[0].name

    def policy(obs: Any) -> Any:
        out = np.asarray(session.run(None, {name: _flat(obs)[None, :]})[0]).ravel()
        return int(np.argmax(out)) if out.size > 1 else float(out[0])

    return policy


_DAY_MS = 86_400_000
PERIODS = ("session", "week")


class Retrain:
    """**Retraining on the session calendar, off the hot path**: at the first close of each new session
    day (``every="session"``) or week (``"week"``, Monday to Sunday) on the workbench's clock, the
    person's ``fit(journal)`` — the decisions so far, a copy — runs in a worker thread, and the
    policy it answers replaces the model's once it is done; the decisions in between keep the policy
    they had. A retrain still running when the next falls due is skipped and counted, never queued
    behind it; one that raises is counted by its kind and the policy stays. The day is the clock's
    own: a candle's time is on the workbench's zone, so a day here is the bench's day."""

    def __init__(self, fit: Callable[[List[Dict[str, Any]]], Callable[[Any], Any]], every: str = "session"):
        if every not in PERIODS:
            raise ValueError(f"{every} is not a period; one of {', '.join(PERIODS)}")
        self.fit, self.every = fit, every
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bentrix-retrain")
        self._pending = None
        self._period: Optional[int] = None
        self.retrains = 0
        self.skipped = 0
        self.failed: Counter = Counter()

    def period(self, t_ms: int) -> int:
        day = t_ms // _DAY_MS
        # The epoch was a Thursday: three days on, a week starts on a Monday.
        return day if self.every == "session" else (day + 3) // 7

    def _swap(self, model: "Model", fut) -> None:
        try:
            model.policy = fut.result()
            self.retrains += 1
        except Exception as e:
            self.failed[type(e).__name__] += 1

    def tick(self, model: "Model", journal: List[Dict[str, Any]], t_ms: int) -> bool:
        """At the close ``t_ms``: whether a retrain was started. The first close only marks its period."""
        p = self.period(t_ms)
        if self._period is None or p == self._period:
            self._period = p
            return False
        self._period = p
        if self._pending is not None and not self._pending.done():
            self.skipped += 1
            return False
        self._pending = self._pool.submit(self.fit, list(journal))
        self._pending.add_done_callback(lambda fut: self._swap(model, fut))
        return True

    def wait(self, timeout: Optional[float] = None) -> None:
        """Until the retrain under way, if any, is done (for a test or the end of a run)."""
        if self._pending is not None:
            try:
                self._pending.result(timeout=timeout)
            except Exception:
                pass


def serve(model: Model, env: LiveEnv, max_steps: Optional[int] = None, retrain: Optional[Retrain] = None) -> Dict[str, Any]:
    """**A model on a live environment**: reset, then each close the model's action (or its fallback)
    stepped, until the environment ends or ``max_steps``; with ``retrain``, its calendar ticked at each
    close. Answers the steps, the model's calls and its fallbacks by reason, the retrains, and the
    journal."""
    obs, _ = env.reset()
    steps = 0
    while max_steps is None or steps < max_steps:
        obs, _, terminated, truncated, _ = env.step(model(obs))
        steps += 1
        if retrain is not None:
            retrain.tick(model, env.journal, env.cursor)
        if terminated or truncated:
            break
    out = {"steps": steps, "calls": model.calls, "fallbacks": dict(model.fallbacks), "journal": env.journal}
    if retrain is not None:
        retrain.wait()
        out["retrain"] = {"retrains": retrain.retrains, "skipped": retrain.skipped, "failed": dict(retrain.failed)}
    return out


@dataclass
class Parity:
    """**The parity report** (module doc)."""

    decisions: int
    compared: int
    #: Decisions whose served features equal the laboratory's, bit for bit.
    equal: int
    #: The decisions' times whose features differ, and the largest difference in any feature.
    differ: List[int] = field(default_factory=list)
    worst: float = 0.0
    #: Decisions the laboratory workbench does not reach (its days end before, or begin after).
    unreachable: int = 0
    actions_compared: int = 0
    actions_equal: int = 0

    @property
    def within(self) -> bool:
        return self.compared > 0 and not self.differ and self.actions_equal == self.actions_compared


def parity(journal: List[Dict[str, Any]], local: Any, bench: str, symbol: str, interval: str = "H1", window: int = 32, model: Optional[Callable[[Any], Any]] = None) -> Parity:
    """**The live journal against a laboratory workbench** over the recording the live one filled
    (module doc). The laboratory workbench is taken back to its start, its deals undone, and
    moved forward to each decision's close: give it one made for the comparison. A journal from
    ``run`` compares the candle each decision saw; one from ``LiveEnv`` the features — the Rust
    environment's own observation at that close, as a step there would read it — and, with ``model``,
    the action."""
    env = TradingEnv(bench, symbol=symbol, interval=interval, window=window, local=local, episode="tape")
    env.reset(seed=0)
    local.call("replay_seek", bench=bench, to_ms=env.first, undo_deals=True)
    report = Parity(len(journal), 0, 0)
    for d in sorted(journal, key=lambda d: d["time_ms"]):
        at = d["time_ms"]
        if "bar" in d:
            at += env.length  # a candle is known at its close
        if at < env.first or at > env.last:
            report.unreachable += 1
            continue
        reached = local.advance(bench, at)
        if reached < at:
            report.unreachable += 1
            continue
        report.compared += 1
        if "bar" in d:
            rows = [r for r in local.call("bars_get", bench=bench, symbol=symbol, timeframe=interval, limit=3).get("bars", []) if r[0] + env.length <= at]
            lab = rows[-1] if rows and rows[-1][0] == d["time_ms"] else None
            gap = max(abs(a - b) for a, b in zip(lab[1:5], d["bar"])) if lab else float("inf")
            if gap == 0.0:
                report.equal += 1
            else:
                report.differ.append(d["time_ms"])
                report.worst = max(report.worst, gap)
            continue
        market = env.observe_at(at)["market"]
        if _digest(market) == d["features"]:
            report.equal += 1
        else:
            report.differ.append(at)
        if model is not None:
            report.actions_compared += 1
            a = model({"market": market, "account": np.asarray(d["account"], dtype=np.float32)})
            a = {k: np.asarray(v).tolist() for k, v in a.items()} if isinstance(a, dict) else np.asarray(a).tolist()
            report.actions_equal += int(a == d["action"])
    return report
