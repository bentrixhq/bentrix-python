"""**The same code on paper and REAL**.

A live workbench's clock is the market's, so these walks stand a laboratory workbench of the CI tape in
for it: the code under test reads the same stream (``data_closed_bars`` through ``wait_for``) and the
same resources, and ``before_wait`` moves the tape's clock the way the market would between two reads.
What a real account answers (``awaiting_person``) is a mapping test over a door that answers
it, said as such.
"""

import time

import pytest

np = pytest.importorskip("numpy")

from bentrix.backtest import Strategy  # noqa: E402
from bentrix.live import LiveEnv, Model, Remote, parity, run, serve  # noqa: E402

H1 = 3_600_000


class Market:
    """The tape's clock moved ``closes`` candles of ``length`` at each read — the market between two
    of a slow reader's reads."""

    def __init__(self, local, bench, closes=1, length=H1):
        self.local, self.bench, self.closes, self.length = local, bench, closes, length
        local.arm(bench)
        self.at = local.call("replay_step", bench=bench, unit="tick", n=1)["cursor_ms"]
        self.at = local.advance(bench, (self.at // length + 40) * length)

    def __call__(self):
        self.at = self.local.advance(self.bench, (self.at // self.length + self.closes) * self.length)


class Flip(Strategy):
    def __init__(self):
        self.events = []

    def on_bar(self, bar, ctx):
        n = len(self.events)
        if bar.close > bar.open and n % 2 == 0:
            ctx.buy(0.1)
        elif bar.close < bar.open and n % 2 == 1:
            ctx.close_all()

    def on_order_event(self, event, ctx):
        self.events.append(event)


def test_a_live_strategy_sees_each_close_once_and_its_candles_are_the_laboratorys(tape):
    """**Each candle once, as it closes, never the forming one**, and the parity report finds every one
    of them equal to the laboratory's candle at that close.

    Mutation (2026-11-01): ``_Closes.next`` re-reading from the first cursor (``after``
    not advanced) — red, candles come again; and the journal's time taken as the close rather than the
    open — red, the journal's newest candle is past where the market stands."""
    local, bench = tape
    market = Market(local, bench)
    s = Flip()
    out = run(s, bench, "XAUUSD", "H1", door=local, max_bars=12, before_wait=market, idle_limit=3)
    times = [d["time_ms"] for d in out["journal"]]
    assert out["bars"] == 12 and len(set(times)) == 12, times
    assert all(b - a == H1 for a, b in zip(times, times[1:])), times
    assert not any(d["late"] for d in out["journal"])
    assert times[-1] + H1 <= market.at, "a candle was handed before it closed"
    assert s.events and all(e["outcome"] == "sent" for e in s.events), s.events
    report = parity(out["journal"], local, bench, "XAUUSD", "H1")
    assert (report.compared, report.equal, report.differ) == (12, 12, []), report


def test_a_slow_reader_is_told_a_candle_is_late_and_none_is_skipped(tape):
    """Three candles closed between two reads: all three are handed, oldest first, the first two
    marked ``late``.

    Mutation (2026-11-01): only the newest of a page handed — red, candles are skipped."""
    local, bench = tape
    out = run(Strategy(), bench, "XAUUSD", "H1", door=local, max_bars=6, before_wait=Market(local, bench, closes=3), idle_limit=3)
    assert [d["late"] for d in out["journal"]] == [True, True, False, True, True, False]
    times = [d["time_ms"] for d in out["journal"]]
    assert all(b - a == H1 for a, b in zip(times, times[1:])), times


def _policy(obs):
    """A deterministic policy of the market alone: long after a rising candle, short after a falling one."""
    m = np.asarray(obs["market"])
    return 1 if m[-1, 3] >= m[-1, 0] else 2


def test_a_live_environment_serves_the_features_the_laboratory_computes(tape):
    """**One feature path**: a ``LiveEnv`` stepped by a policy on the tape's clock, then the parity
    report against the laboratory over the same recording — every decision's features equal, bit for
    bit, and the policy's action there the action it took.

    The report compares at the journal's own instants, so a decision standing an hour behind the market
    would be compared an hour behind too; the walk therefore also asserts each decision stands at the
    close the market had reached.

    Mutation (2026-11-01): the live cursor taken at the last candle's OPEN instead of its
    close — survived the parity report alone (both sides lagged together), red on the instants: the
    last decision stands two candles behind the market rather than one."""
    local, bench = tape
    market = Market(local, bench)
    env = LiveEnv(bench, "XAUUSD", "H1", window=16, door=local, before_wait=market, idle_limit=3, max_steps=10)
    out = serve(Model(_policy, timeout_s=5.0), env)
    assert out["steps"] == 10 and out["fallbacks"] == {}, out
    times = [d["time_ms"] for d in env.journal]
    # Each decision at a close; the market moved one candle past the last (the step's own wait).
    assert times[-1] == market.at - H1 and all(b - a == H1 for a, b in zip(times, times[1:])), (times, market.at)
    assert all(d["outcome"] in (None, "sent") for d in env.journal), env.journal
    assert {d["action"] for d in env.journal} == {1, 2}, "the policy never changed its mind; the walk proves little"
    report = parity(env.journal, local, bench, "XAUUSD", "H1", window=16, model=_policy)
    assert report.compared == 10 and report.equal == 10 and report.differ == [], report
    assert report.actions_compared == 10 and report.actions_equal == 10 and report.within


class RealDoor:
    """**A real account under Confirm, as its answers read**: every order is proposed, never
    sent; the book holds one long position the environment did not open. A mapping test."""

    def __init__(self):
        self.calls = []

    def call(self, tool, **args):
        self.calls.append(tool)
        if tool == "wait_for":
            t = 1_000 * H1 + len(self.calls) * H1
            return {"items": [[t, 2000.0, 2001.0, 1999.0, 2000.5, 10]] if "after" in args else [], "next": str(t)}
        if tool == "resource_read":
            if args["uri"].endswith("/account"):
                return {"equity": 10_000.0, "margin": 0.0}
            if args["uri"].endswith("/book"):
                return {"positions": [{"symbol": "XAUUSD", "side": "buy", "volume": 0.01}, {"symbol": "EURUSD", "side": "sell", "volume": 1.0}]}
            return {"bars": []}
        if tool == "bars_get":
            return {"bars": []}
        if tool in ("order_market", "order_close_all"):
            if self.calls.count("order_close_all") + self.calls.count("order_market") > 1:
                raise RuntimeError("A proposal already waits for the person on this account; nothing was proposed.")
            return {"outcome": "awaiting_person", "positions": []}
        raise AssertionError(tool)


def test_a_strategys_refused_order_is_an_event_not_an_end():
    """On a live workbench a refusal reaches ``on_order_event`` in its own words and the run goes on."""
    door = RealDoor()
    seen = []

    class Twice(Strategy):
        def on_bar(self, bar, ctx):
            ctx.sell(0.01)

        def on_order_event(self, event, ctx):
            seen.append(event)

    out = run(Twice(), "bentrix://real/acct/bench/one", "XAUUSD", "H1", door=door, max_bars=2, idle_limit=1)
    assert out["bars"] == 2
    assert [e["outcome"] for e in seen] == ["awaiting_person", "refused"], seen
    assert "already waits" in seen[1]["refused"]


def test_a_model_answers_within_its_time_or_falls_back_by_reason():
    """Past ``timeout_s`` the fallback; while the slow call still runs, the fallback without asking
    again; an error's fallback named by its kind; a cadence other than the close refused by name."""
    slow = Model(lambda obs: time.sleep(0.3) or 1, timeout_s=0.05, fallback=0)
    assert slow({}) == 0 and slow({}) == 0
    assert slow.fallbacks == {"timeout": 1, "busy": 1}, slow.fallbacks
    time.sleep(0.4)
    broken = Model(lambda obs: 1 / 0, timeout_s=1.0, fallback=0)
    assert broken({}) == 0 and broken.fallbacks == {"error: ZeroDivisionError": 1}
    assert Model(lambda obs: 2, timeout_s=1.0)({}) == 2
    with pytest.raises(ValueError, match="cadence"):
        Model(lambda obs: 0, cadence="tick")


def test_an_onnx_policy_answers_its_largest_score(tmp_path):
    """An ONNX file as a policy: the flattened observation in, the largest score's index out."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    # scores = x @ W, W picking feature 0 for action 0 and feature 1 for action 2.
    w = np.zeros((3, 3), dtype=np.float32)
    w[0, 0], w[1, 2] = 1.0, 1.0
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "p",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor("w", TensorProto.FLOAT, [3, 3], w.ravel().tolist())],
    )
    path = tmp_path / "p.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9), str(path))
    m = Model(str(path), timeout_s=5.0)
    assert m({"a": [0.0, 5.0, 0.0]}) == 2
    assert m({"a": [5.0, 0.0, 0.0]}) == 0


def test_remote_leaves_an_absent_argument_to_the_servers_default():
    class Client:
        def call(self, tool, args):
            return (tool, args)

    assert Remote(Client()).call("order_market", bench="b", sl=None, volume=0.1) == ("order_market", {"bench": "b", "volume": 0.1})


def test_a_model_is_retrained_at_each_new_session_off_the_hot_path():
    """**Retraining on the session calendar**: the first close marks its day; the first close of the
    next day starts the person's ``fit`` on a copy of the journal and swaps the policy when it is done;
    a day's later closes start nothing; a retrain still running when the next day comes is skipped and
    counted; a week calendar retrains once a week (Monday); a period not offered is refused.

    Mutation (2026-11-01): the period read from the hour instead of the day — red, every
    close retrains."""
    from bentrix.live import Retrain

    day = 86_400_000
    seen = []

    def fit(journal):
        seen.append(len(journal))
        return lambda obs: 2

    m, r = Model(lambda obs: 1, timeout_s=5.0), Retrain(fit)
    mon = 19_786 * day  # 2024-03-04, a Monday
    assert not r.tick(m, [{}], mon + 3_600_000) and not r.tick(m, [{}], mon + 7_200_000)
    assert r.tick(m, [{}, {}, {}], mon + day + 3_600_000)
    r.wait(5)
    assert (r.retrains, seen, m({})) == (1, [3], 2)
    assert not r.tick(m, [{}], mon + day + 7_200_000)

    slow = Retrain(lambda j: time.sleep(0.5) or (lambda obs: 0))
    slow.tick(m, [], mon)
    assert slow.tick(m, [], mon + day) and not slow.tick(m, [], mon + 2 * day)
    assert slow.skipped == 1
    slow.wait(5)

    week = Retrain(fit, every="week")
    started = [week.tick(m, [], mon + k * day) for k in range(9)]
    assert started == [False] * 7 + [True, False], started
    week.wait(5)
    with pytest.raises(ValueError, match="period"):
        Retrain(fit, every="hour")


class _Drawn:
    """A door that keeps every ``output_put`` and hands every other call to the workbench."""

    def __init__(self, door, refuse=False):
        self.door, self.refuse, self.puts = door, refuse, []

    def call(self, tool, **args):
        if tool == "output_put":
            if self.refuse:
                raise RuntimeError("output_put is not a tool of this workbench")
            self.puts.append(args)
            return {"ok": True}
        return self.door.call(tool, **args)

    def __getattr__(self, name):
        return getattr(self.door, name)


def _bare_env(kind, draw="gym_ppo"):
    """A LiveEnv's drawing half alone: no workbench, the attributes the drawing reads."""
    env = LiveEnv.__new__(LiveEnv)
    env.action_kind, env.draw, env.symbol, env.bench = kind, draw, "XAUUSD", "b1"
    env.cursor, env.length = 7_200_000, 3_600_000
    env.local = _Drawn(None)
    from collections import Counter

    env.draw_refused = Counter()
    return env


def test_a_policys_decision_is_drawn_as_its_target_and_never_clamped():
    """**Each decision drawn as the model's output `policy`**, anchored at the open of the candle that
    closed: discrete 0/1/2 as a position of 0/+1/-1, continuous as its own target, an order as an action; a continuous
    1.3 is refused and counted, never clamped to 1; a put the workbench refuses never stops a step.

    Mutation (2026-11-11): discrete 2 drawn as +1 - red, *a short was drawn long*."""
    env = _bare_env("discrete")
    for a in (1, 2, 0):
        env._draw(a)
    targets = [p["output"]["values"][0]["target"] for p in env.local.puts]
    assert targets == [1.0, -1.0, 0.0], "a short was drawn long"
    put = env.local.puts[0]
    assert put["bench"] == "b1" and put["output"]["kind"] == "position" and put["output"]["output"] == "policy"
    assert put["output"]["anchor"]["time_ms"] == 3_600_000 and "put_ms" not in put["output"]

    env = _bare_env("continuous")
    env._draw(np.array([0.4], dtype=np.float32))
    env._draw(np.array([1.3], dtype=np.float32))
    assert [round(p["output"]["values"][0]["target"], 6) for p in env.local.puts] == [0.4]
    assert sum(env.draw_refused.values()) == 1, "a target outside [-1, 1] was clamped"

    env = _bare_env("order")
    env._draw({"side": 2, "size": np.array([0.5]), "stop": np.array([0.0]), "target": np.array([0.0])})
    v = env.local.puts[0]["output"]["values"][0]
    assert env.local.puts[0]["output"]["kind"] == "action" and v == {"intent": "sell", "size": 0.5}

    env = _bare_env("discrete")
    env.local = _Drawn(None, refuse=True)
    env._draw(1)
    assert env.draw_refused["RuntimeError"] == 1
    assert _bare_env("discrete", draw=None)._draw(1) is None


def test_a_served_model_on_a_live_workbench_draws_every_decision(tape):
    """**The walk**: a policy served on a live environment with ``draw=`` puts one output per decision, each anchored
    at the candle it decided at, in order."""
    local, bench = tape
    market = Market(local, bench)
    door = _Drawn(local)
    env = LiveEnv(bench, "XAUUSD", "H1", window=16, door=door, before_wait=market, idle_limit=3, max_steps=6, draw="gym_ppo")
    out = serve(Model(_policy, timeout_s=5.0), env)
    assert len(door.puts) == out["steps"] > 0
    anchors = [p["output"]["anchor"]["time_ms"] for p in door.puts]
    assert anchors == sorted(anchors) and len(set(anchors)) == len(anchors)
    assert [d["time_ms"] - env.length for d in out["journal"][: len(anchors)]] == anchors
