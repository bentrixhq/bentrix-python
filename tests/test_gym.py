"""**The Gym**: ``bentrix.gym.TradingEnv`` walked on a recording.

Skipped without the extension module; run on the CI tape (``conftest.tape``).
"""

import random

import pytest

pytest.importorskip("bentrix._native")
np = pytest.importorskip("numpy")

from bentrix.gym import REWARDS, TradingEnv  # noqa: E402

def _episode(env, seed, actions):
    obs, info = env.reset(seed=seed)
    start = info["start_ms"]
    hashes = [info["ledger_hash"]]
    rewards = []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        assert obs["market"].shape == (8, 4) and obs["account"].shape == (3,)
        assert np.isfinite(r), r
        rewards.append(r)
        hashes.append(info["ledger_hash"])
        if term or trunc:
            break
    info["start_ms"] = start
    return hashes, rewards, info


def test_an_episode_is_a_function_of_its_seed_and_actions(tape):
    """**The same seed and the same actions give the same deals**: an episode of a window of candles, its
    observations shaped as declared and every reward finite; run again from the same seed with the same
    actions it ends on the same ``ledger_hash``, and a reset undid the first episode's deals; an episode
    trades (its hash moves); a reward or an action kind not offered is refused by name.

    Mutation (2026-11-01): the reset not undoing the last episode's deals
    (``undo_deals=False``) — red, the second episode's deals are not the first's."""
    local, bench = tape
    env = TradingEnv(bench, local=local, window=8, episode="window", episode_bars=24)
    rng = random.Random(3)
    actions = [rng.choice([0, 1, 2]) for _ in range(24)]
    h1, r1, i1 = _episode(env, 11, actions)
    # The start is a function of (seed, episode index): a fresh environment on the same workbench is
    # episode 0 again, and its reset undoes the first's deals.
    again = TradingEnv(bench, local=local, window=8, episode="window", episode_bars=24)
    h2, r2, i2 = _episode(again, 11, actions)
    # A reset with no seed takes the next episode of the last one: another start.
    _, nxt = env.reset()
    assert nxt["start_ms"] != i1["start_ms"], "the next episode began where the first did"
    assert h1[-1] != h1[0], "the episode traded nothing"
    assert h1 == h2 and r1 == r2, (h1[-1], h2[-1])
    assert i1["reward"] == "dsr_dd_penalty@1"
    with pytest.raises(ValueError):
        TradingEnv(bench, local=local, reward="profit@9")
    with pytest.raises(ValueError):
        TradingEnv(bench, local=local, action="orders")
    assert len(REWARDS) == 8


def test_gymnasiums_own_checker_passes(tape):
    """Gymnasium's own ``check_env`` — spaces, reset and step signatures, seeding — over the environment
    and its continuous action kind; its observation flattened for SB3 and CleanRL.

    Mutation (2026-11-01): a seed given not restarting the episode sampler — red, the
    checker's *deterministic step observations are not equivalent for the same seed and action*."""
    gymnasium = pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    local, bench = tape
    for kind in ("discrete", "continuous"):
        env = TradingEnv(bench, local=local, window=8, episode="window", episode_bars=12, action=kind)
        check_env(env, skip_render_check=True)
    flat = gymnasium.wrappers.FlattenObservation(TradingEnv(bench, local=local, window=8, episode_bars=6))
    obs, _ = flat.reset(seed=1)
    assert obs.shape == (8 * 4 + 3,)


H1 = 3_600_000


def test_an_order_action_opens_with_its_stop_and_target_on_the_right_sides(tape):
    """**The order-level action**: a long of the whole size with a stop 1500 points under and a target 2500
    points over the quote it fills against, a short the reverse, and 0 for none — the venue's book
    holds them where the action put them.

    Mutation (2026-11-01): the stop and the target taken on the same side for a short as
    for a long — red, the short's stop is under its entry."""
    from bentrix.gym import TradingEnv

    local, bench = tape
    env = TradingEnv(bench, interval="H1", window=8, action="order", lots=0.1, episode="window", episode_bars=20, local=local)
    assert set(env.action_space.spaces) == {"side", "size", "stop", "target"}
    env.reset(seed=3)
    point = local.call("symbol_info", bench=bench, symbol="XAUUSD")["point"]
    positions = lambda: [p for p in local.book(bench)["positions"] if p["symbol"] == "XAUUSD"]
    _, _, _, _, info = env.step({"side": 1, "size": [1.0], "stop": [1500.0], "target": [2500.0]})
    held = positions()
    assert info["refused"] is None and len(held) == 1, (info, held)
    (long,) = held
    assert long["side"] == "buy" and abs(long["volume"] - 0.1) < 1e-9, long
    assert long["sl"] < long["price_open"] < long["tp"], long
    assert abs((long["tp"] - long["sl"]) / point - 4000) < 1.5, long
    _, _, _, _, info = env.step({"side": 2, "size": [0.5], "stop": [1500.0], "target": [0.0]})
    held = positions()
    assert info["refused"] is None and len(held) == 1, (info, held)
    (short,) = held
    assert short["side"] == "sell" and abs(short["volume"] - 0.05) < 1e-9, short
    assert short["sl"] > short["price_open"] and not short["tp"], short
    _, _, _, _, info = env.step({"side": 0, "size": [0.0], "stop": [0.0], "target": [0.0]})
    assert positions() == [] and info["refused"] is None


def test_the_participation_cap_cuts_or_refuses_by_name_and_its_impact_is_the_rewards_alone(tape):
    """**Participation, a labelled model**: an order larger than the share of the last candle's volume
    is refused ``participation`` and nothing is sent; one within it is sent, and the impact is charged
    to the reward and said, never to the equity.

    Mutations (2026-11-01): the cap compared to the candle's whole volume rather than its
    share — red, the refused order is sent; the impact left out of the reward — survived while only
    ``info`` was read, red since the same episode without impact is compared."""
    from bentrix.gym import TradingEnv

    local, bench = tape
    env = TradingEnv(bench, interval="H1", window=8, lots=0.01, episode="window", episode_bars=20, local=local, participation=0.001)
    env.reset(seed=1)
    _, _, _, _, info = env.step(1)
    assert info["refused"].startswith("participation:"), info
    assert local.book(bench)["positions"] == [] and info["participation"]["modelled"] is True
    traded = info["participation"]["candle_volume"]
    assert traded > 0
    env = TradingEnv(bench, interval="H1", window=8, lots=0.01, episode="window", episode_bars=20, local=local, participation=1.0, impact=0.5)
    env.reset(seed=1)
    _, r, _, _, info = env.step(1)
    p = info["participation"]
    assert info["refused"] is None and len(local.book(bench)["positions"]) == 1, info
    assert abs(p["impact"] - 0.5 * p["asked"] / p["candle_volume"]) < 1e-12, p
    # The same episode without impact: the reward is higher by the impact alone, the equity the same.
    free = TradingEnv(bench, interval="H1", window=8, lots=0.01, episode="window", episode_bars=20, local=local, participation=1.0)
    free.reset(seed=1)
    _, r0, _, _, info0 = free.step(1)
    assert abs((r0 - r) - p["impact"]) < 1e-9 and info0["equity"] == info["equity"], (r0, r, p, info0["equity"], info["equity"])
    with pytest.raises(ValueError, match="participation"):
        TradingEnv(bench, local=local, participation=2.0)


def test_a_growing_curriculum_lengthens_its_episodes_and_is_a_function_of_its_seed(tape):
    """``growing(4, 12, over=4)``: episodes of 4, 6, 8, 10, 12, 12 candles; the same seed draws the same
    starts.

    Mutation (2026-11-01): the sampler handed the episode index after the increment — red,
    the first episode is already 6 candles."""
    from bentrix.gym import TradingEnv, growing

    local, bench = tape
    env = TradingEnv(bench, interval="H1", window=8, episode="window", local=local, sampler=growing(4, 12, over=4))

    def lengths(seed):
        out, starts = [], []
        for k in range(6):
            _, info = env.reset(seed=seed) if k == 0 else env.reset()
            out.append((env.end - info["start_ms"]) // H1)
            starts.append(info["start_ms"])
        return out, starts

    first, starts = lengths(7)
    assert first == [4, 6, 8, 10, 12, 12], first
    assert lengths(7) == (first, starts)


def test_iqm_cuts_a_quarter_from_each_end():
    from bentrix.gym import iqm

    assert iqm([1, 2, 3, 4, 5, 6, 7, 8]) == 4.5
    assert iqm([0, 0, 0, 1, 1, 1, 1, 1000]) == 0.75, "one lucky seed moved the IQM"
    assert iqm([2.0, 4.0]) == 3.0


def test_a_policy_is_reported_over_ten_seeds_by_iqm_with_its_interval(tape):
    """``evaluate``: ten seeds by default, one score each, the IQM inside its bootstrap interval, the same
    report twice; fewer than ten seeds said.

    Mutation (2026-11-01): the bootstrap's generator unseeded — red, two reports'
    intervals differ."""
    from bentrix.gym import TradingEnv, evaluate

    local, bench = tape
    env = TradingEnv(bench, interval="H1", window=8, episode="window", episode_bars=6, local=local)

    def policy(obs):
        m = obs["market"]
        return 1 if m[-1, 3] >= m[-1, 0] else 2

    a = evaluate(policy, env)
    assert a["seeds"] == list(range(10)) and len(a["scores"]) == 10 and not a["too_few_seeds"]
    lo, hi = a["interval"]
    assert lo <= a["iqm"] <= hi, a
    assert len(set(a["ledger_hashes"])) > 1, "every seed traded the same episode"
    b = evaluate(policy, env)
    assert (b["scores"], b["interval"], b["ledger_hashes"]) == (a["scores"], a["interval"], a["ledger_hashes"])
    assert evaluate(policy, env, seeds=[0, 1])["too_few_seeds"]

