"""**The Gym moved whole into Rust is the Gym it replaced**.

``bentrix.gym.TradingEnv`` is now a client of the Rust environment (``env_make`` …). The Python
implementation it replaced is kept unchanged as ``reference_gym.py``; each walk below runs an episode
from one seed and one list of actions through the reference and then through the Rust environment on
the same workbench (each reset rewinds it and undoes its deals), and requires the same start and end,
the same observations bit for bit, the same rewards (exactly — but the volatility-scaled reward, whose
standard deviation NumPy sums pairwise, within 1e-12), the same refusals, equities, participation and
ledger hash at every step. Skipped without the extension module; run on the CI tape.
"""

import pytest

pytest.importorskip("bentrix._native")
np = pytest.importorskip("numpy")

import reference_gym  # noqa: E402
from bentrix.gym import REWARDS, TradingEnv, growing  # noqa: E402


def _episode(env, seed, actions):
    obs, info = env.reset(seed=seed)
    out = [(obs, None, False, False, info)]
    for a in actions:
        out.append(env.step(a))
        if out[-1][2] or out[-1][3]:
            break
    return out


def _same(want, got, label):
    """The two episodes step for step (module doc)."""
    assert len(want) == len(got) >= 4, (label, len(want), len(got))
    w0, g0 = want[0][4], got[0][4]
    assert (w0["start_ms"], w0["ledger_hash"]) == (g0["start_ms"], g0["ledger_hash"]), (label, w0, g0)
    for k, ((wo, wr, wt, wu, wi), (go, gr, gt, gu, gi)) in enumerate(zip(want, got)):
        for part in ("market", "account"):
            assert np.array_equal(wo[part], go[part]), (label, k, part, wo[part].tolist(), go[part].tolist())
        if k == 0:
            continue
        exact = wi["reward"] != "cost_aware_vol_scaled@1"
        assert (gr == wr) if exact else abs(gr - wr) <= 1e-12 * max(1.0, abs(wr)), (label, k, wr, gr)
        assert (gt, gu) == (wt, wu), (label, k)
        for key in ("ledger_hash", "refused", "equity", "cursor_ms", "participation", "product_hypothesis"):
            assert gi.get(key) == wi.get(key), (label, k, key, wi.get(key), gi.get(key))


def _both(local, bench, seed, actions, **kw):
    ref = reference_gym.TradingEnv(bench, local=local, **kw)
    want = _episode(ref, seed, actions)
    rust = TradingEnv(bench, local=local, **kw)
    got = _episode(rust, seed, actions)
    return want, got, rust


def test_every_reward_and_both_plain_action_kinds_are_the_references(tape):
    """Every reward plugin over one discrete episode, and a continuous one, step for step.

    Mutations (2026-11-01), each red: the Rust observation's log ratio upside down; the
    Rust draw's increment changed; the Rust DSR's numerator taking the second moment for the first."""
    local, bench = tape
    actions = [1, 1, 2, 0, 2, 2, 1, 0, 1, 2, 0, 1]
    for reward in REWARDS:
        want, got, _ = _both(local, bench, 5, actions, interval="H1", window=8, episode="window", episode_bars=12, reward=reward)
        _same(want, got, reward)
    cont = [np.array([x], dtype=np.float32) for x in (0.6, -0.3, 0.05, 1.0, -0.75, 0.125)]
    want, got, _ = _both(local, bench, 1, cont, interval="H1", window=8, action="continuous", episode="tape", reward="dsr@1")
    _same(want[:7], got[:7], "continuous")


def test_the_order_action_is_the_references(tape):
    """The order-level action — a long with a stop and a target, a half-size short with a stop only,
    flat, a long with a target only — step for step: the same stops on the venue, the same deals.

    Mutation (2026-11-01): a short's stop on the long's side in Rust — red, the hashes
    part (the stop is met at another price)."""
    local, bench = tape
    acts = [
        {"side": 1, "size": [1.0], "stop": [1500.0], "target": [2500.0]},
        {"side": 1, "size": [1.0], "stop": [1500.0], "target": [2500.0]},
        {"side": 2, "size": [0.5], "stop": [1500.0], "target": [0.0]},
        {"side": 0, "size": [0.0], "stop": [0.0], "target": [0.0]},
        {"side": 1, "size": [0.37], "stop": [0.0], "target": [80.0]},
        {"side": 1, "size": [0.37], "stop": [0.0], "target": [80.0]},
    ]
    want, got, _ = _both(local, bench, 3, acts, interval="H1", window=8, action="order", lots=0.1, episode="window", episode_bars=20)
    _same(want, got, "order")


def test_participation_and_its_impact_are_the_references(tape):
    """A participation cap that refuses (a share far under one lot), and one that admits with an impact
    charged to the reward, step for step — the refusal's words included.

    Mutation (2026-11-01): the Rust impact left out of the reward — red, the rewards
    part by the impact."""
    local, bench = tape
    actions = [1, 1, 0, 2, 2, 0, 1]
    want, got, _ = _both(local, bench, 1, actions, interval="H1", window=8, lots=0.01, episode="window", episode_bars=20, participation=0.001)
    _same(want, got, "participation refused")
    assert got[1][4]["refused"].startswith("participation:")
    want, got, _ = _both(local, bench, 1, actions, interval="H1", window=8, lots=0.01, episode="window", episode_bars=20, participation=1.0, impact=0.5)
    _same(want, got, "participation with impact")
    assert got[1][4]["participation"]["impact"] > 0


def test_the_curriculum_and_a_persons_own_sampler_are_the_references(tape):
    """``growing(4, 9, over=4)`` over six resets from one seed — the same starts and lengths, the third
    episode's 6.5 candles rounded as Python rounds (to the even 6) — and a sampler of the person's own,
    run on this side and handed over.

    Mutation (2026-11-01): the Rust curriculum's length rounded half away from zero —
    red, the third episode is 7 candles where the reference's is 6."""
    local, bench = tape
    ref = reference_gym.TradingEnv(bench, local=local, interval="H1", window=8, episode="window", sampler=reference_gym.growing(4, 9, over=4))
    rust = TradingEnv(bench, local=local, interval="H1", window=8, episode="window", sampler=growing(4, 9, over=4))
    for env in (ref, rust):
        env.spans = []
        for k in range(6):
            _, info = env.reset(seed=7) if k == 0 else env.reset()
            env.spans.append((info["start_ms"], env.end))
    assert rust.spans == ref.spans and len({e - s for s, e in rust.spans}) > 1, (ref.spans, rust.spans)

    def mine(i, rng, env):
        return env.first + (40 + 3 * i + rng.randrange(5)) * env.length, 6 + i

    want, got, _ = _both(local, bench, 2, [1, 2, 1, 0, 2, 1, 1], interval="H1", window=8, episode="window", sampler=mine)
    _same(want, got, "own sampler")
