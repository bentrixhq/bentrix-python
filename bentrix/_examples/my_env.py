"""**An environment of your own**, over Bentrix's venue: your observation, your actions, your reward.

Written by ``python -m bentrix new env``; the file is yours to change - nothing here is a base class
you inherit. It is a Gymnasium environment over ``bentrix.local.Local``: the venue, its clock and its
book are Bentrix's; everything a model sees and is paid is in this file.

Run it as it stands to check it against Gymnasium's own contract on a practice tape:

    python my_env.py
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import bentrix

HOUR = 3_600_000


class LegsEnv(gym.Env):
    """Bentrix's venue under an environment of your own: your observation, your actions, your reward.

    Actions: 0 hold, 1 open a long leg, 2 open a short leg, 3 close the oldest leg, 4 close every leg.
    One step is five minutes of the recording."""

    metadata = {"render_modes": []}

    def __init__(self, engine, bench, symbol, leg_lots=0.01, max_legs=5, episode_steps=120):
        self.engine, self.bench, self.symbol = engine, bench, symbol
        self.leg_lots, self.max_legs, self.episode_steps = leg_lots, max_legs, episode_steps
        home, owner, _, name = bench.removeprefix("bentrix://").split("/")
        self._where = dict(home=home, owner=owner, bench=name)
        engine.arm(bench)
        bounds = engine.state(bench)["bounds"]
        self.first, self.last = bounds["from_ms"] + 6 * HOUR, bounds["to_ms"] - 12 * HOUR
        self.origin = engine.snapshot(bench)            # the untouched workbench, kept in memory
        self.observation_space = spaces.Box(-1e3, 1e3, shape=(16 + 4 + 4,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    # ---- what the agent sees: yours to design ------------------------------------------------
    def _observe(self):
        got = self.engine.call("bars_window", bench=self.bench, symbol=self.symbol, frames=["M5", "H1"], limit=17)
        frames = got["markets"][self.symbol]
        m5 = np.array([row[4] for row in frames["M5"]["rows"]])          # closes of CLOSED candles only
        h1 = np.array([row[4] for row in frames["H1"]["rows"]][-5:])
        book = self.engine.book(self.bench)
        legs = book["positions"]
        long_lots = sum(p["volume"] for p in legs if p["side"] == "buy")
        short_lots = sum(p["volume"] for p in legs if p["side"] == "sell")
        account = self.engine.read("trade_account", **self._where)
        self.equity = account["equity"]
        self.peak = max(self.peak, self.equity)
        cap = self.leg_lots * self.max_legs
        mine = [
            (long_lots - short_lots) / cap,                               # net exposure, in units of the cap
            (long_lots + short_lots) / cap,                               # gross exposure
            self.equity / self.opening - 1.0,
            self.equity / self.peak - 1.0,                                # drawdown from the episode's peak
        ]
        obs = np.concatenate([np.diff(np.log(m5)) * 1e3, np.diff(np.log(h1)) * 1e3, mine]).astype(np.float32)
        return obs, legs

    # ---- Gymnasium's two methods --------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.restore(self.bench, self.origin)                      # every order and deal undone
        hours = (self.last - self.first) // HOUR
        start = self.first + int(self.np_random.integers(hours)) * HOUR   # the seed draws the start
        self.engine.advance(self.bench, start)
        self.steps = 0
        self.opening = self.peak = self.engine.read("trade_account", **self._where)["equity"]
        obs, _ = self._observe()
        return obs, {"start_ms": start}

    def step(self, action):
        _, legs = self._observe()
        refused = None
        if action in (1, 2) and len(legs) < self.max_legs:
            side = "buy" if action == 1 else "sell"
            answer = self.engine.call("order_market", bench=self.bench, symbol=self.symbol, side=side, volume=self.leg_lots)
            refused = answer["last_answer"].get("reason")
        elif action == 3 and legs:
            self.engine.call("order_close", bench=self.bench, ticket=min(p["ticket"] for p in legs))
        elif action == 4 and legs:
            self.engine.call("order_close_all", bench=self.bench)

        before, peak_before = self.equity, self.peak
        closed = self.engine.next_close(self.bench, "M5", [self.symbol])  # five minutes, tick by tick
        obs, legs = self._observe()
        self.steps += 1

        # ---- the reward: yours to design. Here: the equity's change, less new drawdown. ------
        new_drawdown = max(0.0, (peak_before - self.equity) - (peak_before - before))
        reward = (self.equity - before) - 0.5 * new_drawdown

        terminated = self.equity <= 0.5 * self.opening
        truncated = closed is None or self.steps >= self.episode_steps
        info = {"equity": self.equity, "legs": len(legs), "refused": refused}
        return obs, float(reward), bool(terminated), bool(truncated), info


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    practice = bentrix.practice()                                        # a practice tape, in a folder of its own
    with practice.open() as engine:
        bench = practice.bench(engine)
        env = LegsEnv(engine, bench, "XAUUSD")
        check_env(env, skip_render_check=True)                            # Gymnasium's own contract check
        print("Gymnasium's check passed")

        def episode(seed):
            rng = np.random.default_rng(seed)
            obs, info = env.reset(seed=seed)
            total, done = 0.0, False
            while not done:
                obs, reward, terminated, truncated, info = env.step(int(rng.integers(5)))
                total += reward
                done = terminated or truncated
            return round(total, 4), round(info["equity"], 2), info["legs"]

        print("seed 1:", episode(1))
        print("seed 2:", episode(2))
        print("seed 1 again:", episode(1))
