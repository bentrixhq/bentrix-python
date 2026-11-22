"""**PPO from Stable-Baselines3 on a BENTRIX workbench**, in your own Python process.

    pip install "bentrix[gym]" stable-baselines3 mlflow   # mlflow optional
    python rl_sb3_ppo.py bentrix://lab/<lab>/bench/<bench>

The environment is ``bentrix.gym.TradingEnv`` over the workbench's own venue; the observation is
flattened for SB3; every evaluation episode's ``ledger_hash``, reward plugin and seed are logged to
MLflow or W&B where installed (``bentrix.track``), so a result there leads back to its deals.
"""

import sys

import gymnasium as gym
from stable_baselines3 import PPO

from bentrix.gym import TradingEnv
from bentrix.track import identity, log_run


def main(bench: str, steps: int = 5_000, seed: int = 7, local=None) -> dict:
    """Train PPO for ``steps`` and evaluate one episode; ``local`` an open ``bentrix.local.Local`` (yours
    when ``None``). Answers the evaluation's reward and ledger hash."""
    env = gym.wrappers.FlattenObservation(TradingEnv(bench, window=32, episode="window", episode_bars=200, local=local))
    rollout = min(2048, max(64, steps))  # a short run trains on a rollout as short as itself
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, n_steps=rollout, batch_size=min(64, rollout)).learn(total_timesteps=steps)
    obs, info = env.reset(seed=seed + 1)
    total, done = 0.0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
    log_run(identity(ledger_hash=info["ledger_hash"], reward=info["reward"], seed=seed, bench=bench), {"algo": "PPO", "steps": steps}, {"eval_reward": total})
    out = {"eval_reward": total, "ledger_hash": info["ledger_hash"]}
    print(out)
    return out


if __name__ == "__main__":
    main(sys.argv[1])
