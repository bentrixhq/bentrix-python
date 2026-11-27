"""**A CleanRL-style loop on a Bentrix workbench**: one file, no framework — vectorised
environments over workbenches made alike (``bentrix.gym.make_vec``), a random policy standing in for
yours, and the run's identity logged where a tracker is installed.

    pip install "bentrix[gym]"
    python rl_cleanrl_style.py bentrix://lab/<lab>/bench/<a> bentrix://lab/<lab>/bench/<b>
"""

import sys

import numpy as np

from bentrix.gym import make_vec
from bentrix.track import identity, log_run


def main(benches, steps: int = 200, seed: int = 1) -> None:
    envs = make_vec(benches, window=16, episode="window", episode_bars=50)
    obs, infos = envs.reset(seed=seed)
    rng = np.random.default_rng(seed)
    returns = np.zeros(len(benches))
    for _ in range(steps):
        actions = rng.integers(0, 3, size=len(benches))  # your policy here
        obs, rewards, terms, truncs, infos = envs.step(actions)
        returns += rewards
    log_run(identity(reward="dsr_dd_penalty@1", seed=seed), {"benches": len(benches), "steps": steps}, {"mean_return": float(returns.mean())})
    print({"returns": returns.tolist()})


if __name__ == "__main__":
    main(sys.argv[1:])
