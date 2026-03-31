"""Random baseline agent for ImmunoSim environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class RandomAgent:
    """Uniformly random action selection baseline.

    Provides the lower-bound performance reference. Any learned policy
    should significantly outperform this baseline.
    """

    def __init__(self, env: gym.Env, seed: int = 42) -> None:
        self.env = env
        self.rng = np.random.default_rng(seed)

    def predict(
        self, obs: NDArray[np.float32], deterministic: bool = False
    ) -> tuple[Any, None]:
        """Select a random action.

        Args:
            obs: Current observation (unused).
            deterministic: Ignored for random agent.

        Returns:
            Tuple of (action, None) for API compatibility with SB3.
        """
        return self.env.action_space.sample(), None

    def evaluate(
        self, n_episodes: int = 100, seed: int = 42
    ) -> dict[str, float]:
        """Evaluate the random agent over n episodes.

        Returns:
            Dictionary with mean_reward, std_reward, mean_episode_length.
        """
        rewards = []
        lengths = []

        for i in range(n_episodes):
            obs, _ = self.env.reset(seed=seed + i)
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            rewards.append(total_reward)
            lengths.append(steps)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_episode_length": float(np.mean(lengths)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
        }
