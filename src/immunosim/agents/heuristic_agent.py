"""Clinical protocol heuristic baselines for ImmunoSim environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class CheckpointInhibitorHeuristic:
    """Standard clinical protocol: dose every cycle at fixed level.

    Mimics the standard nivolumab dosing schedule:
    - 3 mg/kg every 2 weeks (high dose every cycle in our 7-day cycle model)
    """

    def __init__(self, dose_action: int = 2) -> None:
        """
        Args:
            dose_action: Fixed dose action (1=low, 2=high). Default 2 = standard clinical.
        """
        self.dose_action = dose_action

    def predict(
        self, obs: NDArray[np.float32], deterministic: bool = True
    ) -> tuple[int, None]:
        return self.dose_action, None


class CombinationTherapyHeuristic:
    """Standard ipilimumab + nivolumab protocol.

    Induction phase: both drugs high dose for 4 cycles (weeks 0-3),
    then maintenance: PD-1 high dose only, CTLA-4 off.
    """

    INDUCTION_CYCLES: int = 4

    def __init__(self) -> None:
        self.cycle_count = 0

    def predict(
        self, obs: NDArray[np.float32], deterministic: bool = True
    ) -> tuple[NDArray[np.int64], None]:
        if self.cycle_count < self.INDUCTION_CYCLES:
            action = np.array([2, 2], dtype=np.int64)  # Both high
        else:
            action = np.array([2, 0], dtype=np.int64)  # PD-1 only
        self.cycle_count += 1
        return action, None

    def reset(self) -> None:
        self.cycle_count = 0


class CARTCellHeuristic:
    """Standard CAR-T infusion protocol.

    Day 0: Standard infusion, then monitor.
    Re-infuse only if tumor rebounds after day 28 (per ZUMA-1 protocol).
    """

    INITIAL_DOSE: int = 2  # Standard infusion
    REINFUSION_THRESHOLD_FACTOR: float = 1.5
    MIN_DAYS_BEFORE_REINFUSION: float = 28.0

    def __init__(self) -> None:
        self.initial_dose_given = False
        self.baseline_tumor: float = 0.0

    def predict(
        self, obs: NDArray[np.float32], deterministic: bool = True
    ) -> tuple[int, None]:
        tumor = obs[0]
        day = obs[5]

        if not self.initial_dose_given:
            self.initial_dose_given = True
            self.baseline_tumor = tumor
            return self.INITIAL_DOSE, None

        # Re-infuse if tumor rebounds significantly after day 28
        if (
            day >= self.MIN_DAYS_BEFORE_REINFUSION
            and tumor > self.baseline_tumor * self.REINFUSION_THRESHOLD_FACTOR
        ):
            self.baseline_tumor = tumor
            return 1, None  # Low dose re-infusion

        return 0, None  # Monitor

    def reset(self) -> None:
        self.initial_dose_given = False
        self.baseline_tumor = 0.0


class AdaptiveDosingHeuristic:
    """RECIST-like adaptive dosing protocol.

    - Continue if stable disease (< 20% change)
    - Escalate if progressive disease (> 20% growth)
    - Reduce if major response (> 30% shrinkage) to minimize toxicity
    - Holiday if sustained complete response
    """

    PROGRESSION_THRESHOLD: float = 0.20
    RESPONSE_THRESHOLD: float = -0.30
    COMPLETE_RESPONSE_THRESHOLD: float = 100.0

    def predict(
        self, obs: NDArray[np.float32], deterministic: bool = True
    ) -> tuple[int, None]:
        tumor = obs[0]
        growth_rate = obs[1]

        # Complete response - holiday
        if tumor < self.COMPLETE_RESPONSE_THRESHOLD:
            return 3, None

        # Progressive disease - escalate
        if growth_rate > self.PROGRESSION_THRESHOLD:
            return 1, None

        # Response - reduce
        if growth_rate < self.RESPONSE_THRESHOLD:
            return 2, None

        # Stable disease - continue
        return 0, None


class HeuristicAgent:
    """Unified heuristic agent that selects the right protocol per environment."""

    HEURISTIC_MAP = {
        "immunosim/CheckpointInhibitor-v0": CheckpointInhibitorHeuristic,
        "immunosim/CombinationTherapy-v0": CombinationTherapyHeuristic,
        "immunosim/CARTCell-v0": CARTCellHeuristic,
        "immunosim/AdaptiveDosing-v0": AdaptiveDosingHeuristic,
    }

    def __init__(self, env: gym.Env, env_id: str) -> None:
        self.env = env
        self.env_id = env_id
        heuristic_cls = self.HEURISTIC_MAP.get(env_id)
        if heuristic_cls is None:
            raise ValueError(f"No heuristic defined for {env_id}")
        self.heuristic = heuristic_cls()

    def predict(
        self, obs: NDArray[np.float32], deterministic: bool = True
    ) -> tuple[Any, None]:
        return self.heuristic.predict(obs, deterministic)

    def evaluate(
        self, n_episodes: int = 100, seed: int = 42
    ) -> dict[str, float]:
        """Evaluate the heuristic agent."""
        rewards = []
        lengths = []

        for i in range(n_episodes):
            if hasattr(self.heuristic, "reset"):
                self.heuristic.reset()
            obs, _ = self.env.reset(seed=seed + i)
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                action, _ = self.predict(obs)
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
