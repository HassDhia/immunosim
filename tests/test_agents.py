"""Tests for agent implementations."""

import gymnasium as gym
import numpy as np
import pytest

import immunosim  # noqa: F401
from immunosim.agents.random_agent import RandomAgent
from immunosim.agents.heuristic_agent import (
    HeuristicAgent,
    CheckpointInhibitorHeuristic,
    CombinationTherapyHeuristic,
    CARTCellHeuristic,
    AdaptiveDosingHeuristic,
)


ENV_IDS = [
    "immunosim/CheckpointInhibitor-v0",
    "immunosim/CombinationTherapy-v0",
    "immunosim/CARTCell-v0",
    "immunosim/AdaptiveDosing-v0",
]


class TestRandomAgent:
    """Tests for random baseline agent."""

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_predict_returns_valid_action(self, env_id):
        env = gym.make(env_id)
        agent = RandomAgent(env, seed=42)
        obs, _ = env.reset(seed=42)
        action, _ = agent.predict(obs)
        assert env.action_space.contains(action)
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_evaluate_returns_metrics(self, env_id):
        env = gym.make(env_id)
        agent = RandomAgent(env, seed=42)
        results = agent.evaluate(n_episodes=5, seed=42)
        assert "mean_reward" in results
        assert "std_reward" in results
        assert "mean_episode_length" in results
        assert results["mean_episode_length"] > 0
        env.close()


class TestHeuristicAgents:
    """Tests for heuristic baseline agents."""

    def test_checkpoint_heuristic_always_doses(self):
        heuristic = CheckpointInhibitorHeuristic(dose_action=2)
        obs = np.zeros(5, dtype=np.float32)
        action, _ = heuristic.predict(obs)
        assert action == 2

    def test_combination_heuristic_induction_phase(self):
        heuristic = CombinationTherapyHeuristic()
        obs = np.zeros(7, dtype=np.float32)
        # First 4 cycles: both drugs
        for _ in range(4):
            action, _ = heuristic.predict(obs)
            assert action[0] == 2 and action[1] == 2
        # After induction: PD-1 only
        action, _ = heuristic.predict(obs)
        assert action[0] == 2 and action[1] == 0

    def test_combination_heuristic_reset(self):
        heuristic = CombinationTherapyHeuristic()
        obs = np.zeros(7, dtype=np.float32)
        for _ in range(10):
            heuristic.predict(obs)
        heuristic.reset()
        action, _ = heuristic.predict(obs)
        assert action[1] == 2  # Back to induction

    def test_cart_heuristic_initial_dose(self):
        heuristic = CARTCellHeuristic()
        obs = np.array([1e7, 0, 0, 0, 0, 0], dtype=np.float32)
        action, _ = heuristic.predict(obs)
        assert action == 2  # Standard infusion first

    def test_cart_heuristic_then_monitors(self):
        heuristic = CARTCellHeuristic()
        obs = np.array([1e7, 0, 0, 0, 0, 0], dtype=np.float32)
        heuristic.predict(obs)  # First dose
        obs2 = np.array([1e7, 1e6, 1e5, 0, 0, 5], dtype=np.float32)
        action, _ = heuristic.predict(obs2)
        assert action == 0  # Monitor

    def test_adaptive_heuristic_continues_stable(self):
        heuristic = AdaptiveDosingHeuristic()
        obs = np.array([1e6, 0.0, 1e5, 10.0, 30.0, 0.0], dtype=np.float32)
        action, _ = heuristic.predict(obs)
        assert action == 0  # Continue for stable

    def test_adaptive_heuristic_escalates_progression(self):
        heuristic = AdaptiveDosingHeuristic()
        obs = np.array([1e6, 0.5, 1e5, 10.0, 30.0, 0.5], dtype=np.float32)
        action, _ = heuristic.predict(obs)
        assert action == 1  # Escalate for progression

    def test_adaptive_heuristic_holiday_for_response(self):
        heuristic = AdaptiveDosingHeuristic()
        obs = np.array([50.0, -0.5, 1e5, 10.0, 30.0, -0.5], dtype=np.float32)
        action, _ = heuristic.predict(obs)
        assert action == 3  # Holiday for complete response

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_heuristic_agent_evaluate(self, env_id):
        env = gym.make(env_id)
        agent = HeuristicAgent(env, env_id)
        results = agent.evaluate(n_episodes=5, seed=42)
        assert "mean_reward" in results
        assert results["mean_episode_length"] > 0
        env.close()

    def test_heuristic_agent_unknown_env(self):
        env = gym.make("immunosim/CheckpointInhibitor-v0")
        with pytest.raises(ValueError):
            HeuristicAgent(env, "unknown/Env-v0")
        env.close()
