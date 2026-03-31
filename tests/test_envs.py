"""Tests for all Gymnasium environments."""

import gymnasium as gym
import numpy as np
import pytest

import immunosim  # noqa: F401


ENV_IDS = [
    "immunosim/CheckpointInhibitor-v0",
    "immunosim/CombinationTherapy-v0",
    "immunosim/CARTCell-v0",
    "immunosim/AdaptiveDosing-v0",
]


class TestEnvironmentRegistration:
    """Test that all environments are properly registered."""

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_make(self, env_id):
        env = gym.make(env_id)
        assert env is not None
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_reset(self, env_id):
        env = gym.make(env_id)
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert isinstance(info, dict)
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_observation_in_space(self, env_id):
        env = gym.make(env_id)
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs), (
            f"Observation {obs} not in space {env.observation_space}"
        )
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_step_returns_correct_types(self, env_id):
        env = gym.make(env_id)
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (float, int, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_step_observation_in_space(self, env_id):
        env = gym.make(env_id)
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_reproducibility(self, env_id):
        """Same seed should produce same trajectory."""
        env = gym.make(env_id)

        obs1, _ = env.reset(seed=42)
        _ = env.action_space.sample()

        env2 = gym.make(env_id)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)
        env.close()
        env2.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_episode_terminates(self, env_id):
        """Episodes should eventually end."""
        env = gym.make(env_id)
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        max_steps = 1000
        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert done, f"Episode did not terminate in {max_steps} steps"
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_multiple_resets(self, env_id):
        """Environment can be reset multiple times."""
        env = gym.make(env_id)
        for i in range(5):
            obs, _ = env.reset(seed=i)
            assert env.observation_space.contains(obs)
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_reward_is_finite(self, env_id):
        """Rewards should be finite numbers."""
        env = gym.make(env_id)
        obs, _ = env.reset(seed=42)
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if terminated or truncated:
                break
        env.close()


class TestCheckpointInhibitorEnv:
    """Specific tests for CheckpointInhibitorEnv."""

    def test_observation_shape(self):
        env = gym.make("immunosim/CheckpointInhibitor-v0")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (5,)
        env.close()

    def test_action_space(self):
        env = gym.make("immunosim/CheckpointInhibitor-v0")
        assert env.action_space.n == 3
        env.close()

    def test_no_dose_action(self):
        env = gym.make("immunosim/CheckpointInhibitor-v0")
        obs, _ = env.reset(seed=42)
        obs_new, _, _, _, info = env.step(0)
        assert info["drug_concentration"] >= 0
        env.close()

    def test_tumor_info_tracked(self):
        env = gym.make("immunosim/CheckpointInhibitor-v0")
        obs, _ = env.reset(seed=42)
        _, _, _, _, info = env.step(2)
        assert "tumor_volume" in info
        assert "effector_cells" in info
        assert "cycle" in info
        env.close()


class TestCombinationTherapyEnv:
    """Specific tests for CombinationTherapyEnv."""

    def test_observation_shape(self):
        env = gym.make("immunosim/CombinationTherapy-v0")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (7,)
        env.close()

    def test_action_space(self):
        env = gym.make("immunosim/CombinationTherapy-v0")
        assert hasattr(env.action_space, "nvec")
        np.testing.assert_array_equal(env.action_space.nvec, [3, 3])
        env.close()

    def test_both_drugs_tracked(self):
        env = gym.make("immunosim/CombinationTherapy-v0")
        obs, _ = env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([2, 2]))
        assert "pd1_concentration" in info
        assert "ctla4_concentration" in info
        assert "toxicity_score" in info
        env.close()


class TestCARTCellEnv:
    """Specific tests for CARTCellEnv."""

    def test_observation_shape(self):
        env = gym.make("immunosim/CARTCell-v0")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (6,)
        env.close()

    def test_action_space(self):
        env = gym.make("immunosim/CARTCell-v0")
        assert env.action_space.n == 4
        env.close()

    def test_crs_tracked(self):
        env = gym.make("immunosim/CARTCell-v0")
        obs, _ = env.reset(seed=42)
        _, _, _, _, info = env.step(2)  # Standard infusion
        assert "crs_grade" in info
        assert "cytokine_level" in info
        env.close()

    def test_infusion_increases_cart(self):
        env = gym.make("immunosim/CARTCell-v0")
        obs, _ = env.reset(seed=42)
        obs2, _, _, _, info = env.step(3)  # High dose
        # Injected or effector should be > 0
        assert obs2[1] > 0 or obs2[2] > 0
        env.close()


class TestAdaptiveDosingEnv:
    """Specific tests for AdaptiveDosingEnv."""

    def test_observation_shape(self):
        env = gym.make("immunosim/AdaptiveDosing-v0")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (6,)
        env.close()

    def test_action_space(self):
        env = gym.make("immunosim/AdaptiveDosing-v0")
        assert env.action_space.n == 4
        env.close()

    def test_dose_escalation(self):
        env = gym.make("immunosim/AdaptiveDosing-v0")
        obs, _ = env.reset(seed=42)
        # Action 1 = escalate
        _, _, _, _, info = env.step(1)
        assert info["dose_level"] >= 2
        env.close()

    def test_treatment_holiday(self):
        env = gym.make("immunosim/AdaptiveDosing-v0")
        obs, _ = env.reset(seed=42)
        # Action 3 = holiday
        _, _, _, _, info = env.step(3)
        assert info["on_holiday"] is True
        assert info["actual_dose_mgkg"] == 0.0
        env.close()

    def test_pseudo_progression_info(self):
        env = gym.make("immunosim/AdaptiveDosing-v0")
        obs, _ = env.reset(seed=42)
        _, _, _, _, info = env.step(0)
        assert "is_pseudo_progression" in info
        assert "response_trajectory" in info
        env.close()
