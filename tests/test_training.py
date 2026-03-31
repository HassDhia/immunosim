"""Tests for training configuration and utilities."""


from immunosim.training.configs import ENV_CONFIGS


class TestENVConfigs:
    """Tests for shared training configurations."""

    def test_all_envs_have_configs(self):
        expected = [
            "immunosim/CheckpointInhibitor-v0",
            "immunosim/CombinationTherapy-v0",
            "immunosim/CARTCell-v0",
            "immunosim/AdaptiveDosing-v0",
        ]
        for env_id in expected:
            assert env_id in ENV_CONFIGS

    def test_configs_have_required_keys(self):
        required = ["total_timesteps", "learning_rate", "n_steps", "batch_size", "gamma"]
        for env_id, config in ENV_CONFIGS.items():
            for key in required:
                assert key in config, f"{env_id} missing {key}"

    def test_timesteps_positive(self):
        for env_id, config in ENV_CONFIGS.items():
            assert config["total_timesteps"] > 0

    def test_learning_rate_reasonable(self):
        for env_id, config in ENV_CONFIGS.items():
            assert 1e-6 <= config["learning_rate"] <= 1e-2

    def test_gamma_in_range(self):
        for env_id, config in ENV_CONFIGS.items():
            assert 0.9 <= config["gamma"] <= 1.0

    def test_batch_size_divides_n_steps(self):
        for env_id, config in ENV_CONFIGS.items():
            assert config["n_steps"] % config["batch_size"] == 0

    def test_checkpoint_inhibitor_config(self):
        cfg = ENV_CONFIGS["immunosim/CheckpointInhibitor-v0"]
        assert cfg["total_timesteps"] == 200_000
        assert cfg["learning_rate"] == 3e-4

    def test_combination_therapy_config(self):
        cfg = ENV_CONFIGS["immunosim/CombinationTherapy-v0"]
        assert cfg["total_timesteps"] == 500_000
        assert cfg["gamma"] == 0.995

    def test_cart_cell_config(self):
        cfg = ENV_CONFIGS["immunosim/CARTCell-v0"]
        assert cfg["total_timesteps"] == 300_000

    def test_adaptive_dosing_config(self):
        cfg = ENV_CONFIGS["immunosim/AdaptiveDosing-v0"]
        assert cfg["total_timesteps"] == 500_000
        assert cfg["gamma"] == 0.998
        assert cfg["n_steps"] == 4096
