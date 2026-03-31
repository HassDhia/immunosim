"""Tests for benchmark runner."""


import immunosim  # noqa: F401
from immunosim.benchmarks.runner import benchmark_env


class TestBenchmarkRunner:
    """Tests for the benchmark runner."""

    def test_benchmark_checkpoint_inhibitor(self):
        result = benchmark_env(
            "immunosim/CheckpointInhibitor-v0",
            n_episodes=5,
            seed=42,
        )
        assert "steps_per_second" in result
        assert result["steps_per_second"] > 0
        assert "random_baseline" in result
        assert "heuristic_baseline" in result

    def test_benchmark_combination_therapy(self):
        result = benchmark_env(
            "immunosim/CombinationTherapy-v0",
            n_episodes=5,
            seed=42,
        )
        assert result["steps_per_second"] > 0

    def test_benchmark_cart_cell(self):
        result = benchmark_env(
            "immunosim/CARTCell-v0",
            n_episodes=5,
            seed=42,
        )
        assert result["steps_per_second"] > 0

    def test_benchmark_adaptive_dosing(self):
        result = benchmark_env(
            "immunosim/AdaptiveDosing-v0",
            n_episodes=5,
            seed=42,
        )
        assert result["steps_per_second"] > 0

    def test_benchmark_includes_spaces(self):
        result = benchmark_env(
            "immunosim/CheckpointInhibitor-v0",
            n_episodes=3,
            seed=42,
        )
        assert "observation_space" in result
        assert "action_space" in result
