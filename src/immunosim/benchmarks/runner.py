"""Benchmark runner for ImmunoSim environments."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym

import immunosim  # noqa: F401
from immunosim.agents.random_agent import RandomAgent
from immunosim.agents.heuristic_agent import HeuristicAgent
from immunosim.training.configs import ENV_CONFIGS


def benchmark_env(
    env_id: str,
    n_episodes: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark a single environment with random and heuristic agents.

    Returns performance metrics and timing information.
    """
    env = gym.make(env_id)

    # Timing: measure steps per second
    start = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    n_timing_steps = 1000
    for _ in range(n_timing_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    elapsed = time.perf_counter() - start
    steps_per_sec = n_timing_steps / elapsed

    # Random baseline
    random_agent = RandomAgent(env, seed=seed)
    random_results = random_agent.evaluate(n_episodes=n_episodes, seed=seed)

    # Heuristic baseline
    heuristic_agent = HeuristicAgent(env, env_id)
    heuristic_results = heuristic_agent.evaluate(n_episodes=n_episodes, seed=seed)

    env.close()

    return {
        "env_id": env_id,
        "steps_per_second": steps_per_sec,
        "random_baseline": random_results,
        "heuristic_baseline": heuristic_results,
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
    }


def run_all_benchmarks(
    output_dir: str = "results",
    n_episodes: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Run benchmarks on all environments."""
    results = {}
    for env_id in ENV_CONFIGS:
        print(f"Benchmarking {env_id}...")
        results[env_id] = benchmark_env(env_id, n_episodes=n_episodes, seed=seed)
        print(
            f"  {results[env_id]['steps_per_second']:.0f} steps/sec, "
            f"random={results[env_id]['random_baseline']['mean_reward']:.2f}, "
            f"heuristic={results[env_id]['heuristic_baseline']['mean_reward']:.2f}"
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark ImmunoSim environments")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all_benchmarks(
        output_dir=args.output_dir, n_episodes=args.episodes, seed=args.seed
    )


if __name__ == "__main__":
    main()
