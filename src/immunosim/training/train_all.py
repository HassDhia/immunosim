"""Train PPO agents on all ImmunoSim environments and save results."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym

import immunosim  # noqa: F401
from immunosim.agents.random_agent import RandomAgent
from immunosim.agents.heuristic_agent import HeuristicAgent
from immunosim.training.configs import ENV_CONFIGS


def evaluate_baselines(
    env_id: str, n_episodes: int = 100, seed: int = 42
) -> dict[str, dict[str, float]]:
    """Evaluate random and heuristic baselines on an environment."""
    env = gym.make(env_id)

    random_agent = RandomAgent(env, seed=seed)
    random_results = random_agent.evaluate(n_episodes=n_episodes, seed=seed)

    heuristic_agent = HeuristicAgent(env, env_id)
    heuristic_results = heuristic_agent.evaluate(n_episodes=n_episodes, seed=seed)

    env.close()
    return {
        "random": random_results,
        "heuristic": heuristic_results,
    }


def train_all(
    output_dir: str = "results",
    seed: int = 42,
    timestep_fraction: float = 1.0,
) -> dict:
    """Train PPO on all environments and collect results.

    Args:
        output_dir: Directory for saving results.
        seed: Random seed.
        timestep_fraction: Fraction of configured timesteps to use (for fast testing).

    Returns:
        Complete training results dictionary.
    """
    from immunosim.agents.ppo import train_ppo

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for env_id, config in ENV_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Training: {env_id}")
        print(f"{'='*60}")

        # Evaluate baselines first
        print("Evaluating baselines...")
        baselines = evaluate_baselines(env_id, n_episodes=100, seed=seed)
        print(f"  Random baseline: {baselines['random']['mean_reward']:.2f}")
        print(f"  Heuristic baseline: {baselines['heuristic']['mean_reward']:.2f}")

        # Train PPO
        timesteps = int(config["total_timesteps"] * timestep_fraction)
        print(f"Training PPO for {timesteps} steps...")
        ppo_results = train_ppo(
            env_id=env_id,
            total_timesteps=timesteps,
            seed=seed,
            output_dir=str(out_path / env_id.replace("/", "_")),
        )

        # Compute ratios
        random_mean = baselines["random"]["mean_reward"]
        heuristic_mean = baselines["heuristic"]["mean_reward"]
        ppo_mean = ppo_results["mean_reward"]

        # Handle negative baselines: use absolute difference method
        if random_mean < 0:
            ppo_vs_random = abs(ppo_mean) / max(abs(random_mean), 1e-6)
            if ppo_mean > random_mean:
                ppo_vs_random = 1.0 + (ppo_mean - random_mean) / max(abs(random_mean), 1e-6)
        else:
            ppo_vs_random = ppo_mean / max(random_mean, 1e-6)

        converged = ppo_mean > random_mean * 1.2 if random_mean > 0 else ppo_mean > random_mean

        all_results[env_id] = {
            "mean_reward": ppo_mean,
            "std_reward": ppo_results["std_reward"],
            "random_baseline": random_mean,
            "heuristic_baseline": heuristic_mean,
            "ppo_vs_random_ratio": float(ppo_vs_random),
            "ppo_vs_heuristic_ratio": (
                ppo_mean / max(heuristic_mean, 1e-6)
                if heuristic_mean > 0
                else (
                    1.0 + (ppo_mean - heuristic_mean) / max(abs(heuristic_mean), 1e-6)
                    if heuristic_mean < 0
                    else float("inf")
                )
            ),
            "training_steps": timesteps,
            "convergence_status": "converged" if converged else "not_converged",
        }

        print(f"  PPO result: {ppo_mean:.2f} +/- {ppo_results['std_reward']:.2f}")
        print(f"  PPO vs Random ratio: {ppo_vs_random:.2f}x")

    # Save results
    results_path = out_path / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return all_results


def main() -> None:
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="Train all ImmunoSim environments")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="Use 10% of timesteps")
    args = parser.parse_args()

    fraction = 0.1 if args.fast else 1.0
    train_all(output_dir=args.output_dir, seed=args.seed, timestep_fraction=fraction)


if __name__ == "__main__":
    main()
