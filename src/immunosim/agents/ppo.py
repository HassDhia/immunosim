"""PPO agent via stable-baselines3 with CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import immunosim  # noqa: F401 - triggers env registration
from immunosim.training.configs import ENV_CONFIGS


def train_ppo(
    env_id: str,
    total_timesteps: int | None = None,
    seed: int = 42,
    output_dir: str = "results",
    **kwargs: Any,
) -> dict[str, Any]:
    """Train a PPO agent on the specified environment.

    Args:
        env_id: Gymnasium environment ID.
        total_timesteps: Override training steps (default from ENV_CONFIGS).
        seed: Random seed.
        output_dir: Directory for saving results.
        **kwargs: Additional PPO hyperparameters.

    Returns:
        Dictionary with training results.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        print("stable-baselines3 required. Install with: pip install immunosim[train]")
        sys.exit(1)

    config = ENV_CONFIGS.get(env_id, {})
    timesteps = total_timesteps or config.get("total_timesteps", 100_000)

    # Create training and eval environments
    train_env = make_vec_env(env_id, n_envs=1, seed=seed)
    eval_env = make_vec_env(env_id, n_envs=1, seed=seed + 1000)

    # PPO hyperparameters from config
    ppo_kwargs = {
        "learning_rate": config.get("learning_rate", 3e-4),
        "n_steps": config.get("n_steps", 2048),
        "batch_size": config.get("batch_size", 64),
        "gamma": config.get("gamma", 0.99),
        "verbose": 1,
        "seed": seed,
    }
    ppo_kwargs.update(kwargs)

    model = PPO("MlpPolicy", train_env, **ppo_kwargs)

    # Setup evaluation callback
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_path / "models"),
        log_path=str(out_path / "logs"),
        eval_freq=max(timesteps // 20, 1000),
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train
    model.learn(total_timesteps=timesteps, callback=eval_callback)

    # Evaluate final model
    eval_rewards = []
    eval_lengths = []
    single_env = gym.make(env_id)
    for i in range(100):
        obs, _ = single_env.reset(seed=seed + 2000 + i)
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = single_env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        eval_rewards.append(total_reward)
        eval_lengths.append(steps)
    single_env.close()

    results = {
        "env_id": env_id,
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "mean_episode_length": float(np.mean(eval_lengths)),
        "training_steps": timesteps,
        "seed": seed,
    }

    # Save model
    model_path = out_path / "models" / f"{env_id.replace('/', '_')}_final"
    model.save(str(model_path))

    train_env.close()
    eval_env.close()

    return results


def main() -> None:
    """CLI entrypoint for PPO training."""
    parser = argparse.ArgumentParser(description="Train PPO on ImmunoSim environments")
    parser.add_argument(
        "--env",
        type=str,
        default="immunosim/CheckpointInhibitor-v0",
        choices=list(ENV_CONFIGS.keys()),
        help="Environment ID",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")

    args = parser.parse_args()

    results = train_ppo(
        env_id=args.env,
        total_timesteps=args.timesteps,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
