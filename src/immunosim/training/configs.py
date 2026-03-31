"""Shared training configurations for all ImmunoSim environments.

This is the single source of truth for environment-specific hyperparameters.
Both ppo.py CLI and train_all.py import from here.
"""

ENV_CONFIGS: dict[str, dict] = {
    "immunosim/CheckpointInhibitor-v0": {
        "total_timesteps": 200_000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "description": "Anti-PD-1 dosing optimization",
    },
    "immunosim/CombinationTherapy-v0": {
        "total_timesteps": 500_000,
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "gamma": 0.995,
        "description": "Dual checkpoint blockade optimization",
    },
    "immunosim/CARTCell-v0": {
        "total_timesteps": 300_000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "description": "CAR-T cell infusion optimization",
    },
    "immunosim/AdaptiveDosing-v0": {
        "total_timesteps": 500_000,
        "learning_rate": 1e-4,
        "n_steps": 4096,
        "batch_size": 128,
        "gamma": 0.998,
        "description": "Adaptive dosing with pseudo-progression handling",
    },
}
