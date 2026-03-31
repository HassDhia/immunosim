"""ImmunoSim: Gymnasium environments for RL in cancer immunotherapy optimization."""

__version__ = "0.1.0"

from gymnasium.envs.registration import register

register(
    id="immunosim/CheckpointInhibitor-v0",
    entry_point="immunosim.envs.checkpoint_inhibitor:CheckpointInhibitorEnv",
)

register(
    id="immunosim/CombinationTherapy-v0",
    entry_point="immunosim.envs.combination_therapy:CombinationTherapyEnv",
)

register(
    id="immunosim/CARTCell-v0",
    entry_point="immunosim.envs.cart_cell:CARTCellEnv",
)

register(
    id="immunosim/AdaptiveDosing-v0",
    entry_point="immunosim.envs.adaptive_dosing:AdaptiveDosingEnv",
)
