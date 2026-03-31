"""CheckpointInhibitorEnv-v0: Optimize anti-PD-1 dosing schedule for tumor control.

Uses Kuznetsov-Taylor (1994) tumor-immune ODE extended with Nikolopoulou (2018)
anti-PD-1 pharmacodynamics.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from immunosim.models.checkpoint_inhibitor import AntiPD1Module
from immunosim.models.tumor_immune import KuznetsovTaylorModel


class CheckpointInhibitorEnv(gym.Env):
    """Gymnasium environment for anti-PD-1 checkpoint inhibitor dosing optimization.

    Observation Space: Box(5)
        [tumor_volume, effector_cells, drug_concentration, time_since_last_dose, treatment_cycle]

    Action Space: Discrete(3)
        0: no dose
        1: low dose (1 mg/kg nivolumab)
        2: high dose (3 mg/kg nivolumab)

    Reward:
        -tumor_volume_change + tumor_reduction_bonus - toxicity_penalty - cumulative_drug_penalty

    Episode: 180 days (26 treatment cycles of ~7 days)
        Terminates early if tumor exceeds lethal threshold or is eliminated.
    """

    metadata = {"render_modes": []}

    # Episode parameters
    MAX_DAYS: float = 180.0
    CYCLE_LENGTH: float = 7.0
    LETHAL_TUMOR_THRESHOLD: float = 1.0e9
    TUMOR_ELIMINATION_THRESHOLD: float = 100.0

    # Reward scaling
    TUMOR_CHANGE_SCALE: float = 1e-7
    REDUCTION_BONUS: float = 5.0
    TOXICITY_PENALTY_SCALE: float = 10.0
    DRUG_COST_SCALE: float = 0.1

    def __init__(
        self,
        tumor_model: KuznetsovTaylorModel | None = None,
        drug_module: AntiPD1Module | None = None,
        initial_tumor: float = 1.0e6,
        initial_effector: float = 3.0e5,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.tumor_model = tumor_model or KuznetsovTaylorModel()
        self.drug_module = drug_module or AntiPD1Module()
        self.initial_tumor = initial_tumor
        self.initial_effector = initial_effector

        # Observation: [tumor, effector, drug_conc, time_since_dose, cycle]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1e10, 1e10, 1000.0, 30.0, 30.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: 0=no dose, 1=low dose, 2=high dose
        self.action_space = spaces.Discrete(3)

        # State tracking
        self.tumor: float = 0.0
        self.effector: float = 0.0
        self.drug_conc: float = 0.0
        self.current_day: float = 0.0
        self.time_since_last_dose: float = 0.0
        self.cycle: int = 0
        self.cumulative_drug: float = 0.0
        self.prev_tumor: float = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        self.tumor = self.initial_tumor
        self.effector = self.initial_effector
        self.drug_conc = 0.0
        self.current_day = 0.0
        self.time_since_last_dose = 0.0
        self.cycle = 0
        self.cumulative_drug = 0.0
        self.prev_tumor = self.tumor

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        action = int(action)
        self.prev_tumor = self.tumor

        # Apply drug action
        self.drug_conc = self.drug_module.drug_concentration_update(
            self.drug_conc, action, 0.0  # bolus at start of cycle
        )
        if action > 0:
            self.time_since_last_dose = 0.0
            dose_amount = (
                self.drug_module.dose_low_pd1 if action == 1
                else self.drug_module.dose_high_pd1
            )
            self.cumulative_drug += dose_amount

        # Simulate ODE for one treatment cycle
        dt = self.CYCLE_LENGTH
        immune_boost = self.drug_module.immune_boost_factor(self.drug_conc)

        # Create boosted model for this step
        boosted_mu = self.tumor_model.mu * immune_boost
        original_mu = self.tumor_model.mu
        self.tumor_model.mu = boosted_mu

        state = np.array([self.effector, self.tumor])
        result = self.tumor_model.simulate(
            state,
            (0.0, dt),
            t_eval=np.array([dt]),
        )

        self.tumor_model.mu = original_mu

        self.effector = float(max(result["E"][-1], 0.0))
        self.tumor = float(max(result["T"][-1], 0.0))

        # Update drug PK (decay over cycle)
        self.drug_conc = self.drug_module.drug_concentration_update(
            self.drug_conc, 0, dt  # no new dose, just decay
        )

        self.current_day += dt
        self.time_since_last_dose += dt
        self.cycle += 1

        # Compute reward
        tumor_change = self.tumor - self.prev_tumor
        reward = -tumor_change * self.TUMOR_CHANGE_SCALE

        # Bonus for reducing tumor
        if self.tumor < self.prev_tumor * 0.9:
            reward += self.REDUCTION_BONUS

        # Toxicity penalty (flat for PD-1 per Shulgin 2020)
        tox = self.drug_module.toxicity_score(self.drug_conc)
        reward -= tox * self.TOXICITY_PENALTY_SCALE

        # Cumulative drug cost
        reward -= self.cumulative_drug * self.DRUG_COST_SCALE / max(self.cycle, 1)

        # Check termination conditions
        terminated = False
        truncated = False

        if self.tumor >= self.LETHAL_TUMOR_THRESHOLD:
            terminated = True
            reward -= 50.0  # Large penalty for tumor escape

        if self.tumor <= self.TUMOR_ELIMINATION_THRESHOLD:
            terminated = True
            reward += 100.0  # Large bonus for elimination

        if self.current_day >= self.MAX_DAYS:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> NDArray[np.float32]:
        return np.array(
            [
                self.tumor,
                self.effector,
                self.drug_conc,
                self.time_since_last_dose,
                float(self.cycle),
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "tumor_volume": self.tumor,
            "effector_cells": self.effector,
            "drug_concentration": self.drug_conc,
            "current_day": self.current_day,
            "cycle": self.cycle,
            "cumulative_drug": self.cumulative_drug,
        }
