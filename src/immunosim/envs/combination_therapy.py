"""CombinationTherapyEnv-v0: Optimize dual anti-PD-1 + anti-CTLA-4 checkpoint blockade.

Uses Nikolopoulou (2021) synergy model with Shulgin (2020) dose-dependent toxicity.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from immunosim.models.checkpoint_inhibitor import DualCheckpointModule
from immunosim.models.tumor_immune import KuznetsovTaylorModel


class CombinationTherapyEnv(gym.Env):
    """Gymnasium environment for dual checkpoint blockade optimization.

    Observation Space: Box(7)
        [tumor_volume, effector_cells, drug1_conc (PD-1), drug2_conc (CTLA-4),
         toxicity_score, time_since_dose1, time_since_dose2]

    Action Space: MultiDiscrete([3, 3])
        drug1_action: [no_dose, low_dose, high_dose]
        drug2_action: [no_dose, low_dose, high_dose]

    Key: CTLA-4 has dose-dependent toxicity, PD-1 does not (Shulgin 2020).
    """

    metadata = {"render_modes": []}

    MAX_DAYS: float = 180.0
    CYCLE_LENGTH: float = 7.0
    LETHAL_TUMOR_THRESHOLD: float = 1.0e9
    TUMOR_ELIMINATION_THRESHOLD: float = 100.0
    GRADE4_TOXICITY_THRESHOLD: float = 0.8

    TUMOR_CHANGE_SCALE: float = 1e-7
    REDUCTION_BONUS: float = 5.0
    TOXICITY_PENALTY_SCALE: float = 15.0
    DRUG_COST_SCALE: float = 0.15

    def __init__(
        self,
        tumor_model: KuznetsovTaylorModel | None = None,
        drug_module: DualCheckpointModule | None = None,
        initial_tumor: float = 1.0e6,
        initial_effector: float = 3.0e5,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.tumor_model = tumor_model or KuznetsovTaylorModel()
        self.drug_module = drug_module or DualCheckpointModule()
        self.initial_tumor = initial_tumor
        self.initial_effector = initial_effector

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1e10, 1e10, 1000.0, 1000.0, 1.0, 30.0, 30.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.MultiDiscrete([3, 3])

        self.tumor: float = 0.0
        self.effector: float = 0.0
        self.pd1_conc: float = 0.0
        self.ctla4_conc: float = 0.0
        self.toxicity: float = 0.0
        self.current_day: float = 0.0
        self.time_since_pd1: float = 0.0
        self.time_since_ctla4: float = 0.0
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
        self.pd1_conc = 0.0
        self.ctla4_conc = 0.0
        self.toxicity = 0.0
        self.current_day = 0.0
        self.time_since_pd1 = 0.0
        self.time_since_ctla4 = 0.0
        self.cycle = 0
        self.cumulative_drug = 0.0
        self.prev_tumor = self.tumor

        return self._get_obs(), self._get_info()

    def step(
        self, action: NDArray[np.int64]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        pd1_action = int(action[0])
        ctla4_action = int(action[1])

        self.prev_tumor = self.tumor

        # Apply drug doses
        self.pd1_conc = self.drug_module.pd1.drug_concentration_update(
            self.pd1_conc, pd1_action, 0.0
        )
        self.ctla4_conc = self.drug_module.ctla4.drug_concentration_update(
            self.ctla4_conc, ctla4_action, 0.0
        )

        if pd1_action > 0:
            self.time_since_pd1 = 0.0
        if ctla4_action > 0:
            self.time_since_ctla4 = 0.0

        # Track cumulative drug exposure
        if pd1_action == 1:
            self.cumulative_drug += self.drug_module.pd1.dose_low_pd1
        elif pd1_action == 2:
            self.cumulative_drug += self.drug_module.pd1.dose_high_pd1
        if ctla4_action == 1:
            self.cumulative_drug += self.drug_module.ctla4.dose_low_ctla4
        elif ctla4_action == 2:
            self.cumulative_drug += self.drug_module.ctla4.dose_high_ctla4

        # Compute combined immune boost (includes synergy)
        combined_boost = self.drug_module.combined_immune_boost(
            self.pd1_conc, self.ctla4_conc
        )

        # Simulate ODE with boosted parameters
        dt = self.CYCLE_LENGTH
        original_mu = self.tumor_model.mu
        original_sigma = self.tumor_model.sigma

        self.tumor_model.mu = original_mu * combined_boost
        # CTLA-4 also boosts immune priming (sigma)
        ctla4_priming = self.drug_module.ctla4.immune_priming_boost(self.ctla4_conc)
        self.tumor_model.sigma = original_sigma * ctla4_priming

        state = np.array([self.effector, self.tumor])
        result = self.tumor_model.simulate(state, (0.0, dt), t_eval=np.array([dt]))

        self.tumor_model.mu = original_mu
        self.tumor_model.sigma = original_sigma

        self.effector = float(max(result["E"][-1], 0.0))
        self.tumor = float(max(result["T"][-1], 0.0))

        # Update drug PK
        self.pd1_conc = self.drug_module.pd1.drug_concentration_update(
            self.pd1_conc, 0, dt
        )
        self.ctla4_conc = self.drug_module.ctla4.drug_concentration_update(
            self.ctla4_conc, 0, dt
        )

        # Update toxicity (combined)
        self.toxicity = self.drug_module.combined_toxicity(self.pd1_conc, self.ctla4_conc)

        self.current_day += dt
        self.time_since_pd1 += dt
        self.time_since_ctla4 += dt
        self.cycle += 1

        # Reward computation
        tumor_change = self.tumor - self.prev_tumor
        reward = -tumor_change * self.TUMOR_CHANGE_SCALE

        if self.tumor < self.prev_tumor * 0.9:
            reward += self.REDUCTION_BONUS

        reward -= self.toxicity * self.TOXICITY_PENALTY_SCALE
        reward -= self.cumulative_drug * self.DRUG_COST_SCALE / max(self.cycle, 1)

        # Termination
        terminated = False
        truncated = False

        if self.tumor >= self.LETHAL_TUMOR_THRESHOLD:
            terminated = True
            reward -= 50.0

        if self.tumor <= self.TUMOR_ELIMINATION_THRESHOLD:
            terminated = True
            reward += 100.0

        if self.toxicity >= self.GRADE4_TOXICITY_THRESHOLD:
            terminated = True
            reward -= 30.0  # Grade 4 toxicity termination

        if self.current_day >= self.MAX_DAYS:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _get_obs(self) -> NDArray[np.float32]:
        return np.array(
            [
                self.tumor,
                self.effector,
                self.pd1_conc,
                self.ctla4_conc,
                self.toxicity,
                self.time_since_pd1,
                self.time_since_ctla4,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "tumor_volume": self.tumor,
            "effector_cells": self.effector,
            "pd1_concentration": self.pd1_conc,
            "ctla4_concentration": self.ctla4_conc,
            "toxicity_score": self.toxicity,
            "current_day": self.current_day,
            "cycle": self.cycle,
            "cumulative_drug": self.cumulative_drug,
        }
