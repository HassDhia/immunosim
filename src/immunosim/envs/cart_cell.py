"""CARTCellEnv-v0: Optimize CAR-T cell infusion timing and dose.

Uses Barros CARTmath (2021) 4-compartment ODE with Santurio (2025) CRS toxicity.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from immunosim.models.cart_cell import CARTmathModel, CRSToxicityModel


class CARTCellEnv(gym.Env):
    """Gymnasium environment for CAR-T cell therapy optimization.

    Observation Space: Box(6)
        [tumor_cells, injected_cart, effector_cart, memory_cart,
         cytokine_level, treatment_day]

    Action Space: Discrete(4)
        0: no infusion
        1: low dose infusion (1e6 cells)
        2: standard infusion (5e6 cells)
        3: high dose infusion (1e7 cells)

    CRS penalty escalates non-linearly with cytokine level (Santurio 2025).
    """

    metadata = {"render_modes": []}

    MAX_DAYS: float = 90.0
    STEP_DAYS: float = 1.0  # Daily decision frequency for CAR-T
    LETHAL_TUMOR_THRESHOLD: float = 1.0e9
    REMISSION_THRESHOLD: float = 10.0

    # Infusion doses (cells)
    DOSE_MAP = {0: 0.0, 1: 1.0e6, 2: 5.0e6, 3: 1.0e7}

    # Reward scaling
    TUMOR_CHANGE_SCALE: float = 1e-7
    KILL_BONUS: float = 3.0
    CRS_PENALTY_SCALE: float = 20.0
    INFUSION_COST_SCALE: float = 0.5

    def __init__(
        self,
        cart_model: CARTmathModel | None = None,
        crs_model: CRSToxicityModel | None = None,
        initial_tumor: float = 1.0e7,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.cart_model = cart_model or CARTmathModel()
        self.crs_model = crs_model or CRSToxicityModel()
        self.initial_tumor = initial_tumor

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1e10, 1e10, 1e10, 1e10, 1e10, 100.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(4)

        self.tumor: float = 0.0
        self.injected: float = 0.0
        self.effector: float = 0.0
        self.memory: float = 0.0
        self.cytokine: float = 0.0
        self.current_day: float = 0.0
        self.prev_tumor: float = 0.0
        self.total_infused: float = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        self.tumor = self.initial_tumor
        self.injected = 0.0
        self.effector = 0.0
        self.memory = 0.0
        self.cytokine = 0.0
        self.current_day = 0.0
        self.prev_tumor = self.tumor
        self.total_infused = 0.0

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        self.prev_tumor = self.tumor

        # Add infusion as bolus to injected compartment
        action = int(action)
        infusion_dose = self.DOSE_MAP[action]
        self.injected += infusion_dose
        self.total_infused += infusion_dose

        # Simulate CAR-T ODE for one day
        dt = self.STEP_DAYS
        state = np.array([self.injected, self.effector, self.memory, self.tumor])
        result = self.cart_model.simulate(state, (0.0, dt), t_eval=np.array([dt]))

        self.injected = float(max(result["I"][-1], 0.0))
        self.effector = float(max(result["E"][-1], 0.0))
        self.memory = float(max(result["M"][-1], 0.0))
        self.tumor = float(max(result["T"][-1], 0.0))

        # Update cytokine level (CRS tracking)
        self.cytokine = self.crs_model.update_cytokine_level(
            self.cytokine, self.effector, self.tumor, dt
        )

        self.current_day += dt

        # Reward computation
        tumor_change = self.tumor - self.prev_tumor
        reward = -tumor_change * self.TUMOR_CHANGE_SCALE

        # Bonus for tumor kill
        if self.tumor < self.prev_tumor * 0.95:
            reward += self.KILL_BONUS

        # CRS toxicity penalty (non-linear, escalating)
        crs_penalty = self.crs_model.toxicity_penalty(self.cytokine)
        reward -= crs_penalty * self.CRS_PENALTY_SCALE

        # Infusion cost
        if infusion_dose > 0:
            reward -= self.INFUSION_COST_SCALE * (infusion_dose / 1.0e7)

        # Termination
        terminated = False
        truncated = False

        if self.tumor <= self.REMISSION_THRESHOLD:
            terminated = True
            reward += 100.0  # Complete remission

        if self.tumor >= self.LETHAL_TUMOR_THRESHOLD:
            terminated = True
            reward -= 50.0

        crs_grade = self.crs_model.crs_grade(self.cytokine)
        if crs_grade >= 4:
            terminated = True
            reward -= 40.0  # Grade 4 CRS

        if self.current_day >= self.MAX_DAYS:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _get_obs(self) -> NDArray[np.float32]:
        return np.array(
            [
                self.tumor,
                self.injected,
                self.effector,
                self.memory,
                self.cytokine,
                self.current_day,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "tumor_cells": self.tumor,
            "injected_cart": self.injected,
            "effector_cart": self.effector,
            "memory_cart": self.memory,
            "cytokine_level": self.cytokine,
            "crs_grade": self.crs_model.crs_grade(self.cytokine),
            "current_day": self.current_day,
            "total_infused": self.total_infused,
        }
