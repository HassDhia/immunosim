"""AdaptiveDosingEnv-v0: Adaptive dosing with pseudo-progression handling.

Uses Butner-Cristini (2020) 3-parameter patient model + Kuznetsov-Taylor dynamics.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from immunosim.models.checkpoint_inhibitor import AntiPD1Module
from immunosim.models.patient import PseudoProgressionDetector
from immunosim.models.tumor_immune import KuznetsovTaylorModel


class AdaptiveDosingEnv(gym.Env):
    """Gymnasium environment for adaptive dosing with pseudo-progression handling.

    Observation Space: Box(6)
        [tumor_volume, tumor_growth_rate, immune_infiltration, drug_conc,
         days_on_treatment, response_trajectory]

    Action Space: Discrete(4)
        0: continue current dose
        1: dose escalate
        2: dose reduce
        3: treatment holiday

    Episode: 360 days, handles pseudo-progression (tumor temporarily grows
    before shrinking due to immune infiltration).
    """

    metadata = {"render_modes": []}

    MAX_DAYS: float = 360.0
    CYCLE_LENGTH: float = 14.0  # 2-week cycles for adaptive dosing
    LETHAL_TUMOR_THRESHOLD: float = 1.0e9
    TUMOR_ELIMINATION_THRESHOLD: float = 100.0

    # Dose levels
    DOSE_LEVELS = [0.0, 1.0, 2.0, 3.0, 5.0]  # mg/kg, index by current level
    INITIAL_DOSE_LEVEL: int = 2  # Start at 2 mg/kg

    # Reward scaling
    LONG_TERM_CONTROL_SCALE: float = 1e-7
    PREMATURE_ESCALATION_PENALTY: float = 3.0
    UNNECESSARY_HOLIDAY_PENALTY: float = 2.0
    STABLE_DISEASE_BONUS: float = 1.0

    def __init__(
        self,
        tumor_model: KuznetsovTaylorModel | None = None,
        drug_module: AntiPD1Module | None = None,
        pseudo_detector: PseudoProgressionDetector | None = None,
        initial_tumor: float = 1.0e6,
        initial_effector: float = 3.0e5,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.tumor_model = tumor_model or KuznetsovTaylorModel()
        self.drug_module = drug_module or AntiPD1Module()
        self.pseudo_detector = pseudo_detector or PseudoProgressionDetector()
        self.initial_tumor = initial_tumor
        self.initial_effector = initial_effector

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0.0, 0.0, 0.0, -np.inf], dtype=np.float32),
            high=np.array([1e10, np.inf, 1e10, 1000.0, 400.0, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(4)

        # State
        self.tumor: float = 0.0
        self.effector: float = 0.0
        self.drug_conc: float = 0.0
        self.current_day: float = 0.0
        self.current_dose_level: int = 0
        self.on_holiday: bool = False
        self.prev_tumor: float = 0.0

        # History for pseudo-progression detection
        self.tumor_history: deque[float] = deque(maxlen=20)
        self.immune_history: deque[float] = deque(maxlen=20)

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
        self.current_dose_level = self.INITIAL_DOSE_LEVEL
        self.on_holiday = False
        self.prev_tumor = self.tumor

        self.tumor_history.clear()
        self.immune_history.clear()
        self.tumor_history.append(self.tumor)
        self.immune_history.append(self.effector)

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        action = int(action)
        self.prev_tumor = self.tumor

        # Process action
        if action == 0:
            # Continue current dose
            pass
        elif action == 1:
            # Dose escalate
            self.current_dose_level = min(
                self.current_dose_level + 1, len(self.DOSE_LEVELS) - 1
            )
            self.on_holiday = False
        elif action == 2:
            # Dose reduce
            self.current_dose_level = max(self.current_dose_level - 1, 0)
            self.on_holiday = False
        elif action == 3:
            # Treatment holiday
            self.on_holiday = True

        # Determine actual dose
        if self.on_holiday:
            actual_dose_mgkg = 0.0
        else:
            actual_dose_mgkg = self.DOSE_LEVELS[self.current_dose_level]

        # Apply dose to drug concentration
        if actual_dose_mgkg > 0:
            self.drug_conc += (
                actual_dose_mgkg * 70.0 / self.drug_module.volume_of_distribution
            )

        # Simulate ODE
        dt = self.CYCLE_LENGTH
        immune_boost = self.drug_module.immune_boost_factor(self.drug_conc)

        original_mu = self.tumor_model.mu
        self.tumor_model.mu = original_mu * immune_boost

        state = np.array([self.effector, self.tumor])
        result = self.tumor_model.simulate(state, (0.0, dt), t_eval=np.array([dt]))

        self.tumor_model.mu = original_mu

        self.effector = float(max(result["E"][-1], 0.0))
        self.tumor = float(max(result["T"][-1], 0.0))

        # Drug PK decay
        self.drug_conc = self.drug_module.drug_concentration_update(
            self.drug_conc, 0, dt
        )

        self.current_day += dt

        # Update histories
        self.tumor_history.append(self.tumor)
        self.immune_history.append(self.effector)

        # Detect pseudo-progression
        is_pseudo = self.pseudo_detector.is_pseudo_progression(
            np.array(list(self.tumor_history)),
            np.array(list(self.immune_history)),
            self.current_day,
        )

        # Response trajectory
        trajectory = self.pseudo_detector.response_trajectory(
            np.array(list(self.tumor_history))
        )

        # Reward computation
        reward = 0.0

        # Long-term tumor control (negative = good)
        tumor_change = self.tumor - self.prev_tumor
        reward -= tumor_change * self.LONG_TERM_CONTROL_SCALE

        # Penalize premature escalation during pseudo-progression
        if action == 1 and is_pseudo:
            reward -= self.PREMATURE_ESCALATION_PENALTY

        # Penalize unnecessary treatment holidays when responding
        if action == 3 and self.tumor < self.prev_tumor:
            reward -= self.UNNECESSARY_HOLIDAY_PENALTY

        # Bonus for stable disease (controlled without escalation)
        if abs(tumor_change) < self.prev_tumor * 0.05 and action == 0:
            reward += self.STABLE_DISEASE_BONUS

        # Termination
        terminated = False
        truncated = False

        if self.tumor >= self.LETHAL_TUMOR_THRESHOLD:
            terminated = True
            reward -= 50.0

        if self.tumor <= self.TUMOR_ELIMINATION_THRESHOLD:
            terminated = True
            reward += 100.0

        if self.current_day >= self.MAX_DAYS:
            truncated = True
            # End-of-episode bonus for tumor control
            if self.tumor < self.initial_tumor * 0.5:
                reward += 20.0

        obs = self._get_obs()
        info = self._get_info()
        info["is_pseudo_progression"] = is_pseudo
        info["response_trajectory"] = trajectory

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> NDArray[np.float32]:
        # Tumor growth rate (log-scale difference)
        if len(self.tumor_history) >= 2:
            prev = max(list(self.tumor_history)[-2], 1.0)
            growth_rate = float(np.log(max(self.tumor, 1.0) / prev))
        else:
            growth_rate = 0.0

        trajectory = self.pseudo_detector.response_trajectory(
            np.array(list(self.tumor_history))
        )

        return np.array(
            [
                self.tumor,
                growth_rate,
                self.effector,
                self.drug_conc,
                self.current_day,
                trajectory,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "tumor_volume": self.tumor,
            "effector_cells": self.effector,
            "drug_concentration": self.drug_conc,
            "current_day": self.current_day,
            "dose_level": self.current_dose_level,
            "on_holiday": self.on_holiday,
            "actual_dose_mgkg": (
                0.0 if self.on_holiday
                else self.DOSE_LEVELS[self.current_dose_level]
            ),
        }
