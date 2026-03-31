"""CAR-T cell therapy mathematical models.

Implements:
  - CARTmathModel: 4-compartment ODE from Barros et al. (2021) "CARTmath"
  - CRSToxicityModel: Cytokine release syndrome model from Santurio et al. (2025)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


CARTMATH_PARAMETER_RANGES: dict[str, dict[str, Any]] = {
    "rho_t": {
        "default": 0.3,
        "range": [0.1, 1.0],
        "unit": "1/day",
        "description": "Tumor cell proliferation rate",
        "source": "Barros 2021 Table 1",
    },
    "K_t": {
        "default": 1.0e9,
        "range": [1.0e7, 1.0e11],
        "unit": "cells",
        "description": "Tumor carrying capacity",
        "source": "Barros 2021 Table 1",
    },
    "k_act": {
        "default": 0.5,
        "range": [0.1, 2.0],
        "unit": "1/day",
        "description": "Rate of injected CAR-T activation to effector state",
        "source": "Barros 2021 Table 1",
    },
    "gamma_e": {
        "default": 1.0e-8,
        "range": [1.0e-10, 1.0e-6],
        "unit": "1/(cell*day)",
        "description": "Effector CAR-T kill rate per tumor cell encounter",
        "source": "Barros 2021 Table 1",
    },
    "rho_e": {
        "default": 0.2,
        "range": [0.05, 1.0],
        "unit": "1/day",
        "description": "Effector CAR-T proliferation rate (antigen-stimulated)",
        "source": "Barros 2021 Table 1",
    },
    "K_e": {
        "default": 1.0e8,
        "range": [1.0e6, 1.0e10],
        "unit": "cells",
        "description": "Effector CAR-T carrying capacity (self-regulation)",
        "source": "Barros 2021 Table 1",
    },
    "delta_e": {
        "default": 0.02,
        "range": [0.005, 0.1],
        "unit": "1/day",
        "description": "Effector CAR-T death rate",
        "source": "Barros 2021 Table 1",
    },
    "k_mem": {
        "default": 0.01,
        "range": [0.001, 0.05],
        "unit": "1/day",
        "description": "Effector-to-memory CAR-T differentiation rate",
        "source": "Barros 2021 Table 1",
    },
    "delta_m": {
        "default": 0.001,
        "range": [0.0001, 0.01],
        "unit": "1/day",
        "description": "Memory CAR-T death rate (very slow)",
        "source": "Barros 2021 Table 1",
    },
    "k_reactivate": {
        "default": 0.05,
        "range": [0.01, 0.2],
        "unit": "1/day",
        "description": "Memory CAR-T reactivation rate upon antigen re-encounter",
        "source": "Barros 2021 Table 1",
    },
    "k_suppress": {
        "default": 1.0e-9,
        "range": [1.0e-11, 1.0e-7],
        "unit": "1/(cell*day)",
        "description": "Tumor-induced immunosuppression of effector CAR-T",
        "source": "Barros 2021 Table 1",
    },
    "delta_i": {
        "default": 0.1,
        "range": [0.01, 0.5],
        "unit": "1/day",
        "description": "Injected (non-activated) CAR-T clearance rate",
        "source": "Barros 2021 estimated",
    },
}


@dataclass
class CARTmathModel:
    """Four-compartment CAR-T cell therapy ODE model.

    Based on Barros et al. (2021) "CARTmath - a mathematical model of CAR-T
    immunotherapy in preclinical studies of hematological cancers,"
    Cancers 13(12), 2941.

    State variables:
        I: Injected (non-activated) CAR-T cells
        E: Effector CAR-T cells (activated, cytotoxic)
        M: Memory CAR-T cells (long-lived, reactivatable)
        T: Tumor cells (antigen-positive)

    ODEs (Barros 2021 Eqs. 1-4, simplified):
        dI/dt = -k_act * I - delta_i * I + infusion(t)
        dE/dt = k_act * I + rho_e * E * T/(K_e + T) - gamma_e * E * T
                - delta_e * E - k_mem * E - k_suppress * T * E
                + k_reactivate * M * T/(K_e + T)
        dM/dt = k_mem * E - delta_m * M - k_reactivate * M * T/(K_e + T)
        dT/dt = rho_t * T * (1 - T/K_t) - gamma_e * E * T

    SIMPLIFICATION: Antigen-negative tumor escape population omitted.
    Barros 2021 and Santurio 2025 include antigen-loss variants, but for
    the RL environment we focus on the primary antigen-positive tumor.
    Antigen loss would require extending the action space to include
    subsequent therapies targeting different antigens.

    SIMPLIFICATION: CAR-T activation is modeled as first-order kinetics
    (k_act * I) rather than antigen-dependent activation requiring explicit
    tumor-T cell contact modeling.
    """

    rho_t: float = 0.3
    K_t: float = 1.0e9
    k_act: float = 0.5
    gamma_e: float = 1.0e-8
    rho_e: float = 0.2
    K_e: float = 1.0e8
    delta_e: float = 0.02
    k_mem: float = 0.01
    delta_m: float = 0.001
    k_reactivate: float = 0.05
    k_suppress: float = 1.0e-9
    delta_i: float = 0.1

    def derivatives(
        self, t: float, state: NDArray[np.float64], infusion_rate: float = 0.0
    ) -> NDArray[np.float64]:
        """Compute derivatives for the 4-compartment system.

        Args:
            t: Current time.
            state: [I, E, M, T] cell counts.
            infusion_rate: External CAR-T infusion rate (cells/day).

        Returns:
            [dI/dt, dE/dt, dM/dt, dT/dt].
        """
        inj, eff, mem, tum = [max(x, 0.0) for x in state]

        # Antigen-dependent stimulation (Michaelis-Menten)
        ag_stimulation = tum / (self.K_e + tum) if (self.K_e + tum) > 0 else 0.0

        dI_dt = infusion_rate - self.k_act * inj - self.delta_i * inj

        dE_dt = (
            self.k_act * inj
            + self.rho_e * eff * ag_stimulation
            - self.gamma_e * eff * tum
            - self.delta_e * eff
            - self.k_mem * eff
            - self.k_suppress * tum * eff
            + self.k_reactivate * mem * ag_stimulation
        )

        dM_dt = self.k_mem * eff - self.delta_m * mem - self.k_reactivate * mem * ag_stimulation

        dT_dt = self.rho_t * tum * (1.0 - tum / self.K_t) - self.gamma_e * eff * tum

        return np.array([dI_dt, dE_dt, dM_dt, dT_dt])

    def simulate(
        self,
        initial_state: NDArray[np.float64],
        t_span: tuple[float, float],
        infusion_rate: float = 0.0,
        t_eval: NDArray[np.float64] | None = None,
        max_step: float = 0.5,
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate the CAR-T ODE system."""
        sol = solve_ivp(
            lambda t, y: self.derivatives(t, y, infusion_rate),
            t_span,
            initial_state,
            method="RK45",
            t_eval=t_eval,
            max_step=max_step,
            rtol=1e-8,
            atol=1e-10,
        )
        return {
            "t": sol.t,
            "I": sol.y[0],
            "E": sol.y[1],
            "M": sol.y[2],
            "T": sol.y[3],
        }

    def validate_parameters(self) -> list[str]:
        """Check parameters against literature ranges."""
        warnings = []
        for name, spec in CARTMATH_PARAMETER_RANGES.items():
            val = getattr(self, name)
            lo, hi = spec["range"]
            if val < lo or val > hi:
                warnings.append(
                    f"{name}={val} outside [{lo}, {hi}] ({spec['source']})"
                )
        return warnings


# --- CRS Toxicity Model ---
CRS_PARAMETER_RANGES: dict[str, dict[str, Any]] = {
    "cytokine_production_rate": {
        "default": 1.0e-6,
        "range": [1.0e-8, 1.0e-4],
        "unit": "pg/(mL*cell*day)",
        "description": "Cytokine production rate per effector CAR-T cell",
        "source": "Santurio 2025 Table 2",
    },
    "cytokine_clearance": {
        "default": 0.5,
        "range": [0.1, 2.0],
        "unit": "1/day",
        "description": "Cytokine clearance rate",
        "source": "Santurio 2025 Table 2",
    },
    "crs_grade1_threshold": {
        "default": 50.0,
        "range": [20.0, 100.0],
        "unit": "pg/mL",
        "description": "Cytokine level threshold for CRS grade 1",
        "source": "Santurio 2025 clinical thresholds",
    },
    "crs_grade2_threshold": {
        "default": 200.0,
        "range": [100.0, 500.0],
        "unit": "pg/mL",
        "description": "Cytokine level threshold for CRS grade 2",
        "source": "Santurio 2025 clinical thresholds",
    },
    "crs_grade3_threshold": {
        "default": 500.0,
        "range": [200.0, 1000.0],
        "unit": "pg/mL",
        "description": "Cytokine level threshold for CRS grade 3",
        "source": "Santurio 2025 clinical thresholds",
    },
    "crs_grade4_threshold": {
        "default": 1000.0,
        "range": [500.0, 5000.0],
        "unit": "pg/mL",
        "description": "Cytokine level threshold for CRS grade 4 (life-threatening)",
        "source": "Santurio 2025 clinical thresholds",
    },
}


@dataclass
class CRSToxicityModel:
    """Cytokine Release Syndrome toxicity model.

    Based on Santurio et al. (2025) "Mathematical modeling unveils the timeline
    of CAR-T cell therapy and macrophage-mediated cytokine release syndrome,"
    PLOS Computational Biology 21(4), e1012908.

    SIMPLIFICATION: The full Santurio 2025 model includes three macrophage
    activation mechanisms (DAMPs from tumor lysis, antigen-binding activation,
    CD40 contact-dependent activation). Here we aggregate these into a single
    cytokine production rate proportional to effector CAR-T cell count and
    tumor kill rate. This captures the key clinical observation that CRS
    severity correlates with tumor burden AND CAR-T expansion, not either alone.

    SIMPLIFICATION: Cytokine dynamics modeled as a single aggregate (e.g., IL-6)
    rather than the full cytokine panel (IL-6, IL-1, TNF-alpha, IFN-gamma).
    """

    cytokine_production_rate: float = 1.0e-6
    cytokine_clearance: float = 0.5
    crs_grade1_threshold: float = 50.0
    crs_grade2_threshold: float = 200.0
    crs_grade3_threshold: float = 500.0
    crs_grade4_threshold: float = 1000.0

    def update_cytokine_level(
        self,
        current_level: float,
        effector_cart: float,
        tumor_cells: float,
        dt: float,
    ) -> float:
        """Update cytokine level based on CAR-T activity and tumor burden.

        Cytokine production is proportional to the product of effector CAR-T
        and tumor cells (representing active killing and subsequent macrophage
        activation from DAMPs and direct contact).

        Args:
            current_level: Current cytokine level (pg/mL).
            effector_cart: Effector CAR-T cell count.
            tumor_cells: Tumor cell count.
            dt: Time step in days.

        Returns:
            Updated cytokine level.
        """
        production = self.cytokine_production_rate * effector_cart * tumor_cells
        clearance = self.cytokine_clearance * current_level

        new_level = current_level + (production - clearance) * dt
        return float(max(new_level, 0.0))

    def crs_grade(self, cytokine_level: float) -> int:
        """Determine CRS grade from cytokine level.

        Returns:
            Grade 0-4 based on Santurio 2025 thresholds.
        """
        if cytokine_level >= self.crs_grade4_threshold:
            return 4
        elif cytokine_level >= self.crs_grade3_threshold:
            return 3
        elif cytokine_level >= self.crs_grade2_threshold:
            return 2
        elif cytokine_level >= self.crs_grade1_threshold:
            return 1
        return 0

    def toxicity_penalty(self, cytokine_level: float) -> float:
        """Compute CRS toxicity penalty for reward function.

        Non-linear penalty that escalates with CRS grade:
          Grade 0: 0.0
          Grade 1: 0.1
          Grade 2: 0.3
          Grade 3: 0.7
          Grade 4: 1.0 (triggers episode termination)
        """
        grade = self.crs_grade(cytokine_level)
        penalties = {0: 0.0, 1: 0.1, 2: 0.3, 3: 0.7, 4: 1.0}
        return penalties[grade]

    def validate_parameters(self) -> list[str]:
        """Check parameters against literature ranges."""
        warnings = []
        for name, spec in CRS_PARAMETER_RANGES.items():
            val = getattr(self, name)
            lo, hi = spec["range"]
            if val < lo or val > hi:
                warnings.append(
                    f"{name}={val} outside [{lo}, {hi}] ({spec['source']})"
                )
        return warnings
