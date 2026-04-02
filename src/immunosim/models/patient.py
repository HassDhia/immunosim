"""Patient generator and pseudo-progression detector.

Implements:
  - PatientGenerator: Domain randomization over physiological parameters
  - PseudoProgressionDetector: Distinguish pseudo-progression from true progression
    based on Butner-Cristini (2020) three-parameter patient model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from immunosim.models.tumor_immune import PARAMETER_RANGES as KT_RANGES


PATIENT_PARAMETER_RANGES: dict[str, dict[str, Any]] = {
    "alpha_patient": {
        "default": 0.18,
        "range": [0.05, 0.5],
        "unit": "1/day",
        "description": "Patient-specific tumor growth rate",
        "source": "Butner 2020 Table 1 (fit from CT imaging, 245 patients)",
    },
    "lambda_immune": {
        "default": 0.1,
        "range": [0.01, 0.5],
        "unit": "dimensionless",
        "description": "Immune infiltration parameter (Lambda in Butner 2020)",
        "source": "Butner 2020 Table 1",
    },
    "mu_amplification": {
        "default": 1.0,
        "range": [0.1, 5.0],
        "unit": "dimensionless",
        "description": "Immunotherapy amplification factor",
        "source": "Butner 2020 Table 1",
    },
    "initial_tumor_volume": {
        "default": 1.0e6,
        "range": [1.0e4, 1.0e8],
        "unit": "cells",
        "description": "Initial tumor burden at diagnosis",
        "source": "Clinical range",
    },
    "initial_effector_cells": {
        "default": 3.0e5,
        "range": [1.0e4, 1.0e7],
        "unit": "cells",
        "description": "Baseline circulating effector immune cells",
        "source": "Kuznetsov-Taylor-Perelson 1994 estimated equilibrium",
    },
}


@dataclass
class PatientGenerator:
    """Generate randomized patient parameter sets for domain randomization.

    Based on Eastman et al. (2021) approach: train RL with randomized patient
    parameters to learn robust policies. Parameter ranges from Butner et al.
    (2020) 245-patient cohort and Kuznetsov-Taylor-Perelson 1994 parameter estimation.

    SIMPLIFICATION: Assumes independent parameter distributions. In reality,
    tumor growth rate and immune infiltration are correlated (fast-growing
    tumors tend to be more immunogenic). Future work could use copula-based
    joint distributions from the Butner 2020 patient cohort.
    """

    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    def generate(self, n: int = 1) -> list[dict[str, float]]:
        """Generate n randomized patient parameter sets.

        Uses log-uniform sampling for parameters spanning orders of magnitude
        and uniform sampling for bounded parameters.

        Args:
            n: Number of patients to generate.

        Returns:
            List of parameter dictionaries.
        """
        patients = []
        for _ in range(n):
            patient = {}

            # Tumor growth rate: uniform within clinical range
            patient["alpha"] = float(
                self.rng.uniform(
                    PATIENT_PARAMETER_RANGES["alpha_patient"]["range"][0],
                    PATIENT_PARAMETER_RANGES["alpha_patient"]["range"][1],
                )
            )

            # Immune infiltration: uniform
            patient["lambda_immune"] = float(
                self.rng.uniform(
                    PATIENT_PARAMETER_RANGES["lambda_immune"]["range"][0],
                    PATIENT_PARAMETER_RANGES["lambda_immune"]["range"][1],
                )
            )

            # Immunotherapy amplification: log-uniform
            lo, hi = PATIENT_PARAMETER_RANGES["mu_amplification"]["range"]
            patient["mu_amplification"] = float(
                np.exp(self.rng.uniform(np.log(lo), np.log(hi)))
            )

            # Initial tumor volume: log-uniform (spans orders of magnitude)
            lo, hi = PATIENT_PARAMETER_RANGES["initial_tumor_volume"]["range"]
            patient["initial_tumor_volume"] = float(
                np.exp(self.rng.uniform(np.log(lo), np.log(hi)))
            )

            # Initial effector cells: log-uniform
            lo, hi = PATIENT_PARAMETER_RANGES["initial_effector_cells"]["range"]
            patient["initial_effector_cells"] = float(
                np.exp(self.rng.uniform(np.log(lo), np.log(hi)))
            )

            # Kuznetsov-Taylor-Perelson model parameters with perturbation
            for param in ["sigma", "delta", "rho", "mu"]:
                default = KT_RANGES[param]["default"]
                lo_kt, hi_kt = KT_RANGES[param]["range"]
                patient[param] = float(
                    np.clip(
                        default * np.exp(self.rng.normal(0, 0.3)),
                        lo_kt,
                        hi_kt,
                    )
                )

            patients.append(patient)

        return patients

    def generate_cohort(
        self, n: int = 100, responder_fraction: float = 0.3
    ) -> list[dict[str, float]]:
        """Generate a virtual patient cohort with specified responder fraction.

        Responders have higher immune infiltration (lambda_immune) and
        immunotherapy amplification (mu_amplification), consistent with
        Butner 2020 and Milberg 2019 virtual patient analyses.
        """
        patients = []
        n_responders = int(n * responder_fraction)

        for i in range(n):
            patient = self.generate(1)[0]
            if i < n_responders:
                # Responder: boost immune parameters
                patient["lambda_immune"] = float(
                    self.rng.uniform(0.2, 0.5)
                )
                patient["mu_amplification"] = float(
                    self.rng.uniform(1.5, 5.0)
                )
            else:
                # Non-responder: lower immune parameters
                patient["lambda_immune"] = float(
                    self.rng.uniform(0.01, 0.15)
                )
                patient["mu_amplification"] = float(
                    self.rng.uniform(0.1, 1.0)
                )
            patient["is_responder"] = i < n_responders
            patients.append(patient)

        return patients


PSEUDO_PROGRESSION_RANGES: dict[str, dict[str, Any]] = {
    "immune_infiltration_delay": {
        "default": 14.0,
        "range": [7.0, 42.0],
        "unit": "days",
        "description": "Typical delay before immune infiltration causes measurable swelling",
        "source": "Butner 2020 (53-day first restaging window)",
    },
    "max_pseudo_growth_factor": {
        "default": 1.3,
        "range": [1.1, 2.0],
        "unit": "dimensionless",
        "description": "Maximum apparent tumor growth before regression in pseudo-progression",
        "source": "Butner 2020 estimated from clinical data",
    },
}


@dataclass
class PseudoProgressionDetector:
    """Detect and model pseudo-progression in immunotherapy.

    Pseudo-progression occurs when immunotherapy causes immune cell infiltration
    into the tumor, temporarily increasing measurable tumor volume before the
    immune cells destroy the tumor. Butner et al. (2020) showed that their
    3-parameter model can distinguish this from true progression at first
    restaging (day 53) with 88% accuracy.

    SIMPLIFICATION: Pseudo-progression detection uses a heuristic based on
    tumor growth rate trajectory rather than the full Butner mechanistic model.
    The detector looks for the characteristic pattern: initial growth followed
    by inflection and decline, with concurrent increase in immune infiltration.
    """

    immune_infiltration_delay: float = 14.0
    max_pseudo_growth_factor: float = 1.3
    window_size: int = 5

    def is_pseudo_progression(
        self,
        tumor_history: NDArray[np.float64],
        immune_history: NDArray[np.float64],
        current_day: float,
        treatment_start_day: float = 0.0,
    ) -> bool:
        """Determine if current tumor growth is likely pseudo-progression.

        Args:
            tumor_history: Recent tumor volume measurements.
            immune_history: Recent effector cell count measurements.
            current_day: Current simulation day.
            treatment_start_day: Day immunotherapy was initiated.

        Returns:
            True if pattern is consistent with pseudo-progression.
        """
        if len(tumor_history) < self.window_size:
            return False

        days_on_treatment = current_day - treatment_start_day

        # Too early for pseudo-progression
        if days_on_treatment < self.immune_infiltration_delay * 0.5:
            return False

        # Check if tumor is growing
        recent_tumor = tumor_history[-self.window_size:]
        tumor_growing = recent_tumor[-1] > recent_tumor[0]
        if not tumor_growing:
            return False

        # Check growth is within pseudo-progression bounds
        growth_factor = recent_tumor[-1] / max(recent_tumor[0], 1.0)
        if growth_factor > self.max_pseudo_growth_factor:
            return False  # Growth too large - likely true progression

        # Check if immune cells are also increasing (hallmark of pseudo-progression)
        recent_immune = immune_history[-self.window_size:]
        immune_increasing = recent_immune[-1] > recent_immune[0] * 1.1

        return bool(immune_increasing)

    def response_trajectory(
        self,
        tumor_history: NDArray[np.float64],
    ) -> float:
        """Compute response trajectory metric.

        Returns a smoothed slope of tumor volume:
          < 0: responding (tumor shrinking)
          ~ 0: stable disease
          > 0: progressive disease or pseudo-progression

        Used as an observation space feature in AdaptiveDosingEnv.
        """
        if len(tumor_history) < 3:
            return 0.0

        # Use log-scale for more stable gradient
        log_tumor = np.log1p(tumor_history[-min(10, len(tumor_history)):])
        if len(log_tumor) < 2:
            return 0.0

        # Linear regression slope
        x = np.arange(len(log_tumor), dtype=np.float64)
        x_mean = x.mean()
        y_mean = log_tumor.mean()
        slope = float(
            np.sum((x - x_mean) * (log_tumor - y_mean))
            / max(np.sum((x - x_mean) ** 2), 1e-10)
        )
        return slope
