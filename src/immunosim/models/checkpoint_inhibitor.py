"""Checkpoint inhibitor pharmacodynamic models.

Implements:
  - AntiPD1Module: PD-1 blockade (Nikolopoulou et al. 2018)
  - AntiCTLA4Module: CTLA-4 blockade with dose-dependent toxicity (Shulgin et al. 2020)
  - DualCheckpointModule: Combined PD-1 + CTLA-4 with synergy (Nikolopoulou et al. 2021)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# --- AntiPD1Module parameters ---
ANTI_PD1_PARAMETER_RANGES: dict[str, dict[str, Any]] = {
    "k_pd1": {
        "default": 0.8,
        "range": [0.3, 1.0],
        "unit": "dimensionless",
        "description": "Maximum PD-1 blockade efficacy (fraction of PD-1/PD-L1 axis blocked)",
        "source": "Nikolopoulou 2018 Table 2",
    },
    "half_life_pd1": {
        "default": 25.0,
        "range": [12.0, 30.0],
        "unit": "days",
        "description": "Anti-PD-1 antibody half-life (nivolumab range from Bajaj 2017)",
        "source": "Bajaj 2017 Table 3",
    },
    "ec50_pd1": {
        "default": 5.0,
        "range": [1.0, 20.0],
        "unit": "mg/L",
        "description": "Half-maximal effective concentration for PD-1 receptor occupancy",
        "source": "Nikolopoulou 2018 estimated",
    },
    "dose_low_pd1": {
        "default": 1.0,
        "range": [0.5, 3.0],
        "unit": "mg/kg",
        "description": "Low-dose anti-PD-1 (nivolumab 1 mg/kg)",
        "source": "Clinical dosing, Bajaj 2017",
    },
    "dose_high_pd1": {
        "default": 3.0,
        "range": [2.0, 10.0],
        "unit": "mg/kg",
        "description": "High-dose anti-PD-1 (nivolumab 3 mg/kg)",
        "source": "Clinical dosing, Bajaj 2017",
    },
    "volume_of_distribution": {
        "default": 8.0,
        "range": [5.0, 12.0],
        "unit": "L",
        "description": "Central compartment volume of distribution",
        "source": "Bajaj 2017 Table 2",
    },
    "toxicity_base_pd1": {
        "default": 0.05,
        "range": [0.02, 0.10],
        "unit": "probability",
        "description": "Flat baseline irAE probability per cycle for anti-PD-1 (dose-independent)",
        "source": "Shulgin 2020 Fig 3: PD-1 toxicity is NOT dose-dependent",
    },
}


@dataclass
class AntiPD1Module:
    """Anti-PD-1 checkpoint inhibitor pharmacodynamics.

    Based on Nikolopoulou et al. (2018) "Tumour-immune dynamics with an immune
    checkpoint inhibitor," Letters in Biomathematics 5(sup1), S137-S159.

    SIMPLIFICATION: Uses 1-compartment PK instead of the 2-compartment model
    from Bajaj 2017. At clinical doses, target-mediated drug disposition is
    saturated and PK is approximately linear, so a single compartment with
    first-order elimination captures the essential concentration-time profile.

    SIMPLIFICATION: PD-1 blockade effect modeled as a scalar multiplier on
    effector cell kill rate rather than explicit receptor-ligand binding
    kinetics. The Hill-type occupancy function approximates the sigmoidal
    dose-response observed clinically.
    """

    k_pd1: float = 0.8
    half_life_pd1: float = 25.0
    ec50_pd1: float = 5.0
    dose_low_pd1: float = 1.0
    dose_high_pd1: float = 3.0
    volume_of_distribution: float = 8.0
    toxicity_base_pd1: float = 0.05

    @property
    def elimination_rate(self) -> float:
        """First-order elimination rate constant k_e = ln(2) / half_life."""
        return np.log(2) / self.half_life_pd1

    def drug_concentration_update(
        self, current_conc: float, dose_action: int, dt: float
    ) -> float:
        """Update drug concentration after time dt with optional dosing.

        Args:
            current_conc: Current drug concentration (mg/L).
            dose_action: 0=no dose, 1=low dose, 2=high dose.
            dt: Time step in days.

        Returns:
            Updated drug concentration.
        """
        # Add dose as bolus (instantaneous absorption)
        if dose_action == 1:
            current_conc += self.dose_low_pd1 * 70.0 / self.volume_of_distribution
        elif dose_action == 2:
            current_conc += self.dose_high_pd1 * 70.0 / self.volume_of_distribution

        # SIMPLIFICATION: First-order elimination only (no peripheral compartment)
        new_conc = current_conc * np.exp(-self.elimination_rate * dt)
        return float(max(new_conc, 0.0))

    def blockade_efficacy(self, drug_conc: float) -> float:
        """Compute PD-1 blockade efficacy as fraction of immune suppression removed.

        Uses Hill-type occupancy: efficacy = k_pd1 * C / (EC50 + C)

        Args:
            drug_conc: Current anti-PD-1 concentration (mg/L).

        Returns:
            Efficacy in [0, k_pd1], where 0 = no blockade, k_pd1 = max blockade.
        """
        if drug_conc <= 0:
            return 0.0
        return self.k_pd1 * drug_conc / (self.ec50_pd1 + drug_conc)

    def immune_boost_factor(self, drug_conc: float) -> float:
        """Factor by which anti-PD-1 enhances effector cell kill rate.

        Returns 1.0 (no boost) when drug_conc=0, up to 1/(1-k_pd1) at saturation.
        The biological interpretation: PD-L1 on tumor cells suppresses T cell killing
        by a fraction k_pd1. Anti-PD-1 removes this suppression proportionally.

        Args:
            drug_conc: Current drug concentration.

        Returns:
            Multiplicative factor >= 1.0 for effector kill rate.
        """
        efficacy = self.blockade_efficacy(drug_conc)
        # When no drug: factor = 1/(1-0) = 1.0 (baseline suppressed state)
        # When fully blocked: factor = 1/(1-k_pd1) (restored killing)
        # SIMPLIFICATION: Linear interpolation of the suppression removal
        return 1.0 + efficacy

    def toxicity_score(self, drug_conc: float) -> float:
        """Compute irAE toxicity contribution from anti-PD-1.

        Per Shulgin 2020: PD-1 inhibitors show dose-INDEPENDENT toxicity.
        The toxicity score is flat regardless of drug concentration.

        Args:
            drug_conc: Current drug concentration (not used, but kept for API symmetry).

        Returns:
            Toxicity score in [0, 1].
        """
        if drug_conc > 0.1:
            return self.toxicity_base_pd1
        return 0.0

    def validate_parameters(self) -> list[str]:
        """Check parameters against literature ranges."""
        warnings = []
        for name, spec in ANTI_PD1_PARAMETER_RANGES.items():
            val = getattr(self, name)
            lo, hi = spec["range"]
            if val < lo or val > hi:
                warnings.append(
                    f"{name}={val} outside [{lo}, {hi}] ({spec['source']})"
                )
        return warnings


# --- AntiCTLA4Module parameters ---
ANTI_CTLA4_PARAMETER_RANGES: dict[str, dict[str, Any]] = {
    "k_ctla4": {
        "default": 0.6,
        "range": [0.3, 0.9],
        "unit": "dimensionless",
        "description": "Maximum CTLA-4 blockade efficacy",
        "source": "Milberg 2019 estimated from virtual patients",
    },
    "half_life_ctla4": {
        "default": 15.0,
        "range": [8.0, 21.0],
        "unit": "days",
        "description": "Anti-CTLA-4 antibody half-life (ipilimumab)",
        "source": "Milberg 2019 Table S2",
    },
    "ec50_ctla4": {
        "default": 8.0,
        "range": [2.0, 20.0],
        "unit": "mg/L",
        "description": "Half-maximal effective concentration for CTLA-4 blockade",
        "source": "Milberg 2019 estimated",
    },
    "dose_low_ctla4": {
        "default": 1.0,
        "range": [0.3, 3.0],
        "unit": "mg/kg",
        "description": "Low-dose anti-CTLA-4 (ipilimumab 1 mg/kg)",
        "source": "Clinical dosing",
    },
    "dose_high_ctla4": {
        "default": 3.0,
        "range": [1.0, 10.0],
        "unit": "mg/kg",
        "description": "High-dose anti-CTLA-4 (ipilimumab 3 mg/kg)",
        "source": "Clinical dosing",
    },
    "volume_of_distribution_ctla4": {
        "default": 7.2,
        "range": [4.0, 12.0],
        "unit": "L",
        "description": "Central compartment volume for ipilimumab",
        "source": "Milberg 2019",
    },
    "toxicity_slope": {
        "default": 0.015,
        "range": [0.005, 0.05],
        "unit": "1/(mg/L)",
        "description": "Dose-dependent toxicity slope for CTLA-4 inhibitors",
        "source": "Shulgin 2020 Fig 3: CTLA-4 toxicity IS dose-dependent",
    },
    "toxicity_base_ctla4": {
        "default": 0.03,
        "range": [0.01, 0.08],
        "unit": "probability",
        "description": "Baseline irAE probability for anti-CTLA-4",
        "source": "Shulgin 2020",
    },
}


@dataclass
class AntiCTLA4Module:
    """Anti-CTLA-4 checkpoint inhibitor pharmacodynamics.

    Based on Milberg et al. (2019) QSP model parameters for ipilimumab.
    Toxicity model from Shulgin et al. (2020): CTLA-4 inhibitors show
    DOSE-DEPENDENT toxicity, unlike PD-1 inhibitors.

    SIMPLIFICATION: 1-compartment PK as with anti-PD-1 module.

    SIMPLIFICATION: CTLA-4 blockade enhances the immune priming phase
    (increasing effector cell proliferation rate) rather than the killing
    phase. In the simplified model, this is represented as a multiplier on
    the immune stimulation rate sigma, rather than explicit dendritic cell
    and T-reg modeling as in Milberg 2019.
    """

    k_ctla4: float = 0.6
    half_life_ctla4: float = 15.0
    ec50_ctla4: float = 8.0
    dose_low_ctla4: float = 1.0
    dose_high_ctla4: float = 3.0
    volume_of_distribution_ctla4: float = 7.2
    toxicity_slope: float = 0.015
    toxicity_base_ctla4: float = 0.03

    @property
    def elimination_rate(self) -> float:
        """First-order elimination rate constant."""
        return np.log(2) / self.half_life_ctla4

    def drug_concentration_update(
        self, current_conc: float, dose_action: int, dt: float
    ) -> float:
        """Update CTLA-4 drug concentration."""
        if dose_action == 1:
            current_conc += self.dose_low_ctla4 * 70.0 / self.volume_of_distribution_ctla4
        elif dose_action == 2:
            current_conc += self.dose_high_ctla4 * 70.0 / self.volume_of_distribution_ctla4

        new_conc = current_conc * np.exp(-self.elimination_rate * dt)
        return float(max(new_conc, 0.0))

    def blockade_efficacy(self, drug_conc: float) -> float:
        """Compute CTLA-4 blockade efficacy."""
        if drug_conc <= 0:
            return 0.0
        return self.k_ctla4 * drug_conc / (self.ec50_ctla4 + drug_conc)

    def immune_priming_boost(self, drug_conc: float) -> float:
        """Factor by which CTLA-4 blockade enhances immune priming (sigma multiplier).

        CTLA-4 blockade primarily acts by:
        1. Removing Treg-mediated suppression of T cell priming
        2. Enhancing T cell activation in lymph nodes

        This is distinct from PD-1 blockade which acts at the tumor site.
        """
        efficacy = self.blockade_efficacy(drug_conc)
        return 1.0 + 0.5 * efficacy

    def toxicity_score(self, drug_conc: float) -> float:
        """Compute dose-dependent irAE toxicity for anti-CTLA-4.

        Per Shulgin 2020: CTLA-4 inhibitors exhibit dose-dependent toxicity.
        Modeled as logistic: P(irAE) = base + slope * C / (1 + slope * C)

        Args:
            drug_conc: Current drug concentration (mg/L).

        Returns:
            Toxicity score in [0, 1].
        """
        if drug_conc <= 0.1:
            return 0.0
        linear_tox = self.toxicity_base_ctla4 + self.toxicity_slope * drug_conc
        return float(min(linear_tox, 1.0))

    def validate_parameters(self) -> list[str]:
        """Check parameters against literature ranges."""
        warnings = []
        for name, spec in ANTI_CTLA4_PARAMETER_RANGES.items():
            val = getattr(self, name)
            lo, hi = spec["range"]
            if val < lo or val > hi:
                warnings.append(
                    f"{name}={val} outside [{lo}, {hi}] ({spec['source']})"
                )
        return warnings


# --- DualCheckpointModule ---
DUAL_CHECKPOINT_PARAMETER_RANGES: dict[str, dict[str, Any]] = {
    "synergy_coefficient": {
        "default": 0.3,
        "range": [0.1, 0.8],
        "unit": "dimensionless",
        "description": (
            "Synergy interaction strength between PD-1 and CTLA-4 blockade. "
            "Nikolopoulou 2021 shows combination requires ~1/3 dose of each drug."
        ),
        "source": "Nikolopoulou 2021 Fig 4",
    },
    "combination_toxicity_amplifier": {
        "default": 1.5,
        "range": [1.0, 3.0],
        "unit": "dimensionless",
        "description": "Toxicity amplification factor for combination therapy",
        "source": "Shulgin 2020: combination amplifies CTLA-4 dose-toxicity",
    },
}


@dataclass
class DualCheckpointModule:
    """Combined PD-1 + CTLA-4 dual checkpoint blockade.

    Based on Nikolopoulou et al. (2021) synergy model showing sub-additive
    dosing requirements in combination therapy.

    SIMPLIFICATION: Synergy modeled as multiplicative interaction rather
    than mechanistic cytokine feedback loops. The synergy coefficient
    captures the empirical observation that combined blockade at 1/3 dose
    each achieves equivalent tumor control to monotherapy at full dose.
    """

    pd1: AntiPD1Module = None  # type: ignore[assignment]
    ctla4: AntiCTLA4Module = None  # type: ignore[assignment]
    synergy_coefficient: float = 0.3
    combination_toxicity_amplifier: float = 1.5

    def __post_init__(self) -> None:
        if self.pd1 is None:
            self.pd1 = AntiPD1Module()
        if self.ctla4 is None:
            self.ctla4 = AntiCTLA4Module()

    def combined_immune_boost(
        self, pd1_conc: float, ctla4_conc: float
    ) -> float:
        """Compute combined immune enhancement from dual blockade.

        The synergy term means that the combined effect is greater than
        the sum of individual effects, following Nikolopoulou 2021.

        Returns multiplicative factor for effector kill rate.
        """
        pd1_boost = self.pd1.immune_boost_factor(pd1_conc)
        ctla4_boost = self.ctla4.immune_priming_boost(ctla4_conc)

        # Synergy: additional boost proportional to product of individual efficacies
        pd1_eff = self.pd1.blockade_efficacy(pd1_conc)
        ctla4_eff = self.ctla4.blockade_efficacy(ctla4_conc)
        synergy_term = self.synergy_coefficient * pd1_eff * ctla4_eff

        return pd1_boost * ctla4_boost * (1.0 + synergy_term)

    def combined_toxicity(
        self, pd1_conc: float, ctla4_conc: float
    ) -> float:
        """Compute combined toxicity score.

        Per Shulgin 2020: combination therapy amplifies the CTLA-4
        dose-toxicity relationship. PD-1 toxicity remains flat.
        """
        pd1_tox = self.pd1.toxicity_score(pd1_conc)
        ctla4_tox = self.ctla4.toxicity_score(ctla4_conc)

        # Amplification applies to CTLA-4 component only
        combined = pd1_tox + ctla4_tox * self.combination_toxicity_amplifier
        return float(min(combined, 1.0))

    def validate_parameters(self) -> list[str]:
        """Validate all sub-module parameters."""
        warnings = self.pd1.validate_parameters() + self.ctla4.validate_parameters()
        for name, spec in DUAL_CHECKPOINT_PARAMETER_RANGES.items():
            val = getattr(self, name)
            lo, hi = spec["range"]
            if val < lo or val > hi:
                warnings.append(
                    f"{name}={val} outside [{lo}, {hi}] ({spec['source']})"
                )
        return warnings
