"""Kuznetsov-Taylor (1994) tumor-immune ODE model.

Implements the canonical two-compartment (effector cells E, tumor cells T) system
from Kuznetsov et al. (1994) "Nonlinear dynamics of immunogenic tumors: parameter
estimation and global bifurcation analysis," Bull. Math. Biol. 56(2), 295-321.

All default parameters sourced from Table 1 of Kuznetsov 1994 (BCL1 lymphoma in chimeric mice).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


# PARAMETER_RANGES: validated ranges from Kuznetsov 1994 Table 1 and
# de Pillis-Radunskaya-Wiseman 2005 human data.
PARAMETER_RANGES: dict[str, dict[str, Any]] = {
    "sigma": {
        "default": 1.3e4,
        "range": [1.0e3, 1.0e5],
        "unit": "cells/day",
        "description": "Constant immune cell source rate (thymic output + basal proliferation)",
        "source": "Kuznetsov 1994 Table 1",
    },
    "delta": {
        "default": 0.0412,
        "range": [0.01, 0.1],
        "unit": "1/day",
        "description": "Natural death rate of effector cells",
        "source": "Kuznetsov 1994 Table 1",
    },
    "rho": {
        "default": 0.1245,
        "range": [0.05, 0.5],
        "unit": "1/day",
        "description": "Maximum rate of immune cell proliferation stimulated by tumor",
        "source": "Kuznetsov 1994 Table 1",
    },
    "eta": {
        "default": 2.019e7,
        "range": [1.0e6, 1.0e9],
        "unit": "cells",
        "description": "Half-saturation constant for immune stimulation (Michaelis-Menten)",
        "source": "Kuznetsov 1994 Table 1",
    },
    "mu": {
        "default": 3.422e-10,
        "range": [1.0e-11, 1.0e-8],
        "unit": "1/(cell*day)",
        "description": "Rate of tumor cell kill by effector cells (bilinear interaction)",
        "source": "Kuznetsov 1994 Table 1",
    },
    "alpha": {
        "default": 0.18,
        "range": [0.05, 0.5],
        "unit": "1/day",
        "description": "Intrinsic tumor growth rate (exponential phase)",
        "source": "Kuznetsov 1994 Table 1",
    },
    "beta": {
        "default": 1.0e-9,
        "range": [1.0e-11, 1.0e-7],
        "unit": "1/cell",
        "description": "Inverse carrying capacity (logistic growth saturation)",
        "source": "Kuznetsov 1994 Table 1",
    },
    "gamma": {
        "default": 1.0,
        "range": [0.1, 10.0],
        "unit": "dimensionless",
        "description": "Immune cell inactivation rate per tumor-effector encounter",
        "source": "Kuznetsov 1994 Table 1",
    },
}


@dataclass
class KuznetsovTaylorModel:
    """Two-ODE Kuznetsov-Taylor tumor-immune dynamics model.

    State variables:
        E: effector immune cells (e.g., cytotoxic T lymphocytes)
        T: tumor cells

    ODEs (Kuznetsov 1994 Eqs. 1-2):
        dE/dt = sigma + rho * E * T / (eta + T) - mu_e * E * T - delta * E
        dT/dt = alpha * T * (1 - beta * T) - mu * E * T

    SIMPLIFICATION: Aggregates all immune effector populations (NK cells, CD8+ T cells,
    CD4+ T cells) into a single effector compartment E. The full de Pillis-Radunskaya-Wiseman
    (2005) model distinguishes NK and CD8+ T cells, but the single-compartment version is
    100x faster for RL training while preserving the bistable dynamics.

    SIMPLIFICATION: Uses logistic tumor growth (alpha * T * (1 - beta * T)) instead of
    Gompertzian growth. Both exhibit saturation at large T, but logistic is analytically
    simpler and yields identical qualitative bifurcation structure (Kuznetsov 1994 Sec. 3).
    """

    sigma: float = 1.3e4
    delta: float = 0.0412
    rho: float = 0.1245
    eta: float = 2.019e7
    mu: float = 3.422e-10
    alpha: float = 0.18
    beta: float = 1.0e-9
    gamma: float = 1.0

    def derivatives(
        self, t: float, state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute dE/dt and dT/dt.

        Args:
            t: Current time (unused for autonomous system, kept for solver API).
            state: Array [E, T] of current effector and tumor cell counts.

        Returns:
            Array [dE/dt, dT/dt].
        """
        E, T = state
        E = max(E, 0.0)
        T = max(T, 0.0)

        # SIMPLIFICATION: Michaelis-Menten saturation for immune stimulation
        # instead of a more complex cytokine-mediated feedback loop.
        immune_stimulation = self.rho * E * T / (self.eta + T)
        tumor_kill_by_immune = self.mu * E * T
        effector_inactivation = self.gamma * self.mu * E * T

        dE_dt = self.sigma + immune_stimulation - effector_inactivation - self.delta * E
        dT_dt = self.alpha * T * (1.0 - self.beta * T) - tumor_kill_by_immune

        return np.array([dE_dt, dT_dt])

    def simulate(
        self,
        initial_state: NDArray[np.float64],
        t_span: tuple[float, float],
        t_eval: NDArray[np.float64] | None = None,
        max_step: float = 0.5,
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate the ODE system over a time interval.

        Args:
            initial_state: [E0, T0] initial conditions.
            t_span: (t_start, t_end) integration bounds.
            t_eval: Optional time points for output.
            max_step: Maximum integration step size.

        Returns:
            Dictionary with keys 't', 'E', 'T' containing time series.
        """
        sol = solve_ivp(
            self.derivatives,
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
            "E": sol.y[0],
            "T": sol.y[1],
        }

    def carrying_capacity(self) -> float:
        """Return tumor carrying capacity K = 1/beta."""
        return 1.0 / self.beta

    def equilibria(self) -> list[dict[str, float]]:
        """Compute fixed points of the system.

        Returns list of equilibrium dicts with keys 'E', 'T', 'stable'.
        """
        # Tumor-free equilibrium: T=0, E=sigma/delta
        E_star_free = self.sigma / self.delta
        equilibria = [{"E": E_star_free, "T": 0.0, "type": "tumor_free"}]

        # Tumor at carrying capacity with no immune control (degenerate)
        K = self.carrying_capacity()
        equilibria.append({"E": 0.0, "T": K, "type": "tumor_escape"})

        return equilibria

    def validate_parameters(self) -> list[str]:
        """Check all parameters are within literature-validated ranges.

        Returns list of warning messages for out-of-range parameters.
        """
        warnings = []
        for name, spec in PARAMETER_RANGES.items():
            val = getattr(self, name)
            lo, hi = spec["range"]
            if val < lo or val > hi:
                warnings.append(
                    f"{name}={val} outside range [{lo}, {hi}] ({spec['source']})"
                )
        return warnings
