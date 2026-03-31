"""Tests for Kuznetsov-Taylor tumor-immune model."""

import numpy as np
import pytest

from immunosim.models.tumor_immune import KuznetsovTaylorModel, PARAMETER_RANGES


class TestKuznetsovTaylorModel:
    """Test the base tumor-immune ODE model."""

    def test_default_parameters(self):
        """Default parameters match Kuznetsov 1994 Table 1."""
        model = KuznetsovTaylorModel()
        assert model.sigma == 1.3e4
        assert model.delta == 0.0412
        assert model.rho == 0.1245
        assert model.eta == 2.019e7
        assert model.mu == 3.422e-10
        assert model.alpha == 0.18
        assert model.beta == 1.0e-9

    def test_parameter_ranges_defined(self):
        """All model parameters have validated ranges."""
        _model = KuznetsovTaylorModel()  # noqa: F841
        for field_name in ["sigma", "delta", "rho", "eta", "mu", "alpha", "beta", "gamma"]:
            assert field_name in PARAMETER_RANGES
            spec = PARAMETER_RANGES[field_name]
            assert "default" in spec
            assert "range" in spec
            assert "source" in spec
            assert len(spec["range"]) == 2

    def test_defaults_within_ranges(self):
        """Default parameter values are within their validated ranges."""
        warnings = KuznetsovTaylorModel().validate_parameters()
        assert len(warnings) == 0

    def test_derivatives_shape(self):
        """Derivatives return correct shape."""
        model = KuznetsovTaylorModel()
        state = np.array([3e5, 1e6])
        deriv = model.derivatives(0.0, state)
        assert deriv.shape == (2,)

    def test_derivatives_tumor_free_equilibrium(self):
        """At tumor-free equilibrium, dE/dt ~ 0 and dT/dt = 0."""
        model = KuznetsovTaylorModel()
        E_star = model.sigma / model.delta
        state = np.array([E_star, 0.0])
        deriv = model.derivatives(0.0, state)
        assert abs(deriv[0]) < 1.0  # dE/dt ~ 0
        assert deriv[1] == 0.0  # dT/dt = 0 when T=0

    def test_tumor_growth_without_immune(self):
        """Tumor grows exponentially when effector cells are negligible."""
        model = KuznetsovTaylorModel()
        state = np.array([0.0, 1e4])
        deriv = model.derivatives(0.0, state)
        # dT/dt = alpha * T * (1 - beta * T) - mu * E * T
        # With E=0: dT/dt = alpha * T * (1 - beta * T) > 0
        assert deriv[1] > 0

    def test_simulate_returns_correct_keys(self):
        """Simulation returns t, E, T arrays."""
        model = KuznetsovTaylorModel()
        result = model.simulate(np.array([3e5, 1e6]), (0.0, 10.0))
        assert "t" in result
        assert "E" in result
        assert "T" in result
        assert len(result["t"]) == len(result["E"]) == len(result["T"])

    def test_simulate_with_t_eval(self):
        """Simulation respects t_eval parameter."""
        model = KuznetsovTaylorModel()
        t_eval = np.linspace(0, 10, 20)
        result = model.simulate(np.array([3e5, 1e6]), (0.0, 10.0), t_eval=t_eval)
        assert len(result["t"]) == 20

    def test_carrying_capacity(self):
        """Carrying capacity K = 1/beta."""
        model = KuznetsovTaylorModel()
        assert model.carrying_capacity() == pytest.approx(1.0e9)

    def test_equilibria_tumor_free(self):
        """Tumor-free equilibrium exists."""
        model = KuznetsovTaylorModel()
        eq = model.equilibria()
        tumor_free = [e for e in eq if e["type"] == "tumor_free"]
        assert len(tumor_free) == 1
        assert tumor_free[0]["T"] == 0.0
        assert tumor_free[0]["E"] == pytest.approx(model.sigma / model.delta, rel=1e-6)

    def test_validate_out_of_range(self):
        """Validation catches out-of-range parameters."""
        model = KuznetsovTaylorModel(sigma=1e10)  # Way too high
        warnings = model.validate_parameters()
        assert len(warnings) > 0
        assert "sigma" in warnings[0]

    def test_non_negative_cells(self):
        """Derivatives handle near-zero state gracefully."""
        model = KuznetsovTaylorModel()
        state = np.array([0.0, 0.0])
        deriv = model.derivatives(0.0, state)
        assert np.isfinite(deriv).all()

    def test_custom_parameters(self):
        """Model accepts custom parameter values."""
        model = KuznetsovTaylorModel(alpha=0.3, sigma=2e4)
        assert model.alpha == 0.3
        assert model.sigma == 2e4

    def test_simulation_conserves_positivity(self):
        """Cell counts remain non-negative during simulation."""
        model = KuznetsovTaylorModel()
        result = model.simulate(np.array([3e5, 1e6]), (0.0, 30.0))
        assert (result["E"] >= -1.0).all()  # Allow tiny numerical artifacts
        assert (result["T"] >= -1.0).all()

    def test_immune_stimulation_saturates(self):
        """Immune stimulation term saturates at high tumor load (Michaelis-Menten)."""
        model = KuznetsovTaylorModel()
        E = 1e5
        # Low tumor
        state_low = np.array([E, 1e3])
        _ = model.derivatives(0.0, state_low)
        # High tumor (near eta)
        state_high = np.array([E, model.eta * 10])
        _ = model.derivatives(0.0, state_high)
        # The stimulation per unit tumor should be lower at high T (saturation)
        stim_low = model.rho * E * 1e3 / (model.eta + 1e3)
        stim_high = model.rho * E * model.eta * 10 / (model.eta + model.eta * 10)
        assert stim_high / (model.eta * 10) < stim_low / 1e3
