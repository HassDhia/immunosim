"""Tests for CAR-T cell therapy models."""

import numpy as np

from immunosim.models.cart_cell import (
    CARTmathModel,
    CRSToxicityModel,
    CARTMATH_PARAMETER_RANGES,
    CRS_PARAMETER_RANGES,
)


class TestCARTmathModel:
    """Tests for the 4-compartment CAR-T ODE model."""

    def test_default_parameters(self):
        model = CARTmathModel()
        assert model.rho_t == 0.3
        assert model.K_t == 1.0e9
        assert model.k_act == 0.5

    def test_parameter_ranges_exist(self):
        for key in ["rho_t", "K_t", "k_act", "gamma_e", "rho_e", "K_e",
                     "delta_e", "k_mem", "delta_m", "k_reactivate", "k_suppress", "delta_i"]:
            assert key in CARTMATH_PARAMETER_RANGES

    def test_defaults_in_range(self):
        assert len(CARTmathModel().validate_parameters()) == 0

    def test_derivatives_shape(self):
        model = CARTmathModel()
        state = np.array([1e6, 1e5, 1e4, 1e7])
        deriv = model.derivatives(0.0, state)
        assert deriv.shape == (4,)

    def test_injected_clearance(self):
        """Injected CAR-T cells should clear without infusion."""
        model = CARTmathModel()
        state = np.array([1e6, 0.0, 0.0, 0.0])  # Only injected, no tumor
        deriv = model.derivatives(0.0, state)
        assert deriv[0] < 0  # I should decrease

    def test_tumor_growth_without_cart(self):
        """Tumor grows without CAR-T cells."""
        model = CARTmathModel()
        state = np.array([0.0, 0.0, 0.0, 1e6])
        deriv = model.derivatives(0.0, state)
        assert deriv[3] > 0  # T grows

    def test_simulate_returns_correct_keys(self):
        model = CARTmathModel()
        result = model.simulate(np.array([1e6, 0.0, 0.0, 1e7]), (0.0, 10.0))
        for key in ["t", "I", "E", "M", "T"]:
            assert key in result

    def test_activation_converts_injected_to_effector(self):
        """Injected CAR-T should activate into effectors."""
        model = CARTmathModel()
        state = np.array([1e7, 0.0, 0.0, 1e6])
        deriv = model.derivatives(0.0, state)
        # dE/dt should be positive (activation from I)
        assert deriv[1] > 0

    def test_memory_differentiation(self):
        """Effectors should differentiate into memory cells."""
        model = CARTmathModel()
        state = np.array([0.0, 1e6, 0.0, 1e3])
        deriv = model.derivatives(0.0, state)
        # dM/dt should be positive (from effector differentiation)
        assert deriv[2] > 0

    def test_infusion_rate(self):
        """External infusion increases injected compartment."""
        model = CARTmathModel()
        state = np.array([0.0, 0.0, 0.0, 1e6])
        deriv = model.derivatives(0.0, state, infusion_rate=1e6)
        assert deriv[0] > 0

    def test_tumor_kill(self):
        """Effector CAR-T cells should kill tumor."""
        model = CARTmathModel()
        # High effector, moderate tumor
        state = np.array([0.0, 1e8, 0.0, 1e6])
        deriv = model.derivatives(0.0, state)
        assert deriv[3] < 0  # Tumor decreasing

    def test_non_negative_handling(self):
        """Model handles negative values gracefully."""
        model = CARTmathModel()
        state = np.array([-1.0, -1.0, -1.0, -1.0])
        deriv = model.derivatives(0.0, state)
        assert np.isfinite(deriv).all()


class TestCRSToxicityModel:
    """Tests for cytokine release syndrome model."""

    def test_default_parameters(self):
        model = CRSToxicityModel()
        assert model.cytokine_production_rate == 1.0e-6
        assert model.cytokine_clearance == 0.5

    def test_parameter_ranges_exist(self):
        for key in CRS_PARAMETER_RANGES:
            assert "source" in CRS_PARAMETER_RANGES[key]

    def test_defaults_in_range(self):
        assert len(CRSToxicityModel().validate_parameters()) == 0

    def test_crs_grade_0(self):
        model = CRSToxicityModel()
        assert model.crs_grade(0.0) == 0
        assert model.crs_grade(10.0) == 0

    def test_crs_grade_1(self):
        model = CRSToxicityModel()
        assert model.crs_grade(50.0) == 1
        assert model.crs_grade(100.0) == 1

    def test_crs_grade_2(self):
        model = CRSToxicityModel()
        assert model.crs_grade(200.0) == 2

    def test_crs_grade_3(self):
        model = CRSToxicityModel()
        assert model.crs_grade(500.0) == 3

    def test_crs_grade_4(self):
        model = CRSToxicityModel()
        assert model.crs_grade(1000.0) == 4
        assert model.crs_grade(5000.0) == 4

    def test_toxicity_penalty_monotonic(self):
        """Penalty increases with cytokine level."""
        model = CRSToxicityModel()
        penalties = [model.toxicity_penalty(x) for x in [0, 50, 200, 500, 1000]]
        assert penalties == sorted(penalties)

    def test_toxicity_penalty_grade4_is_one(self):
        model = CRSToxicityModel()
        assert model.toxicity_penalty(2000.0) == 1.0

    def test_cytokine_update_increases_with_activity(self):
        model = CRSToxicityModel()
        level = model.update_cytokine_level(0.0, 1e6, 1e6, 1.0)
        assert level > 0

    def test_cytokine_clearance(self):
        """Cytokine decays when production stops."""
        model = CRSToxicityModel()
        level = model.update_cytokine_level(100.0, 0.0, 0.0, 1.0)
        assert level < 100.0

    def test_cytokine_non_negative(self):
        model = CRSToxicityModel()
        level = model.update_cytokine_level(0.0, 0.0, 0.0, 1.0)
        assert level >= 0.0
