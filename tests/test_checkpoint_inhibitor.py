"""Tests for checkpoint inhibitor pharmacodynamic models."""

import numpy as np
import pytest

from immunosim.models.checkpoint_inhibitor import (
    AntiPD1Module,
    AntiCTLA4Module,
    DualCheckpointModule,
    ANTI_PD1_PARAMETER_RANGES,
)


class TestAntiPD1Module:
    """Tests for anti-PD-1 pharmacodynamics."""

    def test_default_parameters(self):
        mod = AntiPD1Module()
        assert mod.k_pd1 == 0.8
        assert mod.half_life_pd1 == 25.0

    def test_parameter_ranges_exist(self):
        for key in ["k_pd1", "half_life_pd1", "ec50_pd1", "dose_low_pd1", "dose_high_pd1"]:
            assert key in ANTI_PD1_PARAMETER_RANGES

    def test_defaults_in_range(self):
        assert len(AntiPD1Module().validate_parameters()) == 0

    def test_elimination_rate(self):
        mod = AntiPD1Module()
        assert mod.elimination_rate == pytest.approx(np.log(2) / 25.0)

    def test_no_dose_no_change(self):
        mod = AntiPD1Module()
        conc = mod.drug_concentration_update(10.0, 0, 0.0)
        assert conc == pytest.approx(10.0)

    def test_low_dose_increases_concentration(self):
        mod = AntiPD1Module()
        conc = mod.drug_concentration_update(0.0, 1, 0.0)
        assert conc > 0

    def test_high_dose_more_than_low(self):
        mod = AntiPD1Module()
        conc_low = mod.drug_concentration_update(0.0, 1, 0.0)
        conc_high = mod.drug_concentration_update(0.0, 2, 0.0)
        assert conc_high > conc_low

    def test_concentration_decays(self):
        mod = AntiPD1Module()
        conc = mod.drug_concentration_update(10.0, 0, 7.0)
        assert conc < 10.0
        assert conc > 0

    def test_blockade_efficacy_zero_at_zero_conc(self):
        mod = AntiPD1Module()
        assert mod.blockade_efficacy(0.0) == 0.0

    def test_blockade_efficacy_bounded(self):
        mod = AntiPD1Module()
        eff = mod.blockade_efficacy(1000.0)
        assert 0 < eff <= mod.k_pd1

    def test_blockade_efficacy_hill_function(self):
        """Efficacy at EC50 should be k_pd1/2."""
        mod = AntiPD1Module()
        eff = mod.blockade_efficacy(mod.ec50_pd1)
        assert eff == pytest.approx(mod.k_pd1 / 2.0)

    def test_immune_boost_no_drug(self):
        mod = AntiPD1Module()
        assert mod.immune_boost_factor(0.0) == 1.0

    def test_immune_boost_increases_with_drug(self):
        mod = AntiPD1Module()
        boost = mod.immune_boost_factor(10.0)
        assert boost > 1.0

    def test_toxicity_flat_per_shulgin_2020(self):
        """PD-1 toxicity is dose-independent (Shulgin 2020)."""
        mod = AntiPD1Module()
        tox_low = mod.toxicity_score(5.0)
        tox_high = mod.toxicity_score(50.0)
        assert tox_low == tox_high  # Flat!

    def test_toxicity_zero_no_drug(self):
        mod = AntiPD1Module()
        assert mod.toxicity_score(0.0) == 0.0


class TestAntiCTLA4Module:
    """Tests for anti-CTLA-4 pharmacodynamics."""

    def test_default_parameters(self):
        mod = AntiCTLA4Module()
        assert mod.k_ctla4 == 0.6
        assert mod.half_life_ctla4 == 15.0

    def test_defaults_in_range(self):
        assert len(AntiCTLA4Module().validate_parameters()) == 0

    def test_concentration_update(self):
        mod = AntiCTLA4Module()
        conc = mod.drug_concentration_update(0.0, 2, 0.0)
        assert conc > 0

    def test_toxicity_dose_dependent_per_shulgin_2020(self):
        """CTLA-4 toxicity IS dose-dependent (Shulgin 2020)."""
        mod = AntiCTLA4Module()
        tox_low = mod.toxicity_score(5.0)
        tox_high = mod.toxicity_score(50.0)
        assert tox_high > tox_low  # Dose-dependent!

    def test_toxicity_bounded(self):
        mod = AntiCTLA4Module()
        tox = mod.toxicity_score(10000.0)
        assert tox <= 1.0

    def test_immune_priming_boost(self):
        mod = AntiCTLA4Module()
        boost = mod.immune_priming_boost(10.0)
        assert boost > 1.0

    def test_immune_priming_no_drug(self):
        mod = AntiCTLA4Module()
        assert mod.immune_priming_boost(0.0) == 1.0

    def test_blockade_efficacy_at_ec50(self):
        mod = AntiCTLA4Module()
        eff = mod.blockade_efficacy(mod.ec50_ctla4)
        assert eff == pytest.approx(mod.k_ctla4 / 2.0)


class TestDualCheckpointModule:
    """Tests for combined PD-1 + CTLA-4 module."""

    def test_default_init(self):
        dual = DualCheckpointModule()
        assert dual.pd1 is not None
        assert dual.ctla4 is not None

    def test_synergy_coefficient_in_range(self):
        dual = DualCheckpointModule()
        assert len(dual.validate_parameters()) == 0

    def test_combined_boost_exceeds_individual(self):
        """Synergy: combined > sum of individual effects."""
        dual = DualCheckpointModule()
        boost_pd1_only = dual.combined_immune_boost(10.0, 0.0)
        boost_ctla4_only = dual.combined_immune_boost(0.0, 10.0)
        boost_combined = dual.combined_immune_boost(10.0, 10.0)
        # Combined should be greater due to synergy term
        assert boost_combined > max(boost_pd1_only, boost_ctla4_only)

    def test_combined_toxicity(self):
        dual = DualCheckpointModule()
        tox = dual.combined_toxicity(10.0, 10.0)
        assert tox > 0

    def test_combined_toxicity_bounded(self):
        dual = DualCheckpointModule()
        tox = dual.combined_toxicity(1000.0, 1000.0)
        assert tox <= 1.0

    def test_combined_toxicity_amplifies_ctla4(self):
        """Per Shulgin 2020: combination amplifies CTLA-4 dose-toxicity."""
        dual = DualCheckpointModule()
        ctla4_solo = dual.ctla4.toxicity_score(10.0)
        combined = dual.combined_toxicity(10.0, 10.0)
        pd1_solo = dual.pd1.toxicity_score(10.0)
        # Combined toxicity > sum of individual (amplification)
        assert combined >= pd1_solo + ctla4_solo

    def test_no_boost_without_drugs(self):
        dual = DualCheckpointModule()
        assert dual.combined_immune_boost(0.0, 0.0) == pytest.approx(1.0)
