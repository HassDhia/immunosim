"""Tests for patient generator and pseudo-progression detector."""

import numpy as np

from immunosim.models.patient import (
    PatientGenerator,
    PseudoProgressionDetector,
    PATIENT_PARAMETER_RANGES,
    PSEUDO_PROGRESSION_RANGES,
)


class TestPatientGenerator:
    """Tests for patient parameter randomization."""

    def test_generate_single_patient(self):
        gen = PatientGenerator(rng=np.random.default_rng(42))
        patients = gen.generate(1)
        assert len(patients) == 1
        patient = patients[0]
        assert "alpha" in patient
        assert "lambda_immune" in patient
        assert "mu_amplification" in patient
        assert "initial_tumor_volume" in patient

    def test_generate_multiple_patients(self):
        gen = PatientGenerator(rng=np.random.default_rng(42))
        patients = gen.generate(10)
        assert len(patients) == 10

    def test_parameters_within_ranges(self):
        gen = PatientGenerator(rng=np.random.default_rng(42))
        patients = gen.generate(100)
        for p in patients:
            lo, hi = PATIENT_PARAMETER_RANGES["alpha_patient"]["range"]
            assert lo <= p["alpha"] <= hi
            lo, hi = PATIENT_PARAMETER_RANGES["lambda_immune"]["range"]
            assert lo <= p["lambda_immune"] <= hi

    def test_reproducibility(self):
        gen1 = PatientGenerator(rng=np.random.default_rng(42))
        gen2 = PatientGenerator(rng=np.random.default_rng(42))
        p1 = gen1.generate(5)
        p2 = gen2.generate(5)
        for a, b in zip(p1, p2):
            assert a["alpha"] == b["alpha"]

    def test_generate_cohort(self):
        gen = PatientGenerator(rng=np.random.default_rng(42))
        cohort = gen.generate_cohort(100, responder_fraction=0.3)
        assert len(cohort) == 100
        responders = [p for p in cohort if p.get("is_responder")]
        assert len(responders) == 30

    def test_responders_have_higher_immune(self):
        gen = PatientGenerator(rng=np.random.default_rng(42))
        cohort = gen.generate_cohort(100, responder_fraction=0.5)
        resp = [p for p in cohort if p["is_responder"]]
        non_resp = [p for p in cohort if not p["is_responder"]]
        avg_resp_lambda = np.mean([p["lambda_immune"] for p in resp])
        avg_nonresp_lambda = np.mean([p["lambda_immune"] for p in non_resp])
        assert avg_resp_lambda > avg_nonresp_lambda

    def test_parameter_ranges_defined(self):
        for key in PATIENT_PARAMETER_RANGES:
            spec = PATIENT_PARAMETER_RANGES[key]
            assert "source" in spec
            assert "range" in spec


class TestPseudoProgressionDetector:
    """Tests for pseudo-progression detection."""

    def test_default_parameters(self):
        det = PseudoProgressionDetector()
        assert det.immune_infiltration_delay == 14.0
        assert det.max_pseudo_growth_factor == 1.3

    def test_parameter_ranges_defined(self):
        for key in PSEUDO_PROGRESSION_RANGES:
            spec = PSEUDO_PROGRESSION_RANGES[key]
            assert "source" in spec

    def test_not_pseudo_early(self):
        """Too early on treatment should not be pseudo-progression."""
        det = PseudoProgressionDetector()
        tumor = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        immune = np.array([1000.0, 1100.0, 1200.0, 1300.0, 1400.0])
        assert not det.is_pseudo_progression(tumor, immune, 3.0, 0.0)

    def test_not_pseudo_too_much_growth(self):
        """Growth exceeding threshold is true progression."""
        det = PseudoProgressionDetector()
        tumor = np.array([100.0, 130.0, 160.0, 200.0, 250.0])
        immune = np.array([1000.0, 1100.0, 1200.0, 1300.0, 1400.0])
        assert not det.is_pseudo_progression(tumor, immune, 30.0, 0.0)

    def test_pseudo_progression_detected(self):
        """Moderate growth with immune increase = pseudo-progression."""
        det = PseudoProgressionDetector()
        # Tumor grows ~20% (within 1.3x threshold)
        tumor = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        immune = np.array([1000.0, 1100.0, 1300.0, 1500.0, 1800.0])
        result = det.is_pseudo_progression(tumor, immune, 20.0, 0.0)
        assert result is True

    def test_not_pseudo_without_immune_increase(self):
        """Growth without immune increase is not pseudo-progression."""
        det = PseudoProgressionDetector()
        tumor = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        immune = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        assert not det.is_pseudo_progression(tumor, immune, 20.0, 0.0)

    def test_response_trajectory_declining(self):
        """Declining tumor gives negative trajectory."""
        det = PseudoProgressionDetector()
        tumor = np.array([1000.0, 800.0, 600.0, 400.0, 200.0])
        traj = det.response_trajectory(tumor)
        assert traj < 0

    def test_response_trajectory_growing(self):
        """Growing tumor gives positive trajectory."""
        det = PseudoProgressionDetector()
        tumor = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
        traj = det.response_trajectory(tumor)
        assert traj > 0

    def test_response_trajectory_stable(self):
        """Stable tumor gives near-zero trajectory."""
        det = PseudoProgressionDetector()
        tumor = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        traj = det.response_trajectory(tumor)
        assert abs(traj) < 0.01

    def test_response_trajectory_short_history(self):
        """Short history returns 0."""
        det = PseudoProgressionDetector()
        assert det.response_trajectory(np.array([100.0])) == 0.0
        assert det.response_trajectory(np.array([100.0, 200.0])) == 0.0
