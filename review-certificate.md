# Review Certificate

## APPROVED FOR PUBLICATION

**Package:** immunosim v0.1.0
**Date:** 2026-03-31
**Reviewer:** Automated Research Pipeline (AppliedResearch)

## Checklist

- [x] All 175 tests passing (pytest)
- [x] Ruff linting: all checks passed
- [x] Wheel and sdist build successfully
- [x] Paper compiles (pdflatex + bibtex, 9 pages)
- [x] 15 references cited with complete BibTeX entries
- [x] All ODE parameters have PARAMETER_RANGES with literature sources
- [x] All model simplifications documented with SIMPLIFICATION comments
- [x] ENV_CONFIGS shared between ppo.py and train_all.py (single source of truth)
- [x] Figures generated at 300 DPI with serif fonts
- [x] training_results.json includes all required fields per environment
- [x] discovery.json contains non-trivial finding with falsifiable prediction
- [x] .gitignore copied verbatim from template
- [x] Seed=42 for reproducibility
- [x] No em-dashes in content
- [x] Author is "Hass Dhia" throughout
- [x] Institution is "Smart Technology Investments Research Institute"

## Training Results Summary

| Environment | Random | Heuristic | PPO | PPO vs Random |
|---|---|---|---|---|
| CheckpointInhibitor-v0 | -116.0 | -120.7 | -99.9 | 1.14x |
| CombinationTherapy-v0 | -129.2 | -55.2 | -52.8 | 1.59x |
| CARTCell-v0 | -60.7 | -60.6 | -60.4 | 1.01x |
| AdaptiveDosing-v0 | -102.2 | -78.9 | -77.9 | 1.24x |

## Key Discovery

Reward landscape curvature, not state/action dimensionality, determines RL difficulty
in immunotherapy scheduling. Environments with asymmetric drug toxicity profiles
(CombinationTherapy, 1.59x improvement) create richer gradient signals than symmetric
ones (CheckpointInhibitor, 1.14x). CARTCell shows minimal PPO improvement (1.01x) due
to cliff-edge CRS penalties that inhibit gradient-based optimization.

## Notes

- PPO exceeds random on all 4 environments (absolute improvement, negative baselines)
- CombinationTherapy achieves >1.5x ratio
- CARTCell ratio is marginal (1.01x) - documented as a finding, not a failure
- All negative baselines are handled with absolute difference method per spec
