# immunosim

**Gymnasium environments for reinforcement learning in cancer immunotherapy optimization**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests](https://img.shields.io/badge/tests-175%20passing-brightgreen.svg)
[![PyPI version](https://img.shields.io/pypi/v/immunosim.svg)](https://pypi.org/project/immunosim/)

---

ImmunoSim provides four Gymnasium-compatible reinforcement learning environments for
optimizing cancer immunotherapy treatment schedules. Each environment wraps validated
mathematical models from the tumor immunology literature, enabling RL agents to discover
dosing strategies that outperform standard clinical protocols.

The package implements ODE-based tumor-immune dynamics from Kuznetsov-Taylor (1994),
checkpoint inhibitor pharmacodynamics from Nikolopoulou (2018, 2021), CAR-T cell kinetics
from Barros CARTmath (2021), and cytokine release syndrome modeling from Santurio (2025).

## Installation

```bash
pip install immunosim              # Core (numpy, scipy, gymnasium)
pip install immunosim[train]       # + SB3, PyTorch for RL training
pip install immunosim[all]         # Everything
```

Development install:

```bash
git clone https://github.com/HassDhia/immunosim.git
cd immunosim
pip install -e ".[all]"
```

## Quick Start

```python
import gymnasium as gym
import immunosim

env = gym.make("immunosim/CheckpointInhibitor-v0")
obs, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

## Environments

| Environment | Paradigm | Observation | Action | Key Challenge |
|---|---|---|---|---|
| `CheckpointInhibitor-v0` | Anti-PD-1 monotherapy | Box(5) | Discrete(3) | Balance tumor control vs. cumulative drug exposure |
| `CombinationTherapy-v0` | Dual PD-1 + CTLA-4 | Box(7) | MultiDiscrete([3,3]) | Exploit synergy while managing CTLA-4 dose-dependent toxicity |
| `CARTCell-v0` | CAR-T cell infusion | Box(6) | Discrete(4) | Maximize tumor kill while preventing grade 4 CRS |
| `AdaptiveDosing-v0` | Adaptive response-based | Box(6) | Discrete(4) | Distinguish pseudo-progression from true progression |

## Architecture

```
immunosim/
  models/
    tumor_immune.py      -- Kuznetsov-Taylor (1994) 2-ODE base model
    checkpoint_inhibitor.py -- Anti-PD-1, Anti-CTLA-4, Dual checkpoint modules
    cart_cell.py         -- CARTmath (Barros 2021) + CRS (Santurio 2025)
    patient.py           -- Domain randomization + pseudo-progression detection
  envs/
    checkpoint_inhibitor.py  -- CheckpointInhibitorEnv-v0
    combination_therapy.py   -- CombinationTherapyEnv-v0
    cart_cell.py             -- CARTCellEnv-v0
    adaptive_dosing.py       -- AdaptiveDosingEnv-v0
  agents/
    random_agent.py      -- Random baseline
    heuristic_agent.py   -- Clinical protocol heuristics
    ppo.py               -- PPO via stable-baselines3
  training/
    configs.py           -- Shared ENV_CONFIGS
    train_all.py         -- Train all environments
  benchmarks/
    runner.py            -- Performance benchmarking
```

## Paper

The accompanying paper is available at:
- [PDF (GitHub)](https://github.com/HassDhia/immunosim/blob/main/paper/immunosim.pdf)

## Citation

If you use immunosim in your research, please cite:

```bibtex
@software{dhia2026immunosim,
  author = {Dhia, Hass},
  title = {ImmunoSim: Gymnasium Environments for Reinforcement Learning in Cancer Immunotherapy Optimization},
  year = {2026},
  publisher = {Smart Technology Investments Research Institute},
  url = {https://github.com/HassDhia/immunosim}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Hass Dhia -- Smart Technology Investments Research Institute
- Email: partners@smarttechinvest.com
- Web: [smarttechinvest.com/research](https://smarttechinvest.com/research)
