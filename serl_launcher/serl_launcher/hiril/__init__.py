"""HiRIL — external comparison baseline (NOT part of AQI).

Y. Zhou, Y.-B. Zhao, C. Xu, C. Ouyang, P. Li, "Human-in-the-loop
reinforcement learning with risk-aware intervention and imitation",
Expert Systems With Applications 308 (2026) 131118.

Layout:
    config.py      HiRILConfig: every hyperparameter, paper value or marked
                   adaptation (no JAX; importable anywhere)
    priorities.py  RiskAwarePriorities: TDRD replay priorities (Eq. 26-28)
                   in numpy (no JAX)
    agent.py       HiRILAgent: TD3-style deterministic actor + BIQN critic
                   (Bayesian implicit quantile network, Sec. 4), dual-mode
                   uncertainty DU(s, a) (Eq. 16-20), critic / actor losses
                   (Eq. 29, 32). Imports JAX -- import it explicitly.

Pipeline: examples/train_hiril.py.  Doc: docs/hiril_baseline.md.
"""

from serl_launcher.hiril.config import HiRILConfig  # noqa: F401
from serl_launcher.hiril.priorities import RiskAwarePriorities  # noqa: F401
