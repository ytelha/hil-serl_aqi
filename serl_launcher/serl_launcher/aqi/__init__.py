"""AQI-HIL-SERL: Active and Quality-Aware Intervention HIL-SERL.

Implements the method of the research proposal "Active and Quality-Aware
Human-in-the-Loop Reinforcement Learning for Operator-Robust Robotic
Insertion" on top of the rail-berkeley HIL-SERL codebase.

Package layout (each file maps to a proposal section):
    config.py        AQIConfig, run modes and every hyperparameter (S5, S6.4)
    uncertainty.py   critic-ensemble mu/sigma, LCB, counterfactual advantage
                     (S5.3, S5.5 Option B)
    force_risk.py    R_force, hard force/torque limits, action jerk (S5.4)
    ood.py           k-NN novelty score against D_success (S5.4)
    failure.py       online logistic P_fail model (S5.4)
    query_gate.py    acquisition function A(s_t) and request logic (S5.2)
    quality.py       intervention segments, q_int / w_i scoring, vetoes,
                     Option-A features (S5.5, S5.6)
    normalization.py running z-score utilities used by the gate (S5.2)
    logger.py        per-step / per-segment / per-episode JSONL logging
                     (S7 "Logging extension", feeds the RQ1/RQ2 analysis)
    runtime.py       AQIRuntime: glues everything into the actor loop of
                     examples/train_aqi_rlpd.py

Entry points elsewhere in the repo:
    examples/train_aqi_rlpd.py                  training with --aqi_mode
    examples/aqi_analysis.py                    offline RQ1/RQ2 analysis
    serl_launcher/data/data_store.py            QualityAwareReplayBufferDataStore
    serl_launcher/agents/continuous/sac*.py     quality-weighted imitation loss
"""

from serl_launcher.aqi.config import AQIConfig, MODES
from serl_launcher.aqi.failure import FailureRiskModel
from serl_launcher.aqi.force_risk import ForceRiskMonitor, action_jerk
from serl_launcher.aqi.logger import AQILogger
from serl_launcher.aqi.normalization import RunningVectorZScore, RunningZScore
from serl_launcher.aqi.ood import KnnOODScorer
from serl_launcher.aqi.quality import (
    InterventionSegment,
    QualityScorer,
    disagreement_gate,
    segment_features,
)
from serl_launcher.aqi.query_gate import ActiveQueryGate
from serl_launcher.aqi.runtime import AQIRuntime, make_proprio_slicer
from serl_launcher.aqi.uncertainty import (
    counterfactual_advantage_batch,
    counterfactual_advantage_batch_safe,
    lcb,
    q_stats_single,
    stack_obs,
)

__all__ = [
    "AQIConfig",
    "MODES",
    "AQILogger",
    "AQIRuntime",
    "ActiveQueryGate",
    "FailureRiskModel",
    "ForceRiskMonitor",
    "InterventionSegment",
    "KnnOODScorer",
    "QualityScorer",
    "RunningVectorZScore",
    "RunningZScore",
    "action_jerk",
    "counterfactual_advantage_batch",
    "counterfactual_advantage_batch_safe",
    "disagreement_gate",
    "lcb",
    "make_proprio_slicer",
    "q_stats_single",
    "segment_features",
    "stack_obs",
]

from serl_launcher.aqi.operator_model import (  # noqa: F401
    EXEC_ANCHORS, MON_ANCHORS, OperatorTheta, PhaseOracle, PolicyOracle,
    SyntheticOperator, jitter_theta, make_operator_theta,
    theta_from_json, theta_from_scale, theta_to_json)
from serl_launcher.aqi.operator_gate import (  # noqa: F401
    MonitoringSignal, SyntheticInterventionGate, SyntheticOperatorWrapper)
from serl_launcher.aqi import skill  # noqa: F401
