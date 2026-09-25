# Serl Launcher

 - Dependencies: `jax`, `agentlace`

Code and scripts are modified from [jaxrl_m](https://github.com/dibyaghosh/jaxrl_m) or [jaxrl_m private].(https://github.com/rail-berkeley/jaxrl_minimal)

## AQI-HIL-SERL extension (`serl_launcher/aqi/`)

This fork adds the **Active and Quality-Aware Intervention** subpackage
[`serl_launcher/aqi/`](./serl_launcher/aqi/). It is additive: importing it does
not change any default behaviour, and it is only activated through
`examples/train_aqi_rlpd.py --aqi_mode={hilserl,logged,active,quality,aqi}`
(`hilserl` reproduces the unmodified baseline). See
[docs/aqi_walkthrough.md](../docs/aqi_walkthrough.md) for the full method-to-code map.

| Module | Role |
| --- | --- |
| `config.py` | `AQIConfig`: modes, critic-ensemble size/subsample/bootstrap, trigger terms & thresholds, quality filters. Central place to tune the extension. |
| `runtime.py` | Per-step AQI runtime that assembles the acquisition score and drives the query gate during rollouts. |
| `query_gate.py` | Decides when the robot **requests** an intervention (score `> τ_A`) and handles safety overrides. |
| `uncertainty.py` | Critic-ensemble epistemic uncertainty `σ_Q` (the core active-learning signal). |
| `failure.py` | Short-horizon predicted-failure probability term. |
| `force_risk.py` | Force/torque contact-risk term and hard safety vetoes. |
| `ood.py` | k-NN novelty / out-of-distribution term against successful states. |
| `quality.py` | Counterfactual-advantage intervention scoring → replay priority & imitation weight. |
| `normalization.py` | Running z-score normalization of the individual trigger terms. |
| `logger.py` | Structured logging of AQI decisions/metrics for offline analysis. |
| `stats.py` | Bootstrap/ICC statistics used by the multi-operator study (`examples/aqi_analysis.py`). |

The critic **ensemble size** and per-member training logic that feed `σ_Q`
live in `config.py` (`critic_ensemble_size`, `critic_subsample_size`,
`critic_bootstrap_p`, `critic_per_member_targets`), are wired up in
`examples/train_aqi_rlpd.py`, and are consumed by the SAC critic in
[`agents/continuous/sac.py`](./serl_launcher/agents/continuous/sac.py)
(`critic_loss_fn`) plus the bootstrap masks in
[`data/replay_buffer.py`](./serl_launcher/data/replay_buffer.py).
