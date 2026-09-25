# HiRIL comparison baseline (`examples/train_hiril.py`)

An external baseline for the thesis: **HiRIL** — Y. Zhou, Y.-B. Zhao, C. Xu,
C. Ouyang, P. Li, *"Human-in-the-loop reinforcement learning with risk-aware
intervention and imitation"*, Expert Systems With Applications 308 (2026)
131118, doi:10.1016/j.eswa.2026.131118. No official code is published; this
is an implementation from the paper.

HiRIL is the published method closest to AQI:
- **Robot-triggered interventions.** The robot asks for a human when a
  critic-based risk score is high (AQI: the active gate).
- **Risk-weighted replay.** Human corrections are replayed more when they
  reduced risk (AQI: quality-aware replay).
- **Behaviour cloning** on human actions.

It differs in the learner (TD3 + a Bayesian distributional critic) and in the
risk score, so it runs as its **own pipeline**, not as an `--aqi_mode`.

---

## 1. The method, equation by equation

| paper | what | code |
|---|---|---|
| Eq. 12 | BIQN return samples `Z_{θ,τ}(s,a) = f_τ(s,a;θ)`, `θ ~ N(μ,Σ)`, `τ ~ U(0,1)` | `hiril/agent.py`: `BayesianDense`, `BIQNHead`, `BIQNCritic` (output `(M weight samples, B, N quantiles)`) |
| Eq. 13–15 | TD `δ = r + γ Z_{θ,τ'}(s', π(s')) − Z_{θ,τ}(s,a)`; quantile-Huber loss averaged over M weight samples | `quantile_huber_loss`, `HiRILAgent.critic_loss_fn` |
| Eq. 14 | `Q̂ = mean over K_θ × K_τ samples` | `z.mean(axis=(0, 2))` |
| Eq. 16–20 | risk `DU(s,a) = Var[Z] = E[Z²] − E[Z]² (+σ_z)` over θ and τ | `dual_uncertainty_stats`, `HiRILAgent.dual_uncertainty` |
| Eq. 21 | `a_RL = π(s) + ξ·a_std`, exploration decaying 1.0 → 0.05 | `HiRILAgent.sample_actions(noise_std)`, `train_hiril.exploration_std` |
| Eq. 22–23 | request a human iff `DU(s, a_RL) > τ_risk` | `train_hiril.actor`: `request` → `env.notify_request` |
| Eq. 25 | reward `+ r_p` once at each takeover onset | `stored_reward` in the actor |
| Eq. 26–28 | TDRD priority `k = |δ_TD| + ε (+ exp(DU(s,a_RL) − DU(s,a_H)) for human samples)` | `hiril/priorities.py`, `RiskAwarePERDataStore` |
| Eq. 29 | critic loss `= E_D1[·] + E_D2[·]` (RL and human parts averaged separately) | `split_mean` in `critic_loss_fn` |
| Eq. 32 | actor loss `= E_D1[−Q̂(s,π(s))] + λ E_D2[‖π(s) − a_H‖²]` | `HiRILAgent.actor_loss_fn` |
| Alg. 1 | the loop | `train_hiril.py` (docstring maps every line) |
| Tables A1/A2 | hyperparameters | `hiril/config.py` (each default tagged with its source) |

The risk score decomposes exactly, by the law of total variance, into
`aleatoric = mean_θ Var_τ Z` and `epistemic = Var_θ mean_τ Z`. The paper's
Fig. 6 ablations are `du_mode = aleatoric` ("w/o epistemic") and
`du_mode = epistemic` ("w/o aleatoric").

## 2. How it plugs into this repo

| piece | reused as is | new for HiRIL |
|---|---|---|
| env, task config, cameras, proprio | ✓ (`CONFIG_MAPPING`) | |
| pixel encoder (pretrained ResNet-10 + spatial embeddings), random-crop aug | ✓ same code path as the SAC agents | |
| actor/learner over agentlace, READY banner, STOP file, checkpoints | ✓ same conventions | `train_hiril.py` |
| synthetic operator | ✓ `SyntheticOperatorWrapper` with `respond_to_request=True` (as AQI `active`/`aqi`) | `examples/synthetic_operator_env.py` (flag-free copy of the wrapper setup) |
| convergence rule, certified checkpoint, `CONVERGED.json`, `episodes.jsonl` | ✓ same files and semantics | |
| eval ruler | ✓ | `--agent_type auto|sac|hiril` (auto reads `<ckpt>/agent_type.json`, written by the HiRIL learner) |
| campaign runner | ✓ arms | `mode: hiril` arms launch `train_hiril.py` with only its own flags (`PIPELINES` in `run_campaign.py`); YAML `hiril:` block; `--hiril_*` CLI |
| logs | ✓ `AQILogger` → `aqi_logs/<run>/{meta,episodes,segments}.jsonl` | per-episode DU / request stats, per-takeover DU and RD |
| learner | | `hiril/agent.py` (TD3 + BIQN), `RiskAwarePERDataStore` (single TDRD buffer) |

Mapping the paper's human to the synthetic operator:
- When DU exceeds `τ_risk`, the robot **requests** a takeover.
- The operator responds through its skill-modulated gate: reaction delay,
  misses, false alarms, handback. This is exactly the mechanism AQI's
  active arms use, so HiRIL-vs-AQI comparisons share one operator model.
- The label Δ_t (human experience, D2) is the **actual** takeover, not the
  request.

## 3. Choices the paper leaves open (all are `HiRILConfig` fields; ablate them)

| choice | default here | why |
|---|---|---|
| `τ_risk` | **calibrated once**: the 80th percentile of `DU(s, a_RL)` over 1000 steps after warm-up, then frozen (`tau_risk ≤ 0`); `tau_risk > 0` = absolute, paper-style | The paper's thresholds (12–16) live on CARLA's return scale. A frozen calibrated threshold keeps the paper's semantics: intervention frequency should *fall* as DU falls with learning (their Fig. 7b). The calibrated value is logged in `aqi_logs/<run>/gate_state.json`. |
| `r_p` (intervention penalty) | −0.1 | Not reported. The task reward here is sparse 0/1. |
| `λ` (BC weight) | 1.0 | Not reported ("manually determined"). |
| exploration decay horizon | 20 000 env steps | Paper: over 400 CARLA episodes. |
| which layers are Bayesian | the critic **head** (all layers after the encoder) | The pixel encoder stays the deterministic, shared, pretrained one every method uses. |
| KL term | off (`kl_weight = 0`) | As the paper (Eq. 10–11 drop it). Without KL the weight std is shaped only by the TD loss and may shrink during training; `kl_weight > 0` is available. |
| target policy smoothing, Polyak rate | TD3 defaults (0.2 / 0.5, 0.005) | The paper builds on TD3 but lists neither. |
| offline demos | human experience (D2) with RD term `exp(0) = 1` | Their `a_RL` was never observed. |
| batch size, discount, replay capacity | the task config's (same as every other method) | The comparison should differ only in the learning method; the paper uses batch 128 and γ 0.95. |

## 4. Running it

Single run (two shells):

```bash
cd examples && source experiments/sim_peg_insertion/sim_env.sh
python train_hiril.py --exp_name=sim_peg_insertion --learner \
    --demo_path=<demos.pkl> --checkpoint_path=<ckpt>
python train_hiril.py --exp_name=sim_peg_insertion --actor \
    --checkpoint_path=<ckpt> --synthetic_operator=expert --synthetic_intervention_mode=signal
```

Without `--synthetic_operator` the SpaceMouse human is used. The terminal
rings when HiRIL requests a takeover.

Campaign: [campaigns/hiril_comparison.yaml](../examples/campaigns/hiril_comparison.yaml),
`hilserl` vs `hiril`, with RQ4-identical settings, so RQ4's 18 `hilserl`
cells can be reused (recipe in [e2hil_baseline.md](e2hil_baseline.md) §4). To
include the paper's ablations, add arms:

```yaml
methods: [hilserl, hiril, hiril-no-epi, hiril-no-rd, hiril-no-bc]
arms:
  hiril-no-epi: {mode: hiril, hiril_du_mode: aleatoric}   # Fig. 6 "w/o epistemic"
  hiril-no-rd:  {mode: hiril, hiril_use_rd: false}         # Fig. 8 "w/o RD"
  hiril-no-bc:  {mode: hiril, hiril_lambda_bc: 0.0}        # Fig. 8 "w/o imitation"
```

`hiril_intervention_penalty: 0.0` gives "w/o reward shaping", and
`hiril_du_mode: epistemic` gives "w/o aleatoric".

## 5. What to read out

- **`report/dependency.md`:** SR, ICC, convergence, paired deltas, and the
  "Human effort: intervention rate" section, which gives the paper's
  intervention-rate metric.
- **`aqi_logs/<run>/episodes.jsonl`**, per episode:
  - `du_mean`, `du_max`, `tau_risk`;
  - `n_requests`, `n_request_steps`, `n_intervention_steps`;
  - `explore_std`.

  The requests-per-episode curve is the paper's Fig. 7b.
- **`aqi_logs/<run>/segments.jsonl`**, per takeover:
  - `du_rl_at_trigger`, `du_rl_mean`, `du_h_mean`;
  - `rd_bonus_mean`: does the operator's action actually lower the risk?
- **wandb (learner):**
  - `critic/{critic_loss_rl, critic_loss_human, q, human_frac_batch, is_weight_mean}`;
  - `actor/{actor_q, bc_loss}`;
  - `hiril_replay/{human_frac, human_priority_share, human_bonus_mean}`;
  - actor stats `hiril/{du_mean, request_rate, intervention_rate, tau_risk}`.

## 6. Tests

```bash
python tests/test_hiril_priorities.py   # numpy: TDRD priorities, config
python tests/test_campaign_hiril.py     # campaign wiring; statically checks
                                        # every emitted flag exists
python tests/test_hiril_agent.py        # JAX: Bayesian layer, DU identity,
                                        # IQN loss vs numpy, TD3 delay, D1/D2
                                        # losses, acting, TDRD data store
```
