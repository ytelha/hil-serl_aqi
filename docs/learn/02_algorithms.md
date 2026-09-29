# 02 — The algorithms, from the maths to the code

Each section follows the same pattern:
- **the idea** in plain words;
- **the maths**;
- **the code**: file → function;
- **your results**: what happened in this thesis;
- **check yourself**: a question you should be able to answer before
  moving on.

Read the sections in order: each builds on the previous one.

---

## 1. Reinforcement-learning basics (the vocabulary everything else uses)

**Idea.** An agent observes a state, acts, gets a reward, and wants to
maximise the discounted sum of future rewards.

**Maths.**
- Return: Gₜ = Σₖ γᵏ rₜ₊ₖ.
- The action value Q(s, a) is the expected return after taking a in s and
  then following the policy π. The state value is V(s) = E_{a∼π} Q(s, a).
- The advantage is A(s, a) = Q(s, a) − V(s): how much better a is than the
  policy's average action.
- The Bellman equation, Q(s, a) = r + γ E[V(s')], is what TD learning
  exploits.

**In this task:**
- **states:** a 19-D proprio vector + two 128×128 wrist-camera images;
- **actions:** 6-D end-effector velocity commands in [−1, 1] at 10 Hz;
- **reward:** sparse, 1 on successful insertion, else 0 (`SimPegRewardWrapper`);
- **episodes:** end on success or after `max_episode_length`;
- **discount:** γ = 0.97 (`examples/experiments/config.py`).

**Read:** Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd
ed.), chapters 3, 6 and 13. OpenAI *Spinning Up*, "Key concepts" and
"Kinds of RL algorithms".

**Check yourself:** why does a sparse reward make learning hard, and which
two things in HIL-SERL help against it? (Demos and human interventions.)

## 2. Soft Actor-Critic (SAC): the learner at the core

**Idea.** An off-policy actor–critic that maximises reward **plus policy
entropy**, so it keeps exploring. Its critics learn Q; the actor learns to
pick actions the critics like; a temperature α balances reward against
entropy.

**Maths**, as implemented here:
- **critic target:** y = r + γ · mask · min_{j∈M} Q̄ⱼ(s', a'), with
  a' ∼ π(·|s'). Q̄ are the **target** critics: a slow Polyak copy,
  τ = 0.005. With `backup_entropy=False`, as here, there's no entropy term
  in the target. M is a random subset of the ensemble (§4).
- **critic loss:** mean over critics of (Qᵢ(s, a) − y)².
- **actor loss:** −E[ Q̄(s, ã) − α log π(ã|s) ], with ã ∼ π(·|s)
  reparameterised. Q̄ is the **mean** over critics here; E2HiL uses the min
  (§6).
- **temperature:** α is tuned so the policy's entropy stays near a target,
  by default −|A|/2 = −3 for 6-D actions.
- The policy is a **tanh-squashed Gaussian**, which keeps actions in [−1, 1]
  and needs a log-prob correction.

**Code:**
- `serl_launcher/agents/continuous/sac.py`:
  - `critic_loss_fn` (target: `forward_target_critic`, subsample + min,
    `per_member` branch);
  - `policy_loss_fn`;
  - `temperature_loss_fn`;
  - `update` (one jitted step; `networks_to_update` chooses critic-only or
    all);
  - `target_update` via `common/common.py` → `JaxRLTrainState`.
- The networks: `networks/actor_critic_nets.py` (`Policy`,
  `TanhMultivariateNormalDiag`, `Critic`).

**Read:**
- Haarnoja et al. 2018, *Soft Actor-Critic: Off-Policy Maximum Entropy
  Deep RL with a Stochastic Actor*;
- *Soft Actor-Critic Algorithms and Applications* (automatic α);
- Spinning Up's SAC page is the clearest short explanation.

**Check yourself:** point to the line where the min over critics is taken
in the target, and to the line that makes the actor objective use the mean.

## 3. RLPD and HIL-SERL: SAC with demos and a human in the loop

**Idea.**
- **RLPD** (*Efficient Online RL with Offline Data*, Ball et al. 2023):
  every batch is **half online data, half prior data** (demos). Plus
  layer-norm critics, an ensemble, and many updates per step.
- **SERL** (Luo et al. 2024) packages this for real robots: image encoders,
  and the actor/learner split.
- **HIL-SERL** (Luo et al. 2024, *Precise and Dexterous Robotic
  Manipulation via Human-in-the-Loop RL*) adds **human takeovers** with a
  SpaceMouse. The takeover transitions go into the demo buffer (so they're
  sampled at 50%) as well as the online buffer.

**Code:**
- `examples/train_rlpd.py` (original) and `examples/train_aqi_rlpd.py`
  (thesis):
  - `learner`: two iterators with `batch_size // 2` each →
    `concat_batches` → `agent.update`, with `cta_ratio − 1` critic-only
    updates first;
  - `actor`: interventions routed to `intvn_data_store`.
- Images: a frozen ImageNet ResNet-10 (`vision/resnet_v1.py` →
  `PreTrainedResNetEncoder`) + random-crop augmentation
  (`vision/data_augmentations.py`, the DrQ idea: Kostrikov et al. 2020).
- Demos: `examples/record_demos.py` (human) or
  `examples/collect_synthetic_demos.py` (synthetic operator).

**Your results:** plain HIL-SERL (`hilserl`, 2 critics) is the reference.
It reaches ~84–87% autonomous SR on the cuboid ruler, with low operator
dependency (ICC ≈ 0–0.03; RQ3/RQ4 logs).

**Check yourself:** why are interventions put in the **demo** buffer, and
what fraction of each batch comes from it?

## 4. Critic ensembles (REDQ, per-member targets, bootstrapping)

**Idea.** Train K critics instead of 2. The spread between them, σ_Q, is a
measure of **epistemic uncertainty** (the critic doesn't know yet), which
the active gate uses. **REDQ** (Chen et al. 2021) takes the target as the
min over a random subset of M critics.

Three choices decide whether σ_Q stays meaningful:
- **per-member targets** (each critic its own target);
- **bootstrap masks** (each critic trains on ~80% of the data);
- **M** (M = 2 is pessimistic; M = 1 has no min).

**Code:**
- `networks/actor_critic_nets.py` → `ensemblize(cls, num_qs)`: one module
  whose parameters get a leading axis of size K, via Flax `nn.vmap`.
- `sac.py` → `critic_loss_fn`: the subsample + `per_member_targets`.
- The bootstrap masks live in the replay buffers
  (`data/replay_buffer.py`, `data/memory_efficient_replay_buffer.py`).
- σ_Q at run time: `aqi/uncertainty.py` → `q_stats_single`.

**Your results** (`docs/thesis/rq3_…`, `rq4_…`; [`critic_ensemble.md`](../critic_ensemble.md)):
- K = 10, M = 2 **hurt**: SR −18.5 pp, ICC 0.03 → 0.74 (RQ3).
- The "B fixes" (M = 1, K = 8, 5000 demo-pretraining steps, LR 1.5e-4)
  brought parity with plain HIL-SERL (RQ4).
- M 2 → 1 is the prime suspect: the min under-values the risky "press into
  contact" action.
- On the cluster, σ_Q shrank ~90× in 3000 steps (finding F6).

**Check yourself:** what is the shape of `forward_critic`'s output, and
what does `.std(axis=0)` of it give you?

## 5. AQI — the thesis method

AQI adds two mechanisms to HIL-SERL. Read
[`aqi_walkthrough.md`](../aqi_walkthrough.md) first.

### 5.1 Run modes (the ablation ladder)

`aqi/config.py` → `AQIConfig.mode`:

| mode | ensemble | gate | quality | meaning |
|---|---|---|---|---|
| `hilserl` | 2 critics | — | — | plain HIL-SERL |
| `logged` | K critics | computed, not acted on | computed, not used | same architecture, uniform replay |
| `active` | K | **acts**: operator responds to requests | — | the gate only |
| `quality` | K | computed | **used** (priority / BC) | quality only |
| `aqi` | K | acts | used | both |
| `e2hil` | 2 critics | — | — | external baseline (§6) |

The properties `active_on`, `quality_on`, `logging_on` and `baseline_arch`
decide which code runs. Knobs that don't apply to a mode are "inert", and
each cell's README says so.

### 5.2 The active query gate (when to ask for help)

**Maths.** A(s) = Σᵢ λᵢ · z(termᵢ) − λ_cost · z(workload). The terms:
- σ_Q (uncertainty);
- P_fail (online logistic failure predictor);
- R_force (contact risk);
- novelty (kNN distance to states of successful episodes).

z(·) is a running z-score. The gate requests if A > τ_A, with warm-up,
hysteresis and cool-down; hard force limits always request.

**Code:**
- `aqi/query_gate.py` → `ActiveQueryGate.step`;
- `aqi/normalization.py` → `RunningZScore`;
- `aqi/failure.py`, `aqi/force_risk.py`, `aqi/ood.py`;
- glue in `aqi/runtime.py` → `observe_step`.

**Your results.** Findings F6/F7 in `handoff/FINDINGS_cluster_method_test.md`:
- the uncertainty term fades out (all-time normalisation), and most requests
  were safety overrides;
- the failure predictor and the old evaluation are time-confounded.

The active plan (`handoff/EXPERIMENT_PLAN_active_learning.md`) is how to
test it properly.

**Check yourself:** why does a Welford (all-time) z-score of a signal that
shrinks 90× end up near a constant?

### 5.3 Intervention quality (which human data to trust)

**Idea.** Score each takeover segment with a signed quality q. Then:
- the **replay priority** p = (ε + |δ| + η·max(q, 0))^α;
- optionally a **BC weight** w = sigmoid(γ·q) in an imitation term added to
  the actor loss.

The scores (`quality_score_mode`):

| score | formula (short) | status in your results |
|---|---|---|
| `counterfactual` | Â = mean Q(s, a_H) − mean Q(s, a_robot) | anti-predictive (AUROC 0.175, RQ4 §11); study pending |
| `behavioral` | z(ΔV) − z(t_int) − z(F_peak) | predicts outcome (0.78) but t_int may be timing |
| `hindsight` | n-step advantage on visited states over [onset, handback + K] | ≈ takeover length before any reward (F4) |
| `progress` | change of insertion depth in the same window | to be tested (C7b) |
| `outcome` | ±1 by episode success (a control) | to be tested |
| `uniform` | 0 (no information: a control) | to be tested (Q2/Q3a) |
| consistency | c = 1/(1 + (σβ/σ_ref)²): BC weight only | ran on the cluster; to be tested |

**Code:**
- `aqi/quality.py`: `QualityScorer`, `SEGMENT_FILTERS`,
  `hindsight_advantage`, `apply_segment_score`;
- `aqi/runtime.py` → `end_episode`, `_segment_score`;
- `aqi/consistency.py`;
- the prioritised buffer `data/data_store.py` →
  `QualityAwareReplayBufferDataStore`;
- the BC term in `sac.py` → `policy_loss_fn` (`aqi_bc_weight`).

**Read:**
- PER: Schaul et al. 2016, *Prioritized Experience Replay*;
- AWR: Peng et al. 2019; CRR: Wang et al. 2020 (advantage-weighted
  imitation);
- [`quality_scores_v2.md`](../quality_scores_v2.md),
  [`quality_signal_audit.md`](../quality_signal_audit.md),
  [`counterfactual_score_study.md`](../counterfactual_score_study.md).

**Check yourself:** why is Q(s, a_H) − Q(s, a_robot) negative almost by
construction when the policy is trained to maximise the same critic?

## 6. E2HiL (external baseline): entropy-guided state selection

**Idea.** For each training state, estimate how strongly its actor update
would change the policy's entropy:
- c(s) = −Cov_k(log π(a_k|s), π(a_k|s)·A_soft(s, a_k)), with K = 8 sampled
  actions;
- the actor loss uses only states whose |c| lies between the 5th and 90th
  percentile of the batch.

The paper says the dropped states are mostly human interventions.

**Code:**
- `agents/continuous/e2hil.py`: `influence_per_state`,
  `entropy_bounded_mask`, `e2hil_actor_loss`, called from
  `sac.policy_loss_fn` when `config["e2hil"]`;
- the mode `e2hil` uses the 2-critic architecture.

Read [`e2hil_baseline.md`](../e2hil_baseline.md) (Deng et al., RA-L 2026).

**Your results:** only a 3000-step functional test so far. Each update
costs ~2× (K critic passes), so it gets ~half the updates per env step (F3).

## 7. HiRIL (external baseline): risk-triggered requests with a Bayesian critic

**Idea.**
- A distributional critic (**Bayesian IQN**) models the return
  distribution Z(s, a) with random weights θ and quantiles τ. The **dual
  uncertainty** is DU = Var[Z] over both. It splits into:
  - aleatoric = mean_θ Var_τ Z (the outcome itself is random);
  - epistemic = Var_θ mean_τ Z (the critic doesn't know).
- The robot requests help when DU(s, a_RL) > τ_risk.
- Actor: TD3-style (deterministic, delayed updates) plus a BC term on human
  actions.
- Rewards: a small penalty r_p at each takeover onset.
- Replay priority (TDRD): |δ_TD| + ε, plus a bonus exp(DU(s, a_RL) −
  DU(s, a_H)) for human samples whose action lowers the risk.

**Code:**
- `serl_launcher/hiril/agent.py` (`BayesianDense`, `BIQNCritic`,
  `dual_uncertainty_stats`, `quantile_huber_loss`);
- `hiril/config.py`, `hiril/priorities.py`;
- the whole loop in `examples/train_hiril.py`;
- the buffer `data_store.RiskAwarePERDataStore`.

Read [`hiril_baseline.md`](../hiril_baseline.md) (Zhou et al., ESWA 2026).

**Your results:** with τ_risk calibrated once, DU fell ~1000× and HiRIL
made **no** requests after calibration (F2).

## 8. The synthetic operator (how "humans" are simulated)

**Idea.** An operator has two skill axes: **execution** (how well it
moves) and **monitoring** (when it decides to take over). Its parameters
θ come from tier anchors plus per-operator jitter. When acting:
- **perception:** a biased estimate of the lateral error, with latency;
- **decision:** a phase machine: approach → align → descend,
  retract-and-realign when jammed;
- **motor:** noise that grows with speed, tremor, pauses, lapses.

A monitoring gate decides takeovers from a trouble signal z (thresholds,
reaction delay, misses, false alarms).

**Code:**
- `aqi/operator_model.py`: `OperatorTheta`, `theta_from_scale`,
  `jitter_theta`, `SyntheticOperator` (`_intent_phase`, `_motor`, `act`);
- `aqi/operator_gate.py`: `MonitoringSignal`, `SyntheticInterventionGate`,
  `SyntheticOperatorWrapper`;
- the sim's geometric oracle:
  `examples/experiments/sim_peg_insertion/wrapper.py` → `SimPegPhaseOracle`.

Read [`methods.md`](../methods.md) §1 and
[`oracle_and_synthetic_operator.md`](../oracle_and_synthetic_operator.md).

**Your results:** finding F1 was a real bug here. The jitter looped over a
Python `set`, so every process drew a different operator. Fixed in patch
0007.

## 9. Measuring operator dependency (the statistics)

**Idea.** Train the same method with many operators and several seeds.
Operator dependency is how much of the variance in the final success rate
comes **between** operators vs **within** an operator (between seeds).

**Maths:**
- one-way ANOVA → σ²_between, σ²_within;
- ICC(1,1) = σ²_b / (σ²_b + σ²_w);
- a permutation test on σ²_between;
- bootstrap CIs;
- robustness gain RG_rel = 1 − Var_op(method) / Var_op(baseline);
- paired per-cell deltas with a sign test;
- a non-inferiority guardrail on mean SR.

**Code:** `aqi/stats.py` (`icc_oneway`, `permutation_sigma2`,
`bootstrap_icc`, `robustness_gain`, `paired_bootstrap_rg_rel`,
`wilson_interval`, …), used by `examples/run_campaign.py` → `analyze`.

Read [`dependency_analysis.md`](../dependency_analysis.md): a worked
decode of a real report.

**Check yourself:** why can a method that never asks for help show ICC ≈ 0,
and which other numbers must you look at before calling that a success?
(Human effort; SR — see the active plan §5.)

## 10. The evaluation ruler

**Idea.** Every trained policy is scored the same way, with no operator,
on a fixed, hash-locked grid of start poses (±16 mm, 9 poses). Several
checkpoints are scored and summarised with `best_window`. Training success
rates are never compared across methods.

**Code:** `examples/eval_ruler.py`; `run_campaign.py` → `build_grid`,
`eval_cell`. Read [`eval_ruler.md`](../eval_ruler.md) and
[`architecture.md`](../architecture.md) §3 (the three kinds of "success
rate").
