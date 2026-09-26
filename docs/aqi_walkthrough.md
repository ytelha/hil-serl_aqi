# AQI-HIL-SERL Walkthrough

This document describes the **Active and Quality-Aware Intervention
HIL-SERL (AQI-HIL-SERL)** extension of this repository, implementing the
research proposal *"Active and Quality-Aware Human-in-the-Loop
Reinforcement Learning for Operator-Robust Robotic Insertion"*. Section
numbers (S5.2, S6.4, ...) refer to that proposal.

The extension is additive: `examples/train_rlpd.py` and the default
behaviour of every existing module are unchanged. All new behaviour is
reached through `examples/train_aqi_rlpd.py` and the `--aqi_mode` flag.

---

## 1. What it does

Standard HIL-SERL treats every human correction as expert data and leaves
the timing of interventions entirely to the operator. AQI-HIL-SERL
controls **both sides of the human-feedback loop**:

1. **Acquisition (Active Learning) (S5.2-S5.4).** At every control step an acquisition score

   `A(s_t) = λ1·z(U_Q) + {OPTIONAL} λ2·z(P_fail) + λ3·z(R_force) + λ4·z(OOD) − λ5·z(C_human)`

   combines critic-ensemble uncertainty, a learned short-horizon failure
   probability, force/torque contact risk, k-NN novelty against successful
   states, and a workload penalty. The robot requests intervention when
   `A(s_t) > τ_A`; hard force/torque limits bypass the score entirely
   (safety override).

2. **Usage (S5.5-S5.6).** Each received intervention segment is scored by
   the **counterfactual advantage** (Option B, the primary signal):

   `q_int = clip( Â, 0, ∞ ) · g(σ_Q)`, with `Â = μ_Q(s, a_H) − μ_Q(s, a_robot)`

   i.e. how much better the operator's action was than what the robot
   would have done, as judged by the critic ensemble, gated by ensemble
   disagreement. Layered under it: the outcome filter (Option G) drops
   segments from clearly-failed episodes, and force/torque (and optional
   jerk) act as **hard safety vetoes**, not quality terms. Surviving
   transitions get a replay priority
   `p_i = (ε + |δ_i| + η·max(q_int,0))^α` and a weighted imitation term
   `L_actor = L_RL + λ_BC · Σ w_i·NLL_i / Σ w_i`, with
   `w_i = clip(σ(γ·q̃), w_min, w_max)` and `w_i = 0` for harmful
   corrections (`Â < adv_exclude`).

3. **Logging (S7).** Every step, segment, and episode is logged as JSONL
   with the operator ID, so the multi-operator variance analysis (RQ1)
   and the intervention-quality analysis (RQ2) run offline from
   `examples/aqi_analysis.py`.

## 2. Code map (proposal → repository)

| Proposal component | Location |
|---|---|
| AQIConfig, run modes, all hyperparameters | `serl_launcher/serl_launcher/aqi/config.py` |
| σ_Q, μ_Q, LCB, counterfactual advantage (S5.3, S5.5-B) | `serl_launcher/serl_launcher/aqi/uncertainty.py` |
| R_force, hard limits, jerk (S5.4, S5.5-vetoes) | `serl_launcher/serl_launcher/aqi/force_risk.py` |
| k-NN OOD trigger (S5.4) | `serl_launcher/serl_launcher/aqi/ood.py` |
| Online logistic P_fail (S5.4) | `serl_launcher/serl_launcher/aqi/failure.py` |
| Acquisition function A(s_t), hysteresis/cooldown (S5.2) | `serl_launcher/serl_launcher/aqi/query_gate.py` |
| Segments, q_int / w_i, vetoes, Option-A features (S5.5-S5.6) | `serl_launcher/serl_launcher/aqi/quality.py` |
| Segment-level scores (behavioral, hindsight, progress, outcome, uniform) and the post-release window ([quality_scores_v2.md](quality_scores_v2.md)) | `QualityScorer.apply_segment_score`, `hindsight_advantage` in `serl_launcher/serl_launcher/aqi/quality.py`; `AQIRuntime._post_release` in `runtime.py` |
| Operator-consistency BC weight σβ (SiLRI-style) | `serl_launcher/serl_launcher/aqi/consistency.py` |
| JSONL logging (S7) | `serl_launcher/serl_launcher/aqi/logger.py` |
| Actor-side orchestration | `serl_launcher/serl_launcher/aqi/runtime.py` |
| Quality-prioritized replay (S5.6) | `QualityAwareReplayBufferDataStore` in `serl_launcher/serl_launcher/data/data_store.py` |
| AQI per-transition fields | `include_aqi_fields` in `serl_launcher/serl_launcher/data/replay_buffer.py` |
| Weighted imitation loss (S5.6) | `policy_loss_fn` in `serl_launcher/serl_launcher/agents/continuous/sac.py`, `sac_hybrid_single.py`, `sac_hybrid_dual.py` (enabled by `aqi_bc_weight`) |
| K-critic ensemble plumbing (S5.3) | `critic_ensemble_size` / `critic_subsample_size` kwargs in `serl_launcher/serl_launcher/utils/launcher.py` |
| Bootstrapped ensemble training (S5.3): persistent per-(transition, member) masks + per-member targets | `bootstrap_ensemble_size`/`bootstrap_p` in `data/replay_buffer.py`, masked loss + `critic_per_member_targets` in the three `agents/continuous/sac*.py` critic losses |
| Training entry point (all modes) | `examples/train_aqi_rlpd.py` |
| Task-specific AQI settings | `get_aqi_config()` in `examples/experiments/<task>/config.py` |
| Offline RQ1/RQ2/RQ3 analysis | `examples/aqi_analysis.py` |

## 3. Run modes = proposal baselines (S6.4) and phases (S6.2)

`--aqi_mode` selects the variant; everything else is identical between
modes so comparisons are controlled:

| mode | Proposal name | Phase | Gate | Quality replay/loss | Logging | Critics |
|---|---|---|---|---|---|---|
| `hilserl` | Standard HIL-SERL | 1 | – | – | – | 2 (baseline) |
| `logged` | Logged HIL-SERL (diagnostic control) | 2 | – | – | ✓ | K=10, sub 2 |
| `active` | Active-HIL-SERL | 3 | ✓ | – | ✓ | K=10, sub 2 |
| `quality` | Quality-HIL-SERL | 4 | – | ✓ | ✓ | K=10, sub 2 |
| `aqi` | AQI-HIL-SERL (full) | 5 | ✓ | ✓ | ✓ | K=10, sub 2 |

`hilserl` reproduces `train_rlpd.py` behaviour exactly (2 critics, no
extra losses, no logging), so it is the plain baseline. `logged` is the
**same-architecture diagnostic control**: identical training to the
active/quality modes (K-critic ensemble, min-Q backup over a random
subsample of 2, REDQ-style) plus full logging, isolating the effect of
the gate and the quality module from the ensemble change.

### Which options each mode reads (and which are inert)

Because the VS Code tasks always prompt every input, `main()` **hard-drops
overrides that cannot change the chosen mode** (`_override_relevant` in
[`train_aqi_rlpd.py`](../examples/train_aqi_rlpd.py)) and prints a yellow
"ignoring options that do not apply". Relevance follows the mode switches
(`active_on = {active, aqi}`, `quality_on = {quality, aqi}`, ensemble = every mode
but `hilserl`):

| mode | trigger `terms`/`combiner` + `query_behavior` | quality `gate`/`filters` | ensemble `K`/`subsample` | logging |
|---|---|---|---|---|
| `hilserl` | ignored | ignored | ignored (fixed **2**) | – |
| `logged` | ignored | ignored | **applies** (K=10) | ✓ |
| `active` | **applies** | ignored | **applies** | ✓ |
| `quality` | ignored | **applies** | **applies** | ✓ |
| `aqi` | **applies** | **applies** | **applies** | ✓ |

So `hilserl` treats *every* AQI option as inert; `logged` shares the ensemble but
ignores the gate and quality; `active` ignores quality; `quality` ignores the
trigger and pausing; only `aqi` reads them all. Dropping an inert override also
removes it from that mode's **logs** (e.g. `logged`/`quality` log with the default
gate config) — a deliberate diagnostic control that never changes the net
architecture (only the mode and K do). Each cell's provenance README derives the
same active/inert labels from the live `AQIConfig`, so an artifact always states
which choices were inert for it.

> **The ensemble Q-values themselves** are aggregated three different ways (min
> for the backup, mean for the actor, std for the trigger `σ_Q`) — see
> [critic_ensemble.md](critic_ensemble.md).

### Composable trigger and quality (S6.4 ablations)

The trigger and the quality module are built from **independent sections
combined by explicit logic**, so every ablation is a composition choice
rather than a code change:

**Trigger.** Each term is its own module (uncertainty, failure, force,
OOD, cost); `trigger_terms` selects which sections feed A(s_t) — inactive
sections are not computed at all and their keys are absent from the step
logs — and `trigger_combiner` selects the logic:

* `weighted_sum` (default): `A = Σ λ_i·z_i − λ_cost·z_cost` (proposal S5.2)
* `max` (OR-logic): request when ANY single weighted term exceeds `τ_A`

The minimal trigger is the **ensemble epistemic uncertainty alone**:
`trigger_terms=("uncertainty",)`. The hard force/torque limits are a
safety mechanism, not a term — the safety override and the red safety
prompt stay active under every composition.

**Quality.** The graded signal is always the **counterfactual advantage
Â** (Option B). Around it sit removable layers: the disagreement gate
g(σ_Q) is a switch (`quality_disagreement_gate`), and the binary
keep/drop filters are named sections selected by `quality_filters`
(`safety_veto`, `jerk_veto`, `outcome_filter`, `unsolicited`), combined by
short-circuit OR with the first hit recorded as `exclusion_reason`.
`quality_disagreement_gate=False, quality_filters=()` gives **pure
counterfactual quality**: q = Â exactly (the `adv_exclude` hard-zero is
derived from Â itself and stays with the core; disable via
`adv_exclude=-inf`).

Set compositions in the experiment's `get_aqi_config()` or from the
command line (overrides the config only when passed):

```bash
# pure ensemble trigger + pure counterfactual quality
bash run_actor_aqi.sh --aqi_mode=aqi --operator_id=alice \
    --trigger_terms=uncertainty \
    --quality_disagreement_gate=false --quality_filters=none

# OR-logic gate over uncertainty and force only
bash run_actor_aqi.sh --aqi_mode=aqi \
    --trigger_terms=uncertainty,force --trigger_combiner=max
```

**Ensemble data (S5.3).** For sigma_Q to be an epistemic signal the K
critics must see *different data*: every replay buffer stores a
persistent Bernoulli(`critic_bootstrap_p`, default 0.8) mask per
(transition, member), drawn once at insertion, and each member's TD loss
is weighted by its mask; `critic_per_member_targets` (default on)
additionally gives every member its own independent min-subsample target
draw instead of the single shared REDQ target. Persistence matters:
resampling masks per update over a stream of fresh data is statistically
a no-op, which is why the masks live in the buffer with the transition.
The shared-data ablation (`--bootstrap_p=1.0 --per_member_targets=false`)
recovers the init-diversity-only deep-ensembles variant; `hilserl` mode
ignores both and stays the exact 2-critic shared-target baseline.

Classic S6.4 ablations map directly: *no force term* →
`trigger_terms` without `force` (or `lambda_force=0`); *no OOD term* →
drop `ood`; *no q_int* → `--aqi_mode=active`; *no outcome filter* →
`quality_filters` without it. The composition in force is stored in each
run's `meta.json`, so the offline analysis can group runs by it.

## 4. Running

The workflow is the standard HIL-SERL one (record demos, train the
reward classifier, then actor+learner) with `train_aqi_rlpd.py` in place
of `train_rlpd.py`. For the RAM insertion task:

```bash
cd examples/experiments/ram_insertion

# learner (add your demo pkl paths)
bash run_learner_aqi.sh --aqi_mode=aqi \
    --demo_path=./demo_data/ram_insertion_20_demos.pkl

# actor, one terminal per operator session
bash run_actor_aqi.sh --aqi_mode=aqi --operator_id=alice \
    --demo_path=./demo_data/ram_insertion_20_demos.pkl
```

Notes:

* `--operator_id` tags every logged step/segment/episode; use one ID per
  human operator — the RQ1 analysis groups runs by it.
* Passing `--demo_path` **to the actor as well** seeds the OOD reference
  set `D_success` with demonstration states (S5.4). It is optional; the
  reference set also fills from successful online episodes.
* Actor and learner must use the same `--aqi_mode` (the learner needs it
  to build the right buffers and loss).
* Logs land in `aqi_logs/<run_name>/` next to where the actor was
  launched; `--run_name` overrides the default
  `<exp>_<mode>_<operator>_s<seed>_<timestamp>`.
* When the robot requests intervention, the actor prints a yellow
  `ROBOT REQUESTS INTERVENTION` line (red for the hard force/torque
  safety override). Take over with the SpaceMouse as usual; unsolicited
  takeovers are also honored and logged with `requested=false`
  (`AQIConfig.unsolicited_policy` controls whether they enter the
  imitation buffer).

### Multi-operator study (RQ1, S6.3)

For each mode you compare, run the same training budget once per
operator (ideally with repeated runs per operator for the
within-operator variance / ICC):

```bash
for OP in alice bob chris; do
  bash run_actor_aqi.sh --aqi_mode=logged --operator_id=$OP \
      --checkpoint_path=runs/logged_$OP --seed=0
  # ... paired learner with the same flags ...
done
```

### The operator study (Experiments 1 and 2)

The pre-registered protocol - the frozen evaluation ruler, per-operator
session run-book, the dependency (ICC, permutation) and comparison
(RG_rel, guardrail, interaction model) statistics, and the decision
tables - is a separate pipeline documented in
[experimental_protocol.md](./experimental_protocol.md)
(`examples/experiment_pipeline.py`, `examples/eval_ruler.py`,
`serl_launcher/aqi/stats.py`). The analysis below is the lighter,
training-log-only companion for day-to-day runs.

### Offline analysis

```bash
python examples/aqi_analysis.py --log_root aqi_logs \
    --baseline_mode logged --out aqi_analysis_out
```

Produces `summary.md`, `results.json`, and figures (`.png` + `.svg`):
learning curves, per-operator spread per mode, q_int distributions per
operator, and the P_fail reliability curve. Reported statistics include
the per-mode variance/CV across operators, ICC(1,1), and the robustness
gain vs the baseline mode: `RG_abs = Var_op(base) − Var_op(treat)` and
the primary endpoint `RG_rel = RG_abs / Var_op(base)` (S6.6).

## 5. Calibration recommendations

1. **Phase 2 first.** Run `--aqi_mode=logged` for a few sessions and use
   the logged `f_norm` distribution to set `force_warn` / `force_hard`
   (warn ≈ 90th percentile of nominal contact, hard ≈ safety limit of
   your fixture) in the experiment's `get_aqi_config()`.
2. `query_warmup_steps` should cover at least ~5 episodes so the running
   z-statistics and P_fail model are informed before the gate can fire.
3. `tau_A` around 1.0 with the default unit lambdas fires when the
   combined score is one standard deviation above its running mean;
   lower it for a more talkative robot.
4. Leave `use_td_priority=False` initially: priorities then come purely
   from q_int with the optimistic |δ| init (PER convention), which is
   the proposal's minimal variant. `update_priorities()` on the
   quality buffer is available for the TD-refresh extension.
5. Sanity-check the counterfactual signal offline before trusting it
   online: `aqi_analysis.py` reports whether `q_int` predicts
   post-intervention success better than the hand-weighted Option-A
   features (the central RQ2 comparison).

## 6. Data formats

**Transitions** in quality modes carry three extra float fields —
`q_int`, `quality_weight`, `intervention_mask` — on top of the standard
HIL-SERL keys. Autonomous transitions carry zeros; pre-recorded demos are
defaulted on insert (`demo_weight`, default 0, keeps RLPD semantics: demos
train the critic via sampling, not the imitation loss). Buffers dumped by
a quality-mode run therefore cannot be resumed by `train_rlpd.py`
(baseline-mode buffers resume fine in either script).

**Logs** under `aqi_logs/<run>/`:

* `steps.jsonl` — per-step trigger timeline: raw and z-scored terms,
  `a_score`, `request`, `reason`, `p_fail`, `sigma_q`, `f_norm`,
  `hard_violation`, `intervening`, optional proprio state.
* `segments.jsonl` — per intervention segment: operator, start/length,
  `requested`, per-step `a_hat`/`q_int`/`weights`, exclusion decision and
  reason (`safety_veto` / `jerk_veto` / `outcome_filter` / `unsolicited`),
  and the Option-A feature dict for the RQ2 fit.
* `episodes.jsonl` — outcome, return, length, intervention counts,
  request counts, peak force/torque, hard-violation count.
* `meta.json`, `gate_state.json` — run metadata and final gate /
  P_fail-model state.

## Synthetic operators and skill assessment

The synthetic-operator generator (perception -> decision -> motor model, tier anchors, theta(s), the SDT intervention gate and the SpaceMouse-equivalent wrapper) and the Part-B skill-assessment pipeline (Table-7 features, Methods A-D, S11 validation) are documented in `synthetic_operators_and_skill.md`; the wrapper plugs in via `--synthetic_operator` in `train_aqi_rlpd.py` and produces logs this walkthrough's schema already covers.
