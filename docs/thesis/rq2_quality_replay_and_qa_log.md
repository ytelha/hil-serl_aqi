# RQ2 Quality-Aware Replay — Setup, Implementation & Session Q&A Log

**Period covered:** 2026-09-10 → 2026-09-16
**Task:** MuJoCo `sim_peg_insertion`, **cuboid** peg (5 mm clearance), two wrist cameras + proprioception.
**Methods under test:** `hilserl` (2-critic RLPD baseline) vs `quality` (K=10-critic ensemble + quality-prioritised replay, **imitation loss off**, `bc_loss_weight=0`).
**Operator:** synthetic **geometric** oracle (`SimPegPhaseOracle`), skill-tiered (expert / intermediate / novice). One policy-oracle variant is also prepared.
**Purpose.** A complete, thesis-ready record of the RQ2 line of work: (a) what the intervention-quality score is, how it is stored and used; (b) the HIL-SERL two-buffer architecture; (c) every code capability added this session; (d) the campaigns, configs and results; (e) the "negligible score" diagnosis and its fix; (f) the methodological critique; and (g) **every question asked during the session with its answer** (Section 12). Companion to `rq1_oracle_setup_and_qa_log.md` (which established that the *baseline* depends on the operator). This document is the RQ2 *method + engineering + Q&A* reference.

---

## 0. TL;DR

- **RQ2 question:** does scoring operator interventions by their **counterfactual advantage** and using that score to **prioritise replay** (and nothing else — `bc_loss_weight=0`) improve the policy over the plain HIL-SERL baseline?
- **First full run (`rq2_quality_replayonly`, 18 cells, 3 tiers × 3 seeds):** `quality` did **not** help — mean autonomous SR **80.2% → 59.3%**, ICC **0.037 → 0.568**. Paired sign test (9 matched cells): quality won 3, lost 6, mean Δ **−21.0 pp**, **p = 0.508 (not significant)**.
- **Root cause of the null:** the quality signal was **effectively inert**. Raw `q_int` (a Q-value advantage) is ~1e-3, while the replay-priority formula's constant floor (`per_delta_init = 1.0`) is ~1.05, so the max sampling boost was **1.03×** (median 1.00×). The run therefore mostly measured the **2-critic vs 10-critic architecture change**, not the quality score.
- **Fix applied:** standardise `q_int` by its running scale before it prices the priority (`per_standardize`), plus a small `per_delta_init = 0.1`. Measured on the real `q` distribution this lifts a positive intervention to **2.4× (median) … 28× (max)** the baseline priority — the score now actually moves sampling.
- **Faster follow-up (`rq2_quality_replayonly_ei`, 12 cells):** expert + intermediate only (novice dropped for speed), both methods, **with the fix**. In progress.
- **Key caveat (unresolved by choice):** the comparison is `hilserl` (2-critic) vs `quality` (10-critic ensemble + bootstrap + per-member targets **+** quality replay). The `logged` same-architecture control was deliberately dropped, so a result cannot be cleanly attributed to the quality score alone.

| campaign | tiers × seeds | methods | quality signal | headline |
|---|---|---|---|---|
| `rq2_quality_replayonly` | 3 × 3 (18) | hilserl, quality | **inert** (≤1.03×) | quality worse (80→59% SR), paired p=0.51 |
| `rq2_quality_replayonly_ei` | 2 × 3 (12) | hilserl, quality | **fixed** (2.4–32×) | quality worse (82→49% SR); intermediate collapses 73→6% (Section 15) |
| `rq2_quality_replayonly_policyoracle` | 3 × 3 (18) | hilserl, quality | (as configured) | prepared, not run |
| `rq2_quality_replayonly_ei_warmup` | 2 × 3 (12) | hilserl, quality | fixed + warm-up + EMA + refresh + IS(β=0.5) | collapse FIXED (int 6%→82%); quality still −13pp vs baseline, paired p=0.031; flatter operator spread (Section 18) |

---

## 1. RQ2 definition and endpoints

**RQ2.** Given the RQ1 finding that the baseline depends on the operator, does **quality-aware intervention replay** (counterfactual-advantage score → replay priority, `bc_loss_weight=0`) improve the trained policy's **autonomous success rate (SR)** relative to plain HIL-SERL, and/or reduce its operator-dependence?

**Endpoints (per method, same instrument as RQ1):**
1. Mean autonomous SR on the frozen ±16 mm ruler (9 poses, yaw 0, sha256 `d0721170`, deterministic argmax, no operator).
2. Operator-dependency (ICC, permutation `p(operator matters)`, tier gradient).
3. Steps-to-convergence.
4. **Paired per-cell delta** `SR_quality − SR_hilserl` (added this session — see Section 6.3), which the shared operator×seed grid makes valid.

---

## 2. Capabilities added this session (code)

All changes are backward-compatible (new flags/fields default to the legacy behaviour). Files: `examples/train_aqi_rlpd.py`, `examples/run_campaign.py`, `serl_launcher/serl_launcher/aqi/config.py`, `serl_launcher/serl_launcher/data/data_store.py`, `examples/q_summary.py` (new).

1. **`--bc_loss_weight` override** (train + campaign). Sets λ_BC. `0.0` makes the quality score drive **only** the replay priority (no weighted imitation term). Quality-gated (ignored in `hilserl`).
2. **Segment-filter control via config/CLI** (`--quality_filters`). The RQ2 runs use `safety_veto,jerk_veto` (both inert no-ops in this sim) and **omit `outcome_filter`** so the policy still learns from failed-episode interventions.
3. **`--aqi_log_dir`** (train + campaign). Routes the AQI JSONL logs to `runs/<campaign>/aqi_logs/<cell>_actor/` so each campaign's `q_int` logs are colocated and never mix across campaigns (previously they landed in a shared `examples/aqi_logs/` keyed only by cell id).
4. **`examples/q_summary.py`** (new, wired into the `analyze` stage). Reads the per-segment logs and writes per-episode `q_int` CSVs + per-cell plots (`<cell>_q.png`: q per episode with p10–p90 band, per-step scatter, success/fail markers, and a per-step histogram) + aggregate `q_by_tier.png` + `summary.{md,json}` under `runs/<campaign>/report/q_analysis/`.
5. **`--training_oracle {auto,geometric,policy}`** (train + campaign). `policy` forces the frozen `--oracle_checkpoint` policy to provide the **intervention action** even in the sim (which otherwise always uses its attached geometric oracle), while the geometric monitor still **times** the takeovers. Enables the policy-oracle variant.
6. **Quality-priority fix `per_standardize`** (config field + buffer logic + `--per_standardize/--per_delta_init/--per_eta/--per_eps` flags). Divides `q_int` by its running positive-`q` scale before the priority so `per_eta` is dimension-free. Raw `q_int` is still logged unchanged.
7. **Paired analysis `compute_paired`** (campaign report). Adds a "Paired per-cell comparison" section (`SR_method − SR_baseline` per matched operator×seed + exact two-sided sign test) to `dependency.{md,json}`.

---

## 3. The HIL-SERL two-buffer architecture (RLPD)

HIL-SERL keeps **two** replay buffers and builds every training batch as **half from each**:

| buffer | agentlace name | contents | sampling |
|---|---|---|---|
| **online / replay** | `actor_env` | every actor step — autonomous policy steps **and** human-takeover steps | uniform |
| **demo / intervention** | `actor_env_intvn` | pre-recorded demos (loaded once) + human-intervention steps | uniform (`hilserl`) / **priority** (`quality`) |

```
batch = concat( sample(online, N/2), sample(demo_intervention, N/2) )
```

**Why split:** the online buffer floods with autonomous data; a fixed 50/50 mix guarantees the small pool of demo/human data is always well represented (RLPD "symmetric sampling").

**Key routing fact:** an intervention step lives in **both** buffers — inserted into the online buffer as ordinary RL experience (with the human action) *and* routed to the demo/intervention buffer as teaching data. Autonomous steps live **only** in the online buffer. Demos live **only** in the demo/intervention buffer.

In `hilserl` both buffers are uniform. In `quality` **only the demo/intervention buffer** becomes priority-sampled; the online buffer stays uniform.

---

## 4. The intervention-quality score (counterfactual advantage)

**Definition (S5.5, Option B):**
$$\hat A(s,a_H) = \mu_Q(s,a_H) - \mu_Q(s,a_{\text{robot}}),\qquad q_{\text{signed}} = \hat A \cdot g(\sigma_Q),\qquad q_{\text{int}} = \max(q_{\text{signed}},0)$$
- `μQ` = mean over the K critics; `a_H` = operator action, `a_robot` = policy action at the same state.
- disagreement gate `g(σQ) = 1/(1+(σQ/σref)²)` down-weights where the ensemble disagrees.
- `clip(·,0,∞)` keeps only interventions the critic judged *better* than the policy.

**Where it is computed:** per intervention segment at **episode end** (`AQIRuntime.end_episode` → `counterfactual_advantage_batch_safe`), then attached per transition.

**How it is stored:**
- Per-transition arrays in the replay schema: `q_int`, `quality_weight`, `intervention_mask` (autonomous/demo transitions carry zeros). Allocated by `MemoryEfficientReplayBuffer` when `include_aqi_fields=True`.
- Per-segment JSONL log `runs/<campaign>/aqi_logs/<cell>_actor/segments.jsonl`: `q_int` (per-step list), `q_int_mean`, `w_mean`, `a_hat`, `weights`, `episode`, `episode_success`, `excluded`, `exclusion_reason`, Option-A features, etc.

**How it is used (RQ2, `bc_loss_weight=0`):** ONLY the replay priority of the demo/intervention buffer:
$$p_i = (\varepsilon + |\delta_i| + \eta\cdot\max(q_{\text{int},i},0))^{\alpha}$$
It never enters the loss (λ_BC=0), and it never touches the uniform online buffer.

**Effect on non-intervention transitions:**
- **Autonomous transitions:** unaffected — only in the uniform online buffer; `q_int` unused there.
- **Demos:** stay at the **baseline** priority (`q_int=0`); their absolute priority is unchanged, but their *relative* sampling probability drops slightly as good interventions get boosted (same 50% half).
- **Observations themselves are never modified** — only sampling frequency changes.

---

## 5. Campaigns and configurations

Common settings (all three): cuboid, geometric operator, 1 jittered operator/tier (operator == tier, `sample_seed 0`), 6 filtered demos, **no demo-pretrain**, env-step budget 50 000/cell, convergence-stop (`--converge_success 0.9 --converge_window 20 --converge_min_steps 3000 --converge_eval_threshold 0.6`, 5 ruler-aligned poses), `--synthetic_intervention_mode signal`, `--converge_require_autonomous` (CLI, not a config field), eval `best_window` k=3 on `dx ∈ {−16…16} mm`, yaw 0, `--jax_platform cuda --wandb off`. `bc_loss_weight: 0.0`, `quality_filters: [safety_veto, jerk_veto]` (outcome filter OFF).

### 5.1 `rq2_quality_replayonly.yaml` (18 cells) — the first full run
- tiers {expert, intermediate, novice} × 3 seeds × {hilserl, quality}.
- Quality-priority left at defaults (`per_delta_init=1.0`, no standardisation) → **inert signal** (this was the flaw).

### 5.2 `rq2_quality_replayonly_ei.yaml` (12 cells) — faster, WITH the fix
- tiers {expert, intermediate} × 3 seeds × {hilserl, quality} (novice dropped: it never converged and ran the full cap, so it was the main time sink).
- Adds the fix: `per_standardize: true`, `per_delta_init: 0.1`, `per_eps: 0.05`, `per_eta: 1.0`.

### 5.3 `rq2_quality_replayonly_policyoracle.yaml` (prepared) — policy-oracle variant
- `policy_oracle.training_oracle: policy` → interventions from a **frozen policy oracle** (geometric monitor still times takeovers).
- `policy_oracle.demo_source`: `synthetic` (geometric demos) or `policy` — a free choice.
- Default oracle checkpoint (the one behind `runs/po_rq1_campaign/report/dependency.md`): `runs/rq1_3seed_nopretrain/checkpoints/hilserl_exp00_s0/certified` (`checkpoint_22026`, SR 1.00 on the cuboid ruler). Only its **actor** is used, so a 2-critic `hilserl` oracle loads fine into the 10-critic `quality` agent.
- Header documents how to **train/validate/pick** a policy oracle (validate on the correct ruler first; an un-validated checkpoint gave a 0.07 student in RQ1).

**Launch pattern:**
```
nohup ./.venv/bin/python examples/run_campaign.py \
  --config examples/campaigns/<config>.yaml \
  --campaign runs/<campaign> --converge_require_autonomous \
  > runs/<campaign>.out 2>&1 &
```

---

## 6. Results

### 6.1 `rq2_quality_replayonly` (18 cells, inert quality signal)

| tier | `hilserl` SR | `quality` SR |
|---|---|---|
| expert | 91.4% | 98.8% |
| intermediate | 90.1% | 56.8% |
| novice | 59.3% | 22.2% |
| **mean** | **80.2%** | **59.3%** |
| ICC | 0.037 | 0.568 |
| p(operator matters) | 0.535 | 0.081 |
| converged | 6/9 | 5/9 |
| mean steps-to-converge | 27 154 | 35 636 |

`quality` lowered mean SR and *increased* operator-dependence — but see the caveat (Section 5.1 + Section 10): the signal was inert, so this is largely the 10-critic architecture, not the score.

### 6.2 `q_int` and replay-priority ranges (measured, 234 606 intervention steps)
- `q_int`: min 0, **median 0**, mean 5.4e-4, p90 1.5e-3, p99 7.2e-3, **max 0.061**; **35%** of intervention steps have `q_int>0` (65% clipped to 0).
- **Priority (as-run, `per_delta_init=1.0`):** baseline (any `q=0` transition) = 1.0297; interventions 1.0297–1.0651 → **boost median 1.00×, max 1.03×** (inert).
- **Priority (with the fix, `per_standardize` + `per_delta_init=0.1`):** boost **median 2.4×, p90 5.3×, p99 11.5×, max 28×** over baseline.

### 6.3 Paired per-cell comparison (`rq2_quality_replayonly`)

| method vs `hilserl` | pairs | quality wins | hilserl wins | mean Δ | median Δ | sign-test p |
|---|---|---|---|---|---|---|
| `quality` | 9 | 3 | 6 | −21.0 pp | −3.7 pp | **0.508** |

Per-cell deltas (SR percentage points): exp00 +22.2 / −3.7 / +3.7; int00 −66.7 / −3.7 / −29.6; nov00 −70.4 / −70.4 / +29.6. The "quality is worse" impression is driven by a few catastrophic cells, and the sign test is **not significant** at n=9.

### 6.4 `rq2_quality_replayonly_ei` (with the fix)
Completed — see **Section 15** for the full result and the key within-quality
isolation (inert vs active priority). Headline: the fix made the priority active
(realized boost median 2.4×, max 32×), and with an active signal `quality` still
lost to the baseline (mean 82.1% → 48.8%), driven by an intermediate-tier
collapse (72.8% → 6.2%).

---

## 7. The "negligible score" diagnosis and fix

**Why negligible (two stacked causes):**
1. **Tiny signal.** `q_int` is a Q-value advantage in sparse-reward units (~1e-3), further shrunk by the disagreement gate and clipped (65% → 0).
2. **Swamped formula.** With `use_td_priority=False`, `|δ_i|` is frozen at `per_delta_init=1.0`, so `η·q ≈ 1e-3` is lost inside `(1.05 + q)^0.6`. Root cause: `q_int` was fed **raw** into the priority (the imitation-weight path already standardises it, the priority path did not).

**Fix (`per_standardize`):** in `QualityAwareReplayBufferDataStore`, maintain a running mean of positive `q_int` and use `q/scale` in `_priority`, so `per_eta` acts on a ~unit signal. Config: `per_standardize: true`, `per_delta_init: 0.1`, `per_eps: 0.05`, `per_eta: 1.0`. Raw `q_int` unchanged in logs.

**Known limitations of the fix (documented for honesty — Section 10):** cumulative (non-forgetting) scale, priorities not recomputed as the scale drifts, and **no importance-sampling correction** for the prioritised sampling bias.

---

## 8. Segment filters (S5.5)

Binary keep/drop layers applied to each intervention segment before it enters the intervention buffer (they do **not** change the graded `q_int`):
- `safety_veto` — hard force/torque limit breached (inert in this sim).
- `jerk_veto` — operator action jerk above `jerk_hard` (inert; `jerk_hard=None`).
- `outcome_filter` — drop interventions from **failed** episodes (Option G).
- `unsolicited` — drop non-requested takeovers (only when the active gate is on + `unsolicited_policy=rl_only`).

**RQ2 choice:** `outcome_filter` is **OFF** — the policy must still learn from failed-episode interventions. (Its intended use is during demo *collection*, not online training.)

**Inspecting/disabling:** the effective set is in `meta.json`; per-segment exclusions are logged as `excluded`/`exclusion_reason` in `segments.jsonl`. Disable all with `--quality_filters=none`; select a subset with `--quality_filters=safety_veto,jerk_veto`. Filters are ignored entirely in `hilserl`/`logged`/`active` modes.

---

## 9. Oracles and hardware

- **Geometric oracle** (`SimPegPhaseOracle`): analytic, ground-truth geometry, **no checkpoint**; the sim attaches it as `env.synthetic_oracle` and the trainer prefers it. All RQ2 runs use it.
- **Policy oracle** (`PolicyOracle`): a frozen trained policy acting on the same pixels; used in RQ1's `po_rq1_campaign`. **Important discovery:** `po_rq1_campaign` used the policy oracle for **demos only** — its training interventions were still geometric (all 9 actor logs said "geometric" despite `--oracle_checkpoint` being passed, because the sim's attached oracle wins). The new `--training_oracle policy` switch is what forces the intervention *action* to be the policy oracle (geometric monitor still times takeovers).
- **SpaceMouse:** **not needed** for any of these runs. Synthetic runs set `SIM_DISABLE_SPACEMOUSE=1`, so the `SpacemouseIntervention` wrapper is skipped; interventions come from the synthetic-operator wrapper and eval is autonomous. Hardware is only needed for real human teleoperation (`record_demos.py` / a live human operator).

---

## 10. Holes / problems with the approach (critique)

Ordered by threat to a clean conclusion:

**A. Confounded comparison (biggest).** `hilserl` (2-critic) vs `quality` (10-critic + bootstrap + per-member targets + quality replay) changes ≥3 things at once. The dropped `logged` arm (10-critic, uniform replay) was the control that isolates the score. As-is, a result cannot be attributed to the quality priority.

**B. Prioritised replay without importance-sampling correction.** No β/IS weight; the aggressive fixed priorities (2–28×) bias the value target. Strong prioritisation without IS can itself hurt.

**C. Non-stationary / order-dependent priority scale.** The running `q`-scale is cumulative (never forgets), drifts with the critic, and depends on async insertion order → priorities not comparable across time or bit-reproducible across seeds; stored priorities are never recomputed as the scale evolves.

**D. The counterfactual signal is critic-derived and circular.** Early critics are noise; the critics being trained produce the priority that selects their own training data (feedback loop). `A_hat` compares to a single sampled policy action (high variance). Clipping negatives means harmful interventions can't be pushed below baseline; combined with `outcome_filter` off, the priority can up-weight mis-scored interventions from failed rollouts.

**E. Low statistical power.** 1 operator/tier (operator == tier), 3 seeds; with 0↔100% SR oscillation, seed noise likely dominates any quality effect. `best_window` is an optimistic max whose bias may differ by method.

**F. Unequal training via convergence-stop + uncontrolled intervention volume.** Methods stop at different steps and get different intervention counts, changing buffer composition independently of the priority.

**Remedies:** re-add `logged` (or baseline against it); add IS correction + EMA scale + periodic priority refresh; report fixed-checkpoint SR at equal steps; ≥5 seeds and ≥3 operators/tier; use the paired design (now implemented).

---

## 11. Paired design

The cell grid is `method × operator × seed`; the roster (jittered operator θ), the demos (per operator×seed, reused across methods), and `--seed/--env_seed/--synthetic_seed` are all **matched across methods**. So the design is **paired on the controllable inputs** — only `--aqi_mode` (+ quality knobs) differs.

Caveats: (1) it is paired on *inputs*, not realised trajectories (different policies diverge from step 1, so reset sequences and interventions differ); (2) the analysis previously ignored the pairing — now a **paired per-cell delta + sign test** (Section 6.3) exploits it, which is the right move at this small n.

---

## 12. Complete session Q&A log

Faithful summaries of every question asked and its answer.

### 12.1 "How is the quality score stored, and what about its value?"
`q_int` is the counterfactual advantage `μQ(s,a_H) − μQ(s,a_robot)`, gated and clipped. Stored per-transition (`q_int/quality_weight/intervention_mask`) in the replay schema and per-segment in `segments.jsonl`. It drives replay priority in the intervention buffer and (if λ_BC>0) a weighted imitation loss. Value is nonnegative, ~1e-3 typically.

### 12.2 "If I set λ to zero, the score is just used to increase replay probability, right?"
Yes — with `bc_loss_weight=0` the score only sets replay priority. **Caveat:** segment filters still decide which interventions enter the buffer at all, so keep filters fixed across arms for a clean priority-only ablation.

### 12.3 "What are the segment filters and what are they for?"
`safety_veto`, `jerk_veto`, `outcome_filter`, `unsolicited` — binary keep/drop layers around the graded score (Section 8).

### 12.4 "How do I know which filters are active and how do I deactivate them?"
Startup set in `meta.json`; per-segment `excluded/exclusion_reason` in `segments.jsonl`. Disable via `--quality_filters=none` or a subset; ignored outside quality/aqi modes.

### 12.5 "Run a campaign comparable to RQ1 (geometric), quality score as replay-priority only, outcome filter on. Understood?"
Built `rq2_quality_replayonly.yaml`; added the `--bc_loss_weight` override so `quality` runs replay-priority-only; included `logged` as the same-architecture control; documented the confound.

### 12.6 "The outcome filter should be OFF (the policy learns from failure) — remove it."
Removed `outcome_filter`; `quality_filters = safety_veto,jerk_veto` (both inert). Confirmed via dry-run.

### 12.7 "Do a small smoke test (hil-serl vs quality) first."
Ran a 2-cell smoke; both methods trained + evaluated cleanly; the quality path (10-critic + counterfactual + quality replay + λ_BC=0) ran with the exact intended overrides.

### 12.8 "Log the q score and give per-episode summaries/plots."
Added `--aqi_log_dir` (per-campaign logs) + `q_summary.py` (per-episode CSVs, per-cell plots, `q_by_tier.png`, summary) wired into `analyze`.

### 12.9 "Yes — launch the campaign."
Launched; confirmed running (demos → train).

### 12.10 "What was the oracle, and why did an expert not converge?"
Geometric oracle. `hilserl_exp00_s2` didn't converge because convergence requires ≥90% **autonomous** success over 20 episodes + a mini-eval; a diligent expert keeps intervening (disqualifying episodes) and RL oscillation kept it at a 0.85 peak — seed variance, not low skill; still eval-scored.

### 12.11 "Why 27 cells (should be 18)?"
Because `logged` was included (3 methods × 3 tiers × 3 seeds = 27).

### 12.12 "Drop `logged`."
Dropped it; restarted as an 18-cell run; completed `hilserl` cells skipped via `.done`, no `logged` cell trained.

### 12.13 "Add a policy-oracle .yaml: policy oracle for training, choice of geometric/policy demos, requires the checkpoint dir; document the default path + how to train/pick."
Discovered the sim always uses the geometric oracle for interventions; added `--training_oracle policy`; created `rq2_quality_replayonly_policyoracle.yaml` with the default certified checkpoint and how-to.

### 12.14 "Is the campaign stuck?"
No — actively on a cell (~10 it/s); novice cells run the full cap because they don't converge.

### 12.15 "What happened — VS Code shut down?"
The machine slept/shut down ~08:00 and killed the `nohup` process (VS Code closing was collateral). 12/18 cells were saved; resumed to finish the remaining 6.

### 12.16 "Evaluate the results; explain the counterfactual advantage, the probability range, the buffer transitions, and how to read the plots."
`quality` worse (80→59%), ICC up; `q_int` 0–0.061 (median 0, 35% positive); priority baseline 1.0297, max 1.0651 (≤1.03× boost); buffers per Section 3–4; plots per `q_summary` (Section 2.4).

### 12.17 "Don't add `logged`. Why were the scores negligible, how to fix, and what's the replay probability of non-intervention transitions?"
Negligible = tiny `q` swamped by the `|δ|=1.0` floor + not standardised (Section 7). Fix = standardise `q` + small `per_delta_init`. Non-intervention: autonomous = uniform/online only (untouched); demos = baseline priority (Section 4).

### 12.18 "For faster results, drop novice; run expert vs intermediate, both methods."
Created `rq2_quality_replayonly_ei.yaml` (12 cells).

### 12.19 "Apply the fix, then run it."
Implemented `per_standardize` (config + buffer + flags + campaign wiring); measured boost 2.4–28×; launched the 12-cell run.

### 12.20 "Why 24 cells?"
Not 24 cells — 24 is the flag count (12 cells × 2 processes, learner+actor).

### 12.21 "Does the SpaceMouse need to be connected?"
No — synthetic runs set `SIM_DISABLE_SPACEMOUSE=1`; hardware only for real human teleop.

### 12.22 "Explain the buffer architecture in detail and simply, and whether the score affects demos/autonomous transitions."
Section 3–4: two buffers, 50/50; score prioritises only the intervention buffer; autonomous untouched; demos stay at baseline priority.

### 12.23 "Any holes or problems with this approach?"
Section 10 critique (confound, no IS correction, non-stationary scale, weak/circular signal, low power, unequal training).

### 12.24 "Is it a paired design (same operators/seeds, only method differs)?"
Yes — paired on inputs; caveats that trajectories diverge and the analysis wasn't exploiting it.

### 12.25 "Do it (add the paired per-cell delta + sign test)."
Implemented `compute_paired` + report section; regenerated the 18-cell report (quality 3 wins / 6 losses, mean −21 pp, sign p=0.508).

### 12.26 "Export everything to a thesis-basis document."
This document.

---

## 13. Open items & recommendations

1. **Re-introduce `logged`** (or baseline against it) so the quality effect is separable from the ensemble change — the single most important fix for interpretability.
2. **Make the prioritised replay sound:** add importance-sampling (β) correction; switch the `q`-scale to an EMA; periodically refresh stored priorities.
3. **Report fixed-checkpoint SR at equal steps** alongside `best_window` to guard against selection + unequal-training bias.
4. **Raise power:** ≥5 seeds and ≥3 operators/tier before any "quality helps" claim.
5. **Finish `rq2_quality_replayonly_ei`** (with the fix) and re-run `analyze` to get the paired section; then decide whether the fixed signal changes the picture on expert/intermediate.
6. **Keep:** `bc_loss_weight=0` for the replay-only question, `outcome_filter` OFF, per-campaign `aqi_logs`, the `q_summary` plots, and the paired sign test.

---

## 14. Key file / artifact locations

- Configs: `examples/campaigns/rq2_quality_replayonly.yaml`, `..._ei.yaml`, `..._policyoracle.yaml`.
- Trainer/runner: `examples/train_aqi_rlpd.py`, `examples/run_campaign.py`.
- Quality core: `serl_launcher/serl_launcher/aqi/quality.py`, `.../aqi/config.py`, `.../aqi/uncertainty.py` (`counterfactual_advantage_batch`), `.../data/data_store.py` (`QualityAwareReplayBufferDataStore`).
- q-score analysis: `examples/q_summary.py`; outputs under `runs/<campaign>/report/q_analysis/`.
- Reports: `runs/<campaign>/report/dependency.{md,json}` (now incl. the paired section) + `report/plots/`.
- Per-cell q logs: `runs/<campaign>/aqi_logs/<cell>_actor/{segments,episodes,steps}.jsonl`, `meta.json`.
- Default policy oracle: `runs/rq1_3seed_nopretrain/checkpoints/hilserl_exp00_s0/certified`.

---

## 15. Update (2026-09-17) — `rq2_quality_replayonly_ei` results (fixed signal)

All 12 cells finished; the report was regenerated (`--stages analyze`) so it
carries the paired section.

### 15.1 The fix bit (priority is now active)
Realized replay boost measured from this run's own logs (per_standardize on,
`per_delta_init=0.1`): **median 2.4×, p90 5.1×, max 32×** over a `q=0` baseline
transition (vs 1.00×/1.03× in the inert 18-cell run). So this run genuinely
tested quality-prioritised replay switched ON. `q_int`: 137 817 intervention
steps, 34% positive, median 0, mean 5.8e-4, max 0.082.

### 15.2 Results

| tier | `hilserl` SR | `quality` SR (fixed) |
|---|---|---|
| expert | 91.4% | 91.4% |
| intermediate | 72.8% | **6.2%** |
| **mean** | **82.1%** | **48.8%** |
| ICC | 0.302 | 0.976 |
| converged | 6/6 | 3/6 (expert 3/3, **intermediate 0/3**) |
| mean steps-to-converge | 29 077 | 31 120 |

**Paired (6 matched cells):** quality won 2, lost 4, mean Δ **−33.3 pp**, median
**−37.0 pp**, sign-test **p = 0.688**. Per cell: expert −22.2 / +3.7 / +18.5
(≈ neutral), **intermediate −92.6 / −55.6 / −51.9 (collapse on all 3 seeds)**.

### 15.3 The key comparison: turning the signal ON made it worse
Holding the **architecture fixed** (10-critic in both) and the operators/seeds
fixed, only the priority changed (inert → active):

| tier | quality, inert priority (18-cell) | quality, active priority (ei) |
|---|---|---|
| expert | 98.8% | 91.4% |
| intermediate | 56.8% | **6.2%** |

This is a cleaner isolation than quality-vs-hilserl (no architecture confound):
**activating the quality-prioritised replay hurt** — modestly at expert,
catastrophically at intermediate.

### 15.4 Interpretation
1. It is **no longer a "weak signal" story** — the fix made it bite, and biting
   hurt, concentrated at intermediate skill.
2. **Likely mechanism:** aggressive prioritisation of a noisy, critic-derived,
   circular signal **without importance-sampling correction**. The intermediate
   operator intervenes often with imperfect corrections; the priority
   over-replays a few high-`q` (per-critic-estimate) intermediate interventions,
   biasing the critic and destabilising exactly where interventions are
   frequent-but-mediocre. Expert interventions are rarer/cleaner → expert
   survives.
3. **Not statistically conclusive** (n = 6 pairs, p = 0.69) and quality-vs-
   hilserl still carries the 2-vs-10-critic confound — but the direction is
   consistent (4/6 losses; intermediate collapse and 0/3 convergence on all
   seeds), and the within-quality isolation points at the priority mechanism,
   not just noise.
4. **Reportable finding:** making the score "work" revealed a **failure mode**,
   not a benefit.

---

## 16. Evaluation of the disagreement gate g(σ_Q)

The quality score multiplies the counterfactual advantage by
`g(σ_Q) = 1 / (1 + (σ_Q / σ_ref)²)`, intended to suppress the score where the
critic ensemble disagrees. Verified behaviour in code:

- `σ_ref` is the **running mean of σ_Q** (`QualityScorer._sigma_scale`, updated
  every control step via `observe_sigma`); `quality_sigma_ref` (absolute
  override) is `None` by default.
- Therefore at a state with *average* disagreement, `g = 1/(1+1) = 0.5`; the
  gate only pushes states above/below the **current running-mean** disagreement.

**Is it gating properly? Partly — but it is RELATIVE, not a reliability gate.**
Because `σ_ref` tracks the running mean, it rises and falls *with* the noise.
Early in training the whole ensemble is unreliable and every σ_Q is large — but
σ_ref is large too, so `g` stays ≈ 0.5. **The gate cannot tell that the critics
are *globally* immature; it only flags states that are *relatively* more
uncertain than the current average.** So it does not down-weight the score when
the entire critic is untrustworthy.

**Does it allow a warm-up until the ensemble is reliable? No.** There is **no
warm-up on the quality path.** The active *query* gate has
`query_warmup_steps = 1000` (no requests until the z-stats are seeded), but the
*quality scorer* scores and prioritises interventions **from episode 1**, on
untrained critics. Combined with `per_standardize` (whose running q-scale is
itself polluted by early noise), the early replay priority is driven by noise —
exactly the regime that plausibly drove the intermediate collapse in Section 15.

**How to evaluate the gate (concrete diagnostics):**
1. Log the **distribution of `g(σ_Q)`** over training. If it sits ≈ 0.5 and is
   roughly constant, the gate is doing little beyond relative reweighting.
2. Log **σ_Q (mean/percentiles) vs step** on an absolute scale — does it
   actually shrink as critics mature? If not, the "disagreement" signal is not
   tracking reliability.
3. Compute **AUROC(`q_int`, post-intervention success) EARLY vs LATE** in
   training (aqi_analysis already computes this AUROC). Early ≈ 0.5 and late
   > 0.5 would prove the early score is noise → a warm-up is warranted.
4. **Ablations:** `quality_disagreement_gate` on/off; and relative `σ_ref` vs an
   **absolute** `quality_sigma_ref` (fixed value), to see whether an absolute
   reference gates early noise better.

**Recommended changes to the gate/warm-up:**
- Add a **quality warm-up** (`quality_warmup_steps`): give interventions only the
  **baseline** priority (like demos, `q_int→0` contribution) until the ensemble
  is trained enough — by step count, or until σ_Q (absolute) drops below a
  threshold, or until the critic's `predicted_qs/target_qs` gap stabilises. This
  directly targets the Section 15 failure mode and is cheap.
- Make the gate (optionally) **absolute** — reference σ_Q to a fixed scale or add
  a reliability multiplier that is small while σ_Q is globally high — so early
  global unreliability actually suppresses the score.
- Pair either with **importance-sampling correction** so that whatever the gate
  lets through is not over-fit by the aggressive priority.

---

## 17. Consolidated suggestions / next steps (updated)

Ordered by expected impact on making the RQ2 claim defensible:

1. **Baseline against `logged`** (10-critic, uniform replay), not `hilserl`, to
   remove the 2-vs-10-critic confound. (The within-quality inert-vs-active
   comparison already isolates the priority and shows it hurts — worth featuring.)
2. **Add importance-sampling (β) correction** to the prioritised buffer; the
   aggressive 2.4–32× priorities currently bias the value target uncorrected.
3. **Add a quality warm-up + reliability-aware gate** (Section 16): no q-priority
   until the critic is calibrated; consider an absolute σ_ref.
4. **EMA q-scale + periodic priority refresh** instead of the cumulative,
   never-forgetting scale, so priorities are stationary and comparable.
5. **Report fixed-checkpoint SR at equal steps** alongside `best_window` to guard
   against selection + unequal-training bias.
6. **Raise power:** ≥ 5 seeds and ≥ 3 operators/tier before any "quality
   helps/hurts" claim; the paired sign test needs more pairs than 6.
7. **Thesis framing:** report the inert run (mechanism swamped), the fixed run
   (active priority hurts, intermediate collapse), and the within-quality
   isolation as the clean evidence — then the gate/IS/warm-up as the principled
   remedies whose evaluation is future work.

---

## 18. Update (2026-09-18) — `rq2_quality_replayonly_ei_warmup`: the remedies applied

The three principled remedies from Sections 16-17 were implemented and run
together on the expert+intermediate design (12 cells), this time logged to
**wandb online** (project `hil-serl`, group `rq2_quality_replayonly_ei_warmup`).

### 18.1 What was implemented
- **Quality warm-up** (`quality_warmup_steps=5000`): interventions get BASELINE
  replay priority (q_int→0 contribution, zero BC weight) until the critic is
  past its random-init phase; raw q_int still logged.
- **EMA scale** (`per_scale_decay=0.01`) for the q-standardization, replacing the
  cumulative never-forget mean, so the scale stays stationary.
- **Periodic priority refresh** (`per_refresh_period=2000`): recompute all stored
  priorities against the current scale.
- **Importance-sampling correction** (`per_is_beta=0.5`): per-sample IS weights
  `w_i=(N·P(i))^{-β}/max` de-bias the critic target; the SAC critic loss applies
  them (ones/absent → exact baseline loss). All gated (defaults off).
- **Bug fixed (caught by a 1-cell smoke):** under wandb online, wandb creates
  `<ckpt>/wandb/` early, so the learner's `start_step` hit
  `latest_checkpoint(...)==None` → `basename(None)` crash; guarded. Would have
  crashed every cell under wandb online.
- wandb group set via the `WANDB_RUN_GROUP` env var (the logger already reads it).

### 18.2 Results

| tier | `hilserl` SR | `quality` SR (warm-up+IS) |
|---|---|---|
| expert | 97.5% | 82.7% |
| intermediate | 92.6% | 81.5% |
| **mean** | **95.1%** | **82.1%** |
| converged | 6/6 | 5/6 |
| operator spread (CV) | 0.037 | **0.011** |

**Paired (6 matched cells):** quality won **0/6**, mean Δ **−13.0 pp**, median
**−7.4 pp**, sign-test **p = 0.031**. Per cell: exp00 −3.7 / −33.3 / −7.4;
int00 −18.5 / −7.4 / −7.4 (all negative).

### 18.3 Two outcomes — one good, one not

**(a) The remedies fixed the catastrophic failure.** Holding the 10-critic
architecture fixed (within-quality, no-warm-up → warm-up+IS):

| quality, by tier | no warm-up (Section 15) | warm-up + IS |
|---|---|---|
| expert | 91.4% | 82.7% |
| **intermediate** | **6.2% (0/3 converged)** | **81.5% (3/3 converged)** |
| mean | 48.8% | **82.1%** |

The intermediate collapse is gone (6%→82%, now converges on all seeds). The
warm-up + EMA + refresh + IS did exactly their job: removed the failure mode and
made the quality arm stable and competent (slightly costing expert, 91→83%).

**(b) Quality still does not beat the baseline.** Even fixed, `quality` is
consistently below `hilserl` — all 6 cells negative, ~13 pp mean, paired
p = 0.031. Quality-prioritised replay (as implemented) does not improve
autonomous SR; it modestly but reliably underperforms.

**Operator-robustness (thesis goal):** `quality` is markedly more operator-robust
— CV 0.011 vs hilserl 0.037, expert≈intermediate (82.7 vs 81.5). It *flattens*
the skill gradient (the objective) at the cost of ~13 pp peak SR.

### 18.4 Caveats
- **Architecture confound stands** (2-critic hilserl vs 10-critic quality); the
  `logged` control is still the missing piece for a causal "quality helps" claim.
- **n=3 noise is large.** The `hilserl` baseline itself swung **82.1% (Section 15)
  → 95.1% (here)** on *identical* config — pure RL/seed non-determinism. The one
  big loss (exp00_s1, −33 pp, quality 59.3% and non-converged) is likely such an
  oscillating seed. So p=0.031 is suggestive, not definitive.

### 18.5 Bottom line
The quality-implementation fixes **worked**: they turned a catastrophic failure
(intermediate 6%, non-converging) into a **stable, operator-flattening** method
(intermediate 82%, converges). But quality-prioritised replay **still does not
outperform** plain HIL-SERL on success rate (~13 pp lower, consistent, paired
p=0.031). Honest framing: the remedies trade a modest amount of peak SR for
stability + operator-robustness — not a net SR win.

Artifacts: `runs/rq2_quality_replayonly_ei_warmup/report/dependency.{md,json}`
(with paired section), `report/plots/intervention_curves.png` (method-split;
the quality·intermediate panel now decays instead of staying pinned high),
`report/q_analysis/`, and wandb group `rq2_quality_replayonly_ei_warmup`.
