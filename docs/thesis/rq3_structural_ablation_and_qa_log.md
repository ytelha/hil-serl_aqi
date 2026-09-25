# RQ3 Structural Ablation (plain / logged / quality) — Results, Infra & Session Q&A Log

**Period covered:** 2026-09-20 → 2026-09-21
**Task:** MuJoCo `sim_peg_insertion`, **cuboid** peg (5 mm clearance), two wrist cameras + proprioception.
**Methods under test:** `hilserl` (2-critic RLPD baseline) vs `logged` (K=10-critic ensemble, **uniform** replay, no quality) vs `quality` (K=10-critic ensemble + **quality-prioritised** replay, **extended/stabilised**: warm-up + IS + EMA scale + refresh + standardise, `bc_loss_weight=0`).
**Operator:** synthetic **geometric** oracle (`SimPegPhaseOracle`), skill-tiered (expert / intermediate / novice), **2 jittered operators per tier**.
**Campaign:** `runs/rq3_plain_logged_quality` — 3 methods × 3 tiers × 2 operators × 2 seeds = **36 cells**. wandb **online**, project `hil-serl`, group `rq3_plain_logged_quality`.
**Purpose.** A complete, thesis-ready record of the RQ3 line: (a) why this ablation exists (it finally removes the 2-vs-10-critic confound that RQ2 was warned about); (b) the two clean contrasts it isolates; (c) the results (SR, ICC, gradient, convergence, paired deltas, q-signal); (d) the infrastructure incidents that nearly killed it (disk ENOSPC, `systemd-oomd`, kernel global OOM) and their fixes; (e) the code capabilities added this session; and (f) **every question asked during the session with its answer** (Section 12). Companion to `rq2_quality_replay_and_qa_log.md`.

---

## 0. TL;DR

- **RQ3 question:** separate the two changes that were confounded in the RQ2 `hilserl`-vs-`quality` comparison — the **structural** change (2 → 10 critics) and the **quality-replay** change — by running a same-operators/seeds three-arm ladder: `hilserl` (2-critic) → `logged` (10-critic, uniform) → `quality` (10-critic + prioritised).
- **Result is the OPPOSITE of the AQI thesis claim.** Plain HIL-SERL is both the **best-performing** and the **most operator-independent**. Both the ensemble and the quality replay make things worse.
- **Two clean contrasts:**
  - `hilserl → logged` (structural only): mean SR **86.7% → 68.2%** (−18.5 pp), ICC **0.027 → 0.743**. The 10-critic ensemble **alone** introduces strong operator-dependence.
  - `logged → quality` (quality replay at equal architecture): mean SR **68.2% → 50.9%** (−17.3 pp further), ICC **0.743 → 0.887**. Prioritised replay makes it worse still.
- **Paired sign test vs `hilserl`:** `logged` 4 wins / 6 losses / 2 ties, mean **−18.5 pp**, **p = 0.75** (a wash — fine on experts, catastrophic on a couple of int/novice cells). `quality` 1 win / **11 losses**, mean **−35.8 pp**, **p = 0.006** (reliably, significantly worse).
- **The `quality` arm was the EXTENDED/stabilised one** (warm-up 5000 + IS β=0.5 + EMA scale + refresh + standardise), i.e. the exact configuration that fixed the RQ2 intermediate collapse. The extensions removed the *instability*, but did **not** make the signal helpful: `q_int` is near-zero and non-discriminative (mean ≈ 0.0004–0.0009 across all tiers).
- **Not an artifact:** all 36 cells trained substantially (12k–22k+ steps, 120–420 episodes), no crashes in the scored runs, `best_window` eval (de-noises cuboid oscillation). The low novice/intermediate `quality` scores are genuine non-convergence (`converged=False`), not the OOM artifact that corrupted the earlier cylinder campaign.
- **Infra:** the run survived three separate infrastructure failures — disk full (ENOSPC), `systemd-oomd` killing VS Code, and the kernel global OOM killer — each diagnosed and fixed (Sections 15–17).

| contrast | change isolated | mean SR | ICC | verdict |
|---|---|---|---|---|
| `hilserl` (baseline) | — | 86.7% | 0.027 | best + most operator-independent |
| `hilserl → logged` | 2 → 10 critics, uniform replay | −18.5 pp | 0.027 → 0.743 | structure alone hurts + adds dependence |
| `logged → quality` | + quality-prioritised replay | −17.3 pp | 0.743 → 0.887 | quality replay hurts more (p = 0.006 vs base) |

---

## 1. RQ3 definition and endpoints

**RQ3.** RQ2 compared `hilserl` (2-critic) directly against `quality` (10-critic + bootstrap + per-member targets + quality replay), so a result could not be attributed to the quality score alone (Section 10 of the RQ2 log, hole **A**). RQ3 inserts the missing **same-architecture control** `logged` (10-critic, **uniform** replay, no quality) between them, so the ladder decomposes the effect:

- **`hilserl → logged`** isolates the **structural** (critic-count) effect on SR, operator-dependence and convergence.
- **`logged → quality`** isolates the **quality-replay** effect at **equal architecture** — the clean test the confound previously blocked.

**Endpoints (per method, same instrument as RQ1/RQ2):**
1. Mean autonomous SR on the frozen ±16 mm ruler (9 poses, yaw 0, sha256 `d0721170`, deterministic argmax, no operator).
2. Operator-dependency: ICC, `σ²_between` / `σ²_within`, operator spread (σ_op, CV, range), permutation `p(operator matters)`, tier gradient.
3. Steps-to-convergence + steps-ICC.
4. **Paired per-cell delta** `SR_method − SR_hilserl` (matched operator × seed) + exact-binomial sign test — computed for **both** `logged−hilserl` and `quality−hilserl`.

---

## 2. Capabilities added this session (code)

All changes are backward-compatible (new flags/fields default to legacy behaviour). Files: `examples/train_aqi_rlpd.py`, `examples/run_campaign.py`.

1. **`--buffer_period` override** (train + campaign). The actor pickles the online + demo replay transitions (WITH camera images) to `<ckpt>/buffer/` + `demo_buffer/` every `config.buffer_period` (sim = 1000) steps — **~9 GB/cell** over a full run, only ever used to warm-restart a crashed learner. New flag: `>=0` overrides the config, **`0` disables the dumps**, `<0` keeps the default. The campaign default is now `0`; applied in `main()` before `get_aqi_config`. Wired into `_actor_cmd` (the dumps happen on the actor). Root cause of the disk crash (Section 15).
2. **`--checkpoint_keep` override** (train + campaign). Replaces the four hard-coded `keep=100` in `save_checkpoint` (a 50k-step run at `checkpoint_period=1000` kept ~50 snapshots × 101 MB ≈ several GB/cell). New flag (train default 100 for back-compat; campaign default **25**). Wired into `_learner_cmd` (checkpoints happen on the learner). `best_window` eval still has ample checkpoints to select from.
3. **Campaign YAML `budget.buffer_period` / `budget.checkpoint_keep`** overlay (`apply_config`), so both live in the config.

Net effect: per-cell footprint **~15 GB → ~2.5 GB**; a 36-cell run ~540 GB → ~90 GB. Verified via `--dry_run`: 36× `--buffer_period 0` + `--checkpoint_keep 25`.

(The paired analysis `compute_paired` + report section and `q_summary.py` were added in the RQ2 session; RQ3 reuses them — the paired section now renders **two** contrasts because there are three methods.)

---

## 3. Architecture recap (what actually differs between the three arms)

All three arms share the identical two-buffer RLPD stack (online `actor_env` uniform + demo/intervention `actor_env_intvn`), the same operators/seeds/demos, `--synthetic_intervention_mode signal`, and `bc_loss_weight=0`. Only `--aqi_mode` (+ the quality knobs) differs:

| arm | critics (K) | target aggregation | replay of intervention buffer | quality signal |
|---|---|---|---|---|
| `hilserl` | 2 | min over 2 | **uniform** | off (inert) |
| `logged` | 10 | REDQ min over random subsample | **uniform** | off (inert) |
| `quality` | 10 | REDQ min over random subsample | **quality-prioritised** (extended) | on |

`quality_on` (quality/aqi) gates every `per_*`/`quality_*`/warm-up knob, so `hilserl` and `logged` **ignore** them — verified from the `aqi_logs` `meta.json`: a `logged` cell shows `per_standardize=false, per_is_beta=0.0, quality_warmup_steps=0, per_delta_init=1.0`, i.e. a clean 10-critic uniform-replay control. Only `quality` cells received the extended prioritisation (Section 4).

---

## 4. The `quality` arm was the EXTENDED (stabilised) configuration

Confirmed from the actual learner command line and per-cell manifest — every `quality` cell ran with:

```
--per_standardize            # standardise q_int by its running (EMA) scale
--per_delta_init 0.1         # small optimistic |delta| floor so q can matter
--per_eta 1.0  --per_eps 0.05
--quality_warmup_steps 5000  # baseline priority until the critic matures
--per_scale_decay 0.01       # EMA (forgetting) q-scale, not cumulative
--per_refresh_period 2000    # periodic recompute of stored priorities
--per_is_beta 0.5            # importance-sampling de-biasing
--bc_loss_weight 0.0         # replay-priority only, no imitation loss
```

i.e. **exactly** the four RQ2 remedies (warm-up + EMA + refresh + IS) plus `per_standardize` — the same stack that fixed the RQ2 intermediate collapse (`rq2_quality_replayonly_ei_warmup`, Section 18 of the RQ2 log). So RQ3 is a fair test of the *best* quality configuration to date, not the inert or unstable earlier ones.

---

## 5. Campaign configuration

`examples/campaigns/rq3_plain_logged_quality.yaml`:
- **operators:** 2 jittered operators per tier (expert/intermediate/novice), `jitter 0.15`, `sample_seed 0` → within-tier operator variance **and** seed variance are both estimable.
- **methods:** `[hilserl, logged, quality]`.
- **replicates:** 2 seeds/operator → 4 samples/tier/method; 36 cells total; 12 demo jobs (6 filtered geometric demos per operator×seed, reused across methods).
- **budget:** 50 000 env-step cap; `minutes_per_cell: 300` **safety** wall-clock cap only; `converge_success 0.9 / window 20 / min_steps 3000 / eval_threshold 0.6` (5 ruler-aligned poses); `--converge_require_autonomous` (CLI). `buffer_period: 0`, `checkpoint_keep: 25` (Section 2).
- **aqi:** the extended quality block (Section 4); `quality_filters: [safety_veto, jerk_veto]` (outcome filter OFF).
- **eval:** `best_window` k=3, `dx ∈ {−16…16} mm` (9 poses), yaw 0, grid sha256 `d0721170`.
- **logging:** wandb `online` (group via `WANDB_RUN_GROUP`).

**Launch (as a systemd service — see Section 17):**
```
WANDB_RUN_GROUP=rq3_plain_logged_quality \
systemd-run --user --unit=rq3_campaign --collect \
  -p ManagedOOMPreference=avoid -p WorkingDirectory=<repo> \
  -- bash -c './.venv/bin/python examples/run_campaign.py \
       --config examples/campaigns/rq3_plain_logged_quality.yaml \
       --campaign runs/rq3_plain_logged_quality \
       --converge_require_autonomous --wandb online \
       >> runs/rq3_plain_logged_quality.out 2>&1'
```

---

## 6. Results

### 6.1 Operator dependency per method (final autonomous SR)

| method | mean SR | ICC | σ²_between | σ²_within | ΔICC vs base | cells |
|---|---|---|---|---|---|---|
| `hilserl` (baseline) | **86.7%** | **0.027** | 0.001 | 0.027 | 0.000 | 12 |
| `logged` | 68.2% | 0.743 | 0.113 | 0.039 | −0.716 | 12 |
| `quality` | **50.9%** | **0.887** | 0.172 | 0.022 | −0.860 | 12 |

> **ΔICC sign note.** The report prints ΔICC as negative and labels it "increases operator-dependency by −0.716". ICC went **up** (0.027 → 0.743 → 0.887), so the negative sign means dependence got **worse** — the opposite of the goal. Higher ICC = more operator-dependent.

### 6.2 Operator spread + 'operator matters' test (B.3 / C.2)

| method | σ_op | CV | range | worst operator | p(operator matters) |
|---|---|---|---|---|---|
| `hilserl` | 0.119 | 0.137 | 0.333 | `nov01` (64.8%) | 0.396 |
| `logged` | 0.365 | 0.535 | 0.889 | `nov01` (9.3%) | 0.021 |
| `quality` | 0.427 | 0.839 | 0.907 | `nov01` (1.9%) | 0.007 |

Baseline: operators do **not** matter (p = 0.40, CV = 0.14). `logged` and `quality`: operators **do** matter (p = 0.02 / 0.007), with `quality` the most operator-sensitive (CV 0.84).

### 6.3 Skill gradient (mean SR per tier)

| method | expert | intermediate | novice |
|---|---|---|---|
| `hilserl` | 94.4% | 88.9% | 76.9% |
| `logged` | 97.2% | 61.1% | 46.3% |
| `quality` | 91.7% | 56.5% | **4.6%** |

Baseline is shallow and robust; `logged` steepens; `quality` is steepest and the **novice tier collapses** to ~5%.

### 6.4 Training effort: steps-to-convergence

| method | converged | mean steps | median | steps-ICC |
|---|---|---|---|---|
| `hilserl` | 12/12 (100%) | 30 812 | 28 656 | 0.129 |
| `logged` | 7/12 (58%) | 37 795 | 38 883 | 0.000 |
| `quality` | 7/12 (58%) | 32 630 | 29 149 | 0.698 |

Per tier (steps): `hilserl` 32797/29577/30062; `logged` 38842/37134/34930; `quality` 36366/27648/**n/a** (novice never converged). Only the baseline converges on every cell; both ensemble arms fail to converge on ~40% of cells, and `quality`'s convergence cost is itself strongly operator-dependent (steps-ICC 0.70).

### 6.5 Paired per-cell comparison (matched operator × seed)

| method vs `hilserl` | pairs | wins | losses | ties | mean d | median d | sign-test p |
|---|---|---|---|---|---|---|---|
| `logged` | 12 | 4 | 6 | 2 | −18.5 pp | −3.7 pp | 0.754 |
| `quality` | 12 | 1 | 11 | 0 | −35.8 pp | −18.5 pp | **0.006** |

`logged − hilserl` per cell (pp): exp00 +11.1/+3.7; exp01 −7.4/+3.7; int00 **−92.6**/+0.0; int01 +0.0/−18.5; nov00 +3.7/−14.8; nov01 −33.3/**−77.8**. → near-parity on experts, a couple of catastrophic int/novice cells drive the mean; not significant.

`quality − hilserl` per cell (pp): exp00 +11.1/−11.1; exp01 −3.7/−7.4; int00 **−88.9**/−18.5; int01 −3.7/−18.5; nov00 **−85.2/−77.8**; nov01 −40.7/**−85.2**. → 11/12 losses, consistent and significant.

### 6.6 Quality signal (`q_int`) — why it doesn't help

Per-tier episode-mean `q_int` (2036 quality episodes over 12 cells):

| tier | n episodes | mean `q_int` | q>0 fraction |
|---|---|---|---|
| expert | 645 | 0.0004 | 0.24–0.32 |
| intermediate | 439 | 0.0007 | 0.33–0.45 |
| novice | 874 | 0.0005 | 0.40–0.48 |

Two facts that explain the failure:
1. **The signal is near-zero and non-discriminative** — mean `q_int` ≈ 0.0004–0.0009 for **every** tier; the standardisation/EMA fix rescales it but cannot create information that isn't there.
2. **Low-skill operators intervene on nearly every episode** (q>0 frac 0.40–0.48 for novices vs 0.24 for experts). So quality-prioritised replay up-weights a **flood of frequent, low-value novice corrections**, and the policy leans on the operator crutch → cannot stand alone at eval (novice 4.6%).

---

## 7. Interpretation

- **Structural finding (`hilserl → logged`):** the 10-critic REDQ ensemble with min-Q targets is pessimistic and, undertrained on the harder cuboid, suppresses the risky "press-into-contact" action; this shows up as −18.5 pp SR and a jump to strong operator-dependence (ICC 0.74). The critic-count change alone — not quality — is responsible for a large part of what RQ2 attributed to "quality".
- **Quality finding (`logged → quality`):** adding quality-prioritised replay, even in its **extended/stabilised** form, costs another −17.3 pp and raises ICC to 0.89, significant vs baseline (paired p = 0.006). The extensions (warm-up/IS/EMA/refresh) removed the RQ2 *instability/collapse-to-noise*, but the underlying `q_int` is too small and skill-blind to price replay usefully.
- **Bottom line:** for cuboid peg insertion at this budget, **plain HIL-SERL (2 critics) is the best and most operator-independent method.** This is a legitimate, publishable structural/negative result; it redirects the next investigation from the *replay mechanism* to the **magnitude/quality of the `q_int` signal itself** (why the counterfactual advantage is ~1e-3 and skill-blind).

---

## 8. Holes / caveats (critique)

**A. Small n.** 2 operators/tier × 2 seeds = 12 cells/method. The paired sign test (p = 0.006 for quality) is robust, but per-tier means rest on 4 samples. (The report header's "5/tier" string is a stale template label; the run is 2/tier, 6 operators total.)

**B. Novice `quality` cells were budget-limited.** They hit the 300-min wall-clock safety cap at ~16–22k of 50k steps (novice cells are slow, ~1 s/it). But all were `converged=False` and far from learning, so more budget is unlikely to rescue them; treat novice-quality as "did not learn under a bounded budget".

**C. `best_window` optimism.** Eval selects the best window of 3 consecutive checkpoints — an optimistic max that could bias methods differently. Reported here for all three arms identically, and it *de-noises* the known cuboid 0↔100% oscillation, so it is the right instrument, but a fixed-checkpoint-at-equal-steps readout would be a stronger robustness check.

**D. Convergence-stop → unequal training.** Methods stop at different steps (baseline 12/12 converged, ensembles 7/12), so buffer composition and total updates differ across arms independently of the mechanism under test.

**E. Signal magnitude, not just replay.** The clean negative on `quality` is entangled with the near-zero `q_int`; if a future signal were larger/more discriminative, the replay mechanism might behave differently. RQ3 shows the *current* signal + replay hurts, not that prioritised replay is inherently bad.

---

## 9. Key file / artifact locations

- Config: `examples/campaigns/rq3_plain_logged_quality.yaml`.
- Report: `runs/rq3_plain_logged_quality/report/dependency.{md,json}` (with the two paired contrasts).
- Plots: `runs/rq3_plain_logged_quality/report/plots/` (`icc_by_method.png`, `sr_by_tier.png`, `sr_by_operator.png`, `variance_decomposition.png`, `operator_spread.png`, `steps_to_convergence.png`, `intervention_curves.png`).
- q-signal: `runs/rq3_plain_logged_quality/report/q_analysis/` (per-cell `*_q.png`, `q_by_tier.png`, `summary.{md,json}`).
- Per-cell logs: `runs/rq3_plain_logged_quality/logs/<cell>.{learner,actor,eval}.log`; provenance READMEs + `.done` in `checkpoints/<cell>/`.
- Trainer/runner: `examples/train_aqi_rlpd.py`, `examples/run_campaign.py`.

---

## 10. What RQ3 settles relative to RQ2

RQ2 closed with the caveat (its hole **A**) that `hilserl`-vs-`quality` mixed the architecture and the score. RQ3 resolves it:

- The **structural** change is **not** free — it accounts for roughly half of the total SR drop and *all* of the initial ICC jump (`hilserl → logged`: −18.5 pp, ICC 0.03 → 0.74).
- The **quality** change adds a further, **significant** degradation on top of the matched architecture (`logged → quality`: −17.3 pp, p = 0.006), so RQ2's "quality hurts" is confirmed even after removing the confound — but its magnitude is smaller than the raw `hilserl`-vs-`quality` gap implied, because part of that gap was structural.

---

## 11. Infrastructure incidents (this session) — summary

Three separate failures, each diagnosed and fixed; the run completed only after all three:

| # | failure | root cause | fix | Section |
|---|---|---|---|---|
| 1 | `OSError [Errno 28] No space left` | root disk 98% full; ~9 GB/cell replay dumps + `keep=100` checkpoints; 1.2 TB of old dumps across `runs/` | `--buffer_period 0`, `--checkpoint_keep 25`; deleted all `buffer/`+`demo_buffer/` dumps (freed 1.2 TB, kept checkpoints) | 15 |
| 2 | whole run vanished, no traceback | `systemd-oomd` killed the **VS Code cgroup scope** (21 procs) at 54–69% user-slice memory pressure; the `nohup` run lived inside it | relaunch as an **own** transient systemd user service (`systemd-run --user`), `ManagedOOMPreference=avoid` | 16 |
| 3 | learner killed, no traceback | **kernel global OOM** (RAM 30 GB + 8 GB swap exhausted); `avoid` does not cover the kernel path; the desktop biases session apps to `oom_score_adj=+100` and an unprivileged user manager cannot set a protective negative score | add swap (root, user-run) so global OOM cannot fire; the run then completed | 17 |

---

## 12. Complete session Q&A log

### 12.1 "Compare HIL-SERL (2 critics), HIL-SERL (logged, 10 critics) and quality HIL-SERL (10 critics) — evaluate the structural change and its impact on convergence and stability; wandb active; each tier ≥ 4 samples; include expert, intermediate, novice."
Built `rq3_plain_logged_quality.yaml` (3 methods × 3 tiers), with the extended quality arm gated to quality cells only so `logged` is a clean 10-critic uniform control. This is the `logged` control RQ2 was missing (Section 1, Section 10).

### 12.2 (clarifying) "3 operators, 3 seeds each" → then "reduce to 2 operators × 2 seeds/tier".
Initial answers implied 3 ops × 3 seeds = 81 cells (multi-day). On the scale-confirmation question the choice was **2 operators × 2 seeds/tier = 36 cells**. Config updated; verified 36 cells / 12 demo jobs via `--dry_run`.

### 12.3 "What is the problem with the storage and how can it be solved?"
The campaign crashed with `ENOSPC`: root disk 98% full. Per-cell ~9 GB replay `buffer/` dumps (images) + `keep=100` checkpoints; 1.2 TB of old dumps across `runs/`. Fix: `--buffer_period 0` + `--checkpoint_keep 25` (per-cell ~15 GB → ~2.5 GB) and delete just the `buffer/`+`demo_buffer/` dumps (keeps every checkpoint + result) → freed 1.2 TB, disk 30% used. 33/36 cells were already `.done` so the resume finished the last 3. (Section 15.)

### 12.4 "The same VS Code crash happened again? Solve it — it's crucial the run succeeds."
Two distinct events: (a) `systemd-oomd` killed the VS Code scope again (the run **survived** — the `ManagedOOMPreference=avoid` service isolation worked); (b) the **kernel global OOM** killer then killed the learner (RAM+swap exhausted). `avoid` and `OOMScoreAdjust=-900` do **not** cover the kernel path (negative scores need root; the session biased apps to +100). Definitive fix = add swap (root, user-run) + keep the run in its own service. After swap was added the run completed. (Sections 16–17.)

### 12.5 "Show me the command to monitor the run."
`tail -f runs/rq3_plain_logged_quality.out` (between-cell wrapper log); `tail -f runs/rq3_plain_logged_quality/logs/<cell>.actor.log` (live step bar); `systemctl --user status rq3_campaign.service`; `find runs/rq3_plain_logged_quality/checkpoints -maxdepth 2 -name .done | wc -l` (done/36); `watch -n5 free -h` + `journalctl --since "-15 min" | grep -iE "oom|killed process"`.

### 12.6 "Analyse the run and what can you tell me about its results."
36/36 cells finished cleanly. Headline: plain HIL-SERL best + most operator-independent; `logged` −18.5 pp / ICC 0.74; `quality` −35.8 pp (paired p = 0.006) / ICC 0.89. Verified the low `quality` scores are real non-convergence, not the OOM artifact. (Section 6–7.)

### 12.7 "The quality in this test was the extended one, with importance sampling and warm-up, right?"
Yes — confirmed from the learner command line and manifest: `--per_standardize --quality_warmup_steps 5000 --per_is_beta 0.5 --per_scale_decay 0.01 --per_refresh_period 2000 --per_delta_init 0.1 --bc_loss_weight 0.0`. The `logged` meta shows all knobs at inert defaults → isolation held. (Section 4.)

### 12.8 "What can you tell me about `delta` in the report? What does it measure?"
Two deltas: (a) `delta` in the paired per-cell tables = `SR_method − SR_hilserl` for a matched operator×seed cell, in SR percentage points (positive = method beat baseline); `mean d`/`median d` are their aggregates and the sign-test p is the exact-binomial chance of that many wins/losses. (b) `dICC vs base` = `ICC_method − ICC_hilserl`, the change in operator-dependence (printed negative here because ICC rose, i.e. dependence got worse). (Section 6.1, 6.5.)

### 12.9 "Add all of the new results and the questions asked in this session to a new .md in the same manner as the attached one."
This document.

### 12.10 "Why are `q_int` values small when translated into probabilities, and what can be done to rescale / mitigate it?"
Full analysis in **Section 18**: the signal is a tiny sparse-reward Q-difference (~1e-3), then clipped (65% zeros), gated (×0.5), buried under the additive priority floor, and α/IS-compressed. Rescaling options: per-scale standardization (already on), rank-based priority (scale-free), per-state normalized `A_hat`, softmax/temperature. Honest caveat: the distribution is **skill-blind** (mean q ≈ equal across tiers), so no monotone rescaling creates discrimination — the real fix is a more informative numerator (lower-variance `A_hat` vs the mean/argmax action; n-step/return-to-go advantage; absolute reliability gate).

---

## 13. Open items & recommendations

1. **Investigate the `q_int` magnitude** (~1e-3, skill-blind) — the mechanistic root cause. Why is the counterfactual advantage `μQ(a_H) − μQ(a_robot)` so small, and does it ever become discriminative late in training? (Compute AUROC(`q_int`, post-intervention success) early vs late.)
2. **Report fixed-checkpoint SR at equal steps** alongside `best_window` for all three arms, to remove selection + unequal-training bias (critique C, D).
3. **Raise power:** ≥ 3 operators/tier and ≥ 3 seeds; give novice cells the full 50k (or a larger wall-clock cap) so novice-`quality` is not budget-limited (critique A, B).
4. **Structural follow-up:** since the 10-critic ensemble alone hurts on cuboid, sweep K (2/4/10) and the REDQ subsample/pessimism to find where operator-dependence is introduced.
5. **Infra hardening (permanent):** persist the extra swap (`/etc/fstab`), keep launching campaigns as `systemd-run --user` services with `ManagedOOMPreference=avoid`, and keep `buffer_period=0` + a small `checkpoint_keep` as the campaign defaults.

---

## 14. Environment / reproduction notes

- Repo venv `.venv/bin/python` (3.12); JAX 0.11 CUDA12 on an 8 GB Blackwell GPU; learner XLA mem 0.6 / actor 0.2.
- Machine: 30 GiB RAM, originally 8 GiB swap (the OOM constraint — Section 17); 1.9 TB root disk.
- wandb online works via `~/.netrc` (the "WANDB_API_KEY not set" warning is spurious); group via `WANDB_RUN_GROUP`.
- `--converge_require_autonomous` is a CLI flag, **not** a config field.

---

## 15. Incident 1 — disk ENOSPC (root cause + fix)

**Symptom.** The first launch crashed writing a cell's `.done`: `OSError: [Errno 28] No space left on device`. Root disk 1.9 TB was **98% full (49 GB free)**; `runs/` had grown to **1.4 TB**.

**Root cause.** Per training cell: (a) `<ckpt>/buffer/` + `demo_buffer/` pickle the replay transitions **with camera images** every `config.buffer_period` (sim = 1000) steps → **~9 GB/cell**, used only to warm-restart a crashed learner; (b) checkpoints every 1000 steps with hard-coded `keep=100` → up to ~10 GB/cell. Across all past campaigns the `buffer/` dumps alone were **1.2 TB** (the bulk of `runs/`). 36 cells would need ~540 GB — impossible at 49 GB free.

**Fix.** (1) Code: `--buffer_period 0` (no dumps) + `--checkpoint_keep 25`, wired through the campaign (Section 2) → per-cell ~2.5 GB, 36-cell run ~90 GB. (2) Reclaim: `find runs -type d \( -name buffer -o -name demo_buffer \) -prune -exec rm -rf {} +` deleted **only** the replay dumps (kept every checkpoint + all reports) → freed **1.2 TB**, disk to **30% used**. 33/36 cells were already `.done`, so the resume skipped them and re-ran only the last three with the small footprint.

---

## 16. Incident 2 — `systemd-oomd` killed the VS Code scope

**Symptom.** After the disk fix the run resumed, reached cell 34, then the whole process tree vanished with **no Python traceback** and disk fine.

**Root cause.** `journalctl`: `systemd-oomd … Killed …/app-code-<pid>.scope due to memory pressure for user@1001.service being 69.38% > 50.00% for > 20s … killed 21 process(es) … result 'oom-kill'`. The campaign was launched from the VS Code integrated terminal, so its `nohup` wrapper + learner + actor all lived **inside VS Code's systemd cgroup scope** (`app-code-*.scope`); `systemd-oomd` killed the whole scope on memory pressure. `nohup` blocks only SIGHUP, not a cgroup-level oomd kill.

**Fix.** Relaunch the campaign as its **own** transient systemd **user service**, decoupled from VS Code, and oomd-deprioritised:
```
systemctl --user reset-failed rq3_campaign.service 2>/dev/null
systemd-run --user --unit=rq3_campaign --collect \
  -p ManagedOOMPreference=avoid -p WorkingDirectory=<repo> \
  -p Environment=WANDB_RUN_GROUP=rq3_plain_logged_quality \
  -- bash -c './.venv/bin/python examples/run_campaign.py ... >> runs/....out 2>&1'
```
Child learner/actor inherit the service cgroup (cgroup membership survives `setsid`/`start_new_session`), so `avoid` covers them and they are out of `app-code`. Verified `ManagedOOMPreference=avoid`; cgroup `/user.slice/.../app.slice/rq3_campaign.service`. This worked: at the next oomd event it killed **VS Code**, and the run **survived**.

---

## 17. Incident 3 — kernel global OOM killed the learner (the decisive one)

**Symptom.** The service died again; unit `Failed with result 'oom-kill'`, but this time no `systemd-oomd` line for our unit.

**Root cause.** `journalctl`/`dmesg`: `oom-kill:constraint=CONSTRAINT_NONE … global_oom … task_memcg=…/rq3_campaign.service, task=train_aqi_rlpd`. This is the **kernel** global OOM killer (RAM 30 GB + swap only 8 GB fully exhausted, mostly by VS Code/browser — the service itself peaked at only **3.7 GB**), which selects the largest single process by `oom_score`. Two aggravators, neither fixable from user space:
- `ManagedOOMPreference=avoid` and `OOMScoreAdjust` **only affect `systemd-oomd`, not the kernel OOM killer.**
- This desktop biases session apps to `oom_score_adj=+100` (preferred victims), and an **unprivileged** user manager cannot set a protective **negative** score — the `-900` we requested was clamped to `+100`, verified on `/proc/<pid>/oom_score_adj`. So the learner (largest single process, +100) was the kernel's first choice.

**Fix (needs root, user-run).** Add swap so RAM+swap cannot be exhausted for a ~4 GB run:
```
sudo fallocate -l 48G /swapfile2 || sudo dd if=/dev/zero of=/swapfile2 bs=1M count=49152
sudo chmod 600 /swapfile2 && sudo mkswap /swapfile2 && sudo swapon /swapfile2
# persist: echo '/swapfile2 none swap sw 0 0' | sudo tee -a /etc/fstab
```
No-root alternatives offered: raise the `oom_score_adj` of the heavy apps (browser/VS Code) so the kernel sacrifices *them* first (the run survives VS Code death — proven in Incident 2), and/or close the browser during the run. With swap added, global OOM could no longer fire and the campaign ran to completion (36/36, finished 04:22).

---

## 18. Why `q_int` maps to small probabilities — rescaling & mitigation

The code makes the answer precise. The signal starts small and then gets shrunk four more times before it becomes a sampling probability.

### 18.1 Why `q_int` → tiny probabilities

**1. The numerator is an intrinsically small Q-difference.**
In `uncertainty.py` (`counterfactual_advantage_batch`), `A_hat = μQ(s,a_H) − μQ(s,a_robot)` — the *same state*, two *similar* actions (the human just nudges the policy's action). In a sparse-reward task the Q-values themselves are small, and the **difference** between two nearby actions is smaller still. Measured: median 0, mean ≈ 5e-4, max ≈ 0.06–0.08.

**2. Clipping zeroes ~65% of it.** In `quality.py` (`score_segment`) `q_int = clip(q_signed, 0, ∞)` — every step where the critic didn't judge the human strictly better becomes exactly 0. So the *median* intervention contributes nothing.

**3. The disagreement gate halves the survivors.** `g(σQ)=1/(1+(σQ/σref)²)`; with `σref` = running mean of σQ, a typical state sits at `g=0.5`, so `q_signed = A_hat·g` is ~half of an already-tiny number.

**4. The priority floor buries it.** In `data_store.py` (`_priority`):
$$p_i = (\varepsilon + |\delta_i| + \eta\cdot q)^{\alpha}$$
With `use_td_priority=False`, `|δ|` is frozen at `per_delta_init`. Without standardization (`per_delta_init=1.0`): baseline `p≈(1.05)^{0.6}≈1.03`, and `η·q ≈ 1e-3` is lost inside it → **max boost 1.03×** (the RQ2 "inert" finding).

**5. `^α` (α=0.6) compresses whatever spread remains**, and then the **IS correction** (`w_i=(N·P(i))^{-β}`, β=0.5) deliberately *undoes* part of the boost in the loss. Both are correct, but both mute the net effect.

So "small q → small probability" is really: **small magnitude × 65% zeros × gate × additive-floor domination × α-compression.**

### 18.2 Rescaling options (fix the *units* problem)

**A. Per-scale standardization — already on.** `per_standardize` divides `q` by its running/EMA scale (`_q_scale` in `data_store.py`), and `per_delta_init=0.1` lowers the floor, so a *typical positive* q → ~1 and the boost becomes median **2.4×**, max **32×**. This is the direct rescale, and it's in the RQ3 run. Necessary, but see the caveat below.

**B. Rank-based priority (the robust rescale).** Replace `η·q` with a **rank** term — Schaul's rank-based PER: `p_i ∝ 1/rank_i` over the positive interventions. This is **scale-free by construction**: it doesn't matter whether q is 1e-3 or 1e-1, only the ordering survives, so the "tiny magnitude" problem disappears entirely and outliers can't dominate. This is the cleanest fix for the symptom.

**C. Per-state normalization of `A_hat` (fix the numerator's units).** Instead of a raw Q-difference, use a dimensionless advantage:
$$A_z(s) = \frac{\mu_Q(s,a_H) - \mu_Q(s,a_\text{robot})}{\text{std}_a\,Q(s,\cdot)\ \text{or}\ |\mu_Q(s,a_\text{robot})|}$$
This makes each state's advantage comparable and O(1) regardless of the local Q scale — a principled rescale done *before* clipping/priority.

**D. Softmax / temperature map.** Price the priority as `exp(q̃/τ)` with τ from the running scale, giving explicit, bounded control over how much a given standardized q raises probability (with a max-boost clamp to avoid runaway).

### 18.3 The honest caveat (what rescaling can't fix)

RQ3 showed the deeper problem is **not** just scale: the `q_int` distribution is **skill-blind** — episode-mean q ≈ 0.0004 / 0.0007 / 0.0005 for expert / intermediate / novice, i.e. essentially the same everywhere. Any **monotone rescaling** (A–D) is a strictly-increasing function of q, so it **cannot create discrimination that isn't in the raw signal** — it only redistributes it. That's why the extended, standardized, IS-corrected quality arm still lost by −35.8 pp.

So the real mitigation is to make the **numerator informative**, not just larger:
- **Lower-variance `A_hat`:** compare `a_H` to the policy's **mean/argmax** action (or several sampled robot actions), not one noisy sample — the current single-sample `a_robot` adds variance that swamps a ~1e-3 gap.
- **Temporal credit:** the one-step Q-gap is near-zero in sparse reward; use an **n-step / return-to-go advantage** of the intervention so the signal reflects "did this takeover change the outcome," which *is* skill-dependent.
- **Reliability, not just relative gate:** the current `g` uses a *relative* σref (stays ≈0.5); an **absolute** σ_ref or a warm-up-until-calibrated reliability multiplier would stop early noise from setting the scale.

### 18.4 Recommendation

For a quick, defensible improvement that targets exactly the symptom, add **rank-based priority (B)** as a `per_rank` option (scale-free, robust) and log **AUROC(`q_int`, post-intervention success)** early-vs-late to test whether the signal is *ever* discriminative. If AUROC stays ≈0.5, the fix has to be the numerator (C / temporal credit), and no rescaling will rescue replay priority.

---
