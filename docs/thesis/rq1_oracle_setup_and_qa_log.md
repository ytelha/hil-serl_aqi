# RQ1 Oracle Setup & Session Q&A Log — Geometric vs Policy Oracle

**Period covered:** 2026-09-08 → 2026-09-10
**Task:** MuJoCo `sim_peg_insertion`, **cuboid** peg (5 mm clearance), two wrist cameras + proprioception.
**Algorithm under test:** plain HIL-SERL (`hilserl` — 2-critic RLPD, no AQI components).
**Purpose.** A complete, thesis-ready record of (a) how the **geometric** synthetic operator and the **policy** oracle are set up and used, (b) the measurement bug that had hidden the RQ1 signal and its fix, (c) the corrected results for **both** oracle families, and (d) **every question asked during the working session, with its answer** (Section 8). Companion to `rq1_campaign_engineering_log.md` (the week-by-week failure log); this document is the *resolution* and the *oracle-setup reference*.

---

## 0. TL;DR

- The week-long "RQ1 has no signal (ICC ≈ 0)" conclusion was **substantially a measurement bug**: `eval_ruler.py` always scored on a **cylinder** peg regardless of the trained shape, so every **cuboid** policy was evaluated on the wrong geometry.
- After a one-line-root-cause fix (`--peg_shape`), RQ1 is answered on cuboid by **both** operator families:

| campaign | operator | ICC(1,1) | p(operator matters) | expert / int / novice | verdict |
|---|---|---|---|---|---|
| `rq1_3seed_nopretrain` | **geometric** | **0.604** | **0.035** | 1.00 / 0.82 / 0.59 | ✅ clean pass |
| `rq1_s075_5seed` | geometric (5 seeds) | 0.301 | 0.100 | 0.92 / 0.92 / 0.70 | ◐ direction, underpowered |
| `rq1_3seed_pretrain` | geometric + demo-pretrain | 0.000 | 0.453 | 0.95 / 0.82 / 0.77 | ✗ pretrain flattens |
| `po_rq1_campaign` | **policy oracle** | **0.992** | **0.004** | 1.00 / 0.86 / **0.06** | ✅✅ strongest |

- **Both oracles work; the policy oracle gives the strongest, most human-realistic signal** (novice's wrong-intent takeovers wreck training → 6 % SR and never converges). Demo-pretraining independently destroys the signal (real effect, survives the fix).

---

## 1. RQ1 definition and endpoints (unchanged)

**RQ1.** Does plain HIL-SERL produce a policy whose **autonomous success rate (SR)** depends systematically on *which operator* (expert / intermediate / novice) supervised training?

**Endpoints (all three required, per method):**
1. **ICC(1,1)** (operator-as-random-effect) — high.
2. **`operator_matters` permutation test** — p < 0.05.
3. **Monotone skill gradient** — expert > intermediate > novice.
Secondary axis: **steps-to-convergence ICC** (does training *cost* depend on the operator?).

**Fixed measurement instrument.** A frozen, hash-locked "ruler": 9 poses, lateral offset dx ∈ {−16…+16} mm, yaw = 0, sha256 `d0721170`, scored autonomously (no operator), deterministic argmax.

---

## 2. The measurement bug that hid the signal

### 2.1 The bug
`eval_ruler.py` built its evaluation environment from `CONFIG_MAPPING["sim_peg_insertion"]()`, whose class default is `peg_shape = "cylinder"` (`examples/experiments/sim_peg_insertion/config.py:52`). It had **no `--peg_shape` flag and did not read the shape from the frozen grid**. Training used `--peg_shape cuboid`; evaluation therefore **silently ran on cylinder** — an easier, geometrically different task (8 mm clearance and a round bore vs a 5 mm square bore).

### 2.2 Scope — geometric campaigns were affected too
Confirmed from each `campaign.json`: **every** `rq1_*` / `probe_*` / `cuboid_*` campaign was `peg_shape = cuboid`, and `eval_ruler` always used cylinder. Only the `cylinder_physoracle_*` family (train = eval = cylinder) was scored correctly by accident. So the mismatch affected **all cuboid campaigns, geometric and policy-oracle alike** — it predates the policy-oracle work; it was not new.

### 2.3 The fix
- `eval_ruler.py`: added `--peg_shape` (with an `eval_grid.json['peg_shape']` fallback) and it now prints `[ruler] eval peg_shape = …`.
- `run_campaign.py`: passes `--peg_shape args.peg_shape` to the eval subprocess.
Files: `examples/eval_ruler.py`, `examples/run_campaign.py`.

### 2.4 Impact — re-scoring the same checkpoints on the correct ruler
`rq1_3seed_nopretrain` (geometric, 3 tiers × 3 seeds), *only* the eval peg shape changed:

| tier | buggy eval (cylinder) | **corrected eval (cuboid)** |
|---|---|---|
| expert | 0.30 / 0.52 / 0.74 → **0.52** | 1.00 / 1.00 / 1.00 → **1.00** (SD 0) |
| intermediate | 0.19 / 0.00 / 0.67 → **0.28** | 0.89 / 0.67 / 0.89 → **0.82** |
| novice | 0.33 / 0.37 / 0.48 → **0.39** | 0.74 / 0.33 / 0.70 → **0.59** |

Buggy → non-monotone scatter, **ICC ≈ 0**. Corrected → monotone **100 / 82 / 59**, **ICC = 0.604**, **p = 0.035**. Original buggy evals are preserved alongside each cell as `cell.json.CYLBUG`.

---

## 3. The GEOMETRIC oracle — setup and use

### 3.1 What it is
`SimPegPhaseOracle` (`examples/experiments/sim_peg_insertion/wrapper.py`) — an analytic phase controller that computes corrections from **ground-truth peg/hole geometry** (peg-tip / hole-top / hole-bottom positions + the pinch quaternion). It is an *option-A* oracle: it needs **no checkpoint**. It commands translation (+ small yaw); on cuboid the reset yaw is small (±~4.6°) so it works.

### 3.2 How skill tiers are applied
The operator model (`serl_launcher/serl_launcher/aqi/operator_model.py`) wraps the oracle with a perception→decision→motor pipeline degraded by a skill scalar **s ∈ [0,1]** (`TIER_S = {novice:0.0, intermediate:0.5, expert:1.0}`). `theta_from_scale(s)` interpolates every parameter (motor noise σ₀, perception noise, gate thresholds δ_in, miss rate p_miss, false-alarm ρ_fa, …) between the **novice anchor** (s=0) and the **expert anchor** (s=1). The expert anchor is deliberately **sub-superhuman** (σ₀ ≈ 0.63): the original σ₀ = 0.25 expert *free-rode* (finished the insertion itself → the policy never learned the contact phase → low autonomous SR).

### 3.3 How it's run (demos + interventions)
- **Demos:** `collect_synthetic_demos.py` via `make_synthetic_sim_env` — the geometric operator teleoperates and records training-format demos (fast; pure numpy, no checkpoint).
- **Interventions:** the sim actor attaches the geometric oracle for execution + a geometric monitoring signal for takeover timing; `--synthetic_operator <tier> --synthetic_theta <op>.json` selects the skill.
- **Orchestration:** `run_campaign.py` runs one distributed learner + actor per cell (`method × operator × seed`), then the ruler eval, then the ICC/ANOVA report.

### 3.4 Result (corrected ruler)
`rq1_3seed_nopretrain`: **ICC 0.604, p 0.035, gradient 100 / 82 / 59** — all three RQ1 endpoints met. Report: `runs/rq1_3seed_nopretrain/report/dependency.md`.

---

## 4. The POLICY oracle — setup and use

### 4.1 What it is and why
`PolicyOracle` (`serl_launcher/serl_launcher/aqi/operator_model.py`) — a thin wrapper `policy_fn(obs) → action` around a **frozen trained policy** that acts on the **same wrist-camera pixels** the learner sees (an *option-B*, pixel-consistent oracle). It was introduced (engineering-log Phase 8) on the hypothesis that the geometric oracle's *observation privilege* destabilised training. That hypothesis turned out to be **not** the real blocker (the eval bug was) — but the policy oracle is still valuable: it is a more human-realistic operator model, and it gives the strongest RQ1 signal.

### 4.2 The critical setup lesson: use a *validated* oracle checkpoint
The first policy-oracle attempt produced a **do-nothing student (0.07 on the correct cuboid ruler)**. Root cause: the oracle was an **arbitrary mid-training checkpoint** (`hilserl_exp00_s2 @ step 11000`) that was itself weak on cuboid. **Fix:** use a checkpoint *verified competent on the correct ruler*. We used a **certified** expert checkpoint —

- `runs/rq1_3seed_nopretrain/checkpoints/hilserl_exp00_s0/certified` (`checkpoint_22026`), which had passed the convergence mini-eval and **scored SR = 1.00 on the cuboid ±16 mm ruler** (verified directly before use).

**Lesson:** the policy oracle is only as good as the checkpoint behind it — always validate the oracle on the *correct* ruler first.

### 4.3 How skill tiers are applied
The same tier scalar **s** degrades the oracle, but for a pixel policy the degradation is **temporally-correlated wrong-intent segments + motor noise** (coherent stretches where the operator does the *wrong* thing), rather than per-step Gaussian noise on a P-controller. This is a more realistic model of a bad human takeover.

### 4.4 How it's run
- **Demos:** `collect_policy_synthetic_demos.py` — the tier-degraded policy oracle records demos. (Slower than geometric: it loads the checkpoint + runs ResNet-10 inference per step; `run_campaign` forces these on CPU, which is the main slowdown — see Q&A 8.16.)
- **Interventions:** the actor loads the oracle via `--oracle_checkpoint <certified dir>`; the tier is applied on top.
- **run_campaign support added this session:** `--demo_source policy`, `--policy_oracle_checkpoint`, `--policy_oracle_mode`, `--policy_oracle_step`, `--policy_mm_per_unit` (demos use the policy generator; the actor gets `--oracle_checkpoint`).

### 4.5 Result (corrected ruler, mirrors the geometric campaign exactly)
`po_rq1_campaign` (cuboid, 3 tiers × 3 seeds, same 6 demos / convergence-stop / best_window / ±16 mm ruler as `rq1_3seed_nopretrain` — only the oracle source differs):

- **ICC 0.992, p 0.004, gradient 100 % / 86.4 % / 6.2 %** (σ²_between 0.256, σ²_within **0.002**).
- **Steps-to-convergence:** expert 3/3 @ ~32.4 k, intermediate 3/3 @ ~38.7 k, **novice 0/3 (never converged)** — the training-cost axis shows dependence too.
- Smoke that preceded it: 1 expert cell → best_window **0.926**, converged at step 21902. Report: `runs/po_rq1_campaign/report/dependency.md`.

---

## 5. Head-to-head — geometric vs policy oracle

| | geometric (`rq1_3seed_nopretrain`) | **policy oracle (`po_rq1_campaign`)** |
|---|---|---|
| ICC(1,1) | 0.604 | **0.992** |
| p(operator matters) | 0.035 | **0.004** |
| expert | 100 % | 100 % |
| intermediate | 82 % | 86 % |
| novice | 59 % | **6 %** (never converges) |
| within-tier variance | moderate | **tiny (0.002)** |
| operator realism | noisy ground-truth P-controller | pixel-consistent, wrong-intent takeovers |

**Interpretation.** Both refute "no dependence." The policy oracle widens the gradient because a *novice* pixel operator giving temporally-correlated wrong-intent takeovers genuinely destroys the student, whereas a geometric novice (a noisy but ground-truth-correct P-controller) is more benign. For the thesis, the geometric result is the clean minimal demonstration; the policy-oracle result is the stronger, more human-analogous one.

---

## 6. Cross-cutting mechanics (referenced throughout)

### 6.1 Convergence criterion (still present, used by both)
`train_aqi_rlpd.py:521` — `converge_on = FLAGS.converge_success > 0.0` (flag-controlled, not hardcoded off). The actor keeps a rolling window of the last 20 episodes; an episode counts only if it **succeeded AND was autonomous** (no operator takeover). When rolling autonomous success ≥ `--converge_success` (0.9) and `step ≥ converge_min_steps`, it runs a **fixed-pose mini-eval** (`--converge_eval_threshold 0.6` on 5 ruler poses); if that passes it saves the exact weights to `<ckpt>/certified/`, writes `CONVERGED.json`, and **stops the actor early**. Three uses: (1) actor early-stop; (2) `run_campaign` `.done` marker with `converged_step`; (3) **steps-to-convergence ICC** in the report. *Note:* the mini-eval runs in the actor's own (cuboid, correct) env, so it reported high SR while the buggy ruler reported low — this "converged but scores low" mismatch was another symptom of the eval bug.

- Geometric campaigns: all cells converged, all have `certified/`.
- Policy-oracle campaign: expert + intermediate converged; **novice never converged** (consistent with 6 % SR).

### 6.2 Which checkpoint stage is evaluated
The learner saves a checkpoint **every 1000 steps** (`checkpoint_period = 1000`). `run_campaign` evaluates with `--select best_window --select_window_k 3 --noprefer_certified`: it scores **all** 1000-step checkpoints on the ruler and reports the **best window of 3 consecutive checkpoints** (pooled over their 27 trials) — the **peak of *sustained* competence**, *not* the last checkpoint and *not* the `certified/` snapshot. Rationale: the policy oscillates across checkpoints, so the last one is a random phase; `best_window` de-noises without the upward bias of best-of-1.

### 6.3 The skill scalar `s`
`s ∈ [0,1]`; tiers fixed at novice 0 / intermediate 0.5 / expert 1.0. Higher s = more skilled. **Increasing s is not a fix for a weak between-tier gap** — pushing the expert toward superhuman re-introduces *free-riding* (the expert finishes the task itself → the student learns less → expert-trained SR *drops*). The real lever for a stronger/cleaner signal is **more operators per tier + more seeds** (reduce the operator-mean noise), not a larger s.

### 6.4 `p(operator matters)` (permutation test)
`permutation_sigma2` (`serl_launcher/serl_launcher/aqi/stats.py`): statistic = between-operator variance = `var(ddof=1)` of the per-operator mean SRs. Shuffle the operator labels across cells **B = 10 000** times, recompute the statistic, and `p = (#{perm ≥ observed} + 1) / (B + 1)`. Small p ⇒ the observed tier separation is unlikely by chance ⇒ operators matter. Complementary to ICC (effect size) vs p (significance).

---

## 7. Reproduction — exact commands & settings

**Common settings (both oracle campaigns):** cuboid, `hilserl`, 3 tiers × 3 seeds (1 op/tier), 6 filtered demos, `--steps 50000`, convergence-stop (`--converge_success 0.9 --converge_window 20 --converge_min_steps 500 --converge_eval_threshold 0.6 --converge_eval_poses="-0.016,0.0;-0.008,0.0;0.0,0.0;0.008,0.0;0.016,0.0"`), `--synthetic_intervention_mode signal`, `--eval_select best_window --eval_window_k 3`, `--eval_dx_mm="-16,-12,-8,-4,0,4,8,12,16" --eval_yaw 0`, `--jax_platform cuda --wandb off`. (`--converge_eval_poses` and `--eval_dx_mm` need the `=` form because of the leading minus.)

**Geometric campaign** (baseline): omit the policy flags; demos come from the geometric operator by default.

**Policy-oracle campaign** — add:
```
--demo_source policy \
--policy_oracle_checkpoint runs/rq1_3seed_nopretrain/checkpoints/hilserl_exp00_s0/certified \
--policy_oracle_mode hilserl --policy_oracle_step 0 --policy_mm_per_unit 100
```

**Verify an oracle checkpoint before use** (crucial):
```
SIM_DISABLE_SPACEMOUSE=1 python eval_ruler.py --exp_name sim_peg_insertion \
  --peg_shape cuboid --checkpoint_path <cell_dir> --grid_file <eval_grid.json> \
  --out /tmp/oracle_verify --prefer_certified --select pooled --no_realtime --argmax
# expect SR = 1.00 on the cuboid ruler
```

**Regenerate a corrected report from existing checkpoints** (analyze-only, CPU):
```
python run_campaign.py --campaign <dir> --stages analyze --peg_shape cuboid \
  --methods hilserl --tiers expert,intermediate,novice --ops_per_tier 1 --seeds 3 \
  --sample_seed 0 --baseline hilserl --eval_select best_window --jax_platform cpu --wandb off
```
(`analyze` reads `cell.json["outcomes"]["sr"]`; the GPU pre-flight is skipped with `--jax_platform cpu`.)

**Key file locations**
- Fix: `examples/eval_ruler.py` (`--peg_shape`), `examples/run_campaign.py` (passes it + policy-oracle flags).
- Head-room knob (added, inert by default): `examples/experiments/sim_peg_insertion/config.py` `success_lateral` + `SIM_SUCCESS_LATERAL`.
- Reports: `runs/<campaign>/report/dependency.{md,json}` + `report/plots/`.
- Buggy originals preserved: `runs/<campaign>/eval_runs/<cell>/cell.json.CYLBUG`.

---

## 8. Complete session Q&A log

Every question asked during the working session, with a faithful summary of the answer. (Repeated "give me the monitor command" requests are consolidated at 8.M.)

### 8.1 "Do you have to train a policy from scratch, or can you use one of the trained ones that performs well?"
**No — reuse a trained checkpoint as the policy oracle.** The recommendation was a strong existing expert checkpoint, with the caveat that many runs oscillate, so a smoke check is required before trusting it. (This later proved pivotal: the *specific* checkpoint matters — an un-validated one produced a 0.07 student; a validated certified one produced 0.93.)

### 8.2 "Do the smoke test and verify; pick the best policy with the appropriate settings and run the campaign yourself."
Wired policy-oracle support into `run_campaign` and ran an initial cuboid policy-oracle smoke + campaign. (These early results were later found to be confounded by the eval-shape bug and were superseded.)

### 8.3 "Show me the command to monitor; how was the smoke; are you optimistic?"
Provided a monitor command. Smoke ran end-to-end but scored best_window 0.0 — **cautiously optimistic on infrastructure, not yet on performance**, pending a longer run.

### 8.4 "Are you stupid? I want a copyable CLI monitor command. Also, if SR is zero that means there is a problem — how can you solve it? RETHINK EVERYTHING."
Built a robust monitor **script** (no quoting pitfalls). Diagnosed SR = 0 **with evidence**: the smoke had trained only ~1834 learner steps / 2 checkpoints (severely under-trained); the demos were genuine and skill-graded; the live campaign was already producing successes. Proposed the correct levers (train longer, `best_window` selection, a stronger oracle) rather than declaring failure.

### 8.5 "Call off the run."
Cleanly terminated the orchestrator and its learner/actor children; verified no stragglers. Noted the run is resumable (idempotent).

### 8.6 "Make sure the smoke works; take the best oracle; if needed relax the ruler to maximize SR; think about what can be improved (anchors, which checkpoints to evaluate…). I can train the oracle from scratch if needed."
Ran further smokes (v2: 30 k steps; v3: convergence-stop). Found oscillate-and-collapse and, chasing why the *certified* checkpoint evaluated to a do-nothing policy, **discovered the eval-shape bug** (see 8.7). (Also explored a cylinder + tight-`success_lateral` head-room branch, which was stable but saturated and became moot once the cuboid signal was recovered.)

### 8.7 "Did the environment mismatch happen with the geometric-oracle campaigns, or only now?"
**Both — all cuboid campaigns, geometric included.** The bug is in `eval_ruler` (no peg-shape handling), so every cuboid campaign was scored on cylinder; only the cylinder campaign matched by accident. This reframed the entire engineering log: the "instability / ICC 0" was largely an eval artifact.

### 8.8 "Do it." (re-score the geometric checkpoints on the correct cuboid ruler)
Re-scored `rq1_3seed_nopretrain` (and spot-checks) on the correct cuboid env. The geometric cuboid policies are **good** (best_window 0.63–1.0) with a **monotone gradient** — the RQ1 signal that had been hidden.

### 8.9 "Do both." (regenerate the official report + update the engineering log)
Installed the corrected evals and regenerated the report → **ICC 0.604, p 0.035, 100/82/59**; added the "⚑ CRITICAL CORRECTION" section to the engineering log. Also re-scored the 5-seed and pretrain campaigns.

### 8.10 "Is the solution to #2 to increase the s value? And what is s?"
`s` is the **operator skill scalar** (novice 0 → expert 1) that interpolates all operator noise/gate parameters between the novice and expert anchors. **Increasing s is not the fix and would backfire** (a superhuman expert *free-rides*, lowering the expert-trained SR). Verified that the 3-seed and 5-seed used the **same** anchors, so the 5-seed's weaker result is **sampling + RL noise with 1 operator/tier**, fixed by **more operators/seeds** — and corrected an earlier mistaken "the anchors compressed the tiers" claim.

### 8.11 "What happened to the policy oracle? What can you tell me about the results?"
At that point: it had been the initial pivot; the eval bug was found in the process; the early policy-oracle students scored 0.07 on the correct ruler — but that was traced to a **weak, non-validated oracle checkpoint**, not the method. The geometric oracle was fine once evaluated correctly. The policy-oracle tooling stayed available. (This directly motivated the later, *validated* policy-oracle campaign that succeeded — Section 4.)

### 8.12 "Is the convergence criterion still there? Do the geometric- and policy-oracle policies still have it? How is it used?"
Yes — flag-controlled and functional. All geometric cells converged (with `certified/` checkpoints); the early policy smokes mostly disabled it. Three uses: actor early-stop, orchestration `.done`, and the steps-to-convergence ICC. Flagged that the mini-eval runs on the *correct* cuboid env, so "converged but low ruler SR" was another symptom of the eval bug. (Full detail: Section 6.1.)

### 8.13 "Which checkpoints get evaluated? Which stage of the trained policy is scored?"
The **best sustained 3-consecutive-checkpoint window** across all 1000-step checkpoints (`--select best_window`, `--noprefer_certified`) — the peak of sustained competence, not the last checkpoint and not the certified one. (Full detail: Section 6.2.)

### 8.14 "Now test the policy oracle to answer RQ1: smoke first, then a campaign. Pick the best task and settings."
Chose **cuboid, mirroring the geometric baseline exactly** for a clean A/B (only the oracle source differs). **Validated the oracle** (`hilserl_exp00_s0/certified` = 1.00 on the cuboid ruler — fixing the earlier weak-checkpoint mistake), ran the smoke (best_window **0.926**, converged), then launched the full 9-cell campaign. Result: **ICC 0.992, p 0.004, 100/86/6** (Section 4.5).

### 8.15 "Is everything going well? Show me the correct monitor command."
Yes — demos progressing (4/9 at the time), no errors, no GPU contention. Extended the monitor script to show the demo stage as well as training, and gave the copyable `watch` command.

### 8.16 "Why is recording demos slow? Is that normal? What about a hybrid — geometric operator for demos, policy oracle for interventions?"
**Normal for the policy oracle:** per-job checkpoint load + ResNet-10 **CPU** inference per step + filtered retries (degraded tiers fail more). Geometric demos are far faster (analytic, no net). It's a one-time cost; the real time is training; it could be sped up by letting policy demos use the GPU. **The hybrid idea:** thoughtful and faster, but it makes the tier's "operator" a *chimera* (geometric hands for demos, policy hands for interventions) and mixes two expert action distributions — it muddies the clean "policy vs geometric" A/B. Recommendation: keep the primary comparison pure; use the hybrid deliberately as a **channel-decomposition ablation** (which channel — demos or interventions — carries the operator-dependence) or as a future speed mode.

### 8.M "Show me the (copyable) command to monitor" (asked several times)
Delivered self-contained monitor scripts per run (`runs/<campaign>/monitor.sh`) invoked as `watch -n 20 bash <path>/monitor.sh`. Each prints runner status, demos/trained/converged/evaluated counts, per-cell best_window as cells finish, the in-flight cell's live step, and (for the policy campaign) the geometric baseline for side-by-side comparison.

---

## 9. Open items & recommendations

1. **Tighten the CI:** run a **5-seed** or **multi-operator-per-tier** version of `po_rq1_campaign` (1 op/tier is the main statistical weakness of the geometric 5-seed run).
2. **Speed:** allow policy-oracle demo collection to use the GPU (currently forced to CPU — a geometric-era default), which is the main demo-time cost.
3. **Hybrid ablation (optional):** geometric demos + policy interventions (and vice versa) to attribute the operator-dependence to the demo vs intervention channel.
4. **Keep:** the `--peg_shape` fix, `best_window` selection, certified checkpoints, env-step budget, `wandb: off`, ≥ 3 seeds, no demo-pretraining, and **always validate the oracle checkpoint on the correct ruler**.
5. **Thesis framing:** present the eval-shape bug as the methodological lesson; report the geometric result (ICC 0.60) as the clean minimal RQ1 demonstration and the policy-oracle result (ICC 0.99) as the stronger, human-realistic one; use `rq1_3seed_pretrain` (ICC 0.00) as the demo-pretraining-flattens control.
