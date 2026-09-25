# RQ4 Ensemble Trade-off & Convergence — Design Rationale, Parameters & Run Log

**Period covered:** 2026-09-21 →
**Task:** MuJoCo `sim_peg_insertion`, **cuboid** peg (5 mm clearance), two wrist cameras + proprioception.
**Campaign:** `runs/rq4_ensemble_convergence` — `hilserl` (plain 2-critic) vs `logged` (K=8 critic ensemble **with the "B" fixes**). 3 operators/tier × 3 tiers × 2 seeds × 2 methods = **36 cells**. wandb **online**, project `hil-serl`, group `rq4_ensemble_convergence`.
**Purpose.** A thesis-ready record of *why* each ensemble/optimisation parameter is set the way it is: the **update target** (REDQ min-over-M, per-member targets, bootstrapping), the **number of critics K**, the **learning rate**, and the **warm-start** — **what each was originally, why, what it was changed to for RQ4, and why**. Companion to `rq2_quality_replay_and_qa_log.md` and `rq3_structural_ablation_and_qa_log.md`.

---

## 0. TL;DR

- **Why this study exists.** RQ3 (`runs/rq3_plain_logged_quality`) showed that going from plain 2-critic HIL-SERL to the 10-critic ensemble (`logged`, uniform replay, **no** quality) **cost 18.5 pp of success rate**, jumped operator-dependence ICC **0.03 → 0.74**, and left **5/12 `logged` cells non-converged**. The degradation was traced to two things: (a) **REDQ min-Q pessimism** (the conservative target suppresses the risky "press-into-contact" action) causing genuine under-performance on the hard operators, and (b) a convergence-metric artifact (competent policies that never hit the 90% *autonomous* streak).
- **What RQ4 changes.** It applies the "B" **training** fixes to the ensemble arm and asks whether they recover convergence/SR toward plain HIL-SERL:
  - **Less-pessimistic target:** REDQ subsample **M: 2 → 1** (a single random critic per member → *no* min-Q pessimism).
  - **Fewer critics:** **K: 10 → 8** (mild; less compute, slightly less pessimistic pool, still a real ensemble).
  - **Warm start:** **demo-pretrain 0 → 5000** demo-only gradient steps (a cold ensemble is pessimistic from step 1).
  - **Lower LR:** **3e-4 → 1.5e-4** (damp the 0↔100% cuboid oscillation).
  - **Unchanged** (already on in RQ3): per-member targets (True), bootstrap keep-prob (0.8).
- **The baseline stays truly plain.** `hilserl` = 2-critic, **no** pretrain, default LR — the fixes land *only* on `logged` (via a new `method_overrides` mechanism). So "compare against plain HIL-SERL" is literal.
- **Two read-outs.** In-campaign `hilserl` vs fixed-`logged`, and fixed-`logged` vs **RQ3's** `logged` (K=10, M=2, no pretrain, LR 3e-4) to isolate the fix's effect.

---

## 1. RQ4 definition and endpoints

**RQ4.** Is the ensemble's operator-dependence / non-convergence (RQ3) an **intrinsic** cost of using a critic ensemble, or a **fixable** artifact of the *pessimistic REDQ target* + *cold start* + *high LR*? Concretely: do M=1 (no min-Q), a mild K reduction, demo warm-start, and a halved LR recover SR and convergence toward plain HIL-SERL?

**Endpoints (same instrument as RQ1–RQ3):**
1. Mean autonomous SR on the frozen ±16 mm ruler (9 poses, yaw 0, `best_window` k=3, argmax, no operator).
2. Operator-dependency: ICC, σ²_between/within, operator spread (σ_op, CV), permutation p, tier gradient.
3. **Convergence rate + steps-to-convergence** (the headline for this study).
4. Paired per-cell `SR_logged − SR_hilserl` + sign test; and offline `SR_logged(RQ4) − SR_logged(RQ3)`.

---

## 2. The update target — the heart of the study

The critic TD target in [sac.py](serl_launcher/serl_launcher/agents/continuous/sac.py) (`critic_loss_fn`):

```python
target_next_qs = forward_target_critic(next_obs, next_actions)   # (K, batch)
if critic_subsample_size is not None:
    subsample_idcs = randint(shape=(K, M) if per_member else (M,), 0, K)
    target_next_min_q = target_next_qs[subsample_idcs].min(axis=1 if per_member else 0)
else:
    target_next_min_q = target_next_qs if per_member else target_next_qs.min(axis=0)
target_q = reward + discount * mask * target_next_min_q
```

There are three coupled knobs here: **K** (how many critics), **M = `critic_subsample_size`** (how many are min-ed for the target), and **`per_member_targets`** (whether each critic gets its own target). Below, each is documented as **original → why → RQ4 → why**.

### 2.1 `critic_subsample_size` (M) — the min-over-M pessimism dial *(the primary fix)*

| | value | rationale |
|---|---|---|
| **Original (config default / RQ3 `logged`)** | **M = 2** | This is **REDQ** (Chen et al., ICLR 2021): the TD target is the **minimum over M critics randomly subsampled** from the K-ensemble. The `min` is a *conservative / pessimistic* backup that fights the well-known **over-estimation bias** of off-policy Q-learning (Clipped Double-Q, Fujimoto 2018 generalised). M=2 reproduces Clipped Double-Q per draw. |
| **RQ4 (`logged`)** | **M = 1** | With M=1 the target is a **single randomly-chosen critic — no `min` at all**, so the pessimistic under-estimation from the `min` operator is removed. |

**Why change it (the core hypothesis).** On the cuboid task, insertion requires the policy to **press into contact** — a *risky* action whose Q-value the pessimistic `min` systematically **under-estimates** (the worst-case critic dominates the target). An under-valued "press" action is avoided → the policy hovers, never inserts autonomously, and the actor's rolling autonomous-success stays below the 0.9 convergence bar (RQ3: `logged` non-convergence + `nov01`/`int00` collapse to 7–11%). **M=1 removes exactly this bias.** The trade-off is that the `min` guard against over-estimation is gone; if over-estimation returns, learning could destabilise — which is why the mild K reduction + lower LR + warm-start accompany it.

**Implementation note.** M had **no independent CLI/campaign flag** — it was coupled to `--critic_ensemble_size` (auto-set to `min(M_default, K)`). RQ4 added the standalone `--critic_subsample_size` flag (train + campaign) so M can be set independently of K, clamped to `M ≤ K`, applied *after* the K override so an explicit M wins.

### 2.2 `critic_ensemble_size` (K) — number of critics

| | value | rationale |
|---|---|---|
| **Original (config default / RQ3 `logged`)** | **K = 10** | A large ensemble gives (a) a better **epistemic uncertainty** signal σ_Q = std over K (used by the active gate, not here), and (b) more averaging for the mean-Q used by the actor. REDQ uses K=10. |
| **RQ4 (`logged`)** | **K = 8** | A **mild** reduction. Motivation: fewer critics = less compute/GPU and memory pressure (relevant after the RQ3 OOM incidents), and a *slightly* less pessimistic subsample pool, while **still a genuine ensemble** (8 ≫ 2) so the ablation is honestly "ensemble vs plain", not "ensemble shrunk to baseline". |
| **`hilserl`** | **K = 2** (fixed) | The plain baseline is Clipped Double-Q (2 critics, `min` over both, no subsample). `critic_*` overrides are **mode-gated off** for `hilserl`, so it can never be turned into an ensemble by accident. |

**Why not K=2 for the fixed arm?** That would collapse `logged` into `hilserl` and the study would no longer isolate "does a *fixed ensemble* converge". K=8 keeps a real ensemble whose only relief from RQ3 is the target/optimisation changes.

### 2.3 `critic_per_member_targets` — decorrelated targets *(unchanged; documented for completeness)*

| | value | rationale |
|---|---|---|
| **Original (config default / RQ3 `logged`)** | **True** | Each of the K critics draws its **own** subsample and computes its **own** target, instead of all members regressing onto one shared REDQ target. Decorrelated targets keep the ensemble **diverse**, so σ_Q stays a meaningful *epistemic* signal rather than collapsing. |
| **RQ4 (`logged`)** | **True (kept)** | Not a change. With **M=1 + per_member=True**, each of the 8 critics targets **one independently-chosen member** → maximal decorrelation *and* no min-pessimism. This is the cleanest "diverse ensemble, no conservative bias" setting. |

Interaction with M: `subsample_idcs` has shape `(K, M)` when `per_member` (each member its own draw) vs `(M,)` when shared; the `min` is over `axis=1` (within a member's M draws) vs `axis=0`. So **M=1 + per_member** ⇒ `min` over a single element ⇒ identity ⇒ each member = one random critic's value.

### 2.4 `critic_bootstrap_p` — data diversity *(unchanged)*

| | value | rationale |
|---|---|---|
| **Original / RQ3 / RQ4** | **0.8** | Bernoulli keep-probability of the **persistent per-(transition, member) bootstrap masks** in the replay buffers. Each critic trains on its own ~80% bootstrap of the data (bagging), so members see **different data** — the second ingredient (with per-member targets) that makes σ_Q epistemic. `p=1.0` would be the shared-data (deep-ensembles init-only) ablation. Kept at 0.8. |

---

## 3. Optimisation & warm-start parameters

### 3.1 `learning_rate` — Adam step size

| | value | rationale |
|---|---|---|
| **Original (built-in / RQ3)** | **3e-4** | The standard SAC/RLPD default for the actor, critic and temperature Adam optimisers. |
| **RQ4 (`logged` only)** | **1.5e-4** (halved) | RQ3/earlier runs showed the cuboid policy **oscillates 0↔100% SR** between adjacent checkpoints (a known instability; `best_window` eval was introduced to de-noise it). Halving the LR **damps the oscillation** — smaller gradient steps overshoot less — trading raw speed for a more stable plateau the convergence gate can actually latch onto. Applied **only to `logged`** (the fixed ensemble); `hilserl` keeps 3e-4 so it stays the plain reference. |

### 3.2 `demo_pretrain` / `demo_pretrain_steps` — warm start

| | value | rationale |
|---|---|---|
| **Original (RQ3)** | **off (0 steps)** | Training started from a randomly-initialised critic + actor and learned purely online from the actor's stream (+ demos in the replay mix). |
| **RQ4 (`logged` only)** | **on, 5000 demo-only gradient steps** | A **cold, randomly-initialised 8-critic ensemble is maximally pessimistic** (all members disagree wildly, the `min`/mean is untrustworthy) and — combined with any residual pessimism — suppresses the press action from the very first episode, which is precisely when RQ3's ensemble cells stalled. **Pre-training the critic/actor on the demos for 5000 steps** gives the ensemble a sane initial value landscape *before* online RL, so the policy starts already willing to attempt insertion. Applied **only to `logged`**; `hilserl` starts cold (plain). |

**Budget note.** `demo_pretrain_steps=5000` are learner gradient steps that run **before** waiting for online data; they advance the learner step counter but do **not** consume the actor's 40 000 env-step interaction budget.

---

## 4. Why the baseline must stay plain — the per-arm override

The three "B" fixes split into two kinds:
- **`critic_*` (K, M, per-member)** are **mode-gated**: `train_aqi_rlpd` drops them for `hilserl` (fixed 2-critic), so they only ever touch `logged`.
- **`demo_pretrain` and `learning_rate`** are **not** mode-gated — historically they applied to *every* cell.

If left as-is, the `hilserl` arm would *also* be warm-started + slow-LR, so it would no longer be "plain HIL-SERL". To keep the comparison honest, RQ4 adds a **per-arm override** mechanism:

```yaml
method_overrides:
  hilserl:
    demo_pretrain: false      # no warm start -> plain 2-critic HIL-SERL
    demo_pretrain_steps: 0
    learning_rate: 0          # 0 = built-in default (3e-4)
```

Implemented as `_arm_val(args, method, key, default)` in [run_campaign.py](examples/run_campaign.py); `_learner_cmd` resolves `demo_pretrain*`/`learning_rate` per method. **Verified by dry-run:** `hilserl` learner gets `--nodemo_pretrain` and no `--learning_rate` (→ 3e-4); `logged` gets `--demo_pretrain --demo_pretrain_steps 5000 --learning_rate 0.00015 --critic_ensemble_size 8 --critic_subsample_size 1 --per_member_targets`.

---

## 5. Full configuration (`examples/campaigns/rq4_ensemble_convergence.yaml`)

| block | setting | value | note |
|---|---|---|---|
| task | peg | cuboid | same as RQ3 (comparable) |
| operators | per tier | 3 (exp/int/nov) | within-tier operator variance estimable |
| | jitter / seed | 0.15 / 0 | reproducible roster |
| demos | per operator | 6 filtered geometric | reused across arms |
| methods | | `hilserl`, `logged` | 2 arms |
| replicates | seeds | 2 | 4 samples/tier/method |
| **grid** | **cells** | **36** | 2 × 3 × 3 × 2; 18 demo jobs |
| budget | step cap | 40 000 env steps | (RQ3 was 50k) |
| | wall-clock safety | 300 min/cell | anti-hang only |
| | convergence | 0.9 / win 20 / min 3000 / mini-eval ≥0.6 | **identical to RQ3** (metric NOT changed) |
| | buffer_period | 0 | disk: no ~9 GB/cell dumps (RQ3 fix) |
| | checkpoint_keep | 25 | disk (RQ3 fix) |
| aqi (logged) | K / M / per-member | 8 / 1 / true | the target fixes |
| budget (logged) | demo_pretrain / LR | 5000 / 1.5e-4 | warm start + damped LR |
| method_overrides | hilserl | plain | no pretrain, default LR |
| eval | ruler | ±16 mm, 9 poses, yaw 0 | `best_window` k=3, argmax |
| logging | wandb | online, group `rq4_ensemble_convergence` | |

**Convergence metric deliberately unchanged.** RQ4 isolates the **training** fixes; if convergence improves it must be because the policy genuinely reaches sustained autonomous success, not because the bar was lowered. (Relaxing the metric — mini-eval-based convergence, 0.8 threshold — is a separate, future ablation.)

---

## 6. Launch & infrastructure (carried over from RQ3 incidents)

Launched as an **OOM-protected transient systemd user service**, decoupled from VS Code, so neither `systemd-oomd` (which kills the VS Code scope) nor the kernel OOM killer can take it down as collateral; swap was raised to **55 GB** (persisted) so global OOM cannot fire for a ~4 GB run:

```
WANDB_RUN_GROUP=rq4_ensemble_convergence \
systemd-run --user --unit=rq4_campaign --collect -p ManagedOOMPreference=avoid \
  -p WorkingDirectory=/home/user/Downloads/hil-serl_aqi \
  -- bash -c './.venv/bin/python examples/run_campaign.py \
       --config examples/campaigns/rq4_ensemble_convergence.yaml \
       --campaign runs/rq4_ensemble_convergence \
       --converge_require_autonomous --wandb online \
       >> runs/rq4_ensemble_convergence.out 2>&1'
```
Monitor: `systemctl --user status rq4_campaign.service`; `tail -f runs/rq4_ensemble_convergence.out`; `find runs/rq4_ensemble_convergence/checkpoints -maxdepth 2 -name .done | wc -l`. Disk defaults (`buffer_period 0`, `checkpoint_keep 25`) keep the whole run ~90 GB.

---

## 7. Implementation history (code changes this session)

All backward-compatible (new flags/keys default to legacy behaviour). Files: [examples/train_aqi_rlpd.py](examples/train_aqi_rlpd.py), [examples/run_campaign.py](examples/run_campaign.py).

1. **`--critic_subsample_size` flag** (train). Independent override of M, clamped to K, applied *after* the `--critic_ensemble_size` override so an explicit M wins; mode-gated to non-`hilserl`. Previously M could only be changed by changing K.
2. **Campaign wiring for the critic knobs.** `--critic_ensemble_size`, `--critic_subsample_size`, `--per_member_targets` added to `run_campaign` argparse + `apply_config` (`aqi.{critic_ensemble_size,critic_subsample_size,per_member_targets}`) + `_aqi_flags` emission (both learner and actor).
3. **Per-arm overrides.** New top-level YAML `method_overrides: {<method>: {key: val}}` + `_arm_val()` helper; `_learner_cmd` resolves `demo_pretrain`/`demo_pretrain_steps`/`demo_pretrain_batch_size`/`learning_rate` per method, so a single arm (the plain baseline) can opt out of a campaign-wide fix.

(The disk knobs `--buffer_period` / `--checkpoint_keep` were added in the RQ3 session and are reused here.)

---

## 8. Hypotheses / what a result would mean

- **If fixed-`logged` recovers to ≈ plain HIL-SERL SR and converges on ≥ 10/12 cells:** the RQ3 ensemble penalty was **fixable** — it was REDQ min-Q pessimism + cold start + high LR, not an intrinsic property of using an ensemble. That reframes the RQ3 negative result: the ensemble is usable *if* the target is de-pessimised.
- **If it stays well below plain HIL-SERL / still fails to converge:** the operator-dependence is **intrinsic** to the K-critic architecture on this task, strengthening the RQ3 conclusion that plain 2-critic HIL-SERL is the better choice here.
- **Isolation caveat:** RQ4 changes **four** things on `logged` at once (M, K, LR, pretrain). If it helps, a follow-up one-factor sweep (M only; then + pretrain; then + LR; K∈{2,4,8,10}) is needed to attribute the gain. This campaign is the *combined-best-shot* test; the factorial attribution is future work.

---

## 9. Key comparison table (before → after)

| parameter | plain `hilserl` | RQ3 `logged` | **RQ4 `logged`** | direction of change | why |
|---|---|---|---|---|---|
| critics **K** | 2 | 10 | **8** | ↓ mild | less compute/pessimism, still an ensemble |
| subsample **M** (min-over) | 2 (min of 2) | 2 | **1** (no min) | ↓ **key** | remove min-Q pessimism suppressing "press" |
| per-member targets | n/a | True | True | — | keep ensemble diversity (σ_Q epistemic) |
| bootstrap keep-p | n/a | 0.8 | 0.8 | — | per-member data diversity (bagging) |
| learning rate | 3e-4 | 3e-4 | **1.5e-4** | ↓ | damp 0↔100% oscillation |
| demo-pretrain | off | off | **5000** | ↑ | warm start a cold pessimistic ensemble |
| convergence metric | 0.9 auto | 0.9 auto | 0.9 auto | — | isolate training fixes, don't move the bar |
| step cap | 50k (RQ3) | 50k | **40k** | ↓ | warm start shortens the needed budget |

---

## 10. Results (2026-09-23) — the fixes worked: the ensemble penalty was FIXABLE

All 36 cells finished cleanly (no crashes; verified). `best_window` eval on the frozen ±16 mm ruler.

### 10.1 Headline — fixed ensemble recovers to parity with plain HIL-SERL

| metric | RQ3 `logged` (UNFIXED: K10/M2/3e-4/cold) | **RQ4 `logged` (FIXED: K8/M1/1.5e-4/pretrain)** | RQ4 `hilserl` (plain) |
|---|---|---|---|
| mean autonomous SR | 68.2% | **82.3%** | 84.2% |
| ICC (operator-dependence) | 0.743 | **0.000** | 0.000 |
| σ_op / CV | 0.365 / 0.535 | **0.197 / 0.239** | 0.084 / 0.100 |
| p(operator matters) | 0.021 (they do) | **0.533 (they don't)** | 0.876 |
| converged | 7/12 (58%) | **16/18 (89%)** | 16/18 (89%) |
| paired vs plain hilserl | −18.5 pp, catastrophic cells | **−1.9 pp, median +3.7, sign-test p = 0.302 (tie)** | — |

**The RQ4 hypothesis is confirmed:** de-pessimising the target (**M 2→1**, no min-Q) + warm-start + halved LR **recovered the 8-critic ensemble to statistical parity with plain HIL-SERL**. Specifically, relative to RQ3's unfixed ensemble:
- **Operator-dependence eliminated:** ICC **0.743 → 0.000** — the fixed ensemble is now operator-*independent* like the baseline; the permutation test flips from "operators matter" (p=0.021) to "they don't" (p=0.533).
- **Convergence restored:** **58% → 89%** of cells converge, exactly matching plain HIL-SERL (16/18 each).
- **SR gap closed:** 68.2% → 82.3%, a dead heat with plain (84.2%); the paired sign test is **not significant (p=0.302)** and the *median* per-cell delta is **+3.7 pp** (the fixed ensemble slightly wins the typical cell).

This **reframes the RQ3 negative result**: the ensemble's operator-dependence and non-convergence were an artifact of **REDQ min-Q pessimism + cold start + high LR**, *not* an intrinsic cost of a critic ensemble.

### 10.2 Which fix did what (highlighted)

| fix (RQ3 → RQ4) | mechanism it removed | observed effect |
|---|---|---|
| **M: 2 → 1** (no min-Q) — *the key lever* | the pessimistic `min` under-valued the risky "press-into-contact" action → policy hovered, never inserted autonomously | the dominant cause of RQ3 non-convergence; its removal is the most likely driver of ICC 0.74→0.00 and convergence 58%→89% |
| **demo-pretrain: 0 → 5000** | a cold random 8-critic ensemble is maximally pessimistic from step 1 | gives the ensemble a sane value landscape before online RL so it attempts insertion early |
| **LR: 3e-4 → 1.5e-4** | large steps overshoot → 0↔100% SR oscillation | steadier plateau the convergence gate can latch onto |
| **K: 10 → 8** | (mild) compute/GPU + slightly smaller subsample pool | keeps a genuine ensemble; not expected to be a major driver |

*Attribution caveat:* four factors changed on `logged` at once, so this is the **combined-best-shot** result; a one-factor sweep (M-only → +pretrain → +LR; K∈{2,4,8,10}) is needed to credit the gain. Mechanistically **M 2→1** is the prime suspect (it directly removes the "press"-suppressing pessimism).

### 10.3 Skill gradient — fixed ensemble now beats plain on expert/intermediate

| tier | `hilserl` (plain) | `logged` (fixed) |
|---|---|---|
| expert | 92.6% | **99.4%** |
| intermediate | 84.6% | **90.7%** |
| novice | **75.3%** | 56.8% |

The fixed ensemble **surpasses plain HIL-SERL on expert and intermediate operators**, and the *only* remaining deficit is **novice**.

### 10.4 The remaining issue — novice-operator instability

The novice `logged` mean (56.8%) is dragged down by **high seed variance**, all verified real (no crashes): `nov02` evals **3.7%** (s0, not converged) vs **100%** (s1, converged @33k); `nov00_s1` 14.8% (not converged); `nov01_s1` **converged @26k but evals 48%** (a train↔eval oscillation / distribution mismatch, not a training failure). So the de-pessimised ensemble is competitive on average but **not yet stable on low-skill operators**.

### 10.5 Cost

The fixed ensemble takes **~5k more env steps to converge** (mean 32 478 vs 27 709; steps-ICC 0.123 vs 0.000) — a modest training-effort premium for reaching parity.

### 10.6 Bottom line

Plain HIL-SERL and the **fixed** 8-critic ensemble are now **equivalent** in SR, operator-independence and convergence — with the ensemble even ahead on expert/intermediate. **The ensemble is usable if the target is de-pessimised (M=1) and the critic is warm-started.** The open thread is **novice-operator stability** (§10.4), and the clean **factorial attribution** of the four fixes (§10.2) is the recommended next experiment.

Report: `runs/rq4_ensemble_convergence/report/dependency.{md,json}` + `report/plots/`.

---

## 11. Follow-up diagnostic (2026-09-23) — is the healthy critic enough for quality replay? **No.**

RQ4 gave a **healthy** critic (converged, operator-independent, parity with plain). Since the quality signal `q_int = μQ(s,a_H) − μQ(s,a_robot)` is derived *entirely from the critic*, the natural question was: does the fixed critic finally produce a **larger / skill-discriminative** `q_int`, unlike the ~1e-3 skill-blind signal on RQ3's broken ensemble (RQ3 doc §18)? A cheap 3-cell probe (`quality` mode on the RQ4 base — K8/M1/per-member/pretrain/LR — one operator per tier; `runs/quality_on_rq4_diag`) answered it decisively.

### 11.1 The signal did NOT improve — and is *anti*-aligned with value

| measure | RQ3 (broken critic) | **RQ4 diag (healthy critic)** | needed for "go" |
|---|---|---|---|
| per-tier `q_int` (exp / int / nov) | 0.0004 / 0.0007 / 0.0005 | **0.0003 / 0.0005 / 0.0004** | separate the tiers |
| `q_int` magnitude (mean / max) | ~0.0004 / 0.06 | **0.00042 / 0.009** | materially larger |
| **AUROC(`q_int`, episode success)** | (≈0.5 expected) | **0.175** | > 0.5 |
| raw `A_hat` (pre-clip) | — | **mean −0.004, 64% negative** | — |

(1215 intervention segments; 122 success / 1093 fail.)

### 11.2 Interpretation

1. **Fixing the critic changed nothing about the signal.** `q_int` is still ~1e-3 and still **skill-blind** (0.0003–0.0005 across all tiers) — essentially identical to RQ3. So the weak, undiscriminating signal was **not** an artifact of the broken ensemble.
2. **AUROC = 0.175 is the decisive result.** Far *below* 0.5 ⇒ `q_int` is **anti-predictive**: interventions the critic scores as high-advantage tend to come from **failed** episodes. Prioritising replay by `q_int` would preferentially up-weight the *worst* interventions.
3. **Root cause — an off-distribution asymmetry, not a critic-quality problem.** The raw counterfactual advantage `A_hat` is **mostly negative** (mean −0.004, 64% < 0): the critic systematically values **its own** action `a_robot` above the human correction `a_H`. Reason: `a_robot` is *on-distribution* (the policy generated the training data) while `a_H` is *off-distribution*, so an off-policy critic cannot value it reliably and rates it lower. Clipping to `[0,∞)` keeps only the ~36% noisy positives, which correlate with failure. **This asymmetry is intrinsic to the counterfactual-advantage formulation and independent of critic convergence** — exactly why the RQ4 healthy critic did not help.

### 11.3 Verdict & implication

**No-go for a full "quality on the fixed ensemble" campaign as formulated** — on this evidence it would *hurt* (it replays failure-correlated interventions). This confirms and strengthens RQ3 §18.3: the fix must be the **numerator**, not the critic or a rescaling. Candidate redesigns (future work), in order of promise:
- **Outcome / return-based advantage** — score an intervention by its **n-step return-to-go / did-the-takeover-lead-to-success**, not a one-step critic Q-gap. The AUROC result shows the *outcome* label carries exactly the information the Q-gap is missing (and contradicting).
- **Off-distribution-robust values (CQL/IQL-style)** so `μQ(a_H)` is trustworthy for actions the policy never takes.
- Lower-variance `A_hat` (mean/argmax `a_robot`) — helps the noise but **not** the sign bias, so secondary.

**Thesis takeaway:** *counterfactual-advantage intervention scoring is not merely weak but **anti-aligned** with intervention value, because it asks an off-policy critic to compare an off-distribution expert action against its own on-distribution action; repairing the critic (RQ4) does not repair the signal.* Config: `examples/campaigns/quality_on_rq4_diag.yaml`; logs: `runs/quality_on_rq4_diag/aqi_logs/`.

---

## 12. Score redesign (2026-09-23/24) — a critic-free *behavioral* score fixes the signal

§11 concluded the fix must be the **numerator**, not the critic. This section does exactly that: it identifies which intervention-time signals *are* predictive, replaces the counterfactual Q-gap with a critic-free composite, implements it as an ablatable mode, and validates it live. This is the constructive resolution of the whole quality thread (RQ2 → RQ3 → RQ4).

### 12.1 Offline re-score (zero new training) — which signals predict outcome?

Every intervention segment already logs behavioural features (`d_progress`, `d_value`, `f_peak`, `t_int`) alongside `episode_success`. Re-scoring the §11 counterfactual logs (`runs/quality_on_rq4_diag`, 1215 segments) with **AUROC(score, success)** shows the counterfactual signals are uniquely broken and several behavioural ones carry real information:

| score | AUROC vs success | note |
|---|---|---|
| `q_int` (counterfactual, current) | **0.175** | anti-predictive |
| `a_hat` (raw pre-clip) | 0.168 | anti-predictive |
| `d_value` | 0.599 | weak-positive |
| **brevity (−`t_int`)** | **0.695** | short takeover ⇒ good |
| **B = −`t_int` + `d_value` − `f_peak`** | **0.721** | the composite |
| B + `σ_Q` | 0.816 | strongest (σ caveat below) |

And `t_int` supplies the **skill discrimination** `q_int` never had — per-tier mean takeover length **expert 31.6 < intermediate 45.8 < novice 65.8 steps** (monotone), while `q_int` was flat (0.0004 all tiers).

**σ_Q caveat (and a side-finding).** `σ_Q` has the highest AUROC (0.84) but its per-tier means are flat (0.008/0.006/0.006), so its predictive power is a **temporal confound** (the σ scale drifts over training), not skill — hence it is **excluded** from the core score. It also flags a separate issue: if higher `σ_Q` really is associated with *success*, the disagreement gate `g(σ_Q)` (which *down*-weights high σ) may be **inverted** — worth its own investigation.

### 12.2 The redesigned score (implemented)

$$q_\text{new} \;=\; z(\Delta\text{value}) \;-\; z(t_\text{int}) \;-\; z(f_\text{peak})$$
— *a short, value-increasing, force-controlled takeover scores high* — standardised online (running z-scores), signed, clipped to `[0,∞)` for the replay priority. It is **critic-free** except for `d_value` (an on-distribution value change), so it avoids the off-distribution `μQ(a_H)` evaluation that broke `A_hat`.

Implemented as an **ablatable mode** (default behaviour unchanged):
- `AQIConfig.quality_score_mode ∈ {"counterfactual" (default), "behavioral"}` — `serl_launcher/aqi/config.py`.
- `QualityScorer.behavioral_q(t_int, d_value, f_peak)` (running `RunningZScore` standardisers) — `serl_launcher/aqi/quality.py`.
- `AQIRuntime.end_episode` overrides `seg.q_int`/`weights` with the composite when in behavioral mode (broadcast per-step, clipped, exclusions respected) — `serl_launcher/aqi/runtime.py`.
- Flags: `--quality_score_mode` (train, quality-gated) + campaign argparse / YAML `aqi.quality_score_mode` / `_aqi_flags`. Unit-tested (good takeover +5.48 vs bad −5.25); `counterfactual` stays the default.

### 12.3 Live validation on the RQ4 base

Re-ran the same 3-cell gate with `quality_score_mode: behavioral` (`runs/quality_behavioral_diag`, 1106 segments):

| | `counterfactual` (old) | **`behavioral` (new)** |
|---|---|---|
| **AUROC(`q_int`, success)** | 0.175 (anti) | **0.783** (strongly predictive) |
| per-tier AUROC (exp / int / nov) | — | **0.79 / 0.69 / 0.66** (all > 0.5) |
| per-tier `q_int` mean (exp / int / nov) | 0.0004 / 0.0005 / 0.0004 (blind) | 0.77 / 0.82 / **0.55** (novice separated) |
| magnitude / median / frac>0 | ~1e-3 / 0 / 0.35 | O(1) / 0 / 0.41 |

The online behavioral `q_int` is **positively and skill-aware predictive** (AUROC 0.78, and > 0.5 *within* every tier, so it is not a mere tier confound), selectively boosting the good ~40% of takeovers instead of the failure-correlated subset the old score picked. The offline prediction (0.72) held live (0.78).

### 12.4 Caveats & next step

- **Predicts value ≠ improves training.** AUROC 0.78 proves the score *ranks* interventions by value; it does **not** yet prove that *prioritising replay by it improves the policy* — that requires a real autonomous-SR comparison, not a signal metric.
- Small n (3 cells, 1 seed); expert ≈ intermediate (the score separates novice cleanly but not the top two tiers); `d_value` still touches the critic (though `t_int`/`f_peak` carry most of the signal).
- **Recommended next experiment (RQ5):** a small real campaign on the RQ4 base — **`logged` (uniform) vs `quality`-counterfactual vs `quality`-behavioral** — measuring autonomous SR, convergence, and operator-dependence, to test whether the validated signal translates into a training benefit.

**Thesis takeaway (constructive):** *the intervention-quality signal is salvageable — not by fixing the critic, but by replacing the off-policy counterfactual Q-gap with a critic-free behavioural score (short, value-increasing, force-controlled takeover), which is positively and skill-aware aligned with intervention outcome (AUROC 0.18 → 0.78).* Configs: `examples/campaigns/quality_behavioral_diag.yaml`; logs: `runs/quality_behavioral_diag/aqi_logs/`.

---
