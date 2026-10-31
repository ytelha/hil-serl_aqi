# Baseline comparison: E2HiL and HiRIL vs plain HIL-SERL

A deep reference for comparing the two published HIL-SERL competitors against the
baseline: what each method does, the settings/configs to run them, **every issue and
consideration**, and a concrete campaign design. Implementation docs:
[e2hil_baseline.md](../e2hil_baseline.md), [hiril_baseline.md](../hiril_baseline.md);
issues found in testing: `handoff/handoff/FINDINGS_cluster_method_test.md`.

---

## 0. What you're comparing and why

Both are **published HIL-SERL competitors** that, like AQI, decide *which
human/robot data to trust* — with different signals. The natural anchor is **plain
`hilserl`** (2-critic HIL-SERL): E2HiL *is* hilserl + one change, and HiRIL is the
nearest published cousin of AQI.

| method | learner | difference from `hilserl` | relation to AQI |
|---|---|---|---|
| `hilserl` (anchor) | 2-critic SAC | — | the baseline AQI extends |
| **E2HiL** | 2-critic SAC | **only the actor loss**: entropy-influence sample selection | competitor to AQI's *quality* (which samples to trust) |
| **HiRIL** | **TD3 + Bayesian distributional critic (BIQN)** | whole learner + risk-triggered requests + risk-weighted replay + BC | nearest cousin to *full AQI* (active gate + quality replay) |

These belong to the **RQ5 / "how does AQI compare to prior HiL methods"** chapter —
the ultimate-goal comparison, best run once the active part is settled.

---

## 1. E2HiL — entropy-guided sample selection

H. Deng et al., *E2HiL*, IEEE RA-L 2026 ([arXiv:2601.19969](https://arxiv.org/abs/2601.19969)).

### Mechanism (actor loss only; buffers / critic / temperature unchanged)
For each state $s$ in the batch, draw $K$ fresh policy actions, compute the state's
**influence on policy entropy**, and keep only the middle band:

$$c(s) = -\,\mathrm{Cov}_k\big(\log\pi(a_k\mid s),\, y_k\big),\qquad y_k=\pi(a_k\mid s)\,A_k,$$

$$I(s) = \mathbb{1}\big[\,P_{5}(|c|)\le |c(s)|\le P_{90}(|c|)\,\big],\qquad
\mathcal{L}_{\text{actor}} = \frac{-\sum_s I(s)\,\big[Q-\alpha\log\pi\big]}{\sum_s I(s)+\epsilon}.$$

The top 10% are "shortcut" samples that collapse entropy; the bottom 5% are noise.
The paper reports most clipped samples are human interventions.

### Settings (`--aqi_mode=e2hil`, or `aqi.e2hil_*` in YAML)

| knob | default | meaning |
|---|---|---|
| `e2hil_K` | 8 | MC action samples per state ($K=1$ masks nothing — degenerate) |
| `e2hil_q_low` / `e2hil_q_high` | 0.05 / 0.90 | the keep band |
| `e2hil_q_agg` | `min` | critic aggregation in the actor objective |

**Architecture:** exactly `hilserl` (2 critics, no subsample/bootstrap, no AQI logs);
checkpoints are ruler-compatible unchanged. It **ignores** `critic_ensemble_size` by
design.

### Issues to control
1. **Confound: `min` vs `mean` (E2).** The official default `q_agg=min` makes E2HiL
   differ from `hilserl` by **two** things — the mask **and** min-vs-mean critics. To
   attribute any difference to the *mask*, add an **`e2hil-mean`** arm
   (`e2hil_q_agg=mean`), which isolates the selection. **Report which you ran.**
2. **Compute fairness (F3) — the big one.** The $K=8$ MC forward passes make E2HiL
   **~2× slower** (cluster: 13.7 it/s → **1.3 updates/env-step** vs 2.6 for others).
   At an equal *env-step* budget it gets **half the training**, so a raw SR comparison
   is unfair to E2HiL. Either report updates-per-env-step alongside SR, match on
   updates/wall-clock, or note it explicitly. Measure it on this GPU.

---

## 2. HiRIL — risk-aware intervention + imitation

Y. Zhou et al., *HiRIL*, Expert Systems With Applications 308 (2026) 131118. Runs as
its **own pipeline** `examples/train_hiril.py` (not an `--aqi_mode`).

### Mechanism
- **TD3 + BIQN** Bayesian distributional critic: return samples $Z_{\theta,\tau}(s,a)$,
  $\theta\sim\mathcal N(\mu,\Sigma)$, $\tau\sim U(0,1)$.
- **Risk** $DU(s,a)=\mathrm{Var}[Z]$ = aleatoric + epistemic; **request a human when**
  $DU(s,a_{RL}) > \tau_{\text{risk}}$.
- **Risk-weighted PER** (TDRD priority $k=|\delta_{TD}|+\varepsilon+\exp(DU_{RL}-DU_H)$
  for human samples) + **BC** $\lambda\,\lVert\pi(s)-a_H\rVert^2$.

### Settings (`hiril:` block / `--hiril_*`; ablations via `arms:`)

| knob | default | note |
|---|---|---|
| `tau_risk` | ≤0 → calibrate once (0.8-quantile of DU over 1000 steps, then frozen); >0 = absolute | ⚠️ see the blocker below |
| `du_mode` | `dual` | `aleatoric` / `epistemic` = the paper's Fig. 6 ablations |
| `lambda_bc` | 1.0 | BC weight (`0` = "w/o imitation") |
| `intervention_penalty` ($r_p$) | −0.1 | `0` = "w/o reward shaping" |
| `use_rd` | true | `false` = "w/o RD" |
| `explore_decay_steps` | 20000 | noise 1.0 → 0.05 |
| `kl_weight` | 0 | the paper drops KL |

Same env / encoder / operator / ruler (ruler needs `--agent_type auto|hiril`). The
synthetic operator **responds to requests** (`respond_to_request=True`), the same
stand-in AQI's active arms use.

### ⚠️ HiRIL's blocking issue (F2) — it stops asking for help
On the cluster, HiRIL calibrated `tau_risk` once at step 1700, then made **0 requests
for the entire run** — degenerating to plain TD3 with **no human-in-the-loop**. Cause:
under **sparse reward**, before any success all critic heads fit the same near-constant
target, so $DU$ collapses **~1000× over training** (roughly halves every episode); a
threshold frozen over a decaying signal is never reached again.

**This makes HiRIL a meaningless HiL baseline unless fixed**, and the config has **no
periodic-recalibration knob** (only calibrate-once or absolute):

| option | effect | needs code? |
|---|---|---|
| **(a) periodic re-calibration** (e.g. every 1000 steps over the last 1000) — relative trigger, request rate ≈ $1-q$, like AQI's z-gate | **recommended** | yes (add `recalib_period`) |
| (c) normalise $DU$ by a slow EMA, threshold the ratio | middle ground | yes |
| (b) absolute `tau_risk>0` (paper) | won't work — $DU$ scale changes ~1000× | no |
| (d) leave as-is | report the collapse as a HiRIL property under sparse reward | no |

**Recommendation: (a) or (c)** — fixed **before** the first cell and **never changed
within a campaign**. It is a small code change (add `recalib_period` to `HiRILConfig`
+ the actor loop).

---

## 3. Cross-cutting considerations (both methods)

- **Reuse RQ4's `hilserl` cells to halve compute — but copy `operators/` too
  (F1 / patch 0009).** Both configs match RQ4's `hilserl` arm exactly (same roster,
  6 demos, 40k cap, ruler, no pretrain, default LR), so RQ4's 18 scored `hilserl`
  cells can be copied in. **If you don't also copy `operators/`, the new cells sample
  a different roster and the paired comparison silently breaks.**
- **Compute note (F3)** must accompany any SR table: report updates-per-env-step per
  method (E2HiL ≈ 1.3, others ≈ 2.6).
- **Endpoints** (same instrument as RQ1–RQ4): autonomous SR + **ICC**
  (operator-dependence) + convergence + the **intervention-rate / human-effort**
  section in `report/dependency.md` (the paper metric for both). For HiRIL also read
  per-episode `du_mean`, `request_rate`, `tau_risk` (its Fig. 7b).

---

## 4. Suggested campaign (concrete)

A single, well-controlled comparison on the RQ4 base:

```
arms:
  hilserl      # baseline anchor (reuse RQ4's 18 cells, copy operators/)
  e2hil        # official (q_agg=min)  -> mask + min-vs-mean
  e2hil-mean   # q_agg=mean            -> isolates the entropy mask
  hiril        # WITH the F2 fix (recalib_period), du_mode=dual
# optional ablations: hiril-no-epi / hiril-no-rd / hiril-no-bc
grid: 3 operators/tier x 3 tiers x 2 seeds = 18 cells/arm (hilserl reused)
budget: RQ4-identical (40k cap, convergence rule, +-16 mm ruler)
endpoints: SR, ICC, convergence, intervention rate, updates/env-step (F3 note)
```

- **Why `e2hil-mean`:** isolates the entropy mask from the min-vs-mean confound.
- **Why `hiril` needs the fix first:** otherwise it never intervenes.
- Cost: ~18 (e2hil) + 18 (e2hil-mean) + 18 (hiril) = **54 new cells** (hilserl reused),
  OOM-protected, wandb online.

Existing starting points: `examples/campaigns/e2hil_comparison.yaml` and
`examples/campaigns/hiril_comparison.yaml` (each `hilserl` vs one method, RQ4-identical).
A combined `arms:` config is cleaner for one paired report.

---

## 5. Decisions to make before building

1. **HiRIL trigger fix:** (a) periodic re-calibration [recommended] or (c) DU-EMA —
   both a small code change; or (d) run-as-is-and-report.
2. **E2HiL:** include the `e2hil-mean` control? (Recommended — the clean attribution.)
3. **Ablation depth:** headline only (`hilserl` / `e2hil` / `e2hil-mean` / `hiril`),
   or add HiRIL's 3 ablation arms?
4. **Timing:** now, or after the active part?

**Build steps (on approval):** (1) implement the HiRIL `recalib_period` fix;
(2) create a combined `baseline_vs_e2hil_hiril.yaml` with the arms above (reusing RQ4's
`hilserl` cells + copying `operators/`); (3) dry-run-verify — then ready to launch.
