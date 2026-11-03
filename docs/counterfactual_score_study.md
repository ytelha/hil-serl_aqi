# Counterfactual intervention score: why it fails, and can it be saved?

Status: **decision rule fixed before the results** (this section is
committed before the study cell finished). Results added 2026-11-03 in §5: **abandon**.

## 1. The score and its history

The original AQI quality score grades each human takeover step by

```
A_hat(s, a_H) = mu_Q(s, a_H) - mu_Q(s, a_robot)        (mean over the K critics)
q_signed      = A_hat * g(sigma_Q),   g = 1 / (1 + (sigma_Q / sigma_ref)^2)
q_int         = max(q_signed, 0)       -> replay priority, BC weight
```

- `a_robot` is **one sampled** action of the current policy at the same state.
- `sigma_Q` is the ensemble std of Q **at a_H**.

What happened so far (`docs/thesis/rq4_ensemble_convergence_and_qa_log.md`
§11–12):
- On RQ3's broken critic, `q_int` was ~1e-3 and skill-blind.
- RQ4 repaired the critic (K 10→8, REDQ M 2→1, demo pretraining, lower LR)
  to parity with plain HIL-SERL. `q_int` **did not change**: still ~1e-3,
  skill-blind, and **anti-predictive** of episode success (AUROC 0.175),
  with raw A_hat 64% negative.
- The recorded explanation, "a_H is off-distribution, so the critic
  undervalues it", was **inferred, not tested**. The score was then replaced
  by critic-free scores (behavioral, later hindsight/progress/outcome, patch
  0005).

Two gaps make that verdict incomplete:
1. **The label.** Episode success is confounded by timing
   (`docs/quality_signal_audit.md`). A score whose scale drifts over training
   can look anti-predictive without any per-takeover meaning.
2. **The construction.** The score has three design choices that each
   predict failure on their own, independent of any off-distribution effect.
   They were never separated.

## 2. Hypotheses tested

| # | hypothesis | prediction if true |
|---|---|---|
| H1 | **Noisy baseline.** One policy sample `a_robot` is a noisy estimate of the policy's value; its error dwarfs the advantage | median \|Q(a_robot) − E_π Q\| ≫ median \|A_V\|; A_cf ranks worse than A_V |
| H2 | **Argmax bias.** The policy is trained to maximise this critic, so Q(a_H) − Q(policy) is ≤ 0 almost by construction: the score measures how far a_H is from the policy, not how good it is | A_mode and A_V mostly negative; score correlates with −‖a_H − μπ‖ |
| H3 | **Wrong uncertainty.** The gate uses the ensemble spread of Q, which is mostly common to all actions at a state. The relevant noise is the critics' disagreement on the *difference* | across-critic std of Q(a_H) − V ≪ σ_Q(a_H) |
| H4 | **Label artifact.** The anti-prediction vs episode success is drift over training, not per-takeover information | A_cf correlates with episode index; its time-stratified AUROC is ≈ 0.5, not ≪ 0.5 |
| H5 | **The critic cannot rank single human actions at all.** At 10 Hz, one action changes the future very little, and the critic doesn't resolve it | even the best-built variants (A_V, percentile, t-statistic) do not track what the takeover physically achieved |

H1–H4 are fixable by construction; H5 is not (for a per-step score).

## 3. The experiment

One cell (`examples/campaigns/cf_score_study.yaml`) trained exactly like
the RQ4 quality diagnostic:
- K = 8, M = 1, per-member targets;
- 5000 demo-pretraining steps, LR 1.5e-4;
- counterfactual score, 25 000 env steps;
- one intermediate synthetic operator.

It runs with `quality_cf_diag` (logging only). Per takeover step it records
every critic's Q at a_H, at a_robot, at the policy mode and at 8 fresh
policy samples, plus the simulator's **ground truth** after every step: true
lateral error to the hole, height, jam.

`examples/cf_score_analysis.py` then compares, against **timing-free
ground-truth labels**:
- **Scores:** A_cf (current), A_mode (vs the policy mode), A_V (vs E_π Q),
  `pct` (share of policy samples the critic ranks below a_H), and `t_k` (the
  across-critic t-statistic of Q(a_H) − V).
- **Labels:**
  - per step: the reduction of the true lateral error caused by the executed
    human action;
  - per segment: the lateral-error reduction over the takeover, and whether
    the peg was aligned (< 5 mm clearance) or jammed at handback;
  - also the classic episode-success label, all and time-stratified.

## 4. Decision rule (fixed before the results)

- **Pursue** a redesigned counterfactual score if at least one of A_V /
  `pct` / `t_k` reaches, per segment:
  - AUROC(aligned at handback) ≥ 0.65,
  - Spearman(lateral-error reduction) ≥ 0.25,
  - time-stratified AUROC(episode success) ≥ 0.60,

  and clearly beats A_cf. The next step is then a small controlled comparison
  of that variant against hindsight/progress.
- **Abandon** the per-step counterfactual score if all variants stay ≤ ~0.55
  on the ground-truth labels. The critic then does not rank single human
  actions (H5), and no construction fixes that. The critic-light scores of
  patch 0005 remain the path.
- **Undecided** (0.55–0.65): repeat with 2–3 operators/seeds before deciding.

One cell cannot settle "improves training"; it can only settle whether the
signal exists. That's the question that decides between pursue and abandon.

## 5. Results

Ran on CLAIX on 2026-09-29 (cell job 4529759, 53 min on one H100; analysis job 4530130), read on 2026-11-03.
Report and analysis: `presentation_bundle/results/cf_study/`. One cell (`int00`, seed 0), 25 000 environment
steps, 90 episodes, 264 takeover segments with 11 015 steps. Logged and recomputed values agree to 5e-6; no bad
segments.

**Verdict by the rule of §4: abandon the per-step counterfactual score.** No variant reaches any of the three bars,
and every variant stays below 0.5 on the ground-truth labels.

| score (per segment) | AUROC(aligned at handback) | Spearman(lateral-error reduction) | AUROC(episode success, time-stratified) | AUROC(not jammed at handback) |
|---|---|---|---|---|
| bar to pursue | ≥ 0.65 | ≥ 0.25 | ≥ 0.60 | – |
| A_cf (current) | 0.31 | 0.15 | 0.29 | 0.33 |
| A_mode | 0.30 | 0.12 | 0.27 | 0.32 |
| A_V | 0.29 | 0.13 | 0.28 | 0.32 |
| pct | 0.41 | 0.17 | 0.48 | 0.38 |
| t_k | 0.48 | 0.05 | 0.48 | 0.45 |

| hypothesis | verdict | evidence |
|---|---|---|
| H1 noisy baseline | rejected | one-sample baseline error 0.0007 against median \|A_V\| 0.0072; A_cf and A_V rank alike |
| H2 argmax bias | **supported** | A_mode < 0 in 99.7 %, A_V in 96.4 %, A_cf in 92.7 % of the steps; the score rises the closer a_H is to the policy mean (Spearman 0.46–0.50) |
| H3 wrong uncertainty | partly | the critics' disagreement on the difference (0.0024) is half the gate's σ_Q (0.0049), not much smaller |
| H4 label artifact | partly | strong drift (Spearman with episode index −0.68; mean A_cf −0.006 → −0.019 while success rises 0 → 63 % over the thirds), but the time-stratified AUROC stays 0.29, so drift is not the whole story |
| H5 critic cannot rank single actions | **supported** | per step, Spearman with the change of the lateral error 0.02–0.09 for every variant |

**Labels are informative.** 31 % of the takeovers end aligned and 24 % jammed; ending aligned predicts episode success
(AUROC 0.68). The scores miss what the label carries.

**Reading.** The score measures how far the operator's action is from the policy's, not how good it is. It is lowest
for the corrections that differ most from the policy, and those end aligned more often. Flipping the sign is no way out:
it was not part of the rule, it would still fail the Spearman bar, and it would only measure the distance between the
human and the policy action, which needs no critic. The critic-light scores of patch 0005 and E2HiL remain the paths;
one cell settles that the signal is absent, not how a training campaign would turn out.
