# The inconsistent operator (score validity, discard, re-weight) — results & QA log

Companion to [second_machine_synthesis.md](second_machine_synthesis.md). This is the
test of the hypothesis that **a quality score pays off when the operator is inconsistent**,
as humans are. Design, arms and reading rules: [docs/inconsistent_operator.md](../inconsistent_operator.md).

- Machine: the second training machine (`LSPEM499`, RTX 5090), new stack, score `behavioral`.
- Campaigns: `Q1_validity_new` (6 cells), `headroom_diagnostic_new` (9), `two_arm_pilot_new` (6),
  `two_arm_confirm_new` (36), 25–29 October 2026.
- Raw results: `presentation_bundle/results/<campaign>/` (`dependency.md`, `inconsistent_audit.md`,
  `cells.csv`). The audits there were re-run on 2026-10-31 with patch 0021's tool, which adds the
  within-run columns; `inconsistent_audit_on_the_machine.md` is the original.

---

## 0. Headline

**The hypothesis is neither supported nor refuted: the test was too weak to decide it.**

- **The scores can notice carelessness, moderately.** The best online score (behavioural) separates
  attentive from inattentive takeovers with AUROC 0.66 overall, 0.71 / 0.67 for expert /
  intermediate, and not at all for novices (0.52). Takeover length alone does almost as well (0.65).
- **Using the score made no detectable difference.** Discard against re-weight: 93.2 vs 90.3 % ruler
  success, −2.9 points, 95 % CI −13.4 to +6.2, p 0.39.
- **A perfect filter showed no gain either,** but on 3 pairs near the ceiling: keep-all 98.8,
  oracle drop 93.8, random drop 91.4 %.
- **Baseline and keep-all were not run at scale,** by design of the lean experiment, so nothing here
  says whether quality beats HIL-SERL.

---

## 1. The operator

At every takeover the hidden condition stays with probability 0.7, or else it is redrawn as
inattentive with probability 0.4. Inattentive means *sloppy and wrong*: novice execution
parameters, premature commits with probability 0.5, and an aim point 1.5 commit thresholds beside the
hole. Monitoring stays the operator's own. The condition is logged per takeover as ground truth.

| measured in training (all campaigns) | result | design |
|---|---|---|
| persistence: same condition as the previous takeover | 0.85–0.87 | 0.856 |
| inattentive share | 0.31–0.37 | 0.40 in the long run |

**The condition really is worse, at least without learning:** with rotation control, attentive vs
inattentive success is 100 / 43 % (expert solo), 100 / 30 % (expert rescue), 100 / 50 %
and 100 / 30 % (intermediate), 77 / 43 % and 60 / 27 % (novice). All six comparisons pass the
pre-registered 20 points. Without rotation control, four of six fail, because the attentive operator
itself rescues only 13–17 %. So the original platform was excluded.

**In training the effect depends on the tier** (`two_arm_confirm_new`):

| tier | takeover length, attentive / inattentive | episode success, attentive / inattentive |
|---|---|---|
| expert | 23.6 / 32.6 steps | 81 / 59 % |
| intermediate | 35.8 / 43.8 steps | 66 / 33 % |
| novice | 60.0 / 61.3 steps | 23 / 24 % (**no effect**) |

Inattentive *is* novice-level execution, so for a novice the condition changes only the aim and
the premature commits, and in training that did not show. Effectively, the test of inconsistency
is the expert and intermediate tiers.

## 2. Do the scores find the bad corrections?

AUROC for "attentive" on 3 357 takeovers (`two_arm_confirm_new`). *Within* compares only takeovers of
the same run, training phase (bins of 15 takeovers) and onset time (bins of 50 steps). That is the
cleanest number, since the condition comes in long spells.

| score | online mode | pooled (95 % CI) | within | within: expert | intermediate | novice | verdict |
|---|---|---|---|---|---|---|---|
| behavioural composite | `behavioral` | 0.65 (0.63–0.67) | **0.66** | 0.71 | 0.67 | 0.52 | weak |
| takeover length alone | – | 0.60 (0.57–0.62) | 0.65 | 0.69 | 0.68 | 0.51 | weak |
| hindsight advantage | `hindsight` | 0.55 (0.52–0.58) | 0.62 | 0.66 | 0.63 | 0.49 | weak |
| episode outcome | `outcome` | 0.59 (0.57–0.62) | 0.55 | 0.56 | 0.58 | 0.49 | weak |
| progress | `progress` | 0.50 (0.48–0.52) | 0.50 | 0.51 | 0.50 | 0.46 | invalid |
| counterfactual advantage | `counterfactual` | 0.50 (0.48–0.53) | 0.49 | 0.46 | 0.50 | 0.54 | weak (≈ chance) |
| **smoothness (−jerk)** | none yet | 0.71 (0.69–0.73) | 0.76 | 0.86 | 0.85 | **0.40** | valid, but see below |
| yardstick: takeover start time | – | 0.59 (0.57–0.61) | 0.49 | 0.49 | 0.45 | 0.54 | – |
| yardstick: value at onset | – | 0.55 (0.52–0.59) | 0.58 | 0.60 | 0.62 | 0.48 | – |

- **Q1 agrees in direction:** behavioural 0.61 pooled (0.56–0.65), weak, 641 takeovers.
- **Jerk is circular here.** Mean jerk, attentive / inattentive: expert 0.162 / 0.233, intermediate
  0.175 / 0.251, novice 0.269 / 0.255. It detects the motor noise the model defines inattention by, which
  is why it fails for novices. Do not read it as a validated score for humans.
- **This does not contradict the timing audit.** That audit asked whether a score predicts *episode
  success* on clean takeovers, and found no signal. Here the label is the hidden condition, and
  inattentive takeovers really do last longer. The behavioural score picks that up, mainly through its
  length term (without it: 0.61 within).
- **The counterfactual score stays at chance,** also against a label that has nothing to do with timing.

**The live filter** (`drop-hind`, confirmation, after the warm-up; 649 takeovers):

| | inattentive share | dropped | precision | lift over chance | recall |
|---|---|---|---|---|---|
| all | 0.40 | 0.43 | 0.54 | 1.35 | 0.57 |
| expert | 0.34 | 0.38 | 0.58 | 1.71 | 0.65 |
| intermediate | 0.38 | 0.39 | 0.58 | 1.50 | 0.59 |
| novice | 0.46 | 0.48 | 0.50 | 1.07 | 0.52 |

Sanity checks in the diagnostic: the oracle filter has precision and recall 1.00, the random filter
has precision 0.41 against a base rate of 0.50.

## 3. Does using the score help?

**Discard against re-weight (`two_arm_confirm_new`):** the same 9 operators × 2 seeds.

| | discard (`drop-hind`) | re-weight (`reweight-hind`) | paired, re-weight − discard |
|---|---|---|---|
| ruler SR | 93.2 % | 90.3 % | **−2.9 points**, CI −13.4 to +6.2; 8 / 4 / 6; p 0.39 |
| expert / intermediate / novice | 92.6 / 93.2 / 93.8 % | 92.0 / 95.7 / 83.3 % | |
| SR of the last checkpoint ¹ | 97.5 % | 87.0 % | -10.5 points, CI -25.3 to +1.9; 4 / 4 / 10; p 1.00 |
| steps to converge | 9 264 | 9 359 | +94, CI −1 361 to +1 411 |
| first unaided success (median) | 4 174 | 5 442 | later in 13 of 18, p 0.096 |
| operator share | 41.1 % | 39.4 % | −1.6 points; lower in 13 of 18, p 0.096 |
| SD of SR over cells | 0.107 | 0.171 | |
| ICC | 0.000 | 0.000 | |

¹ The policy training stopped on, 9 trials per cell, from `cells.csv`; a secondary measure, not fixed in advance. The
ruler SR is the best window of three checkpoints.

The arm names are historical: both used the behavioural score, not hindsight. Re-weight's priority was
positive for 48 % of takeovers (54 % of attentive ones, 35 % of inattentive ones).

**The perfect-filter check (`headroom_diagnostic_new`, 3 pairs: one operator per tier, one seed):**

| arm | expert | intermediate | novice | mean | steps to converge | operator share |
|---|---|---|---|---|---|---|
| keep-all | 100 % | 100 % | 96.3 % | **98.8 %** | 9 163 | 42.8 % |
| oracle drop | 100 % | 100 % | 81.5 % | **93.8 %** | 8 177 | 45.7 % |
| random drop | 100 % | 96.3 % | 77.8 % | **91.4 %** | 8 634 | 43.5 % |

Oracle − keep-all −4.9 points (0 / 1 / 2), random − keep-all −7.4 (0 / 2 / 1), oracle − random +2.5
(2 / 0 / 1). Three pairs show direction, not significance, and the ceiling leaves little room.
All differences come from one novice cell per arm.

**The pilot (`two_arm_pilot_new`, 3 pairs):** discard 97.5 %, re-weight 96.3 %.

## 4. Why this was a weak test

1. **The filter started too late.** The 5 000-step warm-up covered 61 % of the takeovers and 58 % of
   the operator steps, and half the cells succeeded unaided before it ended. It dropped **17 % of all
   takeovers, not 40 %** (25 % of operator steps; 11 / 14 / 26 % of takeovers for expert / intermediate /
   novice). In some cells it barely acted: `drop-hind_exp02_s0` had 4 takeovers after the warm-up.
2. **All operators with one seed shared one condition sequence: a bug, now fixed.** `_cond_rng` depended
   only on the seed, so 3 357 takeovers carry just 2 distinct sequences. Their first 25 takeovers are about
   10 % inattentive, and the share is 26 % before the warm-up ends and 38 % after it. So early learning,
   when corrections matter most, saw few bad corrections, and the condition is confounded with training
   phase. Patch 0021 makes the sequence depend on seed and operator, keeping the arms of one operator and
   seed paired.
3. **The ruler is near its ceiling and noisy.** The same cell run twice, with identical operator, demos
   and seed, gave SR 1.00 and 0.67. Across pairs the two arms' SR correlates at −0.21, so pairing does not
   help; ±5 points would take about 75 pairs. Steps to converge pair better (+0.56; about 38 pairs for
   ±1 000 steps), so they are the better primary measure on this stack, with the operator share.
4. **The stop rule catches the policy mid-swing.** Neighbouring checkpoints often differ wildly (for
   example 0.00, 1.00, 1.00; 6.6 % of the steps change by at least 0.5). Of the 5 cells below 0.8, 3 were at
   0.89 or more on their last checkpoint. Three cells, all re-weight, had collapsed to 0.33 or less at the
   stop (`int02_s1`, `nov00_s1`, `nov01_s0`).
5. **No control at scale:** keep-all ran on 3 pairs, plain HIL-SERL on none. The lean design compares two
   uses of the score, not the score with nothing.
6. **BC weight was 0.** These arms use replay only. Whether a score helps imitation is untested here.

## 5. Against the first machine's oracle campaign

`aqi_oracle_quality` on the first machine (old stack, consistent operator) concluded that even a perfect
score amplifies operator dependence, because its novice falls from 65 to 32 %. The two results differ, and
**the first machine's conclusion is not yet established**, because its quality arms changed three things
at once: the oracle score, the default hard filters (`outcome_filter` drops every takeover from a failed
episode, whatever the oracle says) and a BC loss of 0.1. See [docs/novice_ablation.md](../novice_ablation.md).

Here, without those two extras, score-based discarding kept novices at 93.8 %. That is suggestive, but it
is another stack and another machine, and only 26 % of novice takeovers were dropped. The novice test on
the first machine separates the three parts on the old stack, on the code that ran the campaign.

## 6. Verdict

| question | answer | strength |
|---|---|---|
| Does the inconsistent operator work as designed? | yes, for expert and intermediate; no effect for novices in training | strong |
| Do scores reflect correction quality? | moderately for expert and intermediate; not for novices; mostly via takeover length | moderate; label caveat (2 sequences) |
| Is the counterfactual score useful? | no: at chance on a timing-free label too | strong |
| Which use of the score is better? | no detectable difference | weak: wide CI |
| Does quality beat the baseline? | not tested | – |
| Does quality help more with inconsistency? | not tested (no consistent-operator run) | – |

## 7. Q&A log

**Did the experiment test the hypothesis?** Only in part. It tested whether the scores notice the bad
corrections and which use of the score is better. "Beats the baseline" and "helps more because the
operator is inconsistent" need `hilserl`, keep-all and a consistent-operator arm, which the two-arm design
left out on purpose.

**Why the behavioural score if its verdict was "weak"?** It was the best score that can run online. The
original gate stopped there; the two-arm run went ahead on the user's explicit decision, with the weak
verdict kept. No threshold was lowered afterwards.

**Why was the keep-all / oracle / random check run?** Not in the plan: the machine's Claude proposed it after
Q1 and the user agreed. It asks whether removing known-bad corrections can help at all.

**Why did novices show no effect from the condition?** Because inattentive execution equals the novice's
own execution. Only the aim offset and premature commits differ, and novice takeovers mostly fail anyway
(23 % episode success).

**Why does `drop-hind` not use hindsight?** The name dates from before Q1. Only `quality_score_mode` was
edited, so that the arms stay exactly as designed.

**Can these cells be merged with new ones?** Not with cells from other machines, and not for label analyses
with new code, since the sequences have changed.
