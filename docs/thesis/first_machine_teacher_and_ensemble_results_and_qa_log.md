# Does the teacher matter, and can many critics learn? The first machine's first campaigns — results & QA log

Companion to [first_machine_synthesis.md](first_machine_synthesis.md). This log explains, in the style of the other
explained logs, **how operator dependency is measured, what went wrong in the first week, what the two oracles are, and
what the critic ensemble is, why it first failed and how it was fixed.** It covers the teacher-dependency campaigns A1–A4
(4–10 September) and the ensemble campaigns B1 and B2 (18–23 September). Numbers are from the thesis (chapter 6), the
reports in `presentation_bundle/results/` and the W&B export.

- Machine: the first training machine, original platform, cuboid peg, frozen 9-pose ruler.
- Reports: `presentation_bundle/results/{rq1_geometric, rq1_policy_oracle, rq3_structural, rq4_convergence}/`.
- Raw training logs: `wandb_analysis/` (B1 and B2, every episode).
- The machine's own logs: [rq1_campaign_engineering_log.md](rq1_campaign_engineering_log.md),
  [rq1_oracle_setup_and_qa_log.md](rq1_oracle_setup_and_qa_log.md),
  [rq3_structural_ablation_and_qa_log.md](rq3_structural_ablation_and_qa_log.md),
  [rq4_ensemble_convergence_and_qa_log.md](rq4_ensemble_convergence_and_qa_log.md) (old names: `rq3_*` = B1, `rq4_*` = B2).

---

## 0. Headline

- **The teacher matters to plain HIL-SERL.** With one operator per tier and three seeds, success falls from 100 % (expert)
  over 82 % to 59 % (novice); ICC 0.60, p = 0.035. With human-like demonstrations it is extreme: novice 6 %, ICC 0.99.
- **But not equally in every campaign.** With five seeds the ICC is 0.30 (p = 0.10), and demonstration pre-training removes
  it (ICC 0.00). In larger later campaigns it is 0.00–0.30, and 0.66 in the asking-rate campaign.
- **The first week measured nothing,** because the ruler scored every cuboid policy on a cylinder. Fixing that, not any
  change to the operators, made the effect appear.
- **The default critic ensemble hurt:** −18.5 points against HIL-SERL, and it *created* a teacher effect (ICC 0.03 → 0.74).
- **Four fixes brought it to parity:** 82.3 against 84.2 %, p = 0.30, 16 of 18 cells converging in both.

## 1. How "does the teacher matter" is measured

**A cell** is one training run: one method, one synthetic operator, one seed. **A campaign** is a grid of cells, for example
3 tiers × 1 operator × 3 seeds = 9 cells.

**The ruler** is the final exam: the trained policy alone, no operator, deterministic actions, from 9 fixed start positions
(−16 to +16 mm beside the hole), 300 steps each. The score is the share of the 9 that end with the peg inserted, taken over
the best window of 3 consecutive checkpoints.

**ICC (intraclass correlation)** is the share of the success variance that comes from *which operator* trained the policy,
as opposed to seed-to-seed noise. ICC 0 = the operator explains nothing beyond noise; ICC 1 = everything. A **permutation
test** shuffles the operator labels 10 000 times and asks how often chance produces a spread as large as the real one;
p < 0.05 = the operator matters.

**Paired comparisons.** Methods share operators and seeds, so "method A − method B" per cell cancels operator and seed
effects. The exact **sign test** counts wins and losses.

## 2. The two oracles, and where the operator's skill enters

A synthetic operator is an **oracle** that knows the correct move, degraded by skill-dependent perception error, decision
noise (when to take over) and motor noise. Two oracles exist:

| oracle | what it is | used for |
|---|---|---|
| geometric (path planner) | an analytic controller that reads the true peg position from the simulator | all interventions in every campaign; demonstrations everywhere except A4 |
| policy | a trained expert policy acting from the same camera images, with wrong-intent spells | the demonstrations of A4 only |

Skill enters training through two channels: the **demonstrations** (6 per operator, recorded before training) and the
**interventions** (takeovers during training). Both go into the half of every batch that comes from the demonstration
buffer.

## 3. The first week: a measurement bug (4–8 September)

For about a week every attempt to see an operator effect failed: time vs step budgets, checkpoint selection, new operator
anchors, demonstration pre-training, more seeds, learning rates. The log concluded at the time that the geometric oracle
was the problem.

**The real cause:** the evaluation script built its environment from the default configuration, whose peg is the
**cylinder**, and had no option to change it. Every cuboid policy was scored on an easier, different task. After the ruler was
bound to the training geometry, re-scoring the same checkpoints turned a flat scatter into 100 / 82 / 59 %.

Three lessons of that week stayed in the protocol:
- a wall-clock budget gave experts 37 % fewer gradient steps than novices, so budgets are in environment steps;
- policies oscillate between checkpoints, so the ruler takes the best window of three, not the last checkpoint;
- one seed per operator is noise; at least three are needed.

## 4. The teacher-dependency campaigns (A1–A4)

| ID | configuration | seeds | ICC | p | expert / intermediate / novice | reading |
|---|---|---|---|---|---|---|
| A1 | geometric operators | 3 | 0.604 | 0.035 | 1.00 / 0.82 / 0.59 | dependency |
| A2 | geometric operators | 5 | 0.301 | 0.100 | 0.92 / 0.92 / 0.70 | direction only |
| A3 | geometric + demonstration pre-training | 3 | 0.000 | 0.453 | 0.95 / 0.82 / 0.77 | flattened |
| A4 | policy-oracle demonstrations | 3 | 0.992 | 0.004 | 1.00 / 0.86 / 0.06 | strong dependency |

**Reading.**
- **A1:** all three criteria met (ICC, permutation test, a monotone gradient).
- **A2:** same direction, smaller gap, not significant. The size of the effect is uncertain; its direction is not.
- **A3:** pre-training actor and critic on the demonstrations before online training removes the dependency: the novice
  rises from 0.59 to 0.77, the expert drops slightly to 0.95. A policy that has already imitated relies less on the
  interventions, the channel through which skill enters. So pre-training is a cheap baseline mitigation.
- **A4:** when a novice's demonstrations are consistent with the camera images (policy oracle), they wreck training: the
  novice policy reaches 6 % and never converges. Interventions still came from the geometric oracle, so this shows how
  strongly demonstration quality propagates.

**Across later campaigns** (HIL-SERL as reference): ICC 0.04 (C1), 0.30 (C2), 0.03 (B1), 0.00 (B2), 0.66 (D3). The order
expert ≥ intermediate ≥ novice holds in A1–A4, B1, B2 and C1–C3; in D3 one intermediate operator (int00, 28 %) pulls its tier
below the novices (64 against 77 %). Seed variance is large: identical HIL-SERL configurations reached 82 and 95 % in C2
and C3. So a single campaign's ICC is unstable, and a pooled model over all HIL-SERL cells (A5, no training) is planned.

## 5. The critic ensemble: what it is

HIL-SERL learns a **critic** Q(s, a), the reward to expect after taking a in s. It uses 2 critics and takes the
**minimum** of the two as its learning target, a standard guard against over-estimation.

The **ensemble extension** uses K critics (K = 10 by default), because the AQI methods need their disagreement σ_Q (for
asking) and their mean (for the counterfactual score). Its default learning target follows REDQ: the **minimum over M = 2
randomly chosen critics** out of the ten.

## 6. Why the default ensemble failed (B1, 18–21 September)

B1: HIL-SERL, ensemble extended (K 10, M 2) and quality, 2 operators per tier × 2 seeds, 12 cells each, 50 k cap.

| method | success | ICC | p (operator matters) | expert / intermediate / novice | converged | vs HIL-SERL |
|---|---|---|---|---|---|---|
| HIL-SERL | 86.7 % | 0.03 | 0.40 | 94 / 89 / 77 | 12 / 12 | – |
| ensemble extended | 68.2 % | 0.74 | 0.02 | 97 / 61 / 46 | 7 / 12 | −18.5 (p 0.75) |
| quality | 50.9 % | 0.89 | 0.007 | 92 / 57 / 5 | 7 / 12 | −35.8 (1 / 11, p 0.006) |

**The mechanism.** The minimum over two of ten critics is dominated by the most pessimistic one. That critic undervalues
the risky action of pressing the peg into contact. The policy then hovers above the hole and succeeds only when an operator
performs the insertion. The better the operator, the less the policy has to master alone, so the ensemble **creates** a
teacher effect.

**The training logs agree** (W&B): the ensemble's mean predicted value stays below HIL-SERL's in every phase, and is still
negative between 15 000 and 30 000 steps although every reward is 0 or 1. With novices, the ensemble never becomes autonomous:
in the last 20 episodes the novice still took over in 55 % of the episodes, 85 steps per episode (HIL-SERL: 10 %, 8 steps).

## 7. The four fixes (B2, 21–23 September)

| fix | from → to | why |
|---|---|---|
| target critics | min over 2 of 10 → one random critic (M = 1) | removes the pessimism that suppressed pressing in |
| ensemble size | 10 → 8 | slightly smaller |
| warm start | none → 5 000 gradient steps on the demonstrations | starts from successful behaviour |
| learning rate | 3e-4 → 1.5e-4 | slower, more stable |

They apply to the ensemble arm only (per-arm overrides); HIL-SERL stays plain. B2: 3 operators per tier × 2 seeds, 18 cells
each, 40 k cap.

| | ensemble, before (B1) | fixed ensemble (B2) | HIL-SERL (B2) |
|---|---|---|---|
| success | 68.2 % | **82.3 %** | 84.2 % |
| ICC | 0.743 | **0.000** | 0.000 |
| converged | 7 / 12 | 16 / 18 | 16 / 18 |
| expert / intermediate / novice | 97 / 61 / 46 | 99 / 91 / 57 | 93 / 85 / 75 |
| mean steps to converge | 37 800 | 32 500 | 27 700 |
| vs HIL-SERL | −18.5 | −1.9 (median +3.7, p 0.30) | – |

**Reading.** Parity: statistically equal success, same convergence, no teacher effect. With expert and intermediate
operators the fixed ensemble is better (95.1 against 88.6 %, +6.5 points, 7 / 2 / 3, p 0.18). Its weak point is the novice
tier (57 against 75 %), driven by seed variance: one novice reaches 3.7 % with one seed and 100 % with the other. It needs
about 5 000 more steps to converge. Its critic loss stays at or below 0.02 in every run. From here on, every ensemble arm
uses this configuration.

**Which fix did it?** Not known: the four changed together. The attribution sweep B3 (72 cells) is configured, not run.
It is optional, because the later questions only need a competitive ensemble.

## 8. What this means

1. **RQ1 is answered with a caveat.** The teacher matters, most when demonstrations are realistic. A single campaign
   with three operators per tier can show anything from 0 to 0.7, so the thesis relies on paired designs and the planned
   pooled model.
2. **RQ2 is answered.** The ensemble needed for asking and grading is a fair learner after the fixes, so later effects are not
   artefacts of a weaker learner.
3. **Threat to validity** (found later, on the second machine): this operator could rarely rescue a twisted cuboid. The
   comparisons are fair, but the size of the teacher effect may change with an operator who can help. The repeat on the new
   stack (A7) tests it.

## 9. Q&A log

| question | answer |
|---|---|
| What exactly is ICC 0.60? | 60 % of the variance in final success is explained by which operator trained the policy; the rest is seed noise. |
| Why is A4 so extreme? | Its demonstrations came from the policy oracle, which acts from the camera images like the policy does. A novice's bad demonstrations are then very "learnable", and the policy copies them. |
| Why does pre-training remove the effect? | The policy imitates the demonstrations first and then relies less on interventions, the channel that carries the operator's skill. |
| Why did more seeds (A2) weaken the result? | With more seeds the noise estimate grows and the novice-expert gap shrank (0.70 against 0.92); the direction stays. |
| Why would a worse critic create dependence? | A pessimistic critic does not learn to press in, so the policy needs the operator for the insertion, and a better operator does it more often. |
| Is the fixed ensemble worse for novices? | 57 against 75 %, but with large seed variance (one novice: 3.7 and 100 %). Not significant. |
| Was the roster correct in these campaigns? | A bug could change operators between processes in campaigns run in more than one go; the roster audit on this machine found all finished campaigns consistent. |

## 10. Artifacts and how well it is checked

| what | where | checked |
|---|---|---|
| A1, A4, B1, B2 reports | `presentation_bundle/results/` | read |
| A2, A3 | the machine's logs only | log only |
| B1, B2 training logs | `wandb_analysis/` (169 runs, every episode) | checked |
| the first week's attempts | `rq1_campaign_engineering_log.md` | log |
