# Can corrections be graded, and does asking help? The first machine's September campaigns — results & QA log

Companion to [first_machine_synthesis.md](first_machine_synthesis.md). This log explains **quality-prioritised replay,
the counterfactual score and why it is inverted, the behavioural score and the timing audit that undid it, how σ_Q was
tested against physical trouble, and the first asking experiment.** It covers C1–C3 (10–18 September), C4–C6
(23–30 September), D1 (1–2 October) and D2 (2–6 October). The follow-ups have their own explained logs:
[first_machine_asking_rate_results_and_qa_log.md](../first_machine_asking_rate_results_and_qa_log.md) (D3, with the details of
σ_Q and the request gate) and [first_machine_oracle_quality_results_and_qa_log.md](../first_machine_oracle_quality_results_and_qa_log.md)
(C11). Numbers are from the thesis (chapter 6), the reports in `presentation_bundle/results/` and the W&B export.

- Reports: `presentation_bundle/results/{quality_on_rq4_diag, quality_behavioral, active_validity, active_effect}/`.
- The machine's own logs: [rq2_quality_replay_and_qa_log.md](../rq2_quality_replay_and_qa_log.md) (C1–C3, old name `rq2_*`),
  [rq4_ensemble_convergence_and_qa_log.md](../rq4_ensemble_convergence_and_qa_log.md) §11–12 (C4, C5).
- Tool: `examples/quality_signal_audit.py` ([../quality_signal_audit.md](../../quality_signal_audit.md)).

---

## 0. Headline

- **Quality-prioritised replay never helped.** It lost to plain HIL-SERL in all three versions (−21, −33, −13 points) and,
  on the same ensemble, cost 17 more points and raised the teacher effect.
- **The counterfactual score is inverted.** On a healthy critic it prefers corrections from failed episodes (AUROC 0.18).
  The critic rates its own action above the operator's.
- **The behavioural score looked like the fix (0.78) and was timing leakage.** On takeovers the operator ended itself it is
  a coin flip (0.50); knowing only *when* a takeover started does better (0.83).
- **σ_Q is not a trouble detector** (jam 0.51), but asking for help when it is high works: equal success to the operator's
  own monitor (71 against 78 %, p 0.58) at one sixth of the help, and no measurable teacher effect.

## 1. How a quality score enters training (quality-prioritised replay)

Each takeover is graded; the grade q of a kept takeover raises its **replay priority**, so it is sampled more often:

    priority = (ε + |TD error| + η · max(q, 0))^α        (prioritised experience replay)

An imitation loss can also weight takeovers by q, but C1–C3 used replay only. Which score produces q is the question of the
next sections.

| stage | what changed | what happened | vs HIL-SERL |
|---|---|---|---|
| C1 inert | the raw score | the score was about 0.001; the constant δ_init = 1 dominated, so priorities rose by at most 3 %. C1 measured the switch from 2 to 10 critics, not the score. | −21.0 (p 0.51) |
| C2 standardised | score divided by its running scale, smaller floor | priorities 2.4–32× the baseline: a few spuriously high takeovers replayed up to 32× more often; the intermediate collapsed to 6 % on all three seeds | −33.3 (p 0.69) |
| C3 stabilised | + warm-up, moving-average scale, periodic refresh, importance weights | no collapse (the intermediate back at 81.5 %), the most operator-robust arm so far, but it lost in all 6 matched cells | −13.0 (p 0.031) |

One operator per tier, three seeds, the original 10-critic ensemble (K 10, M 2). At equal architecture (B1, the same
ensemble without and with quality replay), quality cost another 17.3 points (68.2 → 50.9 %), raised the ICC from 0.74 to 0.89,
and the novice tier collapsed to 4.6 %.

## 2. The counterfactual score, and why it is inverted (C4)

**Definition.** For every step of a takeover, the critics compare the operator's action with the robot's own action in the
same state:

    Â = μ_Q(s, a_human) − μ_Q(s, a_robot)

(μ_Q = the critics' mean; a_robot = one sample of the policy). Positive should mean "the correction beat what the robot would
have done". The score of a takeover is the positive part, damped where the critics disagree.

**C4:** the score on the fixed, healthy ensemble of B2 (1 operator per tier, 1 215 takeovers). It stayed around 0.001,
flat across tiers, and its AUROC against episode success was **0.175**: it ranks corrections from failed episodes higher.
64 % of the raw Â were negative. In the large asking campaign D2 it was 0.19 on 4 386 takeovers.

**Why.** Three mechanisms bias Â against the operator:
1. **The human action is off-distribution.** Each correction is one noisy transition at its state; the critic is reliable
   only where data are dense around the evaluated action, so there it extrapolates.
2. **The robot's action is evaluated optimistically** (probably decisive). The policy is trained to maximise this same
   critic, so it moves towards actions whose value is high, including wrongly high. Q(s, a_robot) is biased upwards, and Â ≤ 0
   even for good corrections.
3. **Recoveries are rare in the data.** A good takeover often starts by backing off or re-aligning, away from the hole. The
   critic has rarely seen that end in success, so it rates it low; failing episodes contain more such recoveries.

Training the critic longer on the same data removes none of the three, which is why the healthy critic did not help. A
pre-registered study on the cluster (C9, 29 September) later tested five better-built variants against simulator ground
truth: all below chance, so the score is abandoned ([../counterfactual_score_study.md](../../counterfactual_score_study.md) §5).

## 3. The behavioural score, and the timing audit (C5, C6)

**Definition.** A critic-free score from what can be observed about a takeover:

    B = z(ΔV) − z(t_int) − z(F_peak)

ΔV = the critics' value at the hand-back minus at the takeover; t_int = its length; F_peak = its peak force; each standardised
online. A short, value-raising, gentle takeover scores high. Run live (C5, 1 106 takeovers) it reached **0.78** against
episode success, within every tier.

**The audit (C6, 30 September)** asked whether that comes from *when* a takeover happens rather than *how good* it is. The
synthetic operators hand back after at most d_max seconds; late takeovers are cut by the cap and by the episode end, and late
in an episode things are already going badly. Five controls:

| score | all | clean (released by the operator's own hand-back, n 231) | reading |
|---|---|---|---|
| counterfactual Â | 0.15 | 0.15 | inverted |
| behavioural B | 0.73 | **0.50** | timing leakage |
| B without length | 0.63 | 0.49 | no clean signal |
| −length only | 0.70 | 0.56 | timing |
| hindsight | 0.63 | 0.59 | state difficulty (r = −0.81 with the value at onset) |
| progress after release | 0.47 | 0.42 | noise |
| takeover start time only | **0.83** | 0.69 | the yardstick: beats every score |
| σ_Q at the takeover | 0.83 | **0.85** | survives |

61 % of takeovers ended at the cap, and the takeover length correlated perfectly with the operator's cap (r = 1.00): length
re-encodes the operator's tier and the time of the takeover. **No tested score ranks corrections by their value** once timing
and difficulty are controlled. The only quantity that survives is σ_Q, a property of the *situation* at which help was given,
not of the help. That moved the effort from grading to asking.

## 4. Does σ_Q detect physical trouble? (D1, 1–2 October)

**Design.** Nine cells in `active` mode (the robot asks when σ_Q is high; 3 operators per tier, 1 seed, about 297 000 steps)
that log the simulator's ground truth at every step: true lateral error, height, jam flag. Rule fixed in advance: AUROC against
physical trouble ≥ 0.65 pooled, ≥ 0.60 time-stratified, > 0.55 in at least two tiers.

| signal | label | pooled | contact zone (\|h\| < 8 mm) |
|---|---|---|---|
| σ_Q | jam | 0.43 | **0.51** |
| request score | jam | 0.45 | 0.51 |
| −LCB (pessimistic value) | jam | 0.69 | 0.57 (height confound) |
| σ_Q | operator takeover | 0.65 | 0.62 (circular) |

**Reading.** σ_Q does not detect jams. It "predicts" takeovers only because in active mode the operator obeys the gate's
own request (circular; any gate would pass); where the operator uses its own monitor, σ_Q against takeover onsets is
0.49–0.52. The rule is not met. Yet the policies reached 81.1 % with only 8.9 % operator control. So the question changed:
not "does σ_Q detect trouble?" but **"does asking when σ_Q is high train a good policy?"**

## 5. Does asking pay off? (D2, 2–6 October)

**Design.** Four arms that differ only in what decides when the operator helps; the fixed ensemble, the same operators (3 per
tier, 2 seeds), demonstrations, correction quality and ruler; 72 cells, 40 k cap.

| arm | what decides | operator share | success | expert / int. / novice | ICC | p (op.) | converged | vs ensemble extended |
|---|---|---|---|---|---|---|---|---|
| ensemble extended | the operator's own monitor | 54 % | 78.2 % | 93 / 90 / 52 | 0.84 | 0.018 | 15 / 18 | – |
| active | σ_Q gate (fixed threshold) | **8.9 %** | 70.8 % | 76 / 74 / 63 | **0.00** | 0.92 | 10 / 18 | −7.4 (5/8/5, p 0.58) |
| random | random requests, similar rate | 11.9 % | 56.2 % | 58 / 54 / 57 | 0.51 | 0.062 | 10 / 18 | −22.0 (2/13/3, p 0.007) |
| active-loud | σ_Q gate, louder ensemble | 7.0 % | 63.6 % | 68 / 68 / 55 | 0.00 | 0.89 | 5 / 18 | −14.6 (4/11/3, p 0.12) |

**Reading.**
- **Asking with σ_Q is as good as the operator's own judgement** at one sixth of the help, and the teacher effect disappears
  (ICC 0.84 → 0.00).
- **Random requests at a similar rate are significantly worse,** which suggests the timing matters. But the direct contrast
  active − random (+14.6 points, 9 wins, 8 losses, sign test p 1.00, Wilcoxon 0.40) is not significant.
- **The louder ensemble** (each critic sees half the data) asked less and learned less.

**Three qualifications.**
1. **The ICC of 0 is partly flattening and noise:** experts lose (93 → 76 %), novices gain (52 → 63 %), and one expert cell
   swung from 100 to 0 % between its two seeds.
2. **Gated arms converge less often** (10 against 15 of 18), partly because the convergence rule counts no episode with a
   takeover as autonomous; 8 of 16 gated runs that never "converged" scored 33–93 % on the ruler.
3. **Slow operators miss short requests.** The request lasts 10 steps; experts answered about 77 % of the requests,
   intermediates 47 %, novices 2–9 %, and 68 % of novice takeovers were spontaneous. Novice-trained policies therefore learned
   almost without novice corrections. The operator's share of each training batch fell from about 74 % to 37 %.

**Two more observations from the W&B logs.** The gate asks *least* where it is most needed: a policy stuck in familiar states
produces little disagreement, so the gate stays quiet when the policy is confidently wrong. And the fixed threshold fired at a
rate nobody chose (about 9 %), which led to the quantile gate of the asking-rate campaign.

## 6. What this means

1. **Grading (RQ3):** not supported. Every tested score fails for a specific, diagnosable reason: the counterfactual one by
   construction, the behavioural one by timing leakage, hindsight measures difficulty, progress is noise. The thesis keeps
   the null result and a uniform-priority control.
2. **Asking (RQ4):** the gate is not a detector but works as a trigger; the follow-up (D3) showed that asking more often
   beats the operator's own monitor ([first_machine_asking_rate_results_and_qa_log.md](../first_machine_asking_rate_results_and_qa_log.md)).
3. **What stayed open:** whether grading can pay off when an operator's corrections really differ in quality (the
   inconsistent operator of the second machine), and whether a perfect score helps (C11, confounded; the novice test decides).

## 7. Q&A log

| question | answer |
|---|---|
| Why did quality replay lose even when it did not collapse? | Its score was inverted or meaningless, so it replayed the wrong takeovers more often. |
| Why is Â mostly negative? | The policy is trained to maximise the same critic, so its own action looks good to that critic by construction. |
| Why was the behavioural score convincing at first? | Short takeovers come from good operators and early, successful episodes; the score learned *when*, not *how good*. |
| What is "clean"? | Takeovers the operator ended itself, not cut by the time cap or the episode end. Timing cannot leak there. |
| If σ_Q is not a detector, why use it? | Because asking when it is high trains policies as good as the operator's own monitor at one sixth of the effort. |
| Is "one sixth of the help" a real saving? | Partly: novices answered few requests, so their help mostly dropped out. A held request will test this. |
| Why is "random worse than the monitor" not proof that timing matters? | The direct comparison of the gate with random requests (9 wins, 8 losses) is not significant. |

## 8. Artifacts and how well it is checked

| what | where | checked |
|---|---|---|
| C4, C5, C6, D1, D2 reports | `presentation_bundle/results/` | read |
| C1–C3 | the machine's log (C3 also in W&B) | log; C3 checked |
| D1, D2 training logs, answered shares | `wandb_analysis/` | checked |
