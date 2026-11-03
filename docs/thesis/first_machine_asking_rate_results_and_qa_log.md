# How often should the robot ask? The critics' disagreement σ_Q and the asking-rate campaign — results & QA log

Companion to [first_machine_synthesis.md](first_machine_full_history/first_machine_synthesis.md). This log explains **what σ_Q is, how it becomes
a request for help, how a synthetic operator answers**, and what the asking-rate campaign `active_ratematch` (thesis D3)
showed. It is written from the code, the campaign config and the report, and adds measurements made here from the
report's per-takeover table (section 6).

- Machine: the first training machine, original platform (the operator moves the peg but cannot turn it), cuboid.
- Campaign: `runs/active_ratematch`, 108 cells, finished 2026-10-21. Config `examples/campaigns/active_ratematch.yaml`.
- Report: `OneDrive/active_ratematch_report/` = `presentation_bundle/results/active_ratematch/` (`dependency.md`,
  `quality_audit/summary.md` and `segments.csv` with 37 565 takeovers, `plots/`).
- Code: `serl_launcher/aqi/uncertainty.py` (σ_Q), `query_gate.py` (the request), `operator_gate.py` (the operator's answer),
  `runtime.py` (the per-step loop).
- The machine's own log of the same campaign: [active_ratematch_results_and_qa_log.md](active_ratematch_results_and_qa_log.md).

---

## 0. Headline

- **Asking more buys more success.** The robot asking in the 10, 30 and 55 % most uncertain steps reaches 70.2, 81.3 and
  87.4 % ruler success at 15.9, 28.5 and 41.8 % operator share of the steps.
- **Asking in the 55 % most uncertain steps beats the same critics with the operator deciding:** 87.4 against 74.3 %,
  +13.2 points, 13 wins / 3 losses / 2 ties, p = 0.021, with less help (41.8 against 53.8 %).
- **Against plain HIL-SERL it is +8.0 points, not significant** (p = 0.18), with a fifth less help; almost all of the
  gain is among intermediates.
- **The teacher hardly shows when asking most:** ICC 0.01, against 0.66 for HIL-SERL (p = 0.001).
- **Why there is less help, and why the teacher disappears:** experts answer the robot more than they would step in on
  their own (63.8 against 60.5 % of the steps); novices answer little (13.4 against 46.9 %). Novices make about 85
  takeovers per run under the gate against about 936 for experts, and only 38–60 % of their takeovers answer a request.
- **The timing of σ_Q beats random requests at the same rate by 4.9 points,** not significantly (p = 0.30).

## 1. The critics' disagreement σ_Q

**What a critic is.** A critic is a network that predicts Q(s, a): how much reward to expect from state s if the robot
takes action a now and then follows its policy. HIL-SERL has 2 critics. The ensemble extension has **8** (K = 8), trained
on the same data but with different initialisations and their own learning targets (`per_member_targets`, one randomly
chosen member as target, M = 1).

**What σ_Q is.** At every step the runtime asks all 8 critics for the value of the action the robot is about to take,
a_robot, and computes

    μ_Q(s, a) = mean over the 8 critics of Q_i(s, a)
    σ_Q(s, a) = standard deviation over the 8 critics of Q_i(s, a)

(`uncertainty.py`; `runtime.observe_step` calls it with the robot's own action, also during a takeover, where a_robot is
what the policy would have done.)

**What it means.** Where the critics have seen much data they agree, and σ_Q is small. In unfamiliar states, or for
unfamiliar actions, each critic extrapolates differently, and σ_Q is large. So σ_Q measures how little the robot *knows*
about its next move (epistemic uncertainty), not how badly things are going.

**What it is not.** The validity campaign (D1) compared σ_Q with the simulator's ground truth: it does not detect a jam
(AUROC 0.51 in the contact zone). It also fades as training proceeds, because the critics converge and agree more
everywhere (finding F6). That is why the gate uses a **relative** threshold (section 2), and why the question became
"does asking at high σ_Q train a better policy?" instead of "does σ_Q detect trouble?".

## 2. From σ_Q to a request

`query_gate.py`, configured by `trigger_terms: [uncertainty]` and `trigger_combiner: max`:

1. **Normalise.** z = (σ_Q − running mean) / running standard deviation, over all steps of the run so far.
2. **Score.** With one term, the score A(s_t) is simply z.
3. **Threshold.** With `query_target_rate` r, the threshold is the (1 − r) quantile of the last 2 000 scores. So
   "asks 55 %" requests whenever the current score is above the 45th percentile of the recent scores, "asks 10 %" only
   above the 90th. The rate is set explicitly and does not depend on σ_Q's drifting scale.
4. **Hold and pause.** A request stays on for 10 steps (1 s, `query_min_on_steps`), then the gate waits 5 steps
   (`query_cooldown_steps`) before it can ask again. So at most 10 of every 15 steps can be requested (67 %).
5. **The robot keeps moving** while it asks (`query_behavior: continue`); the operator overrides the moment it steps in.

**The random control** (`random-q30`) has no trigger term, so its score is always 0. It asks by a coin flip with
probability 0.08 at every step where a request may start, with the same 10-step hold and 5-step pause. That gave a
similar operator share to "asks 30 %" (25.7 against 28.5 %), so the two differ only in *when* they ask.

**Why the quantile.** In the first asking experiment (D2) the gate used a fixed threshold, τ_A = 0.5 on the z-score.
Because z is normalised, it fired at a rate nobody chose (about 9 % of the steps). A 3-hour probe before this campaign
measured the rates of the quantile gate (intermediate operator, 8 000 steps):

| setting | operator share in the probe |
|---|---|
| fixed threshold τ_A = 0.5 | 3.7 % |
| asks 10 % (cool-down 5) | 14.1 % |
| asks 30 % | 33.9 % |
| asks 55 % | 43.6 % |
| the operator's own monitor | 59.6 % |

With the default cool-down of 20 steps the three rates collapsed onto one (the 10-of-30 duty cycle caps them), so the
campaign uses a cool-down of 5.

## 3. From a request to a takeover: how the synthetic operator answers

In the asking arms the operator does not watch the task itself. Its takeover gate gets z = 1 while a request is on and
0 otherwise (`operator_gate.py`), and decides as a signal detector would:

- it smooths the input with lag and adds monitoring noise (ẑ = EWMA(z) + N(0, σ_mon));
- it takes over once ẑ has stayed above its threshold δ_in for m_in steps, after a further reaction delay;
- it misses whole events with probability p_miss, and sometimes steps in for no reason (false alarms at rate ρ_fa);
- it hands back when ẑ falls below δ_out for k_out steps, or at its time cap d_max.

Novices have more lag, more noise and slower reactions. **A 10-step request often ends before a novice's smoothed
signal crosses its threshold**, and a pending reaction is cancelled when the signal drops. On the W&B logs of the first
asking experiments the share of requests answered was about 77 % for experts, 47 % for intermediates and 2–9 % for
novices. This is a property of the synthetic operator model, and it matters for every result below.

## 4. Design

| arm | what decides when the operator helps |
|---|---|
| `hilserl` | plain HIL-SERL (2 critics); the operator's own monitor |
| `logged` = ensemble extended | the fixed 8-critic ensemble; the operator's own monitor |
| `active-q10` / `q30` / `q55` | the same ensemble; the robot asks in the 10 / 30 / 55 % most uncertain steps |
| `random-q30` | the same ensemble; random requests at the rate of q30 |

Same 9 operators (3 per tier, sample seed 0), 2 seeds, the same demonstrations (6 per operator) and the same ruler for
all six arms: 108 cells. Cap 35 000 environment steps (40 000 in D2). The ensemble arms start from 5 000 steps of
demonstration pre-training with learning rate 1.5e-4; HIL-SERL stays plain. The operator's corrections come from the
geometric oracle (path planner).

## 5. Results

### 5.1 Success and human effort

| arm | operator share of the steps | ruler success | vs ensemble extended (W/L/T, p) | vs HIL-SERL (W/L/T, p) |
|---|---|---|---|---|
| HIL-SERL | 52.5 % | 79.4 % | +5.1 (9/8/1, 1.00) | – |
| ensemble extended | 53.8 % | 74.3 % | – | −5.1 (8/9/1, 1.00) |
| asks 10 % | 15.9 % | 70.2 % | −4.1 (9/7/2, 0.80) | −9.3 (7/10/1, 0.63) |
| asks 30 % | 28.5 % | 81.3 % | +7.0 (9/6/3, 0.61) | +1.9 (9/5/4, 0.42) |
| **asks 55 %** | **41.8 %** | **87.4 %** | **+13.2 (13/3/2, 0.021)** | +8.0 (10/4/4, 0.18) |
| random 30 % | 25.7 % | 76.3 % | +2.1 (11/5/2, 0.21) | −3.1 (8/9/1, 1.00) |

Paired per cell (same operator and seed), exact sign test. Asks 30 % needs less help than HIL-SERL in all 18 pairs;
asks 55 % needs less than ensemble extended in 13 of 18 (p = 0.096). Dose–response: asks 30 vs 10 % +11.1 (p = 0.057),
55 vs 30 % +6.2 (p = 1.00).

### 5.2 By tier

| arm | expert | intermediate | novice |
|---|---|---|---|
| HIL-SERL | 97.5 | 64.2 | 76.5 |
| ensemble extended | 86.4 | 63.6 | 72.8 |
| asks 10 % | 85.8 | 71.0 | 53.7 |
| asks 30 % | 100.0 | 87.0 | 56.8 |
| asks 55 % | 93.2 | 95.1 | 74.1 |
| random 30 % | 89.5 | 71.6 | 67.9 |

Asks 55 % minus HIL-SERL by tier: −4.3 / **+30.9** / −2.4 points. Asks 30 %: +2.5 / +22.8 / −19.7.

### 5.3 Does the teacher show?

| arm | ICC | p (operator matters) | spread of operator means (CV) | worst operator |
|---|---|---|---|---|
| HIL-SERL | **0.662** | **0.001** | 0.29 | int00, 27.8 % |
| ensemble extended | 0.000 | 0.61 | 0.18 | int02, 55.6 % |
| asks 10 % | 0.000 | 0.50 | 0.30 | int01, 38.9 % |
| asks 30 % | 0.421 | 0.12 | 0.28 | nov02, 37.0 % |
| asks 55 % | 0.010 | 0.38 | 0.15 | nov00, 61.1 % |
| random 30 % | 0.000 | 0.66 | 0.20 | int01, 48.1 % |

The ICC of 0.42 under asks 30 % comes mostly from one novice (nov02: 63 and 11 % on its two seeds). The ICC of a single
campaign with three operators per tier is unstable: ensemble extended had 0.84 in D2 and 0.00 here.

### 5.4 Convergence

HIL-SERL converged in 15 of 18 cells, ensemble extended 11, asks 10 / 30 / 55 % 14 / 8 / 11, random 10. The rolling-window
rule (20 autonomous episodes at ≥ 90 % plus a mini-evaluation) sees fewer autonomous episodes when the operator is asked
more, so the gated arms converge less often; the ruler scores every cell either way.

## 6. The mechanism: who answers the robot (measured here)

From the report's per-takeover table (`quality_audit/segments.csv`, 37 565 takeovers in the five arms that log them;
computed 2026-11-03). "Requested" = the takeover answered a request of the robot; the rest are the operator's spontaneous
takeovers (false alarms in the model).

| arm | tier | takeovers per run | requested | mean length (steps) | in successful episodes | end at the time cap |
|---|---|---|---|---|---|---|
| ensemble extended | expert | 615 | – | 31 | 33 % | 62 % |
| | intermediate | 350 | – | 46 | 11 % | 65 % |
| | novice | 206 | – | 65 | **3 %** | 64 % |
| asks 30 % | expert | 857 | 97 % | 19 | 55 % | 16 % |
| | intermediate | 335 | 89 % | 26 | 20 % | 19 % |
| | novice | **71** | **46 %** | 34 | 8 % | 18 % |
| asks 55 % | expert | 936 | 98 % | 24 | 45 % | 31 % |
| | intermediate | 406 | 91 % | 35 | 20 % | 36 % |
| | novice | **85** | **60 %** | 49 | 6 % | 34 % |
| random 30 % | expert | 992 | 98 % | 15 | 24 % | 5 % |
| | intermediate | 566 | 91 % | 17 | 6 % | 3 % |
| | novice | 72 | 50 % | 31 | 4 % | 8 % |

Operator share of the steps by tier (report): asks 55 % 63.8 / 48.1 / 13.4 %, ensemble extended 60.5 / 53.9 / 46.9 %,
HIL-SERL 59.5 / 55.0 / 43.0 %.

**Reading.**
- The gate does not reduce the experts' help: they take over more often (936 against 615 times per run) and, in sum,
  control more steps than with their own monitor. The "less help" of the asking arms is the novices' missing answers.
- Under the gate, takeovers are short and mostly end when the request ends; with the operator's own monitor about two
  thirds run into the time cap.
- So the near-zero ICC is partly mechanical. Novice-trained cells learn mostly from the robot's own experience and the
  novice's six demonstrations, so they look more like the others. This is one reason the thesis calls the asking result
  "largely supported", not "supported".
- **The same table matters for grading:** 97 % of a novice's takeovers under its own monitor fall in failed episodes
  (67 % for experts). A filter that drops failed-episode takeovers removes almost all novice corrections (see
  [first_machine_oracle_quality_results_and_qa_log.md](first_machine_oracle_quality_results_and_qa_log.md)).

## 7. The quality audit on these runs

`quality_audit/summary.md`: the counterfactual advantage Â is inverted in every arm (AUROC against episode success
0.06–0.16, 0.06–0.13 on takeovers released by hand-back). A score that knows only when the takeover started reaches
0.82–0.89. This replicates the earlier finding on another campaign.

## 8. What this means, and its limits

1. **Asking is the thesis's positive result.** It trades effort for success along a monotone curve, and at the high rate
   it beats the same critics with the operator deciding, with less help and no measurable teacher effect.
2. **It does not yet beat plain HIL-SERL significantly,** and its gain over HIL-SERL comes from one tier.
3. **The timing claim is open:** +4.9 points over random requests at a matched rate (10 / 5 / 3, p = 0.30); D2 pointed the
   same way (+14.6, sign test p = 1.00).
4. **Part of the independence is the operator model** (section 6), so a held request, which waits until the operator
   answers, is the next test (implemented as patch 0025, not run).
5. **Threats:** one campaign; 35 000-step cap; fewer gated cells converge; the operator could not turn the peg (second
   machine). The asking result is to be repeated on the new stack (`C_full`).
6. **It replicates:** the oracle campaign ran asks 30 % again on the same roster: 82.9 % (here 81.3 %), tiers 100 / 91 /
   57 % (here 100 / 87 / 57 %), operator share 28.4 % (here 28.5 %).

## 9. Q&A log

| question | answer |
|---|---|
| Is σ_Q a measure of danger? | No. It measures how much the 8 critics disagree about the robot's next action: unfamiliarity. D1 showed it does not detect jams. |
| Why the robot's own action, not the operator's? | The gate must decide before anyone acts, so it can only ask about what the robot would do next. |
| What does "asks 55 %" mean exactly? | A request opens when the current disagreement is above the 45th percentile of the last 2 000 steps, lasts 10 steps, then a 5-step pause. The realised operator share was 41.8 %. |
| Why not simply a fixed threshold? | σ_Q shrinks as the critics converge; a fixed threshold on its z-score fired at about 4–9 %, a rate nobody chose. |
| Does the robot stop while asking? | No (`continue`): it keeps executing its own action until the operator takes over. |
| Why do novices answer so little? | In the model, a novice's monitoring is slow and noisy; a 10-step request often ends before its smoothed signal crosses the threshold. Real people may differ. |
| Is "less help" a saving? | Partly. Experts help more under the gate; the total falls because novices answer little. |
| Then is the ICC of 0 real? | It is measured correctly, but part of it comes from novices dropping out of the data. A held request will test this. |
| Why does HIL-SERL show a strong teacher effect here but ICC 0.00 in B2? | Single-campaign ICC is unstable with three operators per tier; here one intermediate (int00) failed badly with HIL-SERL. Compare only within a campaign. |
| Which arm goes forward? | Asks 55 %, as the asking arm of the repeat at thesis scale on the new stack. |

## 10. Artifacts and how well it is checked

| what | where | checked |
|---|---|---|
| report, tables, plots | `presentation_bundle/results/active_ratematch/` | read; numbers above copied from it |
| pairings against HIL-SERL and by tier | computed here from the report's per-cell tables | yes |
| who answers (section 6) | computed here from `quality_audit/segments.csv` | yes |
| answered share of requests (77 / 47 / 2–9 %) | W&B logs of D1 and D2 (`wandb_analysis/`) | yes, on those campaigns |
| raw logs and checkpoints of this campaign | first machine only | not here |

Report display bugs (the tables are right): the headline puts a minus sign on ICC increases, the header says 5 operators
per tier (it is 3), and `plots/frontier.png` puts HIL-SERL at 0 % help (it is 52.5 %). Fixed in patch 0013, not yet on
the first machine; use `presentation_bundle/output/_siemens/gen/r_rate.png` instead.
