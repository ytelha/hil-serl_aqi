# The second training machine — synthesis, and what changes on the first machine

**Master analysis of everything that ran on the second training machine** (24–29 October 2026),
what it means, and the plan it leaves for the first machine.

Detailed companions:
- [new_stack_operator_results_and_qa_log.md](new_stack_operator_results_and_qa_log.md): the operator
  alone, the orientation tolerance, training speed on the new stack;
- [inconsistent_operator_results_and_qa_log.md](inconsistent_operator_results_and_qa_log.md): score
  validity, discard against re-weight, the perfect-filter check.

Numbers were recomputed from the raw logs of the export `AQI_HIL_SERL_complete_2026-10-29` and agree
with the machine's own reports.

---

## 0. Executive summary

- **What worked:** the code of patches 0014–0019 on the real stack (136 tests, 57 cells, no crash), and
  above all **rotation control**: cuboid rescue success from 7–13 % to 70–100 %.
- **What looks large but is not attributed:** on the new stack the first success without help comes at
  about 4 200–5 400 steps, against about 17 000 on the first machine. Five changes and the machine differ.
- **What did not show a benefit:** using a quality score. Discard vs re-weight: −2.9 points, p 0.39.
  Perfect filter vs keep-all: no gain on 3 pairs.
- **What stays open:** the hypothesis that quality pays off with inconsistent operators. The test was weak
  (late filter, a bug that gave all operators one sequence, ceiling, no control at scale).
- **What it changes:** the operator of every finished cuboid campaign could rarely rescue a twisted peg,
  which is a threat to validity for those campaigns. The first machine's next job is to find out whether
  the conclusions survive with an operator who can.

## 1. What ran

| campaign | what it asks | arms | cells | cap | wall time |
|---|---|---|---|---|---|
| stage 0 (`synthetic_suite`, `inconsistent_operator`) | operator alone; tolerance; tests | – | no training | – | CPU |
| `S_smoke` | does the full stack run end to end? | plain HIL-SERL, cylinder, expert | 1 | 8 k | 0.3 h |
| `Q1_validity_new` | do the scores find the bad corrections? | ensemble extended | 6 (1 per tier × 2 seeds) | 30 k | 9.1 h |
| `headroom_diagnostic_new` | can removing bad corrections help at all? | keep-all, oracle drop, random drop | 9 (1 per tier × 1 seed) | 30 k | 12.4 h |
| `two_arm_pilot_new` | discard or re-weight? | `drop-hind`, `reweight-hind` | 6 (1 per tier × 1 seed) | 30 k | 8.4 h |
| `two_arm_confirm_new` | the same at thesis scale | `drop-hind`, `reweight-hind` | 36 (3 per tier × 2 seeds) | 40 k | 53.8 h |

All on the cuboid and the **new stack**: rotation control, orientation box, orientation channel,
mini-eval convergence, fixed update ratio. The inconsistent operator is on in the last four.

**Not run:** the ladder (`L0`–`L5`, `X_*`) that would attribute the stack's effect; the six-arm Q2 and Q3
at scale; anything on the original platform (it failed the stage-0 rule); plain HIL-SERL on the cuboid.

## 2. What came out

| question | result | strength |
|---|---|---|
| Does the code work on the real stack? | yes: 136 tests, 57 of 57 cells converged, no crash, no deadlock | strong |
| Could the original operator rescue a twisted cuboid? | no: 13 / 10 / 7 %; with rotation control 100 / 100 / 70 % | strong; no learning involved |
| Why? | 21° twist at takeover against 8° yaw tolerance | strong |
| Does the new stack remove the slow start? | first unaided ≈ 4 200–5 400 vs ≈ 17 000; timeouts ≤ 10 k 29–34 % vs 90–99 % | large, but not attributed |
| Is success still skill-graded? | flat in the discard arm (92.6 / 93.2 / 93.8 %); novices need 1.5–1.8 × the steps | suggestive; ceiling |
| Does the orientation channel matter? | it fires in about 1 % of takeovers | suggests not; ladder decides |
| Do scores notice inattention? | behavioural 0.66 (0.71 / 0.67 / 0.52); length alone 0.65; counterfactual 0.49 | moderate |
| Discard or re-weight? | 93.2 vs 90.3 %, −2.9 points, CI −13.4 to +6.2, p 0.39 | no difference shown; wide CI |
| Does a perfect filter help? | keep-all 98.8, oracle drop 93.8, random drop 91.4 % | 3 pairs, ceiling |
| Does quality beat the baseline? | not tested | – |

## 3. What this means

1. **A threat to validity for the finished cuboid campaigns.** They remain correct as paired comparisons
   under equal conditions. But the operator could rarely help, the slow start came largely from that, and the
   measured operator dependency (plain HIL-SERL, ICC 0.66) may partly reflect it. Whether it survives is
   the most important open question on the simulation side.
2. **The quality question is not closed.** The second machine cannot show a benefit, and the first machine's
   oracle campaign reports harm. Both have design limits (below and in
   [docs/novice_ablation.md](../novice_ablation.md)). The honest position today: no evidence that grading
   corrections helps, and a suggestion that aggressive filtering hurts weak operators.
3. **Final success is a poor primary measure on the new stack.** It sits near the ceiling and is noisy. Steps
   to converge and operator share are better.

## 4. What changes on the first machine

**Its state (source files of 2026-10-31):** patches 0001–0010 plus local code (quantile trigger,
random asking, `oracle` score, milder novices, HiRIL fix). It lacks 0011–0019. Nothing is running. E2HiL and HiRIL
have not run. Instructions: `handoff/first_machine/HANDOFF_first_machine.md`.

**Code to move (all off by default; all 33 of its configs keep identical commands):**

| patches | what | why |
|---|---|---|
| 0022 (first) | novice test: config, cell copy, report | must run on the current code |
| 0011–0013 | W&B tags, report fixes | correct metadata and report text |
| 0014 | rotation control | the strongest result |
| 0015 | takeover logging, mini-eval convergence, fixed update ratio | worked in all 57 cells; machine-independent learning |
| 0016 | orientation box and channel, tolerance sweep | box matches the real robots; the ladder decides on the channel |
| 0017 | test suite (ladder, confirmation) | the campaigns below |
| 0018, 0019 | inconsistent operator, drop filters, two-arm configs | the follow-up on quality |
| 0021 | fixes: per-operator condition sequence, report, audit, tests, asking arm in the confirmation | found by the second machine |

Patches 0011–0019 are rebased onto its exact source files, so they apply without conflicts. Patch 0020 is
for the other direction: it brought its local code into the development repo.

**What to take into account in the new campaigns:**

| lesson from the second machine | consequence |
|---|---|
| updates per env step differ between machines | never mix machines; every comparison needs its reference on the first machine |
| fixed update ratio changes the pace | the env waits for the learner; cost per cell must be measured on the smoke test |
| ruler near the ceiling, same cell twice 1.00 vs 0.67 | primary measures: steps to converge, first unaided success, operator share; do not read 3 pairs |
| stopping at mini-eval 0.6 lands mid-swing | the report now shows the last-checkpoint SR; the threshold stays 0.6 until the ladder gives evidence |
| quality warm-up of 5 000 steps outlasts early learning | shorten it for any new filter test |
| one condition sequence per seed (bug) | fixed in 0021; do not merge label analyses with the second machine's |
| orientation channel fires in about 1 % of takeovers; tilt limit is the top of the sweep | ladder includes `X_nochannel`; re-measure limits in stage 0 |
| original platform cannot show the inconsistent operator | use the new stack for it (the driver enforces this) |
| novices show no effect of the condition in training | judge inconsistency on expert and intermediate |

## 5. The campaigns on the first machine

In order. Each step waits for the user's approval. Costs assume about 1.3–1.5 h per cell.

| # | campaign | config | cells | time | question | what follows |
|---|---|---|---|---|---|---|
| 0 | what did the oracle campaign drop? | `novice_ablation_report.py --count` | no training | seconds | was `outcome_filter` the main cause of the dropped novice corrections? | confirms or weakens the doubt before GPU time |
| 1 | **novice test** | `quality_novice_ablation` | 24 new + 12 copied | ≈ 1.5 days | which part collapses the novice: oracle score, hard filters, or BC loss? | decides how the thesis words the quality result |
| – | apply 0011–0021 | throwaway copy first | – | hours | do all configs keep identical commands? | gate |
| 2 | stage 0 + smoke | `synthetic_suite stage0`, `S_smoke` | 1 | ≈ 2 h | same operator results here? measured limits; cost per cell | ladder |
| 3 | **ladder** | `L0_original` → `L5_full`, `X_nobox`, `X_nochannel`, `X_noeval` | 27 | ≈ 40 h | which change removes the slow start? | the stack for everything after |
| 4 | **old vs new stack at thesis scale** | `C_full`; `C_reference` copied from `active_ratematch` | 54 new | ≈ 3–4 days | does the operator dependency of plain HIL-SERL survive? does asking 55 % still win? | re-reads the main results of the thesis |
| 5 | clean quality test (if step 1 clears the score) | keep-all vs oracle drop on the chosen stack, new config | 36 | ≈ 2–3 days | can removing bad corrections help, with a short warm-up and per-operator sequences? | if yes: scores on real data; if no: close the line |
| 6 | E2HiL and HiRIL comparison | `baseline_vs_e2hil_hiril` (on the chosen stack) | 54 new | ≈ 3–4 days | how does the approach compare with the state of the art? | RQ5 |

Held until step 1 is read: `active_mild_novice`, `active_demos_sweep`. Optional: `rq4_attribution_sweep` (72 cells).

**Reading step 1, fixed in advance:** see the table in [docs/novice_ablation.md](../novice_ablation.md). If the hard
filters or BC cause the collapse, the thesis says so, and step 5 is worthwhile. If the oracle's replay priority
does, the first machine's conclusion stands, and the quality line closes.

## 6. Artifacts

| what | where |
|---|---|
| reports, plots, per-cell tables per campaign | `presentation_bundle/results/{Q1_validity_new,headroom_diagnostic_new,two_arm_pilot_new,two_arm_confirm_new,S_smoke}/` |
| operator test, tolerance sweep, limits, machine record | `presentation_bundle/results/second_machine_stage0/` |
| full export (102 GB, W&B files, checkpoints) | `Downloads/AQI_HIL_SERL_complete_2026-10-29/` |
| configs | `examples/campaigns/inconsistent_operator/`, `examples/campaigns/synthetic_suite/` |
| first-machine instructions and patches | `handoff/first_machine/` |
