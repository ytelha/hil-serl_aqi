# The second training machine — synthesis, and what changes on the first machine

**Master analysis of everything that ran on the second training machine** (24–29 October 2026),
what it means, and the plan it leaves for the first machine.

Detailed companions:
- [new_stack_operator_results_and_qa_log.md](new_stack_operator_results_and_qa_log.md): the operator
  alone, the orientation tolerance, training speed on the new stack, the engineering record;
- [inconsistent_operator_results_and_qa_log.md](inconsistent_operator_results_and_qa_log.md): score
  validity, discard against re-weight, the perfect-filter check, every pair.

Numbers were recomputed from the raw logs of the export `AQI_HIL_SERL_complete_2026-10-29` and agree
with the machine's own reports. Checked again on 2026-11-03 against the documentation archive; section 7 has the result.

---

## 0. Executive summary

- **What worked:** the code of patches 0014–0019 on the real stack (136 tests, 58 cells, no crash), and
  above all **rotation control**: cuboid rescue success from 7–13 % to 70–100 %.
- **What looks large but is not attributed:** on the new stack the first success without help comes about 4 × sooner for the
  same method (about 18 000–20 000 → 4 600 steps). Five changes and the machine differ.
- **What did not get much shorter:** a whole cell. Training takes about 30 minutes, but the ruler about 55 more.
- **What did not show a benefit:** using a quality score. Discard vs re-weight: −2.9 points, p 0.39.
  The perfect-filter check gives no usable direction: its 3 pairs flip between the ruler and the last checkpoint.
- **What stays open:** the hypothesis that quality pays off with inconsistent operators. The test was weak
  (late filter, a bug that gave each seed one condition sequence across all campaigns, ceiling, no control at scale).
- **What it changes:** the operator of every finished cuboid campaign could rarely rescue a twisted peg,
  which is a threat to validity for those campaigns. The first machine's next job is to find out whether
  the conclusions survive with an operator who can.

## 1. What ran

| campaign | what it asks | arms | cells | cap | wall time |
|---|---|---|---|---|---|
| stage 0 (`synthetic_suite`, `inconsistent_operator`) | operator alone; tolerance; tests | – | no training | – | CPU |
| `S_smoke` | does the full stack run end to end? | plain HIL-SERL, cylinder, expert | 1 | 8 k | 0.41 h |
| `Q1_validity_new` | do the scores find the bad corrections? | ensemble extended | 6 (1 per tier × 2 seeds) | 30 k | 10.09 h |
| `headroom_diagnostic_new` | can removing bad corrections help at all? | keep-all, oracle drop, random drop | 9 (1 per tier × 1 seed) | 30 k | 12.4 h |
| `two_arm_pilot_new` | discard or re-weight? | `drop-hind`, `reweight-hind` | 6 (1 per tier × 1 seed) | 30 k | 8.42 h |
| `two_arm_confirm_new` | the same at thesis scale | `drop-hind`, `reweight-hind` | 36 (3 per tier × 2 seeds) | 40 k | 53.79 h |

In all 58 cells and 85.1 hours: 57 on the cuboid and 1 smoke cell on the cylinder. All on the **new stack**: rotation control, orientation box, orientation channel,
mini-eval convergence, fixed update ratio. The inconsistent operator is on in the last four.

**Not run:** the ladder (`L0`–`L5`, `X_*`) that would attribute the stack's effect; the six-arm Q2 and Q3
at scale; anything on the original platform (it failed the stage-0 rule); plain HIL-SERL on the cuboid.

## 2. How the run went: gates and decisions

The plan changed three times on the way. This matters for how the results may be read.

| when (machine time) | event | decision | who |
|---|---|---|---|
| 24 Oct 22:35 | environment built, 136 tests pass, both suite plans valid | go | automation |
| 24 Oct 23:21 | Gate 1 **no-go**: a repeated test fails (a test bug), and independently 4 of 12 operator comparisons miss 20 points, all without rotation control | stop; nothing trained | automation |
| 25 Oct 00:24 | scope reduced to the new platform; the threshold, model and code stay; test bundle passes in a clean copy of the same commit | continue, new only | delegated by the user |
| 25 Oct 00:51 | smoke passes all checks | Gate 1 go | automation |
| 25 Oct 09:57 | Gate 2 **no-go**: Q1's best online score is weak (0.61) | stop; no two-arm campaign | automation |
| 25 Oct 19:33 | report: the hard stop was the automation's own conservative rule; proposes a control check first | recommendation | machine's Claude |
| 25 Oct 19:43 | perfect-filter check starts (not in the handoff) | run | user |
| 26 Oct 08:09 | check finishes: "no convincing primary headroom" by its fixed rule | report | automation |
| 26 Oct 12:02 | "Ignore the thresholds from now on that would stop the training at other gates and just do everything until the end." | gates report-only; behavioural score set; pilot and confirmation run | user |
| 29 Oct 02:14 | confirmation finishes | report only | automation |
| 29 Oct 09:59 | archive verified (76 988 entries, 102.2 GB); the originals are then deleted | – | machine |

**How to read this.** The two-arm experiment is a study of a score already known to be weak. It was not cleared by a gate. Nothing was hidden: the failed test, the failed comparisons and both no-go reports are preserved, thresholds were never lowered, and no source file changed. Records: `presentation_bundle/results/second_machine_reports/`.

## 3. What came out

| question | result | strength |
|---|---|---|
| Does the code work on the real stack? | yes: 136 tests, all 58 cells converged, no crash, no deadlock | strong |
| Could the original operator rescue a twisted cuboid? | no: 13 / 10 / 7 %; with rotation control 100 / 100 / 70 % | strong; no learning involved |
| Why? | 21° twist at takeover against 8° yaw tolerance | strong |
| Does the new stack remove the slow start? | same method: first unaided ≈ 18 000–20 000 → 4 600; timeouts ≤ 10 k 92–93 → 27 % | large, but not attributed |
| Does it make a cell faster? | training about 30 min instead of about 78, partly GPU and stop rule; whole cell still about 1.5 h, because of the ruler | measured |
| Is success still skill-graded? | flat in the discard arm (92.6 / 93.2 / 93.8 %); novices need 1.5–1.8 × the steps | suggestive; ceiling |
| Does the orientation channel matter? | it fires in about 1 % of takeovers | suggests not; ladder decides |
| Do scores notice inattention? | behavioural 0.66 (0.71 / 0.67 / 0.52); the live filter score also 0.66; length alone 0.65; counterfactual 0.49, lower late in training | moderate |
| Discard or re-weight? | 93.2 vs 90.3 %, −2.9 points, CI −13.4 to +6.2, p 0.39 | no difference shown; wide CI |
| Does a perfect filter help? | ruler 98.8 (keep-all) vs 93.8 %; last checkpoint 96.3 vs 100 % | undecided: 3 pairs, ceiling, sequence bug |
| Does quality beat the baseline? | not tested | – |

## 4. What this means

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
4. **The model has a flaw for novices.** An inattentive takeover uses fixed novice anchors, and some novices are noisier than those anchors, so their "careless" state can be smoother than their careful one. Novices cannot test inconsistency until this is fixed.

## 5. What changes on the first machine

**Its state (source files of 2026-10-31):** patches 0001–0010 plus local code (quantile trigger,
random asking, `oracle` score, milder novices, HiRIL fix). It lacks 0011–0019. Nothing is running. E2HiL and HiRIL
have not run. Instructions: `handoff/first_machine/HANDOFF_first_machine.md`.

**Code to move (all off by default; all 33 of its configs keep identical commands):**

| patches | what | why |
|---|---|---|
| 0022 (first) | novice test: config, cell copy, report | must run on the current code |
| 0011–0013 | W&B tags, report fixes | correct metadata and report text |
| 0014 | rotation control | the strongest result |
| 0015 | takeover logging, mini-eval convergence, fixed update ratio | worked in all 58 cells; machine-independent learning |
| 0016 | orientation box and channel, tolerance sweep | box matches the real robots; the ladder decides on the channel |
| 0017 | test suite (ladder, confirmation) | the campaigns below |
| 0018, 0019 | inconsistent operator, drop filters, two-arm configs | the follow-up on quality |
| 0021 | fixes: per-operator condition sequence, report, audit, tests, asking arm in the confirmation | found by the second machine |

Patches 0011–0019 are rebased onto its exact source files, so they apply without conflicts. Patch 0020 is
for the other direction: it brought its local code into the development repo. Patches 0024–0026 (HiRIL on the new stack, OR arm and held requests, smooth hand-over) exist in the development repo but are not yet in the first machine's bundle.

**What to take into account in the new campaigns:**

| lesson from the second machine | consequence |
|---|---|
| updates per env step differ between machines | never mix machines; every comparison needs its reference on the first machine |
| fixed update ratio changes the pace | the env waits for the learner; cost per cell must be measured on the smoke test |
| ruler near the ceiling; replicates differ (100 and 67 % for one cell) | primary measures: steps to converge, first unaided success, operator share; do not read 3 pairs |
| stopping at mini-eval 0.6 lands mid-swing | the report now shows the last-checkpoint SR; the threshold stays 0.6 until the ladder gives evidence |
| the ruler takes about two thirds of a cell's time | plan for about 55 min per cell on top of training; decide before a campaign whether to score fewer checkpoints |
| quality warm-up of 5 000 steps outlasts early learning | shorten it for any new filter test |
| one condition sequence per seed, in every campaign (bug) | fixed in 0021; do not merge label analyses with the second machine's |
| `int00` and `nov00` are different operators in 1-per-tier and 3-per-tier rosters | the ladder's pilots and the confirmation cannot be paired by operator name; only `exp00` matches |
| the inattentive novice can be smoother than the attentive one | judge inconsistency on expert and intermediate; change the model first if novices must take part (for example, never less noisy than the operator's own values) |
| orientation channel fires in about 1 % of takeovers; tilt limit is the top of the sweep | ladder includes `X_nochannel`; re-measure limits in stage 0 |
| original platform cannot show the inconsistent operator | use the new stack for it (the driver enforces this) |
| clocks change at the end of October | take durations from logs with offsets, not from file times |

## 6. The campaigns on the first machine

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

## 7. Artifacts, and how complete this is

| what | where |
|---|---|
| reports, plots, per-cell tables per campaign | `presentation_bundle/results/{Q1_validity_new,headroom_diagnostic_new,two_arm_pilot_new,two_arm_confirm_new,S_smoke}/` |
| operator test, tolerance sweep, limits, machine record | `presentation_bundle/results/second_machine_stage0/` |
| the machine's decisions: gates, no-go reports, status report, launch records, execution policy, replay, assumption checks, versions | `presentation_bundle/results/second_machine_reports/` |
| full export (102 GB, W&B files, checkpoints). **The only copy:** the originals on the machine were deleted after it was verified | `Downloads/AQI_HIL_SERL_complete_2026-10-29/` |
| documentation archive (1.3 GB unpacked: documents, reports and plots, no checkpoints) | `Downloads/AQI_HIL_SERL_documentation_2026-10-29.zip` |
| configs | `examples/campaigns/inconsistent_operator/`, `examples/campaigns/synthetic_suite/` |
| first-machine instructions and patches | `handoff/first_machine/` |

**The documentation archive against the full export (compared file by file, 2026-11-03).** 8 517 entries. 4 494 files are also in the export and all are byte-identical. The other 4 023 are documentation unpacked from archives embedded in the export; only 40 of those have content found nowhere in the export tree or the development repo, and they are package licences and vendored documents, older copies of repo documents, one earlier no-go note and two older test outputs. So it holds no new results. Its value is having the decision reports together, and they added the material of section 2 and the corrections listed in the two logs.

**What this check did not do.** It did not re-run training or inspect the W&B cloud. The first machine's own numbers (its oracle campaign) still come from its log. No results from the first machine have come back since.
