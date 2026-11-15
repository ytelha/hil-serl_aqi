# The comprehensive study — results (second training machine, 5–9 Nov 2026)

*Analysed 2026-11-09 from the export `AQI-HIL-SERL-study-results-2026-11-09_065159.zip`. A slim copy is in
`presentation_bundle/results/study/`: reports, run table, audits, plots, ruler scores per run, and the analysis
scripts in `analysis/`. The full export with the takeover logs stays in the zip.*

## 1. What ran

The unattended run finished all steps with no warnings:
- check, stage 0, smoke, Study A, Study B, audit, report;
- 168 study runs plus 8 smoke runs;
- 2 runs side by side on the RTX 5090.

**Setup.**
- **Stack:** the new stack (rotation control, orientation box and twist channel, mini-eval stop rule, exactly one
  learner update per environment step), the shorter ruler (newest 8 checkpoints), budget 30 000 steps, cuboid peg.
- **Local change:** the machine had one, a shell-path fix in `sim_env.sh`
  (`patches/0001-Fix-workstation-setup-with-unset-shell-library-paths.patch`). It doesn't touch training.

| study | operators | arms | runs |
|---|---|---|---|
| **A: critics and asking** | consistent; 3 experts + 3 intermediates × 2 seeds | HIL-SERL; ensemble extended with K = 4 / 8 / 16 critics (operator decides); robot asks in the 40 / 55 / 70 % most uncertain steps; random requests at the 55 % rate | 8 × 12 = 96 |
| **B: grading** | a sometimes-careless expert, 20 % careless; 3 experts × 3 seeds | HIL-SERL, E2HiL, HiRIL; ensemble extended keeping everything (keep-all); behavioural score as discard (lowest 20 %) and as re-weight; quality oracle as re-weight; perfect filter (drops exactly the careless takeovers) | 8 × 9 = 72 |

In B, no arm used the outcome filter or an imitation loss.

## 2. The bottom line

1. **Everything learns the task, fast.**
   - Every arm except HiRIL scores 89–99 % on the ruler, and 97–100 % at the last checkpoint.
   - Runs converge after about 5 000–8 500 steps.
   - **The ruler can't separate the methods any more.** Differences only show in *speed* and *human effort*.
2. **Critics: the number doesn't matter.** K = 4, 8 and 16 and HIL-SERL all end at 94–99 %. The ensemble arms, which
   pre-train on the demos, learn faster early: at 4 000 steps 46–57 % against HIL-SERL's 13 %.
3. **Asking: the first machine's gain does not replicate on the new stack.**
   - Asking reaches the same success as the operator deciding.
   - It costs more help: 45–56 % of the steps against 38 %.
   - Random requests at the same rate do just as well.
4. **Grading: detecting careless takeovers now passes the bar; using the score in training changes nothing visible.**
   - The behavioural score reaches 0.76 within a run, which is "valid".
   - Hindsight reaches 0.74, and the value gained alone 0.73, so the signal is no longer just length.
   - All grading arms equal keep-all at the ceiling.
5. **The baselines.**
   - **E2HiL:** same success, slower.
   - **HiRIL:** worse, but **not a fair result**. Its requests last 2.7 steps on average, too short for the operator
     to answer, so it got 8 % help (§5.3).

## 3. Study A: critics and asking (12 paired runs per arm)

| arm | ruler | last ckpt | first unaided success (median steps) | converged at (median) | operator share | operator steps |
|---|---|---|---|---|---|---|
| HIL-SERL | 93.5 % | 100 % | 5 459 | 7 640 | 39 % | 3 018 |
| ensemble K = 4 | 96.9 % | 98 % | 4 016 | 7 688 | 38 % | 2 806 |
| ensemble K = 8 | 98.5 % | 100 % | 4 750 | 7 440 | 38 % | 3 097 |
| ensemble K = 16 | 95.1 % | 99 % | 4 246 | 7 866 | 37 % | 2 621 |
| asks 40 % | 96.9 % | 98 % | 4 457 | 6 834 | 45 % | 3 372 |
| asks 55 % | 92.6 % | 99 % | 3 744 | 5 828 | 52 % | 3 602 |
| asks 70 % | 98.1 % | 100 % | 4 532 | 6 558 | 56 % | 3 926 |
| random 55 % | 98.1 % | 97 % | 4 623 | 7 316 | 47 % | 3 375 |

### Paired comparisons (sign tests, 12 pairs)

**Success.**
- No arm differs from HIL-SERL or from K = 8 (all p ≥ 0.06).
- Closest: asking 55 % below K = 8, 0 wins, 5 losses, 7 ties, p = 0.06.

**Asking against the operator deciding (K = 8).**
- Asking at 55 % converges sooner: median −1 450 steps, 10 of 12 pairs, p = 0.039.
- But with more operator share in all 12 pairs.
- Total operator steps are not lower: median +450, 4 / 8.
- So asking moves work from the robot to the person; it doesn't save the person's time.

**Critics.**
- K = 16 used fewer operator steps than K = 8: 10 of 12 pairs, p = 0.039.
- That is one of many tests; treat it as a hint.

**Dependency.**
- Every ICC is 0.00–0.07, except K = 16 at 0.28, which comes from one intermediate's 70 %.
- Experts and intermediates on the new stack leave no teacher effect to reduce.

### Learning speed on the ruler

From the per-checkpoint scores already in the export. The value after a run's stop is carried forward; a learner
step minus 5 000 pre-training steps gives the environment step.

| arm | 4 000 steps | 6 000 | 8 000 | 10 000 |
|---|---|---|---|---|
| HIL-SERL | 13 % | 67 % | 94 % | 100 % |
| ensemble K = 4 / 8 / 16 | 57 / 47 / 46 % | 82 / 68 / 86 % | 97 / 86 / 98 % | 98 / 98 / 99 % |
| asks 40 / 55 / 70 % | 43 / 59 / 41 % | 66 / 79 / 82 % | 98 / 88 / 100 % | 98 / 99 / 100 % |
| random 55 % | 56 % | 75 % | 92 % | 97 % |

Only the early gap between HIL-SERL and the pre-trained ensembles is clear: K = 4 against HIL-SERL at 4 000 steps,
8 wins, 1 loss, p = 0.039. That gap is the warm start on the demonstrations, not the number of critics.

## 4. Study B: grading with a 20 % careless expert (9 paired runs per arm)

| arm | ruler | last ckpt | first unaided success (median) | converged at (median) | operator share | operator steps |
|---|---|---|---|---|---|---|
| HIL-SERL | 88.5 % | 99 % | 3 729 | 7 793 | 46 % | 3 417 |
| E2HiL (8 runs) | 98.6 % | 100 % | 6 247 | 8 472 | 45 % | 3 666 |
| HiRIL | 77.0 % | 67 % | 19 265 | 22 976 (5 of 9) | 8 % | 2 073 |
| keep-all | 98.4 % | 97 % | 3 569 | 5 699 | 43 % | 2 729 |
| behavioural, discard 20 % | 99.2 % | 100 % | 2 788 | 5 122 | 44 % | 2 637 |
| behavioural, re-weight | 99.2 % | 99 % | 3 293 | 7 065 | 39 % | 2 640 |
| oracle, re-weight | 97.5 % | 99 % | 2 546 | 5 115 | 41 % | 2 569 |
| perfect filter | 95.5 % | 99 % | 2 992 | 5 357 | 41 % | 2 370 |

### Paired comparisons (sign tests, 9 pairs)

**Against keep-all.**
- **Success:** no grading arm differs (all p ≥ 0.62).
- **Speed:** small hints in favour of filtering, none significant on its own.
  - The perfect filter used fewer operator steps in 8 of 9 pairs (p = 0.039).
  - On the ruler at 6 000 steps: discard 100 % and perfect filter 99 %, against keep-all 90 % (2 / 0 / 7 and 1 / 0 / 8).

**Against HIL-SERL.**
- The arms built on the ensemble are faster early. Discard against HIL-SERL on the ruler at 4 000 and 6 000 steps:
  6 wins, 0 losses, p = 0.031.
- Three of the nine HIL-SERL runs scored 59–78 % on the ruler, though 89–100 % at their last checkpoint; no
  ensemble-based run did. With 9 pairs that is a hint, not a result.

**E2HiL.**
- Same success as keep-all, but slower: converged later in 8 of 9 pairs (p = 0.039), and 33 against 21 minutes of
  robot time per run.

**HiRIL.** 19 points below keep-all, 0 wins, 8 losses, p = 0.008. See §5.3 before using this.

### The score audit

4 583 takeovers with a label, from the 45 runs of the ensemble-based arms. Careless share 19 %. Careless takeovers
last 33.5 against 23.5 steps; their episodes succeed in 60 against 90 %.

| score | within a run | overall (95 % CI) | verdict |
|---|---|---|---|
| behavioural | **0.76** | 0.78 (0.76–0.79) | valid |
| hindsight advantage | **0.74** | 0.77 (0.74–0.78) | valid |
| value gained alone | 0.73 | 0.70 | valid |
| behavioural without length | 0.72 | 0.68 | weak |
| behavioural as computed live in training | 0.70 | 0.73 | valid |
| takeover length | 0.69 | 0.76 | valid |
| episode outcome | 0.55 | 0.65 | weak |
| peak force | 0.51 | 0.54 | weak |
| progress to the hole | 0.47 | 0.49 | invalid |
| critics' comparison | 0.47 | 0.42 | invalid |
| smoothness (circular) | 0.83 | 0.82 | not usable |

**The discard filter:**
- 41 % of what it dropped was careless, against a base rate of 19 %, a lift of 2.2;
- it dropped 18 % of the takeovers after its 1 000-step warm-up.

**The audit's automatic warning** ("an onset yardstick is far from 0.5") comes from the *pooled* start time, 0.69:
careful takeovers start earlier in the episode on average. Within a run, phase and onset time, the start time is
0.52 and the critics' value at the onset 0.47. So the within-run numbers above are clean. Quote within-run numbers
only.

Compared with the earlier series (40 % careless operator, experts and intermediates, 0.70 within), the scores
separate better here. Value gained and hindsight now carry signal of their own, not just length.

## 5. Problems found in the data (fix before the next campaign)

### 5.1 The task is at the ceiling

- 110 of 168 runs score 100 % on the ruler.
- With the new operator (rotation control) and the mini-eval stop rule, experts and intermediates teach the cuboid
  in about 7 000 steps.
- **Final success can no longer show a difference between methods.**

### 5.2 The ruler meets the new stop rule badly

The ruler takes the best window of 3 consecutive checkpoints among the newest 8. The mini-eval rule stops a run 2
passes after it first looks good, so a run's last checkpoints often straddle the moment it learned.

Example, HIL-SERL, expert 1, seed 2:

| checkpoint (steps) | 3 000 | 4 000 | 5 000 | 5 050 |
|---|---|---|---|---|
| ruler score | 67 % | 0 % | 78 % | 100 % |

The best window gives **59 %**, though the policy at the stop scored 100 %.

11 runs scored below 80 % on the ruler. The 7 outside HiRIL all scored 89–100 % at their last checkpoint: their
policy was still unstable when training stopped. The 4 HiRIL runs are different; they never stabilised.

### 5.3 HiRIL got almost no help — not a fair comparison

| measure | value |
|---|---|
| requests | 12 611 in 9 runs, about 9 per episode |
| request length | 2.7 steps on average |
| requests answered | 7.5 % |
| operator share | 8 % (others 40–46 %) |
| episodes that timed out in the first 10 000 steps | 98 % |

HiRIL requests help whenever its risk estimate exceeds a threshold, step by step. The synthetic expert needs about 6
steps of a request before it takes over. This is the same mechanism that made the novice miss the 10-step requests
on the first machine. **The 19-point loss is mostly "no help", not HiRIL's idea.**

Check: `analysis/hiril_requests.py` (needs the full export).

### 5.4 One E2HiL run crashed and was counted as finished

`e2hil_exp02_s2` stopped at step 193 with a simulator error: "Found zero norm quaternions" in
`panda_peg_insert_gym_env.py` line 287, during a takeover. The learner then stopped gracefully, and the driver marked
the run *finished* even though it has no ruler result. E2HiL therefore has 8 runs.

## 6. What this means for the thesis and the deck

*In the Siemens deck since 14 Nov:* result slide page 27 (success against help for critics and asking; AUROC against the earlier test for grading), backup page 50 (§3–4 tables, paired tests, problems), summary page 28 rewritten along the table below.

| claim (deck, first machine) | status after this study |
|---|---|
| The teacher matters | Shown on the old stack, with novices and policy-oracle demos. On the new stack, experts and intermediates leave no teacher effect (ICC ≈ 0 for every arm). Keep the claim tied to its setting. |
| Many critics can learn | **Confirmed and extended:** K = 4 / 8 / 16 all match HIL-SERL; the warm-started ensembles learn faster early. |
| Asking works (94 vs 75 / 81 %) | **Not replicated on the new stack.** At the ceiling, asking gives the same success with more operator share; random timing is as good. The first-machine result stands as a within-campaign result on the old stack, where the operator often couldn't help (rescue 10–13 %) and the operator-decides share was 57 %. Here it is 38 %, and the tested rates (40–70 %) are all above it. |
| Grading is promising | **Detection: stronger** (0.76 within, valid). **Training benefit: not shown,** because there is nothing left to improve at the ceiling. |
| Comparison with E2HiL / HiRIL | E2HiL: same success, slower. HiRIL: not interpretable until its requests can be answered. |

## 7. Suggestions — how to go further

In order of priority.

1. **Change the endpoint before running anything new.** On the new stack, compare methods by *speed* and *human
   effort*, not final success.
   - **Primary:** ruler success at fixed budgets (e.g., 2 000 / 4 000 / 6 000 / 8 000 steps), measured on
     checkpoints saved at those steps.
   - **Secondary:** operator steps until a *stable* policy.
   - For that, either train for a fixed budget without the early stop, or keep training a fixed number of steps (e.g.,
     3 000) after the mini-eval stop before scoring.
   - Score the final policy with 3 tries per pose (27 trials) instead of a 3-checkpoint window.
   - Fix this in the campaign config and the ruler settings, and pre-register it.
2. **Make the task harder, to get headroom back.** Pilot a small ladder (about 10–20 runs, HIL-SERL and the
   fixed ensemble) and pick the setting where both reach roughly 50–70 % at a fixed budget. Candidates:
   - the tight hex peg (needs yaw alignment);
   - a smaller clearance;
   - a wider start range (±25 mm, ±0.2 rad);
   - fewer demonstrations (2–3);
   - no warm start for any arm.
3. **Asking: test the efficiency question properly.**
   - Rates *below* the operator-decides share (10 / 20 / 30 %), so asking could save help.
   - The OR arm (the operator can still step in), with random controls at matched rates.
   - On the harder task, with the budget endpoint.
4. **Grading: make careless corrections costly.** Use 40 % careless, or a careless intermediate, on the harder task,
   so keep-all drops and the filters have something to recover. Keep keep-all, behavioural discard and the perfect
   filter. The detection result (§4) can be written up now.
5. **HiRIL: re-run its 9 runs with held requests**, e.g. `--synthetic_request_hold_steps 10`, or a minimum request
   length like the AQI gate's 10 steps. Report the current run as "requests could not be answered".
6. **Fix the two bugs:**
   - guard the simulator against a zero-norm target orientation (renormalise or skip the rotation update);
   - make the driver mark a run *failed* when the actor exits with an error or no ruler result exists.
   - Then re-run `e2hil_exp02_s2`.
7. **Update 10 Nov: the ablation ran anyway, and confirmed this.** The first version doesn't fail on the new stack;
   see [ensemble_fix_ablation_results.md](ensemble_fix_ablation_results.md). The original advice was: **hold the
   ensemble-fix ablation (first machine, 96 runs) until it has headroom.**
   - On the new stack HIL-SERL and the fixed ensemble both reach the ceiling. Its question, "which part of the fix
     matters", needs the first version to fail.
   - Run a 4-run pilot first: first version against fixed version, one expert and one intermediate. Or run the
     ablation on the harder task with the budget endpoint.

## 8. How to reproduce the numbers

```bash
python -I presentation_bundle/results/study/analysis/study_analysis.py presentation_bundle/results/study
```

```bash
python -I presentation_bundle/results/study/analysis/study_budget.py presentation_bundle/results/study
```

The study's own report is `presentation_bundle/results/study/report.md`, the audit
`presentation_bundle/results/study/B_quality/report/inconsistent_audit.md`, and the plots are in
`presentation_bundle/results/study/plots/`.
