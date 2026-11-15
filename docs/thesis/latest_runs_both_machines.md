# The latest runs on both training machines (November 2026) — what they show and how to go further

*Written 2026-11-10.*

| machine | latest campaign | details |
|---|---|---|
| first machine | the ensemble-fix ablation (96 runs) | [ensemble_fix_ablation_results.md](ensemble_fix_ablation_results.md) |
| second machine | the comprehensive study (168 runs) | [comprehensive_study_results.md](comprehensive_study_results.md) |

All data is in `results/`; see [data_inventory.md](data_inventory.md).

## 1. What ran

| | first machine | second machine |
|---|---|---|
| campaign | ensemble-fix ablation | comprehensive study: A (critics, asking) and B (grading) |
| runs | 96: 8 arms × 6 operators × 2 seeds | 168: A 8 arms × 12; B 8 arms × 9 |
| operators | 3 experts + 3 intermediates, consistent | A: 3 experts + 3 intermediates, consistent; B: 3 experts, 20 % careless |
| stack | new: rotation control, twist channel, mini-eval stop, one update per step | the same |
| ruler | standard (all checkpoints) | shorter (newest 8 checkpoints) |
| what came back | the report (9 Nov), then the slim run folder (13 Nov) | the full export (reports, ruler per checkpoint, training records, takeover logs) |

## 2. What both show

1. **On the new stack, experts and intermediates make the task easy.**
   - Every method reaches about 90–99 % on the ruler within about 6 000–9 000 steps.
   - Final success no longer separates methods, and there is no teacher dependency left (ICC ≈ 0).
2. **The two machines now agree.**
   - HIL-SERL: 93.5 % on both, operator share 39 % on both, about 8 400 and 7 600 steps to converge.
   - The full ensemble fix: 91 and 98.5 %.
   - With one learner update per step the machines give the same level, which the old stack did not (finding F3).
     Still, run each comparison on one machine.
3. **Ensembles learn as well as HIL-SERL and faster.**
   - Any number of critics (4, 8, 10, 16) reaches the same success: 96–100 % at the last checkpoint on the first
     machine.
   - Ensembles converge sooner and, on the first machine, with slightly less help.
   - **The warm start on the demonstrations is what makes them faster:**
     - first unaided success earlier than the first version in all 12 pairs;
     - 78 against 23 % (HIL-SERL) on the ruler after 4 000 steps.
   - It slows plain HIL-SERL down (30 against 71 % at 6 000 steps).
4. **The first ensemble version no longer fails, and de-pessimisation isn't needed.**
   - Its old loss was mainly slower convergence, which is gone; even on the old stack it was not significant on
     success.
   - Of the four changes of the fix, only the warm start matters on the new stack. De-pessimisation alone slows the
     early phase, and the warm start alone is at least as fast as the full fix.
   - What removed the old problem is open: the operator, the update ratio, the stop rule or the budget.
   - On both machines the low ruler scores are mostly policies still unstable at the stop (11 of 13 on the first
     machine): the best-of-3 window meets the early stop badly.
5. **Asking does not replicate.** With 40–70 % requests the robot gets more help than when the operator decides (38 %),
   for the same success; random requests do as well.
6. **Grading.**
   - Detection works with the careless operator: the behavioural score reaches 0.76 within a run.
   - The critics' comparison fails again. It is inverted in the arms with the de-pessimised target or the warm start,
     at chance with the pessimistic target: a hint that the critics' optimism about the robot's own move causes it.
   - Using a score in training shows no gain, because there is nothing left to improve.

## 3. What this means for the thesis story

| question | old stack (first machine, Sept–Oct) | new stack (both machines, Nov) |
|---|---|---|
| Does the robot inherit its teacher? | yes, clearly, with novices and policy-oracle demonstrations | not with experts and intermediates: ICC ≈ 0 |
| Can many critics learn? | the fixed version matches HIL-SERL | every version matches HIL-SERL, and faster |
| Does asking help? | asking more often beat the operator's own judgement within one campaign | no gain at the ceiling; it costs more help |
| Can a score grade corrections? | the critics' comparison fails | behavioural score valid; no measurable training gain at the ceiling |

How to tell it: the new stack fixed the operator (it can now turn the peg) and the learning rhythm, and that made the
task easy. The open questions — whether asking saves human effort, and whether grading helps learning — must be
asked on a harder task with a measure that isn't at the ceiling.

## 4. How to go further — one plan for both machines

In order. Each step says where it runs and roughly how long it takes.

1. ~~Finish the ablation analysis.~~ **Done 14 Nov** with the slim run folder:
   - settings confirmed;
   - low runs explained;
   - learning speed at fixed budgets ([ensemble_fix_ablation_results.md](ensemble_fix_ablation_results.md)).

   One consequence for the next campaigns: use the warm-started ensemble with the pessimistic target (`ens-wu`) as the
   ensemble base, after a check on the second machine.
2. **Change the endpoint, then pre-register it (code, here).**
   - Score each run at fixed budgets of 2 000, 4 000, 6 000 and 8 000 steps on the ruler.
   - Keep training a fixed 3 000 steps after the mini-eval stop, or use fixed budgets without the stop.
   - Score the final policy with 3 tries per pose.
   - Human effort to a stable policy as the second measure.
   - This is a small change to the ruler selection and the campaign settings, with tests, as one patch.
3. **Find a harder task (a pilot, about 1 day on one machine, 10–20 runs of HIL-SERL and the fixed ensemble).**
   - Try in order: the hex peg, a tighter clearance, a wider start range (±25 mm, ±0.2 rad), 2–3 demonstrations, no
     warm start.
   - Pick the first setting where both reach about 50–70 % at 8 000 steps.
4. **On the harder task, with the new endpoint, one study per machine:**
   - **Second machine, asking (about 2 days):**
     - asking in 10 / 20 / 30 % of the steps, below the operator's own 38 %;
     - the OR arm (the operator may still step in), with random requests at the same rates;
     - HIL-SERL and the fixed ensemble as references.
   - **First machine, grading (about 2–3 days):**
     - a careless share of 40 %, or a careless intermediate, so that keeping everything actually loses;
     - keep-all, behavioural discard, behavioural re-weight, perfect filter and HIL-SERL.
5. **Repair the baselines, then compare (second machine, about 1 day).**
   - HiRIL with held requests (`--synthetic_request_hold_steps 10`), so the operator can answer.
   - Fix the simulator's zero-norm-quaternion crash and make the driver mark crashed runs as failed.
   - Then E2HiL and HiRIL against the best AQI arm on the harder task.
6. ~~Optional: a ladder that switches the stack changes on one at a time.~~ **It already ran** on the first machine
   (5–6 Nov, HIL-SERL only): rotation control removes most of the slow start, one update per step the rest
   ([first_machine_novice_test_and_ladder.md](first_machine_novice_test_and_ladder.md)). Still open: the same
   ladder for the first ensemble, if the thesis must explain why it was slow on the old stack.
7. **Real robot.** The UR5 lab port waits for the lab's repository. That stage tests what simulation can't: real
   operators, their timing and their carelessness.

**What to update now, without new runs:**
- deck page 20 ("lost … four changes fix it"): narrow it to the old stack, and say that the success gap was not
  significant there;
- ~~deck page 23 and the summary: add that asking did not replicate on the new stack~~ **done 14 Nov:** the Siemens
  deck has a new result slide on the comprehensive study (page 27), its numbers per arm (backup 50), a rewritten summary
  (page 28) and pointers in the notes of pages 21 and 23;
- the thesis results chapter: split it into the old stack (teacher dependency, asking rate) and the new stack
  (ceiling, scores, ablation).
