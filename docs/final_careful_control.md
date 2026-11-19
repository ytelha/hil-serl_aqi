# Final study: careful-operator control

**Question.** What does the operator's carelessness cost, and how much of that could grading recover at most?

The final study ([final_study.md](final_study.md)) ran every arm with a sometimes-careless operator. It compares
grading with keeping everything, both with that operator. It has no arm in which the same operators are always
careful, so it cannot measure the cost of the carelessness itself. That cost is the most any grading of corrections
can win back. This control adds the missing arm.

## Design

| | |
|---|---|
| campaigns | `F3_careful_hard` (3 demonstrations) and `F4_careful_standard` (6), copies of `F1_hard` and `F2_standard` |
| arm | `keepall` only: the ensemble keeps every correction, the operator decides when to help |
| the one difference | the operator is never careless (`synthetic_inconsistent: false`) |
| the same as the final study | operators (2 experts, 2 intermediates, 2 novices at skill 0.3), seeds 0–1, demonstrations, ruler, budget, checkpoints, platform, code |
| size | 2 campaigns × 6 operators × 2 seeds = 24 runs, about 10–14 h on the second machine (2 side by side) |
| machine | the second training machine only (the one that ran the final study); never mixed with another machine |

The inputs are copied from the final study's folders and checked by hash before the first run: the operators
(`operators/*.json`), the demonstrations (`demos/*/demos.pkl`), the ruler (`eval_grid.json`) and the stage-0
orientation limits. `tests/test_final_careful.py` checks that every training and evaluation command equals the final
study's `keepall` command without the three careless-operator flags.

## Analysis, fixed before the run

**Pairs.** Same campaign, operator and seed: `F3_careful_hard/keepall` with `F1_hard/keepall`, and
`F4_careful_standard/keepall` with `F2_standard/keepall`; 12 pairs per campaign, 24 pooled.

**Contrasts.**

1. careful keep-everything − careless keep-everything: the cost of the carelessness;
2. careful keep-everything − perfect filter (careless operator, every careless correction dropped): what is left of
   that cost after the best possible grading.

**Measures.**

- Primary: environment steps until the run stopped, and operator steps.
- Secondary: evaluation success (best window of three checkpoints), last checkpoint, success after 4 000, 6 000 and
  8 000 environment steps, first success without help, operator share.

**Tests.** Two-sided sign test and exact Wilcoxon signed-rank test, per campaign and pooled; no correction for
multiple comparisons. Per-run records with `results/analysis/scripts/run_tables.py`.

**Reading.**

| result | reading |
|---|---|
| careful ≈ careless on every primary measure | the carelessness cost nothing on this task: grading had nothing to recover |
| careful better on a primary measure (sign p < 0.05, same direction in both campaigns), perfect filter ≈ careless | the carelessness costs time and operator effort, but the cost arises while a careless correction happens; removing its data afterwards cannot undo it |
| careful better, and the perfect filter closes part of the gap | dropping careless corrections recovers part of the cost |

**Caveats.** The control runs after the final study, not interleaved with it (same machine, code and inputs; the
fixed update ratio makes learning per environment step independent of wall-clock speed). The careless operator takes
one extra random draw when it starts, so the two arms' operator noise differs from the first step; pairs share
operator, seed and demonstrations, as every pair of the final study does.

## How to run (second training machine)

```bash
python tests/test_final_careful.py               # config checks, no GPU
bash tools/run_final_careful_unattended.sh       # copies and checks the inputs, trains, retries crashes once, reports
tail -f runs/final_careful/unattended.out        # progress
```

The script reads `runs/final/` and never writes there. It stops before training if an input differs from the final
study's, if the training code has local changes, or if the disk has less room than the 24 runs need (it never keeps
only the newest checkpoints, which would lose the budget curves). Results go to `results/final_careful/` only, with
`CAREFUL_RUN.md` as the run record; nothing is pushed.
