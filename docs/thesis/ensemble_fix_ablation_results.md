# Ensemble-fix ablation — results (first training machine, November 2026)

*Analysed 2026-11-10 from the campaign report, and **completed 2026-11-14 from the full run folder** (the user's slim
export, `ensemble_fix_ablation_slim_2026-11-13.tgz`).*

- Data: `results/first_machine/2026-11_ensemble_fix_ablation/` (campaign layout: `campaign.json`, `checkpoints/`,
  `eval_runs/`, `aqi_logs/`, `logs/`, `report/`); the originals are in `results/_received/`.
- Analysis: `results/analysis/scripts/ablation_full_analysis.py` (output `results/analysis/outputs/ablation_full_analysis.txt`)
  and `ablation_analysis.py` (from the report).
- Plan and arms: [`../ensemble_fix_ablation.md`](../ensemble_fix_ablation.md).

## 1. What ran

| | |
|---|---|
| campaign | 8 arms × 6 operators (3 experts, 3 intermediates) × 2 seeds = **96 runs**, all finished and converged |
| setting | cuboid peg, budget 30 000 steps, the new stack (rotation control, twist channel, mini-eval stop, one learner update per step), the standard ruler (9 poses, best window of 3 over the kept checkpoints) |

| arm | critics K | target M | warm start | learning rate | question |
|---|---|---|---|---|---|
| `ens` | 10 | min of 2 | – | 3e-4 | the first version |
| `ens-m1` | 10 | 1 | – | 3e-4 | de-pessimisation alone |
| `ens-k8` | 8 | min of 2 | – | 3e-4 | fewer critics alone |
| `ens-wu` | 10 | min of 2 | 5 000 | 3e-4 | warm start alone |
| `ens-wu-lr` | 10 | min of 2 | 5 000 | 1.5e-4 | warm start + halved learning rate |
| `ens-fix` | 8 | 1 | 5 000 | 1.5e-4 | the full fix |
| `hilserl` | 2 | – | – | 3e-4 | plain HIL-SERL |
| `hilserl-wu` | 2 | – | 5 000 | 3e-4 | HIL-SERL with warm start |

**Checked against the run folder:**
- The settings match `campaign.json` and every run's own learner command line.
- All runs ran at one update per step.
- None of the 300 logs contains an error.

## 2. Results

| arm | ruler (best window) | last checkpoint | stable (lowest of last 3) | first unaided success (median steps) | converged at (median) | operator share |
|---|---|---|---|---|---|---|
| first version `ens` | 95.4 % | 96.3 % | 88 % | 4 838 | 7 064 | 34.9 % |
| de-pessimisation `ens-m1` | 96.6 % | 100 % | 92 % | 4 698 | 8 038 | 36.3 % |
| fewer critics `ens-k8` | 96.0 % | 99.1 % | 89 % | 4 502 | 6 750 | 35.1 % |
| **warm start `ens-wu`** | 96.6 % | 97.2 % | 89 % | **3 411** | 6 289 | 38.2 % |
| warm start + LR `ens-wu-lr` | 91.4 % | 98.1 % | 75 % | 3 817 | 6 130 | 35.4 % |
| full fix `ens-fix` | 91.4 % | 99.1 % | 75 % | 4 414 | 6 421 | 37.5 % |
| HIL-SERL | 93.5 % | 100 % | 82 % | 5 272 | 8 612 | 39.1 % |
| HIL-SERL + warm start | 89.8 % | 96.3 % | 79 % | 6 268 | 9 538 | 38.4 % |

**Learning speed on the ruler.** Success of the policy alone at fixed training budgets (environment steps; for the
warm-started arms, checkpoint step minus the 5 000 warm-start steps). After a run stopped, its last score is carried
forward.

| arm | 2 000 | 4 000 | 6 000 | 8 000 | 10 000 |
|---|---|---|---|---|---|
| first version | 11 % | 42 % | 84 % | 96 % | 96 % |
| de-pessimisation | 6 % | 28 % | 69 % | 92 % | 99 % |
| fewer critics | 1 % | 49 % | 99 % | 99 % | 99 % |
| **warm start** | 13 % | **78 %** | **95 %** | 97 % | 97 % |
| warm start + LR | 7 % | 69 % | 90 % | 98 % | 98 % |
| full fix | 28 % | 50 % | 79 % | 90 % | 99 % |
| HIL-SERL | 5 % | 23 % | 71 % | 90 % | 93 % |
| HIL-SERL + warm start | 4 % | 18 % | 30 % | 77 % | 88 % |

### Paired comparisons (12 pairs, sign test)

**The final policy:** at the last checkpoint every arm scores 96–100 %, and no arm differs from any other (all p ≥ 0.5).
The same holds for the ruler's best window and for stability.

**Speed: the warm start is what helps an ensemble.**
- First unaided success: the warm-started ensemble got there before the first version in **all 12 pairs**
  (p < 0.001), and before HIL-SERL in 10 of 12 (p = 0.039).
- Ruler at 4 000 steps:
  - warm start ahead of HIL-SERL in 11 of 12 pairs (p = 0.006);
  - warm start + LR ahead of HIL-SERL in 10 of 11 (p = 0.012);
  - full fix ahead of HIL-SERL in 7 of 7 (p = 0.016).
- **De-pessimisation does not help.** Alone it learns more slowly than the warm start: behind at 4 000 steps in 10 of
  11 pairs, p = 0.012. Added to the warm start (in the full fix) it does not make learning faster. Against the warm
  start alone, the full fix is behind more often than ahead at 4 000 and 6 000 steps (7 / 3 and 5 / 1 of the pairs
  that differ), but not significantly.
- Fewer critics (8 instead of 10) and the halved learning rate show no clear effect.
- Steps to converge, from the report: every warm-started ensemble converges sooner than HIL-SERL (10–11 of 12 pairs,
  p ≤ 0.039).

**The warm start slows plain HIL-SERL down.**
- Its first unaided success came later than without it in 10 of 12 pairs (p = 0.039).
- On the ruler at 6 000 steps it reaches 30 against 71 %, and it is behind the first ensemble version in 10 of 11
  pairs (p = 0.012).
- Why is open. A guess: two critics pre-trained only on successful demonstrations start overconfident and must first
  unlearn that online. The learner logs (critic loss after the warm start) can check this.

**Human effort:** the first version and the 8-critic version need less help than HIL-SERL (−5 and −3 points, both
10 / 2, p = 0.039).

**Teacher dependency:** none (ICC 0 for most arms; 0.65 for warm start + LR comes from one intermediate's two runs).

### The low runs

13 runs scored below 80 % on the ruler. With the scores per checkpoint now available:
- **11 were still unstable when training stopped.** Their last checkpoint scored 88–100 %, but the 3-checkpoint window
  also caught earlier, bad checkpoints. Examples (score at environment step):
  - HIL-SERL, expert 2, seed 0: 55 % at 2 000, 0 % at 3 000, 88 % at 4 000, 0 % at 5 000, 100 % at 6 000;
  - full fix, intermediate 2, seed 1: 88 % at 7 000, 0 % at 8 000, 100 % at 9 000.
- **2 were genuinely weak at the stop:**
  - the first version, intermediate 0, seed 1, ended at 78 %;
  - HIL-SERL + warm start, intermediate 1, seed 0, ended at 67 %.

So the ruler's best-of-3 window mostly measures whether a run stopped right after it learned. That is the same finding
as on the second machine.

### The score audit

The label here is episode success: the operator is consistent, so there is no careless label.

| arm | target | critics' comparison | with timing controls | behavioural score | time-only yardstick |
|---|---|---|---|---|---|
| first version | min of 2 | 0.56 | 0.53–0.56 | 0.67 | 0.72 |
| fewer critics | min of 2 | 0.53 | 0.49–0.55 | 0.69 | 0.69 |
| de-pessimisation | 1 | 0.37 | 0.34–0.40 | 0.65 | 0.73 |
| warm start | min of 2 | 0.35 | 0.35–0.39 | 0.71 | 0.76 |
| warm start + LR | min of 2 | 0.40 | 0.40–0.43 | 0.67 | 0.70 |
| full fix | 1 | 0.34 | 0.31–0.35 | 0.63 | 0.73 |

- The **critics' comparison** sits at chance in the two arms without a warm start and with the pessimistic target. It
  points the wrong way (0.31–0.43) in the de-pessimised arms and the warm-started arms, under every timing and
  difficulty control.
- That fits the explanation on page 24 of the deck: when the critics are optimistic about the robot's own actions
  (single-critic target, or trained early on the demonstrations), the robot's move is over-rated and the correction
  loses. **A hint, not a proof:** 12 runs per arm, label = episode success.
- The **behavioural score** against episode success is again no better than a clock that only knows when the takeover
  happened (findings F4/F7). The clean test is the careless operator on the second machine, where it reaches 0.76.

## 3. What it means

1. **The first version no longer fails, so the ablation can't say which change "fixed" the old problem.**
   - On the new stack the first ensemble (10 critics, pessimistic target, no warm start) reaches 96 % at its last
     checkpoint and learns faster than HIL-SERL.
   - On the old stack its loss was small: for experts and intermediates, 79 against 92 %, but 3 wins and 3 losses,
     p = 1.0. Its clearest symptom was slower convergence (later in 7 of 8 pairs, p = 0.07), and that has reversed.
2. **Ruled out: less learning per step.** On the old stack the 10-critic ensemble and HIL-SERL both got 0.45 learner
   updates per step (W&B export).
3. **What changed between the stacks** is several things at once:
   - the operator turns the peg;
   - one update per step;
   - the mini-eval stop;
   - a 30 000-step budget.

   Which of these removed the problem is open; a ladder adding one change at a time would answer it.
4. **What the ablation shows on the new stack:**
   - **For the final policy, none of the four changes is needed:** every arm ends at 96–100 %.
   - **For speed, the warm start is the one change that matters.** The warm start alone is at least as fast as the
     full fix.
   - **De-pessimisation, once thought to be the main fix, does not help;** alone it slows the early phase.
   - The fastest ensemble here is the **pessimistic target with the warm start** (`ens-wu`): 78 % already at 4 000
     steps, and the only arm with no run below 89 % on the ruler.
   - The warm start helps ensembles but slows plain HIL-SERL.
   - Ensembles need slightly less help than HIL-SERL.
5. **The two machines agree on the new stack.**

   | | first machine | second machine (comprehensive study) |
   |---|---|---|
   | HIL-SERL, ruler success | 93.5 % | 93.5 % |
   | HIL-SERL, steps to converge | about 8 600 | about 7 600 |
   | HIL-SERL, operator share | 39 % | 39 % |
   | full fix, ruler success | 91.4 % | 98.5 % |
   | full fix, steps to converge | about 6 400 | about 7 400 |

   With one update per step the two machines give the same level, unlike the old stack (finding F3).

## 4. Consequences for the deck and the thesis

- **Page 20** ("the first ensemble extension lost to HIL-SERL – four changes fix it"): true only for the old stack, and
  even there the success gap was not significant for experts and intermediates. Suggested wording: "slower convergence
  on the old stack, gone on the new stack; there, the warm start is what speeds an ensemble up".
- **Page 21** (the fixed ensemble matches HIL-SERL): confirmed on both machines on the new stack.
- **Thesis:**
  - report that no part of the fix is needed for final success on the new stack;
  - the warm start alone makes ensembles learn faster, and de-pessimisation adds nothing;
  - the warm start slows plain HIL-SERL;
  - the counterfactual result above supports the bias explanation of page 24.
- **For later campaigns:** the warm-started ensemble with the pessimistic target (`ens-wu`) is a sound default for the
  ensemble base on the new stack. It would be worth confirming on the second machine before switching.

## 5. Measurement

- **Everything is at the ceiling for the final policy.** Speed and effort separate the methods.
- **The best-of-3 window clashes with the early stop** (11 of the 13 low runs).
- Ruler success at fixed budgets, as in the table above, is the measure that showed the effects here.
- For future campaigns:
  - keep the budget checkpoints (2 000 / 4 000 / 6 000 / 8 000 environment steps), which `checkpoint_keep 12` can
    delete;
  - map learner to environment steps;
  - score the final policy with more attempts.
