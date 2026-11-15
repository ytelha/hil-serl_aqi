# The final study: does asking help, does grading help, do they add up?

*Set up 2026-11-15 for the second training machine. Plan and reasoning: `docs/thesis/final_experiments_plan.md`.*

## 1. The question and the design

The earlier campaigns answered part of it:
- asking changes how much help the robot uses, and who decides when;
- the behavioural score finds careless takeovers (AUROC 0.70–0.76).

They did not answer three things:
1. **Can asking save help?** Every earlier asking rate used as much help as the operator, or more.
2. **Does grading help training?** Careless takeovers never cost anything there: keeping everything did as well as the
   perfect filter.
3. **Do asking and grading add up?** AQI (both) has never run cleanly.

So one design, asking × grading on the same operators and seeds:

| arm | asking | grading | what it is |
|---|---|---|---|
| `keepall` | – | – | the ensemble; the operator decides when to help; keeps every correction |
| `asks-15` | ✓ | – | the robot asks in its 15 % most uncertain recent steps |
| `beh-discard` | – | ✓ | the behavioural score drops the lowest-scored 40 % of corrections |
| `aqi-15` | ✓ | ✓ | both |
| `perfect-filter` | – | oracle | drops exactly the careless corrections: the most grading could win |
| `random-15` | random | – | requests at random times, about as often: does the timing matter? |
| `hilserl` | – | – | plain HIL-SERL: 2 critics, no warm start |

**The four factorial cells run the same pipeline:** every correction is scored at the end of its episode. They differ
only in the two switches; a test checks that their commands are identical apart from them.

**Settings shared by every arm:**
- **Operators:**
  - 2 experts, 2 intermediates and 2 novices;
  - the novice is stronger than before: skill 0.3 for execution *and* monitoring (the old one was 0);
  - every operator is careless in 40 % of its takeovers, in spells (persistence 0.7), with a hidden label.
- **Requests** (the asking and random arms): held up to 30 steps until the operator takes over. They replace the
  operator's own monitor.
- **Ensemble base** (the ablation's fastest): 10 critics, a minimum over 2 in the target, a warm start of 5 000 steps,
  learning rate 3e-4.
- **New stack:** rotation control, orientation box and channel, the mini-evaluation stop, one update per step.
- **No outcome filter and no imitation loss anywhere.**

**Two campaigns, identical apart from the demonstrations:**
- `F1_hard`: 3 per operator;
- `F2_standard`: 6 per operator.

Each has 7 arms × 6 operators × 2 seeds = 84 runs. F1 runs first. The smoke test `F0_smoke` comes before both: 3
arms × 1 expert and 1 novice, 4 000 steps.

**Measured:**
- ruler success at every kept checkpoint: the newest 8 plus every 2 000th learner step up to 17 000, which gives
  success at fixed budgets;
- steps to the first success without help, and operator steps;
- the final ruler and the last checkpoint;
- score validity, audited separately for operator-decided and robot-asked takeovers (`report/audit_robot_asks/`).

## 2. Running it

For the agent on the training machine: [final_study_agent.md](final_study_agent.md) (rules and steps).

**Once, on a fresh clone:**

```bash
git clone https://github.com/<owner>/hil-serl_aqi.git && cd hil-serl_aqi
```

```bash
bash tools/setup_workstation.sh
```

```bash
./.venv/bin/wandb login
```

The W&B login is optional; without it, W&B logs offline on the machine.

**Then everything, unattended, in the background:**

```bash
bash tools/run_final_unattended.sh
```

It runs:
1. the machine check;
2. stage 0 (tests, the operator in each condition with the tiers expert, intermediate and `s=0.3`, the tolerance
   sweep);
3. the smoke campaign;
4. `F1_hard`, then `F2_standard`;
5. one retry of any run whose actor crashed;
6. the score audit, the report and the plots.

**Nothing waits for a person.** Every gate becomes a decision in `results/final/RUN_STATUS.md`:
- **W&B:** online if logged in, else offline;
- **cells side by side:** 2 if the smoke runs show that two at once beat one;
- **disk, per campaign:** if the free disk is short, that campaign keeps only its newest checkpoints;
- **retries:** each one is listed.

A failing step is recorded and the run goes on.

**Watch it** (optional):

```bash
tail -f runs/final/unattended.out
```

```bash
cat results/final/RUN_STATUS.md
```

**The results:** everything the analysis needs is in **`results/final/`**, refreshed after every step and every 6 hours:
- reports and plots;
- ruler outcomes and training records;
- takeover logs and compacted logs;
- `RUN_STATUS.md`, `machine.json` and stage 0.

There are no checkpoints and no demonstrations in it. Copy that folder. It is never pushed anywhere: the repository is
public.

**Time:** about 1 h for check and stage 0, about 1 h for the smoke, then about 2–3 days per main campaign with two runs
at a time. F1's results are complete in `results/final/` as soon as F1 finishes. Keep the machine on: lock the screen,
don't suspend.

## 3. Code changes for this study (all tested in `tests/test_final_suite.py`)

| change | where | why |
|---|---|---|
| keep every 2 000th checkpoint besides the newest | `examples/ckpt_retention.py`; `train_aqi_rlpd.py --checkpoint_keep_every/--checkpoint_keep_until`; `budget.checkpoint_keep_every/until` | success at fixed budgets in runs that train long; off (0) by default, so earlier campaigns run unchanged |
| simulator guard | `franka_sim/.../orientation_box.py` `safe_quat`, `safe_action`; `panda_peg_insert_gym_env.py` step | a non-finite step left a zero quaternion that crashed the next step ("Found zero norm quaternions": one E2HiL, one perfect-score and one novice-test run) |
| crashed runs are not finished runs | `run_campaign.py` `actor_crashed`, `.crashed` marker, eval skips them | a crash used to be marked `.done` and scored |
| retry once, decide disk per campaign, no shorter-ruler check | `synthetic_suite.py` (`--suite final`) | unattended |
| unattended wrapper without push | `tools/run_final_unattended.sh` | results stay on the machine |
