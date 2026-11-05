# The comprehensive study on the second training machine

Planned 5 November 2026. Two studies on the new stack, each with a smoke campaign first, driven by
`examples/synthetic_suite.py --suite study`. Configs in `examples/campaigns/study/`. The second training machine
(RTX 5090) clones this repo from GitHub; its agent follows [comprehensive_study_agent.md](comprehensive_study_agent.md);
the results come back as one folder, `results/study/`.

**Decisions taken for it (5 Nov):**
- **No novice.** The synthetic novice is too weak: it answers 2–9 % of the robot's ten-step requests (it would need
  step 14), 97 % of its takeovers fall in failed episodes, and its careless mode barely differs from its normal one.
  It may come back with a recalibrated model.
- **The careless operator stays, at 20 %:** the same implementation, with a smaller share (0.2 instead of 0.4,
  persistence 0.7).
- **The behavioural score is used both ways** (discard and re-weight); the simulator-truth quality oracle re-weights;
  the perfect filter gives the ceiling.
- **About one week** of the second machine.
- **Critics: K = 4, 8 and 16.**

## The platform: the new stack with the shorter ruler

| part | setting | why |
|---|---|---|
| operator | rotation control; orientation channel with the stage-0 limits | the old operator rescued a twisted cuboid in 7–13 % of the tries, the new one in 70–100 % |
| environment | orientation box ±0.1 / ±0.1 / ±0.4 rad | as on the real robot |
| stop rule | mini-eval: 5 poses, ≥ 0.6 twice in a row, every 2 000 steps; cap 30 000 | training starts about four times sooner (≈ 4 600 steps to the first unaided success) |
| update ratio | exactly 1 learner step per environment step | the same learning per step on every machine and for every method (finding F3) |
| ruler | 9 poses, best window of 3; pre-training checkpoints skipped, newest 8 scored | 36 % of the ruler time, 0 of 57 scores changed on the second machine |
| logging | W&B online (tags `comprehensive_study`, campaign, arm, tier, operator, seed), takeover channels | |

## Study A: the number of critics and the asking rate (`A_consistent`, 96 cells)

Consistent operator, 3 experts + 3 intermediates, 2 seeds. One campaign, so every arm pairs with every other by
operator and seed.

| arm | what it is | question |
|---|---|---|
| `hilserl` | plain HIL-SERL: 2 critics, the operator decides | the published reference |
| `ens-k4`, `ens-k8`, `ens-k16` | ensemble extended with 4, 8, 16 critics (warm start, LR 1.5e-4, one random critic as target) | does the number of critics change success, effort and the teacher effect? |
| `asks-40`, `asks-55`, `asks-70` | the robot asks when its σ_Q is in the top 40 / 55 / 70 % of the last 2 000 steps (8 critics; 10 steps on, 5 off) | which asking rate gives the best success with a low ICC? |
| `random-55` | state-blind requests, p = 0.25 per free step (about 56 % requested) | does the timing of σ_Q beat chance at the same amount of asking? |

**Why these rates:** on the first machine success rose 70 → 81 → 87 % from asks 10 to 55 %. The 10-on/5-off timing caps
the requested share at 67 %, and asks 55 % already requests about 57 %; asks 70 % is the practical ceiling ("ask
whenever allowed"), asks 40 % checks the lower side.

**Measures and rules (fixed before the runs):**
- **Primary:** ruler success, environment steps until training stopped, operator share of the steps, first success
  without help; ICC per arm (with the permutation p).
- **Paired:** every arm against `ens-k8` and against `hilserl`, exact sign test over the 12 pairs (same operator and
  seed).
- **Critics:** the recommended K is the smallest whose success is not below the best K's (sign test p ≥ 0.05) and that
  does not need more steps; minutes per cell are reported next to it (K = 16 makes the learner slower).
- **Asking:** the recommended rate is the one with the highest success among those with ICC ≤ 0.2; the timing claim
  holds only if `asks-55` beats `random-55` with sign-test p < 0.05.

## Study B: can bad corrections be found, and does using a score help? (`B_quality`, 72 cells)

A sometimes-careless **expert** (3 experts, 3 seeds). A careless takeover moves like a novice (motor noise, tremor, lag,
speed, pauses, recovery), pushes down too early in half the attempts and aims about 6 mm beside the hole; 20 % of
takeovers are careless, in spells of about 4 (careful spells about 17). Every takeover logs its condition; only the
perfect filter reads it.

| arm | critics | how corrections are used |
|---|---|---|
| `hilserl` | 2 | every correction, uniform (the published method) |
| `e2hil` | 2 | E2HiL: drops batch states with extreme entropy influence from the actor loss |
| `hiril` | 2 (distributional) | HiRIL: risk-triggered requests, risk-difference replay, imitation |
| `keepall` | 8 | every correction, uniform: the same-architecture reference for the quality arms |
| `beh-discard` | 8 | drops a takeover from the demonstration half if its behavioural score is in the lowest 20 % of the last 200 |
| `beh-reweight` | 8 | keeps every takeover; replay priority rises with the behavioural score |
| `oracle-reweight` | 8 | keeps every takeover; replay priority from the simulator-truth oracle (trouble removed − jam + outcome) |
| `perfect-filter` | 8 | drops exactly the careless takeovers: the ceiling of any discard filter |

No outcome filter and no imitation loss in any AQI arm. The filters act after 1 000 steps (was 5 000, which covered
61 % of the takeovers last time). HiRIL and E2HiL run as in their papers (2 critics, no demo pre-training), like
`hilserl`.

**Questions and rules (fixed before the runs):**
1. **Can bad corrections be found?** AUROC of every score against the careless label, within run, training phase and
   onset time, on the takeovers of the quality arms (`audit`). A score is *valid* if AUROC ≥ 0.70 and the 95 % CI's
   lower bound > 0.5.
2. **Does using it help?** Each quality arm paired with `keepall` (same critics) and with `hilserl`: steps until training
   stopped, operator share and first unaided success are primary, ruler success secondary (it sits near the ceiling).
   - If `perfect-filter` does not beat `keepall`, there is no headroom: no score can help in this setting.
   - A score helps if its arm beats `keepall` on a primary measure with sign-test p < 0.05 and loses on none.
3. **Against the published methods:** `e2hil` and `hiril` paired with `hilserl` on the same measures.

## Size and time

| campaign | cells | purpose |
|---|---|---|
| `S0_smoke_A` | 3 (16 critics, asks 70 %, random) | new code paths of A run end to end; minutes per cell; full ruler for the ruler check |
| `S0_smoke_B` | 5 (E2HiL, HiRIL, discard, oracle, perfect filter) | new code paths of B, incl. HiRIL with the careless operator |
| `A_consistent` | 96 | study A |
| `B_quality` | 72 | study B |

About 45–70 minutes per cell: training about 25–40 minutes, the shorter ruler about 20. K = 16 and E2HiL train
more slowly, because the robot waits for the learner. That is about 7 days one cell at a time. Two cells side by side
(`--parallel 2`) should bring it down to 4–5 days; the smoke run measures whether two at once actually beat one.

## Run it

On the second training machine (Linux, RTX 5090). The repo comes from GitHub; the setup needs an NVIDIA driver with
CUDA 12.8 support or newer (the lock's CUDA 12.9 wheels support the RTX 5090; the earlier second machine ran the same
lock).

```bash
git clone <the GitHub URL of this repo> hil-serl_aqi && cd hil-serl_aqi
bash tools/setup_workstation.sh
./.venv/bin/wandb login                                     # once, by the user; never paste the key anywhere else
./.venv/bin/python examples/synthetic_suite.py --suite study check
./.venv/bin/python examples/synthetic_suite.py --suite study stage0
./.venv/bin/python examples/synthetic_suite.py --suite study run smoke --parallel 2
./.venv/bin/python examples/synthetic_suite.py --suite study rulercheck
./.venv/bin/python examples/synthetic_suite.py --suite study audit S0_smoke_B
./.venv/bin/python examples/synthetic_suite.py --suite study report smoke
./.venv/bin/python examples/synthetic_suite.py --suite study run A_consistent B_quality --parallel 2
./.venv/bin/python examples/synthetic_suite.py --suite study audit B_quality
./.venv/bin/python examples/synthetic_suite.py --suite study report
./.venv/bin/python examples/synthetic_suite.py --suite study plots
```

**Gates between the steps:**

| after | go on if | otherwise |
|---|---|---|
| `check` | READY (packages, JAX on the GPU, the sim renders, W&B key present, disk ≥ 50 GB) | fix what it lists |
| `stage0` | every test passes; careless rescues clearly below careful ones; orientation limits written | fix before training |
| smoke | all 8 cells trained and scored; HiRIL and E2HiL logged a condition per takeover; careless share between 0.1 and 0.3; `rulercheck` reports no changed score | read the cell's logs; a changed score means the full ruler for A and B (remove `skip_pretrain` and `newest`) |
| smoke, timing | two cells side by side finished in less than twice the time of one | run A and B with `--parallel 1` |

Everything resumes: re-running a `run` command skips finished cells.

## The results folder

Every driver command (`check`, `stage0`, `run`, `rulercheck`, `audit`, `report`, `plots`) ends by refreshing
**`results/study/`** in the repo, and `run` also refreshes it after every finished campaign. The folder is the only
thing to hand over for the analysis: no repo copy, no checkpoints. It is git-ignored, so pushing code never
carries it along. To refresh it by hand: `./.venv/bin/python examples/synthetic_suite.py --suite study export`.

| in `results/study/` | what |
|---|---|
| `README.md`, `export.json` | when, from which machine and commit, cells done per campaign, what was left out |
| `machine.json`, `stage0/` | the machine check; tests, operator conditions, tolerance sweep, orientation limits |
| `report.md`, `report_cells.csv`, `plots/` | all campaigns: per-arm table, paired tests, per-cell metrics, study figures |
| `<campaign>/report/` | operator dependency (`dependency.md/json`, its plots), `inconsistent_audit.*` (B), `ruler_reselect.*` (smoke) |
| `<campaign>/eval_runs/<cell>/` | ruler outcome (`cell.json`) and every trial (`trials.jsonl`) |
| `<campaign>/checkpoints/<cell>/` | the training record only: `episodes.jsonl`, `convergence_evals.jsonl`, `.done`, README, manifest |
| `<campaign>/aqi_logs/<run>/` | takeover segments with the careful/careless label, episodes, run metadata |
| `<campaign>/logs/`, `*.out` | every log, with the progress bars compacted to their last update and long logs cut to head and tail |

Left out: checkpoints, demonstrations, W&B folders and the per-step logs (`steps.jsonl`; `export --with_steps`
adds them). The layout is the runs folder's, so `report`, `plots` and `audit` run on it directly:
`python examples/synthetic_suite.py --suite study --root results/study report`. On the second machine's earlier
archive the export kept 46 MB of 86 GB.

**Figures** (`results/study/plots/`):
- `<campaign>_arms.png`: every arm;
- `A_consistent_critics.png`: success, steps, effort and minutes against K;
- `A_consistent_asking.png`: success against human effort, with ICC;
- `B_quality_scores.png`: AUROC of each score against the careless label.

W&B: project `hil-serl`, filter by the tag `comprehensive_study`.

## What is new in the code for it

- **HiRIL with the careless operator:** `train_hiril.py` takes `--synthetic_inconsistent`, `--synthetic_bad_share`
  and `--synthetic_bad_persistence` (built through `synthetic_operator_env.wrap_synthetic_operator`) and logs each
  takeover's condition; the campaign runner no longer refuses the combination.
- **The study suite:** `synthetic_suite.py --suite study`:
  - its own stage 0: tests, the operator in each condition, the tolerance sweep;
  - `run --parallel N`: shards with their own ports and a share of the GPU memory each;
  - a W&B login check;
  - `plots`;
  - the score audit restricted to the quality arms (`inconsistent_operator_audit.py --modes quality`).
  - `rulercheck`: would the shorter ruler change a smoke score;
  - `export`: the results folder, refreshed after every command.
- **`tools/setup_workstation.sh`:** the training machines' environment on a fresh Linux machine.
- **`tests/test_study_suite.py`:** the configs, every arm's flags, the design in the commands, the driver.

## Caveats

- **Different critics:** HIL-SERL, E2HiL and HiRIL use 2 critics, the quality arms 8. Compare a quality arm first with
  `keepall`.
- **HiRIL decides when to ask for help itself,** so against the quality arms it differs in who decides when help comes
  as well as in how corrections are used.
- **The careless expert keeps success high,** so success may sit near the ceiling for every arm. The primary measures
  are steps, effort and detection.
- **Synthetic operators:** absolute numbers say little about people; the comparisons are paired.
- **Never pool these cells** with another machine's or the first machine's (updates per step, platform).
