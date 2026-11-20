# Extension study: phases 0–2

The final study ([final_study.md](final_study.md)) left three gaps in its design:
- every arm met the same sometimes-careless operator, so no arm shows what the carelessness costs;
- corrections were learned from but never imitated, so a discarded correction could change little: its steps stay in
  the online buffer;
- it used a single task, the cuboid peg, which every careful tier performs reliably.

The extension study closes these gaps in four phases. This document covers phases 0–2, which run unattended in one go
on the second training machine. Phase 3, the main study on the task chosen in phase 2, is prepared after its decision.

```bash
bash tools/run_extension_unattended.sh            # phases 0-2, in the background
tail -f runs/extension_pipeline/pipeline.out      # progress
```

| phase | campaigns | runs | what it answers |
|---|---|---|---|
| 0 | `final/F3_careful_hard`, `final/F4_careful_standard` | 24 | What does the carelessness cost when corrections are not imitated? |
| 1 | `final/F5_bc_standard`, `final/F6_bc_careful_standard` | 60 | Does grading matter once the learner imitates the corrections? |
| 2 | operator screen, `extension/P1–P4` pilots, decision, `extension/S1–S4` smoke | ≤ 56 | Which harder task has headroom and operators that differ? Do the new arms run? |

## Phases 0 and 1: twins of the final study

Phase 0 also ran on its own (19–20 Nov, commit a290c12, into `runs/final_careful/`). The twins runner does not
repeat a twin that finished there.

A twin is a final-study campaign with one deliberate difference. It uses the final study's operators, demonstrations,
ruler (`eval_grid.json`) and orientation limits. These are copied from `runs/final/<source>/` and checked by hash before
the first run. A twin whose inputs differ is skipped, never trained on other inputs. Results are paired by operator
and seed with the source campaign.

| twin | source | operator | arms (12 runs each: 2 experts, 2 intermediates, 2 novices at s = 0.3, seeds 0–1) |
|---|---|---|---|
| `F3_careful_hard` | `F1_hard` | careful | `keepall` |
| `F4_careful_standard` | `F2_standard` | careful | `keepall` |
| `F5_bc_standard` | `F2_standard` | careless 40 % | `bc`, `bc-discard`, `bc-random`, `bc-perfect` |
| `F6_bc_careful_standard` | `F2_standard` | careful | `bc` |

**Imitation arms.**
- **Common to all:** an imitation (behaviour-cloning) loss on the corrections, weight 0.1 (the value of every earlier
  imitation run). They use `quality_gamma: 0`, so every kept correction weighs the same, and the arms differ only in
  which corrections are dropped from the intervention buffer and the imitation loss.
- **`bc`:** none.
- **`bc-discard`:** the lowest 40 % by the behavioural score.
- **`bc-random`:** a random 40 % (the same loss of data, chosen blindly).
- **`bc-perfect`:** exactly the careless ones (reads the label; the ceiling).

`tests/test_final_twins.py` checks that every learner, actor and evaluation command equals the source's `keepall`
command except for these flags.

**Analysis (fixed now).** Pairs by campaign, operator and seed. Contrasts:
- **The cost of carelessness:**
  - careful keepall − careless keepall (F3 − F1, F4 − F2);
  - careful bc − careless bc (F6 − F5).
- **What grading recovers:**
  - with imitation: bc-discard − bc, bc-perfect − bc;
  - without imitation: the final study's beh-discard and perfect-filter against keepall.
- **Score or data loss:** bc-discard − bc-random.
- **Does imitation help at all:** bc − keepall, careful (F6 − F4) and careless (F5 − F2).

Measures: evaluation success, success after 4 000 / 6 000 / 8 000 environment steps, environment steps until the stop,
operator steps. Exact sign and Wilcoxon tests; per-run records in every results document.

## Phase 2: a harder task

**Candidates** (pilot configs `extension/P*`; the task knobs reach every process: demonstrations, training, ruler,
operator test):

| candidate | task |
|---|---|
| `hex` | hex peg: 3 mm clearance and yaw alignment (cuboid: 5 mm) |
| `tight` | cuboid, lateral success tolerance 6 mm instead of 12 mm (`task.success_lateral: 0.006`) |
| `wide` | cuboid from a wider start range, ±25 mm and ±0.2 rad instead of ±15 mm and ±0.08 rad (`task.random_xy`, `task.random_rz`); ruler and mini-evaluation poses widened to ±24 mm |
| `hexwide` | hex peg from the wider start range |

**Steps** (`python examples/synthetic_suite.py --suite extension all`):

1. **Stage 0:** tests, and the orientation tolerance sweep for the cuboid and the hex peg (orientation-channel
   limits).
2. **Operator screen** (CPU), per candidate:
   - tiers: expert, intermediate, novice skill s = 0.3 / 0.15 / 0;
   - careful and careless;
   - alone and rescue;
   - rotation control on, 30 episodes each.

   **Rule (fixed now):**
   - a candidate passes if the careful expert succeeds alone ≥ 90 % and rescues ≥ 80 %, the careless expert succeeds
     alone ≤ 60 %, and some novice skill succeeds alone in 40–70 %;
   - that skill becomes the candidate's novice (closest to 55 %; on a tie, the higher skill).
3. **Learning pilots** of the passing candidates:
   - arms: HIL-SERL, `keepall`, `bc`;
   - one careful intermediate and one careful novice at the screened skill (passed as `--tier_s_exec` /
     `--tier_s_mon novice=<s>`);
   - seeds 0–1, 12 runs per candidate.
4. **Decision** (`decision.json`, `DECISION.md`). **Rule (fixed now):**
   - **A candidate qualifies** if keepall with the intermediate reaches 30–70 % evaluation success after 8 000
     environment steps and ≥ 80 % evaluation success in the end.
   - **Choice among qualifiers:** the one with the largest intermediate − novice gap at 8 000 steps.
   - **If none qualifies:** the one with the lowest success at 8 000 steps among those still ≥ 50 % in the end, with
     a step cap of 40 000 for phase 3.
   - **If none is learnable:** phase 3 stays on the standard task.
   - **Budget B** for phase 3's primary endpoint: the environment step (2 000–20 000, every 2 000) where keepall with
     the intermediate is closest to 50 %.
5. **Smoke test** on the chosen task:
   - every new arm type once: `bc-discard`, `bc-reweight`, `bc-outcome`, `bc-perfect`, `asks-15-bc`, `aqi`, `asks-30`,
     E2HiL;
   - a careless intermediate, seed 0, a cap of 8 000 steps.

**Phase 3** (after the decision) uses the chosen task, its novice skill and B:
- **the main campaign, sometimes-careless operator:**
  - references: HIL-SERL, E2HiL;
  - without imitation: keepall, discard, perfect filter;
  - with imitation: bc, bc-discard, bc-reweight, bc-outcome, bc-random, bc-perfect;
  - asking: asks-15, asks-30, random-15, asks-15-bc, AQI;
- **the careful twin:** HIL-SERL, keepall, bc, bc-discard, asks-15.

## Guardrails

- `runs/final/` is only read. The twins live in `runs/final_twins/`, phase 2 in `runs/extension/`.
- Results go to `results/final_twins/`, `results/extension/` and `results/extension_pipeline/`, never pushed.
- **Disk.** After each campaign is scored, `synthetic_suite.py prune` deletes the checkpoints the ruler result no
  longer needs. It keeps each run's best window, its last scored and its newest checkpoint.
  - It works only in `runs/final_twins/` and `runs/extension/`, or in a root that holds `.prune_scored_ok`.
  - If the disk is still short, a campaign keeps only its newest checkpoints, and this is recorded.
- **No stops.**
  - Nothing waits for a person: a failing step is recorded (`TWINS_RUN.md`, `RUN_STATUS.md`) and the next one starts.
  - A crashed run is retrained once.
  - A restart resumes: finished runs, stage 0 and the screen are kept.
- **New switches.** All are off by default; every earlier config's commands are byte-identical (checked by a dry-run
  snapshot of all 74 configs):
  - `quality_gamma` (trainer flag, per arm);
  - the task knobs (`task.success_lateral`, `task.random_xy`, `task.random_rz` → `$SIM_SUCCESS_LATERAL`,
    `$SIM_RANDOM_XY`, `$SIM_RANDOM_RZ`, read by `SimPegEnv` via `franka_sim/envs/task_knobs.py`);
  - `--tier_s_exec` / `--tier_s_mon` on the command line.
