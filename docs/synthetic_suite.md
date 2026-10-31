# Synthetic-operator test suite

This suite tests the operator and platform changes of patches 0014–0016 on any
machine that hosts the repo. Every setting is a normal campaign config in
`examples/campaigns/synthetic_suite/`, and one driver runs, checks and compares
them: `examples/synthetic_suite.py`.

| group | change | patch | what it does |
|---|---|---|---|
| behaviour | rotation control | 0014 | the synthetic operator also turns the peg back to an insertable orientation ([oracle_and_synthetic_operator.md §6.4](oracle_and_synthetic_operator.md)) |
| signal | orientation box | 0016 | the sim env clips roll, pitch and yaw like the real robot's safety box (§6.6) |
| signal | orientation channel | 0016 | a tilt or yaw error raises the operator's trouble signal `z_t` (§6.6) |
| signal | takeover-channel logging | 0015 | logging only: which channel caused each takeover and how it ended (§6.5) |
| convergence | mini-eval convergence | 0015 | the autonomous fixed-pose mini-eval decides when a cell has converged ([methods.md §3.6](methods.md)) |
| convergence | fixed update ratio | 0015 | actor and learner hold the learner steps per env step fixed, so learning per env step no longer depends on the GPU (§3.6) |

**Why.** In the first 10 000 env steps of the cuboid campaigns 90–99 % of the
episodes ran out of time, and the first success without help came only after
about 17 000 steps. Most of that traces back to the operator and the task: the
operator never turned the peg, the task never bounded the twist, and the trouble
signal never saw it. The suite measures how much each change helps.

## The settings

Each setting is its own campaign, because the switches are campaign-wide. All
pilots record the takeover channels; that is logging only.

| setting | rotation control | orientation box | orientation channel | convergence | update ratio | role |
|---|---|---|---|---|---|---|
| `L0_original` | – | – | – | rolling | asynchronous | reference: the platform of every reported campaign |
| `L1_rotation` | ✓ | – | – | rolling | asynchronous | ladder |
| `L2_box` | ✓ | ✓ | – | rolling | asynchronous | ladder |
| `L3_channel` | ✓ | ✓ | ✓ | rolling | asynchronous | ladder |
| `L4_evalconv` | ✓ | ✓ | ✓ | mini-eval | asynchronous | ladder |
| `L5_full` | ✓ | ✓ | ✓ | mini-eval | fixed, 1 | ladder: the full stack |
| `X_nobox` | ✓ | – | ✓ | mini-eval | fixed, 1 | leave-one-out from L5 |
| `X_nochannel` | ✓ | ✓ | – | mini-eval | fixed, 1 | leave-one-out from L5 |
| `X_noeval` | ✓ | ✓ | ✓ | rolling | fixed, 1 | leave-one-out from L5 |
| `S_smoke` | ✓ | ✓ | ✓ | mini-eval | fixed, 1 | one-cell smoke test (round peg, expert) |
| `C_reference` | – | – | – | rolling | asynchronous | confirmation, original platform |
| `C_full` | ✓ | ✓ | ✓ | mini-eval | fixed, 1 | confirmation, the chosen stack |

- **Ladder (L0 → L5):** each step adds one change, so the step-to-step
  difference is that change's added value.
- **Leave-one-out (X):** each removes one change from the full stack, so the
  difference to L5 is what that change still contributes once all the others
  are on. Removing the update ratio is L4; rotation control cannot be removed
  alone, because the channel needs it.
- **Pilots:** plain HIL-SERL, 1 operator per tier, 1 seed (3 cells), cuboid,
  30 000-step cap, the usual convergence stop with the mini-eval gate.
- **Confirmation:** the thesis base configuration (3 operators per tier × 2
  seeds, the fixed ensemble) with the 35 000-step cap of `active_ratematch`. Arms:
  baseline, ensemble extended and the best asking arm (requests in the 55 % most
  uncertain steps), defined exactly as in `active_ratematch`.
- **Common to all:**
  - orientation box ±0.1 rad roll and pitch, ±0.4 rad yaw;
  - update ratio 1 learner step per env step;
  - mini-eval every 2 000 env steps, stop after 2 passes in a row;
  - orientation-channel limits from stage 0 (below).

## Prerequisites (any machine)

1. The repo with patches 0014, 0015 and 0016 applied (or a later commit that
   contains them), its `.venv`, and the sim environment
   (`examples/experiments/sim_peg_insertion/sim_env.sh`). A separate copy of the
   repo needs a `.venv` too, or a link to an existing one
   (`ln -s <repo>/.venv <copy>/.venv`). The campaign runner starts `python`
   after sourcing `sim_env.sh`, which activates `<copy>/.venv`; without it, the
   system Python would run.
2. One GPU that nothing else uses. The campaign runner's pre-flight refuses
   a busy GPU.
3. Disk: measure after the smoke test (`du -sh runs/synthetic_suite/S_smoke`)
   and multiply by the number of cells.
4. W&B (optional): put `WANDB_API_KEY` into the environment yourself and pass
   `--wandb online`. Never paste the key into a chat or a file in the repo.

All paths are relative to the repo. `--root` moves the suite folder, for
example to a larger disk: `--root /data/synthetic_suite`.

## Stage 0 – no training (CPU, about 1 h)

```bash
cd <repo>
./.venv/bin/python examples/synthetic_suite.py check
./.venv/bin/python examples/synthetic_suite.py stage0
```

- **`check`** writes `runs/synthetic_suite/machine.json` and prints READY or
  what is missing. It checks:
  - which of the patches are present;
  - the packages, and that JAX sees the GPU;
  - that `franka_sim`, `serl_launcher` and `experiments` are imported from this repo;
  - that the sim env renders, with the orientation box on;
  - the free disk space.
- **`stage0`** runs, all into `runs/synthetic_suite/stage0/`:
  - **the tests**: rotation control, update ratio, takeover channels, campaign switches, orientation box, orientation channel, the sim orientation test and the suite itself;
  - **the operator test**: the operator alone in the cuboid and cylinder task, solo and after 60 random actions, per tier, with and without rotation control;
  - **the tolerance sweep**: success against an initial tilt or yaw of 0–22°.

  It writes `limits.json`: the expert's tolerance with rotation control off,
  which becomes the orientation channel's limits. A shape or axis without a
  measured value keeps the default: cuboid yaw 0.25 rad, tilt 0.10 rad.

**Go on only if:**
- every test passes;
- with rotation control on, the cuboid's solo and rescue success is clearly
  above rotation control off;
- the sweep gives a tolerance for the cuboid's tilt and yaw.

If rotation-on success is lower than off, the target orientation is wrong:
stop and report (assumption RC2 of `handoff/EVALUATE_FIRST.md`).

## Stage 1 – smoke test (GPU, about 30–45 min)

```bash
./.venv/bin/python examples/synthetic_suite.py run smoke
./.venv/bin/python examples/synthetic_suite.py report smoke
grep -h "orientation box\|orientation monitor\|fixed update ratio\|mini-eval at step" runs/synthetic_suite/S_smoke/logs/*.log | sort | uniq -c
```

**Expected:**
- the box line in the actor, learner and ruler logs;
- the orientation monitor and its limits in the actor's start-up line;
- `fixed update ratio` in both processes;
- mini-evals after the first unaided success;
- takeover channels in `episodes.jsonl`;
- a scored cell.

Its numbers are not results.

## Stage 2 – pilots (GPU, about 1.5–2 days on one GPU)

```bash
nohup ./.venv/bin/python examples/synthetic_suite.py run pilots --wandb online > runs/synthetic_suite/pilots.out 2>&1 &
./.venv/bin/python examples/synthetic_suite.py plan        # progress per setting
./.venv/bin/python examples/synthetic_suite.py report pilots
```

27 cells, one campaign after the other. A cell takes about 0.8–1.5 h. The
settings with the fixed update ratio may take up to about 100 min, because the
env then waits for the learner. If a run stops, the same command resumes:
finished cells are kept, unfinished ones restart.

**Reading the report** (`runs/synthetic_suite/report.md`, per cell in `report_cells.csv`):

| measure | meaning | primary? |
|---|---|---|
| first unaided success | env steps to the first success without any takeover (median over the cells) | **yes** – the same under every convergence rule |
| ruler SR | success on the fixed 9-pose ruler, as in every campaign | **yes** |
| timeouts / success ≤ 10 k | share of the first-10 k episodes that ran out of time / succeeded | the slow start itself |
| operator share | operator-controlled steps / all steps | help needed |
| orientation takeovers | share of takeovers caused by the orientation channel (first 10 k) | does the channel fire? |
| time-cap hand-backs | share of takeovers that ended at `d_max` (first 10 k) | early hand-back |
| convergence step | compares only between settings with the same rule | – |
| actor minutes | wall-clock time of the actor | cost |

**From the pilots to the confirmation.** One cell per tier shows the direction
and size of an effect, not its significance. As a working rule, keep a change
in the confirmation stack if:
1. adding it (ladder) or removing it (leave-one-out) moves the first unaided
   success in the same direction in at least 2 of the 3 tiers, and
2. the ruler SR does not drop by more than 10 percentage points.

Drop a change that does not pass: a smaller stack is easier to defend. Two
special cases:
- **Mini-eval convergence** changes only when a cell stops, so judge it by the
  ruler SR and the wall-clock time, not by the convergence step.
- **The fixed update ratio** is also what makes cells from different machines
  comparable. Keep it if the confirmation or later campaigns may run on another
  machine and it does not hurt.

## Stage 3 – confirmation (GPU, about 2–6 days per campaign)

1. Edit the switches in `C_full.yaml` to the stack chosen from the pilots, and
   keep `C_reference.yaml` as it is.
2. **On a machine with finished `active_ratematch` cells, skip `C_reference`.**
   Its commands equal those of `active_ratematch` for these three arms, apart from
   the takeover logging. Copy the cells instead (they must come from this machine):

   ```bash
   ./.venv/bin/python examples/copy_campaign_cells.py --src runs/active_ratematch --dst runs/synthetic_suite/C_reference --arms hilserl,logged,active-q55
   ```

3. Run them (`C_reference` first, unless copied), then report:

```bash
nohup ./.venv/bin/python examples/synthetic_suite.py run confirm --wandb online > runs/synthetic_suite/confirm.out 2>&1 &
./.venv/bin/python examples/synthetic_suite.py report confirm
```

The report adds a paired comparison: the same arm, operator and seed in
`C_full` and `C_reference`, with wins, losses and an exact sign test. Each
campaign also writes the usual dependency report (`report/dependency.md`),
with ICC and the comparisons between arms.

`C_reference` may be skipped only if finished cells with this roster, these
arms and this budget exist **from the same machine**. Check with
`examples/roster_audit.py` and `machine.json`.

## A new machine

- Copy or clone the repo with the patches. Create the `.venv` and run `check`
  until it prints READY. Then run stage 0 on that machine: the sweep is cheap,
  and the limits belong to the machine's own simulator build.
- **One machine runs the whole suite.** Updates per env step differ between
  machines (finding F3). The report reads the machine of every campaign and
  warns when they differ. If the machine changes mid-suite, re-run `L0_original`
  (and `C_reference`) on the new machine before comparing.
- Results with the fixed update ratio on are the most portable: the learning
  per env step is then the same everywhere, and only the wall-clock time differs.

## Guardrails

- The suite never touches a finished campaign. Every setting has its own folder
  under `runs/synthetic_suite/`. The driver refuses a folder that was started with
  another config or other limits (`suite_setting.json`).
- With all switches off (`L0_original`, `C_reference`), the operator, the task
  and the training commands are those of the reported campaigns. Only the
  takeover logging is added, which is logging only.
- Results with any switch on are a new operator or platform definition. Using
  them in the thesis means re-running the comparisons they affect; that is a
  decision for after the pilots.

## Files

| file | what |
|---|---|
| `examples/campaigns/synthetic_suite/*.yaml` | the 12 configs (9 pilots, smoke, 2 confirmation) |
| `examples/synthetic_suite.py` | driver: `check`, `stage0`, `plan`, `run`, `report` |
| `tests/test_synthetic_suite.py` | the ladder and leave-one-out structure, the runner accepts every config, the confirmation keeps the thesis base, the driver's limits, guards and report |
| `runs/synthetic_suite/` | `machine.json`, `stage0/`, one folder per setting, `suite_log.jsonl`, `report.md`, `report_cells.csv` |
