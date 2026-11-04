# `eval_ruler.py` — the fixed evaluation ruler

Evaluates one frozen policy (one experimental cell) on a pre-generated
initial-state grid: autonomous trials in a fixed order, policy frozen, no
human in the loop, active gate disabled.

---

## Purpose

Produces the **unbiased outcome metric** used by both `experiment_pipeline.py`
and `run_campaign.py`: success rate on new starting poses the policy never saw
during training, without any operator help.  The ruler's result is what the
whole operator-dependency analysis is built on.

---

## Inputs

| Input | Flag | Notes |
|---|---|---|
| Trained checkpoint dir | `--checkpoint_path` | e.g. `runs/cylinder_logged_op0_s42` |
| Pose grid file | `--grid_file` | `eval_grid.json` written by `experiment_pipeline.py init` or `run_campaign.py` |
| Experiment name | `--exp_name` | same as training, e.g. `sim_peg_insertion` |
| Output directory | `--out` | written per-cell, e.g. `eval_runs/logged_op0_s42` |

Optional flags:

| Flag | Default | Meaning |
|---|---|---|
| `--eval_checkpoint_step` | 0 (latest) | specific step to evaluate |
| `--argmax` | `True` | deterministic policy (recommended for ruler) |
| `--control_hz` | 10.0 | used to convert steps → seconds |
| `--f_max` | `None` | force threshold for the violation flag (N) |
| `--max_steps` | 0 (env default) | step cap per trial |
| `--cell_id` | basename of `--out` | identifier stored in cell.json |
| `--save_video` | `False` | save per-trial videos |
| `--select_min_step` | 0 (all) | with `--select best/mean/best_window`: score only checkpoints saved after this learner step (see *A shorter ruler*) |
| `--select_stride` | 1 (all) | every n-th checkpoint, counted back from the newest; changes the score |
| `--select_newest` | 0 (all) | only the newest N checkpoints |

---

## Outputs (written to `--out/`)

| File | What it contains |
|---|---|
| `trials.jsonl` | one JSON record per trial: `success`, `steps`, `time_s`, `f_peak`, `f_mean`, `tau_peak`, `violation`, `dx_mm`, `dyaw_rad` |
| `cell.json` | aggregated outcomes: `sr` (success rate), `sr_wilson95` CI, `time_to_success_mean`, `f_peak_mean`, `violation_rate`, grid SHA-256, checkpoint step |
| `manifest.json` / `README.md` | provenance: operator, method, conditions, reproduce command |

---

## Example

```bash
cd examples

# evaluate the latest checkpoint of a logged baseline run for op0
python eval_ruler.py \
    --exp_name sim_peg_insertion \
    --checkpoint_path runs/cylinder_logged_op0_s42 \
    --grid_file ~/campaigns/study1/eval_grid.json \
    --out ~/campaigns/study1/eval_runs/logged_op0_s42 \
    --argmax
```

---

## Grid integrity

The ruler refuses a grid whose SHA-256 does not match its own content.
The grid is generated **once** by `experiment_pipeline.py init` or
`run_campaign.py` and then frozen for the entire campaign, guaranteeing
every cell is scored on exactly the same starting poses.

---

## A shorter ruler (optional, off by default)

**Why the ruler is slow.** Campaigns use `--select best_window`: the ruler scores *every* saved checkpoint (one per
1000 learner steps) on the 9 grid trials and reports the best mean over 3 consecutive checkpoints. Most of its
time goes into checkpoints that fail every trial, because a failed trial runs to the 300-step limit and a success
ends after 12–17 steps. On the second training machine a cell's ruler simulated about 30 000 steps, three times
its training, about 55 minutes (`docs/thesis/new_stack_operator_results_and_qa_log.md` §5c).

**Three shortenings** (`examples/ruler_select.py`, shared by the ruler and the re-selection tool):

| YAML `eval:` key | runner flag | ruler flag | what is scored |
|---|---|---|---|
| `skip_pretrain: true` | `--eval_skip_pretrain` | `--select_min_step N` | only checkpoints after the arm's demo pre-training (N = its `demo_pretrain_steps`; an arm without pre-training keeps all) |
| `stride: 2` | `--eval_stride 2` | `--select_stride 2` | every 2nd checkpoint, counted back from the newest |
| `newest: 8` | `--eval_newest 8` | `--select_newest 8` | only the newest 8; the older checkpoint files stay on disk |

**Measured on the 57 cells of the second machine** (new stack, cuboid), by re-selecting from their own
`trials.jsonl` (`examples/ruler_reselect.py`):

| selection | checkpoints scored | ruler time | cells whose score changes |
|---|---|---|---|
| every checkpoint (now) | 14.9 | 100 % (≈ 55 min) | – |
| skip pre-training | 9.9 | 55 % | 0 |
| newest 8 | 8.0 | 39 % | 0 |
| **skip pre-training + newest 8** | **7.6** | **36 % (≈ 20 min)** | **0** |
| every 2nd checkpoint | 7.8 | 51 % | 46, by −21 points on average (up to −33) |
| every 2nd, window of 2 | 7.8 | 51 % | 24, −1 point on average (−17 to +15) |

Why the first two keep every score: no cell succeeded at any pre-training checkpoint, and the best window always
sat among the newest checkpoints, because training stops once the mini-eval passes twice. Why *every 2nd* changes
the score: only the last two or three checkpoints are good, so thinning removes one of them and the window of three
has to reach back into failing ones. It is implemented, but it measures something else; do not use it to save time.
Newest 4–6 also changed no cell here (7–22 % of the time); 8 leaves a margin for a policy that oscillates.

**Not measured on the old stack.** Without the mini-eval stop (rolling-window convergence, a 35 k cap) the best
window may sit earlier. Measure on that machine's own finished cells before switching (read-only, seconds):

```bash
./.venv/bin/python examples/ruler_reselect.py --campaign runs/active_ratematch --skip_pretrain --newest 8
```

**Rules.**
1. Choose the selection before a campaign starts and keep it for every cell of that comparison. The runner
   refuses, before any training, to evaluate a campaign whose already-scored cells used another selection.
2. Reference cells copied from another campaign (`copy_campaign_cells.py` copies results, not policies) can be
   re-selected in the new folder: `ruler_reselect.py --campaign <new> --skip_pretrain --newest 8 --apply`. It
   rewrites only the cells in `copied_cells.json` and keeps each original as `cell.full_ruler.json`. Do this only
   after `--verify` (below) showed IDENTICAL on that machine.
3. Never add the keys to a finished or running campaign; use a new campaign folder.
4. `budget.checkpoint_keep` is something else: it deletes older checkpoint files during training and changes the
   learner's command. To shorten the ruler, use `eval.newest`.

**The assumption, and its check.** Re-selecting is exact only if a checkpoint's trials do not depend on what was
scored before it. The eval reset is deterministic (fixed pose, `mj_resetData`, no random draw) and the actions are
the policy's mode, so it should hold. Check it once on a GPU (10–20 minutes; the cell needs its `checkpoint_*`
folders): `--check_cmd` prints the cell's own ruler command from `logs/<cell>.eval.log`, re-pointed to
`/tmp/ruler_check/<cell>` with the shorter ruler's flags. Run what it prints, then compare trial by trial:

```bash
./.venv/bin/python examples/ruler_reselect.py --campaign runs/<campaign> --skip_pretrain --newest 8 --check_cmd <cell>
```

```bash
./.venv/bin/python examples/ruler_reselect.py --verify /tmp/ruler_check/<cell> --against runs/<campaign>/eval_runs/<cell>
```

Expected: `IDENTICAL`. Pick a cell whose newest checkpoints have both successes and failures, so the check means
something.

**Records.** `cell.json` gets `checkpoint_filter` (`min_step`, `stride`, `newest`, `n_saved`, `n_scored`) and
`campaign.json` gets `ruler_checkpoints`; both are absent when the ruler scores every checkpoint. With all keys
off, every existing config gives the same commands as before (64 configs, 8 555 command lines compared).

---

## Connections

```
experiment_pipeline.py init  ──writes──▶  eval_grid.json
run_campaign.py              ──writes──▶  eval_grid.json

eval_ruler.py  ◀──reads──  checkpoint dir  +  eval_grid.json
               ──writes──▶  eval_runs/<cell>/trials.jsonl + cell.json

experiment_pipeline.py evaluate  ──reads──▶  cell.json  (primary outcome)
run_campaign.py analyze          ──reads──▶  cell.json  (primary outcome)

ruler_select.py   ◀──imports──  eval_ruler.py, run_campaign.py, ruler_reselect.py
ruler_reselect.py ──reads──▶  eval_runs/<cell>/cell.json + trials.jsonl (--apply: copied cells only)
```
