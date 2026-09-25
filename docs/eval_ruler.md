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

## Connections

```
experiment_pipeline.py init  ──writes──▶  eval_grid.json
run_campaign.py              ──writes──▶  eval_grid.json

eval_ruler.py  ◀──reads──  checkpoint dir  +  eval_grid.json
               ──writes──▶  eval_runs/<cell>/trials.jsonl + cell.json

experiment_pipeline.py evaluate  ──reads──▶  cell.json  (primary outcome)
run_campaign.py analyze          ──reads──▶  cell.json  (primary outcome)
```
