# `aqi_analysis.py` — offline analysis of training logs

Reads the JSONL logs written by `AQILogger` during training and answers the
three research questions from the training-time perspective.

---

## Purpose

A **quick diagnostic tool**: works on any subset of runs (one run, one mode,
partial data) and requires no eval ruler output.  The success rates it
reports come from the **training episode log** (online SR, includes operator
help) — not the autonomous frozen-policy ruler.  Use it to:

- check learning curves and intervention statistics during or after a run
- calibrate the trigger (RQ3) before enabling the active gate
- explore whether interventions correlated with improvement (RQ2)
- compare operator spread across modes once you have multiple operators (RQ1)

For the **formal study result** (autonomous SR, bootstrapped CIs, decision
tables) use `experiment_pipeline.py evaluate` or `run_campaign.py analyze`
instead — both read the eval ruler output.

---

## Inputs

All log directories must contain `episodes.jsonl` (written by every mode
except plain `hilserl`).  `steps.jsonl` is needed only for RQ3 trigger
calibration; `segments.jsonl` only for RQ2 intervention quality.

| Flag | Description |
|---|---|
| `--log_root <dir>` | scan all subdirs of this directory (the `aqi_logs/` folder) |
| `--run_dirs d1 d2 ...` | explicit list of run dirs (alternative to `--log_root`) |
| `--out <dir>` | output directory (default: `aqi_analysis_out/`) |
| `--baseline_mode <mode>` | mode used as the RQ1 variance baseline (default: `logged`) |
| `--metric <name>` | outcome metric for RQ1 (`final_success_rate` / `success_rate` / `mean_f_peak` / `intervention_step_frac`; default: `final_success_rate`) |
| `--fail_horizon <N>` | steps ahead used to build failure labels for RQ3 (default: 20) |

---

## Outputs (written to `--out/`)

| File | What it contains |
|---|---|
| `summary.md` | human-readable tables and RQ1/RQ2/RQ3 numbers |
| `results.json` | machine-readable version of the same data |
| `learning_curves.png/.svg` | rolling success rate per run over episodes |
| `operator_variance.png/.svg` | per-mode boxplot of final SR across operators (RQ1; needs ≥2 operators) |
| `q_int_by_operator.png/.svg` | counterfactual quality distribution per operator (RQ2) |
| `p_fail_calibration.png/.svg` | reliability curve: predicted P_fail vs empirical failure rate (RQ3) |

---

## What each research question needs

| RQ | Question | Minimum data |
|---|---|---|
| RQ1 operator variance | Does mode X reduce operator spread? | ≥2 operators × ≥2 modes |
| RQ2 intervention quality | Do interventions predict improved outcome? | ≥1 run with `segments.jsonl` |
| RQ3 trigger calibration | Does `sigma_q`/`P_fail`/`a_score` predict failure? | ≥1 run with `steps.jsonl` (logged/active/quality/aqi) |

With a single run you get: learning curve + intervention stats + RQ2 (if segments exist) + RQ3 (if steps exist). RQ1 metrics appear as `nan`.

---

## Examples

```bash
cd examples

# single run — learning curve + trigger calibration
python aqi_analysis.py \
    --run_dirs aqi_logs/sim_peg_insertion_logged_op0_s42_20260814-100000 \
    --out single_run_out/

# full multi-operator comparison
python aqi_analysis.py \
    --log_root aqi_logs \
    --baseline_mode logged \
    --out full_analysis_out/

# two specific runs side by side
python aqi_analysis.py \
    --run_dirs aqi_logs/sim_peg_insertion_logged_op0_s42_... \
               aqi_logs/sim_peg_insertion_aqi_op0_s42_... \
    --baseline_mode logged \
    --out comparison_out/
```

---

## Connections

```
train_aqi_rlpd.py (actor)  ──writes──▶  aqi_logs/<run>/
                                          ├── steps.jsonl
                                          ├── segments.jsonl
                                          ├── episodes.jsonl
                                          └── meta.json

aqi_analysis.py  ◀──reads──  aqi_logs/<run>/
                 ──writes──▶  summary.md, results.json, *.png
```

`aqi_analysis.py` is **fully standalone** — it does not call or import any
other script in `examples/`.  The shared statistical primitives (ICC,
Wilson interval) live in `serl_launcher/serl_launcher/aqi/stats.py`.
