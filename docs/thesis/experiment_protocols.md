# Experiment protocols — exact runbook (setup → run → save → analyse → plot)

Companion to [experiment_plan.md](experiment_plan.md). Every command is copy-paste
runnable and states its **working directory** and **where results are saved**. All
study outputs live under `runs/<name>/` at the repo root.

## Conventions & prerequisites

```bash
# Set these once per shell. $PY is absolute, so it works from any directory.
REPO=/home/user/Downloads/hil-serl_aqi
PY="$REPO/.venv/bin/python"
cd "$REPO"

# Interpreter: the repo venv (python 3.12). Do NOT use system python.
$PY -c "import serl_launcher, jax; print('jax', jax.default_backend())"   # sanity

# GPU is shared (8 GB). BEFORE launching a campaign, check nobody else is training:
nvidia-smi
```

Every block below opens with its working directory. Rule of thumb:

- **`run_campaign.py`, `campaign_plots.py`** → from **`$REPO`**.
- **`operator_skill.py`, `aqi_analysis.py`, `experiment_pipeline.py`, `eval_ruler.py`** → from **`$REPO/examples`**.
- Training writes JSONL logs to **`$REPO/examples/aqi_logs/<run>/`** (the actor's cwd).

### Where a campaign saves everything

```
runs/e1_cylinder/
├── campaign.json            # frozen roster + grid + settings
├── eval_grid.json           # the ruler (SHA-256 locked)
├── operators.csv            # roster (+ auto skill types)
├── operators/<id>.json      # each operator's θ
├── demos/<op>_s<seed>/      # demos.pkl + groundtruth + README + manifest
├── checkpoints/<cell>/      # trained policy + CONVERGED.json + README + manifest
├── eval_runs/<cell>/        # trials.jsonl + cell.json (ruler SR)  + README
├── logs/                    # per-cell learner/actor stdout
└── report/
    ├── dependency.md        # ICC / σ²between-within / CV / permutation-p / RG / steps
    ├── dependency.json      # machine-readable
    └── plots/*.png          # 6 figures (see §E1)
```
`<cell>` = `<method>_<operator>_s<seed>`.

---

## E0 — Validate the synthetic operators (prerequisite)

```bash
cd "$REPO/examples"
$PY operator_skill.py --mode validate-synthetic \
    --val_ops_per_tier 10 \
    --out_dir "$REPO/runs/e0_validation"
```

**Saved:** `runs/e0_validation/skill_report.md` + `skill.json`
(accuracy / macro-F1 / QWK / Spearman-ρ / episodes-needed / test-retest ICC).
**Gate:** these must clear the S8/S11 targets before trusting E1. Put the table in the thesis.

---

## E1 — Main operator-dependency campaign (synthetic) → RQ1

### 1. The config (the 5-mode ladder)

The ready-to-run config is committed at **`examples/campaigns/e1_cylinder.yaml`** — edit
it in place to change operators/seeds/task. Its key fields (for reference):

```yaml
task: { exp_name: sim_peg_insertion, peg_shape: cylinder }
operators: { expert: 5, intermediate: 5, novice: 5, jitter: 0.15, sample_seed: 0 }
demos: { per_operator: 6, filtered: true, max_attempts: 300 }
methods: [hilserl, logged, active, quality, aqi]     # full ablation ladder
replicates: { seeds: 3, seed_base: 0 }
budget: { converge_success: 0.9, steps_cap: 20000, minutes_per_cell: 60, learner_max_steps: 1000000 }
eval: { dx_mm: [-16,-12,-8,-4,0,4,8,12,16], yaw: [0.0], reps: 1, max_steps: 300 }
gpu: { jax_platform: cuda, mem_learner: 0.6, mem_actor: 0.2 }
logging: { wandb: "offline" }
analysis: { baseline: hilserl, plots: true }
```

### 2. Preview (no training), then run

```bash
cd "$REPO"
# Dry preview: roster, grid hash, cell count, budgets
$PY examples/run_campaign.py --config examples/campaigns/e1_cylinder.yaml \
    --campaign runs/e1_cylinder --plan

# Smoke-test the loop first (2 methods, tiny) — optional but recommended
$PY examples/run_campaign.py --config examples/campaigns/e1_cylinder.yaml \
    --campaign runs/e1_smoke --methods hilserl,aqi --ops_per_tier 2 --seeds 2

# Full run in the background, logged to a file (resumable: re-run the same line)
nohup $PY examples/run_campaign.py --config examples/campaigns/e1_cylinder.yaml \
    --campaign runs/e1_cylinder > runs/e1_cylinder.out 2>&1 &
tail -f runs/e1_cylinder.out          # watch progress
```

Stages run in order `sample → demos → train → eval → analyze` and **skip work already
done**, so an interrupted run resumes cleanly.

### 3. Results & re-analysis / plots

The `analyze` stage writes the report automatically. To (re)generate just the report/plots:

```bash
cd "$REPO"
$PY examples/run_campaign.py --config examples/campaigns/e1_cylinder.yaml \
    --campaign runs/e1_cylinder --stages analyze
$PY examples/campaign_plots.py --campaign runs/e1_cylinder
```

**Saved:** `runs/e1_cylinder/report/dependency.{md,json}` and `report/plots/`:
`icc_by_method.png`, `sr_by_operator.png`, `sr_by_tier.png`,
`variance_decomposition.png`, `operator_spread.png`, `steps_to_convergence.png`.
→ **RQ1** (ICC, $\sigma^2_{\text{between/within}}$, CV, permutation-$p$, $RG_{\text{rel}}$, steps-ICC).

### 4. Offline RQ2 / RQ3 from the same run's logs

```bash
cd "$REPO/examples"
$PY aqi_analysis.py \
    --log_root aqi_logs --baseline_mode logged --fail_horizon 20 \
    --out "$REPO/runs/e1_cylinder/rq_analysis"
```

**Saved:** `runs/e1_cylinder/rq_analysis/`: `summary.md`, `results.json`,
`learning_curves.*`, `operator_variance.*` (RQ1), `q_int_by_operator.*` (**RQ2**),
`p_fail_calibration.*` (**RQ3**). Run this right after E1 (before other runs add to
`aqi_logs/`), or pass explicit `--run_dirs aqi_logs/sim_peg_insertion_*` to scope it.

---

## E2 — Ablations → RQ2 / RQ3 mechanisms

The **ladder in E1 already decomposes AQI** (`logged`→`active`→`quality`→`aqi`). Add
finer arms as **separate campaigns** (own `--campaign` dir ⇒ own report), reusing the
same operators/seeds so only the knob changes.

### RQ3 — trigger-term ablation (uncertainty-only, OR-logic)

```bash
cd "$REPO"
$PY examples/run_campaign.py --config examples/campaigns/e1_cylinder.yaml \
    --campaign runs/e2_trigger_uncert \
    --methods active --trigger_terms uncertainty --trigger_combiner max
```

### RQ2 — quality-composition ablation (drop the disagreement gate / filters)

```bash
cd "$REPO"
$PY examples/run_campaign.py --config examples/campaigns/e1_cylinder.yaml \
    --campaign runs/e2_quality_nogate \
    --methods quality --no-quality_disagreement_gate --quality_filters outcome_filter
```

Then compare reports across `runs/e1_cylinder` vs `runs/e2_*` (same ruler hash).
For each ablation campaign, run the offline `aqi_analysis.py` block (E1 §4) pointed at
its logs to get the RQ2/RQ3 curves.

> **Advanced arms** (not campaign-exposed): the ensemble-diversity control for RQ3
> (`--bootstrap_p 1.0 --noper_member_targets` on `train_aqi_rlpd.py`) runs as a manual
> learner+actor pair (see the sim tasks / `run.sh`). The usage split — replay-only
> (`bc_loss_weight=0`) vs BC-only (`priority_sampling=false`) — is set in the task's
> `get_aqi_config()` (AQIConfig fields, not CLI flags), then trained the same way. Analyse
> the resulting `aqi_logs/` runs with `aqi_analysis.py`.

---

## E3 — Human confirmation study (interactive) → RQ1 + human RQ2/RQ3

### 1. Freeze the protocol (once)

```bash
cd "$REPO/examples"
$PY experiment_pipeline.py init \
    --campaign "$REPO/runs/e3_human" --exp_name ram_insertion \
    --operators O1,O2,O3,O4 --methods logged,aqi --seeds 0 \
    --delta 0.05 --budget "45 min wall-clock" --rig_epoch rig-v1
```

**Saved:** `runs/e3_human/`: `campaign.json`, `eval_grid.json` (+SHA-256),
`counterbalancing.json`, `PROTOCOL_FREEZE.md`, `operators.csv`. **Do not edit after this.**

### 2. One session per operator (interactive)

```bash
cd "$REPO/examples"
$PY experiment_pipeline.py session \
    --campaign "$REPO/runs/e3_human" --operator O1 \
    --demo_path ./experiments/ram_insertion/demo_data/ram_20_demos.pkl
```

It prints the exact **training** command and the **eval-ruler** command for that
operator's counterbalanced cell, collects the exit form, and verifies artifacts. Run the
printed training command, then the printed ruler command, e.g.:

```bash
cd "$REPO/examples"
$PY eval_ruler.py --exp_name=ram_insertion \
    --checkpoint_path="$REPO/runs/e3_human/checkpoints/<cell>" \
    --grid_file="$REPO/runs/e3_human/eval_grid.json" \
    --out="$REPO/runs/e3_human/eval_runs/<cell>"
```

Repeat `session` for O2, O3, O4 (order is auto-counterbalanced).

### 3. Evaluate + skill-type

```bash
cd "$REPO/examples"
$PY experiment_pipeline.py evaluate \
    --campaign "$REPO/runs/e3_human" --out "$REPO/runs/e3_human/report"
$PY experiment_pipeline.py assess --campaign "$REPO/runs/e3_human"
```

**Saved:** `runs/e3_human/report/`: `report.md`, `results.json`, `money_plot.png`
(**RQ1**), `interaction_plot.png` (**RQ1**), `learning_curves.png`. `assess` writes
operator skill types into `operators.csv`.

---

## Optional — hex/cuboid headroom (policy oracle)

The geometric oracle is translation-only (cylinder). For yaw shapes, bootstrap a **policy
oracle** from a converged cylinder baseline, then point demo collection / the actor at it:

```bash
# 1) after a cylinder baseline exists at run_cylinder_hilserl/checkpoint_XXXX/
cd "$REPO/examples"
$PY collect_policy_synthetic_demos.py \
    --exp_name sim_peg_insertion --checkpoint_path "$REPO/run_cylinder_hilserl/checkpoint_XXXX" \
    --profile expert --episodes 6 --seed 0 --out "$REPO/runs/e1_hex/demos/hex_expert.pkl"
# 2) run an E1-style campaign with task.peg_shape: hex and SIM_ORACLE_CHECKPOINT set
```

See [oracle_and_synthetic_operator.md](../oracle_and_synthetic_operator.md) §7.1.

---

## RQ → artifact map (what to cite in the thesis)

| RQ | file(s) | figure |
|---|---|---|
| RQ1 | `e1_*/report/dependency.md`; `e3_human/report/report.md` | `icc_by_method.png`, `variance_decomposition.png`, `money_plot.png`, `interaction_plot.png`, `steps_to_convergence.png` |
| RQ2 | `*/rq_analysis/summary.md` | `q_int_by_operator.png` |
| RQ3 | `*/rq_analysis/summary.md` | `p_fail_calibration.png` |
| validation | `e0_validation/skill_report.md` | — |

---

## Reproducibility & gotchas

- **Frozen ruler:** the grid SHA-256 is pinned in `campaign.json`; every cell (and every
  resume) is scored on the byte-identical ruler. `eval_ruler.py` refuses a mismatched grid.
- **Seeds:** each replicate drives `--seed = --synthetic_seed = --env_seed = r` → genuinely
  independent runs; the roster is reproducible from `sample_seed`.
- **Provenance:** every `demos/`, `checkpoints/`, `eval_runs/` folder carries a
  `manifest.json` + `README.md` stating the task, conditions, and which options were **inert**.
- **W&B:** `logging.wandb: "offline"` (or `off`) — the learner opens W&B; an unset key with
  `online` crashes background runs. Offline is safe and needs no key.
- **GPU sharing (8 GB):** `nvidia-smi` first; the campaign aborts if others use
  > `--gpu_free_mb` (override with `--force`). If an `active`/`quality`/`aqi` actor OOMs on
  reset, set `gpu.jax_platform: cpu` (actor inference on CPU is fine).
- **Convergence / stop:** a cell writes `checkpoints/<cell>/CONVERGED.json` when it hits the
  rolling autonomous-success criterion; a `STOP` sentinel stops a learner gracefully.
- **Before trusting any AQI delta:** open `report/dependency.md` and confirm the **vanilla
  baseline shows a real skill gradient** (expert ≫ novice). If it's at ceiling/floor, harden
  the task (hex/cuboid, fewer demos) — otherwise RQ1 is uncomputable.
