# Experimental protocol - operator study pipeline

Implements the thesis document *"Experimental Design - Measuring Operator
Dependency and Comparing AQI-HIL-SERL to Vanilla"* (Parts A-G). Section
references below (A.2, B.3, D.2, E.1, ...) point into that document; its
worked numerical examples are reproduced verbatim by this code's test
suite, so the implementation and the pre-registration agree by
construction.

Three components:

| Component | File | Document part |
|---|---|---|
| Statistics library | `serl_launcher/serl_launcher/aqi/stats.py` | B.1-B.4, C.2, D.2, C.3/D.3 |
| The frozen evaluation ruler | `examples/eval_ruler.py` | A.2 |
| Campaign pipeline (init / session / evaluate / simulate) | `examples/experiment_pipeline.py` | A, C-F |

## The campaign workflow (the run-book, Part E)

```bash
cd examples

# E.1 - freeze the protocol ONCE, before the first session
python experiment_pipeline.py init \
    --campaign ~/campaigns/ram_2026 --exp_name ram_insertion \
    --operators O1,O2,O3,O4 --methods logged,aqi --seeds 0 \
    --budget "45 min wall-clock" --delta 0.05 --rig_epoch rig-v1 \
    --protocol_file Experimental_Design.docx
# writes: campaign.json, eval_grid.json (+SHA-256), counterbalancing.json,
# PROTOCOL_FREEZE.md (date-stamped pre-registration incl. the decision
# rules), operators.csv

# E.2 - one command per operator session; it walks you through the
# verbatim briefing, prints the exact training + ruler commands for the
# counterbalanced cell, collects the 3-question exit form and anomalies,
# verifies the artifacts, and commits the session sheet
python experiment_pipeline.py session \
    --campaign ~/campaigns/ram_2026 --operator O1 \
    --demo_path ./experiments/ram_insertion/demo_data/ram_20_demos.pkl

# A.2 - the ruler (the session command prints this line pre-filled)
python eval_ruler.py --exp_name=ram_insertion \
    --checkpoint_path=<campaign>/checkpoints/<cell> \
    --grid_file=<campaign>/eval_grid.json \
    --out=<campaign>/eval_runs/<cell>

# B-D + F - after any number of sessions
python experiment_pipeline.py evaluate --campaign ~/campaigns/ram_2026
# -> report/report.md, results.json, money_plot / interaction_plot /
#    learning_curves (.png + .svg)

# dry-run everything without a robot (also: --preset doc_example
# reproduces the design document's worked numbers end to end)
python experiment_pipeline.py simulate --campaign <fresh> --preset realistic
```

For the simulation grid (Experiment 2, 4 methods x 4 operators x >=5
seeds) use `--methods logged,active,quality,aqi --seeds 0,1,2,3,4`; the
counterbalancing table becomes a balanced Latin square automatically.

## Metric -> code map

| Document metric | Function in `aqi/stats.py` |
|---|---|
| SR + 95% Wilson interval (B.1) | `cell_outcomes`, `wilson_interval` |
| T-bar, F-bar_peak, violation rate (B.1) | `cell_outcomes` |
| S80 (B.2) | `s80` (rolling window, default 20; `math.inf` if never) |
| Intervention count / time (B.2) | `intervention_summary` (episode-level events AND distinct takeovers) |
| r(t) intervention-rate curve, final autonomy (B.2) | `intervention_rate_curve` |
| Learning-curve AUC (B.2) | `learning_curve_auc` |
| sigma_op, CV, range, worst case (B.3) | `operator_spread` |
| ICC one-way random effects (B.3) | `icc_oneway` (+ `bootstrap_icc`, hierarchical over operators and seeds) |
| Permutation test "operator matters" (C.2) | `permutation_sigma2` |
| RG / RG_rel (B.4-1, primary) | `robustness_gain`, `paired_bootstrap_rg_rel` (operators are the resampling unit) |
| Non-inferiority guardrail dbar vs delta (B.4-2) | `mean_diff_guardrail` |
| Worst-operator gain (B.4-3) | `worst_operator_gain` |
| Intervention efficiency to 80% (B.4-4) | `steps_to_threshold` with interventions / human-seconds as cost |
| Interaction model mu + alpha + b + gamma (B.4-5) | `interaction_decomposition` (exact algebra; reconstruction certified) |
| Trial-level Binomial GLMM (B.4-5 / D.2) | `glmm_success` (optional: `pip install statsmodels pandas`; the exact decomposition is the assumption-light substitute) |
| Brown-Forsythe, paired Wilcoxon + effect sizes (D.2) | `brown_forsythe`, `wilcoxon_paired` |
| Pre-registered decision rules (C.3 / D.3) | `exp1_decision`, `exp2_decision`; rendered as the report's decision table |

## Practical notes

**The vanilla arm.** `logged` is the recommended vanilla: HIL-SERL
training with the gate suppressed and the quality module off, plus the
instrumentation that the process metrics (S80, r(t), AUC, intervention
counts) require. A pure `hilserl` arm is supported - its ruler-based
outcome metrics work normally - but it writes no training logs by
design, so its process metrics are reported as unavailable. Whichever
you choose is recorded in `PROTOCOL_FREEZE.md` at init time; do not
change it afterwards.

**Start-pose control for the ruler.** The grid offsets are applied, in
preference order: (1) via a `set_eval_reset_pose(dx_m, dyaw_rad)` method
if any wrapper exposes one (implement this on simulation envs);
(2) on FrankaEnv-style bases automatically, through the same
`resetpos` / `euler_2_quat` mechanics the env's own randomization uses,
restored afterwards; (3) otherwise the env's own reset runs, the
intended pose is still logged, and every trial carries
`pose_control="env_random"` so the limitation is visible in the data.

**One ruler per campaign.** `eval_ruler.py` refuses a grid file whose
content does not match its recorded SHA-256, and the campaign manifest
pins that hash - every cell is measured with the identical ruler, and
the familiarisation poses of E.2 must stay disjoint from it.

**Contaminated trials.** If a human intervenes during the frozen
evaluation the trial is flagged (`intervened=true`) and excluded from
every statistic; the count of exclusions is reported per cell.

**Logging schema (E.3).** Training already logs per step
(t, proprio state, sigma_Q, gate score, `a_robot`, `a_executed`, `a_h`,
intervention flag), per intervention segment (start/length, requested,
q_int / A-hat / sigma at trigger, force stats, exclusion decision,
episode outcome), per episode, and per run (`meta.json` with the full
composition); the session sheet adds operator, method, order index, rig
epoch, budget used, anomalies, exit form, hyperparameter hash and
eval-grid hash.

**Ensemble bootstrap ablation.** Trigger quality (RQ3 / D.3) depends on
sigma_Q being a real epistemic signal, which the bootstrapped ensemble
training provides (persistent per-transition member masks + per-member
targets). Running an arm with `--bootstrap_p=1.0 --per_member_targets=false`
gives the shared-data control for exactly that claim.

**Statistics at small K.** With 4-6 operators the hierarchical bootstrap
CI for ICC is wide (its lower tail is dominated by low-diversity
resamples) - that is honest, and it is why the pre-registered Exp-1 rule
tests the ICC point estimate together with the permutation p, not the CI
bound. Likewise the paired Wilcoxon's smallest attainable two-sided p at
n=4 is 0.125; per D.2 it is secondary and effect sizes carry the weight.

## Operator instruments (synthetic operators, skill typing)

Experiment-1 synthetic operators (op1-op4 incl. the skilled-but-jumpy off-diagonal profile) and the automatic operator typing that annotates `operators.csv` are specified in `synthetic_operators_and_skill.md`; the generator's S8 calibration check is the Milestone-1.2 exit criterion stated as a generator property, and `experiment_pipeline.py assess` runs the Part-B typing over a campaign's aqi_logs.

---

## Script CLI reference — `experiment_pipeline.py`

Five subcommands, each maps to one stage of the run-book:

| Subcommand | What it does | When to run |
|---|---|---|
| `init` | Freeze the protocol: campaign manifest, eval grid (+ SHA-256), counterbalancing table, `PROTOCOL_FREEZE.md` | Once, before the first session |
| `session` | Walk through one operator session: verbatim briefing, training + ruler commands, exit form, artifact verification, session sheet | Once per cell |
| `evaluate` | Ingest all cells, compute full metric catalogue (SR, S80, r(t), ICC, RG_rel, bootstrap CIs, GLMM, decision tables), generate figures | After all sessions are done |
| `assess` | Operator skill typing from `aqi_logs/` (Part B) | Optionally after sessions |
| `simulate` | Fill a campaign with synthetic cells for a dry run | Testing / smoke test |

```bash
cd examples

# freeze the protocol
python experiment_pipeline.py init \
    --campaign ~/campaigns/ram_2026 --exp_name ram_insertion \
    --operators O1,O2,O3,O4 --methods logged,aqi --seeds 0 \
    --budget "45 min wall-clock" --delta 0.05

# one operator session (interactive)
python experiment_pipeline.py session \
    --campaign ~/campaigns/ram_2026 --operator O1

# evaluate all completed cells
python experiment_pipeline.py evaluate \
    --campaign ~/campaigns/ram_2026 --out ~/campaigns/ram_2026/report

# skill-type all operators
python experiment_pipeline.py assess \
    --campaign ~/campaigns/ram_2026

# dry run with synthetic data
python experiment_pipeline.py simulate \
    --campaign ~/campaigns/ram_2026 --preset doc_example
python experiment_pipeline.py evaluate \
    --campaign ~/campaigns/ram_2026
```

### Key flags for `evaluate`

| Flag | Default | Meaning |
|---|---|---|
| `--vanilla` | first method | the arm treated as the baseline |
| `--B` | 10 000 | bootstrap resamples |
| `--window` | 20 | rolling window for S80 / threshold metrics |
| `--stat_seed` | 0 | RNG seed for bootstrap / permutation |

### Outputs of `evaluate`

| File | What it contains |
|---|---|
| `report.md` | full metric catalogue, Exp-1 ICC table, Exp-2 RG_rel + guardrail, decision table |
| `results.json` | machine-readable version of the same |
| `money_plot.png` | paired per-operator SR arrows (vanilla → treatment) with Wilson whiskers |
| `interaction_plot.png` | one line per operator across methods (converging = robustness) |
| `learning_curves.png` | rolling SR + intervention rate r(t) per method |

### What `experiment_pipeline.py` does NOT do

- It does **not** launch training subprocesses — it prints the exact commands and waits for the researcher to run them.
- It does **not** call `aqi_analysis.py` — its `evaluate` subcommand has its own metric implementation using the eval ruler SR (not the training-log SR).
- For automated synthetic-operator campaigns use `run_campaign.py` instead.
