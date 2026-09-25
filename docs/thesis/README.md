# Thesis hub — start here

Everything you need to **run the study, analyse it, and write it up**, in one place.
If you feel lost, read this page top to bottom; it links to the exact doc, command,
and output for each piece.

## The two core files

| file | what it gives you |
|---|---|
| **[experiment_plan.md](experiment_plan.md)** | *what* to run and *why*: RQ1–RQ3 → endpoints → the four experiments (E0–E3) → decision rules → deliverables |
| **[experiment_protocols.md](experiment_protocols.md)** | *how* to run it: copy-paste commands, where results are saved, how to analyse and plot |

Pseudocode for the algorithms chapter: **[../../tex/aqi_hil_serl_algorithms.tex](../../tex/aqi_hil_serl_algorithms.tex)**
(compiles to a 5-page PDF; Alg. 1–6 + notation + ablation tables).

> **Engineering log / failure analysis:** **[rq1_campaign_engineering_log.md](rq1_campaign_engineering_log.md)** — the week-by-week record of every attempt to make the RQ1 campaign yield a valid signal (premise → result → why it failed), with a decision-tree diagram, a summary table, and the root-cause pivot to the policy oracle. Use it for the methodology / lessons-learned chapter or appendix.

## Start here (the shortest path to a result)

```bash
# 0. one-time shell setup (see protocols for details)
REPO=/home/user/Downloads/hil-serl_aqi; PY="$REPO/.venv/bin/python"; cd "$REPO"

# 1. validate the synthetic operators (gate)
cd "$REPO/examples" && $PY operator_skill.py --mode validate-synthetic --out_dir "$REPO/runs/e0_validation"

# 2. smoke-test the campaign loop, then run E1
cd "$REPO" && $PY examples/run_campaign.py --config examples/campaigns/e1_cylinder.yaml \
    --campaign runs/e1_smoke --methods hilserl,aqi --ops_per_tier 2 --seeds 2
nohup $PY examples/run_campaign.py --config examples/campaigns/e1_cylinder.yaml \
    --campaign runs/e1_cylinder > runs/e1_cylinder.out 2>&1 &

# 3. the report + plots land in runs/e1_cylinder/report/ automatically
```

Full step-by-step (E0–E3, ablations, human study, plotting) → **[experiment_protocols.md](experiment_protocols.md)**.

## Map: thesis chapter → repo doc → artifact

| thesis chapter | read | produces |
|---|---|---|
| Introduction / problem | [../methods.md](../methods.md) §0, [../getting_started.md](../getting_started.md) | — |
| Background (RL, HIL-SERL, ensemble) | [../aqi_walkthrough.md](../aqi_walkthrough.md), [../critic_ensemble.md](../critic_ensemble.md) | — |
| Method: AQI (trigger + quality) | [../aqi_walkthrough.md](../aqi_walkthrough.md), [../critic_ensemble.md](../critic_ensemble.md), [tex](../../tex/aqi_hil_serl_algorithms.tex) | Alg. 1–4 |
| Method: synthetic operators | [../oracle_and_synthetic_operator.md](../oracle_and_synthetic_operator.md), [../operator_skill_assessment.md](../operator_skill_assessment.md) | Alg. 5, E0 table |
| Experimental design | [experiment_plan.md](experiment_plan.md), [../experimental_protocol.md](../experimental_protocol.md), [../dependency_analysis.md](../dependency_analysis.md), [../eval_ruler.md](../eval_ruler.md) | Alg. 6 |
| Implementation / reproducibility | [../architecture.md](../architecture.md), [../campaign_runner.md](../campaign_runner.md), [experiment_protocols.md](experiment_protocols.md) | — |
| Results & analysis | [../dependency_analysis.md](../dependency_analysis.md), [../aqi_analysis.md](../aqi_analysis.md), [experiment_protocols.md](experiment_protocols.md#rq--artifact-map-what-to-cite-in-the-thesis) | all figures |
| Methodology / lessons learned | [rq1_campaign_engineering_log.md](rq1_campaign_engineering_log.md) | RQ1 attempt log + decision-tree diagram |

## Map: RQ → metric → script → figure to cite

| RQ | metric | script | figure |
|---|---|---|---|
| **RQ1** dependency | ICC, $RG_{\text{rel}}$, permutation-$p$, steps-ICC | `run_campaign.py analyze` / `experiment_pipeline.py evaluate` | `icc_by_method.png`, `variance_decomposition.png`, `money_plot.png` |
| **RQ2** usage | AUROC, corr$(q_{\text{int}},\text{outcome})$ | `aqi_analysis.py` | `q_int_by_operator.png` |
| **RQ3** trigger | AUROC + ECE of $A(s_t)$/$P_{\text{fail}}$ | `aqi_analysis.py` | `p_fail_calibration.png` |
| validation | accuracy/QWK/ρ/ICC | `operator_skill.py --mode validate-synthetic` | E0 table |
| quality-signal validity (RQ4 §11–12, RQ5) | AUROC all / clean / time-stratified / post-release, takeover-cap shares | `quality_signal_audit.py` ([doc](../quality_signal_audit.md)) | `report/quality_audit/summary.md`, `<arm>_auroc.png` |

## Where results live

All study outputs go under `runs/<name>/` at the repo root. A campaign folder holds
`report/dependency.{md,json}`, `report/plots/*.png`, per-cell `eval_runs/<cell>/cell.json`
and `checkpoints/<cell>/`, `demos/`, and `operators.csv` — every folder carries a
`README.md` + `manifest.json` describing its conditions. Layout details in
[experiment_protocols.md](experiment_protocols.md#where-a-campaign-saves-everything).
