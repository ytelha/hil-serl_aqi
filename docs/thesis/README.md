# Thesis hub — start here

Everything you need to **run the study, analyse it, and write it up**, in one place.
If you feel lost, read this page top to bottom; it links to the exact doc, command,
and output for each piece.

## The training machines — start here

| where it ran | read first | details |
|---|---|---|
| **first training machine** (4 Sep – 31 Oct 2026, 459 cells, original platform: RQ1–RQ4) | **[first_machine_synthesis.md](first_machine_synthesis.md)** | the per-campaign logs listed in its header |
| **second training machine** (24–29 Oct 2026, 58 cells, new stack: rotation control, inconsistent operator) | **[second_machine_synthesis.md](second_machine_synthesis.md)** | [new_stack_operator_results_and_qa_log.md](new_stack_operator_results_and_qa_log.md), [inconsistent_operator_results_and_qa_log.md](inconsistent_operator_results_and_qa_log.md) |
| **RWTH cluster** (method test 26 Sep; counterfactual study C9, 29 Sep; small E2HiL campaign queued 3 Nov) | [../counterfactual_score_study.md](../counterfactual_score_study.md) §5, [../e2hil_inconsistent_operator.md](../e2hil_inconsistent_operator.md) | `cluster/claix/README.md`, `handoff/FINDINGS_cluster_method_test.md` |
| presenting both machines | **[presenting_the_training_machines.md](presenting_the_training_machines.md)** | slide map, key numbers, likely questions |

Never pool cells of different machines: learner updates per environment step differ (about 0.45 on the first,
1 on the second, more on the cluster).

## The two core files

| file | what it gives you |
|---|---|
| **[experiment_plan.md](experiment_plan.md)** | *what* to run and *why*: RQ1–RQ3 → endpoints → the four experiments (E0–E3) → decision rules → deliverables |
| **[experiment_protocols.md](experiment_protocols.md)** | *how* to run it: copy-paste commands, where results are saved, how to analyse and plot |

Pseudocode for the algorithms chapter: **[../../tex/aqi_hil_serl_algorithms.tex](../../tex/aqi_hil_serl_algorithms.tex)**
(compiles to a 5-page PDF; Alg. 1–6 + notation + ablation tables).

> **Engineering log / failure analysis:** **[rq1_campaign_engineering_log.md](rq1_campaign_engineering_log.md)** — the week-by-week record of every attempt to make the RQ1 campaign yield a valid signal (premise → result → why it failed), with a decision-tree diagram, a summary table, and the root-cause pivot to the policy oracle. Use it for the methodology / lessons-learned chapter or appendix.

> **Ensemble tuning & quality-score findings (RQ2–RQ5) — read in this order:**
> These logs contain the *full reasoning* behind every ensemble/quality design
> decision (the thesis-critical negative and constructive results):
> 1. **[rq2_quality_replay_and_qa_log.md](rq2_quality_replay_and_qa_log.md)** — the original quality-replay study + the "negligible score" critique (§7/§10/§16/§18).
> 2. **[rq3_structural_ablation_and_qa_log.md](rq3_structural_ablation_and_qa_log.md)** — the default 10-critic ensemble **hurts** (SR −18.5 pp, ICC 0.03→0.74); quality replay hurts most.
> 3. **[rq4_ensemble_convergence_and_qa_log.md](rq4_ensemble_convergence_and_qa_log.md)** — the **structural fixes** that recover parity (§2, §7–§10), the quality-signal **diagnostic** (§11), and the **behavioral-score redesign** (§12).
> The distilled, reference version of both topics lives in **[../critic_ensemble.md](../critic_ensemble.md)** — §7 (structural tuning) and §4.2 (quality-score problems). Audit tooling: **[../quality_signal_audit.md](../quality_signal_audit.md)**.

> **AQI active + quality results — START HERE for the Oct-2026 arc:**
> **[active_quality_campaigns_synthesis.md](active_quality_campaigns_synthesis.md)** —
> the master synthesis of every campaign (A active trigger, C quality-gate oracle
> ceiling, demo probe), written on the first machine: **timing helps (use
> `active`)**; its reading that quality filtering hurts operator-independence even at
> the oracle ceiling is **not established**, because the quality arms also ran the hard
> `outcome_filter` and a BC loss (note of 2026-10-31; `../novice_ablation.md` decides).
> Links to the two detailed per-campaign logs below.

> **The second training machine — START HERE for the new stack and the inconsistent operator:**
> **[second_machine_synthesis.md](second_machine_synthesis.md)** — what ran (58 cells on the new stack, 57 of them on the
> cuboid), how the run went gate by gate, what came out, and the plan and campaigns it leaves for the first machine. Details in
> **[new_stack_operator_results_and_qa_log.md](new_stack_operator_results_and_qa_log.md)** (rotation control: cuboid
> rescue 7–13 % → 70–100 %; orientation tolerance; training speed) and
> **[inconsistent_operator_results_and_qa_log.md](inconsistent_operator_results_and_qa_log.md)** (scores against a
> hidden attentive/inattentive label; discard vs re-weight: no difference shown; why the test was weak).

> **Active trigger — the rate-matched frontier (the active half of the thesis):**
> **[active_ratematch_results_and_qa_log.md](active_ratematch_results_and_qa_log.md)** —
> campaign A (108 cells): does σ_Q-timed help beat the engineered monitor at a
> *matched* intervention budget? Headline: `active-q55` dominates the SR-vs-cost
> frontier (+13.2 pp vs `logged`, p=0.021, lifts weak tiers, ICC≈0); honest
> near-nulls on "timing beats chance" (p=0.30) and "beats the no-trigger baseline".
> Includes the quantile-trigger mechanism, the calibration probe, both baselines,
> and the frontier plot.

> **Quality gate — the oracle ceiling (the quality half of the thesis):**
> **[aqi_oracle_quality_results_and_qa_log.md](aqi_oracle_quality_results_and_qa_log.md)** —
> campaign C (90 cells): does a PERFECT simulator-oracle quality score reduce
> operator-dependence on top of σ_Q timing? Its arms raised the ICC (0.18 → 0.93) and
> collapsed the novice tier (65 % → 9–36 %). The log calls this a *paradigm* problem; that
> does **not** follow yet: the quality arms changed three things at once (the oracle score,
> the hard `outcome_filter`, a BC loss of 0.1). The novice test (`../novice_ablation.md`,
> 24 cells, not run) separates them.

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
