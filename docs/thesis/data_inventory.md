# Data inventory — every dataset, where it lives, what it supports

*Created 2026-11-09.* Use this file when writing the thesis: every number in the thesis and the decks should trace to
a row here.

**The data home is `results/` at the repo root.**
- It is **git-ignored**: the GitHub repo is public, and the data stays private.
- **So back it up yourself** (OneDrive or an external disk). It is about 4.3 GB (14 Nov).
- Originals as received are in `results/_received/`, with SHA-256 checksums in
  `results/_received/SHA256SUMS.txt`.

```
results/
  _received/                      originals exactly as received (+ SHA256SUMS.txt)
  first_machine/                  campaigns/ (raw runs of the valid campaigns), campaign reports, W&B export, the ablation
  second_machine/
    2026-10_new_stack_and_careless_operator/   runs (slim), agent reports, curated summaries
    2026-11_comprehensive_study/  the full export of the comprehensive study
  cluster/                        RWTH CLAIX campaigns
  analysis/                       the scripts behind the deck and the reports, with their outputs
```

"Slim" means without network weights (`checkpoint_*`, `certified`), demonstrations, W&B folders and per-step logs.
Everything an analysis reads is kept:
- reports;
- `cells.csv`;
- ruler results (`eval_runs/*/cell.json`, `trials.jsonl`);
- training records (`episodes.jsonl`, `convergence_evals.jsonl`);
- takeover logs (`aqi_logs/*/segments.jsonl`);
- logs and configs.

---

## 1. First training machine

Old stack: asynchronous learner, about 0.45 updates per step, finding F3.

**Raw run folders since 14 Nov:** `results/first_machine/campaigns/` holds the 15 valid campaigns from
`first_machine_runs_slim_2026-11-14.tgz`. Its `README.md` lists each campaign's caveats and the 18 campaigns left out,
with the reasons (cylinder-ruler bug, aborted, smoke tests, confounded). The tgz itself (857 MB, SHA-256
`c07a7188…35e7a4`) is **not copied**: it stays in `C:\Users\youss\OneDrive\`. Keep it; it is the only complete copy.

| dataset | stored in | contents | supports |
|---|---|---|---|
| **Raw runs of every valid campaign** (A1–A4, B1, B2, C4, C5, D1–D3, C11, demonstration probe, novice test, ladder) | `results/first_machine/campaigns/<campaign>/` | `campaign.json`, `checkpoints/` (episodes), `eval_runs/` (ruler), `aqi_logs/` (takeovers), `logs/`, `report/` | everything below; the bundled reports are byte-identical to these runs' `report/` |
| **Novice test (36 runs, 4–5 Nov)** | `…/campaigns/quality_novice_ablation/` | report `report/novice_ablation.md` | the imitation loss collapses the novice, not the score: `docs/thesis/first_machine_novice_test_and_ladder.md` |
| **Ladder (27 runs, 5–6 Nov) and stage 0** | `…/campaigns/synthetic_suite/` | `report.md`, `report_cells.csv`, `stage0/` | rotation control and one update per step remove the slow start: same doc |
| Teacher dependency, geometric and policy oracle | `results/first_machine/campaign_reports/rq1_geometric/`, `rq1_policy_oracle/` | dependency reports, plots | teacher dependency (ICC 0.60 / 0.99); deck page 19 |
| Structural ablation: HIL-SERL, first ensemble, critics' score in training | `…/rq3_structural/` | reports, quality audit | deck pages 20, 24 |
| Ensemble convergence: fixed ensemble vs HIL-SERL | `…/rq4_convergence/` | reports | deck pages 20–21 |
| Quality diagnostics | `…/quality_on_rq4_diag/`, `…/quality_behavioral/` | reports, audits | thesis quality chapter (early audits) |
| Asking: validity and first experiment | `…/active_validity/`, `…/active_effect/` | reports, plots, audits | deck page 22 |
| Asking rate (108 runs) | `…/active_ratematch/` | reports (two baselines), plots, quality audit with `segments.csv` | deck page 23, backup 39, the counterfactual AUROC 0.15 (page 24) |
| Index of the campaigns above | `…/campaign_reports/RESULTS_INDEX_from_presentation_bundle.md` | configs, headline numbers | — |
| W&B export (campaigns B1 structural ablation, B2 ensemble convergence, C3, D1 active validity, D2 first asking experiment) | `results/first_machine/wandb_export/` (identical copy of `wandb_analysis/`, which the figure scripts read) | `analysis/episodes.csv.gz`, `learner.csv.gz`, `runs.csv`, raw exports, scripts | page 21 learning curves, backup 40 (same setup in three campaigns), backup 43 (first success) |
| **Ensemble-fix ablation (96 runs, Nov 2026)** | `results/first_machine/2026-11_ensemble_fix_ablation/`: the whole run folder, slim; originals in `results/_received/`: `ensemble_fix_ablation_slim_2026-11-13.tgz` and the report received earlier as `C:\Users\youss\OneDrive\report\` | `campaign.json`, `checkpoints/` (episodes, convergence evaluations), `eval_runs/` (ruler per checkpoint), `aqi_logs/`, `logs/`, `report/` (dependency report, plots, score audit) | `docs/thesis/ensemble_fix_ablation_results.md`; scripts `ablation_analysis.py` (report) and `ablation_full_analysis.py` (run folder) |
| Code snapshot of the first machine, 31 Oct | `results/_received/first_machine_code.tgz` | its source tree | what that machine ran (patches 0001–0010, 0014–0016, local features) |
| Quality oracle (90 runs) | `results/first_machine/campaigns/aqi_oracle_quality/` (since 14 Nov) | raw runs and report; numbers in `docs/thesis/aqi_oracle_quality_results_and_qa_log.md` | deck page 25, backups 51–52. Checked against the raw runs on 14 Nov: the report matches; the backup's 86.0 and 95.5 % were rounded-tier means and are now 85.8 and 95.7 %; the takeovers the filter kept (expert about 110 per run, not 200) are counted in these runs' own logs |

The asking-rate report was also received as `C:\Users\youss\OneDrive\active_ratematch_report\`; it is identical to
the copy above.

## 2. Second training machine, October 2026

New stack and the sometimes-careless operator; 24–29 Oct, 58 runs.

| dataset | stored in | contents | supports |
|---|---|---|---|
| All run folders, slim | `results/second_machine/2026-10_new_stack_and_careless_operator/runs/` | `inconsistent_operator/` (Q1_validity_new, headroom_diagnostic_new, two_arm_pilot_new, two_arm_confirm_new, stage0, reports, status files); `synthetic_suite/` (S_smoke, stage0) | quality scores (page 26, backups 47–49), rescue and tolerance (backups 43–44) |
| The machine's own agent records | `…/automation/` | decisions, proposals, validation, logs | how the run went (`docs/thesis/second_machine_synthesis.md`) |
| Summaries made for the deck | `…/curated_summaries/` | `cells.csv`, dependency reports, stage-0 results, the agent's status reports | the same slides |
| Result archives as delivered | `results/_received/inconsistent_operator_*_LSPEM499_2026-10-2*.tgz` | 4 archives (24, 25, 26, 29 Oct) | originals |
| **The complete archive** with weights, demos, `.venv` and W&B: 95.7 GB | **not copied:** `C:\Users\youss\Downloads\AQI_HIL_SERL_complete_2026-10-29.zip` and its extracted folder | everything | only needed to re-run evaluations. Its file list with SHA-256 per file is kept in `results/_received/second_machine_complete_archive_index/MANIFEST.json`. **Move the zip to an external disk.** |
| Documentation subset of that archive (914 MB) | not copied: `C:\Users\youss\Downloads\AQI_HIL_SERL_documentation_2026-10-29.zip` | repo and earlier archives, contained in the complete archive | — |

## 3. Second training machine, November 2026: the comprehensive study

5–9 Nov, 168 runs plus 8 smoke runs.

| dataset | stored in | contents | supports |
|---|---|---|---|
| Full export | `results/second_machine/2026-11_comprehensive_study/` | reports, plots, `report_cells.csv`, ruler results, `episodes.jsonl`, `convergence_evals.jsonl`, takeover logs, compacted logs, run status, the machine's one local patch | `docs/thesis/comprehensive_study_results.md` |
| Original zip | `results/_received/AQI-HIL-SERL-study-results-2026-11-09_065159.zip` | as received | — |
| Slim working copy for the deck | `presentation_bundle/results/study/` | reports, cells, audits, plots, `cell.json`, analysis scripts | Siemens deck page 27 (figure `r_study`: success against help from `report_cells.csv`, AUROCs from §4 of the results doc) and backup 50 (tables of §3–4) |

Re-run the study's own report with
`python examples/synthetic_suite.py --suite study --root results/second_machine/2026-11_comprehensive_study report`.

## 4. RWTH cluster (CLAIX)

| dataset | stored in | supports |
|---|---|---|
| Method test, 26 Sep | `results/cluster/cluster_method_test/` | findings F4, F7 (`handoff/FINDINGS_cluster_method_test.md`) |
| Counterfactual score study, 29 Sep | `results/cluster/cf_study/` | better-grounded critic scores below chance (`docs/counterfactual_score_study.md`) |
| Small E2HiL campaign, 5 Nov | `results/cluster/e2hil_small_claix/` | failed technically: rendering stall at step 201, no result |

## 5. The analysis behind the numbers

Scripts and their saved outputs are in `results/analysis/scripts/` and `results/analysis/outputs/`.

| script | computes | used for |
|---|---|---|
| `recompute_no_novice.py` | results for experts and intermediates (ICC, sign tests, convergence, help) from each campaign's `dependency.json` | deck pages 20–23, backup 39 |
| `ei_details.py` | worst operators, paired help | page 23 notes, backup 39 |
| `careless_cells.py` | discard vs re-weight, perfect filter, keep-all | page 26, backups 47, 49 |
| `audit_compare.py`, outputs `audit_ei/`, `audit_all/` | score AUROCs with and without novices | page 26 chart, backup 48, `docs/learn/06_quality_scores.md` |
| `ratematch_audit_ei.py` | critics' comparison against success: AUROC 0.15 on 5 790 corrections | page 24 |
| `first_success.py`, `first_success_hil.py`, `clock_time_ei.py`, `train_time_split.py`, `steps_ratio_ei2.py` | first unaided success, time-outs, clock time per run | backup 43 |
| `why_inconsistent.py` | the same setup in three campaigns; run-to-run spread | backup 40 |
| `why_ratematch.py`, `why_ratematch_b.py` | why asking won on page 23: references, takeover timing | page 23 notes. Part B of `why_ratematch.py` stops with an error, because the first asking experiment has no `segments.csv` |
| `verify_p21.py` | every run behind page 22, by tier | page 22 |
| `study_analysis.py`, `study_budget.py`, `hiril_requests.py` | comprehensive study: speed, effort, ruler at fixed budgets, HiRIL's requests | `docs/thesis/comprehensive_study_results.md` |
| `ablation_analysis.py` | ensemble-fix ablation: paired success, steps and operator share against the first version and HIL-SERL | `docs/thesis/ensemble_fix_ablation_results.md` |
| `ablation_full_analysis.py` | ensemble-fix ablation from the run folder: last checkpoint, stability, ruler at fixed budgets, first unaided success, the low runs | `docs/thesis/ensemble_fix_ablation_results.md` §2 |
| `archive_data.py` | builds this `results/` folder | — |
| `examples/novice_ablation_report.py` (repo, fixed 14 Nov); outputs `outputs/first_machine_campaigns/novice_ablation_report_fixed.*`, `oracle_campaign_dropped_by_arm_fixed.md` | the novice test's table and what each arm of the perfect-score campaign dropped, with "would" for arms whose filters only logged | backup 51, `docs/thesis/first_machine_novice_test_and_ladder.md` |
| `fm_inventory.py`, `fm_validity.py`, `fm_check_numbers.py`; outputs in `outputs/first_machine_campaigns/` | the first machine's archive: runs per campaign, crashes and bugs, per-arm tier means from the raw runs | `results/first_machine/campaigns/README.md`; the check of deck page 25 and backup 51 |

**Paths.** The scripts keep the paths they ran with. Translate them like this:

| path in the script | now in |
|---|---|
| `C:\Users\youss\Downloads\AQI_HIL_SERL_complete_2026-10-29\repo\runs\…` | `results/second_machine/2026-10_new_stack_and_careless_operator/runs/…` |
| `…\scratchpad\study_results\study` | `results/second_machine/2026-11_comprehensive_study` |
| `…\scratchpad\ei_logs` | the expert and intermediate runs of `…/runs/inconsistent_operator/two_arm_confirm_new/aqi_logs/` |
| `presentation_bundle/results/<campaign>` | unchanged; also in `results/first_machine/campaign_reports/<campaign>` |

**The audit for experts and intermediates** was made with:

```bash
python examples/inconsistent_operator_audit.py --log_root <folder with the 24 exp*/int* runs of two_arm_confirm_new/aqi_logs> --out <dir>
```

## 6. Other material you gave me (not data, kept where it is)

| what | where |
|---|---|
| your edited versions of the Siemens deck | `presentation_bundle/output/_backup/` (every version since 5 Nov, 21:36); current: `presentation_bundle/output/AQI-HIL-SERL_Siemens_short_updated.pptx` |
| deck generators | `presentation_bundle/output/_siemens/` (short deck), `presentation_bundle/output/_build/` (long deck) |
| thesis LaTeX draft | `C:\Users\youss\OneDrive\Downloads\Thesis+Starter+Kit++(Status+24.03.25)\…\50 Thesis\20_MA_Thesis_LaTeX` |
| papers and planning documents | `C:\Users\youss\OneDrive\Work Computer\` |

## 7. Gaps — what is not here yet

- ~~The first machine's raw run folders~~: complete since 14 Nov (`results/first_machine/campaigns/`).
- ~~The ensemble-fix ablation's run folders~~: complete since 14 Nov.
- Network weights of any machine: not kept anywhere in `results/`. Only needed to re-run a ruler.
