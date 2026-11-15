# Your questions, and where they are answered

Every question asked while preparing the mid-term presentation (4–5 November 2026), with a one-line answer and the file
and section that explain it in full. All files are in `docs/thesis/` unless a path is given.

**How to read these files.** In VS Code: open the file and press `Ctrl+Shift+V` (preview) or `Ctrl+K V` (preview beside
the text). On GitHub (`ytelha/hil-serl_aqi`) they render directly. The links below are relative, so they work in both.

**Start here:**
- [presentation_summary_both_machines.md](presentation_summary_both_machines.md): the one file to show, with both machines,
  formulas, worked examples and a glossary;
- [training_iteration_and_statistics.md](training_iteration_and_statistics.md): how the learner works, and the
  statistical tests with worked examples.

## Asking for help (σ_Q and the asking rate)

| question | short answer | where |
|---|---|---|
| How does the asking rate work, with numbers that are easy to follow? | Every step: the 8 critics' spread σ_Q → its z-score → compared with a line that 55 % of the last 2 000 scores exceed; a crossing opens a 10-step request, then 5 steps off | [asking-rate log §3.1](first_machine_asking_rate_results_and_qa_log.md#31-a-worked-example-step-by-step); [summary §3.1–3.3](presentation_summary_both_machines.md#3-first-machine-how-often-should-the-robot-ask-108-runs) |
| Is σ_Q stored for every step, or fitted to a normal distribution? | Three running numbers for the z-score, the last 2 000 scores for the line; nothing is fitted, the line is a count | [asking-rate log §3.2](first_machine_asking_rate_results_and_qa_log.md#32-what-the-gate-keeps-in-memory-and-how-its-line-moves); [summary §8](presentation_summary_both_machines.md#8-questions-asked-while-preparing-with-short-answers) |
| What did we learn from it? | A fixed line gives an accidental rate (19 → 36 % within a run); the rank line holds it (53–60 %); a deployed system needs a floor | same |
| Does this make the threshold adaptive? | Yes, recomputed every step; it fell from 0.0131 to 0.0061 as σ_Q fell | same |
| Does "asks 55 %" request 55 % of the steps? | No: about 57 % requested, 42 % helped | [asking-rate log §9](first_machine_asking_rate_results_and_qa_log.md#9-qa-log) |

## The presentation summary

| question | short answer | where |
|---|---|---|
| Can the .md files be summarised into one file for the presentation, both machines? | Yes, with the parts you chose: the story, asking, grading, new stack, careless operator, closing questions, glossary | [presentation_summary_both_machines.md](presentation_summary_both_machines.md) |
| Symbols, worked examples, why decisions were made, which arms are compared with HIL-SERL | Explained in each section; the glossary at the end | [summary §10](presentation_summary_both_machines.md#10-glossary) |
| Is the novice too weak? | Yes: it answers 2–9 of 100 requests, and 97 % of its takeovers fall in failed episodes; the next study leaves it out | [summary §7](presentation_summary_both_machines.md#7-the-synthetic-novice-is-too-weak); Siemens deck backup page 52 |

## The cluster

| question | short answer | where |
|---|---|---|
| What came out of the cluster runs (5 November)? | The 4-run E2HiL test failed technically: the camera rendering stalled in every run once the policy acted (step 201); no result | `presentation_bundle/results/e2hil_small_claix/RESULT.md`; [e2hil_inconsistent_operator.md](../e2hil_inconsistent_operator.md) |
| Does the September run make sense to use? | It was a method test and a 1-cell counterfactual study: it supports "the counterfactual score cannot be repaired", nothing more | `presentation_bundle/results/cluster_method_test/RESULT.md` |

## The comprehensive study (second machine)

| question | short answer | where |
|---|---|---|
| Which campaigns, with the new operator model, the careless operator and no novice? | Study A (96 cells): HIL-SERL, 4 / 8 / 16 critics, asks 40 / 55 / 70 % and random; Study B (72 cells): HIL-SERL, E2HiL, HiRIL, keep-all, behavioural score (discard, re-weight), oracle, perfect filter, with a 20 % careless expert | [comprehensive_study.md](../comprehensive_study.md) |
| Only the results should come back, in a folder called `results` | `results/study/`: reports, plots, audits, compacted logs, no checkpoints | [comprehensive_study.md, "The results folder"](../comprehensive_study.md#the-results-folder) |
| It must run once, unattended, without stopping | `bash tools/run_study_unattended.sh`; decisions are recorded in `results/study/RUN_STATUS.md` instead of waiting | [comprehensive_study.md, "Run it"](../comprehensive_study.md#run-it-one-command-unattended) |
| Does the prompt for the other Claude set up the venv and dependencies? Is anything missing? | Yes, `tools/setup_workstation.sh`; the disk check (about 250 GB) and the prerequisites were added | [comprehensive_study_agent.md](../comprehensive_study_agent.md) |

## Today: the learner and the statistics

| question | short answer | where |
|---|---|---|
| How does one training iteration work? | The actor observes, acts (or the operator takes over), steps and stores; the learner samples 128 + 128 and makes 2 updates per learner step | [training_iteration_and_statistics.md §2–4](training_iteration_and_statistics.md#2-one-environment-step-on-the-actor); Siemens deck page 64 |
| How is the data stored and sampled; what are the buffers? | Online buffer (everything) and intervention buffer (demonstrations + takeovers; in the grading arms only kept takeovers, with a priority); uniform sampling, half from each | [§3](training_iteration_and_statistics.md#3-the-buffers-where-a-transition-goes) |
| How are both networks updated? | Critics towards y = r + 0.97 · Q̄(s′, a′); actor towards the critics' mean minus α · log π; temperature tuned; target critics follow at 0.5 % | [§4](training_iteration_and_statistics.md#4-one-learner-step); Siemens deck page 66 |
| How much data does each ensemble critic see compared with plain HIL-SERL? | 410 against 512 transitions per environment step (its own 80 %), plus about 2 M in the warm start; 3.7 M + 2 M against 4.6 M after 9 000 steps | [§5](training_iteration_and_statistics.md#5-the-pace-and-how-much-data-each-critic-sees); Siemens deck page 65 |
| Simple worked examples of the statistical tests | ICC 0.77, sign test p = 0.021, AUROC 0.67, standard error (the permutation test is in the doc but off the slides for now) | [§6](training_iteration_and_statistics.md#6-the-statistical-tests-with-worked-examples); Siemens page 16 and the notes of page 69 |

## Earlier questions (already in the summary)

Why the ruler takes about 55 minutes; whether it can be shorter; why time halved when the steps fell to a quarter; the
fixed update ratio and whether it slows convergence; solo and rescue; validity, perfect-filter check, pilot and
confirmation; how carelessness was implemented; what the scores mean; whether the first machine used the old setup:
[summary §8](presentation_summary_both_machines.md#8-questions-asked-while-preparing-with-short-answers).
