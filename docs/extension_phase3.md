# Extension study, phase 3: the main study on the harder task

Phases 0–2 ([extension_study.md](extension_study.md)) chose the task and the operators of the main study: the hex peg
from the wider start range (`hexwide`), with the novice at skill 0.15. Phase 3 runs the main study there, on the second
training machine, unattended and in one go:

```bash
bash tools/run_phase3_unattended.sh          # in the background; resumable
tail -f runs/phase3/pipeline.out             # progress
```

Status and results: `results/phase3/RUN_STATUS.md`, `results/phase3/PHASE3_REPORT.md` (never pushed).

| step | campaign | runs | what it is for |
|---|---|---|---|
| probe | `phase3/X0_probe` | 2 | The NaN probe: phase 2's hex pilot run that diverged (keepall, intermediate int00, seed 0), twice, 3 000 environment steps, on its own inputs. |
| smoke | `phase3/X_smoke` | 8 | Every new code path once: one careless intermediate, seed 0, 4 000 environment steps. |
| main | `phase3/X1_careless` | 144 | 12 arms with the sometimes-careless operator (careless in 40 % of takeovers, persistence 0.7). |
| twin | `phase3/X2_careful` | 72 | 6 arms on the same operators, demonstrations and ruler; the operator is never careless. |

Each main campaign has 2 experts, 2 intermediates and 2 novices at skill 0.15, with seeds 0–1, so 12 runs per arm.
Runs are paired by operator and seed. The ensemble is set as in the final study: 10 critics, minimum over 2 in the
target, warm start of 5 000 updates. Imitation weight λ_BC = 0.1, uniform (γ = 0); discard share 40 %.

## Arms

| arm | X1 | X2 | asking | use of the corrections |
|---|---|---|---|---|
| `hilserl` | x | x | – | plain HIL-SERL: 2 critics, no warm start |
| `e2hil` | x | x | – | E2HiL (Deng et al., RA-L 2026): the entropy-bounded actor loss, minimum over the critics, as published |
| `e2hil-mean` | x | | – | E2HiL's mask with the mean over the critics (HIL-SERL's actor): the mask alone |
| `keepall` | x | x | – | ensemble, keep every correction, no imitation |
| `bc` | x | x | – | imitate every correction |
| `bc-discard` | x | x | – | imitate; the behavioural score drops the lowest 40 % |
| `bc-random` | x | | – | imitate; a random 40 % dropped (same loss of data, blind) |
| `bc-perfect` | x | | – | imitate; exactly the careless corrections dropped (reads the label): the ceiling |
| `asks-15` | x | | σ_Q, 15 % | keep every correction, no imitation |
| `random-15` | x | | random, as often | keep every correction, no imitation |
| `asks-15-bc` | x | | σ_Q, 15 % | imitate every correction |
| `aqi` | x | x | σ_Q, 15 % | imitate; the score drops the lowest 40 % |

The no-imitation grading arms of earlier designs (`discard`, `perfect` without imitation) are left out: without
imitation, grading had nothing to act on in every earlier test. HiRIL is left out: its requests are too short for the
synthetic operators to answer, so it would not be a fair test.

## Fixed budget and measurement

- **Budget:** every run trains exactly B = 20 000 environment steps (`steps_cap`).
- **The mini-evaluation observes:** `converge_mode: observe` keeps the 5-pose mini-evaluation every 2 000 steps, but it
  never stops the run. The first time it passes twice in a row is written to `WOULD_STOP.json`. So the earlier
  measures (environment steps and operator steps until the stop) come from the same runs.
- **Checkpoints on environment steps:** `checkpoint_env_every: 2000`. The k-th checkpoint is saved once the learner has
  done k · 2 000 online updates under the fixed update ratio of 1, i.e. after it has received at least k · 2 000
  environment steps. The warm start does not shift them. At the stop the learner first catches up with the steps it
  has received, then saves the last checkpoint (about 20 000). `env_checkpoints.jsonl` lists them.
- **Ruler:** every checkpoint is scored with the 9-pose ruler, each run right after it is trained
  (`eval.after_each_cell`). Then its checkpoints are pruned to the best window and the last three
  (`eval.prune_keep_last: 3`), so the disk holds one run's full set at a time.
- **Primary measures:**
  - *success at B*: mean ruler success of the last three checkpoints (16 000, 18 000, 20 000): 27 trials;
  - *AULC*: mean ruler success over all ten checkpoints;
  - *operator steps until the would-be stop* (runs without one: over the whole budget, marked).
- **Secondary:** operator steps over the whole budget. The quantile gate keeps asking until the end, so this measure
  penalises asking by construction; it is reported, but it is not the effort measure.

## Divergence guard and the NaN probe

Some earlier runs diverged numerically (NaN in the learner), on the hex peg within the first episodes. The cause is
open.

- **Guard** (`divergence_guard_every: 100`): every 100 learner steps the learner checks that all its metrics are
  finite. On the first NaN or inf it writes `DIVERGED.json` (step, phase, the non-finite metrics, the last finite ones)
  and exits without a checkpoint. The actor sees the marker at its next episode end and stops.
- **Retry:** the runner keeps a copy in `<campaign>/diverged/` and marks the run crashed. The suite retrains it once
  from scratch with the same inputs. A second divergence makes it a technical failure: its pair is left out and listed.
- **Probe** (`X0_probe`): learner metrics and the guard every 50 learner steps. `PROBE.md` says whether the divergence
  reproduced and which metric broke first. A fix, such as gradient clipping in every arm, would be a design change and
  is decided after reading it; the main campaign runs with the guard either way.

## New instrumentation (diagnostics only; training is unchanged)

- **Careless counts in every arm:** with the inconsistent operator, `episodes.jsonl` gets `careless_takeovers` and
  `careless_steps` per episode. Plain HIL-SERL and E2HiL write no takeover log, so this is their only record of how
  careless their operator was.
- **E2HiL's careless label:** a step of a careless takeover gets source label 3 (careful takeovers keep 2). E2HiL logs
  the count, the mean |c(s)| and the kept share per source: `demo`, `online`, `intervention` (careful), `careless` and
  `takeover` (both). Labels never enter a loss.
- **Local learner metrics** (`metrics_log_every: 200`): `learner_metrics.jsonl` next to the checkpoints, whatever W&B
  does.

## Analysis (fixed now)

`python examples/phase3_analysis.py --root runs/phase3 --out results/phase3` (the suite runs it at the end):

| hypothesis | contrast | primary measure |
|---|---|---|
| H4b grading under imitation | `bc-discard` − `bc`; `bc-discard` − `bc-random`; `bc` careless − careful | success at B |
| H3′ asking | `asks-15` − `keepall`; `asks-15` − `random-15` | operator steps until the would-be stop (and success at B) |
| H5′ combination | (`aqi` − `asks-15-bc`) − (`bc-discard` − `bc`); `aqi` − `hilserl` | success at B |
| H6 E2HiL | `e2hil` − `hilserl`, careless and careful; their difference (does E2HiL gain more with a careless operator?); `e2hil` − `e2hil-mean` | success at B |
| H6 E2HiL's criterion | per E2HiL run: kept share of careless minus careful takeover states | paired over runs |
| H1′ dependency | success at B by tier, expert − novice, per arm | descriptive |

- Exact two-sided sign and Wilcoxon signed-rank tests, paired by operator and seed; Holm within each hypothesis on its
  primary measure. Every contrast also lists AULC and both operator-step measures.
- Diverged or crashed runs never enter a contrast; their pairs are listed.
- `bc-perfect` is reported as the ceiling of grading, not tested.
- The interaction of operator carelessness rests on experts and intermediates: a careless novice executes like a
  careful one.

## Guardrails

- **Inputs:** the probe copies `operators/int00.json`, `demos/int00_s0` and `eval_grid.json` from
  `runs/extension/P1_hex/`; X2 copies `operators/`, `demos/` and `eval_grid.json` from X1. Both are checked by hash
  (`inputs_from.json`). Inputs that differ are never trained on: that step is skipped and recorded.
- **Read only:** `runs/extension/` (phase 2: the decision, the hex orientation limits, P1_hex's inputs).
- **No stops:** nothing waits for a person. Failures are recorded in `RUN_STATUS.md` and the next step starts. A rerun
  resumes: finished runs are kept.
- **Never pushes:** results go to `results/phase3/`.
- **New switches** are all off by default: `converge_mode: observe`, `checkpoint_env_every`, `divergence_guard_every`,
  `metrics_log_every`, `eval.after_each_cell`, `eval.prune_keep_last`. All 84 earlier configs give byte-identical
  commands (dry-run snapshot before and after).
- **Time:** about 226 runs at roughly 0.8 runs per hour with two side by side, so about 12 days.

Tests: `tests/test_run_guard.py` (the instrumentation), `tests/test_phase3.py` (commands, analysis, suite helpers),
`tests/test_e2hil.py` (the careless label in E2HiL's diagnostics; needs JAX).
