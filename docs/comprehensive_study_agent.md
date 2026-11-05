# The comprehensive study: instructions for the agent on the second training machine

For the Claude working on the second training machine (Linux, RTX 5090), which gets this repo by cloning it from
GitHub. The design, the decision rules fixed before the runs and the commands are in
**[comprehensive_study.md](comprehensive_study.md)**: read it first. This page says how to work and what to report.

## Rules

- **Do not change** a config, an arm, the seeds, the careless share or the decision rules. If something looks wrong,
  stop and report it with the evidence; the user decides.
- **Credentials:** never type, paste or read the W&B key or any other credential. The user runs
  `./.venv/bin/wandb login` themselves, and handles any GitHub login.
- **Data:** never delete or overwrite a run folder in `runs/`. Re-running the same `run` command resumes and skips
  finished cells. Never mix these cells with another machine's.
- **Results:** everything the user hands over for the analysis is in **`results/study/`**. The driver refreshes it after
  every command and after every finished campaign. Do not hand over `runs/` or checkpoints.
- **Code fixes,** if a check fails because of a bug:
  - one commit per fix, with a test;
  - `git format-patch -1 HEAD -o results/study/patches/`, so the fix travels with the results;
  - push only if the user asks.
- **Report after each step** with the numbers asked for below, not only "done".

## 1. Set up (about 15 minutes)

```bash
bash tools/setup_workstation.sh
```

The user, once: `./.venv/bin/wandb login`

```bash
./.venv/bin/python examples/synthetic_suite.py --suite study check
```

**Report:**
- GPU, driver, CPU count and free disk;
- whether `check` says READY, and if not, its problem list;
- the feature lines: 0014–0021, 0037 and 0048 must all be ok.

## 2. Stage 0, no training (about 1 hour, CPU)

```bash
./.venv/bin/python examples/synthetic_suite.py --suite study stage0
```

**Report:**
- **tests:** every test file passes (`results/study/stage0/tests.json`);
- **operator conditions:** careful against careless rescue, with rotation control on and off;
- **orientation limits:** `results/study/stage0/limits.json`; the earlier second machine measured tilt 0.384, yaw 0.1396 rad.

**Gate:** all tests pass and careless rescues are clearly below careful ones.

## 3. Smoke (about 2–4 hours)

```bash
./.venv/bin/python examples/synthetic_suite.py --suite study run smoke --parallel 2
./.venv/bin/python examples/synthetic_suite.py --suite study rulercheck
./.venv/bin/python examples/synthetic_suite.py --suite study audit S0_smoke_B
./.venv/bin/python examples/synthetic_suite.py --suite study report smoke
```

**Report, per cell (8):**
- trained and scored;
- actor minutes and ruler minutes;
- env steps when training stopped, and ruler success.

**Report, overall:**
- **HiRIL and E2HiL:** both log a condition per takeover (`aqi_logs/*/segments.jsonl`), and HiRIL makes requests;
- **careless share** per arm of `S0_smoke_B` (expected about 0.2);
- **E2HiL's keep fraction** from its learner log (expected about 0.85);
- **ruler check:** how many smoke scores the shorter ruler would change (`<campaign>/report/ruler_reselect.txt`);
- **parallel timing:** wall time of the two side-by-side shards against the sum of their cells' times;
- **W&B:** the runs appear under the tag `comprehensive_study`.

**Gate:**
- all 8 cells trained and were scored;
- the conditions are logged and the careless share is 0.1–0.3;
- **the ruler check changes no score.** If it does, stop and report: A and B would then need the full ruler.
- **Parallel:** if two side by side are not faster than one after the other, use `--parallel 1` below.

## 4. The studies (about 4–7 days)

```bash
./.venv/bin/python examples/synthetic_suite.py --suite study run A_consistent B_quality --parallel 2
```

**Report once a day:** cells done per campaign (`results/study/README.md`, or
`synthetic_suite.py --suite study plan`), any failed cell and why (`results/study/<campaign>/logs/<cell>.*.log`),
minutes per cell.

## 5. Analysis

```bash
./.venv/bin/python examples/synthetic_suite.py --suite study audit B_quality
./.venv/bin/python examples/synthetic_suite.py --suite study report
./.venv/bin/python examples/synthetic_suite.py --suite study plots
```

Then answer the questions of [comprehensive_study.md](comprehensive_study.md) with its rules, quoting the numbers:
- **A, critics:** success, steps, effort and minutes per K; paired against `ens-k8` and `hilserl`; the recommended K.
- **A, asking:** success against effort; ICC per arm; the recommended rate; `asks-55` against `random-55`.
- **B, detection:** AUROC (within) of each score with CI and verdict.
- **B, benefit:** each quality arm against `keepall` and `hilserl`; `perfect-filter` against `keepall` (headroom);
  `e2hil` and `hiril` against `hilserl`.

Write that report to `results/study/ANALYSIS.md`.

## 6. Hand over

The user hands over the folder `results/study/` (for example
`cd results && tar czf study_results_$(date +%F).tgz study`). It holds:
- the reports, plots and audits;
- ruler outcomes, training records, takeover logs and compacted logs;
- `ANALYSIS.md` and any `patches/`.

No repo copy and no checkpoints.
