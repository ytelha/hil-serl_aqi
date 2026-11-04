# The operator that can turn the peg, and the new stack — results & QA log

Companion to [second_machine_synthesis.md](second_machine_synthesis.md). This log covers
what was measured on the **second training machine** without or before the quality
question: the operator alone in the simulator, the orientation tolerance, the smoke test,
how fast training runs on the new stack, and the engineering record.

- Machine: `LSPEM499`, one RTX 5090 (32 GB, driver 610.88), 24 CPU threads, Linux under WSL2, repo at commit `d191f9d`
  (patches 0001–0019), Python 3.12.3, JAX 0.11.0, MuJoCo 3.10.0: the first machine's exact
  package lock.
- Dates: 24–29 October 2026.
- Raw results: `presentation_bundle/results/second_machine_stage0/`,
  `presentation_bundle/results/S_smoke/` and `presentation_bundle/results/second_machine_reports/`.
- Tools: `examples/oracle_insertion_test.py`, `examples/synthetic_suite.py stage0`.
- All tables are generated from those files and the raw logs. Checked on 2026-11-03 against the machine's
  reports; section 9 lists what that check changed.

---

## 0. Headline

**The synthetic operator of every finished cuboid campaign could rarely rescue a twisted
peg.** With rotation control it can. Success of a takeover after 60 random actions rises
from 13 / 10 / 7 % (expert / intermediate / novice) to 100 / 100 / 70 %. The cause is
measured: the peg is about 21° twisted at the takeover, and without rotation control the
expert tolerates about 8° of yaw.

**On the new stack the slow start of the cuboid is gone.** For the same method (ensemble extended), the first
success without help comes after about 4 600 environment steps, against about 18 000–20 000 on the first machine.
This compares two machines and five changes at once, so it shows the size of what can be gained, not which change gains it. The ladder that
separates them (`synthetic_suite` pilots) has **not** been run.

**Learning got faster; a whole cell did not get much shorter.** Training takes about 30 minutes per cell, but the final ruler takes about 55
more, so a cell still costs about 1.5 h. A shorter ruler that leaves every one of these cells' scores unchanged would take
about 20 minutes (§5 (c); implemented, not yet used in a campaign).

---

## 1. The new stack

"New stack" means all of these switched on, with the settings of `L5_full`:

| change | patch | what it does | type |
|---|---|---|---|
| rotation control | 0014 | the operator also turns the tool to the nearest insertable orientation | operator |
| orientation box | 0016 | the task clips the commanded orientation: ±0.1 rad roll and pitch, ±0.4 rad yaw | task |
| orientation channel | 0016 | a twist raises the operator's trouble signal | operator |
| mini-eval convergence | 0015 | the autonomous mini-eval decides when a cell stops (threshold 0.6, 2 passes in a row, 5 poses, every 2 000 steps) | training |
| fixed update ratio | 0015 | 1 learner step per environment step, so learning per step does not depend on the GPU | training |

The "original" platform is the operator and task of the finished campaigns.

## 2. The operator alone (no training)

30 episodes per row, cuboid and cylinder. *Solo* starts from a training reset; *rescue*
takes over after 60 random actions. No learning is involved, so these numbers should carry
over to any machine with the same package versions.

| peg | tier | start | success, rotation off → on | median steps of a success run, off → on | twist when the operator starts (mean) |
|---|---|---|---|---|---|
| cuboid | expert | solo | 96.7 → 100.0 % | 17 → 17 | 2.4° (max 4.6°) |
| cuboid | expert | rescue | 13.3 → 100.0 % | 69.5 → 31 | 21.0° (max 32.3°) |
| cuboid | intermediate | solo | 86.7 → 100.0 % | 24.5 → 22 | 2.4° (max 4.6°) |
| cuboid | intermediate | rescue | 10.0 → 100.0 % | 87 → 56 | 21.0° (max 32.3°) |
| cuboid | novice | solo | 40.0 → 83.3 % | 59 → 103 | 2.4° (max 4.6°) |
| cuboid | novice | rescue | 6.7 → 70.0 % | 158.5 → 109 | 21.0° (max 32.3°) |
| cylinder | expert | solo | 100.0 → 100.0 % | 16.5 → 16.5 | 0.2° (max 0.2°) |
| cylinder | expert | rescue | 96.7 → 100.0 % | 36 → 28 | 15.0° (max 29.7°) |
| cylinder | intermediate | solo | 100.0 → 100.0 % | 20.5 → 20 | 0.2° (max 0.2°) |
| cylinder | intermediate | rescue | 86.7 → 100.0 % | 56.5 → 40 | 15.0° (max 29.7°) |
| cylinder | novice | solo | 90.0 → 83.3 % | 41 → 43 | 0.2° (max 0.2°) |
| cylinder | novice | rescue | 70.0 → 76.7 % | 140 → 80 | 15.0° (max 29.7°) |

- **The cuboid's rescue is what changes:** the expert goes from 13 to 100 %, and when it succeeds it also takes less than half the steps.
- **The cylinder barely changes,** as expected: a round peg has no yaw to get wrong. This is why
  the problem only showed on the cuboid.
- **Solo starts are not the problem.** The twist there is 2.4° on average, within the tolerance below.
- **The novice is slower when it can turn** from a training reset (59 → 103 steps), probably because it now also waits to align the yaw. It succeeds twice as often.

## 3. Orientation tolerance (no training)

10 episodes per angle, starting centred over the bore with a set twist. Success in %:

| peg | axis | tier | rotation | 0° | 2° | 4° | 6° | 8° | 10° | 12° | 15° | 18° | 22° |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuboid | yaw | expert | off | 100 | 100 | 100 | 90 | 90 | 50 | 30 | 0 | 0 | 0 |
| cuboid | yaw | expert | on | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| cuboid | yaw | novice | off | 50 | 20 | 30 | 10 | 40 | 30 | 30 | 0 | 0 | 0 |
| cuboid | yaw | novice | on | 50 | 70 | 80 | 60 | 80 | 60 | 70 | 80 | 50 | 80 |
| cuboid | tilt | expert | off | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 90 |
| cuboid | tilt | expert | on | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| cuboid | tilt | novice | off | 50 | 30 | 40 | 50 | 40 | 40 | 30 | 30 | 10 | 20 |
| cuboid | tilt | novice | on | 50 | 70 | 60 | 70 | 80 | 60 | 80 | 90 | 50 | 20 |
| cylinder | tilt | expert | off | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| cylinder | tilt | expert | on | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| cylinder | tilt | novice | off | 90 | 100 | 100 | 90 | 90 | 90 | 90 | 80 | 90 | 90 |
| cylinder | tilt | novice | on | 90 | 100 | 70 | 80 | 90 | 70 | 100 | 80 | 90 | 90 |

By the rule "largest angle with at least 80 % of the success at 0°":

| peg | axis | tier | tolerance by the rule |
|---|---|---|---|
| cuboid | tilt | expert | 22° |
| cuboid | tilt | novice | 0° |
| cuboid | yaw | expert | 8° |
| cuboid | yaw | novice | 0° |
| cylinder | tilt | expert | 22° |
| cylinder | tilt | novice | 22° |

- **With rotation control, the expert succeeds at every angle tested (0–22°),** in yaw and
  in tilt.
- **Yaw tolerance of the expert: 8°** (0.1396 rad). That is well below the geometric 19.6° of a perfectly centred
  peg, and below the placeholder limit of 0.25 rad.
- **Tilt tolerance: at least 22°.** The value 0.384 rad written to `limits.json` is only the
  top of the sweep, not a measured limit. With the orientation box at ±0.1 rad the tilt part of
  the channel practically cannot fire; the channel works through yaw.
- **Novice tolerance reads 0°** because the novice's success at 0° is already only 50 %, with
  10 noisy episodes per angle. It does not mean a novice cannot insert. With rotation control it holds 50–80 % across the whole yaw sweep.
- **Limits used by all later campaigns:** cuboid tilt 0.384, yaw 0.1396 rad; cylinder tilt 0.384 rad, yaw free.

## 4. Smoke test

One cell: plain HIL-SERL, expert, cylinder, full stack, W&B off, 0.41 h in all. All nine checks passed: the box in the actor, learner and ruler; the
orientation monitor; the fixed ratio in both processes; takeover channels; mini-eval; scored. Ruler 92.6 %, first unaided success at 1 221 steps, converged at 4 286 steps, operator
share 28 %, 92 training episodes with 88 successes, 61 of them without help. Not a result, as intended.

| mini-eval at env step | 1 221 | 2 275 | 3 283 | 4 286 |
|---|---|---|---|---|
| success | 0.0 | 0.2 | 1.0 | 1.0 |
| passes in a row | 0 | 0 | 1 | 2: stop |

## 5. Training speed on the new stack

"Speed" means three different things, so they are kept apart.

**(a) Learning per environment step: much faster.** Same-method rows are the fairest comparison: ensemble extended on both sides, recomputed from the raw logs (W&B episodes on the first machine, `episodes.jsonl` on the second).

| machine and platform | arm | cells | first success without help (median steps) | episodes out of time, first 10 000 steps | success, first 10 000 steps | steps when training stopped (median) | operator share | env steps per second | training time per cell |
|---|---|---|---|---|---|---|---|---|---|
| first machine, original | `hilserl`, first ensemble campaign | 12 | 18 836 | 95 % | 5 % | 28 404 | 53 % | 9.8 | 49 min |
| first machine, original | `hilserl`, fixed-ensemble campaign | 18 | 18 164 | 95 % | 5 % | 26 801 | 52 % | 9.8 | 47 min |
| first machine, original | ensemble extended (`logged`), fixed-ensemble campaign | 18 | 20 179 | 92 % | 8 % | 33 288 | 54 % | 9.4 | 79 min |
| first machine, original | ensemble extended (`logged`), asking experiment | 18 | 18 330 | 93 % | 7 % | 30 395 | 54 % | 9.0 | 76 min |
| second machine, new stack | ensemble extended (`logged`), validity (Q1) | 6 | 4 599 | 27 % | 73 % | 9 560 | 41 % | 6.3 | ≈ 36 min |
| second machine, new stack | keep-all, perfect-filter check | 3 | 5 585 | 22 % | 78 % | 8 166 | 43 % | 6.5 | ≈ 29 min |
| second machine, new stack | discard, pilot | 3 | 4 578 | 20 % | 80 % | 9 877 | 43 % | 6.1 | ≈ 30 min |
| second machine, new stack | re-weight, pilot | 3 | 6 382 | 20 % | 80 % | 8 401 | 41 % | 6.4 | ≈ 30 min |
| second machine, new stack | discard, confirmation | 18 | 4 174 | 29 % | 71 % | 9 007 | 41 % | 6.1 | ≈ 31 min |
| second machine, new stack | re-weight, confirmation | 18 | 5 442 | 34 % | 66 % | 8 802 | 39 % | 6.5 | ≈ 31 min |

- **Same method: about 4 × sooner.** First success without help 18 300–20 200 → 4 600 steps; out of time 92–93 → 27 %; success in the first 10 000 steps 7–8 → 73 %.
- **Steps at the stop are not comparable** (about 30 000 → 9 600). The stop rule changed from a rolling window of untouched episodes at 0.9 to a mini-eval at 0.6. A lower bar stops earlier by itself.
- **The new side has the harder operator:** about a third of its takeovers are inattentive.
- **Env steps per second** are averages over the online training (medians over cells); on the second machine from the actor's progress lines (corrected 2026-11-04, see (b)).
- Plain HIL-SERL on the first machine took about 18 000–19 000 steps. The docs earlier rounded this to about 17 000 from another count; the direction is the same.

**(b) Clock time of training: 2.5 × shorter, while the steps at the stop fell 3.4 ×.** Split measured on 2026-11-04 from
the progress lines in `logs/<cell>.learner.log` and `logs/<cell>.actor.log` of all 57 cells; the first machine from
the table above.

| part of a cell's training | first machine, ensemble extended | second machine, all arms |
|---|---|---|
| learner steps per env step | about 0.45 (finding F3) | 1 |
| demo pre-training (5 000 learner steps) | about 20 min (about 4 per second; engineering log, 6 Sep) | 5.1 min median (4.7–7.5; 16–21 per second) |
| online training: env steps | 30 400–33 300 | 8 700 median |
| online: env steps per second, average | 9.0–9.4 (paced at 10 Hz) | 6.2–6.4 (about 7 while stepping) |
| online training time | 56–59 min | 23–25 min (validity campaign: 29) |
| training per cell | 76–79 min | 29–36 min |

Check: plain HIL-SERL on the first machine has no pre-training, and 26 801 steps at 9.8 per second are 46 minutes,
against 47 measured. The earlier version of this table gave 7.7–8.4 steps per second for the second machine: that was
the progress bar's smoothed rate at the end of a run, not the average.

Why the time fell less than the steps:
1. **The 4 × is the first success without help, not where training stops.** Training runs on until the mini-eval passes
   twice: 3.4 × fewer steps than on the first machine.
2. **Each env step takes about 1.45 × longer** (9.2 → 6.3 per second on average). On the first machine the 10 Hz pacing
   was the limit, and the learner did what it could alongside, about 4 updates per second. With the fixed update ratio
   the pacing is off, but the learner must make one update per step, about 7 per second, on the same GPU that renders
   the two cameras and runs the actor's policy. The actor waited for the learner only 0.5–0.7 minutes per cell (its own
   log line), so the limit is the actor's step on a shared GPU. Even the ruler, with no learner at all, ran at only
   about 9 steps per second on that machine. The two or three mini-evals and the start-up take another 1–3 minutes.
3. **Pre-training does not shrink with fewer steps.** It is a fixed 5 000 learner steps. The RTX 5090 made it 4 × faster
   (20 → 5 minutes), but it is still a sixth of a new-stack cell's training.

From 77 to 31 minutes: fewer env steps save about 40 minutes, faster pre-training about 15, and the slower steps cost
about 8 back. At the first machine's step rate a new-stack cell would train in about 23 minutes. Whether the slower
step comes from the fixed update ratio or from the machine (another GPU, under WSL2) shows only on one machine: the
ladder rung that adds the fixed update ratio measures it. At its measured rate of about 5 learner steps per second,
the first machine should train a new-stack cell in about 50 minutes instead of about 78. That is an estimate; the smoke
test will measure it.

**(c) Clock time of a whole cell: about 1.5 h, mostly the ruler.**

| campaign | cells | total | demos | training (per cell) | ruler and reports (per cell) |
|---|---|---|---|---|---|
| smoke | 1 | 0.41 h | – | – | – |
| validity (Q1) | 6 | 10.09 h | 0.19 h | ≈ 3.6 h (36 min) ¹ | ≈ 6.3 h (63 min) |
| perfect-filter check | 9 | 12.4 h | reused | 4.3 h (29 min) | 8.1 h (54 min) |
| pilot | 6 | 8.42 h | 0.13 h | 3.0 h (30 min) | 5.3 h (53 min) |
| confirmation | 36 | 53.79 h | 0.74 h | 18.6 h (31 min) | 34.4 h (57 min) |
| **all 58 cells** | | **85.1 h** | | | |

¹ The machine's own estimate; Q1 spans the clock change, so file times cannot give it directly. The other rows add up to the recorded totals.

The ruler scores every saved checkpoint: on average 14–17 per cell (a checkpoint every 1 000 learner steps, including the 5 000 pre-training steps), 9 trials each. Pre-training takes 5–7 minutes of the training time. Scoring fewer checkpoints is the obvious way to shorten a campaign, but it changes the ruler's "best window", so it would have to be decided before a campaign and kept the same across a comparison. For the first machine there is only a rough total: its 18-cell probe took about a day, about 1.3 h per cell.

**Why the ruler takes longer than training** (measured from `eval_runs/<cell>/trials.jsonl` of the four campaigns, 2026-11-04):

| campaign | checkpoints scored per cell | of them from pre-training | trials per cell | ruler steps per cell | trials that run the full 300 steps |
|---|---|---|---|---|---|
| validity (Q1) | 16.5 | 5 | 148 | 33 157 | 71 % |
| perfect-filter check | 14.1 | 5 | 127 | 28 198 | 72 % |
| pilot | 14.2 | 5 | 128 | 27 907 | 70 % |
| confirmation | 14.9 | 5 | 134 | 30 372 | 73 % |

1. **The ruler scores every saved checkpoint,** not only the last, because it reports the best window of three in a row.
2. **A checkpoint is saved every 1 000 learner steps, and the first five come from the demonstration pre-training,**
   before the policy has acted once.
3. **Early checkpoints fail every trial, and a failed trial runs to the 300-step limit.** About 72 % of all trials do.
   Example `drop-hind_exp00_s0`: checkpoints 1 000–8 000 score 0 of 9, all at 300 steps; from 13 000 on, 9 of 9 at 12–17 steps.
4. **So the ruler simulates about 30 000 steps per cell, three times the training's 9 000–10 000.** Each step renders two
   camera images and runs the policy, so it runs at about 9 steps per second (30 372 steps in about 57 minutes), only a
   little faster than training (6.2–6.4 on average, see (b)), even without the 10 Hz pacing and without a learner.
5. **Faster learning did not reduce the number of checkpoints.** Checkpoints count learner steps, and the fixed update ratio
   doubled the learner steps per environment step (1 against about 0.45). The old setup made about 30 000 × 0.45 + 5 000 ≈
   18 500 learner steps per cell; the new one about 9 500 + 5 000 = 14 500: almost as many checkpoints, in a third of the
   environment steps.

**Shortening it (implemented 2026-11-04, off by default; `docs/eval_ruler.md`, *A shorter ruler*).** Three options, measured
by re-selecting the 57 cells from their own trials (`examples/ruler_reselect.py`):

| selection | ruler time | cells whose score changes |
|---|---|---|
| skip the pre-training checkpoints (`eval.skip_pretrain`) | 55 % | 0 |
| only the newest 8 (`eval.newest: 8`) | 39 % | 0 |
| **both** | **36 % (about 20 instead of 55 minutes)** | **0** |
| every second checkpoint (`eval.stride: 2`) | 51 % | 46, by −21 points on average |

Skipping costs nothing here because no cell succeeded at a pre-training checkpoint and the best window always sat among
the newest checkpoints (training stops right after the mini-eval passes twice). Every second checkpoint does change the
score: only the last two or three checkpoints are good, so the window of three reaches back into failing ones. With both
safe options, a cell would take about 30 + 20 minutes instead of 1.5 h. It must still be chosen before a campaign and kept
across a comparison, and on the old stack it has to be measured first on that machine's own cells.

**What each change is likely doing.** No training run has tested any of them alone.

| change | what we know | effect on speed |
|---|---|---|
| rotation control | rescue 7–13 → 70–100 %, measured without learning | probably the main cause; not isolated in training |
| orientation box | never tested alone; it also changes the task | unknown |
| orientation channel | triggers about 1 % of takeovers | probably small |
| mini-eval convergence | stops training earlier; can stop in a dip | saves time, does not make learning faster |
| fixed update ratio | 1 instead of about 0.45 learner steps per env step, but each step takes longer | may explain part of the 4 ×; costs clock time |

**Is success still skill-graded?** Final success is flat across tiers in the discard arm (92.6 / 93.2 / 93.8 %, against 94 / 78 / 65 % for ensemble extended in the first
machine's oracle campaign), but novices need 1.5–1.8 times the steps of experts (12 616 vs 6 946 and 11 060 vs 7 514). The ICC of 0.000 says little on its own: ensemble extended already has an ICC of 0.00, 0.18 and 0.84 in three campaigns on the original platform, so a single campaign's ICC is unstable. Whether plain HIL-SERL stays
operator-dependent on the new stack is unknown, because it was not run on the cuboid.

## 6. How the takeovers end and start

Over the 36 confirmation cells:

| what triggered the takeover | share | | how it ended | share |
|---|---|---|---|---|
| corridor | 45.0 % | | time cap $d_{\max}$ | 42.0 % |
| hover | 27.2 % | | the operator's signal | 36.8 % |
| false alarm | 9.2 % | | episode end | 21.2 % |
| off bore | 9.2 % | | | |
| force excess | 6.9 % | | | |
| **orientation** | **1.0 %** | | | |
| stagnation | 1.0 % | | | |
| time | 0.5 % | | | |

**The orientation channel hardly fires** (0.3–1.4 % across campaigns). Together with the
tilt limit above, this makes it the least justified piece of the stack. The ladder's
leave-one-out setting `X_nochannel` will tell whether it matters. Operator share stays high
at about 40 %.

## 7. Engineering record

**Environment.**
- All 153 retained lock requirements match their pins or the pinned VCS commit. Installed with `--no-deps`, without `setup_env.sh`, sudo or system packages.
- Requested adaptations: `opencv-python` → `opencv-python-headless` at the same version; robot hardware packages left out (`easyhid-ng`, `evdev`, `hidapi`, `pymodbus`, `pynput`, `pyrealsense2`, `pyspacemouse`, `python-xlib`, `rospkg`), with the repo's headless stubs on `PYTHONPATH`.
- `pip check` reports metadata conflicts, recorded and not resolved: the omitted packages, the OpenCV name, `tf-keras` asking for `tensorflow` while `tensorflow_cpu` is installed, and `dm-robotics-transformations` declaring NumPy < 2 against the pinned 2.1.3. The installer also notes that the pinned cuDNN package is yanked. The import, render and GPU checks all passed.
- No source was changed. The only tracked changes are `quality_score_mode` in the two `_new` two-arm configs. The repo reads as dirty only because of those and of untracked copied files.

**Tests.** 136 repository tests passed with no skips, across 18 files, on the real stack (`test-summary.json`). Two test bugs appeared and were reported rather than hidden; both are fixed in patch 0021:
- `test_run_passes_limits...` failed once real stage-0 limits existed, because `read_limits` fell back on them. It caused the first no-go together with the 20-point rule. The whole seven-file bundle (71 tests) then passed in a clean `git archive` of the same commit; the failed result is preserved.
- `test_suite_configs` failed after the intended score edit in the two-arm configs (13 of 14 in that file).

**Assumptions checked on the machine** (`handoff/EVALUATE_FIRST.md`, final state):

| | assumption | result |
|---|---|---|
| IO1 | patches 0014–0019 present | holds |
| IO2 | the actor uses the geometric oracle | holds (actor logs) |
| IO3, IO4 | the condition reaches the takeover's segment | holds (tests) |
| IO5 | segments carry `condition`, `drop_score`, `drop_warmup` | holds (every Q1 segment labelled) |
| IO6 | inattentive at least 20 points worse | holds on the new platform; **fails on the original** in 4 of 6 |
| IO7 | scores computed with window 10, discount 0.97 | holds (`meta.json`) |
| IO8 | a dropped takeover leaves only the intervention buffer | holds |
| IO9 | the warm-up holds the filter off | holds (pilot and confirmation) |
| IO10 | keep-all and the drop arms replay the same way | holds |
| IO11 | `_new` equals the full stack | holds |
| IO12 | GPU free, disk sufficient | holds |
| IO13 | a campaign without `hilserl` reports normally | holds |

**Stability.** All 57 cuboid training cells and the smoke cell converged. No crash, no traceback, no repeated
cell. The fixed update ratio worked between real processes with no deadlock.

**A non-fatal error in every learner.** One `CUDA_ERROR_OUT_OF_MEMORY` allocating 4 GiB at the
start of demo pre-training (`XLA_PYTHON_CLIENT_MEM_FRACTION` 0.6 on 32 GB). Training continues
normally. The log does not show which allocation failed. Worth watching on a smaller GPU.

**Other interventions on the machine.** At the user's request an unrelated backend (`Alive_Dashboard`) was stopped. About 1.6 GiB of GPU memory stayed allocated with no compute process, owner unknown, so the automation's idle check was raised from 1 500 to 4 096 MiB; the campaign memory fractions did not change. An early CPU stage-0 attempt was restarted while the automation was being tested. The automation itself lived outside the repo.

**Storage.** Up to 2.4 GB per cell, checkpoints included:

| campaign | cells | checkpoints per cell, mean | largest cell |
|---|---|---|---|
| smoke | 1 | 0.46 GB | 0.46 GB |
| validity (Q1) | 6 | 1.74 GB | 2.38 GB |
| perfect-filter check | 9 | 1.50 GB | 1.79 GB |
| pilot | 6 | 1.51 GB | 1.69 GB |
| confirmation | 36 | 1.58 GB | 2.38 GB |

**The archive.** After the run the originals on the machine were deleted. The archive was verified first: 76 988 entries, 102.2 GB, every entry decompressed and compared by SHA-256 (`VERIFICATION.json`). So `AQI_HIL_SERL_complete_2026-10-29` is the only copy of the checkpoints. W&B runs of Q1, the check, the pilot and the confirmation are in the project `hil-serl`.

**Scope decision on the machine.** Stage 0 showed the inconsistent operator cannot be separated
on the original platform (below 20 points in 4 of 6 rotation-off comparisons), so nothing
was trained on `_orig`. The driver now refuses it unless `--allow_orig` is given.

## 8. Q&A log

**Why did the Cuboid train so slowly in all the earlier campaigns?** The policy's near-random
rotation commands twist the peg by about 21° before the operator takes over. The original
operator commands only translation, and it can insert only up to about 8° of yaw. So most
takeovers could not succeed, and the critic saw almost no reward.

**Does this invalidate the finished campaigns?** No. All arms of a campaign shared the condition,
so the paired comparisons still compare the methods under equal conditions. But absolute
convergence times, the size of the effects and possibly the operator dependency itself may change
with an operator who can turn the peg. That is what the comparison of old and new stack on the
first machine has to find out.

**Did the changes speed up training?** Learning per step, yes, by about 4 × for the same method. Training clock time, yes, 2.5 × (77 → 31 minutes), less than the steps (3.4 × fewer at the stop): each step takes about 1.45 × longer now, and the 5-minute pre-training does not shrink with the steps (§5 (b)). The whole cell hardly, because the ruler takes about 55 minutes. Which change does it is open.

**Can the ruler be made shorter without changing the results?** On these cells, yes: skipping the pre-training checkpoints
and scoring only the newest 8 keeps 36 % of the ruler time and leaves all 57 scores unchanged (§5 (c)). Scoring every
second checkpoint is not a shortcut: it changes 46 of 57 scores. The options are in the code since 2026-11-04, off by
default; they have to be chosen before a campaign, and the first machine has to measure them on its own cells first.

**Can these numbers be compared with the first machine?** Stage 0 yes: no learning is involved.
Training numbers no, except as a stated caveat: updates per environment step differ between
machines (finding F3).

**Is the ICC of 0.000 good news?** It says little here. The ruler is near its ceiling, and
seed noise is large, so the between-operator variance estimates to zero. The flat tier means say
more, and only a plain HIL-SERL arm on the new stack can show whether the dependency has gone.

## 9. What the check of 2026-11-03 changed in this log

- **Added:** steps of a success run and twist per row (2); the full tolerance table including novice, tilt and cylinder (3); smoke timeline and checks (4); speed split into three meanings, with a same-method comparison and a clock breakdown per campaign (5); environment, tests, assumptions, interventions, storage and archive (7).
- **Corrected:**
  - **Wall times:** Q1 took 10.09 h, not 9.1 h; the earlier figure missed the clock change that night. The smoke test took 0.41 h, not 0.3 h.
  - **Storage:** up to 2.4 GB per cell, not 1.7 GB.
  - **"The ICC of 0.000 is not new: ensemble extended already had 0.00–0.18":** that left out the 0.84 of the asking experiment.
  - **"About 17 000 steps":** this compared plain HIL-SERL with other arms. The fair same-method figure is 18 000–20 000 → 4 600.
  - **Counts:** 57 cuboid cells plus 1 smoke cell on the cylinder, 58 in all.
