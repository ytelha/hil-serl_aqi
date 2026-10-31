# The operator that can turn the peg, and the new stack — results & QA log

Companion to [second_machine_synthesis.md](second_machine_synthesis.md). This log covers
what was measured on the **second training machine** without or before the quality
question: the operator alone in the simulator, the orientation tolerance, the smoke test
and how fast training runs on the new stack.

- Machine: `LSPEM499`, one RTX 5090 (32 GB), 24 CPU threads, repo at commit `d191f9d`
  (patches 0001–0019), Python 3.12.3, JAX 0.11.0, MuJoCo 3.10.0: the first machine's exact
  package lock.
- Dates: 24–29 October 2026.
- Raw results: `presentation_bundle/results/second_machine_stage0/` and
  `presentation_bundle/results/S_smoke/`.
- Tools: `examples/oracle_insertion_test.py`, `examples/synthetic_suite.py stage0`.

---

## 0. Headline

**The synthetic operator of every finished cuboid campaign could rarely rescue a twisted
peg.** With rotation control it can. Success of a takeover after 60 random actions rises
from 13 / 10 / 7 % (expert / intermediate / novice) to 100 / 100 / 70 %. The cause is
measured: the peg is about 21° twisted at the takeover, and without rotation control the
expert tolerates about 8° of yaw.

**On the new stack the slow start of the cuboid is gone:** the first success without help
comes after about 4 200–5 400 environment steps, against about 17 000 in the first
machine's logs. This compares two machines and five changes at once, so it shows the size
of what can be gained, not which change gains it. The ladder that separates them
(`synthetic_suite` pilots) has **not** been run.

---

## 1. The new stack

"New stack" means all of these switched on, with the settings of `L5_full`:

| change | patch | what it does |
|---|---|---|
| rotation control | 0014 | the operator also turns the tool to the nearest insertable orientation |
| orientation box | 0016 | the task clips the commanded orientation: ±0.1 rad roll and pitch, ±0.4 rad yaw |
| orientation channel | 0016 | a twist raises the operator's trouble signal |
| mini-eval convergence | 0015 | the autonomous mini-eval decides when a cell stops (threshold 0.6, 2 passes in a row, 5 poses) |
| fixed update ratio | 0015 | 1 learner step per environment step, so learning per step does not depend on the GPU |

The "original" platform is the operator and task of the finished campaigns.

## 2. The operator alone (no training)

30 episodes per row, cuboid and cylinder. *Solo* starts from a training reset; *rescue*
takes over after 60 random actions. No learning is involved, so these numbers should carry
over to any machine with the same package versions.

| peg | tier | solo, rotation off → on | rescue, rotation off → on | twist at takeover |
|---|---|---|---|---|
| cuboid | expert | 96.7 → 100 % | **13.3 → 100 %** | 21.0° |
| cuboid | intermediate | 86.7 → 100 % | **10.0 → 100 %** | 21.0° |
| cuboid | novice | 40.0 → 83.3 % | **6.7 → 70.0 %** | 21.0° |
| cylinder | expert | 100 → 100 % | 96.7 → 100 % | 15.0° |
| cylinder | intermediate | 100 → 100 % | 86.7 → 100 % | 15.0° |
| cylinder | novice | 90.0 → 83.3 % | 70.0 → 76.7 % | 15.0° |

The cylinder barely changes, as expected: a round peg has no yaw to get wrong. This is why
the problem only showed on the cuboid.

## 3. Orientation tolerance (no training)

10 episodes per angle, starting centred over the bore with a set twist.

| expert, cuboid, rotation off | 0–4° | 6–8° | 10° | 12° | ≥ 15° |
|---|---|---|---|---|---|
| success at that yaw | 100 % | 90 % | 50 % | 30 % | 0 % |

- **With rotation control, the expert succeeds at every angle tested (0–22°),** in yaw and
  in tilt.
- **Yaw tolerance of the expert: 8°** (0.1396 rad), by the rule "largest angle with at least
  80 % of the success at 0°". That is well below the geometric 19.6° of a perfectly centred
  peg, and below the placeholder limit of 0.25 rad.
- **Tilt tolerance: at least 22°.** The value 0.384 rad written to `limits.json` is only the
  top of the sweep, not a measured limit. With the orientation box at ±0.1 rad the tilt part of
  the channel practically cannot fire; the channel works through yaw.
- **Novice tolerance reads 0°** because the novice's success at 0° is already only 50 %, with
  10 noisy episodes per angle. It does not mean a novice cannot insert.

## 4. Smoke test

One cell: plain HIL-SERL, expert, cylinder, full stack. All checks passed (box in every
process, orientation monitor, fixed ratio in both processes, takeover channels, mini-eval,
scored). Ruler 93 %, first unaided success at 1 221 steps, converged at 4 286 steps, operator
share 28 %. Not a result, as intended.

## 5. Training speed on the new stack

All cells below are cuboid, new stack, inconsistent operator (see
[inconsistent_operator_results_and_qa_log.md](inconsistent_operator_results_and_qa_log.md)).

| | first machine, original platform (B1/B2 W&B logs) | second machine, new stack (36 cells) |
|---|---|---|
| first success without help | ≈ 17 000 steps (median over the plain HIL-SERL cells) | 4 174 / 5 442 steps (median, discard / re-weight arm) |
| episodes running into the 300-step limit, first 10 000 steps | 90–99 % | 29 / 34 % |
| success in the first 10 000 steps | about 7 % at 75–90 % operator control | 71 / 66 % |
| converged | 15 of 18 (`logged`, oracle campaign) | 36 of 36 (at mini-eval 0.6) |

**Read this with care.** The left column has a consistent operator and asynchronous updates
on another GPU. The right column has an inconsistent operator, a fixed update ratio and a
different stop rule. The direction is very clear, and the operator being inconsistent makes
it more remarkable, not less. But it does not say which change does the work.

Skill still shows, in time rather than in outcome. Final success is flat across tiers in the
discard arm (92.6 / 93.2 / 93.8 %, against 94 / 78 / 65 % for ensemble extended in the first
machine's oracle campaign), but novices need 1.5–1.8 times the steps of experts (12 616 vs
6 946 and 11 060 vs 7 514). The ICC of 0.000 is **not** new: ensemble extended already had
0.00–0.18 on the old stack; only plain HIL-SERL is high (0.66). Whether plain HIL-SERL stays
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

- **Tests:** 136 passed on the real stack. Two test bugs appeared and were reported rather
  than hidden; both are fixed in patch 0021:
  - `test_run_passes_limits...` failed once real stage-0 limits existed, because `read_limits`
    fell back on them;
  - `test_suite_configs` failed after the intended score edit in the two-arm configs.
- **Stability:** all 57 cuboid training cells converged. No crash, no traceback, no repeated
  cell. The fixed update ratio worked between real processes with no deadlock.
- **Cost:** about 1.5 h per cell, up to 1.7 GB per cell (checkpoints included).
- **A non-fatal error in every learner:** one `CUDA_ERROR_OUT_OF_MEMORY` allocating 4 GiB at the
  start of demo pre-training (`XLA_PYTHON_CLIENT_MEM_FRACTION` 0.6 on 32 GB). Training continues
  normally. Worth watching on a smaller GPU.
- **Scope decision on the machine:** stage 0 showed the inconsistent operator cannot be separated
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

**Can these numbers be compared with the first machine?** Stage 0 yes: no learning is involved.
Training numbers no, except as a stated caveat: updates per environment step differ between
machines (finding F3).

**Is the ICC of 0.000 good news?** It says little here. The ruler is near its ceiling, and
seed noise is large, so the between-operator variance estimates to zero. The flat tier means say
more, and only a plain HIL-SERL arm on the new stack can show whether the dependency has gone.
