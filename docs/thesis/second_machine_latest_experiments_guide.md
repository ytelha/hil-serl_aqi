# The second machine's experiments — a guide

Everything the second training machine ran (24–29 October 2026, 58 runs, 85.1 hours), with the mechanisms: **rotation
control and the new stack**, the **sometimes-careless operator**, and **how discarding and re-weighting acted on training**.
Details and every table: [second_machine_synthesis.md](second_machine_synthesis.md),
[new_stack_operator_results_and_qa_log.md](new_stack_operator_results_and_qa_log.md),
[inconsistent_operator_results_and_qa_log.md](inconsistent_operator_results_and_qa_log.md).

## The story in three sentences

The synthetic operator of every earlier cuboid run could rarely rescue a twisted peg, because it could move the peg but
not turn it; with rotation control it can, and training starts about four times sooner. Then an operator who is sometimes
careless was tested: a score notices careless corrections only moderately, and discarding or re-weighting by it made no
difference. That test was too weak to decide, and no gate cleared it.

## What ran

| step | experiment | runs | question | answer |
|---|---|---|---|---|
| 1 | operator alone, no training | – | can the operator rescue a twisted peg? | without rotation control 7–13 %, with it 70–100 % |
| 2 | smoke test (cylinder) | 1 | does the new setup run end to end? | yes; all 58 runs finished, none crashed |
| 3 | score validity (`Q1_validity_new`) | 6 | does a live score spot careless corrections? | best 0.61: weak → gate says no-go |
| 4 | perfect-filter check (`headroom_diagnostic_new`) | 9 | could dropping the careless ones help at all? | no clear direction (3 pairs, flips) |
| – | your instruction | – | – | ignore the stopping thresholds, run to the end |
| 5 | discard vs re-weight, pilot and confirmation (`two_arm_*_new`) | 6 + 36 | which use of the score is better? | 93 vs 90 %, p 0.39: no difference |

Steps 3–5 use the careless operator and the fixed 8-critic ensemble on the new stack, cuboid. On the cluster: the
counterfactual-score study (1 run, abandon) and a queued E2HiL test (4 runs).

## Words you need

| word | meaning |
|---|---|
| rotation control | the operator also turns the peg, not only moves it |
| new stack | rotation control + orientation limit + orientation channel + mini-eval stop + fixed update ratio |
| careful / careless | the hidden state of a takeover; careless = sloppy and aimed wrong |
| label | each takeover records whether it was careful or careless; nothing in training reads it |
| behavioural score | a live grade: value rise − takeover length − peak force |
| AUROC | how well a score separates careful from careless (0.5 = coin flip, 0.70 = the bar) |
| warm-up | the first 5 000 training steps, during which no filter acts |
| demonstration half | half of every batch: the demonstrations plus takeovers (as on the first machine) |

## 1. Why the old operator failed, and the new stack

**The problem.** At a takeover the peg is about **21 degrees twisted**. The old operator commanded translation only; the
expert tolerates about **8 degrees** of yaw (100 % up to 4, 50 % at 10, 0 % from 15). So it centred a twisted peg and stayed
stuck at the rim.

**Rotation control.** The operator also turns the peg, towards the nearest orientation that fits (every 90 degrees for the
cuboid), by 30 % of the remaining error per step, and pushes down only once aligned within 0.03 rad. A twist of more than
0.08 rad in contact counts as a jam.

| tier | rescue after 60 random actions, without → with rotation control | alone from a normal start |
|---|---|---|
| expert | 13 → 100 % | 97 → 100 % |
| intermediate | 10 → 100 % | 87 → 100 % |
| novice | 7 → 70 % | 40 → 83 % |

No learning is involved, so this carries over to any machine. It is a **threat to validity** for every finished cuboid
campaign: the comparisons stay fair (every method had the same operator), but the slow start, and possibly part of the
teacher effect, came from it.

**The other four changes** of the new stack: an orientation limit like the real robot (±0.1 rad tilt, ±0.4 rad yaw); a twist
that raises the operator's trouble signal; a stop rule from a small autonomous test (5 poses, threshold 0.6, twice in a row,
every 2 000 steps); and exactly one learner update per environment step.

**Speed,** same method (ensemble extended) on both machines:

| | first machine, old setup | second machine, new stack |
|---|---|---|
| first success without help | 18 300–20 200 steps | 4 600 |
| episodes out of time, first 10 000 steps | 92–93 % | 27 % |
| learner updates per step | about 0.45 | 1 |
| training per run | 76–79 min | about 30 min (+ 55 min ruler ≈ 1.5 h; a shorter ruler with the same scores: ≈ 20 min) |

About four times sooner in steps, but **not attributed**: another machine, another operator and five changes at once. The
ladder (27 runs) will separate them.

## 2. The sometimes-careless operator

**When the state is drawn.** Once per takeover, at its start, held until the hand-back. With probability 0.7 it stays what it
was last time; otherwise it is drawn anew: careless with probability 0.4. So 40 % of takeovers are careless in the long run,
and two in a row agree 86 % of the time: careless spells last several takeovers.

**What careless changes.**
1. Execution at novice level: motor noise, tremor, lag, speed, pauses, recovery take the novice's values.
2. A premature push in half the attempts.
3. An aim point about **6 mm beside the hole** (1.5 × the 4 mm commit threshold), fixed for the takeover; the cuboid allows 5 mm.

**What it does not change:** *when* the operator steps in (its own monitor), so the start of a takeover gives nothing away.

**It matters, without learning** (rotation control on, rescue): expert 100 against 30 %, intermediate 100 against 30 %,
novice 60 against 27 %. Without rotation control even careful rescues fail (13–17 %), so this operator runs only on the new
stack.

**A flaw for novices.** Careless execution uses fixed novice values, so a novice barely changes, and novices with more
tremor than those values are even smoother when careless. In training, careless takeovers hurt the expert (episode success 81
against 59 %) and the intermediate (66 against 33 %), not the novice (23 against 24 %). **The test rests on the upper two
tiers.**

## 3. Do scores spot careless corrections?

Confirmation, 3 357 takeovers, compared within one run, training phase and onset time:

| score | all | expert / intermediate / novice |
|---|---|---|
| behavioural (value rise − length − peak force) | **0.66** | 0.71 / 0.67 / 0.52 |
| takeover length alone | 0.65 | 0.69 / 0.68 / 0.51 |
| hindsight | 0.62 | 0.66 / 0.63 / 0.49 |
| counterfactual (critics' score) | 0.49 | 0.46 / 0.50 / 0.54 |
| smoothness (not available live; circular) | 0.76 | 0.86 / 0.85 / 0.40 |

**Reading.** Moderate at best, and mostly length: careless takeovers last longer (44 against 38 steps). The live score
reaches the same 0.66, but none passes the 0.70 bar. The critics' score is at chance.

## 4. How the arms acted on training

All arms: the fixed 8-critic ensemble, the behavioural score, **no outcome filter and no imitation loss** (unlike the first
machine's perfect-score campaign).

| arm | what happens to a takeover |
|---|---|
| keep-all | always enters the demonstration half |
| discard | after the 5 000-step warm-up: dropped from the demonstration half if its score is in the lowest 40 % of the last 200 (its steps stay in the online half) |
| re-weight | always kept; replayed more often the higher its score |
| perfect filter | after the warm-up: dropped if it was careless |
| random drop | after the warm-up: dropped with probability 0.4 |

**Results.**

| test | runs | result |
|---|---|---|
| keep-all / perfect filter / random drop | 3 pairs | ruler 98.8 / 93.8 / 91.4 %; last checkpoint 96.3 / 100 / 100 %: the order flips, no direction |
| discard vs re-weight, pilot | 6 | 97.5 vs 96.3 % |
| discard vs re-weight, confirmation | 18 pairs | 93.2 vs 90.3 %, −2.9 points, 95 % CI −13.4 to +6.2, p 0.39: no difference |

The discard arm kept novices at 93.8 %, with no outcome filter and no imitation, and with 26 % of their takeovers dropped.

## 5. Why this decides nothing

1. **The filter started late:** the warm-up covered 61 % of all takeovers; half the runs had succeeded alone by then; it
   dropped 17 % instead of 40 %.
2. **Two condition sequences for everything** (a bug, now fixed): every operator with the same seed saw the same
   careful/careless order in all 57 cuboid runs.
3. **Success sits near the ceiling, and it is noisy:** the same expert with the same seed gave 100 and 67 % in two series.
4. **No control at scale:** no keep-all or HIL-SERL arm with 36 runs.
5. **Novices cannot test it** in this model.
6. **No gate cleared it:** the first gate said no-go (a test bug), then go; the second said no-go (score 0.61); you then told
   the automation to ignore the thresholds and run to the end.

## If you remember only five things

1. The old operator could move the peg but not turn it: rescue 7–13 %. With rotation control: 70–100 %. No learning involved.
2. With the new stack, training starts about four times sooner, but which change does it is not yet known.
3. The careless operator: one hidden state per takeover, in spells; careless = novice-level execution, early push, aim 6 mm
   beside the hole. Novices barely change, so the test rests on experts and intermediates.
4. The best live score reaches 0.66, mostly through takeover length; discarding against re-weighting made no difference
   (93 vs 90 %, p 0.39).
5. The test was weak (late filter, shared sequences, ceiling, no control), and no gate cleared it.
