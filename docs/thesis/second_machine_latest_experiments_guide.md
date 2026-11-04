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
| fixed update ratio | the learner makes exactly one update per environment step (before: as many as its GPU allowed) |
| ruler | the final test of a run: every saved checkpoint on 9 fixed starting poses, best 3 in a row |
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
| steps when training stopped | 30 400–33 300 | about 9 000 |
| learner updates per step | about 0.45 | 1 |
| env steps per second (average) | 9.0–9.4 (paced at 10 Hz) | 6.2–6.4 |
| training per run | 76–79 min (20 pre-training + 56–59 online) | about 30 min (5 pre-training + 23–25 online) |
| ruler per run | – | about 55 min (a shorter ruler with the same scores: about 20) |

About four times sooner in steps, but **not attributed**: another machine, another operator and five changes at once. The
ladder (27 runs) will separate them.

**Why the training time fell less than the steps** (2.5 × against 3.4 ×): the 4 × is the first success, and training stops
later; each step takes about 1.45 × longer (one learner update per step on a GPU that also renders the cameras); and the
demonstration pre-training is a fixed 5 000 updates. Fewer steps save about 40 minutes, faster pre-training about 15, the
slower steps cost about 8 back.

**The fixed update ratio.** Before, the robot stepped at 10 Hz and the learner trained as fast as its GPU allowed, so the
learning per step depended on the machine (about 0.45 updates per step on the first machine, 2.5–2.7 on the cluster's H100)
and on the method (E2HiL got about half, its update costs twice as much). Now the learner makes exactly one update per step;
if it falls behind, the robot waits between episodes. Learning per step is then the same everywhere, which makes machines and
methods comparable. Convergence in steps should get faster (more learning per step); in clock time it costs little on the
second machine but about twice the time per step on the first, so it breaks even only if a cell needs about half the steps.
Not measured alone yet: the ladder's settings with and without it will.

**Why the ruler takes longer than training.** It scores every saved checkpoint (one per 1 000 learner steps, the first five
from pre-training), and early checkpoints fail every trial, which runs to the 300-step limit: about 30 000 simulated steps
per run. Skipping the pre-training checkpoints and scoring only the newest 8 keeps 36 % of the time and changes none of the
57 scores; scoring every second checkpoint changes 46. Implemented, off by default, not yet used.

## 2. The sometimes-careless operator

**When the state is drawn.** Once per takeover, at its start, held until the hand-back. With probability 0.7 it stays what it
was last time; otherwise it is drawn anew: careless with probability 0.4. So 40 % of takeovers are careless in the long run,
and two in a row agree 86 % of the time: careless spells last several takeovers.

$$
c_n = \begin{cases} c_{n-1} & \text{prob. } \rho = 0.7 \\ \text{careless} & \text{prob. } (1-\rho)\,b = 0.12 \\ \text{careful} & \text{prob. } (1-\rho)(1-b) = 0.18 \end{cases}
\qquad \pi_{careless} = b = 0.4, \qquad P(c_n = c_{n-1}) = \rho + (1-\rho)\big(b^2 + (1-b)^2\big) = 0.856
$$

Careless spells last $1/0.18 \approx 5.6$ takeovers on average, careful ones $1/0.12 \approx 8.3$, often across episodes.
Code: `operator_model.py` (`Inconsistency`, `inattentive_theta`, `begin_takeover`); every step in the
[log, section 2.1](inconsistent_operator_results_and_qa_log.md).

**What careless changes.**
1. Execution at novice level: motor noise, tremor, lag, speed, pauses, recovery take the novice's values.
2. A premature push in half the attempts.
3. An aim point about **6 mm beside the hole** (1.5 × the 4 mm commit threshold), fixed for the takeover; the cuboid allows 5 mm.

The operator's command each step is $a_t = (1-\lambda)\,a_{t-1} + \lambda\,(u_t + \varepsilon_t + \tau_t)$ with motor noise
$\varepsilon_t \sim \mathcal N(0, (\sigma_0 + k_{sdn}\lVert u_t\rVert)^2)$, tremor $\tau_t$ of size $\sigma_{tr}$ and
$\lVert a_t \rVert \le v_{max}$. Careless sets $\sigma_0, k_{sdn}, \sigma_{tr}, \lambda, v_{max}$, the pauses and the recovery to
the novice anchor (for example $\sigma_0$: expert 0.63, novice 10), $p_{early} = 0.5$, and replaces the small aiming error
$\mathbf b \sim \mathcal N(0, \sigma_{perc}^2 I)$ by $\mathbf b = 1.5\, c_{commit}\, \mathbf d / \lVert \mathbf d \rVert$, a fixed
6 mm offset in a random direction. Every episode starts with the operator's own values, so the demonstrations are always
careful.

**What it does not change:** *when* the operator steps in (its own monitor), so the start of a takeover gives nothing away.

**It matters, without learning** (rotation control on, rescue): expert 100 against 30 %, intermediate 100 against 30 %,
novice 60 against 27 %. Without rotation control even careful rescues fail (13–17 %), so this operator runs only on the new
stack.

**A flaw for novices.** Careless execution uses fixed novice values, so a novice barely changes, and novices with more
tremor than those values are even smoother when careless. In training, careless takeovers hurt the expert (episode success 81
against 59 %) and the intermediate (66 against 33 %), not the novice (23 against 24 %). **The test rests on the upper two
tiers.**

## 3. Do scores spot careless corrections?

Confirmation, 3 357 takeovers, compared within one run, training phase and onset time. A takeover covers steps $t_0..t_1$
($T$ steps), $a^H$ is the operator's action, $a^R$ the policy's at the same state, $\mu_Q$ the mean of the 8 critics, $z(\cdot)$
a running standardisation over the run's takeovers:

| score | formula (higher = more careful) | all | expert / intermediate / novice |
|---|---|---|---|
| behavioural | $B = z(\Delta V) - z(T) - z(F_{peak})$, $\ \Delta V = \mu_Q(s_{t_1}, a^H_{t_1}) - \mu_Q(s_{t_0}, a^R_{t_0})$, $F_{peak} = \max_t F_t$ | **0.66** | 0.71 / 0.67 / 0.52 |
| takeover length alone | $-T$ | 0.65 | 0.69 / 0.68 / 0.51 |
| hindsight | $A_{hind} = \sum_{t=t_0}^{t_e-1} \gamma^{t-t_0} r_t + \gamma^{t_e-t_0} V(s_{t_e}) - V(s_{t_0})$, $t_e = t_0 + T + 10$, $V(s) = \mu_Q(s, a^R)$ | 0.62 | 0.66 / 0.63 / 0.49 |
| counterfactual (critics' score) | $\frac1T \sum_t \big[\mu_Q(s_t, a^H_t) - \mu_Q(s_t, a^R_t)\big]$ | 0.49 | 0.46 / 0.50 / 0.54 |
| smoothness (not available live; circular) | $-\frac1T \sum_t \lVert a^H_t - 2a^H_{t-1} + a^H_{t-2} \rVert$ | 0.76 | 0.86 / 0.85 / 0.40 |

The number is the AUROC, $P(\text{score of a random careful takeover} > \text{score of a random careless one})$, counting only
pairs within the same run, training phase and onset time. Every score's formula, with the yardsticks and the reasons: the
[log, section 3.0](inconsistent_operator_results_and_qa_log.md).

**Reading.** Moderate at best, and mostly length: careless takeovers last longer (44 against 38 steps). The live score
reaches the same 0.66, but none passes the 0.70 bar. The critics' score is at chance.

## 4. How the arms acted on training

All arms: the fixed 8-critic ensemble, the behavioural score, **no outcome filter and no imitation loss** (unlike the first
machine's perfect-score campaign).

| arm | what happens to a takeover |
|---|---|
| keep-all | always enters the demonstration half |
| discard | after the 5 000-step warm-up: dropped from the demonstration half if $B < Q_{0.4}$ of the last 200 scores, its 40th percentile (its steps stay in the online half) |
| re-weight | always kept; replay priority $p_i = (0.05 + \lvert\delta_i\rvert + \max(B, 0)/\bar q)^{0.6}$, so a higher score is replayed more often |
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
