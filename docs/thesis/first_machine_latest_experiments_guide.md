# The first machine's latest experiments — a guide

Only the newest runs of the first training machine (October 2026): **how often the robot should ask**, **a perfect
grading score**, **more demonstrations**, and the **novice test** that comes next. Everything earlier is in
[first_machine_full_history/](first_machine_full_history/). Details and every table:
[first_machine_asking_rate_results_and_qa_log.md](first_machine_asking_rate_results_and_qa_log.md) and
[first_machine_oracle_quality_results_and_qa_log.md](first_machine_oracle_quality_results_and_qa_log.md).

## The story in three sentences

Letting the robot ask for help when its critics disagree works, and asking more often works better: asking in the 55 %
most uncertain steps beats the operator's own judgement (87 against 74 %) with less help. Grading corrections with a
*perfect* score from the simulator then collapsed the novices, but that campaign also switched on a filter that threw away
almost every novice correction, and an imitation loss. Which of the three did it is what the novice test decides.

## What ran

| when | experiment (config) | runs | question | answer |
|---|---|---|---|---|
| to 21 Oct | how often to ask (`active_ratematch`) | 108 | does asking more buy more success, and does it beat the operator deciding? | yes, and yes at the high rate (p 0.021); vs plain HIL-SERL +8, not significant |
| to 29 Oct | perfect grading score (`aqi_oracle_quality`) | 90 | does grading corrections with a perfect score help weak operators? | novices collapse, but three things changed at once |
| to 31 Oct | more demonstrations (`quality_starvation_probe`) | 18 | do 20 demonstrations instead of 6 rescue the novice? | no |
| next | novice test (`quality_novice_ablation`) | 24 | which of score, filter and imitation collapses the novice? | ready, not run |

All on the original platform (the operator moves the peg but cannot turn it), cuboid peg, the fixed 8-critic ensemble,
3 operators per tier × 2 seeds, the same 9 operators in all four.

## The setup they ran on: the original platform, no rotation control

The first machine's code is patches 0001–0010 plus its own local features (the quantile trigger, random asking, the oracle
score, milder novices, the HiRIL fix). Rotation control came later, in patch 0014, and the copy of its code from 31 October
does not contain it at all. So none of these campaigns had any part of the new stack:

| | first machine (asking rate, perfect score, more demonstrations) | new stack (second machine) |
|---|---|---|
| the operator can turn the peg | **no**: rescue of a twisted cuboid 13 / 10 / 7 % (expert / intermediate / novice) | yes: 100 / 100 / 70 % |
| orientation limit, twist channel | no | yes |
| stop rule | 90 % success over the last 20 untouched training episodes, with the mini-eval (0.6) as an extra check | the mini-eval alone: 5 poses, 0.6, twice in a row |
| learner updates per environment step | free-running, about 0.45 (the robot paced at 10 Hz) | fixed at 1 |
| step cap | 35 000 (asking rate); 40 000 (perfect score, more demonstrations) | 30 000–40 000 |
| operator | always careful (consistent) | sometimes careless |

**What it means for reading them:**
- **The comparisons inside each campaign stay fair.** Every arm had the same operator, so "asking beats the operator
  deciding" and "the perfect-score arms collapse the novice" hold as paired comparisons on this platform.
- **The operator's weakness shapes the numbers, especially for novices.** At a takeover the peg is about 21 degrees twisted,
  and without rotation control even the expert tolerates only about 8, so most takeovers could not rescue it. In the
  asking-rate campaign 97 % of a novice's takeovers sat in failed episodes, so the outcome filter of the perfect-score
  campaign threw away almost all novice corrections. With an operator who can turn the peg that picture may change.
- **Absolute numbers do not carry over:** steps to converge, minutes, and possibly the size of the teacher effect may differ
  on the new stack. Never compare these runs one by one with the second machine's.

**What comes next on this machine** (`handoff/first_machine/HANDOFF_first_machine.md`):
1. **The novice test stays on the original platform, on purpose.** It pairs with the perfect-score campaign's cells, so it
   runs on the current code before any patch.
2. **Then the new stack is installed:** patches 0011–0021, and 0037 for the shorter ruler.
3. **Then the ladder (27 runs)** adds the changes one at a time, starting from this original platform, so it shows which
   change (rotation control or another) removes the slow start on this machine, and what the fixed update ratio costs here.
   After it, the old-against-new comparison at thesis scale repeats plain HIL-SERL, ensemble extended and asking with the
   chosen stack.

## Words you need

| word | meaning |
|---|---|
| critic | a network that predicts how much reward to expect from a state and an action |
| σ_Q | how much the 8 critics disagree about the robot's next move: how unfamiliar the situation is, not how bad |
| request | the robot asks for help for 10 steps (1 s) |
| answered | the operator actually takes over during a request |
| ensemble extended | the 8 critics, with the operator deciding when to help (the reference) |
| demonstration half | every learner batch is half online experience, half a buffer of the 6 demonstrations plus takeovers |
| quality oracle | a grade per takeover from the simulator's truth (section 2) |
| outcome filter | throws a takeover out of the demonstration half if its episode failed |
| imitation loss | the policy also copies the takeovers in the demonstration half (weight 0.1) |
| ruler / ICC | the final exam (9 start positions, no operator) / how much of the success differences comes from the operator |

## 1. How often should the robot ask? (108 runs)

**How the asking works.**
1. At every step the 8 critics each value the robot's *next* action; **σ_Q** is the spread of their 8 answers.
2. σ_Q becomes a z-score against its running mean.
3. "Asks 55 %": a request opens when the current score is above the **45th percentile of the last 2 000 steps** ("asks 10 %":
   above the 90th). The rate is set directly, so it does not drift as the critics converge.
4. The request lasts **10 steps**, then a **5-step pause**. The robot keeps moving while it asks.
5. The synthetic operator gets "1" while a request is on. It smooths that signal, waits until it crosses its threshold, then
   reacts after a delay. **A slow, noisy novice often reacts too late:** the request is over first. Measured answered share:
   experts about 77 %, intermediates 47 %, novices 2–9 %.

**Random control:** a coin flip (probability 0.08 per step) opens requests with the same 10-step / 5-step timing, so only the
*timing* differs from "asks 30 %".

**In formulas** (`query_gate.py`, `operator_gate.py`; every step in the
[asking-rate log, sections 1–4](first_machine_asking_rate_results_and_qa_log.md)):

$$
\sigma_Q(s, a) = \mathrm{std}_{k=1..8}\, Q_k(s, a), \qquad
z_t = \frac{\sigma_Q(s_t, a^R_t) - m_t}{s_t}, \qquad
\text{request opens if } z_t > Q_{1-r}\big(z_{t-1999}, \dots, z_t\big)
$$

with $m_t, s_t$ the running mean and standard deviation over the run, $r$ = 0.10 / 0.30 / 0.55, no request in the first
500 steps (the simulation's warm-up), then 10 steps on and 5 off, so at most $10/15 = 67\,\%$ of the steps are requested. The random arm requests
about $10 / (14 + 1/0.08) \approx 38\,\%$ of the steps. The line is recomputed at every step from the last 2 000
scores, so it follows σ_Q's drift; no distribution is fitted (a worked example with numbers, and what the gate keeps
in memory: [asking-rate log, sections 3.1–3.2](first_machine_asking_rate_results_and_qa_log.md)). The operator sees the request through
$\hat z_t = 0.6\, \hat z_{t-1} + 0.4\, z_t + \mathcal N(0, \sigma_{mon}^2)$, and takes over after $\hat z > \delta_{in}$ for
$m_{in}$ steps plus a reaction of $l_{react}$ steps, unless $\hat z$ drops below $\delta_{in}$ first. Run through the real gate,
the tier anchors answer an isolated 10-step request:

| tier anchor | threshold $\delta_{in}$ / steps above / reaction | takeover without noise | requests answered |
|---|---|---|---|
| expert | 0.54 / 2 / 4 | step 6 | 87.5 % |
| intermediate | 0.67 / 3 / 6 | step 10, the request's last | 54.3 % |
| novice | 0.80 / 4 / 8 | never: it would need step 14, but the request ends at 10 | 0.4 % |

The novice's reaction is cancelled because its smoothed signal falls below the threshold as soon as the request ends.

**Results.**

| arm | operator share of the steps | success | vs ensemble extended | vs HIL-SERL |
|---|---|---|---|---|
| HIL-SERL | 52.5 % | 79.4 % | +5.1 (p 1.00) | – |
| ensemble extended | 53.8 % | 74.3 % | – | −5.1 (p 1.00) |
| asks 10 % | 15.9 % | 70.2 % | −4.1 (p 0.80) | −9.3 (p 0.63) |
| asks 30 % | 28.5 % | 81.3 % | +7.0 (p 0.61) | +1.9 (p 0.42) |
| **asks 55 %** | **41.8 %** | **87.4 %** | **+13.2 (13 / 3 / 2, p 0.021)** | +8.0 (10 / 4 / 4, p 0.18) |
| random 30 % | 25.7 % | 76.3 % | +2.1 (p 0.21) | −3.1 (p 1.00) |

- **The teacher effect:** HIL-SERL ICC 0.66 (p 0.001); asks 55 % ICC 0.01. ICC = $\sigma^2_{op} / (\sigma^2_{op} + \sigma^2_{seed})$,
  the share of the spread between cells that comes from the operator. The p-values of the comparisons are exact sign tests
  over the 18 pairs of the same operator and seed.
- **By tier**, asks 55 % against HIL-SERL: experts −4, intermediates **+31**, novices −2 points. The gain is one tier.
- **Timing against random** at the same rate: +4.9 points, p 0.30, not shown.

**Who really answers** (from the per-takeover table of this campaign):

| | takeovers per run, expert | takeovers per run, novice | novice takeovers that answered a request |
|---|---|---|---|
| the operator's own monitor | 615 | 206 | – |
| asks 55 % | 936 | 85 | 60 % |

Experts help *more* when asked (64 % of the steps against 61 % on their own); novices barely answer (13 % against 47 %). So
the "less help" is the novices missing requests, and part of the vanished teacher effect is that novice corrections dropped
out of the data.

## 2. A perfect grading score (90 runs)

**How a score enters training.** A takeover is graded once, at the end of its episode.
- In the arms **without** grading, every takeover goes into the demonstration half of every batch.
- In the **grading** arms, a takeover first has to pass the filters. If it does, the score sets its **replay priority**
  (sampled more often when positive) and its **imitation weight**, and the **imitation loss** (weight 0.1) makes the policy
  copy it.

**What the quality oracle computed** for each takeover, from the simulator's truth:

    q = (trouble at the takeover − trouble at the hand-back) − (1 if the peg is jammed at the hand-back) + 0.5 × (+1 if the episode succeeded, −1 if it failed)

"Trouble" is the operator's own trouble signal, computed on the true state (for example, lateral error relative to the 5 mm
clearance). q is then divided by its running average size. Examples:
- an expert takes over 6 mm off (trouble 1.2), centres the peg (0.1), no jam, success: q = 1.1 + 0.5 = **1.6**, high priority;
- a novice takes over at 0.8, ends at 0.9, jammed, failure: q = −0.1 − 1 − 0.5 = **−1.6**, lowest priority.

The oracle **never drops** a takeover; it only ranks. It uses information no real system has, so it is an upper bound.

**What else was switched on.** The grading arms left the filters at their defaults. One of them, the **outcome filter**,
throws out every takeover of a failed episode, *before* the oracle sees it. And the **imitation loss** of 0.1 was on, only in
these arms. **Every earlier grading campaign had both switched off**; this one was the first with both on.

**Why that matters for novices.** In the asking-rate campaign (same operators, same platform), 97 % of a novice's
takeovers fall in failed episodes, against 67 % of an expert's. So the filter leaves a novice run with about **7**
takeovers in its demonstration half, an expert run with about **200**.

**Results** (from the machine's log; its raw data has not come back):

| arm | when help comes | grading | success | ICC | expert / intermediate / novice |
|---|---|---|---|---|---|
| ensemble extended | operator decides | none | 79.0 % | 0.18 | 94 / 78 / **65** |
| asks 30 % | robot asks | none | 82.9 % | 0.23 | 100 / 91 / 57 |
| quality, oracle | operator decides | oracle + filter + imitation | 73.3 % | 0.63 | 97 / 91 / **32** |
| asks + oracle | robot asks | oracle + filter + imitation | 78.8 % | 0.93 | 100 / 100 / **36** |
| asks + critics' score | robot asks | critics' score + filter + imitation | 73.4 % | 0.97 | 100 / 100 / **10** |

**Reading.** Experts and intermediates rise to the ceiling, novices collapse, so the teacher effect explodes. The machine's
log concludes "even a perfect score amplifies dependence". That does not follow yet: the critics' score, known to be
inverted, collapses novices in the same way, and the two share the filter and the imitation loss, not the score.

## 3. More demonstrations (18 runs)

Novices only, the same three, 20 demonstrations instead of 6; everything else as before.

| arm | novice, 6 demonstrations | novice, 20 demonstrations |
|---|---|---|
| ensemble extended | 65.4 % | 60.5 % |
| asks 30 % | 57.4 % | 64.2 % |
| asks + oracle | 36.4 % | 27.2 % |

**Reading.** More of the novice's *own* demonstrations do not rescue it. But the filter and the imitation loss stayed on, so
this does not separate the three suspects either.

## 4. Next: the novice test (24 runs, ready)

Novices only, on the same code; two reference arms copied, so every new run pairs with an old one.

| arm | score | outcome filter | imitation |
|---|---|---|---|
| ensemble extended (copied) | – | – | – |
| quality, oracle (copied) | oracle | on | 0.1 |
| oracle without filter | oracle | off | 0.1 |
| oracle without filter and imitation | oracle | off | 0 |
| filter only | none | on | 0 |
| imitation only | none | off | 0.1 |

| if | then the cause is |
|---|---|
| filter only collapses, oracle without filter recovers | the outcome filter |
| imitation only collapses, oracle without imitation recovers | copying novice corrections |
| both harmless, the oracle alone collapses | the oracle's priority: the log's conclusion stands |

A one-second count first: what the campaign's filter actually dropped. About 1.5 days on the first machine.

## What it means

1. **Asking is the positive result of the thesis,** at the high asking rate. It is not yet significantly better than plain
   HIL-SERL, and its gain comes from intermediates.
2. **Part of asking's "independence" is the operator model:** novices miss short requests. A held request, which waits until
   the operator answers, is implemented and not yet run.
3. **Grading is not ruled out by the perfect score,** because the campaign changed three things at once. The thesis words the
   result only after the novice test.

## If you remember only five things

1. σ_Q = how much 8 critics disagree about the robot's next move; the robot asks when it is in the top share of recent steps.
2. Asking in the 55 % most uncertain steps: 87 against 74 % success with less help (p 0.021); against HIL-SERL +8, not significant.
3. Experts answer most requests, novices almost none; that drives both the saving and the vanished teacher effect.
4. The quality oracle = trouble removed − jam + outcome, from the simulator. It only ranks; it never drops.
5. The outcome filter dropped about 97 % of novice corrections and an imitation loss was on; the novice test decides which
   part collapsed the novices.
