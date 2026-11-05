# Both training machines — summary for the presentation

Status 5 November 2026. One file for the supervisors: what ran on the two training machines, how the mechanisms work
(with formulas and small worked examples), what improved, and the questions asked while preparing. Every symbol is
explained where it first appears and again in the glossary (section 10). Details: the guides
[first_machine_latest_experiments_guide.md](first_machine_latest_experiments_guide.md) and
[second_machine_latest_experiments_guide.md](second_machine_latest_experiments_guide.md).

**How to read the tables.** "93 / 95 / 74" means expert / intermediate / novice. "pp" means percentage points (87 % − 74 %
= 13 pp). "13 / 3 / 2" means 13 wins, 3 losses, 2 ties over 18 pairs of runs with the same operator and seed. "p" is the
chance of a result at least this lopsided if the two methods were really equal (below 0.05 = significant).

---

## 1. The story in one minute

The first training machine answered the four research questions on the original platform (459 runs). The teacher
matters to plain HIL-SERL; a critic ensemble can be made to learn as well as HIL-SERL; no learned score could grade
corrections; and letting the robot ask for help saves human effort and removes the measurable teacher effect. Asking in
the 55 % most uncertain steps beats the same critics with the operator deciding (87 against 74 %) with less help.

A second machine then showed that the synthetic operator of all those runs could rarely rescue a twisted peg, because it
could move the peg but not turn it. With rotation control it can, and training starts about four times sooner. It also
tested an operator who is sometimes careless: a score notices carelessness moderately (0.66), and using the score
showed no difference in a test that was too weak to decide.

Next: check whether the main results hold with the operator who can turn the peg.

## 2. What ran where

| | first training machine | second training machine | RWTH cluster |
|---|---|---|---|
| when | 4 Sep – 31 Oct | 24–29 Oct | 26 Sep, 29 Sep, 5 Nov |
| runs | 459 in 16 campaigns; latest: asking rate (108), perfect grading score (90), more demonstrations (18) | 58 in 5 campaigns, 85.1 hours | method test; 1-run counterfactual study; 4-run E2HiL test (ran 5 Nov and failed technically: in all 4 runs the camera rendering stalled once the policy took over (step 201) and aborted; no result) |
| platform | original: the operator moves the peg but cannot turn it | new stack: rotation control + four other changes (section 5.2) | – |
| learner updates per environment step | about 0.45 | exactly 1 | 2.5–2.7 (H100) |
| latest answers | asking more buys more success; a perfect score collapsed the novices, but three things changed at once (the novice test, 24 runs, decides which) | the operator can now rescue a twisted peg; a score notices a careless operator moderately; discard against re-weight: no difference | the counterfactual score cannot be repaired; the E2HiL test has no result yet |

**Never compare runs across machines.** The learning per step differs (0.45 against 1 update per step), so every
comparison has its reference on its own machine.

**Which results are against plain HIL-SERL** (HIL-SERL = the published method: 2 critics, the operator decides when to help):

| result | against plain HIL-SERL? | outcome |
|---|---|---|
| asks 55 % (robot asks in its 55 % most uncertain steps) | **yes**, same campaign, 18 pairs | **+8.0 pp** (10 / 4 / 4, p 0.18), not yet significant; **intermediates +31 pp**; a fifth less help (41.8 against 52.5 % of the steps) |
| asks 30 % | **yes** | +1.9 pp (p 0.42) with **about half the help** (28.5 against 52.5 %, lower in all 18 pairs) |
| asks 10 % | **yes** | −9.3 pp (p 0.63) with less than a third of the help (15.9 %) |
| random requests | **yes** | −3.1 pp (p 1.00) |
| ensemble extended (8 critics, operator decides) | **yes** | −5.1 pp here (p 1.00); 82 against 84 % (p 0.30) in its own campaign: parity |
| quality replay, learned score (earlier) | **yes** | always below HIL-SERL; −13.0 pp (p 0.031) |
| teacher effect (earlier) | HIL-SERL itself | ICC 0.60 (p 0.035); 0.66 (p 0.001) in the asking-rate campaign |
| perfect grading score, more demonstrations | **no** HIL-SERL arm | reference: ensemble extended |
| everything on the second machine | **no** HIL-SERL arm at scale (one smoke run on the cylinder) | references: keep-all, each other; the speed-up compares ensemble extended across machines |
| E2HiL with the careless operator (cluster, 4 runs) | **yes** by design (E2HiL against HIL-SERL, same operators) | no result yet: the runs failed technically (rendering stall), so nothing was trained |

---

## 3. First machine: how often should the robot ask? (108 runs)

Six arms, the same 9 operators (3 per tier) and 2 seeds, the same 6 demonstrations per operator, cuboid peg, original
platform, cap 35 000 steps.

### 3.1 How asking works, in five steps

1. **How unsure is the robot?** Every step (10 per second), the 8 critics each value the action the robot is about to
   take. Their spread is **σ_Q**. Critics that have seen similar situations agree (small σ_Q); in unfamiliar ones they
   disagree (large σ_Q). σ_Q measures unfamiliarity, not danger.
2. **Is that unusual for this run?** σ_Q is turned into a z-score against its running mean and spread.
3. **Is it among the most uncertain recent steps?** "Asks r %" opens a request when the z-score is above the line that
   only r % of the last 2 000 steps exceed. "Asks 55 %": above the 45th percentile; "asks 10 %": above the 90th.
4. **Timing.** A request lasts 10 steps (1 s), then 5 steps of pause. The robot keeps moving while it asks. No request in
   the first 500 steps.
5. **The operator answers** if its slow, noisy attention notices the request in time (section 3.3).

**Why it was built this way:**
- **A rank line, not a fixed threshold.** σ_Q's size drifts as the critics learn. The first asking experiment used a
  fixed threshold (z > 0.5): it requested about 15 % of the steps (9 % help), a rate nobody chose. A rank line sets
  the rate on purpose, so the arms can be compared at known amounts of asking.
- **A 5-step pause, not 20.** With a 20-step pause the three rates collapsed onto almost the same value.
- **A random arm** asks state-blind at about the rate of asks 30 %, so it tests whether *when* the robot asks matters.
- **Ensemble extended as the main reference:** the same 8 critics, but the operator decides. Only the asking differs.
- **The robot's own next action** goes into σ_Q, because the gate must decide before anyone acts.

### 3.2 Worked example (made-up numbers, real rules)

**1 · How unsure:**

| situation | the 8 critics' values | σ_Q |
|---|---|---|
| familiar: they agree | 5.0, 5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.0 | **0.11** |
| unfamiliar: they disagree | 5.0, 6.1, 4.2, 5.5, 3.9, 6.0, 4.6, 5.3 | **0.75** |

**2 · Unusual for this run:** σ_Q has so far averaged 0.40 with a spread of 0.20, so
z = (0.11 − 0.40) / 0.20 = **−1.4** (calm) and z = (0.75 − 0.40) / 0.20 = **+1.75** (unsure).

**3 · The line:** if the last 2 000 z-values were bell-shaped, asks 10 % / 30 % / 55 % would need z > 1.27 / 0.51 / −0.15.

**4 · The timeline** of one tricky moment (computed with the gate code). `ask` = request on, `-` = pause, `.` = nothing:

```
step         1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20
z         -1.4 -0.9 -0.5  0.0  0.3  0.7  1.1  1.6  2.0  1.8  1.5  1.2  0.9  0.6  0.4  0.2  0.1  0.0 -0.2 -0.4
asks 10 %    .    .    .    .    .    .    .  ask  ask  ask  ask  ask  ask  ask  ask  ask  ask    -    -    -
asks 30 %    .    .    .    .    .  ask  ask  ask  ask  ask  ask  ask  ask  ask  ask    -    -    -    -    -
asks 55 %    .    .    .  ask  ask  ask  ask  ask  ask  ask  ask  ask  ask    -    -    -    -    -    .    .
```

All three ask about the same moment; the lower line starts sooner and also catches milder moments.

**5 · Does the operator answer?** Its attention ẑ rises during the request (0.40, 0.64, 0.78, 0.87, 0.92, …) and falls
after it (0.60, 0.36, …). Without noise:

| tier | needs | takes over |
|---|---|---|
| expert | ẑ above 0.54 for 2 steps, then reacts in 4 | step 6, inside the request |
| intermediate | above 0.67 for 3 steps, then 6 | step 10, the request's last step |
| novice | above 0.80 for 4 steps, then 8 | would need step 14; at step 11 ẑ drops below 0.80 and the reaction is cancelled |

With noise and misses, 87.5 / 54 / 0.4 of 100 isolated requests are answered (simulation); in training 77 / 47 / 2–9.

### 3.3 The formulas

**Request gate** (code: `query_gate.py`):

$$
\sigma_Q(s_t, a^R_t) = \sqrt{\frac18 \sum_{k=1}^{8} \big(Q_k(s_t, a^R_t) - \mu_Q\big)^2}, \qquad
z_t = \frac{\sigma_Q(s_t, a^R_t) - m_t}{s_t}, \qquad
\tau_t = \operatorname{quantile}_{1-r}\big(z_{t-1999}, \dots, z_t\big)
$$

$$
\text{a request opens at step } t \text{ if } z_t > \tau_t,\ \text{no request is on, the pause is over and } t > 500;
\qquad \text{requested share} \le \frac{10}{10 + 5} = 67\,\%
$$

| symbol | meaning |
|---|---|
| $s_t$, $a^R_t$ | the state at step $t$; the action the robot (policy) is about to take |
| $Q_k$, $\mu_Q$ | critic $k$'s value estimate; the mean of the 8 |
| $\sigma_Q$ | the spread of the 8 critics |
| $m_t$, $s_t$ (in $z_t$) | running mean and standard deviation of σ_Q over every step of the run so far (three running numbers, no list) |
| $\tau_t$ | the line: the value that a share $r$ of the last 2 000 z-values exceed, recomputed every step; nothing is fitted |
| $r$ | the target share: 0.10, 0.30 or 0.55 |

**Random arm:** at each step where a request may start, it opens one with probability $p = 0.08$. On average
$(1-p)/p = 11.5$ steps pass first, then 10 on and 5 off, so it requests about $10 / (14 + 1/p) \approx 38\,\%$ of the steps.

**The operator's answer** (code: `operator_gate.py`). The input is $z = 1$ while a request is on, else 0:

$$
\hat z_t = 0.6\, \hat z_{t-1} + 0.4\, z_t + \nu_t, \qquad \nu_t \sim \mathcal N(0, \sigma_{mon}^2)
$$

It takes over once $\hat z_t > \delta_{in}$ for $m_{in}$ steps in a row and a reaction of $l_{react}$ steps has passed,
unless $\hat z$ falls below $\delta_{in}$ first (then the reaction is cancelled). A whole request is missed with
probability $p_{miss}$. It hands back when $\hat z < \delta_{out}$ for $k_{out}$ steps, or after $d_{max}$ seconds.

| tier anchor | $\delta_{in}$ threshold | $m_{in}$ steps above | $l_{react}$ reaction steps | $\sigma_{mon}$ noise | $p_{miss}$ |
|---|---|---|---|---|---|
| expert | 0.538 | 2 | 4 | 0.072 | 0.125 |
| intermediate | 0.669 | 3 | 6 | 0.126 | 0.237 |
| novice | 0.80 | 4 | 8 | 0.22 | 0.35 |

### 3.4 Results

| arm | help (% of steps) | success | expert / intermediate / novice | ICC | vs ensemble extended | vs plain HIL-SERL |
|---|---|---|---|---|---|---|
| HIL-SERL (operator decides) | 52.5 % | 79.4 % | 97.5 / 64.2 / 76.5 | 0.66 | +5.1 (9 / 8 / 1, p 1.00) | – |
| ensemble extended (operator decides) | 53.8 % | 74.3 % | 86.4 / 63.6 / 72.8 | 0.00 | – | −5.1 (p 1.00) |
| asks 10 % | 15.9 % | 70.2 % | 85.8 / 71.0 / 53.7 | 0.00 | −4.1 (9 / 7 / 2, p 0.80) | −9.3 (7 / 10 / 1, p 0.63) |
| asks 30 % | 28.5 % | 81.3 % | 100 / 87.0 / 56.8 | 0.42 | +7.0 (9 / 6 / 3, p 0.61) | +1.9 (9 / 5 / 4, p 0.42) |
| **asks 55 %** | **41.8 %** | **87.4 %** | 93.2 / **95.1** / 74.1 | **0.01** | **+13.2 (13 / 3 / 2, p 0.021)** | +8.0 (10 / 4 / 4, p 0.18) |
| random 30 % | 25.7 % | 76.3 % | 89.5 / 71.6 / 67.9 | 0.00 | +2.1 (11 / 5 / 2, p 0.21) | −3.1 (8 / 9 / 1, p 1.00) |

Help = the share of training steps the operator controlled. Success = the ruler: the frozen policy alone on 9 fixed
starting poses, best 3 checkpoints in a row (27 trials). ICC = the share of the differences between runs that comes from
which operator trained them (0 = none, 1 = all).

**What improved:**
- **More asking, more success:** 70 → 81 → 87 %, at 16 → 29 → 42 % help.
- **Asks 55 % beats the same critics with the operator deciding** by 13 pp with less help: the first significant gain
  over the operator's own judgement in the thesis.
- **Asks 30 % matches plain HIL-SERL with about half the help.**
- **Intermediates gain most:** 95 % under asks 55 %, against 64 % with HIL-SERL (+31 pp). Experts stay at 93–100 % at the two higher rates.
  Novices rise with more asking (54 → 57 → 74 %).
- **The teacher effect disappears:** ICC 0.66 with HIL-SERL, 0.01 with asks 55 %. Part of this is that novices answer few
  requests (section 7).

**Worked example of p** (asks 55 % against ensemble extended): 13 wins, 3 losses, 2 ties. Ties are dropped, leaving 16
pairs. If the arms were equal, each pair is a coin flip; a split of 3 or fewer on either side has probability
$2 \times (1 + 16 + 120 + 560) / 2^{16} = 2 \times 697 / 65\,536 = 0.021$.

**Worked example of ICC:** two operators, two seeds each. A: 90, 94; B: 60, 64. Almost all differences are between
operators: ICC near 1. A: 90, 60; B: 64, 94. The differences are within operators: ICC near 0. The formula is
$\mathrm{ICC} = \sigma^2_{op} / (\sigma^2_{op} + \sigma^2_{seed})$ (operator variance over total variance).

### 3.5 Timing against random

At the same rate, asking at high σ_Q beats random requests by **+4.9 pp** (asks 30 % against random 30 %, 10 / 5 / 3,
p 0.30). The first asking experiment pointed the same way (+14.6 pp, not significant). The direction is positive in
both, but not yet shown; more seeds are planned. Most of the gain may come from asking more, in short structured requests.

---

## 4. First machine: grading corrections with a perfect score (90 runs)

### 4.1 The setup: the original platform, no rotation control

| | first machine (asking rate, perfect score, more demonstrations) | new stack (second machine) |
|---|---|---|
| the operator can turn the peg | **no**: rescue of a twisted cuboid 13 / 10 / 7 % | yes: 100 / 100 / 70 % |
| orientation limit, twist channel | no | yes |
| stop rule | 90 % success over the last 20 unaided training episodes, plus the mini-eval | the mini-eval alone: 5 poses, 0.6, twice in a row |
| learner updates per environment step | about 0.45 | 1 |
| step cap | 35 000 (asking rate); 40 000 (perfect score, more demonstrations) | 30 000–40 000 |
| operator | always careful | sometimes careless |

The comparisons inside each campaign stay fair, because every arm had the same operator. Absolute numbers (steps,
minutes, maybe the size of the teacher effect) do not carry over to the new stack.

### 4.2 How a score enters training

Every learner batch is **half online experience** (everything the robot did, takeovers included) and **half a
demonstration buffer** (the 6 demonstrations plus takeovers). A takeover (from the operator's first step to the
hand-back) is graded once, at the end of its episode.

- **Without grading** (HIL-SERL, ensemble extended, asking): every takeover goes into the demonstration half.
- **With grading:** a takeover first has to pass the **filters**. Here the defaults were on, including the
  **outcome filter**: a takeover from a failed episode is dropped from the demonstration half. A kept takeover gets:

| use | formula | effect |
|---|---|---|
| replay priority | $p = (\epsilon + \lvert \delta \rvert + \eta \max(q, 0))^{\alpha}$ | a positively scored takeover is sampled more often |
| imitation weight | $w = \mathrm{clip}(\mathrm{sigmoid}(2q),\ 0.05,\ 1)$ | $q = 0 \to 0.5$, $q = +1 \to 0.88$, $q = -1 \to 0.12$ |
| imitation loss | $L_{actor} = L_{RL} + 0.1 \cdot \mathrm{mean}(w \cdot L_{imitate})$ | the policy also copies the kept takeovers |

$q$ = the takeover's score; $\delta$ = its TD error (how surprised the critic is); $\epsilon = 0.05$, $\eta = 1$,
$\alpha = 0.6$ (how strongly priority matters); $L_{RL}$ the usual actor loss, $L_{imitate}$ the copying loss.

**Why a perfect score:** every learned score had failed. The oracle uses the simulator's truth, so it is the upper bound
that a learned score could reach: if even it does not help, grading itself is in doubt.

### 4.3 The oracle score, with examples

$$
q_{raw} = \big(z_{task}(\text{takeover}) - z_{task}(\text{hand-back})\big) - \mathbf{1}[\text{jammed at hand-back}] + 0.5 \cdot \text{outcome},
\qquad q = \frac{q_{raw}}{\text{running mean of } \lvert q_{raw} \rvert}
$$

$z_{task}$ = the operator's trouble signal computed on the true state (for example the sideways error relative to the
5 mm clearance); outcome = +1 if the episode succeeded, −1 if not.

- **An expert** takes over 6 mm off (trouble 1.2), centres the peg (0.1), no jam, success:
  $q_{raw} = 1.1 - 0 + 0.5 = 1.6$, high priority, weight near 1.
- **A novice** takes over at 0.8, ends at 0.9 with the peg jammed, failure: $q_{raw} = -0.1 - 1 - 0.5 = -1.6$, lowest
  priority, weight 0.05. In practice the outcome filter had already dropped it.

The oracle never drops a takeover; it only ranks. The result (novices fell from 65 to 32 %) came with the outcome
filter and the imitation loss switched on for the first time, so the novice test (24 runs) separates the three.

---

## 5. Second machine: an operator who can turn the peg, and the new stack

### 5.1 Why the old operator failed

At a takeover the cuboid is on average **21 degrees twisted** (up to 32). The old operator only moved the peg; even the
expert tolerates about **8 degrees** (100 % up to 4, 50 % at 10, 0 % from 15). So it centred a twisted peg and stayed
stuck at the rim. Measured without any learning, 30 episodes each:

| tier | solo: success from a normal start, without → with rotation control | rescue: success after 60 random actions, without → with |
|---|---|---|
| expert | 97 → 100 % | **13 → 100 %** |
| intermediate | 87 → 100 % | **10 → 100 %** |
| novice | 40 → 83 % | **7 → 70 %** |

*Solo* = the operator does a whole episode from a training reset. *Rescue* = it takes over after 60 random actions, a
messed-up, twisted state like a real takeover. The cylinder barely changes (a round peg has no twist to get wrong), which
is why the problem showed only on the cuboid. **Every tier improves; the novice's rescue rises tenfold.**

### 5.2 The five changes of the new stack, and why

| change | what | why |
|---|---|---|
| rotation control | the operator also turns the peg toward the nearest orientation that fits (every 90 degrees), 30 % of the remaining error per step, and pushes down only once aligned within 0.03 rad; a twist above 0.08 rad in contact counts as a jam | without it, rescues of a twisted peg failed |
| orientation limit | tilt within ±0.1 rad, twist within ±0.4 rad, as on the real robot | keeps the simulation close to the hardware |
| twist channel | a twist raises the operator's trouble signal | so the operator notices a twisted peg |
| mini-eval stop | every 2 000 steps, 5 poses without help; stop at ≥ 0.6 twice in a row | a direct test of the policy instead of training episodes |
| fixed update ratio | exactly one learner update per environment step | the same learning per step on every machine and for every method (section 5.5) |

### 5.3 Training starts about four times sooner

Same method (ensemble extended) on both machines:

| | first machine, old setup | second machine, new stack |
|---|---|---|
| first success without help | 18 300–20 200 steps | **4 600** |
| episodes that ran out of time, first 10 000 steps | 92–93 % | 27 % |
| steps when training stopped | 30 400–33 300 | about 9 000 |

**Not attributed:** another machine, another operator and five changes at once. A ladder of 27 runs will add the changes
one at a time to find out which one does it.

### 5.4 Time per run, and why it fell less than the steps

| | first machine | second machine |
|---|---|---|
| demonstration pre-training | about 20 min | about 5 min |
| online training | 30 400–33 300 steps at 9.0–9.4 per second: 56–59 min | about 9 000 steps at 6.2–6.4 per second: 23–25 min |
| training in total | about 77 min | about 31 min |
| ruler (final test) | not measured | about 55 min (a shorter one: about 20, section 8) |

**Worked example:** 3.4 × fewer steps, but each step is 1.45 × slower, and pre-training is a fixed 5 000 updates.
Fewer steps save about 40 minutes, faster pre-training about 15, the slower steps cost about 8 back: 77 − 40 − 15 + 8
≈ 30 minutes, so 2.5 × shorter, not 3.4 ×. The "4 × sooner" is the first success; training stops later.

### 5.5 The fixed update ratio

**What:** two processes run side by side. The actor renders the two cameras, runs the policy and steps the robot; the
learner trains the networks on batches and sends new weights. Now the learner makes exactly **one update per environment
step**. If it falls behind, the actor waits between episodes.

**Why:** before, the actor stepped at 10 Hz and the learner trained as fast as its GPU allowed, so the learning per step
depended on the machine and the method:

| setting | updates per environment step |
|---|---|
| first machine | about 0.45 |
| cluster H100 | 2.5–2.7 |
| E2HiL on the H100 (its update costs twice as much) | about 1.3 |
| new stack, everywhere | 1 |

Same settings, six times more learning per step on one machine than on another, and a costlier method got less learning
for the same budget. Fixing it makes machines and methods comparable.

**Does it slow convergence?** In steps, probably not: each step now gets more learning (about 2.2 × on the first
machine). In minutes it depends on the learner. On the second machine it cost little: the actor waited 0.5–0.7 minutes
per run. On the first machine the learner makes about 5 updates per second, so a step takes about twice as long:
30 000 steps would take about 100 minutes instead of 55, and a run breaks even only if it needs about half the steps. The
ladder measures it.

---

## 6. Second machine: the sometimes-careless operator

### 6.1 How it works

Once per takeover, at its start, the operator is **careful** or **careless**, and stays so until the hand-back. With
probability 0.7 the state stays what it was last time; otherwise it is drawn anew, careless with probability 0.4. So 40 %
of takeovers are careless, and they come in spells, like a person who gets tired or distracted for a while.

A careless takeover:
1. **moves like a novice:** novice motor noise, tremor, lag, speed, pauses and recovery;
2. **pushes down too early** in half the attempts;
3. **aims about 6 mm beside the hole**, more than the 5 mm the cuboid allows.

What stays: *when* the operator steps in (its own monitor), so the start of a takeover gives nothing away; and the
demonstrations, which are always careful. Every takeover records whether it was careless; nothing in training reads that
label except the perfect filter.

**Why:** to test whether a score can find bad corrections, the bad ones must be known. The label gives that ground truth.
Spells make it realistic and keep a score from guessing by training phase alone.

**Worked example:** ten takeovers of one run could come out as careful, careful, careful, careless, careless, careless,
careless, careless, careful, careful: one careless spell of five.

### 6.2 The formulas

**The hidden state** $c_n$ of takeover $n$:

$$
c_n = \begin{cases} c_{n-1} & \text{prob. } \rho = 0.7 \\ \text{careless} & \text{prob. } (1-\rho)\,b = 0.12 \\ \text{careful} & \text{prob. } (1-\rho)(1-b) = 0.18 \end{cases}
\qquad P(\text{careless}) = b = 0.4, \qquad P(c_n = c_{n-1}) = \rho + (1-\rho)\big(b^2 + (1-b)^2\big) = 0.856
$$

Careless spells last $1/0.18 \approx 5.6$ takeovers on average, careful ones $1/0.12 \approx 8.3$.

**The motor model** (each step the operator turns its intent into a command):

$$
a_t = (1-\lambda)\,a_{t-1} + \lambda\,(u_t + \varepsilon_t + \xi_t), \qquad
\varepsilon_t \sim \mathcal N\big(0,\ (\sigma_0 + k_{sdn}\,\lVert u_t \rVert)^2\big), \qquad \lVert a_t \rVert \le v_{max}
$$

**The aim point:** careful $\mathbf b \sim \mathcal N(0, \sigma_{perc}^2 I)$ (small perception error); careless
$\mathbf b = 1.5\, c_{commit}\, \mathbf d / \lVert \mathbf d \rVert$, a fixed 6 mm offset in a random direction ($c_{commit} = 4$ mm).

| symbol | meaning | expert / novice value (careless = novice) |
|---|---|---|
| $\rho$, $b$ | how often the state is kept; the careless share | 0.7; 0.4 |
| $a_t$, $u_t$ | the command sent; the intent (from the path planner) | – |
| $\lambda$ | lag: how fast a command follows the intent | 0.64 / 0.40 |
| $\varepsilon_t$, $\sigma_0$, $k_{sdn}$ | motor noise; its base size; its growth with speed | $\sigma_0$ 0.63 / 10; $k_{sdn}$ 0.03 / 0.10 |
| $\xi_t$, $\sigma_{tr}$ | tremor (a slowly wandering jitter); its size | $\sigma_{tr}$ 1.7 / 9.0 |
| $v_{max}$ | top speed | 79.6 / 55.0 |
| $p_{early}$ | chance of pushing down too early | 0.05 / 0.18, careless 0.5 |
| $\mathbf b$, $c_{commit}$ | the aim offset; the distance at which the operator commits to pushing | – ; 4 mm |

**It matters even without learning** (rescue, rotation control on): careful against careless, expert 100 against 30 %,
intermediate 100 against 30 %, novice 60 against 27 %.

### 6.3 Do scores spot careless takeovers?

A score is one number per takeover, higher for a careful one. It is judged by the **AUROC**: the chance that a random
careful takeover scores higher than a random careless one (0.5 = coin flip, 1 = perfect; the bar fixed in advance: 0.70).
Pairs are only compared within the same run, training phase and onset time, so a score cannot win by timing.

**Worked example of AUROC:** careful takeovers score 2, 1, 0; careless ones 1.5 and −1. Of the 6 careful–careless
pairs, the careful one is higher in 4 (2 > 1.5, 2 > −1, 1 > −1, 0 > −1): AUROC = 4 / 6 = 0.67.

Confirmation run, 3 357 takeovers:

| score | AUROC | expert / intermediate / novice |
|---|---|---|
| behavioural (the one both arms used) | **0.66** | 0.71 / 0.67 / 0.52 |
| takeover length alone | 0.65 | 0.69 / 0.68 / 0.51 |
| hindsight | 0.62 | 0.66 / 0.63 / 0.49 |
| counterfactual (the critics' score) | 0.49 | 0.46 / 0.50 / 0.54 |
| smoothness (not available live; circular) | 0.76 | 0.86 / 0.85 / 0.40 |

**What it shows:** the behavioural score notices carelessness moderately, mostly through length (careless takeovers last
44 against 38 steps). It works for experts (0.71) and intermediates (0.67), not for novices (0.52, section 7). The
critics' score is at chance, which confirms why it was abandoned.

### 6.4 The score formulas

A takeover covers steps $t_0 \dots t_1$, so it lasts $T = t_1 - t_0 + 1$ steps. $a^H_t$ is the operator's action, $a^R_t$
the robot's at the same state, $\mu_Q$ the mean of the 8 critics, $z(\cdot)$ a running standardisation over the run's
earlier takeovers.

| score | formula | in words |
|---|---|---|
| behavioural | $B = z(\Delta V) - z(T) - z(F_{peak})$, with $\Delta V = \mu_Q(s_{t_1}, a^H_{t_1}) - \mu_Q(s_{t_0}, a^R_{t_0})$, $F_{peak} = \max_t F_t$ | value gained, minus length, minus peak force: a short, value-raising, gentle correction scores high |
| hindsight | $A_{hind} = \sum_{t=t_0}^{t_e-1} \gamma^{t-t_0} r_t + \gamma^{t_e-t_0} V(s_{t_e}) - V(s_{t_0})$, $t_e = t_0 + T + 10$, $V(s) = \mu_Q(s, a^R)$ | what the robot reached 10 steps after getting control back ($\gamma = 0.97$, $r_t$ the reward) |
| counterfactual | $\frac1T \sum_t \big[\mu_Q(s_t, a^H_t) - \mu_Q(s_t, a^R_t)\big]$ | how much better the critics rate the operator's move than the robot's; fails because the robot is trained to maximise these same critics |
| smoothness | $-\frac1T \sum_t \lVert a^H_t - 2a^H_{t-1} + a^H_{t-2} \rVert$ | jerky commands score low; circular, because careless is novice jitter by construction |

**Worked example of $B$** (each term already standardised):
- careful: value gain $z = +1.0$, shorter than usual $z(T) = -0.5$, normal force $z = 0$: $B = 1.0 + 0.5 - 0 = 1.5$;
- careless: value gain $z = -0.5$, longer $z(T) = +1.2$, harder push $z = +0.8$: $B = -0.5 - 1.2 - 0.8 = -2.5$.

**How the arms used $B$:** *discard* drops a takeover from the demonstration half if $B$ is below the 40th percentile of
the last 200 scores (after a 5 000-step warm-up); *re-weight* keeps every takeover and replays it more often the higher
$\max(B, 0)$. No outcome filter and no imitation loss, unlike the first machine. Result: 93.2 against 90.3 % (p 0.39), no
difference. The discard arm kept novices at 93.8 % while dropping 26 % of their takeovers.

---

## 7. The synthetic novice is too weak

Several results depend on how weak the synthetic novice is. This is a property of the operator model, not proven for
real people, and it may change in the future.

| where | what the novice does |
|---|---|
| asking | answers 2–9 of 100 ten-step requests: its attention needs until step 14, the request ends at 10. Takeovers per run fall from 206 to 85; its share of the steps from 47 to 13 % |
| the old platform | rescues a twisted peg in 7 % of the tries (experts 13 %) |
| the perfect-score campaign | 97 % of its takeovers fall in failed episodes (experts 67 %), so the outcome filter left about 7 novice takeovers per run, against about 200 for an expert |
| the careless operator | careless execution uses novice values, so a careless novice barely changes (episode success 23 against 24 %); scores cannot test novices (AUROC 0.52) |

**Consequences:** part of the vanished teacher effect under asking is that novice corrections drop out of the data;
the careless-operator test rests on experts and intermediates.

**What may change:**
- **A held request** stays on until the operator answers or a timeout expires, so a slow operator can answer. It is
  implemented, not yet run.
- **Rotation control** already lifts the novice's rescue from 7 to 70 %.
- **Calibrating the novice against human data** (detection delay and patience separately) is a future step; the first
  machine also has a milder novice variant.

---

## 8. Questions asked while preparing, with short answers

| question | answer |
|---|---|
| Why does the ruler take about 55 minutes when training takes about 30? | It scores every saved checkpoint (one per 1 000 learner steps, the first five from pre-training) on 9 poses. Early checkpoints fail every trial, and a failed trial runs to the 300-step limit: about 30 000 simulated steps per run, three times the training. |
| Can it be shorter? | Yes: skip the pre-training checkpoints and score only the newest 8. That keeps 36 % of the time (about 20 minutes) and changes none of the 57 scores. Scoring every second checkpoint would change 46 of 57, so it is not used. Implemented, off by default, to be checked on the first machine first. |
| The steps fell to about a quarter, why did training time only halve? | The quarter is the first success; training stops later (3.4 × fewer steps). Each step is 1.45 × slower, and pre-training is fixed (section 5.4). |
| What is the fixed update ratio and why was it made? | One learner update per environment step, so learning per step no longer depends on the GPU or the method (section 5.5). |
| Does it make convergence slower? | In steps probably not; in minutes little on the second machine, about twice per step on the first (section 5.5). |
| What do solo and rescue mean? | Solo: the operator does a whole episode from a normal start. Rescue: it takes over after 60 random actions, from a twisted, messed-up state (section 5.1). |
| What do validity, perfect-filter check, pilot and confirmation mean? | Four steps of the careless-operator study. Validity (6 runs): does a live score spot careless takeovers? (0.61, weak.) Perfect-filter check (9 runs): could dropping exactly the careless ones help at all? (no clear direction, 3 pairs.) Pilot (6 runs): a small first run of discard against re-weight. Confirmation (36 runs): the same at thesis scale (93 against 90 %, p 0.39). |
| How was the careless behaviour implemented? | A hidden state per takeover, kept with probability 0.7, else redrawn (careless 0.4); careless = novice movement, early push, aim 6 mm off (section 6). |
| What do the scores mean? | One number per takeover, higher = more careful; formulas and examples in sections 6.3–6.4. |
| Did the first machine's latest experiments use the old setup, without rotation control? | Yes. Its code is patches 0001–0010 plus local features; rotation control came in patch 0014 (section 4.1). |
| How does the asking rate work? | Sections 3.1–3.3, with the worked example in 3.2. |
| Is σ_Q stored for every step, or fitted to a normal distribution? | For the decision, no list: the z-score keeps three running numbers (count, mean, spread) over the whole run. The line keeps the last 2 000 scores. Nothing is fitted: the line counts. Every step's σ_Q is also written to a log file, for analysis only. |
| Does this make the threshold adaptive? | Yes, it is recomputed every step. Replayed on logged σ_Q from the second machine: in a run where σ_Q fell from 0.0147 to 0.0085, the asks-55 % line fell from 0.0131 to 0.0061. A frozen line would have put 28 % of the later steps above it instead of 55 %. |
| Does "asks 55 %" request 55 % of the steps? | No. About 55 % of the recent steps are above the line; each crossing buys 10 steps, so about 57 % are requested (replay), and the operator's answers turn that into 42 % help. |
| What did we learn from it? | A fixed line gives an accidental rate (it drifted from 19 to 36 % within a run in the replay); a rank line holds it (53–60 % in every run). A set rate makes the arms comparable. With one signal the z-score hardly matters. And a rank line never stops asking, even when the robot is confident: fine for a comparison, but a deployed system needs a floor or a falling rate. |

## 9. Likely questions from the supervisors

| question | short answer |
|---|---|
| Does asking beat plain HIL-SERL? | Not yet significantly: +8 pp (p 0.18), mostly intermediates (+31). It beats the same critics with the operator deciding (+13 pp, p 0.021), and asks 30 % matches HIL-SERL with half the help. |
| Is the gate's timing better than random? | It points that way in both asking experiments (+15 and +5 pp) without significance; more seeds are planned. |
| Is the ICC of 0 under asking real? | It is measured correctly, but partly mechanical: novices answer few short requests. The held request tests this. |
| Did the perfect score prove that grading is wrong? | No: score, outcome filter and imitation loss changed at once. The novice test (24 runs) decides which one hurt. |
| Can I compare the speed-up across machines? | Only as the size of what can be gained: same method, about 19 000 → 4 600 steps, but a different machine, operator and five settings. |
| Why is 0.66 not "valid"? | The bar fixed in advance was 0.70. |
| Why two machines, and why not the cluster? | The first ran the thesis campaigns; the second was available for a week for the new stack and the careless operator. The cluster quota is 3 GPU-hours a month; a campaign needs 80–135. |
| What runs next? | The novice test, then the new stack on the first machine, its 27-run ladder, and the repeat at thesis scale with plain HIL-SERL, ensemble extended and asking; the 4-run E2HiL test is repeated once the rendering stall on the cluster is fixed, or run on a training machine. |

## 10. Glossary

**Words**

| word | meaning |
|---|---|
| HIL-SERL | the published method: reinforcement learning with demonstrations and human corrections, 2 critics, the operator decides when to help |
| critic | a network that predicts how much reward to expect from a state and an action |
| ensemble extended | HIL-SERL with 8 critics; the operator still decides when to help (the reference for asking and grading) |
| asks r % | the robot asks when its σ_Q is in the top r % of the last 2 000 steps |
| request / answered | the robot asks for 10 steps (1 s) / the operator actually takes over during it |
| takeover, hand-back | the operator takes control / gives it back |
| tier | expert, intermediate, novice: three skill levels of the synthetic operator (3 operators each) |
| cell, arm, campaign | one training run / one method in a comparison / a set of arms × operators × seeds |
| ruler | the final test: every saved checkpoint, 9 fixed starting poses, no operator; best 3 checkpoints in a row |
| demonstration half | half of every learner batch: the 6 demonstrations plus takeovers |
| outcome filter | drops a takeover from the demonstration half if its episode failed |
| imitation loss | the policy also copies kept takeovers (weight 0.1) |
| new stack | rotation control, orientation limit, twist channel, mini-eval stop, fixed update ratio |
| mini-eval | a small test during training: 5 poses without help, every 2 000 steps |
| solo, rescue | the operator alone from a normal start / taking over after 60 random actions |
| careful, careless | the hidden state of a takeover on the second machine |
| warm-up | steps before a mechanism acts: 500 for the asking gate, 5 000 for the discard filter |
| pp | percentage points |

**Symbols**

| symbol | meaning |
|---|---|
| $Q_k$, $\mu_Q$, $\sigma_Q$ | critic $k$'s value; the mean of the 8; their spread (how unsure the robot is) |
| $s_t$, $a^R_t$, $a^H_t$ | state at step $t$; the robot's action; the operator's action |
| $z_t$, $m_t$, $s_t$ | the z-score of σ_Q; its running mean and spread |
| $r$, $\tau_t$ | the target share of asking; the line (a rank among the last 2 000 steps) |
| $p$ (random arm) | probability of starting a request at a free step (0.08) |
| $\hat z_t$, $\sigma_{mon}$ | the operator's smoothed, noisy attention; its noise |
| $\delta_{in}$, $m_{in}$, $l_{react}$, $p_{miss}$ | threshold, steps above it, reaction steps, chance of missing a request |
| $\delta_{out}$, $k_{out}$, $d_{max}$ | hand-back threshold, steps below it, time cap |
| $q$, $q_{raw}$, $z_{task}$ | a takeover's score; before scaling; the operator's trouble signal |
| $\delta$, $\epsilon$, $\eta$, $\alpha$ | TD error; base priority 0.05; score weight 1; priority exponent 0.6 |
| $w$ | imitation weight of a kept takeover |
| $c_n$, $\rho$, $b$ | careful/careless state of takeover $n$; keep probability 0.7; careless share 0.4 |
| $a_t$, $u_t$, $\lambda$ | command; intent; lag |
| $\varepsilon_t$, $\sigma_0$, $k_{sdn}$, $\xi_t$, $\sigma_{tr}$, $v_{max}$ | motor noise and its sizes; tremor and its size; top speed |
| $p_{early}$, $\mathbf b$, $c_{commit}$ | early push chance; aim offset; commit distance (4 mm) |
| $t_0$, $t_1$, $T$ | first and last step of a takeover; its length |
| $B$, $\Delta V$, $F_{peak}$ | behavioural score; value gained; peak force |
| $A_{hind}$, $\gamma$, $r_t$, $V$ | hindsight score; discount 0.97; reward; the critics' value at the robot's action |
| AUROC | chance that a careful takeover scores higher than a careless one (0.5 chance, 0.70 the bar) |
| ICC | share of the differences between runs that comes from the operator |
| p | chance of a result at least this lopsided if the methods were equal (exact sign test over pairs) |
| W / L / T | wins / losses / ties over the paired runs |
