# How often should the robot ask? The critics' disagreement σ_Q and the asking-rate campaign — results & QA log

Companion to [first_machine_synthesis.md](first_machine_full_history/first_machine_synthesis.md). This log explains **what σ_Q is, how it becomes
a request for help, how a synthetic operator answers**, and what the asking-rate campaign `active_ratematch` (thesis D3)
showed. It is written from the code, the campaign config and the report, and adds measurements made here from the
report's per-takeover table (section 6).

- Machine: the first training machine, original platform (the operator moves the peg but cannot turn it), cuboid.
- Campaign: `runs/active_ratematch`, 108 cells, finished 2026-10-21. Config `examples/campaigns/active_ratematch.yaml`.
- Report: `OneDrive/active_ratematch_report/` = `presentation_bundle/results/active_ratematch/` (`dependency.md`,
  `quality_audit/summary.md` and `segments.csv` with 37 565 takeovers, `plots/`).
- Code: `serl_launcher/aqi/uncertainty.py` (σ_Q), `query_gate.py` (the request), `operator_gate.py` (the operator's answer),
  `runtime.py` (the per-step loop).
- The machine's own log of the same campaign: [active_ratematch_results_and_qa_log.md](active_ratematch_results_and_qa_log.md).

---

## 0. Headline

- **Asking more buys more success.** The robot asking in the 10, 30 and 55 % most uncertain steps reaches 70.2, 81.3 and
  87.4 % ruler success at 15.9, 28.5 and 41.8 % operator share of the steps.
- **Asking in the 55 % most uncertain steps beats the same critics with the operator deciding:** 87.4 against 74.3 %,
  +13.2 points, 13 wins / 3 losses / 2 ties, p = 0.021, with less help (41.8 against 53.8 %).
- **Against plain HIL-SERL it is +8.0 points, not significant** (p = 0.18), with a fifth less help; almost all of the
  gain is among intermediates.
- **The teacher hardly shows when asking most:** ICC 0.01, against 0.66 for HIL-SERL (p = 0.001).
- **Why there is less help, and why the teacher disappears:** experts answer the robot more than they would step in on
  their own (63.8 against 60.5 % of the steps); novices answer little (13.4 against 46.9 %). Novices make about 85
  takeovers per run under the gate against about 936 for experts, and only 38–60 % of their takeovers answer a request.
- **The timing of σ_Q beats random requests at the same rate by 4.9 points,** not significantly (p = 0.30).

## 1. The critics' disagreement σ_Q

**What a critic is.** A critic is a network that predicts Q(s, a): how much reward to expect from state s if the robot
takes action a now and then follows its policy. HIL-SERL has 2 critics. The ensemble extension has **8** (K = 8), trained
on the same data but with different initialisations and their own learning targets (`per_member_targets`, one randomly
chosen member as target, M = 1).

**What σ_Q is.** At every step the runtime asks all 8 critics for the value of the action the robot is about to take,
a_robot, and computes

$$
\mu_Q(s, a) = \frac1K \sum_{k=1}^{K} Q_k(s, a), \qquad
\sigma_Q(s, a) = \sqrt{\frac1K \sum_{k=1}^{K} \big(Q_k(s, a) - \mu_Q(s, a)\big)^2}, \qquad K = 8
$$

(`uncertainty.py`; `runtime.observe_step` calls it with the robot's own action, also during a takeover, where a_robot is
what the policy would have done.)

**What it means.** Where the critics have seen much data they agree, and σ_Q is small. In unfamiliar states, or for
unfamiliar actions, each critic extrapolates differently, and σ_Q is large. So σ_Q measures how little the robot *knows*
about its next move (epistemic uncertainty), not how badly things are going.

**What it is not.** The validity campaign (D1) compared σ_Q with the simulator's ground truth: it does not detect a jam
(AUROC 0.51 in the contact zone). It also fades as training proceeds, because the critics converge and agree more
everywhere (finding F6). That is why the gate uses a **relative** threshold (section 2), and why the question became
"does asking at high σ_Q train a better policy?" instead of "does σ_Q detect trouble?".

## 2. From σ_Q to a request

`query_gate.py`, configured by `trigger_terms: [uncertainty]` and `trigger_combiner: max`:

1. **Normalise.** z = (σ_Q − running mean) / running standard deviation, over all steps of the run so far.
2. **Score.** With one term, the score A(s_t) is simply z.
3. **Threshold.** With `query_target_rate` r, the threshold is the (1 − r) quantile of the last 2 000 scores. So
   "asks 55 %" requests whenever the current score is above the 45th percentile of the recent scores, "asks 10 %" only
   above the 90th. The rate is set explicitly and does not depend on σ_Q's drifting scale.
4. **Hold and pause.** A request stays on for 10 steps (1 s, `query_min_on_steps`), then the gate waits 5 steps
   (`query_cooldown_steps`) before it can ask again. So at most 10 of every 15 steps can be requested (67 %).
5. **The robot keeps moving** while it asks (`query_behavior: continue`); the operator overrides the moment it steps in.

**The gate in formulas** (`query_gate.ActiveQueryGate.step`, once per control step, 10 steps per second). With the robot's
own next action $a^R_t$:

$$
z_t = \frac{\sigma_Q(s_t, a^R_t) - m_t}{s_t + \epsilon}, \qquad
A_t = \lambda_{uq}\, z_t = z_t, \qquad
\tau_t = Q_{1-r}\big(A_{t-1999}, \dots, A_t\big)
$$

- $m_t, s_t$: running (Welford) mean and standard deviation of $\sigma_Q$ over every step of the run so far, including this
  one; $z_t = 0$ for the first 10 steps.
- $A_t$: the request score. With the single term `uncertainty`, combiner `max`, $\lambda_{uq} = 1$ and no workload term, it
  is just $z_t$.
- $\tau_t$: the empirical $(1 - r)$ quantile of the last 2 000 scores, used once 10 are in; $r$ = `query_target_rate`
  (0.10, 0.30 or 0.55). Without $r$ the threshold would be the fixed $\tau_A = 0.5$.

A request **opens** at step $t$ if no request is on, the cool-down is over, $t > 500$ (the gate's warm-up: 500 steps in the simulation config
`sim_peg_insertion/config.py`, 1 000 is the code default), and
$A_t > \tau_t$. It then stays on for exactly 10 steps ($t, \dots, t+9$, `query_min_on_steps`), followed by 5 steps without a
request (`query_cooldown_steps`). About a share $r$ of all recent steps lie above $\tau_t$, because $\tau_t$ is that
quantile of the recent scores. At the steps where a request may start the chance is lower, because unsure stretches
are already covered by a request (section 3.2). A breach of the hard force or torque limit opens or extends a request whatever the score (a safety
override, reason `safety`). The 10-on, 5-off cycle caps the requested share of the steps:

$$
\text{requested share} \;\le\; \frac{10}{10 + 5} = 67\,\%
$$

**The random control** (`random-q30`) has no trigger term, so its score is always 0. It asks by a coin flip with
probability 0.08 at every step where a request may start, with the same 10-step hold and 5-step pause. That gave a
similar operator share to "asks 30 %" (25.7 against 28.5 %), so the two differ only in *when* they ask.

In formulas: $A_t \equiv 0$ never exceeds a threshold, and at every eligible step a request opens with probability
$p = 0.08$ (`random_request_rate`). The step that opens a request is its first requested step, so on average
$(1 - p)/p = 11.5$ steps pass without a request before it. One cycle is then about 11.5 waiting, 10 on and 5 off:

$$
\text{requested share} \approx \frac{10}{10 + 5 + (1 - p)/p} = \frac{10}{14 + 1/p} = \frac{10}{26.5} \approx 38\,\%, \qquad
p = \frac{1}{10/f - 14} \ \text{ for a target share } f
$$

Replaying the real gate with $p = 0.08$ on 9 logged runs gives 37.9 % (37.4–38.6 %; section 3.2). An earlier version of
this log wrote $10 / (15 + 1/p) \approx 36\,\%$, which counts the opening step twice.

The config chose $p = 0.08$ to match the roughly 34 % that asks 30 % reached in the probe.

**Why the quantile.** In the first asking experiment (D2) the gate used a fixed threshold, τ_A = 0.5 on the z-score.
Because z is normalised, it fired at a rate nobody chose (about 9 % of the steps). A 3-hour probe before this campaign
measured the rates of the quantile gate (intermediate operator, 8 000 steps):

| setting | operator share in the probe |
|---|---|
| fixed threshold τ_A = 0.5 | 3.7 % |
| asks 10 % (cool-down 5) | 14.1 % |
| asks 30 % | 33.9 % |
| asks 55 % | 43.6 % |
| the operator's own monitor | 59.6 % |

With the default cool-down of 20 steps the three rates collapsed onto one (the 10-of-30 duty cycle caps them), so the
campaign uses a cool-down of 5.

## 3. From a request to a takeover: how the synthetic operator answers

In the asking arms the operator does not watch the task itself. Its takeover gate gets z = 1 while a request is on and
0 otherwise (`operator_gate.py`), and decides as a signal detector would:

- it smooths the input with lag and adds monitoring noise (ẑ = EWMA(z) + N(0, σ_mon));
- it takes over once ẑ has stayed above its threshold δ_in for m_in steps, after a further reaction delay;
- it misses whole events with probability p_miss, and sometimes steps in for no reason (false alarms at rate ρ_fa);
- it hands back when ẑ falls below δ_out for k_out steps, or at its time cap d_max.

**The operator's answer in formulas** (`operator_gate.SyntheticInterventionGate.update`). The stimulus is $z_t = 1$ while a
request is on and $0$ otherwise; the operator sees it through a smoothed, noisy estimate

$$
\hat z_t = (1 - \lambda)\,\hat z_{t-1} + \lambda\, z_t + \nu_t, \qquad \nu_t \sim \mathcal N(0, \sigma_{mon}^2), \qquad \lambda = 0.4
$$

- **Detection:** once $\hat z_t > \delta_{in}$ for $m_{in}$ steps in a row, the event is missed with probability $p_{miss}$;
  otherwise a reaction countdown of $l_{react}$ steps starts. If $\hat z$ falls below $\delta_{in}$ before it ends, the
  countdown is **cancelled**.
- **Takeover** when the countdown reaches zero.
- **Hand-back** once $\hat z_t < \delta_{out}$ for $k_{out}$ steps in a row, or after $d_{max}$ seconds in control.
- **False alarms:** a takeover for no reason with probability $\rho_{fa}\, \Delta t / 60$ per step ($\rho_{fa}$ per minute,
  $\Delta t = 0.1$ s).

Without noise, after $n$ request steps $\hat z = 1 - 0.6^n$: 0.40, 0.64, 0.78, 0.87, 0.92, 0.95, 0.97, 0.98, 0.99, 0.99. That
is enough to see why a 10-step request suits experts and loses novices. With the tier anchors, run through the real gate
(computed here on 2026-11-04, 20 000 isolated 10-step requests per tier):

| tier anchor | $\delta_{in}$ | $m_{in}$ | $l_{react}$ | $\sigma_{mon}$ | $p_{miss}$ | takeover without noise | requests answered |
|---|---|---|---|---|---|---|---|
| expert | 0.538 | 2 | 4 | 0.072 | 0.125 | step 6 | 87.5 % |
| intermediate | 0.669 | 3 | 6 | 0.126 | 0.237 | step 10, the request's last | 54.3 % |
| novice | 0.80 | 4 | 8 | 0.22 | 0.35 | never: it would need step 14, but at step 11 $\hat z$ drops to 0.60 and the countdown is cancelled | 0.4 % |

The novice's own monitoring is slower on every count: a higher threshold, more steps above it, a longer reaction, more
noise and more misses. The campaign's operators are jittered around these anchors, and requests often come in trains, so
the logs show 77 / 47 / 2–9 % (below). The direction is the same.

Novices have more lag, more noise and slower reactions. **A 10-step request often ends before a novice's smoothed
signal crosses its threshold**, and a pending reaction is cancelled when the signal drops. On the W&B logs of the first
asking experiments the share of requests answered was about 77 % for experts, 47 % for intermediates and 2–9 % for
novices. This is a property of the synthetic operator model, and it matters for every result below.

### 3.1 A worked example, step by step

One tricky moment in a run, followed through the gate and the three kinds of operator. The σ_Q values are made up so that
the arithmetic is easy (in the logs σ_Q is about 0.005–0.03). The rules and the timelines are from running the real
`query_gate.py` and `operator_gate.py` on these numbers (computed here on 2026-11-04).

**Step 1 – How unsure is the robot?** The 8 critics value the action the robot is about to take. σ_Q is their spread.

| situation | the 8 critics' values | σ_Q |
|---|---|---|
| familiar: they agree | 5.0, 5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.0 | **0.11** |
| unfamiliar: they disagree | 5.0, 6.1, 4.2, 5.5, 3.9, 6.0, 4.6, 5.3 | **0.75** |

**Step 2 – Is that unusual for this run?** Say σ_Q has so far averaged 0.40 in this run, with a typical spread of 0.20.

- familiar: z = (0.11 − 0.40) / 0.20 = **−1.4**, calmer than usual;
- unfamiliar: z = (0.75 − 0.40) / 0.20 = **+1.75**, much less sure than usual.

**Step 3 – Unusual enough to ask?** The gate ranks the current z among the last 2 000 steps (200 seconds):

| arm | asks when this step is more uncertain than … | the line, if the last 2 000 z were bell-shaped |
|---|---|---|
| asks 10 % | 1 800 of the last 2 000 steps | z > 1.27 |
| asks 30 % | 1 400 of them | z > 0.51 |
| asks 55 % | 900 of them | z > −0.15 |

Real σ_Q is lopsided (many calm steps, a few very unsure ones), so the real lines sit lower. In a logged run at step
2 000 they were +0.56, −0.27 and −0.70 (section 3.2).

**Step 4 – The timing rule: 10 steps on, 5 off.** The robot meets a tricky moment; z rises from −1.4 to 2.0 and falls
again. `ask` = request on, `-` = forced pause (it may not ask), `.` = no request:

```
step         1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20
z         -1.4 -0.9 -0.5  0.0  0.3  0.7  1.1  1.6  2.0  1.8  1.5  1.2  0.9  0.6  0.4  0.2  0.1  0.0 -0.2 -0.4
asks 10 %    .    .    .    .    .    .    .  ask  ask  ask  ask  ask  ask  ask  ask  ask  ask    -    -    -
asks 30 %    .    .    .    .    .  ask  ask  ask  ask  ask  ask  ask  ask  ask  ask    -    -    -    -    -
asks 55 %    .    .    .  ask  ask  ask  ask  ask  ask  ask  ask  ask  ask    -    -    -    -    -    .    .
```

- Asks 55 % crosses its line first (step 4, z 0.0 > −0.15), asks 30 % at step 6 (0.7 > 0.51), asks 10 % at step 8
  (1.6 > 1.27). All three ask about the same tricky moment; the low line simply starts sooner and also catches the
  milder moments.
- A request stays on for 10 steps (1 second) even when z drops, then 5 steps of pause.
- So at most 10 of every 15 steps can be requested: in 1 000 steps (100 seconds) at most 66 requests and 667 requested
  steps. The first 500 steps of a run never ask (the simulation's warm-up).

**Step 5 – The random arm.** No σ_Q: at every step where a request may start, a coin comes up "ask" with probability
0.08. On average 11.5 steps pass before it does; then 10 steps on and 5 off. One cycle is about 26.5 steps, 10 of them
requested: **about 38 %** of the steps, close to asks 30 %. Only *when* it asks differs.

**Step 6 – Does the operator answer?** While a request is on, the operator's input is 1. Its attention ẑ builds up
slowly (0.40, 0.64, 0.78, 0.87, 0.92, …, 0.99) and decays when the request ends (0.60, 0.36, 0.22). Each tier needs a
level, held for some steps, and then reaction time. Without noise:

| step | ẑ | expert: above 0.54 for 2 steps, reacts in 4 | intermediate: above 0.67 for 3, reacts in 6 | novice: above 0.80 for 4, reacts in 8 |
|---|---|---|---|---|
| 1 | 0.40 | – | – | – |
| 2 | 0.64 | above, 1 of 2 | – | – |
| 3 | 0.78 | 2 of 2 → **reacting** | above, 1 of 3 | – |
| 4 | 0.87 | reacting | 2 of 3 | above, 1 of 4 |
| 5 | 0.92 | reacting | 3 of 3 → **reacting** | 2 of 4 |
| 6 | 0.95 | **TAKES OVER** | reacting | 3 of 4 |
| 7 | 0.97 | | reacting | 4 of 4 → **reacting** (would take over at step 14) |
| 8–9 | 0.98–0.99 | | reacting | reacting |
| 10 | 0.99 | | **TAKES OVER**, on the request's last step | reacting |
| 11 | 0.60 | | | request over; ẑ below 0.80 → **reaction cancelled** |

With noise and the chance of missing a request (20 000 simulated requests per tier):

| tier | answers, of 100 requests | why |
|---|---|---|
| expert | **87.5** | only its 12.5 % chance of missing; 4 steps to spare |
| intermediate | **54** | 24 % misses, and one step of delay from noise is already too late |
| novice | **0.4** | always 4 steps too slow; the few cases are late takeovers just after the request ended |

In the campaign, with operators spread around these anchors, the logs show 77, 47 and 2–9 answers per 100 requests.

**Step 7 – Over a whole run (asks 55 % against the operator's own monitor, section 6).** An expert takes over 936 times
per run instead of 615 and controls 63.8 % of the steps instead of 60.5 %. A novice takes over 85 times instead of 206
and controls 13.4 % instead of 46.9 %. The "less help" of asks 55 % is the novices' missing answers.

### 3.2 What the gate keeps in memory, and how its line moves

**What is stored.**

- **For the z-score: three numbers, not a list.** A running count, mean and sum of squared deviations (Welford's method,
  `normalization.RunningZScore`) over every step since the run started. Each new σ_Q updates them and is then dropped.
  This memory never forgets: step 30 000 is still compared with an average that includes step 1.
- **For the line: the last 2 000 scores, as a list** (`deque(maxlen=2000)`), about 200 seconds of robot time. Each new
  score pushes the oldest one out. Every step goes in: during requests, pauses and takeovers too.
- **For the analysis only:** σ_Q, z, the score and the request flag of every step are written to `steps.jsonl`
  (`log_every_step`, on by default). The gate never reads that file.

**No normal distribution is fitted.** The z-score only shifts and rescales σ_Q. The line is the empirical quantile of the
2 000 stored scores (`np.quantile`): it counts, it does not assume a shape. The bell-shaped lines in section 3.1 are an
illustration only.

**The line is adaptive: it is recomputed at every step** from the last 2 000 scores. Below, the real gate replays real
σ_Q traces: 9 runs of the second machine's headroom diagnostic (inconsistent-operator suite, quality mode, 5 700–11 900
steps; `runs/inconsistent_operator/headroom_diagnostic_new/aqi_logs/*/steps.jsonl` in the 2026-10-29 export). These are
not the asking campaign's runs (its raw logs stay on the first machine), and the replay is open loop: the requests do
not change the data. It shows the mechanism, not a result.

Asks 55 %, its line translated back into σ_Q:

| run | typical σ_Q (median), steps 1 000–2 000 → 4 000–5 000 | line at step 2 000 | at 4 000 | at 6 000 |
|---|---|---|---|---|
| expert run, σ_Q falls | 0.0147 → 0.0085 | 0.0131 | 0.0061 | 0.0083 |
| novice run, σ_Q rises | 0.0060 → 0.0092 | 0.0045 | 0.0072 | 0.0088 |

A line fixed at its step-2 000 value would have gone wrong both ways. In the expert run, σ_Q above 0.0131 at steps
4 000–5 000: 28 % of the steps instead of 55 %. In the novice run, σ_Q above 0.0045: 88 %.

Requested share of the steps after the first 1 000 (the replay uses the code-default warm-up of 1 000 steps), mean over
the 9 runs (range across runs):

| rule | requested | first 2 000 steps → last 2 000 | of all recent steps above the line | chance at a step where a request may start |
|---|---|---|---|---|
| fixed line z > 0.5 (the old τ_A) | 28.8 % (19.8–36.0) | 18.6 → 36.0 % | – | 5.0 % |
| asks 10 % | 23.8 % (20.0–29.0) | 19.9 → 26.4 % | 11.5 % | 3.6 % |
| asks 30 % | 42.9 % (40.2–45.3) | 41.1 → 44.4 % | 33.4 % | 10.8 % |
| asks 55 % | 57.1 % (53.1–59.6) | 58.5 → 56.5 % | 59.3 % | 29.2 % |
| random, p = 0.08 | 37.9 % (37.4–38.6) | 35.5 → 37.7 % | – | 8.1 % |

Reading the table:

- **"Asks r %" holds what it promises:** about r of the recent steps are above the line (11.5 / 33.4 / 59.3 %).
- **The requested share is different:** 24 / 43 / 57 %. Each crossing buys 10 steps of request, even when z drops back,
  and the pause skips some crossings. A brief crossing of the 10 % line therefore costs 10 steps.
- **At a step where a request may start, the chance is lower than r** (3.6 / 10.8 / 29 %). Unsure stretches are already
  covered by a request, so the free steps are mostly calm ones.
- **The fixed line drifts:** its share doubles within a run (18.6 → 36 %) and differs between runs (20–36 %). On the
  first machine the same rule gave about 9 % in D2. The rank line keeps asks 55 % at 53–60 % in every run, early and late.

**What this taught us.**

1. **A fixed line gives an accidental rate**, set by σ_Q's shape and drift: about 9 % on the first machine, 20–36 % on
   the second machine's traces. A rank line sets the rate on purpose.
2. **A set rate makes the arms comparable** at known budgets. That is what produced the clean ladder 70 → 81 → 87 %
   success at 16 → 29 → 42 % operator share.
3. **With one signal the z-score matters little.** Within 2 000 steps the running mean and spread barely move, so ranking z
   is almost the same as ranking σ_Q: the same above/below decision at 92 % of the steps (at least 88 % in every run of
   the replay). The z-score matters when several signals are added.
4. **The label is not the help.** "Asks 55 %" means 55 % of recent steps are above the line. The 10-on, 5-off timing turns
   that into the requested share, and the operator's answers into the operator share (41.8 % on the first machine).
5. **A rank line always asks.** Even a confident robot has a "most uncertain 10 %", so the requests never fade as the
   policy improves. That is fair for a comparison, but a deployed system would need a floor (never ask below a fixed
   σ_Q) or a falling rate. It is also one reason the gated arms see fewer autonomous episodes and converge less often
   (section 5.4).
6. **The timing rules shape the rate as much as the line:** with a 20-step pause the three rates collapsed onto one
   (section 2).

## 4. Design

| arm | what decides when the operator helps |
|---|---|
| `hilserl` | plain HIL-SERL (2 critics); the operator's own monitor |
| `logged` = ensemble extended | the fixed 8-critic ensemble; the operator's own monitor |
| `active-q10` / `q30` / `q55` | the same ensemble; the robot asks in the 10 / 30 / 55 % most uncertain steps |
| `random-q30` | the same ensemble; random requests at the rate of q30 |

Same 9 operators (3 per tier, sample seed 0), 2 seeds, the same demonstrations (6 per operator) and the same ruler for
all six arms: 108 cells. Cap 35 000 environment steps (40 000 in D2). The ensemble arms start from 5 000 steps of
demonstration pre-training with learning rate 1.5e-4; HIL-SERL stays plain. The operator's corrections come from the
geometric oracle (path planner).

**How the results are measured.** A *cell* is one arm × operator × seed; each arm has 18 cells (9 operators × 2 seeds).

- **Ruler success** $SR_c$: the frozen policy's success on the 9 fixed starting poses, best mean over 3 consecutive
  checkpoints (`best_window`), resolution $1/27 \approx 3.7$ points.
- **Operator share** of a cell: $\dfrac{\text{steps the operator controlled}}{\text{all environment steps of the training}}$.
- **Paired comparison** of an arm with a reference: $d_c = SR^{arm}_c - SR^{ref}_c$ over the 18 cells with the same
  operator and seed; $W, L, T$ count $d_c > 0$, $< 0$ and $= 0$. Exact two-sided sign test over the $n = W + L$ untied pairs:

  $$
  p = \min\Big(1,\; 2 \sum_{k=0}^{\min(W, L)} \binom{n}{k}\, 0.5^{\,n}\Big)
  $$

  For example asks 55 % against ensemble extended, 13 / 3 / 2: $n = 16$, $p = 2 \cdot 697 / 65\,536 = 0.021$.
- **ICC** (how much of the spread between cells is the operator): the one-way random-effects ICC(1,1) over operator groups
  of per-seed success rates,

  $$
  \mathrm{ICC} = \frac{\sigma^2_{op}}{\sigma^2_{op} + \sigma^2_{seed}}, \qquad
  \sigma^2_{op} = \max\Big(\frac{MS_{between} - MS_{within}}{k_0},\, 0\Big), \qquad
  \sigma^2_{seed} = MS_{within}
  $$

  with the ANOVA mean squares between and within operators and $k_0$ the mean group size (2 here). 0 means the operator
  explains nothing; 1 means everything.
- **p (operator matters):** a permutation test. Shuffle the operator labels across the arm's 18 cells, recompute the
  variance of the operator means, and count how often it reaches the observed one: $p = (\#\{\text{shuffled} \ge
  \text{observed}\} + 1) / (B + 1)$.
- **CV:** the standard deviation of the 9 operator means over their mean.

## 5. Results

### 5.1 Success and human effort

| arm | operator share of the steps | ruler success | vs ensemble extended (W/L/T, p) | vs HIL-SERL (W/L/T, p) |
|---|---|---|---|---|
| HIL-SERL | 52.5 % | 79.4 % | +5.1 (9/8/1, 1.00) | – |
| ensemble extended | 53.8 % | 74.3 % | – | −5.1 (8/9/1, 1.00) |
| asks 10 % | 15.9 % | 70.2 % | −4.1 (9/7/2, 0.80) | −9.3 (7/10/1, 0.63) |
| asks 30 % | 28.5 % | 81.3 % | +7.0 (9/6/3, 0.61) | +1.9 (9/5/4, 0.42) |
| **asks 55 %** | **41.8 %** | **87.4 %** | **+13.2 (13/3/2, 0.021)** | +8.0 (10/4/4, 0.18) |
| random 30 % | 25.7 % | 76.3 % | +2.1 (11/5/2, 0.21) | −3.1 (8/9/1, 1.00) |

Paired per cell (same operator and seed), exact sign test. Asks 30 % needs less help than HIL-SERL in all 18 pairs;
asks 55 % needs less than ensemble extended in 13 of 18 (p = 0.096). Dose–response: asks 30 vs 10 % +11.1 (p = 0.057),
55 vs 30 % +6.2 (p = 1.00).

### 5.2 By tier

| arm | expert | intermediate | novice |
|---|---|---|---|
| HIL-SERL | 97.5 | 64.2 | 76.5 |
| ensemble extended | 86.4 | 63.6 | 72.8 |
| asks 10 % | 85.8 | 71.0 | 53.7 |
| asks 30 % | 100.0 | 87.0 | 56.8 |
| asks 55 % | 93.2 | 95.1 | 74.1 |
| random 30 % | 89.5 | 71.6 | 67.9 |

Asks 55 % minus HIL-SERL by tier: −4.3 / **+30.9** / −2.4 points. Asks 30 %: +2.5 / +22.8 / −19.7.

### 5.3 Does the teacher show?

| arm | ICC | p (operator matters) | spread of operator means (CV) | worst operator |
|---|---|---|---|---|
| HIL-SERL | **0.662** | **0.001** | 0.29 | int00, 27.8 % |
| ensemble extended | 0.000 | 0.61 | 0.18 | int02, 55.6 % |
| asks 10 % | 0.000 | 0.50 | 0.30 | int01, 38.9 % |
| asks 30 % | 0.421 | 0.12 | 0.28 | nov02, 37.0 % |
| asks 55 % | 0.010 | 0.38 | 0.15 | nov00, 61.1 % |
| random 30 % | 0.000 | 0.66 | 0.20 | int01, 48.1 % |

The ICC of 0.42 under asks 30 % comes mostly from one novice (nov02: 63 and 11 % on its two seeds). The ICC of a single
campaign with three operators per tier is unstable: ensemble extended had 0.84 in D2 and 0.00 here.

### 5.4 Convergence

HIL-SERL converged in 15 of 18 cells, ensemble extended 11, asks 10 / 30 / 55 % 14 / 8 / 11, random 10. The rolling-window
rule (20 autonomous episodes at ≥ 90 % plus a mini-evaluation) sees fewer autonomous episodes when the operator is asked
more, so the gated arms converge less often; the ruler scores every cell either way.

## 6. The mechanism: who answers the robot (measured here)

From the report's per-takeover table (`quality_audit/segments.csv`, 37 565 takeovers in the five arms that log them;
computed 2026-11-03). "Requested" = the takeover answered a request of the robot; the rest are the operator's spontaneous
takeovers (false alarms in the model).

| arm | tier | takeovers per run | requested | mean length (steps) | in successful episodes | end at the time cap |
|---|---|---|---|---|---|---|
| ensemble extended | expert | 615 | – | 31 | 33 % | 62 % |
| | intermediate | 350 | – | 46 | 11 % | 65 % |
| | novice | 206 | – | 65 | **3 %** | 64 % |
| asks 30 % | expert | 857 | 97 % | 19 | 55 % | 16 % |
| | intermediate | 335 | 89 % | 26 | 20 % | 19 % |
| | novice | **71** | **46 %** | 34 | 8 % | 18 % |
| asks 55 % | expert | 936 | 98 % | 24 | 45 % | 31 % |
| | intermediate | 406 | 91 % | 35 | 20 % | 36 % |
| | novice | **85** | **60 %** | 49 | 6 % | 34 % |
| random 30 % | expert | 992 | 98 % | 15 | 24 % | 5 % |
| | intermediate | 566 | 91 % | 17 | 6 % | 3 % |
| | novice | 72 | 50 % | 31 | 4 % | 8 % |

Operator share of the steps by tier (report): asks 55 % 63.8 / 48.1 / 13.4 %, ensemble extended 60.5 / 53.9 / 46.9 %,
HIL-SERL 59.5 / 55.0 / 43.0 %.

**Reading.**
- The gate does not reduce the experts' help: they take over more often (936 against 615 times per run) and, in sum,
  control more steps than with their own monitor. The "less help" of the asking arms is the novices' missing answers.
- Under the gate, takeovers are short and mostly end when the request ends; with the operator's own monitor about two
  thirds run into the time cap.
- So the near-zero ICC is partly mechanical. Novice-trained cells learn mostly from the robot's own experience and the
  novice's six demonstrations, so they look more like the others. This is one reason the thesis calls the asking result
  "largely supported", not "supported".
- **The same table matters for grading:** 97 % of a novice's takeovers under its own monitor fall in failed episodes
  (67 % for experts). A filter that drops failed-episode takeovers removes almost all novice corrections (see
  [first_machine_oracle_quality_results_and_qa_log.md](first_machine_oracle_quality_results_and_qa_log.md)).

## 7. The quality audit on these runs

`quality_audit/summary.md`: the counterfactual advantage Â is inverted in every arm (AUROC against episode success
0.06–0.16, 0.06–0.13 on takeovers released by hand-back). A score that knows only when the takeover started reaches
0.82–0.89. This replicates the earlier finding on another campaign.

## 8. What this means, and its limits

1. **Asking is the thesis's positive result.** It trades effort for success along a monotone curve, and at the high rate
   it beats the same critics with the operator deciding, with less help and no measurable teacher effect.
2. **It does not yet beat plain HIL-SERL significantly,** and its gain over HIL-SERL comes from one tier.
3. **The timing claim is open:** +4.9 points over random requests at a matched rate (10 / 5 / 3, p = 0.30); D2 pointed the
   same way (+14.6, sign test p = 1.00).
4. **Part of the independence is the operator model** (section 6), so a held request, which waits until the operator
   answers, is the next test (implemented as patch 0025, not run).
5. **Threats:** one campaign; 35 000-step cap; fewer gated cells converge; the operator could not turn the peg (second
   machine). The asking result is to be repeated on the new stack (`C_full`).
6. **It replicates:** the oracle campaign ran asks 30 % again on the same roster: 82.9 % (here 81.3 %), tiers 100 / 91 /
   57 % (here 100 / 87 / 57 %), operator share 28.4 % (here 28.5 %).

## 9. Q&A log

| question | answer |
|---|---|
| Is σ_Q a measure of danger? | No. It measures how much the 8 critics disagree about the robot's next action: unfamiliarity. D1 showed it does not detect jams. |
| Why the robot's own action, not the operator's? | The gate must decide before anyone acts, so it can only ask about what the robot would do next. |
| What does "asks 55 %" mean exactly? | A request opens when the current disagreement is above the 45th percentile of the last 2 000 steps, lasts 10 steps, then a 5-step pause. The realised operator share was 41.8 %. Worked example with numbers: section 3.1. |
| Does "asks 55 %" request 55 % of the steps? | No. About 55 % of the recent steps are above the line; the 10-on, 5-off timing turns that into the requested share (57 % in a replay on logged σ_Q; 24 % for asks 10 %, 43 % for asks 30 %), and the operator's answers into the operator share (41.8 %). Section 3.2. |
| Is σ_Q stored for every step, or fitted to a normal distribution? | For the z-score only three running numbers are kept (count, mean, sum of squared deviations) over the whole run; for the line, the last 2 000 scores. No distribution is fitted: the line is the empirical quantile of those 2 000 values. Every step's σ_Q is also written to `steps.jsonl`, for the analysis only. |
| Is the threshold adaptive? | Yes: it is recomputed every step from the last 2 000 scores, so it follows σ_Q down as the critics converge (and up when σ_Q rises). It keeps the rate fixed, so it never stops asking; a deployed system would need a floor or a falling rate. Section 3.2. |
| Why not simply a fixed threshold? | Its rate is an accident of σ_Q's shape and drift: about 4–9 % on the first machine in D2; in a replay on second-machine traces 20–36 % per run, doubling within a run. The rank line keeps asks 55 % at 53–60 % in every run. |
| Does the robot stop while asking? | No (`continue`): it keeps executing its own action until the operator takes over. |
| Why do novices answer so little? | In the model, a novice's monitoring is slow and noisy; a 10-step request often ends before its smoothed signal crosses the threshold. Real people may differ. |
| Is "less help" a saving? | Partly. Experts help more under the gate; the total falls because novices answer little. |
| Then is the ICC of 0 real? | It is measured correctly, but part of it comes from novices dropping out of the data. A held request will test this. |
| Why does HIL-SERL show a strong teacher effect here but ICC 0.00 in B2? | Single-campaign ICC is unstable with three operators per tier; here one intermediate (int00) failed badly with HIL-SERL. Compare only within a campaign. |
| Which arm goes forward? | Asks 55 %, as the asking arm of the repeat at thesis scale on the new stack. |

## 10. Artifacts and how well it is checked

| what | where | checked |
|---|---|---|
| report, tables, plots | `presentation_bundle/results/active_ratematch/` | read; numbers above copied from it |
| pairings against HIL-SERL and by tier | computed here from the report's per-cell tables | yes |
| who answers (section 6) | computed here from `quality_audit/segments.csv` | yes |
| answered share of requests (77 / 47 / 2–9 %) | W&B logs of D1 and D2 (`wandb_analysis/`) | yes, on those campaigns |
| raw logs and checkpoints of this campaign | first machine only | not here |

Report display bugs (the tables are right): the headline puts a minus sign on ICC increases, the header says 5 operators
per tier (it is 3), and `plots/frontier.png` puts HIL-SERL at 0 % help (it is 52.5 %). Fixed in patch 0013, not yet on
the first machine; use `presentation_bundle/output/_siemens/gen/r_rate.png` instead.
