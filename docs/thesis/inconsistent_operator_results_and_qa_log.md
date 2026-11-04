# The inconsistent operator (score validity, discard, re-weight) — results & QA log

Companion to [second_machine_synthesis.md](second_machine_synthesis.md). This is the
test of the hypothesis that **a quality score pays off when the operator is inconsistent**,
as humans are. Design, arms and reading rules: [docs/inconsistent_operator.md](../inconsistent_operator.md).

- Machine: the second training machine (`LSPEM499`, RTX 5090), new stack, score `behavioral`.
- Campaigns: `Q1_validity_new` (6 cells), `headroom_diagnostic_new` (9), `two_arm_pilot_new` (6),
  `two_arm_confirm_new` (36), 25–29 October 2026.
- Raw results: `presentation_bundle/results/<campaign>/` (`dependency.md`, `inconsistent_audit.md`,
  `cells.csv`). The audits there were re-run on the raw logs with patch 0021's tool, which adds the
  within-run columns; `inconsistent_audit_on_the_machine.md` is the original.
- The machine's own decision records: `presentation_bundle/results/second_machine_reports/`.
- All tables below are generated from those files and the raw logs of the export. Checked on 2026-11-03
  against the machine's reports and the documentation archive; section 9 lists what that check changed.

---

## 0. Headline

**The hypothesis is neither supported nor refuted: the test was too weak to decide it.**

- **The scores can notice carelessness, moderately.** The best online score (behavioural) separates
  attentive from inattentive takeovers with AUROC 0.66 overall, 0.71 / 0.67 for expert /
  intermediate, and hardly at all for novices (0.52). Takeover length alone does almost as well (0.65). The
  filter's own live score reaches the same 0.66.
- **Using the score made no detectable difference.** Discard against re-weight: 93.2 vs 90.3 % ruler
  success, −2.9 points, 95 % CI −13.4 to +6.2, p 0.39.
- **The perfect-filter check gives no usable direction.** On the ruler, keep-all wins (98.8 vs 93.8 %). At the
  last checkpoint the order flips (96.3 vs 100 %). It has 3 pairs, sits near the ceiling, and its one visible
  difference falls in a long inattentive spell that only the novice cell hit after the warm-up.
- **Baseline and keep-all were not run at scale,** by design of the lean experiment, so nothing here
  says whether quality beats HIL-SERL.
- **The score's weak verdict was never lowered.** The automation stopped at that verdict. The two-arm
  experiment then ran on the user's explicit instruction, as a study of a weak score (section 1).

---

## 1. How this study actually ran

Times are the machine's local times. Clocks went back one hour on 25 October at 03:00.

| when | what happened | decision | who decided |
|---|---|---|---|
| 24 Oct 23:21 | Stage 0: 4 of 12 attentive / inattentive comparisons below 20 points, all without rotation control | **no-go** for the two-platform plan | automation |
| 25 Oct 00:24 | scope reduced to the new platform; threshold, model and code unchanged | go on, new platform only | user delegated it, recorded in `scope-decision.json` |
| 25 Oct 00:51 – 09:57 | Q1, 6 cells | – | – |
| 25 Oct 09:57 | best online score weak (0.61; a valid score needs AUROC ≥ 0.70 and a CI above 0.5) | **no-go** at Gate 2; automation stops, no two-arm campaign | automation |
| 25 Oct 19:33 | the machine's Claude corrects itself: the hard stop was its own conservative rule, the repo allows controlled experiments without a valid score. It replays the filter on Q1 and proposes the perfect-filter check instead of a 36-cell run | recommendation | machine's Claude |
| 25 Oct 19:43 – 26 Oct 08:09 | perfect-filter check, 9 cells; not in the handoff | "no convincing primary headroom" by its fixed rule (oracle beats keep-all in at least two tiers) | user authorised the run |
| 26 Oct 12:02 | user: ignore the thresholds that would stop training and run to the end | scientific gates report-only; pilot goes on to confirmation whatever its direction | user, recorded in `twoarm_execution_policy.json` |
| 26 Oct 12:02 – 20:27 | pilot, 6 cells, behavioural score | – | – |
| 26 Oct 20:27 – 29 Oct 02:14 | confirmation, 36 cells | report only | – |

So the original pilot-to-confirmation rule was not what let the confirmation start, and "weak" stayed "weak" in every
report. Only `quality_score_mode` changed in the two `_new` configs (`score-config.diff`). Six-arm Q2 and Q3 were left out on the
user's instruction, so the design had no keep-all, no baseline and no consistent-operator arm at scale.

## 2. The operator

At every takeover the hidden condition stays with probability 0.7, or else it is redrawn as
inattentive with probability 0.4. Inattentive means *sloppy and wrong*: novice execution
parameters, premature commits with probability 0.5, and an aim point 1.5 commit thresholds beside the
hole. Monitoring stays the operator's own. The condition is logged per takeover as ground truth.

**How it works in the code** (`operator_model.py`: `Inconsistency`, `inattentive_theta`, `SyntheticOperator._draw_bias`; step by step and in formulas in section 2.1):
- **When the condition is drawn.** Once per takeover, at its onset, and held until the hand-back. With probability 0.7 it
  stays as it was; otherwise it is drawn anew, inattentive with probability 0.4. So 40 % of takeovers are inattentive in
  the long run, and two in a row agree with probability 0.7 + 0.3 × (0.4² + 0.6²) = 0.856: careless spells span several
  takeovers and episodes.
- **What inattentive changes.** Eight execution parameters take the novice anchor's values: motor noise σ₀,
  speed-dependent noise k_sdn, tremor σ_tr, lag, top speed, pause probability and length, recovery. The operator commits
  (starts pressing down) too early in half of the attempts. And it aims beside the hole: 1.5 commit thresholds away
  (about 6 mm on the cuboid, whose commit threshold is 4 mm), in a random direction that stays fixed for the takeover.
- **What it does not change.** The monitoring: when the operator takes over is its own, so the start of a takeover
  does not depend on the condition. Its end can: a careless takeover leaves the peg in trouble longer.
- **What reads the label.** Nothing in training, except the perfect filter of the check. The demonstrations come from the
  consistent operator, so only corrections are careless.

Measured in training: two takeovers in a row share the condition with probability 0.85–0.87 (design 0.856), and the inattentive share is 0.31–0.37 by campaign (0.40 in the long run; lower here because each sequence starts mostly attentive).

**Without learning, the condition really is worse, once the operator can turn the peg.** 30 episodes per row, cuboid:

| rotation | tier | start | attentive | inattentive | gap (points) | ≥ 20 points? |
|---|---|---|---|---|---|---|
| on | expert | solo | 100.0 % | 43.3 % | 56.7 | yes |
| on | expert | rescue | 100.0 % | 30.0 % | 70.0 | yes |
| on | intermediate | solo | 100.0 % | 50.0 % | 50.0 | yes |
| on | intermediate | rescue | 100.0 % | 30.0 % | 70.0 | yes |
| on | novice | solo | 76.7 % | 43.3 % | 33.3 | yes |
| on | novice | rescue | 60.0 % | 26.7 % | 33.3 | yes |
| off | expert | solo | 100.0 % | 13.3 % | 86.7 | yes |
| off | expert | rescue | 13.3 % | 3.3 % | 10.0 | **no** |
| off | intermediate | solo | 86.7 % | 20.0 % | 66.7 | yes |
| off | intermediate | rescue | 16.7 % | 3.3 % | 13.3 | **no** |
| off | novice | solo | 30.0 % | 23.3 % | 6.7 | **no** |
| off | novice | rescue | 13.3 % | 3.3 % | 10.0 | **no** |

All six comparisons with rotation control pass the fixed 20 points. Without it, four of six fail, because the attentive
operator itself rescues only 13–17 %. Raising the aim offset could not fix that, so the threshold and the model stayed as they were
and the original platform was excluded.

**In training the effect depends on the tier:**

| campaign | tier | takeovers | inattentive share | takeover length, attentive / inattentive (steps) | episode success, attentive / inattentive |
|---|---|---|---|---|---|
| validity (Q1) | expert | 261 | 0.32 | 23.3 / 31.4 | 90 / 67 % |
| validity (Q1) | intermediate | 212 | 0.32 | 39.1 / 43.0 | 65 / 56 % |
| validity (Q1) | novice | 168 | 0.32 | 57.7 / 60.9 | 26 / 25 % |
| perfect-filter check | expert | 481 | 0.36 | 22.9 / 31.1 | 89 / 55 % |
| perfect-filter check | intermediate | 210 | 0.37 | 35.6 / 45.6 | 68 / 50 % |
| perfect-filter check | novice | 222 | 0.39 | 60.5 / 60.8 | 14 / 24 % |
| pilot | expert | 342 | 0.35 | 24.8 / 33.3 | 80 / 52 % |
| pilot | intermediate | 132 | 0.35 | 35.7 / 43.5 | 70 / 78 % |
| pilot | novice | 124 | 0.36 | 63.1 / 65.0 | 23 / 40 % |
| confirmation | expert | 1367 | 0.32 | 23.6 / 32.6 | 81 / 59 % |
| confirmation | intermediate | 1081 | 0.32 | 35.8 / 43.8 | 66 / 33 % |
| confirmation | novice | 909 | 0.30 | 60.0 / 61.3 | 23 / 24 % |

- **Expert:** attentive is better in every campaign.
- **Intermediate:** attentive is better in three of four; the pilot's single operator reverses it (70 vs 78 %, 132 takeovers).
- **Novice:** the inattentive condition never does measurable harm, and in two small campaigns it looks better (24 vs 14 %, 40 vs
  23 %). Effectively, the test of inconsistency rests on the expert and intermediate tiers.

**Why novices differ: an inattentive takeover uses fixed novice anchors, and a novice can be noisier than those anchors.**

| operator | takeovers | σ₀ | k_sdn | tremor σ_tr | mean jerk, attentive / inattentive | AUROC of smoothness |
|---|---|---|---|---|---|---|
| **novice anchors** (what an inattentive takeover uses) |  | 10.00 | 0.100 | 9.00 |  |  |
| `nov00`, validity (Q1) | 168 | 9.54 | 0.083 | 8.83 | 0.246 / 0.275 | 0.66 |
| `nov00`, perfect-filter check | 222 | 9.54 | 0.083 | 8.83 | 0.252 / 0.307 | 0.67 |
| `nov00`, pilot | 124 | 9.54 | 0.083 | 8.83 | 0.238 / 0.289 | 0.71 |
| `nov00`, confirmation | 348 | 9.54 | 0.122 | 11.79 | 0.275 / 0.240 | 0.35 |
| `nov01`, confirmation | 262 | 9.99 | 0.122 | 9.55 | 0.268 / 0.241 | 0.41 |
| `nov02`, confirmation | 299 | 9.06 | 0.107 | 11.68 | 0.263 / 0.282 | 0.55 |

The novice of the 1-per-tier roster sits below the anchors on all three noise parameters, so inattentive means rougher. All three
novices of the 3-per-tier roster have more tremor than the anchor, so inattentive can mean *smoother*: two of them measurably are.
That is a property of the model, not of a real careless person. It should be fixed before novices are used to test inconsistency (see
[second_machine_synthesis.md](second_machine_synthesis.md) §5).

**Operators with the same name are not the same operator.** `int00` and `nov00` have different parameters in the 1-per-tier campaigns (Q1,
check, pilot) and in the 3-per-tier confirmation, because the roster's jitter differs. Only `exp00` is identical. Never pair those two names across
these campaigns.

**All operators with one seed shared one condition sequence, in all four campaigns: a bug, now fixed.** `_cond_rng` depended only on the seed,
so the 57 cells carry just two sequences:

| seed | cells sharing it (all campaigns) | longest run seen (takeovers) | inattentive share per 25 takeovers (%) | longest inattentive spell |
|---|---|---|---|---|
| 0 | 36 | 206 | 12 / 44 / 68 / 12 / 36 / 44 / 40 / 28 | 19 takeovers, from takeover 41 |
| 1 | 21 | 140 | 8 / 44 / 12 / 56 / 12 / 47 | 8 takeovers, from takeover 77 |

Early takeovers are therefore mostly attentive (about 10 % inattentive in the first 25). The long spell of seed 0 starts at takeover 41. Novice
cells reach the end of the warm-up after 34–38 takeovers (section 3.4), just before it, and experts well after it. So exposure to bad
corrections is confounded with training phase and with tier. Patch 0021 makes the sequence depend on seed and operator, keeping the arms of one
operator and seed paired.

### 2.1 The implementation, step by step and in formulas

Code: `serl_launcher/aqi/operator_model.py` (`Inconsistency`, `inattentive_theta`, `SyntheticOperator.begin_takeover`,
`_next_condition`, `_apply_condition`, `_draw_bias`); switched on with `--synthetic_inconsistent`, `--synthetic_bad_share`
($b = 0.4$) and `--synthetic_bad_persistence` ($\rho = 0.7$). "Attentive" is careful, "inattentive" careless.

**1. The hidden condition is a two-state chain over takeovers.** Let $c_n \in \{A, I\}$ be the condition of the $n$-th
takeover of a run. The first is drawn with $P(c_0 = I) = b$; then, at each takeover onset,

$$
c_n =
\begin{cases}
c_{n-1} & \text{with probability } \rho \\
I & \text{with probability } (1-\rho)\,b \\
A & \text{with probability } (1-\rho)(1-b)
\end{cases}
\qquad
P = \begin{pmatrix} P(A \to A) & P(A \to I) \\ P(I \to A) & P(I \to I) \end{pmatrix}
  = \begin{pmatrix} 0.88 & 0.12 \\ 0.18 & 0.82 \end{pmatrix}
$$

- Long-run share of careless takeovers: $\pi_I = b = 0.4$.
- Two takeovers in a row agree with $\rho + (1-\rho)\,(b^2 + (1-b)^2) = 0.856$.
- Mean spell length: careless $1 / 0.18 \approx 5.6$ takeovers, careful $1 / 0.12 \approx 8.3$. Spells cross episode
  boundaries, because the condition is kept between episodes.
- The chain has its own random stream. Since patch 0021 it is seeded by the operator's seed **and** its name (a draw from
  the seed's generator together with `crc32(operator_id)`), so the arms of one operator and seed see the same sequence (a
  paired comparison) and different operators do not. On the second machine it was seeded by the seed alone: every operator
  with one seed shared one sequence (the bug of section 2).

**2. What is drawn when.**

| moment | what happens |
|---|---|
| episode reset | the operator's own parameters $\theta_{own}$ are restored; the condition is kept for the next takeover |
| takeover onset (`begin_takeover`) | draw $c_n$; set $\theta \leftarrow \theta_{bad}$ if $c_n = I$, else $\theta_{own}$; redraw the premature-commit event and the aim offset |
| during the takeover | the operator acts with $\theta$ and the fixed aim offset |
| hand-back | decided by the operator's own monitoring, unchanged |

So the demonstrations, which are recorded from resets, always use $\theta_{own}$: only corrections can be careless.

**3. The motor model, and which parameters a careless takeover replaces.** Each step the operator turns its intent $u_t$
(from the geometric oracle) into a command:

$$
a_t = (1-\lambda)\,a_{t-1} + \lambda\,(u_t + \varepsilon_t + \tau_t), \qquad
\varepsilon_t \sim \mathcal N\!\big(0,\, (\sigma_0 + k_{sdn}\,\lVert u_t \rVert)^2\big), \qquad \lVert a_t \rVert \le v_{max}
$$

with $\tau_t$ an Ornstein-Uhlenbeck tremor of size $\sigma_{tr}$, random pauses (probability $p_{pause}$ per step, mean length
`pause_len_mean`) and a recovery behaviour $q_{rec}$ (retract and re-align rather than push on). A careless takeover sets
$\theta_{bad}$ = the operator's own $\theta$ with these eight execution parameters replaced by the **novice anchor**, and
$p_{early} = 0.5$:

| parameter | meaning | expert anchor | intermediate anchor | novice anchor = careless |
|---|---|---|---|---|
| $\sigma_0$ | motor noise | 0.629 | 2.508 | 10.0 |
| $k_{sdn}$ | speed-dependent noise | 0.03 | 0.055 | 0.10 |
| $\sigma_{tr}$ | tremor | 1.732 | 3.948 | 9.0 |
| $\lambda$ (`lag`) | how fast a command follows the intent | 0.641 | 0.506 | 0.40 |
| $v_{max}$ | top speed | 79.6 | 66.2 | 55.0 |
| $p_{pause}$ | pause probability per step | 0.028 | 0.054 | 0.08 |
| `pause_len_mean` | mean pause length (s) | 0.186 | 0.472 | 1.2 |
| $q_{rec}$ | recovery | 0.17 | 0.125 | 0.08 |
| $p_{early}$ | premature commit per attempt | 0.052 | 0.116 | 0.18 → **0.5** when careless |

Real operators are jittered around these anchors (roster jitter 0.15), so `nov00` is not exactly the novice anchor; that
is why a careless takeover can be smoother than a novice's own (section 2). A premature commit means the operator starts
pressing down after a random 0.2–0.7 s, aligned or not.

**4. The aim offset.** The attentive operator aims at the hole with a small perception error, the careless one beside it:

$$
\text{attentive: } \mathbf b \sim \mathcal N(0,\, \sigma_{perc}^2 I_2), \qquad
\text{careless: } \mathbf b = 1.5\, c_{commit}\, \frac{\mathbf d}{\lVert \mathbf d \rVert},\ \ \mathbf d \sim \mathcal N(0, I_2)
$$

drawn at the onset and fixed for the takeover. On the cuboid $c_{commit} = 4$ mm, so $\lVert \mathbf b \rVert = 6$ mm,
beyond the 5 mm the peg allows.

**5. What it leaves alone.** The monitoring parameters ($\sigma_{mon}$, $\delta_{in}$, $m_{in}$, $l_{react}$, $p_{miss}$,
$\rho_{fa}$, $\delta_{out}$, $k_{out}$, $d_{max}$) are not in $\theta_{bad}$, so *when* a takeover starts does not depend on
the condition; rotation control works in both conditions.

**6. The label.** Every takeover segment stores `condition` (in `segments.jsonl`). Only the perfect filter reads it.

## 3. Do the scores find the bad corrections?

### 3.0 The scores, defined

A *score* is one number per takeover that should be high for a careful correction and low for a careless one. Live scores
are computed at the end of each episode by `serl_launcher/aqi/runtime.py` and `quality.py`; the offline ones by
`examples/inconsistent_operator_audit.py` (with `quality_signal_audit.py`).

**Notation.** A takeover covers steps $t_0, \dots, t_1$, so its length is $T = t_1 - t_0 + 1$, and control returns to the
robot at $t_1 + 1$. At each step, $s_t$ is the state, $a^H_t$ the operator's action and $a^R_t$ the action the policy
would have taken in the same state. The ensemble has $K = 8$ critics:

$$
\mu_Q(s,a) = \frac1K \sum_{k=1}^{K} Q_k(s,a), \qquad \sigma_Q(s,a) = \sqrt{\frac1K \sum_{k=1}^{K} \big(Q_k(s,a) - \mu_Q(s,a)\big)^2}
$$

$F_t$ is the contact-force magnitude, $r_t$ the reward, $p(s) = -z_{tcp}(s)$ the insertion depth (the tool's height below
the reset pose), $\gamma = 0.97$ and $H = 10$ steps after the hand-back. The critics are those of the moment the episode is
scored.

**Running standardisation** (live scores). For a takeover feature $x$, after adding $x$ to this run's earlier values:

$$
z(x) = \frac{x - m_n}{s_n + \epsilon}, \qquad m_n, s_n = \text{running (Welford) mean and std over the run's first } n \text{ takeovers}
$$

and $z(x) = 0$ while $n < 10$. The offline audit uses the mean and std of the whole run instead, which a live system
cannot; the live score reaches the same AUROC (section 3.3), so this does not inflate the result.

#### How a score is judged

$$
\mathrm{AUROC} = \frac{1}{n_A\, n_I} \sum_{i \in A} \sum_{j \in I} \Big[\mathbb 1(S_i > S_j) + \tfrac12\,\mathbb 1(S_i = S_j)\Big]
$$

the probability that a random careful takeover ($A$) scores higher than a random careless one ($I$): 0.5 is chance, 1 is
perfect, below 0.5 the score points the wrong way. **Within** counts only pairs inside the same stratum $g$ = (run,
training-phase bin of 15 takeovers, onset-time bin of 50 steps):

$$
\mathrm{AUROC}_{within} = \frac{\sum_g U_g}{\sum_g n_{A,g}\, n_{I,g}}, \qquad U_g = \text{the pair count above, inside } g
$$

Because careless takeovers come in spells, the condition is tied to the training phase, and a pooled AUROC can reward a
score that only tells early from late. The 95 % CI resamples whole episodes (500 times). Verdict: **valid** if
AUROC ≥ 0.70 and the CI's lower bound > 0.5, **weak** between, **invalid** at or below 0.5.

#### The score both arms used

**Behavioural composite** (`quality_score_mode: behavioral`):

$$
B = z(\Delta V) - z(T) - z(F_{peak}), \qquad
\Delta V = \mu_Q\big(s_{t_1}, a^H_{t_1}\big) - \mu_Q\big(s_{t_0}, a^R_{t_0}\big), \qquad
F_{peak} = \max_{t_0 \le t \le t_1} F_t
$$

The critics' value at the takeover's last step (at the operator's action) minus at its first (at the robot's action), minus
its length, minus its peak force. A short, value-raising, gentle correction scores high.

**How the two arms turn $B$ into training:**

- **Discard** (`score_filter`, `quality_drop_fraction` 0.4). Keep the last 200 scores in a window $W$. After the warm-up
  (env step ≥ 5 000) and once $\lvert W \rvert \ge 20$:

  $$
  \text{drop the takeover if } B < Q_{0.4}(W)
  $$

  where $Q_{0.4}$ is the 40th percentile; ties at the threshold are dropped at random so the share stays 40 %. A dropped
  takeover leaves the demonstration buffer only; its steps stay in the online buffer. $B$ is logged as `drop_score`.
- **Re-weight** (no filter, `per_eta` 1). Every takeover is kept, and its steps get the replay priority

  $$
  q_{int} = \max(B, 0), \qquad
  p_i = \Big(\epsilon + \lvert \delta_i \rvert + \eta\, \frac{q_{int,i}}{\bar q}\Big)^{\alpha}, \qquad
  P(i) = \frac{p_i}{\sum_j p_j}, \qquad
  w_i = \frac{(N\,P(i))^{-\beta}}{\max_{j \in \text{batch}}\, (N\,P(j))^{-\beta}}
  $$

  with $\epsilon = 0.05$, $\eta = 1$, $\alpha = 0.6$, $\beta = 0.5$; $\delta_i$ the item's TD error, $\bar q$ a running mean
  of $q_{int}$ (priorities refreshed every 2 000 steps), and $w_i$ the importance weight in the loss. The clip at 0 gives
  every below-average takeover the same priority, which is why the logged $q_{int}$ separates less well than $B$.

The other filters: the **perfect filter** drops a takeover if $c = I$; the **random drop** drops it with probability 0.4.
Both act only after the warm-up.

#### All scores, with their formula and their AUROC (confirmation, 3 357 takeovers)

| score | formula (higher = more careful) | live? | within | within: expert / interm. / novice | pooled |
|---|---|---|---|---|---|
| behavioural composite | $B = z(\Delta V) - z(T) - z(F_{peak})$ | yes, both arms | **0.66** | 0.71 / 0.67 / 0.52 | 0.65 |
| the score as logged | $q_{int} = \max(B, 0)$ | yes (re-weight) | 0.63 | 0.67 / 0.66 / 0.50 | 0.62 |
| takeover length | $-T$ | part of $B$ | 0.65 | 0.69 / 0.68 / 0.51 | 0.60 |
| value gain | $\Delta V$ | part of $B$ | 0.60 | 0.66 / 0.59 / 0.49 | 0.60 |
| behavioural without length | $z(\Delta V) - z(F_{peak})$ | no | 0.61 | 0.65 / 0.60 / 0.53 | 0.59 |
| peak force | $-F_{peak}$ | part of $B$ | 0.58 | 0.61 / 0.56 / 0.53 | 0.56 |
| hindsight advantage | $A_{hind}$, below | yes (`hindsight`) | 0.62 | 0.66 / 0.63 / 0.49 | 0.55 |
| episode outcome | $+1$ if the episode succeeded, $-1$ if not | yes (`outcome`) | 0.55 | 0.56 / 0.58 / 0.49 | 0.59 |
| consistency | $-\bar\sigma_\beta$, below | BC weight only | 0.58 | 0.63 / 0.55 / 0.50 | 0.60 |
| progress after hand-back | $p(s_{t_p}) - p(s_{t_0})$, $t_p = \min(t_0 + T + H,\ \text{episode end})$ | yes (`progress`) | 0.50 | 0.51 / 0.50 / 0.46 | 0.50 |
| progress during the takeover | $p(s_{t_1}) - p(s_{t_0})$ | no | 0.49 | 0.48 / 0.51 / 0.47 | 0.49 |
| counterfactual advantage, mean | $\bar A = \frac1T \sum_{t=t_0}^{t_1} \big[\mu_Q(s_t, a^H_t) - \mu_Q(s_t, a^R_t)\big]$ | yes (`counterfactual`) | 0.49 | 0.46 / 0.50 / 0.54 | 0.50 |
| counterfactual at onset | $\mu_Q(s_{t_0}, a^H_{t_0}) - \mu_Q(s_{t_0}, a^R_{t_0})$ | no | 0.50 | 0.50 / 0.45 / 0.56 | 0.50 |
| critic disagreement | $\bar\sigma = \frac1T \sum_{t=t_0}^{t_1} \sigma_Q(s_t, a^H_t)$ | no | 0.48 | 0.50 / 0.45 / 0.49 | 0.49 |
| smoothness | $-\frac1T \sum_{t=t_0}^{t_1} j_t$, $\ j_t = \lVert a^H_t - 2a^H_{t-1} + a^H_{t-2} \rVert$ (0 for the first two steps) | no | 0.76 | 0.86 / 0.85 / **0.40** | 0.71 |
| yardstick: start time | $-t_0$ | – | 0.49 | 0.49 / 0.45 / 0.54 | 0.59 |
| yardstick: end time | $-(t_1 + 1)$, the hand-back step | – | 0.60 | 0.61 / 0.64 / 0.55 | 0.60 |
| yardstick: value at onset | $\mu_Q(s_{t_0}, a^R_{t_0})$ | – | 0.58 | 0.60 / 0.62 / 0.48 | 0.55 |

**Hindsight advantage** judges a correction by what the robot reached once it had control back. With
$t_e = t_0 + T + H$ (clipped to the episode) and the value taken only at the policy's own action,
$V(s_t) = \mu_Q(s_t, a^R_t)$:

$$
A_{hind} = \sum_{t=t_0}^{t_e - 1} \gamma^{\,t - t_0}\, r_t \;+\; \gamma^{\,t_e - t_0}\, V(s_{t_e}) \;-\; V(s_{t_0})
$$

without the bootstrap term if the episode terminated inside the window. Live it is divided by the running mean of
$\lvert A_{hind} \rvert$. Before the critics have seen a reward, $r_t = 0$ and $V$ is nearly flat, so
$A_{hind} \approx (\gamma^{T+H} - 1)\, V(s_{t_0})$: a measure of length and timing (finding F4). That is why it is weak
pooled (0.55) and better within run and late in training.

**Consistency** asks whether the operator acts as it did at similar states in its earlier takeovers (the SiLRI idea). For
a state $s$, take the nearest state $s_j$ from each of the $k = 5$ earlier takeovers closest to $s$ (states standardised):

$$
\sigma_\beta(s) = \sqrt{\frac1d \sum_{i=1}^{d} \mathrm{Var}_{j = 1..k}\big[a^H_i(s_j)\big]}, \qquad
\bar\sigma_\beta = \text{mean of } \sigma_\beta \text{ over evenly spaced steps of the takeover}
$$

A small spread means a consistent operator. Only one neighbour per earlier takeover, since steps of one trajectory are
always close to each other.

**Why each one does what it does:**
- **Behavioural is moderate and mostly length.** Careless takeovers last longer (43.7 against 37.5 steps): the wrong aim and
  the novice motor noise keep the peg in trouble, and the monitor hands back only when the trouble clears or at the time
  cap. Without its length term the composite falls from 0.66 to 0.61.
- **Counterfactual is at chance and below it once the critics are trained** (0.41 in the second half). The actor is trained
  to maximise these same critics, so $a^R$ always looks good: the score measures distance from the policy, not quality.
  The pre-registered cluster study abandoned it.
- **Progress, outcome and critic disagreement carry little or nothing.** Probably because depth after the hand-back depends
  on the robot as much as on the correction, and an episode's success on far more than one takeover.
- **Smoothness is circular.** Careless *is* novice motor noise, which is what jerk measures, so it would say nothing about
  a real careless person. It fails for the novices because their own tremor exceeds the anchor (section 2), was chosen
  after seeing the validity results, and has no live mode.
- **The yardsticks are checks, not scores.** Start time and value at the onset must sit near 0.5, because the condition is
  drawn independently of when and where a takeover starts: within run they are 0.49 and 0.58. Pooled, start time reaches
  0.59, which is only the spells, and the reason "within" is the number to read. End time may differ by design.

### 3.1 The confirmation

AUROC for "attentive" on 3 357 takeovers (`two_arm_confirm_new`). *Within* compares only takeovers of
the same run, training phase (bins of 15 takeovers) and onset time (bins of 50 steps). That is the
cleanest number, since the condition comes in long spells. *2nd half* uses each run's second half of episodes, when the critics are trained.

| score | online mode | n | pooled (95 % CI) | 2nd half | within | within: expert | within: intermediate | within: novice | pooled by tier (expert / intermediate / novice) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| behavioural composite | behavioral | 3357 | 0.65 (0.63–0.67) | 0.65 | 0.66 | 0.71 | 0.67 | 0.52 | 0.73 / 0.63 / 0.53 | weak |
| the score as logged (`q_int`) | – | 3357 | 0.62 (0.60–0.64) | 0.65 | 0.63 | 0.67 | 0.66 | 0.50 | 0.69 / 0.62 / 0.51 | weak |
| takeover length alone | – | 3357 | 0.60 (0.57–0.62) | 0.60 | 0.65 | 0.69 | 0.68 | 0.51 | 0.72 / 0.64 / 0.51 | weak |
| value gain | – | 3357 | 0.60 (0.58–0.62) | 0.60 | 0.60 | 0.66 | 0.59 | 0.49 | 0.66 / 0.62 / 0.51 | weak |
| behavioural without length | – | 3357 | 0.59 (0.57–0.62) | 0.60 | 0.61 | 0.65 | 0.60 | 0.53 | 0.65 / 0.58 / 0.53 | weak |
| peak force | – | 3357 | 0.56 (0.53–0.58) | 0.53 | 0.58 | 0.61 | 0.56 | 0.53 | 0.57 / 0.55 / 0.56 | weak |
| hindsight advantage | hindsight | 3357 | 0.55 (0.52–0.58) | 0.59 | 0.62 | 0.66 | 0.63 | 0.49 | 0.65 / 0.54 / 0.42 | weak |
| episode outcome | outcome | 3357 | 0.59 (0.57–0.62) | 0.57 | 0.55 | 0.56 | 0.58 | 0.49 | 0.61 / 0.67 / 0.50 | weak |
| consistency (σ_β) | – | 3052 | 0.60 (0.57–0.62) | 0.59 | 0.58 | 0.63 | 0.55 | 0.50 | 0.64 / 0.59 / 0.50 | weak |
| progress after hand-back | progress | 3357 | 0.50 (0.48–0.52) | 0.53 | 0.50 | 0.51 | 0.50 | 0.46 | 0.50 / 0.51 / 0.47 | invalid |
| progress during takeover | – | 3357 | 0.49 (0.47–0.51) | 0.52 | 0.49 | 0.48 | 0.51 | 0.47 | 0.48 / 0.52 / 0.47 | invalid |
| counterfactual advantage, mean | counterfactual | 3357 | 0.50 (0.48–0.53) | 0.41 | 0.49 | 0.46 | 0.50 | 0.54 | 0.47 / 0.47 / 0.59 | weak |
| counterfactual advantage at onset | – | 3357 | 0.50 (0.47–0.52) | 0.44 | 0.50 | 0.50 | 0.45 | 0.56 | 0.48 / 0.46 / 0.57 | invalid |
| critic disagreement | – | 3357 | 0.49 (0.46–0.52) | 0.51 | 0.48 | 0.50 | 0.45 | 0.49 | 0.50 / 0.49 / 0.48 | invalid |
| smoothness (−jerk) | – | 3357 | 0.71 (0.69–0.73) | 0.71 | 0.76 | 0.86 | 0.85 | 0.40 | 0.82 / 0.83 / 0.44 | valid |
| yardstick: start time | – | 3357 | 0.59 (0.57–0.61) | 0.60 | 0.49 | 0.49 | 0.45 | 0.54 | 0.63 / 0.60 / 0.51 | yardstick |
| yardstick: end time | – | 3357 | 0.60 (0.58–0.62) | 0.61 | 0.60 | 0.61 | 0.64 | 0.55 | 0.67 / 0.63 / 0.51 | yardstick |
| yardstick: value at onset | – | 3357 | 0.55 (0.52–0.59) | 0.58 | 0.58 | 0.60 | 0.62 | 0.48 | 0.55 / 0.57 / 0.58 | yardstick |

The verdict uses the repo's fixed rule: valid = AUROC ≥ 0.70 with a CI above 0.5; weak = between; invalid = at or below 0.5.

### 3.2 All four campaigns

Number of takeovers in brackets. Each cell is pooled AUROC / second half of each run:

| pooled / 2nd half of each run | validity (Q1) (641) | perfect-filter check (913) | pilot (598) | confirmation (3357) |
|---|---|---|---|---|
| behavioural composite | 0.61 / 0.61 | 0.65 / 0.70 | 0.67 / 0.65 | 0.65 / 0.65 |
| the score as logged (`q_int`) | 0.50 / 0.43 | 0.65 / 0.68 | 0.65 / 0.66 | 0.62 / 0.65 |
| takeover length alone | 0.59 / 0.63 | 0.62 / 0.63 | 0.63 / 0.70 | 0.60 / 0.60 |
| value gain | 0.54 / 0.57 | 0.57 / 0.68 | 0.57 / 0.56 | 0.60 / 0.60 |
| behavioural without length | 0.53 / 0.52 | 0.58 / 0.67 | 0.58 / 0.58 | 0.59 / 0.60 |
| peak force | 0.55 / 0.52 | 0.57 / 0.57 | 0.60 / 0.51 | 0.56 / 0.53 |
| hindsight advantage | 0.53 / 0.63 | 0.55 / 0.62 | 0.55 / 0.62 | 0.55 / 0.59 |
| episode outcome | 0.56 / 0.55 | 0.60 / 0.60 | 0.55 / 0.48 | 0.59 / 0.57 |
| consistency (σ_β) | 0.62 / 0.60 | 0.60 / 0.63 | 0.57 / 0.53 | 0.60 / 0.59 |
| progress after hand-back | 0.51 / 0.48 | 0.48 / 0.54 | 0.50 / 0.43 | 0.50 / 0.53 |
| progress during takeover | 0.53 / 0.49 | 0.47 / 0.46 | 0.50 / 0.40 | 0.49 / 0.52 |
| counterfactual advantage, mean | 0.44 / 0.37 | 0.45 / 0.43 | 0.48 / 0.53 | 0.50 / 0.41 |
| counterfactual advantage at onset | 0.46 / 0.42 | 0.47 / 0.48 | 0.52 / 0.46 | 0.50 / 0.44 |
| critic disagreement | 0.50 / 0.49 | 0.50 / 0.42 | 0.47 / 0.51 | 0.49 / 0.51 |
| smoothness (−jerk) | 0.75 / 0.79 | 0.77 / 0.72 | 0.79 / 0.76 | 0.71 / 0.71 |
| yardstick: start time | 0.55 / 0.62 | 0.58 / 0.67 | 0.51 / 0.46 | 0.59 / 0.60 |
| yardstick: end time | 0.58 / 0.64 | 0.61 / 0.65 | 0.55 / 0.54 | 0.60 / 0.61 |
| yardstick: value at onset | 0.55 / 0.48 | 0.55 / 0.53 | 0.53 / 0.60 | 0.55 / 0.58 |

By tier, pooled:

| pooled, expert / intermediate / novice | validity (Q1) | perfect-filter check | pilot | confirmation |
|---|---|---|---|---|
| behavioural composite | 0.68 / 0.58 / 0.51 | 0.67 / 0.71 / 0.56 | 0.69 / 0.67 / 0.61 | 0.73 / 0.63 / 0.53 |
| takeover length alone | 0.69 / 0.57 / 0.53 | 0.70 / 0.68 / 0.50 | 0.73 / 0.61 / 0.54 | 0.72 / 0.64 / 0.51 |
| counterfactual advantage | 0.41 / 0.55 / 0.38 | 0.35 / 0.44 / 0.71 | 0.34 / 0.60 / 0.78 | 0.47 / 0.47 / 0.59 |
| smoothness (−jerk) | 0.80 / 0.80 / 0.66 | 0.84 / 0.81 / 0.67 | 0.81 / 0.91 / 0.71 | 0.82 / 0.83 / 0.44 |

### 3.3 Is 0.66 a number the online score can reach?

The audit's behavioural composite normalises with statistics of the whole run, which a live system cannot do. Two checks show this does not
inflate the result:

- **The filter's own live score, logged during training,** reaches the same level:

| campaign | score as logged during training | takeovers | AUROC | expert | intermediate | novice |
|---|---|---|---|---|---|---|
| pilot | `drop_score` (what the filter ranks) | 341 | 0.70 | 0.72 | 0.77 | 0.56 |
| confirmation | `drop_score` (what the filter ranks) | 1657 | 0.66 | 0.74 | 0.67 | 0.53 |
| pilot | `q_int` (what re-weighting replays by) | 257 | 0.61 | 0.67 | 0.55 | 0.53 |
| confirmation | `q_int` (what re-weighting replays by) | 1700 | 0.61 | 0.67 | 0.61 | 0.54 |

- **The machine's replay of the real filter on Q1** (read-only, 641 takeovers, running normalisation) gives 0.591 against the audit's 0.606, a
  small loss. It replays a finished run, so it does not show what a retrained policy would do:

| score, replayed as it would run live | tier | AUROC, all takeovers | takeovers after the warm-up | inattentive (= precision of a random drop) | dropped | precision | recall |
|---|---|---|---|---|---|---|---|
| behavioral | all | 0.591 | 297 | 0.41 | 0.41 | 0.48 | 0.48 |
| behavioral | expert |  | 85 | 0.38 | 0.45 | 0.58 | 0.69 |
| behavioral | intermediate |  | 116 | 0.38 | 0.43 | 0.48 | 0.55 |
| behavioral | novice |  | 96 | 0.48 | 0.36 | 0.37 | 0.28 |
| hindsight | all | 0.554 | 297 | 0.41 | 0.35 | 0.44 | 0.38 |
| hindsight | expert |  | 85 | 0.38 | 0.25 | 0.52 | 0.34 |
| hindsight | intermediate |  | 116 | 0.38 | 0.47 | 0.41 | 0.50 |
| hindsight | novice |  | 96 | 0.48 | 0.31 | 0.43 | 0.28 |
| jerk | all | 0.751 | 297 | 0.41 | 0.45 | 0.61 | 0.66 |
| jerk | expert |  | 85 | 0.38 | 0.40 | 0.68 | 0.72 |
| jerk | intermediate |  | 116 | 0.38 | 0.56 | 0.58 | 0.86 |
| jerk | novice |  | 96 | 0.48 | 0.35 | 0.59 | 0.43 |

In that replay the behavioural filter's novice precision (0.37) is *below* a random drop's (0.48). The jerk row is a development result: jerk was chosen
after seeing Q1, it has no online mode, and the machine's report says so.

### 3.4 Reading the scores

Definitions and formulas: section 3.0.

- **Behavioural is moderate, and mostly takeover length.** Without its length term it falls from 0.66 to 0.61 within run. Inattentive takeovers really do last longer
  (43.7 vs 37.5 steps).
- **No contradiction with the timing audit.** That audit asked whether a score predicts *episode success* on clean takeovers, and found no signal. Here the
  label is the hidden condition.
- **Counterfactual advantage: at or below chance, and lower once the critics are trained.** Pooled 0.44, 0.45, 0.48 and 0.50 in the four campaigns (Q1's CI is 0.39–0.50, verdict invalid);
  in the second half of the runs 0.37, 0.43, 0.53 and 0.41. The pilot's 0.53 is the exception, on far fewer takeovers. Its tier values swing widely between campaigns (novice 0.38 to 0.78), so
  they should not be read one by one.
- **Smoothness is circular, and its novice result is an artefact.** It detects the motor noise that defines inattention. Its novice AUROC is 0.66–0.71 in the three 1-per-tier
  campaigns and 0.44 in the confirmation; section 2 shows why. It says nothing about real operators.
- **Consistency (σ_β)** reaches 0.60, with no online score mode for discarding; it is listed for completeness.

**The filter in training:**

| campaign | arm / tier | takeovers after the warm-up | inattentive | dropped | precision | lift | recall | attentive lost | inattentive kept |
|---|---|---|---|---|---|---|---|---|---|
| confirmation | drop-hind | 649 | 0.40 | 0.43 | 0.54 | 1.35 | 0.57 | 0.33 | 0.30 |
| confirmation | drop-hind / expert | 202 | 0.34 | 0.38 | 0.58 | 1.71 | 0.65 | 0.24 | 0.19 |
| confirmation | drop-hind / intermediate | 185 | 0.38 | 0.39 | 0.58 | 1.50 | 0.59 | 0.27 | 0.26 |
| confirmation | drop-hind / novice | 262 | 0.46 | 0.48 | 0.50 | 1.07 | 0.52 | 0.45 | 0.43 |
| pilot | drop-hind | 171 | 0.47 | 0.42 | 0.69 | 1.48 | 0.62 | 0.24 | 0.30 |
| pilot | drop-hind / expert | 114 | 0.34 | 0.33 | 0.61 | 1.77 | 0.59 | 0.20 | 0.21 |
| pilot | drop-hind / intermediate | 28 | 0.68 | 0.68 | 0.79 | 1.16 | 0.79 | 0.44 | 0.44 |
| pilot | drop-hind / novice | 29 | 0.76 | 0.52 | 0.80 | 1.05 | 0.55 | 0.43 | 0.71 |
| perfect-filter check | drop-oracle | 110 | 0.45 | 0.45 | 1.00 | 2.24 | 1.00 | 0.00 | 0.00 |
| perfect-filter check | drop-oracle / expert | 70 | 0.33 | 0.33 | 1.00 | 3.04 | 1.00 | 0.00 | 0.00 |
| perfect-filter check | drop-oracle / intermediate | 6 | 0.00 | 0.00 | – | – | – | 0.00 | 0.00 |
| perfect-filter check | drop-oracle / novice | 34 | 0.76 | 0.76 | 1.00 | 1.31 | 1.00 | 0.00 | 0.00 |
| perfect-filter check | drop-random | 119 | 0.50 | 0.47 | 0.41 | 0.83 | 0.39 | 0.55 | 0.57 |
| perfect-filter check | drop-random / expert | 86 | 0.36 | 0.42 | 0.19 | 0.54 | 0.23 | 0.53 | 0.48 |
| perfect-filter check | drop-random / intermediate | 13 | 0.69 | 0.54 | 0.57 | 0.83 | 0.44 | 0.75 | 0.83 |
| perfect-filter check | drop-random / novice | 20 | 0.95 | 0.65 | 0.92 | 0.97 | 0.63 | 1.00 | 1.00 |

Precision = dropped takeovers that were inattentive; lift = precision over the inattentive share. A random drop has lift 1 and the oracle has precision 1. The tier rows of the check
and the pilot rest on very few takeovers. The check's random arm has precision 0.41 against a base rate of 0.50 by chance, and its novice cell had 95 % inattentive takeovers
after the warm-up.

**Exposure, cell by cell (`two_arm_confirm_new`):**

| cell (`drop-hind`) | takeovers | during the warm-up | after it | inattentive after it | dropped |
|---|---|---|---|---|---|
| exp00 s0 | 167 | 86 | 81 | 33 % | 25 |
| exp00 s1 | 110 | 68 | 42 | 33 % | 17 |
| exp01 s0 | 77 | 70 | 7 | 71 % | 6 |
| exp01 s1 | 125 | 85 | 40 | 22 % | 14 |
| exp02 s0 | 101 | 97 | 4 | 25 % | 1 |
| exp02 s1 | 92 | 64 | 28 | 46 % | 14 |
| int00 s0 | 115 | 45 | 70 | 43 % | 24 |
| int00 s1 | 53 | 49 | 4 | 0 % | 1 |
| int01 s0 | 105 | 53 | 52 | 37 % | 28 |
| int01 s1 | 65 | 58 | 7 | 0 % | 3 |
| int02 s0 | 74 | 67 | 7 | 86 % | 4 |
| int02 s1 | 93 | 48 | 45 | 36 % | 13 |
| nov00 s0 | 102 | 37 | 65 | 46 % | 33 |
| nov00 s1 | 77 | 35 | 42 | 33 % | 23 |
| nov01 s0 | 89 | 37 | 52 | 56 % | 27 |
| nov01 s1 | 78 | 34 | 44 | 34 % | 20 |
| nov02 s0 | 70 | 37 | 33 | 70 % | 12 |
| nov02 s1 | 64 | 38 | 26 | 38 % | 12 |
| **all** | 1657 | 1008 | 649 | 40 % | 277 |

The filter acted on 649 of 1 657 takeovers and dropped 277, which is 17 % of all takeovers and 25 % of operator steps. Five cells had 7 or fewer takeovers after the warm-up. After the warm-up the inattentive share
ranges from 0 to 86 % by cell.

## 4. Does using the score help?

### How the arms act on training (from the configs and `runtime.py`)

Every arm is the fixed 8-critic ensemble in quality mode. Every learner batch is half online experience and half a
demonstration buffer (the demonstrations plus takeovers). No arm has an imitation loss (`bc_loss_weight: 0`), and the
filters are set explicitly, so the outcome filter and the safety veto are **off**. That is the clean contrast with the
first machine's oracle campaign, where both the outcome filter and an imitation loss were on
([first_machine_oracle_quality_results_and_qa_log.md](first_machine_oracle_quality_results_and_qa_log.md)).

| arm | what happens to a takeover |
|---|---|
| keep-all | enters the demonstration buffer; uniform priority |
| discard (`drop-hind`) | after the 5 000-step warm-up and once 20 scores are in: dropped from the demonstration buffer if its score is below the 40th percentile of the last 200 scores; its steps stay in the online half; uniform priority |
| perfect filter (`drop-oracle`) | after the warm-up: dropped if it was inattentive |
| random drop | after the warm-up: dropped with probability 0.4 |
| re-weight (`reweight-hind`) | always kept; replay priority rises with the score (standardised, refreshed every 2 000 steps, importance weights β 0.5) |

**The score the arms used** is the behavioural composite, z(Δvalue) − z(t_int) − z(f_peak): the change of the critics'
value estimate from takeover to hand-back, minus the takeover's length, minus its peak force, each standardised online. A
short, value-raising, gentle takeover scores high. The names `drop-hind` and `reweight-hind` predate the choice; they
do not use hindsight.

### 4.1 Discard against re-weight (`two_arm_confirm_new`)

The same 9 operators × 2 seeds.

| | discard (`drop-hind`) | re-weight (`reweight-hind`) | paired, re-weight − discard |
|---|---|---|---|
| ruler SR | 93.2 % | 90.3 % | **−2.9 points**, CI −13.4 to +6.2; 8 / 4 / 6; p 0.39 |
| expert / intermediate / novice | 92.6 / 93.2 / 93.8 % | 92.0 / 95.7 / 83.3 % | |
| SR of the last checkpoint ¹ | 97.5 % | 87.0 % | -10.5 points, CI -25.3 to +1.9; 4 / 4 / 10; p 1.00 |
| steps to converge | 9 264 | 9 359 | +94, CI −1 361 to +1 411 |
| first unaided success (median) | 4 174 | 5 442 | later in 13 of 18, p 0.096 |
| operator share | 41.1 % | 39.4 % | −1.6 points; lower in 13 of 18, p 0.096 |
| SD of SR over cells | 0.107 | 0.171 | |
| ICC | 0.000 | 0.000 | |

¹ The policy training stopped on, 9 trials per cell, from `cells.csv`; a secondary measure, not fixed in advance. The
ruler SR is the best window of three checkpoints.

The side measures point in opposite directions: first unaided success favours discard, operator share favours re-weight. The arm names are historical: both used the
behavioural score, not hindsight. Re-weight's priority was positive for 48 % of takeovers (54 % of attentive ones, 35 % of inattentive ones).

**Every pair** (SR and shares in %):

| cell | ruler SR, discard | re-weight | last ckpt, discard | re-weight | first unaided, discard | re-weight | converged, discard | re-weight | operator share, discard | re-weight |
|---|---|---|---|---|---|---|---|---|---|---|
| exp00 s0 | 100 | 67 | 100 | 100 | 7 321 | 7 799 | 9 397 | 9 840 | 46 | 43 |
| exp00 s1 | 100 | 100 | 100 | 100 | 2 203 | 4 943 | 7 247 | 9 007 | 38 | 34 |
| exp01 s0 | 63 | 89 | 89 | 100 | 2 803 | 3 469 | 5 223 | 5 477 | 40 | 45 |
| exp01 s1 | 100 | 100 | 100 | 100 | 2 250 | 2 721 | 7 201 | 7 272 | 44 | 42 |
| exp02 s0 | 93 | 96 | 89 | 100 | 3 055 | 3 324 | 5 109 | 5 333 | 57 | 45 |
| exp02 s1 | 100 | 100 | 100 | 100 | 5 495 | 4 100 | 7 501 | 8 156 | 34 | 39 |
| int00 s0 | 100 | 100 | 100 | 100 | 10 503 | 1 266 | 12 548 | 11 333 | 36 | 40 |
| int00 s1 | 82 | 96 | 89 | 100 | 2 448 | 5 856 | 5 305 | 7 876 | 38 | 38 |
| int01 s0 | 93 | 96 | 100 | 100 | 7 279 | 9 212 | 9 315 | 11 215 | 40 | 32 |
| int01 s1 | 100 | 100 | 100 | 100 | 3 876 | 5 027 | 5 882 | 7 028 | 42 | 32 |
| int02 s0 | 85 | 100 | 89 | 100 | 2 113 | 7 814 | 5 278 | 10 009 | 55 | 44 |
| int02 s1 | 100 | 82 | 100 | 33 | 9 051 | 7 360 | 11 057 | 9 553 | 33 | 46 |
| nov00 s0 | 74 | 100 | 100 | 100 | 8 926 | 6 865 | 15 239 | 17 376 | 45 | 40 |
| nov00 s1 | 100 | 56 | 100 | 33 | 629 | 4 511 | 11 326 | 6 586 | 42 | 41 |
| nov01 s0 | 100 | 44 | 100 | 11 | 3 618 | 6 593 | 16 433 | 8 596 | 31 | 38 |
| nov01 s1 | 96 | 100 | 100 | 100 | 5 480 | 3 081 | 11 573 | 7 112 | 40 | 37 |
| nov02 s0 | 100 | 100 | 100 | 89 | 8 414 | 10 453 | 12 427 | 14 578 | 35 | 34 |
| nov02 s1 | 93 | 100 | 100 | 100 | 4 471 | 8 099 | 8 699 | 12 111 | 45 | 40 |

**Dependency reports of all campaigns:**

| campaign | arm | cells | mean ruler SR | SD over cells | ICC | SR by tier (expert / intermediate / novice, %) | worst operator | p (operator matters) |
|---|---|---|---|---|---|---|---|---|
| validity (Q1) | `logged` | 6 | 98.1 % | 0.031 | 0.000 | 98.1 / 100.0 / 96.3 | `nov00` 96.3 % | 1.00 |
| perfect-filter check | `keepall` | 3 | 98.8 % | 0.021 | – | 100.0 / 100.0 / 96.3 | `nov00` 96.3 % | – |
| perfect-filter check | `drop-random` | 3 | 91.4 % | 0.119 | – | 100.0 / 96.3 / 77.8 | `nov00` 77.8 % | – |
| perfect-filter check | `drop-oracle` | 3 | 93.8 % | 0.107 | – | 100.0 / 100.0 / 81.5 | `nov00` 81.5 % | – |
| pilot | `drop-hind` | 3 | 97.5 % | 0.043 | – | 100.0 / 100.0 / 92.6 | `nov00` 92.6 % | – |
| pilot | `reweight-hind` | 3 | 96.3 % | 0.064 | – | 100.0 / 100.0 / 88.9 | `nov00` 88.9 % | – |
| confirmation | `drop-hind` | 18 | 93.2 % | 0.107 | 0.000 | 92.6 / 93.2 / 93.8 | `exp01` 81.5 % | 0.99 |
| confirmation | `reweight-hind` | 18 | 90.3 % | 0.171 | 0.000 | 92.0 / 95.7 / 83.3 | `nov01` 72.2 % | 1.00 |
| smoke | `hilserl` | 1 | 92.6 % | – | – | 92.6 | `exp00` 92.6 % | – |

| campaign | arm | converged | mean steps | median steps | steps ICC | mean steps by tier |
|---|---|---|---|---|---|---|
| validity (Q1) | `logged` | 6/6 | 10 689 | 9 560 | 0.232 | 7 417 / 11 152 / 13 496 |
| perfect-filter check | `keepall` | 3/3 | 9 163 | 8 166 | – | 7 447 / 8 166 / 11 876 |
| perfect-filter check | `drop-random` | 3/3 | 8 634 | 8 200 | – | 11 050 / 6 651 / 8 200 |
| perfect-filter check | `drop-oracle` | 3/3 | 8 177 | 9 082 | – | 9 082 / 5 705 / 9 745 |
| pilot | `drop-hind` | 3/3 | 9 459 | 9 877 | – | 10 663 / 7 836 / 9 877 |
| pilot | `reweight-hind` | 3/3 | 8 062 | 8 401 | – | 8 401 / 7 100 / 8 686 |
| confirmation | `drop-hind` | 18/18 | 9 264 | 9 007 | 0.267 | 6 946 / 8 231 / 12 616 |
| confirmation | `reweight-hind` | 18/18 | 9 359 | 8 802 | 0.053 | 7 514 / 9 502 / 11 060 |
| smoke | `hilserl` | 1/1 | 4 286 | 4 286 | – | 4 286 |

An ICC of 0.000 means the variance between operators estimates to zero here. Near the ceiling and with this much noise between seeds, that says little; see the Q&A.

### 4.2 The pilot (`two_arm_pilot_new`)

3 pairs: discard 97.5 %, re-weight 96.3 %; 0 wins, 1 loss, 2 ties. It carried no decision, because the confirmation was going to run whatever its direction.

| arm | operator | seed | ruler SR % | last ckpt % | first unaided | converged | operator share | takeovers | inattentive | dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| `drop-hind` | exp00 | 0 | 100.0 | 100 | 4 578 | 10 663 | 54 % | 206 | 72 | 38 |
| `drop-hind` | int00 | 0 | 100.0 | 100 | 3 668 | 7 836 | 31 % | 66 | 23 | 19 |
| `drop-hind` | nov00 | 0 | 92.6 | 78 | 7 818 | 9 877 | 43 % | 69 | 26 | 15 |
| `reweight-hind` | exp00 | 0 | 100.0 | 100 | 6 382 | 8 401 | 44 % | 136 | 48 | 0 |
| `reweight-hind` | int00 | 0 | 100.0 | 100 | 2 857 | 7 100 | 37 % | 66 | 23 | 0 |
| `reweight-hind` | nov00 | 0 | 88.9 | 100 | 6 686 | 8 686 | 42 % | 55 | 19 | 0 |

### 4.3 The perfect-filter check (`headroom_diagnostic_new`)

3 pairs: one operator per tier, seed 0. The three seed-0 demonstration sets were copied from Q1, checked by hash, and no trained cell was reused. Primary measure, fixed before the run: paired
ruler SR against keep-all. Rule for a hint of headroom: the oracle beats keep-all in at least two tiers.

| arm | mean ruler SR | mean last checkpoint | ruler, expert / intermediate / novice | last checkpoint | first unaided | converged | takeovers dropped |
|---|---|---|---|---|---|---|---|
| `keepall` | 98.8 % | 96.3 % | 100 / 100 / 96 | 100 / 100 / 89 | 5 438 / 6 161 / 5 585 | 7 447 / 8 166 / 11 876 | 0 / 0 / 0 |
| `drop-oracle` | 93.8 % | 100.0 % | 100 / 100 / 82 | 100 / 100 / 100 | 4 815 / 3 696 / 7 738 | 9 082 / 5 705 / 9 745 | 23 / 0 / 26 |
| `drop-random` | 91.4 % | 100.0 % | 100 / 96 / 78 | 100 / 100 / 100 | 4 693 / 4 637 / 3 947 | 11 050 / 6 651 / 8 200 | 36 / 7 / 13 |

| arm | operator | seed | ruler SR % | last ckpt % | first unaided | converged | operator share | takeovers | inattentive | dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| `drop-oracle` | exp00 | 0 | 100.0 | 100 | 4 815 | 9 082 | 45 % | 162 | 56 | 23 |
| `drop-oracle` | int00 | 0 | 100.0 | 100 | 3 696 | 5 705 | 47 % | 66 | 23 | 0 |
| `drop-oracle` | nov00 | 0 | 81.5 | 100 | 7 738 | 9 745 | 45 % | 74 | 30 | 26 |
| `drop-random` | exp00 | 0 | 100.0 | 100 | 4 693 | 11 050 | 40 % | 174 | 64 | 36 |
| `drop-random` | int00 | 0 | 96.3 | 100 | 4 637 | 6 651 | 47 % | 78 | 32 | 7 |
| `drop-random` | nov00 | 0 | 77.8 | 100 | 3 947 | 8 200 | 43 % | 59 | 23 | 13 |
| `keepall` | exp00 | 0 | 100.0 | 100 | 5 438 | 7 447 | 52 % | 145 | 52 | 0 |
| `keepall` | int00 | 0 | 100.0 | 100 | 6 161 | 8 166 | 30 % | 66 | 23 | 0 |
| `keepall` | nov00 | 0 | 96.3 | 89 | 5 585 | 11 876 | 46 % | 89 | 33 | 0 |

- **By its rule, no headroom:** oracle − keep-all −4.9 points (0 / 1 / 2), random − keep-all −7.4 (0 / 2 / 1), oracle − random +2.5 (2 / 0 / 1), all p ≥ 0.5. All differences come from the novice cell and one intermediate cell.
- **The secondary and last-checkpoint measures disagree with the ruler.** At the last checkpoint both dropping arms are at 100 % and keep-all at 96.3 %. Random drop reached its first unaided success earlier than keep-all in all three pairs. Three pairs cannot settle this.
- **The novice difference is confounded by the sequence bug.** After the warm-up, the novice cell of the oracle arm saw 76 % inattentive takeovers and dropped 26 of 34; the random arm's saw 95 %. That is the seed-0 spell of section 2.
- **The filter had nothing to do in one pair.** In `drop-oracle_int00` all 6 takeovers after the warm-up were attentive, so nothing was dropped, and the cell is effectively a re-run of `keepall_int00`. It still differs by about 2 500 steps in first unaided success (3 696 vs 6 161), about 2 500 in convergence (5 705 vs 8 166) and 17 points of operator share. That is the noise any effect here must beat.

### 4.4 Q1, cell by cell (`Q1_validity_new`)

Ensemble extended (`logged`), 1 operator per tier × 2 seeds: SR 98.1 %, converged at 10 689 steps (mean), first unaided at 4 599 (median), operator share 40.5 %.

| arm | operator | seed | ruler SR % | last ckpt % | first unaided | converged | operator share | takeovers | inattentive |
|---|---|---|---|---|---|---|---|---|---|
| `logged` | exp00 | 0 | 96.3 | 100 | 4 126 | 6 316 | 58 % | 141 | 52 |
| `logged` | exp00 | 1 | 100.0 | 100 | 4 453 | 8 518 | 36 % | 120 | 32 |
| `logged` | int00 | 0 | 100.0 | 100 | 2 467 | 9 125 | 29 % | 72 | 28 |
| `logged` | int00 | 1 | 100.0 | 100 | 4 745 | 13 180 | 45 % | 140 | 40 |
| `logged` | nov00 | 0 | 100.0 | 100 | 7 992 | 9 994 | 41 % | 63 | 23 |
| `logged` | nov00 | 1 | 92.6 | 100 | 14 995 | 16 999 | 34 % | 105 | 30 |

This campaign also shows how much a filter on episode outcome would hit novices. It only marks the takeovers here; `logged` keeps them all:

| tier (validity campaign, 6 cells) | takeovers | marked by the default filters | share | takeovers in failed episodes |
|---|---|---|---|---|
| expert | 261 | 66 | 25 % | 18 % |
| intermediate | 212 | 102 | 48 % | 38 % |
| novice | 168 | 129 | 77 % | 74 % |

## 5. Why this was a weak test

1. **The filter started too late.** The 5 000-step warm-up covered 61 % of the takeovers and 58 % of
   the operator steps, and half the cells succeeded unaided before it ended. It dropped **17 % of all
   takeovers, not 40 %** (25 % of operator steps; 11 / 14 / 26 % of takeovers for expert / intermediate /
   novice). In some cells it barely acted: `drop-hind_exp02_s0` had 4 takeovers after the warm-up.
2. **One condition sequence per seed, in every campaign.** See section 2. Early learning, when corrections matter most, saw few bad corrections, and exposure varies with tier
   and phase. Patch 0021 fixes it.
3. **The ruler is near its ceiling and noisy.** Across pairs the two arms' SR
   correlates at −0.21, so pairing does not help; ±5 points would take about 75 pairs. Steps to converge pair better (+0.56; about 38 pairs for
   ±1 000 steps), so they are the better primary measure on this stack, with the operator share.
4. **Replicates differ a lot.** Only `exp00`, seed 0, ran with the same operator and the same demonstrations in two campaigns (the caps differ, 30 k and 40 k, but every run stopped long before them):

   | `exp00`, seed 0, pilot / confirmation | ruler SR % | last checkpoint % | first unaided | converged | operator share |
   |---|---|---|---|---|---|
   | `drop-hind` | 100 / 100 | 100 / 100 | 4 578 / 7 321 | 10 663 / 9 397 | 54 / 46 % |
   | `reweight-hind` | 100 / 67 | 100 / 100 | 6 382 / 7 799 | 8 401 / 9 840 | 44 / 43 % |

   One ruler value falls from 100 to 67 %, although the last checkpoint is 100 % in both. First unaided success moves by up to 2 700 steps. Two pairs give a feel for the noise, not an estimate of it.
5. **The stop rule catches the policy mid-swing.** Neighbouring checkpoints often differ wildly (for
   example 0.00, 1.00, 1.00; 6.6 % of the steps change by at least 0.5). Of the 5 cells below 0.8, 3 were at
   0.89 or more on their last checkpoint. Three cells, all re-weight, had collapsed to 0.33 or less at the
   stop (`int02_s1`, `nov00_s1`, `nov01_s0`).
6. **No control at scale:** keep-all ran on 3 pairs, plain HIL-SERL on none. The lean design compares two
   uses of the score, not the score with nothing.
7. **The novice's inattentive condition is not clearly worse,** partly for the model reason in section 2.
8. **BC weight was 0.** These arms use replay only. Whether a score helps imitation is untested here.

## 6. Against the first machine's oracle campaign

`aqi_oracle_quality` on the first machine (old stack, consistent operator) concluded that even a perfect
score amplifies operator dependence, because its novice falls from 65 to 32 %. The two results differ, and
**the first machine's conclusion is not yet established**, because its quality arms changed three things
at once: the oracle score, the default hard filters (`outcome_filter` drops every takeover from a failed
episode, whatever the oracle says) and a BC loss of 0.1. See [docs/novice_ablation.md](../novice_ablation.md).

Here, without those two extras, score-based discarding kept novices at 93.8 %. That is suggestive, but it
is another stack and another machine, and only 26 % of novice takeovers were dropped. The novice test on
the first machine separates the three parts on the old stack, on the code that ran the campaign.

## 7. Verdict

| question | answer | strength |
|---|---|---|
| Does the inconsistent operator work as designed? | yes for expert, mostly for intermediate; no harm for novices in training, partly a model artefact | strong for expert; caveat for the rest |
| Do scores reflect correction quality? | moderately for expert and intermediate; hardly for novices; mostly via takeover length | moderate; label caveat (2 sequences) |
| Is the online 0.66 real? | yes: the filter's live score gives the same in the confirmation | strong |
| Is the counterfactual score useful? | no: at or below chance, lower late in training, on a timing-free label too | strong |
| Which use of the score is better? | no detectable difference | weak: wide CI |
| Can a perfect filter help? | undecided: the ruler and the last checkpoint disagree on 3 confounded pairs | very weak |
| Does quality beat the baseline? | not tested | – |
| Does quality help more with inconsistency? | not tested (no consistent-operator run) | – |

## 8. Q&A log

**Did the experiment test the hypothesis?** Only in part. It tested whether the scores notice the bad
corrections and which use of the score is better. "Beats the baseline" and "helps more because the
operator is inconsistent" need `hilserl`, keep-all and a consistent-operator arm, which the two-arm design
left out on purpose.

**Why the behavioural score if its verdict was "weak"?** It was the best score that can run online. The
original gate stopped there; the two-arm run went ahead on the user's explicit decision, with the weak
verdict kept. No threshold was lowered afterwards.

**Why was the keep-all / oracle / random check run?** Not in the plan: the machine's Claude proposed it after
Q1 and the user agreed. It asks whether removing known-bad corrections can help at all.

**Did the pilot decide that the confirmation should run?** No. On the user's instruction, the gates after Q1 only reported.

**Why did novices show no effect from the condition?** Three reasons. Inattentive execution is close to the novice's
own execution, and for some novices smoother. Only the aim offset and premature commits differ, and novice takeovers mostly fail anyway
(23 % episode success). The rosters differ too: read section 2.

**Does ICC 0.000 mean the operator no longer matters?** Not on this evidence. The ruler is near its ceiling and seeds vary a lot, so the between-operator variance estimates to zero. Plain HIL-SERL, the arm
whose dependency is the question, did not run on the cuboid.

**Why does `drop-hind` not use hindsight?** The name dates from before Q1. Only `quality_score_mode` was
edited, so that the arms stay exactly as designed.

**Can these cells be merged with new ones?** Not with cells from other machines, and not for label analyses
with new code, since the sequences have changed.

## 9. What the check of 2026-11-03 changed in this log

- **Added:** how the study ran and who decided (section 1); the whole confirmation score table and all four campaigns (3.1, 3.2); the live score and the machine's replay of the filter (3.3); filter exposure per cell (3.4); every pair and all dependency reports (4.1); per-cell tables of Q1, the pilot and the check (4.2–4.4); the seed-sequence table; novice parameters against the anchors.
- **Corrected:**
  - "Smoothness fails for novices, which is why…" was wrong. It holds only in the confirmation (0.44 pooled, 0.40 within). The three 1-per-tier campaigns give 0.66–0.71. Section 2 has the cause.
  - "A perfect filter showed no gain" was too strong. The ruler and the last checkpoint disagree on those three pairs, and the sequence bug confounds the novice cell.
  - "The counterfactual score stays at chance" was incomplete. It is at or below chance, and lower late in training in three of four campaigns.
  - "The same cell run twice" rests on one operator only: `exp00`. `int00` and `nov00` are different operators between those campaigns.
- **Restored:** the consistency score row. My re-run of 2026-10-31 had lost it because an import failed silently in that environment; all other numbers of the re-run were unchanged.
- **Confirmed unchanged:** every other headline number of this log.
