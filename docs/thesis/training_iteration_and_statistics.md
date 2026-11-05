# One training iteration, the buffers, what each critic sees, and the statistical tests

How the learner works, with the numbers in the code, and how the numbers on the result slides are tested, with
simple worked examples. Written on 5 November 2026 for the mid-term presentation. The slides are the backups of the
Siemens deck (`presentation_bundle/output/AQI-HIL-SERL_Siemens_short.pptx`): page 58 (one training iteration), 59
(what each critic sees), 60 (the learner in formulas), 63 (the statistics); their speaker notes carry the same
examples.

**How to read it.** Open it in VS Code and press `Ctrl+Shift+V` for the rendered preview (formulas show as code, tables
as tables), or open it on GitHub (`docs/thesis/` in `ytelha/hil-serl_aqi`). Sections 1–5 are the learner, section 6 is
the statistics, and section 7 lists where each piece is in the code. Symbols are explained where they first appear.

---

## 1. The two processes

HIL-SERL, and every method built on it here, runs two processes at the same time:

- the **actor** drives the robot (in simulation): it observes, acts, steps the simulator and stores what happened;
- the **learner** samples stored data and updates the networks: the **critics** (Q-networks, "how much success do I
  expect from here if I do this action?") and the **actor network**, also called the policy ("which action do I take?").

The actor gets fresh policy weights from the learner every 50 learner steps (`steps_per_update`).

## 2. One environment step, on the actor

| step | what happens | numbers |
|---|---|---|
| observe | two wrist-camera images and the robot's own state (proprioception) | cameras `wrist_1`, `wrist_2` |
| act | the policy picks an action; during the first steps the action is random | random for the first 200 steps (`random_steps`) |
| takeover | if the synthetic operator's monitor fires (or, in the asking arms, the robot asks and the operator answers), the operator's action replaces the policy's | |
| step | the simulator advances | 0.1 s; reward 1 when the peg is inserted, else 0 |
| store | the step becomes one **transition** (s, a, r, s′, done): state, action, reward, next state, episode over | |

Symbols: s = state (what the robot sees), a = action, r = reward, s′ = the next state, done = 1 if the episode ended.

## 3. The buffers: where a transition goes

| buffer | what goes in | when | size | sampled |
|---|---|---|---|---|
| **online buffer** | every transition, the operator's included | immediately | up to 200 000 | uniformly |
| **intervention buffer** (also called the demo buffer) | the operator's 6 demonstrations (about 250 transitions) + every step in which the operator was in control | demonstrations before training; takeovers immediately | up to 200 000 | uniformly |
| intervention buffer, **grading arms** (quality, AQI) | only the takeovers the score keeps | at the end of the episode, after the score (and, on the first machine, the filters) | up to 200 000 | by a **replay priority** from the score |

In the grading arms a takeover waits in an episode buffer until the episode ends, because the score needs to know how
the episode went. Then it is scored, and either kept or dropped (discard), or kept with a sampling weight (re-weight,
replay priority). The online buffer is not affected: it keeps every transition.

**Bootstrap masks (ensemble arms only).** When a transition is stored, it gets one mask bit per critic: 1 with
probability p = 0.8 (`critic_bootstrap_p`), else 0. The bits are drawn once and never change. A critic learns only
from the transitions whose bit is 1 for it. Plain HIL-SERL has no masks (the baseline architecture).

Example with 3 critics and 4 transitions:

| transition | critic 1 | critic 2 | critic 3 |
|---|---|---|---|
| A | 1 | 1 | 0 |
| B | 1 | 0 | 1 |
| C | 1 | 1 | 1 |
| D | 0 | 1 | 1 |

Critic 1 learns from A, B, C; critic 2 from A, C, D; critic 3 from B, C, D. Each sees a different 3 of 4 (on average
80 %), so where data is thin the critics learn different functions, and their spread σ_Q measures what the robot has
not seen.

## 4. One learner step

**The batch.** Every batch has 256 transitions: 128 from the online buffer and 128 from the intervention buffer (the
RLPD recipe: half own experience, half the operator's).

**Two updates per learner step** (`cta_ratio` = 2, critic-to-actor ratio), each on a fresh batch:

1. **update 1:** the critics only;
2. **update 2:** the critics again, then the actor and the temperature.

**The critic update.** For a transition (s, a, r, s′, done) the target is

```
y = r + γ · (1 − done) · Q̄(s′, a′),     a′ drawn from the policy at s′,     γ = 0.97
```

- γ (gamma) = the discount: success one step later is worth 0.97 of success now.
- Q̄ = a **target critic**: a slow copy of a critic, used so the target does not chase itself.
- Which target critic:
  - **plain HIL-SERL** (K = 2 critics, M = 2): the smaller of the two target critics (clipped double Q), a pessimistic,
    steady target shared by both critics;
  - **ensemble extended** (K = 8, M = 1, per-member targets): every critic draws one target critic at random, separately,
    at every update.
- No entropy term in the target (`backup_entropy` False).

Each critic k then moves its prediction Q_k(s, a) towards y: the loss is the squared difference (Q_k(s, a) − y)²,
averaged over the transitions of the batch that its mask allows.

**The target critics** follow the critics slowly after every critic update:

```
Q̄ ← 0.995 · Q̄ + 0.005 · Q          (soft update, rate 0.005)
```

**The actor update.** The actor draws an action a for every state s of the batch and is pushed towards actions the
critics like, while staying random enough to explore:

```
actor loss = mean over the batch of [ α · log π(a | s) − (1/K) · Σ_k Q_k(s, a) ]
```

- π(a | s) = the probability the policy gives the action a; − log π is its randomness (entropy).
- (1/K) Σ_k Q_k = the mean of all critics (2 for HIL-SERL, 8 for the ensemble). E2HiL uses the minimum instead, plus its
  entropy filter.
- α (alpha) = the **temperature**: how much randomness is worth.

**The temperature update.** α is tuned automatically (a Lagrange penalty): it rises when the policy's randomness falls
below a target, and falls when it is above.

**Publish.** Every 50 learner steps the learner sends the new policy weights to the actor.

## 5. The pace, and how much data each critic sees

### 5.1 How many updates per environment step

- **New stack (second machine, the next study):** the fixed update ratio (`--utd_ratio 1`) gives exactly **one learner
  step per environment step**: 2 critic updates and 1 actor update, on 2 × 256 = 512 sampled transitions. If the
  learner is ahead, it waits for data; if it falls behind, the actor waits between episodes.
- **First machine (original platform):** no fixed ratio. The learner ran as fast as the GPU allowed, about **0.45
  learner steps per environment step** (about 0.9 critic updates). Fair between methods, because every comparison has
  its reference on the same machine, but never compare runs across the machines.

The learner starts once the online buffer holds 100 transitions (`training_starts`).

### 5.2 The warm start (ensemble arms)

Before online training, the ensemble arms learn from the demonstrations alone: 5 000 learner steps, each with the same
2 updates, on batches of 256 from the demonstrations, at learning rate 1.5 · 10⁻⁴. Plain HIL-SERL has no warm start and
learns at 3 · 10⁻⁴. (The shorter ruler skips the checkpoints of this phase when it scores a run; the training still
does it.)

### 5.3 What each critic sees

| | plain HIL-SERL | ensemble extended |
|---|---|---|
| critics | 2 | 8 (next study: 4, 8, 16) |
| data a critic learns from | all 256 transitions of a batch | its own 80 %: about 205 of 256 |
| target | the smaller of the 2 target critics | one random target critic, per critic |
| updates per environment step (fixed ratio) | 2 critic, 1 actor | the same |
| transitions per critic per environment step | 512 | about 410 |
| warm start | none | 5 000 learner steps: 5 000 × 2 × 256 × 0.8 ≈ 2 M samples per critic |
| learning rate | 3 · 10⁻⁴ | 1.5 · 10⁻⁴ |
| the actor follows | the mean of 2 critics | the mean of 8 critics |

### 5.4 Worked example: a run that stops after 9 000 environment steps (fixed ratio)

- **Updates:** 9 000 learner steps = 18 000 critic updates and 9 000 actor updates.
- **Sampled transitions per critic:**
  - HIL-SERL: 9 000 × 512 ≈ **4.6 M**.
  - Ensemble: 9 000 × 410 ≈ **3.7 M** online, plus ≈ **2 M** in the warm start.
- **What is in the buffers:** say the operator controlled 4 000 of the 9 000 steps. The online buffer then holds 9 000
  transitions, and the intervention buffer 250 + 4 000 = 4 250.
- **How often a correction is replayed:** per learner step 256 draws go to each buffer.
  - A given correction is drawn with chance 256 / 4 250 = 0.060 from the intervention buffer, plus 256 / 9 000 = 0.028
    from the online buffer: 0.088 in total.
  - An own step is drawn only from the online buffer: 0.028.
  - So a correction is replayed about **three times** as often as an own step.
- **Early data is replayed most.** The online buffer grows by one transition per step, so a transition stored at step t
  is drawn about 256 × ln(T / t) times by step T:
  - stored at step 1 000: 256 × ln 9 ≈ **560** times by step 9 000;
  - stored at step 8 000: 256 × ln(9/8) ≈ **30** times.
  - For an ensemble critic it is all or nothing: the same count if its mask bit is 1, zero if it is 0.

## 6. The statistical tests, with worked examples

Every claim rests on several runs. A **run** (cell) is one method, one operator, one seed. The **ruler** scores every
run the same way: success on 9 fixed starting poses. **pp** = percentage points (87 % − 74 % = 13 pp).

### 6.1 ICC: how much of the differences comes from who taught

**Example campaign:** three teachers, three seeds each, success in %:

| teacher | seed 1 | seed 2 | seed 3 | average |
|---|---|---|---|---|
| expert | 96 | 85 | 90 | 90.3 |
| intermediate | 80 | 66 | 74 | 73.3 |
| novice | 58 | 70 | 48 | 58.7 |

Overall average: 74.1. Number of teachers g = 3, runs per teacher n = 3, all runs N = 9.

1. **Between teachers** (how far the teacher averages lie apart):
   MS_between = n · Σ (teacher average − overall)² / (g − 1)
   = 3 · (16.2² + 0.8² + 15.4²) / 2 = **753.4**.
2. **Within teachers** (the seed spread):
   MS_within = Σ (run − its teacher's average)² / (N − g)
   = (60.7 + 98.7 + 242.7) / 6 = 402 / 6 = **67.0**.
3. **Teacher variance:** σ²_teacher = (MS_between − MS_within) / n = (753.4 − 67.0) / 3 = **228.8**.
4. **ICC** = σ²_teacher / (σ²_teacher + MS_within) = 228.8 / (228.8 + 67.0) = **0.77**.

Read: 77 % of the differences between runs come from who taught; 0 = the teacher does not matter, 1 = only the teacher
matters. If MS_between is smaller than MS_within, the teacher variance would be negative and is set to 0: that is how
"ICC 0.00" arises. With only three operators per tier the ICC of one campaign is unstable, so it is compared only within
a campaign. (This is ICC(1,1), the one-way model; `icc_oneway` in the code.)

### 6.2 Permutation test: could the teacher effect be chance?

1. **Observed:** the spread (variance) of the three teacher averages, 90.3, 73.3 and 58.7: **251.1**.
2. **Shuffle:** forget which run belongs to which teacher; deal the 9 numbers at random into three groups of three, and
   compute the spread of the three new averages. Repeat 10 000 times.
3. **Count:** how many shuffles reach at least 251.1? **71** of 10 000.
4. **p = (71 + 1) / (10 000 + 1) = 0.007.**

Read: if the teacher did not matter, a spread this large would arise by chance in about 7 of 1 000 shuffles. (There are
only 1 680 different ways to deal 9 runs into three groups of three; exactly 12 of them reach 251.1, 12 / 1 680 =
0.007, the same answer.) `permutation_sigma2` in the code; the slides use 10 000 shuffles.

### 6.3 Sign test: is method A better than method B?

Pairs: the same operator and seed, run with both methods. Count who wins each pair.

**Example** (asking in the 55 % most uncertain steps against ensemble extended, 9 operators × 2 seeds = 18 pairs):
13 wins, 3 losses, 2 ties.

1. Drop the ties: n = 16 pairs.
2. If both methods were equal, each pair is a coin flip. The chance of 3 or fewer losses out of 16:
   (C(16,0) + C(16,1) + C(16,2) + C(16,3)) / 2¹⁶ = (1 + 16 + 120 + 560) / 65 536 = 697 / 65 536 = 0.0106.
3. Both directions (A could also lose): **p = 2 × 0.0106 = 0.021.**

C(16,k) = the number of ways to pick k of the 16 pairs. Read: so clear a win would happen by chance in about 2 of 100
campaigns. It only counts wins, not their size, so it is robust to one run with a huge difference. `_sign_test_p` in
`examples/run_campaign.py`: two-sided exact binomial over the non-tied pairs.

### 6.4 AUROC: does a score tell careful from careless takeovers?

Take every pair of one careful and one careless takeover; count the pairs in which the careful one has the higher score
(a tie counts half). AUROC = that share.

**Example:** careful takeovers score 2, 1 and 0; careless ones 1.5 and −1. The 6 pairs:

| careful | careless | careful higher? |
|---|---|---|
| 2 | 1.5 | yes |
| 2 | −1 | yes |
| 1 | 1.5 | no |
| 1 | −1 | yes |
| 0 | 1.5 | no |
| 0 | −1 | yes |

**AUROC = 4 / 6 = 0.67.** 0.5 = a coin flip, 1 = perfect. The validity bar fixed before the runs was 0.70, so the best
online score (0.66) did not pass.

### 6.5 Standard error (the bars on the asking-rate figure)

SE = standard deviation of the runs / √(number of runs). Example: 18 runs with standard deviation 12 pp: SE = 12 / √18 =
2.8 pp. Two means further apart than about two standard errors are usually also significant.

## 7. Where it is in the code

| what | file |
|---|---|
| actor loop, learner loop, batches 128 + 128, cta_ratio updates, publish every `steps_per_update` | `examples/train_aqi_rlpd.py` (learner around lines 1550–1740) |
| buffers, masks, priorities | `serl_launcher/serl_launcher/data/` (replay buffers), `serl_launcher/serl_launcher/aqi/` (masks, scores, gate) |
| critic, actor and temperature losses | `serl_launcher/serl_launcher/agents/continuous/sac.py` |
| fixed update ratio | `serl_launcher/serl_launcher/utils/update_ratio.py` (`UpdateRatio`), flag `--utd_ratio` |
| ICC, permutation test | `serl_launcher/serl_launcher/aqi/stats.py` (`icc_oneway`, `permutation_sigma2`) |
| sign test | `examples/run_campaign.py` (`_sign_test_p`) |
| AUROC of the scores, validity bar 0.70 | `examples/quality_signal_audit.py`, used by `examples/inconsistent_operator_audit.py` |
