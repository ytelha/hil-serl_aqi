# 06 — The six quality scores, in depth

*Added 2026-11-08, for the presentation (slide "Scores that watch the takeover find careless corrections" and
its backups). Every number here is for **experts and intermediates** (the novice is left out of all results
except the teacher-dependency slide), recomputed from the logs of the second training machine.*

*Updated 2026-11-10.* A second, cleaner test now exists: the comprehensive study, with a 20 % careless expert. There
the behavioural score passes the "valid" bar, 0.76 within a run. The ensemble-fix ablation adds a hint about why the
critics' comparison fails. Both are in **§6b**; §7 and §9 are updated.

You should leave this part able to:
- say in one sentence what each of the six scores measures, and write its formula;
- explain how a score is tested (the careless operator, AUROC, "within a run");
- say why three scores work, three don't, and what the numbers do and don't prove;
- answer the questions on the list at the end without notes.

---

## 1. The question

During training the operator takes over many times. Some takeovers are good corrections, others are sloppy. The
idea of **grading** is to give every takeover one number, its quality score q, and to use it in training:
- **replay priority**: good takeovers are replayed more often from the intervention buffer;
- **imitation weight**: if an imitation loss is on, good takeovers are copied more strongly;
- **discard**: the worst-scored takeovers are dropped.

To know whether a score is any good, you need takeovers whose quality you **know**. A consistent synthetic operator
doesn't give you that: all its takeovers are equally good. So we built the **sometimes-careless operator**.

## 2. The test: the sometimes-careless operator

At every takeover the operator is in a hidden state, **careful** or **careless**:
- **careful:** the operator's own skill (expert or intermediate);
- **careless:** novice motor skill (more noise and tremor), it often pushes down too early, and it aims beside the hole;
- **in spells:** a takeover keeps the previous state with probability 0.7; in the long run 40 % of the takeovers are
  careless (`--synthetic_bad_share 0.4`, `--synthetic_bad_persistence 0.7`);
- **monitoring unchanged:** *when* the operator steps in does not depend on the state. So the start time of a
  takeover can't give the state away. That leak fooled the earlier audits (finding F4/F7).

The state is logged with every takeover (`condition` in `segments.jsonl`). It is a **ground-truth label of correction
quality**.

Alone, without any learning, the difference is real: an expert rescues the peg in 100 % of the tries when careful
and in 30 % when careless. The intermediate is the same, 100 vs 30 %.

Code: `examples/train_aqi_rlpd.py` (flags `--synthetic_inconsistent`, `--synthetic_bad_share`,
`--synthetic_bad_persistence`), `serl_launcher/serl_launcher/aqi/operator_model.py`. Background:
[`docs/inconsistent_operator.md`](../inconsistent_operator.md).

**The data on the slide:** second training machine, new stack, cuboid peg. 24 training runs of experts and
intermediates: 6 operators × 2 seeds × 2 ways of using the score. In total 2 448 labelled takeovers, of which
about a third are careless.

## 3. How a score is judged: AUROC, and "within a run"

**AUROC** = the chance that a random careful takeover scores **higher** than a random careless one.
- 0.5 = a coin flip;
- 1.0 = perfect;
- below 0.5 = the score points the wrong way.

*Tiny example.* Careful takeovers score 0.9, 0.4 and 0.1; careless ones score 0.5 and −0.2. There are 3 × 2 = 6
pairs. The careful one wins in 4 of them (0.9 > 0.5, 0.9 > −0.2, 0.4 > −0.2, 0.1 > −0.2), so AUROC = 4/6 = 0.67.

**Within a run.** Pairs are only compared inside the same *stratum*:
- the same run;
- the same training phase (blocks of 15 consecutive takeovers);
- the same onset time in the episode (blocks of 50 steps).

**Why within?** The state comes in spells, and runs differ. If you pool everything, a score could look good just
because it knows *which run* or *which phase of training* a takeover came from. Inside a stratum it can't. The
within-run number is the clean one; it's the number on the slide.

Code: `examples/quality_signal_audit.py` → `stratified_auroc`, and `examples/inconsistent_operator_audit.py` →
`analyse`. The 95 % interval resamples whole episodes, because the takeovers of one episode aren't independent.

**Verdict rule, fixed before the runs:**

| verdict | condition |
|---|---|
| valid | AUROC ≥ 0.70 and the interval's lower end > 0.5 |
| weak | between 0.5 and 0.70, or the interval includes 0.5 |
| invalid | ≤ 0.5 |

**Yardsticks**, which know nothing about quality:
- the start time of the takeover;
- the critics' value at its start.

Both should come out near 0.5. If they don't, the label is tied to the situation (see §7).

## 4. What every score has in common

Each score is **one number per takeover**, computed when the episode ends (`AQIRuntime.end_episode` in
`aqi/runtime.py`). It is put on a common scale so that "about +1 = good, about −1 = bad":

| score | scale used in training |
|---|---|
| behavioural | each term is a z-score over the run's earlier takeovers (running mean and spread) |
| hindsight, progress | divided by the running mean of its magnitude (the sign is kept) |
| outcome | +1 or −1 already |
| critics' comparison | divided by the running mean of \|Â\| |

How it enters training (`aqi/quality.py` → `apply_segment_score`; the buffer `data/data_store.py`):
- **replay priority** in the intervention half of each batch: p = (0.05 + |δ| + max(q, 0))^0.6;
- **imitation weight**, only if an imitation loss is on: w = clip(sigmoid(2q), 0.05, 1);
- **discard**, on the second machine: after a warm-up of 5 000 steps, drop a takeover if its score is in the lowest
  40 % of the last 200 scores. Settings: `quality_drop_fraction 0.4`, `quality_drop_window 200`,
  `quality_warmup_steps 5000`; code: `AQIRuntime._discard`.

The quality mode is chosen with `quality_score_mode` in the campaign config:
- `behavioral`, `hindsight`, `progress`, `outcome`, `counterfactual` (= the critics' comparison);
- plus `oracle` (simulator truth, page 25) and `uniform` (0, a control).

Takeover length is not a mode of its own; it is one of the three terms of the behavioural score.

---

## 5. The six scores

Notation:
- a takeover starts at step t₀ (the onset) and lasts T steps, so it hands back at t₀ + T (10 steps = 1 second);
- Q̄(s, a) = the mean over the critics of their value for action a in state s;
- the robot's own move a^robot is the action the policy proposed in the same state while the operator was in control.

### 5.1 Behavioural score — B = z(ΔV) − z(T) − z(F_peak)

**In words:** a short, gentle takeover after which the critics think the situation is better scores high.

**The three terms:**
- **ΔV, value gained:** ΔV = Q̄(s_last, a^op_last) − Q̄(s_onset, a^robot_onset). The first term is the critics'
  value of the operator's move at the last step of the takeover; the second is their value of the robot's own move at
  its first step. In short: how much better the critics rate the situation at the end of the takeover than at its
  start.
- **T, length:** the number of steps the operator was in control.
- **F_peak, peak force:** the largest wrist force during the takeover.
- **z(·):** each term is standardised over the run's takeovers, so the three can be added. During training this is a
  running z-score over the earlier takeovers; it gives 0 until enough takeovers have been seen.

**Worked example** (illustrative numbers). Say this run averages ΔV = 0.05 (spread 0.10), T = 33 steps (spread 12)
and F_peak = 12 (spread 5).
- **Careful takeover:** ΔV 0.20, T 20, F 9.
  - z(ΔV) = (0.20 − 0.05)/0.10 = +1.5
  - z(T) = (20 − 33)/12 = −1.1
  - z(F) = (9 − 12)/5 = −0.6
  - B = 1.5 + 1.1 + 0.6 = **+3.2**
- **Careless takeover:** ΔV −0.05, T 48, F 20.
  - z(ΔV) = −1.0
  - z(T) = +1.25
  - z(F) = +1.6
  - B = −1.0 − 1.25 − 1.6 = **−3.85**

**Result:**
- within a run **0.70** (expert 0.71, intermediate 0.67); over all takeovers 0.69 (0.67–0.71);
- **computed live during training: 0.67.** The audit standardises with the whole run, which a live system can't, so
  0.67 is what the filter really saw.

**Where its signal comes from:**

| part | AUROC within a run |
|---|---|
| length alone | 0.69 |
| value gained alone | 0.63 |
| peak force alone | 0.59 |
| B without the length term | 0.63 |

So most of it is length; the value term adds a little, and force less.

**Verdict:** "weak" by the rule (0.698 within a run, just under 0.70). It is the best score that can run online.

**Note:** B uses the critics only to read the *value of the situation*. It never compares the operator's move with
the robot's, which is why it avoids the failure of §5.6.

**Code:** `QualityScorer.behavioral_q` (`aqi/quality.py`). ΔV is computed in `AQIRuntime.end_episode` as
`mu_h[-1] - mu_r[0]`. The audit's version is `_add_composites` in `quality_signal_audit.py`.

### 5.1b Where the name "behavioural score" comes from, and what the literature calls it

**Origin: this project.** No paper defines a "behavioural score" for takeovers.
- The research proposal listed design options for the quality score. The thesis draft keeps them as the table
  *Design options for the intervention quality score* (methodology chapter).
- Option **A is "hand-weighted features":** progress, ΔV, success, force, jerk, duration. Option B, the counterfactual
  advantage, was chosen in the proposal.
- When B failed (AUROC 0.18, §5.6), option A was built with three of those features: ΔV, duration and peak force.
  The code named it `behavioral` (`aqi/config.py`, "a critic-FREE composite of the takeover's behaviour").
- The name says what the score reads: how the takeover went, not what the critics think of the operator's move.
- "Critic-free" is not quite true. ΔV reads the critics' value of the situation; the score never asks them to compare
  the operator's move with the robot's.

**Where the idea has roots outside RL:**

| field | what they call it | what it measures | closest to |
|---|---|---|---|
| human factors (workload, vigilance) | **performance-based (behavioural) measures**, one of three classic families next to subjective and physiological measures | the operator's own task performance | the whole idea: judge the operator by what they do |
| driver monitoring | **steering entropy** (Nakayama, Futami, Nakamura & Boer, SAE 1999) | how unpredictable the steering corrections are; rises with distraction | detecting a careless operator from control behaviour |
| driver drowsiness | *vehicle-based* measures (steering, lane keeping) vs *behavioural* measures (eye closure, yawning, head pose) (Sahayadhas et al., Sensors 2012) | – | careful: here "behavioural" means the face, and our score would be "vehicle-based" |
| surgical skill assessment | **kinematic / motion-based metrics** | completion time, path length, jerk, sometimes force | T and F_peak are exactly these |
| learning from demonstration | **demonstration quality** (Sena & Howard, IJRR 2020; Belkhale, Cui & Sadigh, NeurIPS 2023) | how useful a demonstration is to the learner | the goal of grading |

**What the RL and imitation-learning literature calls the same job** (a weight per demonstration or correction, so
that bad ones count less):

| name | papers | how the weight is found |
|---|---|---|
| **confidence score** | Wu et al., ICML 2019 (2IWIL, IC-GAIL); Zhang et al., NeurIPS 2021 (CAIL) | given for a few demonstrations (Wu), or learned in a bilevel loop (CAIL) |
| **demonstrator expertise** | Beliaev et al., ICML 2022 (ILEED) | learned per demonstrator and state, with the policy |
| **ranking / preference** | Brown et al., ICML 2019 (T-REX); PACT (arXiv 2606.03949) | ranked trajectories; preference pairs at the takeover |
| **advantage weighting** | AWR (Peng 2019), CRR (Wang 2020) | the critic's advantage; the lineage of our hindsight score and the critics' comparison |
| **intervention weighting** | IWR (Mandlekar 2020), Sirius (Liu et al., RSS 2023) | by data source: takeovers up, the robot steps just before them down; not by quality |
| **intervention as reward** | RLIF (Luo et al., ICLR 2024) | the takeover itself is a negative reward |
| **uncertainty of the corrections** | SiLRI (Zhao et al. 2025), GAINS (arXiv 2608.15707) | per-state spread of the operator's actions; distributional values |
| **data curation** | Demo-SCORE (arXiv 2503.03707), CUPID (arXiv 2506.19121) | a success classifier; influence on the policy's return |
| **outcome filter** | Hi-ORS (Lu et al. 2025) | episode success (our `outcome` score and `outcome_filter`) |

**The closest established term is "confidence score"** (Wu 2019, Zhang 2021). Ours is a *hand-crafted* one: three
fixed features, standardised, no learning.

**A trap with the word.** In RL, "behaviour" usually means the *behaviour policy* that collected the data
(behaviour-regularised offline RL) or *behavioural cloning*. A reviewer may read "behavioural score" as "a score of
the behaviour policy". **Candidate names, each with the literature it borrows from:**

| name | borrowed from | for | against |
|---|---|---|---|
| **heuristic quality score** | Sirius calls its weights an "approximate quality score"; RINSE and CUPID contrast learned data quality with "heuristic" filters (success, length) | keeps the deck's word "quality score" for the whole family; "heuristic" says hand-crafted, not learned | sounds modest, which is honest |
| **heuristic confidence score** | "confidence score" for a per-demonstration trust weight: Wu et al. 2019, CAIL (Zhang et al. 2021); "confidence weighting" in NetLfD (RA-L 2023) | the closest established IL term | "confidence" may be read as the critics' certainty (σ_Q) |
| **takeover data-quality score** | data quality in imitation learning (Belkhale et al. 2023); an episode-level data-quality score from teleoperation telemetry, normalised subscores with weights (arXiv 2605.26349) | the structure matches ours: normalised features, weighted sum | long |
| **performance-based score** | human factors: performance-based measures of the operator, next to subjective and physiological ones | explains "behavioural" to a human-factors reader | rare in RL |
| **takeover-feature score** | the proposal's option A, "hand-weighted features" | plain description | no literature behind it |

Length as a quality signal is not ours alone: SiLRI (arXiv 2512.24288) treats shorter successful episodes as
higher-quality intervention data, and RINSE scores demonstrations by smoothness.

**Recommendation: "heuristic quality score".**
- It keeps "quality score", the word the deck and the thesis use for the whole family.
- It removes the clash with "behaviour policy".
- On first mention, define it: "a heuristic quality score from three observable features of the takeover (value
  gained, length, peak force), in the sense of a hand-crafted confidence score (Wu et al. 2019; Zhang et al. 2021)".
- The code name `behavioral` stays.

If you keep "behavioural score", add the same defining sentence.

Sources: [Nakayama et al. 1999](https://saemobilus.sae.org/papers/development-a-steering-entropy-method-evaluating-driver-workload-1999-01-0892),
[Sahayadhas et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3571819/),
[workload measure families (DTIC)](https://apps.dtic.mil/sti/pdfs/ADA289568.pdf),
[surgical motion smoothness metrics](https://www.mdpi.com/1424-8220/23/6/3146/html),
[Sena & Howard 2020](https://arxiv.org/pdf/1905.04218v1.pdf), [Belkhale et al. 2023](https://arxiv.org/abs/2306.02437v1),
[Wu et al. 2019](https://proceedings.mlr.press/v97/wu19a.html), [Zhang et al. 2021](https://arxiv.org/abs/2110.14754),
[Beliaev et al. 2022](https://arxiv.org/pdf/2202.01288), [Brown et al. 2019](https://arxiv.org/abs/1904.06387),
[Mandlekar et al. 2020](https://arxiv.org/pdf/2012.06733), [Liu et al. 2023](https://arxiv.org/pdf/2211.08416),
[Luo et al. 2024](https://arxiv.org/pdf/2311.12996), [SiLRI](https://arxiv.org/pdf/2512.24288),
[PACT](https://arxiv.org/pdf/2606.03949), [GAINS](https://arxiv.org/pdf/2608.15707),
[Demo-SCORE](https://pdf.arxiv.org/pdf/2503.03707), [CUPID](https://arxiv.org/html/2506.19121v1),
[RINSE, smoothness-driven data quality](https://arxiv.org/html/2604.23000),
[episode-level data-quality assessment in teleoperation](https://arxiv.org/pdf/2605.26349),
[NetLfD, confidence weighting](https://fis.tu-dresden.de/portal/en/publications/netlfd(c4a7553e-ac39-4004-a620-1cb603256c09).html).

### 5.2 Takeover length — −T

**In words:** shorter is better.

**Example:** careful takeovers of experts and intermediates last 29 steps on average (2.9 s); careless ones last 38
steps (3.8 s).

**Result:** within a run **0.69** (expert 0.69, intermediate 0.68); overall 0.66.

**The honest caveat:** in our model, careless takeovers last longer **partly by design**. Aiming beside the hole and
pushing too early take longer to undo. The audit script itself says so ("inattentive takeovers last longer by
design"). Whether a real careless person also takes longer is plausible, but it is a hypothesis to test with people.

### 5.3 Hindsight advantage — A_hind

**In words:** what the robot actually reached shortly after it got control back, compared with what the critics
expected at the start of the takeover.

**Formula:**
A_hind = Σ_{k=0}^{n−1} 0.97^k r_{t₀+k} + 0.97^n V(s_{t₀+n}) − V(s_{t₀})

- **Window:** from the onset to 10 steps after the hand-back, so n = T + 10. It is cut at the episode end; if the
  episode ended by success there is nothing to bootstrap.
- **r:** the task reward, 1 at the insertion.
- **V(s):** the critics' mean value **at the robot's own action**, Q̄(s, a^robot). The critics never judge the
  operator's move.
- **0.97:** the training discount.

**Worked examples:**
- **Careful.** The critics expected 0.40 at the takeover, and the peg is inserted 15 steps later:
  0.97¹⁵ × 1 − 0.40 = 0.633 − 0.40 = **+0.23**.
- **Careless.** The peg is stuck, the value is 0.30 at the end of an 18-step window:
  0.97¹⁸ × 0.30 − 0.40 = 0.173 − 0.40 = **−0.23**.

**Result:** within a run **0.65** (expert 0.66, intermediate 0.63); overall 0.60.

**Caveat (finding F4):** early in training the critic has seen no reward, so r = 0 and V is almost flat. Then
A_hind ≈ (0.97ⁿ − 1)·V(s_t₀): negative, and set by the window length and the starting value rather than by what the
takeover achieved. So early on, hindsight is partly a *length* signal too.

**Code:** `hindsight_advantage` and `post_release_window` (`aqi/quality.py`); `AQIRuntime._post_release`
(`aqi/runtime.py`). Settings: `quality_post_release_steps 10`, `quality_hindsight_discount 0.97`.

### 5.4 Episode outcome — ±1

**In words:** +1 if the episode ended in success, −1 if not. **Every takeover of an episode gets the same score.**

**Example:** episodes with a careless takeover succeeded in 48 % of cases, episodes with a careful one in 74 %.

**Result:** overall **0.63**, but within a run only **0.56** (expert 0.56, intermediate 0.58).

**Why it drops within a run:** pooled, the outcome mostly tracks *when* and *where* the takeover happened. Late in
training and in good runs more episodes succeed, and careless spells aren't spread evenly. Inside one run and phase
that information is gone. And the outcome also depends on everything the robot does after the hand-back, not only on
the correction.

**Why it matters anyway:** the same idea is the **outcome filter** ("drop every takeover of a failed episode",
Hi-ORS, Lu et al. 2025). On the first machine the oracle runs had it on. There, 47 % of an expert's takeovers lay in
failed episodes and were dropped; an expert run still kept about 110, a novice run about 7 (counted in those
runs' takeover logs on 14 Nov; the earlier "67 %, about 200" came from the asking-rate campaign).

### 5.5 Progress to the hole — Δd

**In words:** how much deeper the peg got over the same window as hindsight.

**Formula:** Δd = d(s_{hand-back+10}) − d(s_onset), where d = the depth of the tool below its start pose. In the
code this is the negative height of the tool in the start frame.

**Example:** going from 0 to 4 mm deeper scores +4; staying stuck at the rim scores about +0.5.

**Result:** within a run **0.51** (expert 0.51, intermediate 0.50): chance.

**Why (likely, not proven):** within 10 steps after the hand-back the depth hardly changes in either state.
- A careful operator often hands back *above* the hole, aligned but not yet inserted.
- A careless one pushes beside it and gets stuck at the surface.
- Depth doesn't see alignment, which is exactly what the careful operator fixes.

**Code:** `AQIRuntime._post_release` (`d_progress_post`); the depth function `_progress_fn` in `AQIRuntime.__init__`.

### 5.6 Critics' comparison — Â (the counterfactual advantage)

**In words:** in each state of the takeover, the critics compare the operator's move with the move the robot would
have made.

**Formula:** Â = (1/T) Σ_t [Q̄(s_t, a^op_t) − Q̄(s_t, a^robot_t)]

**Example:** the correction scores 0.55, the robot's own move 0.70, so −0.15 for that step. The score is the average
over the takeover.

**Result:** within a run **0.48** (expert 0.46, intermediate 0.50): chance. And against episode success, with the
fixed ensemble's critics on 5 790 corrections of experts and intermediates, it is **0.15**, pointing the wrong way.
The critics rated the robot's move higher in 80 % of the corrections.

**Why it fails** (page 24 of the deck):
1. **It never saw the operator's move.** The critics learn from the robot's own actions. A human correction lies
   outside that cloud, so its grade is an extrapolation, a guess.
2. **The robot's own move is favoured.** The actor is trained to maximise these very critics, so its action sits
   where the critics are most optimistic. The correction loses by default. This is probably the main reason.
3. **Recoveries are rare in its data.** Backing off and realigning is right, but the critics have rarely seen such a
   recovery end in success, so they rate it low.

**Better-grounded versions don't help.** A pre-registered study on the cluster tried three of them:
- Q(s, a_H) − V(s);
- the rank of the human action among the policy's own samples;
- a per-critic t-test.

All ranked below chance against the simulator's truth
([`counterfactual_score_study.md`](../counterfactual_score_study.md)).

**Code:** `counterfactual_advantage_batch` (`aqi/uncertainty.py`); `QualityScorer.score_segment` (`aqi/quality.py`).
There is an optional disagreement gate, q = Â · 1/(1 + (σ/σ_ref)²).

---

## 6. All six at a glance

AUROC for careful > careless, experts and intermediates, 24 runs, 2 448 takeovers:

| score | measures | within a run | expert / interm. | overall (95 % CI) | verdict |
|---|---|---|---|---|---|
| behavioural | value gained − length − peak force | **0.70** | 0.71 / 0.67 | 0.69 (0.67–0.71) | weak (just under) |
| takeover length | shorter is better | **0.69** | 0.69 / 0.68 | 0.66 (0.63–0.68) | weak |
| hindsight advantage | what the robot reached after the hand-back vs what was expected | **0.65** | 0.66 / 0.63 | 0.60 (0.57–0.63) | weak |
| episode outcome | did the episode succeed | 0.56 | 0.56 / 0.58 | 0.63 (0.60–0.67) | weak |
| progress to the hole | depth gained over the same window | 0.51 | 0.51 / 0.50 | 0.51 (0.48–0.53) | ≈ chance |
| critics' comparison | Q̄(operator) − Q̄(robot) | 0.48 | 0.46 / 0.50 | 0.47 (0.44–0.50) | ≈ chance |

Not on the slide: **smoothness** (mean jerk of the operator's actions) reaches 0.85. But it is **circular**: the
model defines carelessness partly by motor noise, and smoothness measures motor noise. So it says nothing about real
people.

The pattern to remember: the scores that **watch the takeover itself** carry the signal (its length, the value it
gained, what the robot reached afterwards). The **critics' comparison** of two moves does not.

## 6b. The second test: the comprehensive study (20 % careless expert, November)

Second machine, new stack. 4 583 labelled takeovers from 45 runs of the five ensemble-based grading arms. Careless
share 19 %. Careless takeovers last 33.5 against 23.5 steps; their episodes succeed in 60 against 90 %.

| score | within a run | overall (95 % CI) | verdict | first test (40 % careless) |
|---|---|---|---|---|
| behavioural | **0.76** | 0.78 (0.76–0.79) | **valid** | 0.70 |
| hindsight advantage | **0.74** | 0.77 (0.74–0.78) | valid | 0.65 |
| value gained alone | 0.73 | 0.70 | valid | 0.63 |
| behavioural without length | 0.72 | 0.68 | weak | 0.63 |
| behavioural as computed live | 0.70 | 0.73 | valid | 0.67 |
| takeover length | 0.69 | 0.76 | valid | 0.69 |
| episode outcome | 0.55 | 0.65 | weak | 0.56 |
| peak force | 0.51 | 0.54 | weak | 0.59 |
| progress to the hole | 0.47 | 0.49 | invalid | 0.51 |
| critics' comparison | 0.47 | 0.42 | invalid | 0.48 |

What changed against the first test:
- **The scores separate better.** The behavioural score passes the bar.
- **The value gained and hindsight now carry signal of their own:** without its length term the behavioural score
  keeps 0.72. So the answer to "isn't it just length?" is now "no, not only".
- **The label is cleaner.** Within a run, the start time sits at 0.52 and the critics' value at the onset at 0.47, both
  near 0.5. The audit's warning comes only from the pooled start time (0.69), which the within-run comparison
  removes.
- **The filter worked as a filter:** 41 % of what it dropped was careless, against a base rate of 19 % (a lift of
  2.2).
- **But training still showed no gain.** Every grading arm reached the same 95–99 % as keeping everything. The new
  stack makes the task easy, so there was nothing left to improve.

**The hint from the ensemble-fix ablation** (first machine, consistent operator, label = episode success):
- The critics' comparison sits at chance (0.49–0.56) in the two arms with the pessimistic target and no warm start.
- It points the wrong way (0.31–0.43) in the arms with the optimistic one-critic target or the warm start.
- This holds under every timing control.
- It fits the explanation of §5.6, reason 2: the more optimistic the critics are about the robot's own move, the more
  the correction loses.
- 12 runs per arm, so treat it as a hint.

Details: [`../thesis/comprehensive_study_results.md`](../thesis/comprehensive_study_results.md) §4,
[`../thesis/ensemble_fix_ablation_results.md`](../thesis/ensemble_fix_ablation_results.md) §2.

## 7. What the numbers do *not* prove (say this before someone else does)

- **Much of the signal is length,** and in our model careless takeovers are longer partly by design. Without the
  length term the behavioural score drops to 0.63 in the first test, but keeps 0.72 in the second.
- **Two series of runs, one machine,** experts and intermediates only (40 % and 20 % careless). Never with real
  people.
- **The label check raised a flag** for experts and intermediates in the first test. In the second, the within-run
  yardsticks are clean (§6b).
  - Pooled over all takeovers, the start time of a takeover reaches 0.62 (careful ones start earlier on average).
    The automatic rule flags anything more than 0.1 from 0.5. Inside the strata it is 0.48, so the within-run
    numbers are protected against timing.
  - But the critics' value at the onset reaches 0.61 within a run: careful takeovers start in states the critics
    rate higher. A likely reason is the spells: a careless takeover leaves the peg worse off, and the next one, often
    careless too, starts from there.
  - Scores that subtract the onset value (ΔV, A_hind) are, if anything, pushed *against* the label by this. Length
    doesn't use it. With novices included the flag did not fire (0.59 and 0.58).
- **A good AUROC does not yet mean better training.** In the second test, too, every grading arm equalled keeping
  everything (§6b). The first test in detail:
  - Discarding by the behavioural score versus keeping everything with a replay weight gave the same success: 93 vs
    94 %, p = 0.45.
  - Discarding converged faster: 7 600 vs 8 500 steps, faster in 10 of 12 pairs, p = 0.039.
  - Of what the filter dropped, 58 % was careless, against a base rate of 36 %.
  - It dropped only 13 % of all takeovers, because 67 % fell inside its 5 000-step warm-up.
  - A perfect filter that knows the hidden state reached 100 %, and so did keeping everything. Only 2 pairs, at the
    ceiling.
- **Arm names.** The second machine's training arms are called `drop-hind` and `reweight-hind`, but they ran
  `quality_score_mode: behavioral`. The config was switched after the pilot, and the arm names stayed (check:
  `campaign.json` in the archive). In this repo the config already says `behavioral`.

## 8. Related, but not one of the six: the quality oracle (page 25)

This score uses the **simulator's truth**:

q_raw = (z_onset − z_hand-back) − 1[jam] + ½·outcome, then divided by its running mean magnitude.

- z is the trouble signal of the operator's monitor, from 0 to 1.
- It is a ceiling, not deployable on a real robot.
- It ran on the first machine with a **consistent** operator, together with the outcome filter and an imitation
  loss. With experts and intermediates success rose 85.8 → 94.1 % (operator decides) and 95.7 → 100 % (robot asks).
- The three changes were made at once, so the gain is not yet the score's alone.
- **The novice test (4–5 Nov) separated them for novices:** the imitation loss collapses the novice, not the score.
  Used only as replay priority, the oracle gave 68 % against 65 % for ensemble extended
  ([`thesis/first_machine_novice_test_and_ladder.md`](../thesis/first_machine_novice_test_and_ladder.md)).

Code: `QualityScorer.oracle_q`.

## 9. Questions you should be able to answer

| question | short answer |
|---|---|
| Which score is best? | The behavioural score: 0.70 within a run in the first test, 0.76 (valid) in the second. Hindsight is next (0.65 / 0.74). |
| Isn't length trivial? | Partly: in our model careless takeovers are longer by design. But without length the composite still reaches 0.63 in the first test and 0.72 in the second. For real people it's a hypothesis to test. |
| Why does the critics' comparison fail? | It grades a move the critics never saw, against a move the actor was trained to make the critics like. And recoveries are rare in the critics' data. Better-grounded versions failed too. |
| Why is the outcome good overall but weak within a run? | Pooled, it mostly encodes the run and the training phase. It also gives every takeover of an episode the same grade. |
| Why is progress at chance? | Likely because depth barely changes within 10 steps after the hand-back in either state, and depth doesn't see alignment. |
| What does AUROC 0.70 mean? | In 70 % of the (careful, careless) pairs from the same run, phase and onset time, the careful one scores higher. |
| Why compare within a run? | The state comes in spells and runs differ. Pooled numbers would reward knowing the run or the phase. |
| Is the label clean? | Timing is controlled within a run (start time 0.48). The onset value is 0.61, so careful takeovers start in slightly better states, likely from the spells. That's stated as a limit. |
| Did grading help training? | Not in success, in either test. First test: 93 vs 94 %, and discarding made learning faster (p = 0.039). Second test: every grading arm 95–99 %, the same as keeping everything. On the new stack the task is too easy for a gain to show. That needs a harder task and a measure of learning speed. |
| Why is the critics' comparison sometimes inverted and sometimes at chance? | In the ablation it is inverted when the critics are optimistic about the robot's own move (one-critic target or warm start), at chance otherwise. That supports reason 2 of §5.6. |
| Why not smoothness (0.85)? | Circular: the model's carelessness *is* motor noise. |
| Why without novices? | The model's careless novice is almost its normal novice (0.52), so it can't test a score. |
| What comes next? | Detection is shown. Next is the training benefit: careless corrections that actually hurt (a 40 % careless share or a careless intermediate), on a harder task, measured by learning speed and human effort; then the comparison with HiRIL, once its requests can be answered. |

## 10. Exercises

1. Open `aqi/quality.py` and find, for each of the six scores, the line that computes it. For the critics'
   comparison, follow `counterfactual_advantage_batch` in `aqi/uncertainty.py`.
2. Compute the AUROC of the tiny example in §3 by hand, then with `quality_signal_audit.auroc` in a Python shell.
3. Re-run the audit on the second machine's logs. This works on the Windows PC; it needs only NumPy:

   ```bash
   python examples/inconsistent_operator_audit.py --log_root "C:/Users/youss/Downloads/AQI_HIL_SERL_complete_2026-10-29/repo/runs/inconsistent_operator/two_arm_confirm_new/aqi_logs" --out runs/audit_two_arm
   ```

   This run includes the novice cells; the per-tier columns show expert and intermediate separately. Compare with §6.
4. Explain to someone, in 60 seconds and without notes, why the behavioural score works and the critics' comparison
   doesn't.
5. Write down one change to the operator model that would make length *not* informative. Then say what you would
   expect from the behavioural score in that case.

**Further reading:**
- [`docs/inconsistent_operator.md`](../inconsistent_operator.md) (the model and every score);
- [`docs/thesis/inconsistent_operator_results_and_qa_log.md`](../thesis/inconsistent_operator_results_and_qa_log.md)
  (all tiers, all series);
- [`docs/quality_signal_audit.md`](../quality_signal_audit.md);
- [`docs/counterfactual_score_study.md`](../counterfactual_score_study.md);
- references: Peng et al. (AWR) 2019 and Wang et al. (CRR) 2020 for advantages; Spencer et al. 2020 and Liu et al.
  (Sirius) 2023 for grading human interventions; Lu et al. (Hi-ORS) 2025 for the outcome filter.
