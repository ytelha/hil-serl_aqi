# Inconsistent operator: do the quality scores find the bad corrections?

**The question.** A human operator is not one fixed skill level. One takeover
is careful, the next is rushed and lands next to the hole. If a quality score
can tell those takeovers apart, the learner can keep the good corrections and
drop the bad ones. This suite tests that in two steps:

1. **Do the scores reflect the quality of a correction?** The model logs the
   true condition of every takeover, so each score's ability to find the bad
   ones can be measured directly.
2. **Does using the score help?** It compares dropping the low-scored
   corrections, re-weighting them, and keeping all of them, against plain
   HIL-SERL.

**Why a new operator.** The synthetic operator of every finished campaign is
*consistent*. Its skill vector θ is drawn once and never changes. Its noise
within an episode is zero-mean: motor tremor, a per-approach aim bias, pauses
and the rare lapse. Nothing marks a correction as bad, so quality scores could
only be judged by proxies. This was limitation C10 of the thesis, and the
earlier score audit was confounded by takeover timing
([quality_signal_audit.md](quality_signal_audit.md)).

## 1. The model (patch 0018)

`SyntheticOperator(..., inconsistency=Inconsistency(bad_share, persistence))`
in `serl_launcher/serl_launcher/aqi/operator_model.py`. Off (`None`, the default),
the operator is exactly the previous one: its actions are identical in 15 runs
× 400 steps against the previous code.

| | |
|---|---|
| **hidden condition** | `attentive` or `inattentive`, one per takeover |
| **switch** | at every takeover onset: with probability `persistence` (default 0.7) the condition stays; otherwise it is redrawn, inattentive with probability `bad_share` (default 0.4) |
| **long-run share** | `bad_share` of the takeovers are inattentive. Two takeovers in a row share the condition with probability ρ + (1 − ρ)(b² + (1 − b)²) = 0.86 at the defaults, so bad spells come in runs, like fatigue or distraction |
| **inattentive = sloppy** | the execution fields of θ become the novice anchors: `sigma0`, `k_sdn`, `sigma_tr` (tremor), `lag`, `v_max`, `p_pause`, `pause_len_mean`, `q_rec` (recovery) |
| **inattentive = wrong** | premature commit with probability 0.5 (`p_early`), and the aim is off by 1.5 × the commit threshold in a random direction for that takeover, so it descends next to the hole and jams |
| **unchanged** | the monitoring fields (when the operator steps in, reacts and hands back), so *when* a takeover starts does not depend on the condition. The demos are recorded by the consistent operator. Episodes start from the operator's own θ |
| **label** | `info["synthetic_operator"]["condition"]` in the actor and `condition` on every segment in `aqi_logs/<run>/segments.jsonl` |

The persistence works across episodes: a bad spell can continue after a reset.

Switches: `--synthetic_inconsistent`, `--synthetic_bad_share`,
`--synthetic_bad_persistence` (actor only; YAML `aqi.synthetic_inconsistent`,
`aqi.synthetic_bad_share`, `aqi.synthetic_bad_persistence`). It needs the
geometric phase oracle (the sim); a policy oracle refuses it.

## 2. Using a score: discard or re-weight

Three discard filters join the quality filter pipeline (`quality_filters`).
None of them is in the default pipeline:

| filter | drops a takeover when | role |
|---|---|---|
| `score_filter` | its final score is below the `quality_drop_fraction` quantile (default 0.4) of the last `quality_drop_window` (200) scores. It acts from the `quality_drop_min_segments`-th (20th) score; ties at the threshold are dropped at random, so the share stays right for a score with few values | the score in use |
| `oracle_filter` | it was inattentive (the hidden label) | upper bound: a perfect score |
| `random_filter` | at random with probability `quality_drop_fraction` | control: losing the same amount of data without information |

- The score is the segment score of `quality_score_mode` (hindsight,
  behavioral, progress, outcome), or the mean counterfactual gap for
  `counterfactual`.
- Nothing is dropped during the quality warm-up (`quality_warmup_steps`). The
  score window fills from the start.
- A dropped takeover leaves the **intervention buffer** only: its steps stay in
  the online replay buffer like every other environment step. This is the
  quality filters' existing behaviour.
- Every segment logs `drop_score`, and `drop_warmup` tells whether the warm-up
  still held the filter off. An excluded takeover has `exclusion_reason` set
  to the filter's name.

**Re-weighting** is the existing quality replay priority,
`(per_eps + |δ| + per_eta · q)^α`. It keeps every correction and replays the
well-scored ones more often.

## 3. The experiments

All configs are in `examples/campaigns/inconsistent_operator/`, and they share
one base. In the finished campaigns, this base is called the *fixed ensemble*:
8 critics, 1 sub-sampled, per-member targets, demo pre-training and a learning
rate of 1.5·10⁻⁴. Plain HIL-SERL opts out of the pre-training as usual.
`_orig` keeps the operator and task of the finished campaigns. `_new` uses the
full stack of the synthetic-operator suite: rotation control, the orientation
box and channel, mini-eval convergence and the fixed update ratio
([synthetic_suite.md](synthetic_suite.md)).

| setting | operator | arms | cells | cap |
|---|---|---|---|---|
| `Q1_validity_{orig,new}` | inconsistent | ensemble extended (logs every score) | 1 per tier × 2 seeds = 6 | 30 k |
| `Q2_pilot_{orig,new}` | inconsistent | the six below | 6 × 3 = 18 | 30 k |
| `Q3_consistent_{orig,new}` | consistent | the six without the oracle | 5 × 3 = 15 | 30 k |
| `Q2_confirm_{orig,new}` | inconsistent | the arms the pilot keeps | 6 × 3 per tier × 2 seeds = 108 | 40 k |

| arm | what it does with the corrections | compared with |
|---|---|---|
| `hilserl` | plain HIL-SERL | – (the baseline) |
| `keepall` | quality mode, keeps all, uniform replay (`per_eta 0`) | the matched control of every quality arm |
| `drop-hind` | drops the lowest-scored 40 % | keepall, drop-random |
| `drop-oracle` | drops the inattentive ones | keepall |
| `drop-random` | drops 40 % at random | keepall |
| `reweight-hind` | keeps all, replay priority from the score (the settings of the quality-score campaign) | keepall |

The quality arms differ only in what they drop and how they replay. They all
use the score chosen by Q1 and a 5 000-step warm-up, and no other filter, not
even the default outcome filter. The configs ship with `hindsight` as a
placeholder. The discard fraction 0.4 equals the bad share, so the score, oracle
and random arms drop about the same amount.

### Stage 0 – the operator in each condition (CPU, about 1 h)

```bash
./.venv/bin/python examples/synthetic_suite.py --suite inconsistent_operator check
./.venv/bin/python examples/synthetic_suite.py --suite inconsistent_operator stage0
```

- **Tests:** the inconsistent operator, rotation control, takeover channels,
  quality scores, score audit, campaign switches, the suite driver.
- **Operator test:** cuboid, solo and rescue, per tier, rotation off and on,
  with the operator held attentive or inattentive
  (`stage0/operator_conditions.json`).

**Go on only if** inattentive success is at least 20 percentage points below
attentive, in solo and rescue, for every tier. If it is not, the bad condition
is not bad enough to matter: raise `BAD_AIM_OFF` in `operator_model.py`, re-run
stage 0 and report the change.

### Q1 – score validity (GPU, about 6–9 h per platform)

```bash
./.venv/bin/python examples/synthetic_suite.py --suite inconsistent_operator run Q1_validity_orig
./.venv/bin/python examples/synthetic_suite.py --suite inconsistent_operator audit Q1_validity_orig
```

`audit` runs `examples/inconsistent_operator_audit.py` and writes
`<setting>/report/inconsistent_audit.{md,json}`. For every score it reports the
AUROC for "attentive" (higher score = better correction), with a 95 % CI that
resamples whole episodes, and the AUROC per tier.

**Pre-registered reading:**

| verdict | rule |
|---|---|
| valid | AUROC ≥ 0.70 and the CI's lower bound > 0.5 |
| weak | 0.5 < AUROC < 0.70, or the CI includes 0.5 |
| invalid | AUROC ≤ 0.5 |

- **Label check:** the inattentive share is 0.3–0.5 in every tier. The two
  onset yardsticks (takeover start time, critic value at the onset) are within
  0.5 ± 0.1. Otherwise the audit warns that the label is confounded: report it,
  and do not choose a score from it. The end-time yardstick may differ, since
  inattentive takeovers last longer by design.
- **The score for Q2/Q3:** the audit names the best score that can run online
  (counterfactual, behavioral, hindsight, progress, outcome). Set
  `quality_score_mode` of every quality arm in `Q2_*` and `Q3_*` of the same
  platform to it **before** their first run. The driver refuses to start Q2 or
  Q3 until the configs match that platform's Q1 audit (`--score_as_is`
  overrides).
- **If no score is valid,** run Q2 anyway. The oracle and random arms still
  measure whether discarding bad corrections can help at all. The score arms
  then show what an uninformative score does; the expectation is
  drop-hind ≈ drop-random.

### Q2 – discard or re-weight (pilot about 15–27 h per platform)

```bash
./.venv/bin/python examples/synthetic_suite.py --suite inconsistent_operator run Q2_pilot_orig
./.venv/bin/python examples/synthetic_suite.py --suite inconsistent_operator report Q2_pilot_orig
./.venv/bin/python examples/synthetic_suite.py --suite inconsistent_operator audit Q2_pilot_orig
```

`report` pairs every arm with `keepall` and with `hilserl`: same operator,
same seed, ruler SR difference, wins and losses. `audit` adds a table per
discard arm with what it dropped after the warm-up:

- **precision:** the dropped takeovers that were inattentive. A random drop
  scores the inattentive share (0.4).
- **recall:** the inattentive takeovers that were dropped.
- **attentive lost** and **inattentive kept**.

The oracle has precision and recall 1.

**Primary measure:** ruler SR, paired with `keepall`. Secondary measures: the
first unaided success, the operator share, and the comparison with `hilserl`.

| difference | answers |
|---|---|
| **headroom** = drop-oracle − keepall | do bad corrections hurt? > 0: yes, and dropping them helps. ≈ 0: this learner is robust to them, so no score can help by discarding. < 0: even bad corrections carry useful data |
| **data loss** = drop-random − keepall | the cost of losing 40 % of the corrections without information |
| **score value** = drop-hind − drop-random | what the score's information is worth. As a share of the headroom: (drop-hind − drop-random) / (drop-oracle − drop-random), when the denominator is clearly > 0 |
| **re-weighting** = reweight-hind − keepall | keeping everything but replaying by score |
| **the claim** = best quality arm − hilserl | quality-extended HIL-SERL against the baseline |

**From the pilot to the confirmation.** One cell per tier shows direction and
size, not significance.

- **Full confirmation:** run all six arms if drop-oracle *or* drop-hind beats
  keepall in at least 2 of the 3 tiers.
- **Reduced confirmation:** otherwise run only `hilserl`, `keepall` and
  `drop-oracle` (54 cells), to establish at scale that discarding does not
  help.
- Remove arms from `methods` in `Q2_confirm_*`; never change the arms that
  stay.

### Q3 – the consistent operator (about 12–23 h per platform)

Run the same arms with the consistent operator (no oracle arm: there is no
inattentive takeover to find). The hypothesis predicts that quality helps
*more* when the operator is inconsistent. Per tier, the pilot difference
drop-hind − keepall (and reweight-hind − keepall) should be larger in Q2 than
in Q3. With a consistent operator, a score-based drop should act like
drop-random.

### Q2 confirmation (about 4.5–9 days per platform)

```bash
nohup ./.venv/bin/python examples/synthetic_suite.py --suite inconsistent_operator run Q2_confirm_orig --wandb online > runs/inconsistent_operator/confirm.out 2>&1 &
./.venv/bin/python examples/synthetic_suite.py --suite inconsistent_operator report Q2_confirm_orig
./.venv/bin/python examples/synthetic_suite.py --suite inconsistent_operator audit Q2_confirm_orig
```

That is 9 operators × 2 seeds = 18 pairs per arm. The report gives the exact
sign test per pair of arms. The campaign's own dependency report
(`report/dependency.md`) adds the ICC and the comparisons between arms.
Significance: two-sided p < 0.05 on the paired ruler SR.

## 4. Platforms, order and cost

"Both, decide later":

- Run **Q1 on both platforms** (6 cells each), because it is cheap and shows
  whether the score's validity depends on the platform.
- Run the Q2/Q3 pilots on the platform the synthetic-operator suite's pilots
  favour. Run them on both if time allows.
- Edit the `_new` switches to the stack that suite chose.
- Run the confirmation on one platform.

| step | cells | GPU time (one GPU, about 0.8–1.5 h per 30 k cell) |
|---|---|---|
| stage 0 | – | about 1 h CPU |
| Q1 per platform | 6 | 5–9 h |
| Q2 pilot per platform | 18 | 15–27 h |
| Q3 pilot per platform | 15 | 12–23 h |
| Q2 confirmation (6 arms) | 108 | 4.5–9 days |

**Order:** stage 0 → Q1 → set the score → Q2 pilot → Q3 pilot → decide →
confirmation. `run orig` (or `run new`) runs a platform's Q1, Q2 pilot and Q3
in this order and stops before Q2 until the score is set. The confirmation runs
only by name (`run Q2_confirm_orig`, or the group `q2confirm`).

**One machine runs the whole suite** (finding F3). The `_new` settings need
the orientation-channel limits from the synthetic-operator suite's stage 0. The
driver reads `runs/inconsistent_operator/stage0/limits.json` first, then
`runs/synthetic_suite/stage0/limits.json`. Without either, pass
`--default_limits`.

## 5. Guardrails and limits

- Off, nothing changes. The operator, the monitor and the takeover gate give
  identical results against the previous code. The 37 existing campaign
  configs produce identical commands. The discard filters run only when named
  in `quality_filters`.
- Every setting has its own folder under `runs/inconsistent_operator/`. No
  finished campaign is touched.
- The condition is a model of one kind of inconsistency: a correction that is
  sloppy and aimed wrong, in spells. Real lapses vary more. The label is per
  takeover, not per step: the whole takeover is good or bad.
- `oracle_filter` needs the hidden label, so it exists only in simulation. It
  is the upper bound, not a method.
- A dropped takeover still trains the critic through the online replay buffer.
  The suite tests the intervention buffer's selection, as the quality filters
  do.

## Files

| file | what |
|---|---|
| `serl_launcher/serl_launcher/aqi/operator_model.py` | `Inconsistency`, `inattentive_theta`, the condition switch, `force_condition` |
| `serl_launcher/serl_launcher/aqi/operator_gate.py` | `condition` in the wrapper's info (inconsistent operator only) |
| `serl_launcher/serl_launcher/aqi/runtime.py`, `quality.py`, `config.py` | the label on the segments; `score_filter`, `oracle_filter`, `random_filter` (`DISCARD_FILTERS`), `quality_drop_fraction`, `quality_drop_window`, `quality_drop_min_segments` |
| `examples/train_aqi_rlpd.py`, `examples/run_campaign.py` | the switches; `quality_drop_fraction` per arm |
| `examples/oracle_insertion_test.py` | `--conditions attentive,inattentive` |
| `examples/inconsistent_operator_audit.py` | score validity (Q1) and what the discard arms dropped (Q2, Q3) |
| `examples/campaigns/inconsistent_operator/*.yaml` | the 8 configs |
| `examples/synthetic_suite.py --suite inconsistent_operator` | driver: `check`, `stage0`, `plan`, `run`, `audit`, `report` |
| `tests/test_inconsistent_operator.py` | the model, the label, the filters, the wiring, the audit, the suite |
