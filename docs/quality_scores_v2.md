# Quality scores v2 — four intervention scores from the literature

Status: implemented (patch 0005), **not yet run**. Evaluate first
(`handoff/EVALUATE_FIRST.md`, section "Quality scores v2").

## 1. Why

The quality module grades each human takeover and uses the grade for replay
priority (`q_int`) and, optionally, for the weight of the imitation loss
(`w_i`). The two scores so far:

- **counterfactual** Â = μQ(s, a_H) − μQ(s, a_robot) is **anti-predictive**
  (AUROC 0.175, RQ4 §11). The critic is evaluated at the human action, which
  lies off the policy's distribution.
- **behavioral** z(d_value) − z(t_int) − z(f_peak) predicts well (0.78 live),
  but `t_int` may be timing leakage (`docs/quality_signal_audit.md`).

The literature review (thesis ch. 3) suggested four more signals. Each one
avoids evaluating Q at a_H, or measures something the others do not:

| # | score / weight | lineage | formula | acts on | config |
|---|---|---|---|---|---|
| 1 | **hindsight advantage** | AWR (Peng et al. 2019), CRR (Wang et al. 2020), GAE (Schulman et al. 2016) | Σ γ^k r + γ^n V(s_{t0+n}) − V(s_{t0}), window onset → handback + K | priority + BC weight | `quality_score_mode: hindsight` |
| 2 | **task progress** | Option A feature `d_progress`, never used online | Δ insertion depth, onset → handback + K | priority + BC weight | `quality_score_mode: progress` |
| 3 | **outcome** (control) | Hi-ORS (Lu et al. 2025), proposal Option G | +1 / −1 by episode success; with `outcome_filter`: rejection sampling | priority + BC weight | `quality_score_mode: outcome` |
| 4 | **operator consistency** | SiLRI (Zhao et al. 2025) | c = 1 / (1 + (σβ/σ_ref)²) | BC weight **only** | `quality_consistency_weight: true` |

Plus `quality_score_mode: uniform` (q = 0 for every segment): the "no
grading" control, e.g. plain BC in C8.

## 2. The scores

### 2.1 Hindsight advantage (`hindsight`)

For a takeover starting at step t0 with length L (handback at t0 + L):

```
t_e    = t0 + L + K                        (K = quality_post_release_steps)
A_hind = sum_{t=t0}^{t_e-1} g^(t-t0) r_t + g^(t_e-t0) V(s_te) - V(s_t0)
V(s_t) = mu_Q(s_t, a_robot_t)              (already computed every step)
```

- **Only states that were actually visited are evaluated**, at the policy's
  own action. The critic is never queried at a_H, which is the failure mode
  of the counterfactual score.
- It asks: *after the operator handed back, did the robot stand in a better
  state than before the takeover, counting any reward on the way?*
- `g` = `quality_hindsight_discount`. The training script sets it to the
  experiment's TD discount (`config.discount`, 0.97) unless the flag is given.
- **Episode end inside the window:** the mask of the last transition
  decides, exactly as in the critic's TD target.
  - mask 0 (done): rewards to the end, no bootstrap. **In this codebase
    every episode ends this way**, success or time limit: the sim and
    `franka_env` both set `done` at `max_episode_length`. A takeover that
    runs into a failed episode's end therefore gets A_hind = −V(s_t0) plus
    discounted zeros.
  - mask 1 (an env that reports a time limit as `truncated`): the window
    ends at the last step, bootstrapping from its V. This does not occur in
    the current tasks.
- V(s) = μQ(s, a_robot) is a one-sample estimate of the soft value. It ignores
  the entropy term. During `random_steps`, a_robot is a random action.
- **Scaling:** q = A_hind / running mean |A_hind|. The sign is kept, not
  centred, so a harmful takeover stays negative. A missing value (NaN)
  scores 0.

**Known bias:** when the window reaches a successful episode end, A_hind
contains the success reward, so it partly re-labels the outcome. The audit's
**window-clean** column measures how much of its AUROC is left without that.

### 2.2 Task progress (`progress`)

d_progress_post = progress(s_{t0+L+K}) − progress(s_{t0}).

- `progress_fn` defaults to −z of `tcp_pose`, i.e. insertion depth below the
  reset pose in the relative frame.
- The window ends at the final next-state when the episode ends inside it.
- Scaled like the hindsight score (running mean magnitude, sign kept).
- **Lateral error is not included**: the hole position is not part of the
  proprio state. Pass a `progress_fn` to `AQIRuntime` for a richer measure.
- It is critic-free, and the same feature within the takeover (`d_progress`)
  was already logged. The audit now reports both.

### 2.3 Outcome (`outcome`) — the control

- q = +1 if the episode succeeded, −1 otherwise.
- With the `outcome_filter` section (arm `quality-outc`), segments from failed
  episodes are dropped from the intervention buffer. All survivors get the
  same q: Hi-ORS-style rejection sampling.
- It answers: *does any graded score beat simply keeping the interventions
  of successful episodes?*

### 2.4 Uniform (`uniform`) — no grading

q = 0 for every segment:
- the priority is the baseline priority;
- w = sigmoid(0) = 0.5 for every kept segment.

The BC loss is normalised by Σw, so this is **plain BC**.

### 2.5 Operator consistency (`quality_consistency_weight`)

SiLRI models the human as a noisy policy with a state-dependent spread σβ and
trusts it less where the spread is large. Here σβ(s) is estimated
non-parametrically (`serl_launcher/aqi/consistency.py`):

1. **References:** the operator's own earlier takeover (state, a_H) pairs
   from the same run. One run is one operator in every campaign. The
   references are added only after the episode is scored, so a segment is
   never compared with itself.
2. **Neighbours:** for each of the k (`quality_consistency_k`, default 5)
   earlier segments closest to s, take their nearest state. That is **at
   most one neighbour per segment**. Consecutive states of one trajectory
   are each other's nearest neighbours and their actions are smooth, so an
   unrestricted kNN would call every operator consistent.
3. **Spread:** σβ(s) = sqrt(mean over action dims of the variance of those
   k actions). States are compared after running standardisation, as in the
   OOD scorer.
4. **Weight:** c(s) = 1 / (1 + (σβ/σ_ref)²). σ_ref is
   `quality_consistency_sigma_ref`, or the running mean of σβ by default.
   It multiplies the BC weight `w_i` **only**. The replay priority is
   unchanged, so the weight is inert unless `bc_loss_weight > 0`.
5. **Not ready:** c = 1 until `quality_consistency_min_ref` (200) reference
   steps from ≥ k segments exist. The log field `consistency_ready` shows it.

Since the BC loss is Σ w·NLL / Σ w, only **relative** weights within a batch
matter. A uniformly noisy operator is therefore not down-weighted as a whole.
What changes is the weighting between states where the operator is
consistent and states where they are not.

## 3. How q and w are formed

Every mode except `counterfactual` gives **one signed score q_seg per
segment**. `QualityScorer.apply_segment_score` then sets:

```
q_int = max(q_seg, 0)                                   replay priority
w     = clip(sigmoid(quality_gamma * q_seg), w_min, w_max)   BC weight
w    *= c(s_i)   (only with quality_consistency_weight)
```

- The behavioral mode uses exactly this mapping, as before; a test guards
  the refactor.
- The disagreement gate and `adv_exclude` act on the counterfactual Â only.
- The filters (`safety_veto`, `jerk_veto`, `outcome_filter`, `unsolicited`)
  work as before in every mode. Excluded segments get w = 0.

## 4. What is logged (all logging modes)

| where | new fields |
|---|---|
| `segments.jsonl` → `features` | `a_hind`, `hind_window`, `d_progress_post`, `progress_window`, `v_onset` (μQ at onset = state difficulty); non-finite values are `null` |
| `segments.jsonl` (consistency on) | `sigma_beta_mean`, `consistency_mean`, `consistency_ready` |
| `steps.jsonl` | `reward`, `mask` per step, so the audit can rebuild A_hind offline |
| `meta.json` → `config` | `proprio_slices` (layout of the logged `state`), plus the new config fields |

The window features are computed in `logged` runs too, so the audit can
evaluate hindsight and progress **without training on them**.

## 5. Knobs

| AQIConfig field | flag (train / campaign) | default | relevant in |
|---|---|---|---|
| `quality_score_mode` | `--quality_score_mode` | `counterfactual` | quality, aqi |
| `quality_post_release_steps` | `--quality_post_release_steps` | 10 | every logging mode (features); quality/aqi (score) |
| `quality_hindsight_discount` | `--quality_hindsight_discount` | experiment TD discount | same |
| `quality_consistency_weight` | `--quality_consistency_weight` / `--noquality_consistency_weight` | off | quality, aqi, with `bc_loss_weight > 0` |
| `quality_consistency_k` | `--quality_consistency_k` | 5 | same |
| `quality_consistency_sigma_ref` | `--quality_consistency_sigma_ref` | running mean | same |
| `quality_consistency_min_ref`, `quality_consistency_capacity` | config only | 200, 20000 | same |

All of them are per-arm knobs in campaign YAMLs (`ARM_KEYS`). The condition
table of a run README marks them active or inert with the reason.

## 6. Order of work

1. **No training:** run the audit's PART C on the existing logs
   (`docs/quality_signal_audit.md` §7). Check the window-clean AUROC of
   `a_hind` / `d_progress_post`, the difficulty stratification, and whether
   σβ rises from expert to novice.
2. **C7b** `examples/campaigns/rq5_quality_score_ext.yaml`: `logged`,
   `quality-beh`, `quality-hind`, `quality-prog`, `quality-outc`. That is 54
   new cells if `logged_*` / `quality-beh_*` are copied from C7. Drop an arm
   whose audit AUROC is ≈ 0.5.
3. **C8** `examples/campaigns/rq5_quality_imitation.yaml`: `logged`,
   `quality-unif-bc`, `quality-beh-bc`, `quality-beh-bc-cons`, with
   λ_BC = 0.1 (**tune in a smoke cell first**). That is 54 new cells if
   `logged_*` is copied.

## 7. Limits

- **The window length K is a free parameter** (10 steps = 1 s at 10 Hz).
  Sweep it offline with the audit (`--post_release_steps`) before fixing it
  for a campaign.
- **V(s) comes from the same ensemble** whose counterfactual use failed.
  Evaluated on-policy it should be far better behaved, but that is an
  assumption the audit tests, not a guarantee.
- **σβ assumes one operator per run and a roughly stationary operator.**
  Synthetic operators satisfy both; a human who learns during the session
  does not.
- The window features of a segment use steps **after** it, so they are only
  available at episode end. That was already true for all quality scoring.
