# W&B metrics: what is logged, how to read it, how to extract it

Setup (account, `online`/`offline`, syncing) is in [`wandb.md`](wandb.md).
This page is about the **numbers**:
- every metric the code logs, with where it comes from and what normal and
  abnormal look like;
- a checklist for reading one run;
- how to compare arms of a campaign fairly;
- how to export the data;
- what else could be logged, and why.

**The rule to keep in mind:** W&B shows the **training dynamics**. The thesis
results (autonomous success on the frozen ruler, operator dependency) come
from `runs/<campaign>/eval_runs/*/cell.json` and `report/dependency.md`, not
from W&B. Use W&B to understand *why* a cell ended where it did, and to spot
problems while it runs.

---

## 1. How data reaches W&B

```
ACTOR (every episode end)                   LEARNER (owns the W&B run)
 {"environment": info}  ── send-stats ──►   stats_callback: wandb.log(payload, step=<learner step>)
 {"timer": actor timers} (every log_period)
                                            every log_period (10) learner steps:
                                              update_info   → critic/*, actor/*, temperature/*
                                              timer         → timer/* (learner)
                                              priority_stats→ aqi/*   (quality, aqi modes)
                                            during demo pretraining → demo_pretrain/*
```

- **One W&B run per cell, owned by the learner.** The actor doesn't open a
  run; it forwards its statistics.
  - Run name: `<exp>_<arm or mode>_<operator>_<timestamp>`.
  - A resumed cell starts a **new** run (new timestamp).
  - The group is `WANDB_RUN_GROUP` (the campaign).
- **The run config contains every training flag**: `aqi_mode`, `arm`,
  `operator_id`, `synthetic_operator` (tier), `seed`,
  `critic_ensemble_size`, `quality_score_mode`, `bc_loss_weight`, … Group
  and filter by them in the UI.
- **The x-axis ("Step") is the learner's update count, for everything,**
  including the actor's episode statistics (logged at the learner's current
  step when they arrive).
  - Methods make different numbers of updates per env step (≈ 1.3–2.7 on an
    H100, finding F3), so **curves of different arms are not aligned by
    Step**.
  - Compare by episode index or env steps instead (§4).
- **W&B also records system metrics** automatically (GPU utilisation and
  memory, CPU, RAM) in the run's System tab. They're useful for memory
  problems.
- On the cluster, runs are **offline** by default (`WANDB_MODE=offline` in
  `cluster/claix/env.sh`). They sit in `checkpoints/<cell>/wandb/` until you
  `wandb sync` them.

## 2. Metric reference

"Normal" describes this task (sparse 0/1 reward, success and time-out both
terminal, γ = 0.97, 6-D actions). Use it as a heuristic, not a hard
threshold.

### 2.1 Critic: `critic/*`

From `sac.py → critic_loss_fn`, averaged over the batch and the K critics.

| key | meaning | normal | warning signs |
|---|---|---|---|
| `critic/critic_loss` | mean squared TD error | high at the start, falls, then moves with the data (new successes raise it temporarily) | steady growth, or sudden spikes that don't recover → instability; with prioritised replay, check `aqi/*` too |
| `critic/predicted_qs` | mean Q(s, a) over batch and critics | rises from ≈ 0 once successes enter the buffers. With a sparse 0/1 terminal reward, **Q should stay within [0, 1]** | **> 1: over-estimation** (e.g. no min over critics, M = 1); stuck near 0 after many successes → the critic isn't propagating reward; M = 2 arms sit lower (pessimism) |
| `critic/target_qs` | mean TD target | tracks `predicted_qs` | a growing gap → the critic lags its targets |
| `critic/rewards` | mean reward in the batch = the share of success transitions the critic sees | small (a few %), grows as online successes appear | ≈ 0 for long → no success data; prioritised replay changes it by design |
| `critic/critic_bootstrap_frac` | share of (transition, critic) pairs used | ≈ 0.8 with bootstrapped ensembles (K > 2 modes), 1.0 otherwise | a different value → bootstrap masks not what you think |

### 2.2 Actor and temperature: `actor/*`, `temperature/*`

From `sac.py → policy_loss_fn`, `temperature_loss_fn`.

| key | meaning | normal | warning signs |
|---|---|---|---|
| `actor/actor_loss` | −E[Q − α log π] (+ λ·BC) | goes down (more negative) as Q rises. **Not a performance measure** on its own | wild oscillation together with the critic |
| `actor/entropy` | policy entropy −E[log π] | falls from high values towards the **target −3** (= −6/2) and hovers there | far below −3 early → premature collapse; E2HiL claims a smoother decline |
| `actor/temperature` | α | starts at 0.01, adapts so that entropy ≈ target | α exploding → the policy can't keep entropy at target; α → 0 with entropy far above target → reward dominates |
| `temperature/temperature_loss` | the α objective; its sign says whether entropy is above or below target | small, noisy around 0 | constantly large → α hasn't converged |
| `actor/aqi_bc_loss` | NLL of operator actions under the policy (weighted BC) | decreases as the policy imitates the kept takeovers | very large (≈ 10 nats per saturated action dimension) → saturated operator actions dominate (EVALUATE_FIRST Q9) |
| `actor/aqi_bc_weight_mean`, `actor/aqi_bc_samples` | mean BC weight in the batch; number of batch samples with weight > 0 | grow as interventions accumulate | ≈ 0 → BC is inert (no interventions in the batch, or all weights 0) |
| **compare** λ·`aqi_bc_loss` with the RL part of `actor_loss` | which term drives the actor | — | λ·BC ≫ RL part → mostly imitation; ≪ → BC irrelevant: tune λ (quality plan Q2a) |
| `actor/e2hil_keep_ratio`, `actor/e2hil_abs_c_lo/hi`, `actor/e2hil_{demo,online,intervention}_{abs_c_mean,keep_frac}` | E2HiL's selection | keep ratio ≈ 0.85 by construction; the paper expects the **intervention** keep fraction to be lowest | keep ratio far from 0.85 → the mask isn't working |
| `critic_lr`, `actor_lr`, … | learning rates, if the optimiser exposes them | constant | — |

### 2.3 Quality-aware replay: `aqi/*` (quality and aqi modes)

From `data_store.py → QualityAwareReplayBufferDataStore.priority_stats`,
over the whole demo/intervention buffer.

| key | meaning | normal | warning signs |
|---|---|---|---|
| `aqi/intervention_frac` | share of the demo/intervention buffer that is interventions | grows from 0 as takeovers are added | stays 0 → no interventions reach the buffer |
| `aqi/q_int_mean` | mean stored quality score (0 for demos and warm-up) | > 0 after `quality_warmup_steps` if some takeovers score positive | exactly 0 for long → warm-up not over, all scores ≤ 0, or `uniform` mode |
| `aqi/priority_mean`, `aqi/priority_max` | sampling priorities | **the ratio max/mean shows whether the score moves sampling at all** | max/mean ≈ 1 → **priority inert** (RQ2's original problem); a huge ratio → a few transitions dominate the batches |

### 2.4 Actor episodes: `environment/*` (one point per episode)

From `train_aqi_rlpd.actor`: the `info` dict of the last step of each
episode.

| key | meaning | normal | warning signs |
|---|---|---|---|
| `environment/episode/r` | episode return = **1 if the episode succeeded** (sparse reward), else 0 | its rolling mean is the **training** success rate | this includes **operator-assisted** successes, so it isn't autonomous success and isn't the ruler |
| `environment/episode/l` | episode length in steps | successful episodes are short; mean length falls as the policy improves | stuck at the limit (300) → failures time out |
| `environment/episode/t` | episode wall time | ≈ length × 0.1 s at 10 Hz | much longer → the actor is slow (see `timer/*`; broken rendering, F5) |
| `environment/episode/intervention_count` | takeover onsets in the episode | falls as the policy improves (HIL-SERL expectation) | stays high → the policy doesn't learn from takeovers |
| `environment/episode/intervention_steps` | steps under operator control = **human effort** | falls over training; lower for skilled operators | — |
| `environment/episode/aqi_requests` | the gate's request onsets in the episode (`active`/`aqi`) | > 0 while the policy needs help | ≈ 0 for most of training → the gate stopped asking (F6); many per episode → an annoying request rate |
| `environment/succeed` | success flag of the last step | = r | — |
| `environment/synthetic_operator/*` (`z`, `zhat`, `taking_over`, …) | the operator's state **at the last step only** | — | little use as a curve; per-step data is in `steps.jsonl` |
| `environment/episode/aqi_mode` | a text label | — | ignore in plots |

HiRIL runs add `hiril/du_mean`, `hiril/request_rate`,
`hiril/intervention_rate`, `hiril/tau_risk` and `hiril/explore_std` per
episode, plus `hiril_replay/*` (human share, human priority share, bonus)
from the learner. `request_rate` ≈ 0 after `tau_risk` is set is finding F2.

### 2.5 Timing: `timer/*` (seconds, averaged over each logging interval)

| key | who | normal | warning signs |
|---|---|---|---|
| `timer/total` | actor, one env step | ≈ 0.1 s (10 Hz) | **≫ 0.1 s → the actor is slow**: rendering (F5), a slow AQI section, CPU starvation |
| `timer/step_env`, `timer/sample_actions`, `timer/aqi_trigger` | actor parts | `step_env` dominates | `aqi_trigger` large → a costly gate section (e.g. a huge OOD buffer) |
| `timer/train`, `timer/train_critics`, `timer/sample_replay_buffer` | learner, one update | milliseconds | 1 / (`train` + `train_critics` + sampling) ≈ updates per second; prioritised sampling is slower |
| `demo_pretrain/step` | learner | counts up to `demo_pretrain_steps` | — |

## 3. Reading one run: a 10-minute checklist

1. **Is it alive and at speed?** `timer/total` ≈ 0.1 s; the System tab shows
   GPU memory stable, not creeping up.
2. **Is data flowing?** `environment/*` points appear regularly (one per
   episode), and `aqi/intervention_frac` grows (quality modes).
3. **Is the critic learning?** `critic/critic_loss` falls early.
   `predicted_qs` leaves ≈ 0 after the first successes and stays ≤ 1.
4. **Is the policy healthy?** Entropy falls towards ≈ −3 without crashing
   early; the temperature settles.
5. **Is the task being learned?** The rolling mean of
   `environment/episode/r` rises; `episode/l` falls; `intervention_steps`
   falls.
6. **Mode-specific:**
   - quality: `aqi/priority_max` / `priority_mean` clearly > 1;
     `q_int_mean` > 0 after warm-up;
   - BC: `aqi_bc_loss` falls, and λ·BC vs the RL part is balanced;
   - active: `aqi_requests` > 0 and not fading to 0 early;
   - E2HiL: keep ratio ≈ 0.85; HiRIL: `request_rate` > 0 after calibration.
7. **Then check the truth:** the cell's `eval_runs/<cell>/cell.json`
   (ruler SR). A run can look good in training (assisted successes) and
   still score low on the ruler, or the other way round.

## 4. Comparing arms and operators fairly

- **Group** runs by `config.arm` (or `aqi_mode`), and colour by
  `config.synthetic_operator` (tier) or `operator_id`. Use
  `config.seed` for replicates.
- **Don't align by Step** (learner updates, which differ by method).
  Instead:
  - plot against **episode index**: the n-th `environment/*` point of each
    run;
  - or against **env steps**: the cumulative sum of `environment/episode/l`.

  Both need an export (§5); the W&B UI can't do this directly unless env
  steps are logged (§6, W1).
- **Smooth** episode metrics: a rolling mean over 10–20 episodes, as the
  convergence rule does (window 20).
- **Separate autonomous from assisted success.** Episodes with
  `intervention_count == 0` and `r == 1` are autonomous successes; that's
  what the convergence rule counts.
- **Loss values are not comparable across architectures.** 2 vs 8 critics,
  mean vs min over critics, and BC terms all change the scale. Compare
  trends, not levels.
- **Report the human effort:** total `intervention_steps` per cell, and
  success per 1000 intervention steps. This matters especially when
  comparing `active` with `logged` (EXPERIMENT_PLAN_active_learning §5).
- **Resumed cells have several runs.** Concatenate them by run name prefix
  and config.

## 5. Exporting the data

The W&B public API, run from any machine with `wandb login`. This builds
one per-episode table for a whole campaign group:

```python
import wandb, pandas as pd
api = wandb.Api()
runs = api.runs("<entity>/hil-serl", filters={"group": "<campaign group>"})
rows = []
for run in runs:
    cfg = run.config
    # keys present in EVERY mode; scan_history drops rows missing any listed key,
    # so mode-specific keys (e.g. environment/episode/aqi_requests) need their own scan
    keys = ["_step", "environment/episode/r", "environment/episode/l",
            "environment/episode/intervention_count",
            "environment/episode/intervention_steps"]
    for i, h in enumerate(run.scan_history(keys=keys)):   # full history, not sampled
        rows.append(dict(run=run.name, arm=cfg.get("arm") or cfg.get("aqi_mode"),
                         operator=cfg.get("operator_id"), tier=cfg.get("synthetic_operator"),
                         seed=cfg.get("seed"), episode=i, learner_step=h["_step"],
                         success=h["environment/episode/r"], length=h["environment/episode/l"],
                         takeovers=h["environment/episode/intervention_count"],
                         effort=h["environment/episode/intervention_steps"]))
df = pd.DataFrame(rows)
df["env_steps"] = df.groupby("run")["length"].cumsum()
df["autonomous_success"] = (df.success > 0) & (df.takeovers == 0)
print(df.groupby(["arm", "tier"]).agg(episodes=("episode", "max"),
                                      effort=("effort", "sum"),
                                      auto_sr=("autonomous_success", "mean")))
```

Notes:
- `scan_history(keys=...)` only returns rows where **all** those keys
  exist, so query per-episode keys separately from learner keys
  (`critic/*`, `actor/*`, `aqi/*`).
- For learner metrics, use `keys=["_step", "critic/predicted_qs",
  "actor/entropy", ...]`.
- Offline runs must be `wandb sync`ed before the API sees them.

**For the thesis analyses, prefer the campaign's own files**, which are
more complete and don't depend on W&B:

| question | best source |
|---|---|
| autonomous success on the ruler, operator dependency | `eval_runs/*/cell.json`, `report/dependency.{md,json}` |
| per-episode success, length, takeovers (every mode) | `checkpoints/<cell>/episodes.jsonl` |
| per-step gate terms, requests, σ_Q, ground truth | `aqi_logs/<cell>_actor/steps.jsonl` |
| per-takeover quality scores and features | `aqi_logs/<cell>_actor/segments.jsonl` |
| critic / actor / entropy / priority dynamics | **W&B** (not stored elsewhere) |

W&B is the only place the learner's optimisation metrics are kept. That's
its unique value.

## 6. What else could be logged (proposals, not implemented)

Each needs a small logging-only change, a test, and for the training
machine a handoff patch. Propose them before implementing.

| id | add | why (which question / finding) | where | cost |
|---|---|---|---|---|
| **W1** | the actor's env step (and episode index) in the `environment` payload; the learner step at arrival → `train/updates_per_env_step` | lets the UI plot against env steps; shows updates per env step live (**F3**) | `train_aqi_rlpd.actor` (payload), `stats_callback` | trivial |
| **W2** | σ_Q of the training batch: mean and 90th percentile of the std over critics of `predicted_qs`; Q percentiles | watch the ensemble's disagreement collapse during training (**F6**, active plan AQ1) | `sac.critic_loss_fn` info | tiny, inside jit |
| **W3** | per-episode gate summary: request onsets by reason (score / safety), mean z per term, which term decided each onset | shows the gate's behaviour live instead of only counting requests (**F6**, the safety-override share) | `aqi/runtime.end_episode` → the payload | small |
| **W4** | per-episode quality summary: takeover segments, mean q, mean w, share excluded, and the priority **boost** of interventions vs demos (median and 90th-percentile ratio) | shows whether the score actually moves sampling (RQ2's inert-priority lesson) and what it rewards | `runtime.end_episode`; `priority_stats` | small |
| **W5** | the convergence mini-evaluation SR when it runs, and `CONVERGED` events | ties the training curve to the autonomous check | the actor's convergence block | trivial |
| **W6** | gradient norm per network | diagnoses instability (spikes before a loss blow-up) | `JaxRLTrainState.apply_loss_fns` | small, inside jit |
| **W7** | per-episode outcome of each takeover (lateral error removed, aligned or jammed at handback), with patch 0010 ground truth | a live view of intervention quality with timing-free labels (quality plan Q1) | `runtime.end_episode` | small |

W1 and W2 give the most insight for the least code. Remember that anything
computed inside the jitted update costs learner time, which slightly shifts
updates per env step. Keep additions light, and never mix cells logged with
and without an addition in one comparison if the addition is expensive.

## 7. Pitfalls when interpreting

1. **Step = learner updates**, not env steps (§1, §4).
2. **Training success ≠ autonomous success ≠ ruler SR.** `episode/r`
   counts assisted successes. The thesis numbers come from the ruler.
3. **Loss levels across architectures** aren't comparable (§4).
4. **Early episodes** (before the learner's first update) are logged at the
   same Step, and W&B keeps only the last value per key per step, so a few
   first episodes can be missing. The complete record is
   `checkpoints/<cell>/episodes.jsonl`.
5. **Last-step values:** `environment/synthetic_operator/*` shows the final
   step of each episode, not the episode.
6. **Resumed cells** are split over several runs (§4).
7. **Offline runs** are invisible until synced (§1).
8. **Sampled vs full history:** `run.history()` downsamples; use
   `scan_history()` for analysis.
9. **Groups on the cluster:** Slurm jobs start with a clean environment
   (`SLURM_EXPORT_ENV=NONE`), so `WANDB_RUN_GROUP` from your shell doesn't
   reach them, and cluster runs are **ungrouped**. Filter them by run name
   (it contains the arm and operator) and config instead. On the training
   machine, set it in the launch command, as the campaign headers do.
