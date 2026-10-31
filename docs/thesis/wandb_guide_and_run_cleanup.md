# W&B metrics guide + run-cleanup recommendation

A distilled reference: **what is logged to Weights & Biases, what to look at, how to
interpret it**, and **which `runs/` to keep vs delete**. The exhaustive metric
reference is `handoff/handoff/WANDB_metrics_guide.md`; setup (account, online/offline,
syncing) is in [../wandb.md](../wandb.md).

> **The golden rule:** W&B shows **training dynamics** (*why* a cell ended where it
> did, and problems while it runs). The **thesis numbers** — autonomous success on the
> frozen ruler, operator dependency (ICC) — come from
> `runs/<campaign>/report/dependency.md` and `eval_runs/*/cell.json`, **not** from W&B.

---

## 1. How the data is organised

- **One run per cell**, owned by the learner (the actor forwards its episode stats).
- Project `hil-serl`; **group = the campaign** (`WANDB_RUN_GROUP`); entity
  `ytelha-rwth-aachen-university`.
- The run **config** holds every flag — `aqi_mode` / `arm`, `operator_id`,
  `synthetic_operator` (tier), `seed`, `critic_ensemble_size`, `quality_score_mode`,
  `bc_loss_weight`, … — so you **group and filter by them** in the UI.
- The **x-axis ("Step") is the learner's update count** for everything, including the
  actor's episode stats. Methods make different numbers of updates per env step, so
  **curves of different arms are not aligned by Step** — compare by episode index or
  env steps (Section 4).
- A **resumed cell = a new run** (new timestamp); concatenate by run-name prefix.
- The **System tab** (GPU/CPU/RAM) is logged automatically — use it for memory issues.

---

## 2. The metrics, grouped

### 2.1 Critic — `critic/*`

| key | meaning | normal | warning sign |
|---|---|---|---|
| `critic/critic_loss` | mean squared TD error | high early, falls, then moves with new successes | steady growth or spikes that don't recover |
| `critic/predicted_qs` | mean $Q(s,a)$ over batch and critics | with sparse 0/1 reward, stays in $[0,1]$ | $>1$ = **over-estimation** — watch on the $M{=}1$ fixed base |
| `critic/target_qs` | mean TD target | tracks `predicted_qs` | a growing gap = critic lags targets |
| `critic/rewards` | share of success transitions in the batch | small, grows with online successes | $\approx 0$ for long = no success data |
| `critic/critic_bootstrap_frac` | share of (transition, critic) pairs used | $\approx 0.8$ for $K>2$ modes | any other value = bootstrap masks wrong |

The Q-bound heuristic: with a sparse terminal reward and $\gamma=0.97$,

$$Q(s,a)\in[0,1]\quad\Rightarrow\quad \texttt{predicted\_qs}>1 \text{ is a red flag.}$$

### 2.2 Actor and temperature — `actor/*`, `temperature/*`

| key | meaning | normal | warning sign |
|---|---|---|---|
| `actor/actor_loss` | $-\mathbb{E}[Q-\alpha\log\pi]$ (+ $\lambda_{\text{BC}}\cdot$BC) | goes more negative as $Q$ rises; not a performance measure | wild oscillation with the critic |
| `actor/entropy` | policy entropy $-\mathbb{E}[\log\pi]$ | falls toward the target $-3$ and hovers | far below $-3$ early = premature collapse |
| `actor/temperature` | $\alpha$ | starts 0.01, adapts to hold entropy at target | $\alpha$ exploding, or $\to 0$ with high entropy |
| `temperature/temperature_loss` | the $\alpha$ objective | small, noisy around 0 | constantly large = $\alpha$ not converged |
| `actor/aqi_bc_loss` | NLL of operator actions (weighted BC) | falls as the policy imitates kept takeovers | very large = saturated operator actions dominate |
| `actor/aqi_bc_weight_mean`, `actor/aqi_bc_samples` | mean BC weight; samples with weight $>0$ | grow as interventions accumulate | $\approx 0$ = BC inert |

### 2.3 Quality-aware replay — `aqi/*` (quality / aqi modes)

| key | meaning | normal | warning sign |
|---|---|---|---|
| `aqi/intervention_frac` | share of the demo/intervention buffer that is interventions | grows from 0 | stays 0 = no interventions reach the buffer |
| `aqi/q_int_mean` | mean stored quality score (0 for demos and warm-up) | $>0$ after `quality_warmup_steps` if some takeovers score positive | exactly 0 for long = warm-up not over / `uniform` mode |
| `aqi/priority_mean`, `aqi/priority_max` | sampling priorities | the **ratio** below shows whether the score moves sampling | ratio $\approx 1$ = priority inert (RQ2's original bug) |

Priority informativeness (the one number that says whether the score does anything):

$$\frac{p_{\max}}{p_{\text{mean}}}\;\gg\;1 \quad\text{(good)} \qquad\text{vs}\qquad \frac{p_{\max}}{p_{\text{mean}}}\approx 1 \quad\text{(inert).}$$

### 2.4 Actor episodes — `environment/*` (one point per episode)

| key | meaning | normal | warning sign |
|---|---|---|---|
| `environment/episode/r` | episode return = 1 if success, else 0 | rolling mean = **assisted** training SR | not the ruler / not autonomous SR |
| `environment/episode/l` | episode length (steps) | short on success; falls as policy improves | stuck at 300 = failures time out |
| `environment/episode/intervention_count` | takeover onsets | falls as the policy improves | stays high = not learning from takeovers |
| `environment/episode/intervention_steps` | steps under operator control = **human effort** | falls over training; lower for skilled operators | — |
| `environment/episode/aqi_requests` | gate request onsets (`active` / `aqi`) | $>0$ while the policy needs help | fades to $\approx 0$ early = the gate went silent (failure F6) |

### 2.5 Timing and system — `timer/*`, System tab

| key | meaning | normal | warning sign |
|---|---|---|---|
| `timer/total` | actor, one env step | $\approx 0.1$ s (10 Hz) | $\gg 0.1$ s = actor slow (rendering, costly AQI section) |
| `timer/aqi_trigger` | the gate section | small | large = costly gate (e.g. huge OOD buffer) |
| `timer/train_critics` | learner, one update | milliseconds | large = slow updates (prioritised sampling is slower) |
| System tab: GPU mem | — | flat | creeping up = leak / impending OOM |

`active` runs also log `aqi_requests`; `aqi` adds `aqi/*`; HiRIL adds `hiril/*`
(`du_mean`, `request_rate`, `tau_risk`, …); E2HiL adds `actor/e2hil_*`.

---

## 3. Reading one run — a 10-minute checklist

1. **Alive and at speed?** `timer/total` $\approx 0.1$ s; System-tab GPU memory flat.
2. **Data flowing?** `environment/*` points appear each episode; `aqi/intervention_frac` grows (quality modes).
3. **Critic learning?** `critic_loss` falls; `predicted_qs` leaves 0 after first successes and stays $\le 1$.
4. **Policy healthy?** `actor/entropy` $\to -3$ without early collapse; temperature settles.
5. **Task learned?** rolling mean of `episode/r` rises; `episode/l` and `intervention_steps` fall.
6. **For the active part (your focus):** `aqi_requests` $>0$ and **not fading early**; report human effort and success per unit effort (Section 4).
7. **Then check the truth:** the cell's `eval_runs/<cell>/cell.json` (ruler SR). A run can look good in training (assisted successes) and still score low on the ruler.

---

## 4. Comparing arms fairly

- **Group** by `config.arm` (or `aqi_mode`); **colour** by `config.synthetic_operator`
  (tier) or `operator_id`; use `config.seed` for replicates.
- **Do not align by Step.** Plot against episode index (the n-th `environment/*`
  point) or against env steps (cumulative sum of `episode/l`). Both need an export.
- **Smooth** episode metrics over a rolling window of 20 (as the convergence rule does).
- **Separate autonomous from assisted success.** An autonomous success is

  $$\big(r = 1\big)\;\wedge\;\big(\texttt{intervention\_count}=0\big),$$

  which is what the convergence rule and the thesis count.
- **Loss levels are not comparable across architectures** (2 vs 8 critics, mean vs min,
  BC terms all change the scale) — compare **trends**, not levels.
- **Report human effort** — the key `active`-vs-`logged` number is success per unit of
  operator effort:

  $$\text{SR}_{1000}\;=\;\frac{\#\,\text{successes}}{\sum_{\text{episodes}} \texttt{intervention\_steps}}\times 1000.$$

### Export (per-episode table for a whole group)

```python
import wandb, pandas as pd
api = wandb.Api()
runs = api.runs("ytelha-rwth-aachen-university/hil-serl",
                filters={"group": "<campaign group>"})
rows = []
for run in runs:
    cfg = run.config
    keys = ["_step", "environment/episode/r", "environment/episode/l",
            "environment/episode/intervention_count",
            "environment/episode/intervention_steps"]
    for i, h in enumerate(run.scan_history(keys=keys)):
        rows.append(dict(run=run.name, arm=cfg.get("arm") or cfg.get("aqi_mode"),
                         operator=cfg.get("operator_id"),
                         tier=cfg.get("synthetic_operator"), seed=cfg.get("seed"),
                         episode=i, success=h["environment/episode/r"],
                         length=h["environment/episode/l"],
                         takeovers=h["environment/episode/intervention_count"],
                         effort=h["environment/episode/intervention_steps"]))
df = pd.DataFrame(rows)
df["env_steps"] = df.groupby("run")["length"].cumsum()
df["autonomous_success"] = (df.success > 0) & (df.takeovers == 0)
```

---

## 5. Run cleanup — keep vs delete

Context: disk is **37% used (1.1 TB free)**, `runs/` is **266 GB** — no space pressure,
so this is about tidiness. (The bulky replay-buffer dumps were already deleted.)

### Keep — canonical thesis results (cited by docs / reports)

| run | size | why |
|---|---|---|
| `rq4_ensemble_convergence` | 57 G | the fixed base for everything (RQ4) |
| `rq3_plain_logged_quality` | 61 G | RQ3 structural ablation |
| `rq2_quality_replayonly` | 30 G | RQ2 quality thread |
| `rq2_quality_replayonly_ei` | 18 G | RQ2 (fixed signal) |
| `rq2_quality_replayonly_ei_warmup` | 17 G | RQ2 (remedies applied) |
| `quality_on_rq4_diag` | 4.5 G | quality diagnostic (rq4 §11) + audit source |
| `quality_behavioral_diag` | 4.6 G | behavioral diagnostic + leakage-audit source |
| `rq1_3seed_nopretrain` | 9.6 G | holds the policy-oracle `certified` checkpoint referenced by configs |

### Delete — smoke / aborted / probe junk (~7 G, safe)

| run | size | note |
|---|---|---|
| `rq1_baseline` | 92 K | empty (0 cells) |
| `rq2_smoke` | 369 M | smoke test |
| `po_cyl_2cell` | 1.5 G | 2-cell cylinder smoke |
| `policy_oracle_rq1_hilserl_cuboid_2op3seed` | 1.7 G | 0 cells done (aborted) |
| `probe_longer` | 1.9 G | probe |
| `probe_longer_halflr` | 1.9 G | probe |
| `active_validity` | 856 M | the aborted A1 run — restarts fresh anyway |

### Your call — RQ1 exploration (~50 G)

`rq1_3seed_pretrain`, `rq1_s075_5seed`, `po_rq1_campaign`, `rq1_newanchors`,
`rq1_newanchors_pretrain`, `rq1_unified_mon`, `rq1_mini_check`. The RQ1 thread was
abandoned (pivot to the policy oracle; see
[rq1_campaign_engineering_log.md](rq1_campaign_engineering_log.md)). If the
lessons-learned chapter cites them, keep their `report/` folders but delete the bulky
`checkpoints/` inside; otherwise delete whole.

**Also small and safe:** stray `runs/*.out` / `*.nohup` logs, `examples/aqi_logs/`
(old shared logs), `examples/MUJOCO_LOG.TXT`.

**Recommendation:** delete the junk tier now; keep the canonical runs; decide on the
RQ1 tier when writing the methodology chapter.
