# Active rate-matched frontier (campaign A) — results & QA log

Companion to the RQ2–RQ4 logs. This is the **active half** of the thesis: does
timing operator help by the policy's own uncertainty (σ_Q ensemble disagreement)
beat the hand-engineered geometric monitor **at a matched intervention budget**?

- Campaign: `runs/active_ratematch` (108 cells, finished 2026-10-21, no crashes).
- Config: [`examples/campaigns/active_ratematch.yaml`](../../examples/campaigns/active_ratematch.yaml).
- Report: `runs/active_ratematch/report/dependency.md` (logged-baseline, primary),
  `…/dependency_baseline_hilserl.md` (hilserl-baseline), `…/plots/frontier.png`.

---

## 0. Headline

**`active-q55` dominates the SR-vs-cost frontier.** At 42% intervention it reaches
**87.4% autonomous success** — the best of all arms — vs `logged`'s 74.3% at 54%
cost, i.e. **more autonomous success at less human help**, and operator-independent
(ICC 0.01). The paired per-cell test is **significant: +13.2 pp, 13W/3L, sign-p =
0.021**. It **lifts** the weak tiers (intermediate 63.6 → 95.1) rather than leveling.

Two deeper claims are **weaker** (honest): σ_Q's *timing* is only *suggestively*
better than random at matched rate (+4.9 pp, p = 0.30), and no arm significantly
beats the plain `hilserl` baseline on raw SR (though all online-help arms erase
`hilserl`'s operator-dependence).

---

## 1. The question and the confound it fixes

The prior `active_effect` run compared `logged` (~54% intervention) vs `active`
(~9%) and read the SR gap as "trigger quality". That is confounded by **volume**:
6× less help trivially lowers SR for good operators and raises it for bad ones
(level, not lift). The only clean question is **at a matched intervention budget,
does σ_Q-timed help beat engineered-timed help on autonomous SR, and dominate the
SR-vs-cost frontier?**

## 2. New mechanism + code (this session)

- **Quantile trigger** `query_target_rate` (new `AQIConfig` field): request the
  top-`rate` fraction of recent A(s_t) via a running quantile, so the request rate
  is **explicit and scale-free** (robust to σ_Q's drift as the RQ4 ensemble
  converged — a fixed `tau_A` fired at an uncontrolled ~9%). This is why the old
  `active-loud` failed: A(s_t) is z-normalised, so a louder ensemble (`bootstrap_p`)
  **cannot** change the rate, only the ranking.
- **Arm-routable duty cycle** `query_cooldown_steps` / `query_min_on_steps`: the
  realized rate is capped by `min_on/(min_on+cooldown)`, so the quantile only has
  leverage when cooldown is small. `min_on=10` is the floor for the operator to
  react; **cooldown is the rate lever**.
- (Also added this session, parked for later campaigns: an **oracle** quality score
  `quality_score_mode=oracle` for the ceiling study, and a **milder-novice** knob
  `operators.tier_s_exec`. See §9.)

Files touched: `serl_launcher/serl_launcher/aqi/{config.py, query_gate.py}`,
`examples/{train_aqi_rlpd.py, run_campaign.py}`.

## 3. Calibration probe (the "is it worth it" gate)

Offline simulation **cannot** predict the realized rate (it depends on σ_Q's
temporal clustering) and i.i.d. sims mis-estimate it. A worst-case i.i.d. sim showed
the quantile sweep **collapses at the default cooldown=20** (q15/q30/q50 →
0.28/0.31/0.32, indistinguishable) and separates at cooldown=5 — but the absolute
numbers were unreliable. So before committing ~6 days, a **3-hour probe**
(`runs/active_ratematch_probe`, int00, 8k steps) measured the real rates:

| arm | measured intervention rate |
|---|---|
| `active-taua` (tau_A=0.5) | 3.7% |
| `active-q10` (cd=5) | 14.1% |
| `active-q30` (cd=5) | 33.9% |
| `active-q55` (cd=5) | 43.6% |
| `logged` | 59.6% |

Clean separation, no collapse, brackets `logged`. A was then calibrated to
`active-q10/q30/q55 @ cd=5`, `random-q30` at `random_request_rate=0.08` (matched to
q30), `steps_cap=35000`.

## 4. Design

6 arms × 3 tiers × 3 operators × 2 seeds = **108 cells**. RQ4 fixed base
(K=8 critics, M=1 subsample, per-member targets, 5000-step demo warm start,
LR 1.5e-4), cuboid peg, `signal` handback, `quality_cf_diag` on, best-window eval
on the frozen 9-pose ruler.

| arm | timing | role |
|---|---|---|
| `hilserl` | none (plain 2-critic) | floor / baseline |
| `active-q10` | σ_Q quantile, cd=5 | low-rate frontier point |
| `active-q30` | σ_Q quantile, cd=5 | mid-rate point |
| `active-q55` | σ_Q quantile, cd=5 | high-rate point |
| `random-q30` | random, matched to q30 | chance-timing control |
| `logged` | engineered geometric monitor | reference |

## 5. Results

### 5.1 The frontier (autonomous SR vs measured human cost)

Measured intervention rate = operator-controlled steps / all env steps (from
`aqi_logs/*/steps.jsonl`). SR = best-window autonomous SR on the frozen ruler.

| arm | interv. rate | autonomous SR | ICC | SR per interv-pt |
|---|---|---|---|---|
| `hilserl` | 0% | 79.4% | 0.662 | — |
| `active-q10` | 15.9% | 70.2% | 0.000 | 4.42 |
| `random-q30` | 25.7% | 76.3% | 0.000 | 2.97 |
| `active-q30` | 28.5% | 81.3% | 0.421 | 2.85 |
| **`active-q55`** | **41.8%** | **87.4%** | **0.010** | 2.09 |
| `logged` | 53.8% | 74.3% | 0.000 | 1.38 |

Plot: `runs/active_ratematch/report/plots/frontier.png`. The active arms (q30, q55)
sit above-and-left of the `logged` point → active dominates the frontier.

### 5.2 Skill gradient (mean SR per tier — lift vs level)

| arm | expert | intermediate | novice |
|---|---|---|---|
| `hilserl` | 97.5% | 64.2% | 76.5% |
| `active-q10` | 85.8% | 71.0% | 53.7% |
| `active-q30` | 100.0% | 87.0% | 56.8% |
| `active-q55` | 93.2% | 95.1% | 74.1% |
| `random-q30` | 89.5% | 71.6% | 67.9% |
| `logged` | 86.4% | 63.6% | 72.8% |

`active-q55` **lifts** the middle/top without dropping the novice — the thesis's
lift-not-level claim at matched budget.

### 5.3 Operator spread + "operator matters" permutation test

| arm | sigma_op | CV | p(operator matters) |
|---|---|---|---|
| `hilserl` | 0.227 | 0.286 | 0.001 |
| `active-q10` | 0.209 | 0.298 | 0.498 |
| `active-q30` | 0.229 | 0.282 | 0.119 |
| `active-q55` | 0.131 | 0.150 | 0.382 |
| `random-q30` | 0.153 | 0.200 | 0.662 |
| `logged` | 0.134 | 0.181 | 0.610 |

### 5.4 Training effort (converged fraction)

`hilserl` 15/18, `active-q10` 14/18, `active-q30` 8/18, `active-q55` 11/18,
`random-q30` 10/18, `logged` 11/18.

## 6. Paired per-cell tests (shared operator × seed grid)

### vs `logged` (baseline for active-vs-engineered)

| method | W/L/T | mean d | sign-p | rate vs logged |
|---|---|---|---|---|
| `active-q55` | 13/3/2 | **+13.2** | **0.021** | 42% vs 54% |
| `active-q30` | 9/6/3 | +7.0 | 0.607 | 29% vs 54% |
| `random-q30` | 11/5/2 | +2.1 | 0.210 | 26% vs 54% |
| `active-q10` | 9/7/2 | −4.1 | 0.804 | 16% vs 54% |
| `hilserl` | 9/8/1 | +5.1 | 1.000 | 0% vs 54% |

### vs `hilserl` (does any trigger beat no-trigger?)

| method | W/L/T | mean d | sign-p |
|---|---|---|---|
| `active-q55` | 10/4/4 | +8.0 | 0.180 |
| `active-q30` | 9/5/4 | +1.9 | 0.424 |
| `active-q10` | 7/10/1 | −9.3 | 0.629 |
| `random-q30` | 8/9/1 | −3.1 | 1.000 |
| `logged` | 8/9/1 | −5.1 | 1.000 |

### The decisive timing-vs-chance test (matched rate)

**`active-q30` vs `random-q30`: +4.9 pp, 10W/5L/3T, sign-p = 0.302** (rates 28.5%
vs 25.7%). σ_Q times help slightly better than chance, but **not significantly**.

## 7. Verdict (with honest strength ratings)

1. **`active` > `logged` — WELL SUPPORTED.** `active-q55` beats the engineered
   monitor on autonomous SR (+13.2 pp, p=0.021) at lower cost, lifts the weak
   tiers, and is operator-independent. Lead the thesis with this.
2. **σ_Q timing beats chance — WEAK / suggestive.** +4.9 pp, p=0.30 at matched
   rate. Consistent with the `active_validity` audit (σ_Q AUROC ≈ 0.5 vs
   ground-truth trouble). Most of active's win is efficient *volume*, not smart
   *timing*. Report as a transparent near-null.
3. **`logged`/`active` > baseline — on DEPENDENCE, not SR.** No arm significantly
   beats `hilserl` on SR (strong at 79.4%). But `hilserl` is operator-**dependent**
   (ICC 0.662, p=0.001) because demos are per-operator, and **any** online help
   (`logged`, `active-q55`, even `random`) washes that to ICC ≈ 0. Robust, but not
   unique to active.

**Caveats:** `active-q10` is unstable at low help (sigma2_within 0.093, some cells
collapse). `active-q30`'s ICC 0.421 is almost certainly n=9 noise (q10/q55 ≈ 0).
The active-vs-random gap is not perfectly rate-matched. All need **more seeds**
(2→4) to firm up claims 2 and 3.

## 8. Artifacts

| what | path |
|---|---|
| primary report (logged baseline) | `runs/active_ratematch/report/dependency.md` |
| hilserl-baseline report | `runs/active_ratematch/report/dependency_baseline_hilserl.md` |
| logged-baseline backup | `runs/active_ratematch/report/dependency_baseline_logged.md` |
| frontier plot | `runs/active_ratematch/report/plots/frontier.png` |
| standard plots + intervention_rates.csv | `runs/active_ratematch/report/plots/` |
| per-cell eval | `runs/active_ratematch/eval_runs/<cell>/cell.json` |
| per-step logs (rates, σ_Q, gt) | `runs/active_ratematch/aqi_logs/<cell>_actor/steps.jsonl` |
| calibration probe | `runs/active_ratematch_probe/` |
| config | `examples/campaigns/active_ratematch.yaml` |

Reproduce the frontier + paired tests: the script reads `eval_runs/*/cell.json`
(`best_window_sr`) and `aqi_logs/*_actor/steps.jsonl` (`intervening`), pairs by
operator × seed, and writes `report/plots/frontier.png`.

## 9. Next steps + parked campaigns

- **+2-seed extension** of `active-q55`/`active-q30`/`random-q30`/`logged` to settle
  the timing (claim 2) and baseline (claim 3) tests — cheapest high-value follow-up.
- **C `aqi_oracle_quality.yaml`** — the natural next headline: does *selecting which
  corrections to keep* (the simulator-ground-truth quality ceiling) add to *when to
  ask*? Directly addresses why timing alone was only suggestive.
- Parked configs (all calibrated, `--plan`/`--dry_run` verified, not launched):
  `active_trigger_loudness.yaml` (B, σ_Q ranking at matched rate),
  `aqi_oracle_quality.yaml` (C, quality ceiling),
  `active_mild_novice.yaml` (D, weak-but-informative novice),
  `active_demos_sweep.yaml` (E, demo robustness).
