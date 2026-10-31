# AQI active + quality campaigns — synthesis & interpretation

**Master analysis of every campaign run in this experimental arc** (Oct 2026):
the active-query trigger (campaign A), the quality-gate oracle ceiling
(campaign C), and the demo mechanism probe. It ties the detailed per-campaign
logs together into one interpretation and the thesis-level conclusions.

Detailed companions (read for full tables, per-cell data, caveats):
- [active_ratematch_results_and_qa_log.md](active_ratematch_results_and_qa_log.md) — campaign A
- [aqi_oracle_quality_results_and_qa_log.md](aqi_oracle_quality_results_and_qa_log.md) — campaign C + the demo probe (§8)

All on the same base: RQ4 fixed ensemble (K=8 critics, M=1 subsample, per-member
targets, 5000-step demo warm start, LR 1.5e-4), cuboid peg, synthetic operators
at 3 skill tiers (3 operators/tier × 2 seeds), autonomous success on a frozen
9-pose ruler, `signal` handback.

> **Note added 2026-10-31.** The quality arms below changed three things at once: the oracle score, the default hard
> filters (`outcome_filter` drops every takeover from a failed episode, whatever the oracle says) and a BC loss of 0.1.
> So "a perfect score amplifies dependence" is not yet established. See [../novice_ablation.md](../novice_ablation.md)
> for the test that separates them.

---

## 0. Executive summary

AQI has two components: **when to ask** the operator for help (the active σ_Q
trigger) and **which corrections to keep** (the quality gate). This arc tested
each at the ceiling.

- **Timing helps (campaign A).** σ_Q-timed help at a moderate rate (`active-q55`)
  reaches the best autonomous success (87.4%) at **lower** human cost than the
  engineered monitor (42% vs 54%), is operator-independent (ICC 0.01), and lifts
  the weak tiers — a **significant** win over `logged` (+13.2 pp, p=0.021). But
  σ_Q's *timing* is only **suggestively** better than random at matched rate
  (+4.9 pp, p=0.30): most of the gain is efficient *volume*, not smart *timing*.
- **Quality hurts (campaign C).** A **perfect simulator-oracle** quality score
  does NOT reduce operator-dependence — it **dramatically amplifies** it (ICC
  0.18 → 0.93) by **collapsing the novice tier** (65% → 9–36%). It keeps good
  operators' corrections (expert/intermediate → 100%) and correctly discards bad
  novice ones, starving the novice policy. Because the **oracle** proves this,
  it is a *paradigm* problem, not a broken-score problem.
- **The harm is robust (demo probe).** Tripling the novice's demos (6→20) does
  **not** rescue the quality-collapsed novice (stays ~27–36% vs `logged` ~60%).
  So the quality-induced dependence is not a demo-quantity artifact.

**Headline conclusion:** use the **active trigger** (timing); **drop the quality
gate** (filtering). `aqi` (timing + quality) is dominated by `active` (timing
alone). The two AQI objectives *conflict* — see §4.

---

## 1. Campaign A — the active trigger (108 cells)

**Question:** at a *matched* intervention budget, does timing help by the
policy's own uncertainty σ_Q beat the hand-engineered monitor, and dominate the
success-vs-cost frontier? (Fixes the volume confound of the earlier
`active_effect` run.) The `query_target_rate` quantile sets the request rate
explicitly; `active-q10/q30/q55` sweep it. A calibration probe fixed
`cooldown=5` so the rates separate (see A doc §3).

| arm | timing | interv. rate | autonomous SR | ICC |
|---|---|---|---|---|
| `hilserl` | none | 0% | 79.4% | **0.662** |
| `active-q10` | σ_Q | 16% | 70.2% | 0.000 |
| `random-q30` | random | 26% | 76.3% | 0.000 |
| `active-q30` | σ_Q | 28% | 81.3% | 0.421 |
| **`active-q55`** | σ_Q | 42% | **87.4%** | 0.010 |
| `logged` | engineered | 54% | 74.3% | 0.000 |

Per-tier (expert / intermediate / novice): `active-q55` 93 / **95** / 74 vs
`logged` 86 / 64 / 73 — it **lifts** the middle without dropping the top.

**Findings:**
1. **`active` > `logged` (well supported).** `active-q55` dominates the frontier:
   best SR at lower cost, paired +13.2 pp, **p=0.021**. The active arms sit
   above-and-left of the `logged` point.
2. **Timing beats chance — only suggestive.** `active-q30` vs rate-matched
   `random-q30`: +4.9 pp, **p=0.302** (not significant). Consistent with the
   `active_validity` audit (σ_Q AUROC ≈ 0.5 vs ground-truth trouble). Most of the
   win is efficient *budget*, not *timing*.
3. **Beats the no-trigger baseline — on dependence, not SR.** No arm beats plain
   `hilserl` on SR (it is a strong 79.4%), but `hilserl` is operator-**dependent**
   (ICC 0.662, because demos are per-operator); any online help washes that to
   ICC ≈ 0.

## 2. Campaign C — the quality gate at the oracle ceiling (90 cells)

**Question:** does *selecting which corrections to keep* — with a **perfect
simulator-oracle** score (not the known-broken counterfactual) — reduce
operator-dependence on top of σ_Q timing? This isolates the quality *paradigm*
from the quality *score*.

| arm | timing | quality | SR | ICC | expert / inter / **novice** |
|---|---|---|---|---|---|
| `logged` | engineered | — | 79.0% | **0.18** | 94 / 78 / **65** |
| `active-q30` | σ_Q | — | 82.9% | **0.23** | 100 / 91 / **57** |
| `quality-oracle` | engineered | oracle | 73.3% | **0.63** | 97 / 91 / **32** |
| `aqi-oracle` | σ_Q | oracle | 78.8% | **0.93** | 100 / 100 / **36** |
| `aqi-cf` | σ_Q | counterfactual | 73.4% | **0.97** | 100 / 100 / **10** |

**Findings (a decisive NEGATIVE, verified real — collapsed novice cells trained
130+ episodes, not the 2026-08 OOM artifact):**
1. **Quality AMPLIFIES operator-dependence** (ICC 0.18 → 0.63 / 0.93 / 0.97). It
   lifts expert/intermediate to the ceiling (100/100) and **collapses the
   novice** (65 → 9–36%) → the tier gap, and ICC, explode.
2. **Mechanism = the quality gate starves the weak operator.** The oracle (and
   the outcome filter) correctly give q_int=0 to corrections from failed
   episodes. Novices fail most episodes, so their corrections are (correctly)
   discarded → the novice policy loses its learning signal. `logged` uses the
   novice's mediocre corrections and reaches 65%. **Filtering bad help is worse
   than using it.**
3. **It is the paradigm, not the score.** The **oracle** — the best possible
   score — makes dependence worst. So this is not fixable by a better score; it
   deepens the RQ2–RQ4 finding ("score is anti-predictive, AUROC 0.175") to
   "**even a perfect score makes dependence worse**."
4. **Oracle vs broken score:** `aqi-oracle` (novice 36%, ICC 0.93) beats `aqi-cf`
   (10%, 0.97) — the oracle *is* the better score, as hypothesized — but **both
   collapse the novice.** "Oracle beats counterfactual" holds; "quality helps"
   fails even at the ceiling.

## 3. The demo mechanism probe (18 cells)

Is the novice collapse demo-**starvation** (fixable with more data) or quality
doing **active harm**? Rerun the novice tier at **20 demos** (vs C's 6), same
roster, paired per operator×seed.

| arm | novice @ 6 demos | novice @ 20 demos | Δ |
|---|---|---|---|
| `logged` | 65.4% | 60.5% | −4.9 |
| `active-q30` | 57.4% | 64.2% | +6.8 |
| `aqi-oracle` | 36.4% | 27.2% | **−9.3** |

**More demos do not rescue the quality-collapsed novice** — it stays ~27–36%,
the quality gap persists (29 → 33 pp). So C's result is **robust to demo
quantity**, not a starvation artifact; the interpretation tilts toward quality
being **actively harmful** to weak operators. (Caveat: this adds the novice's own
mediocre demos — 20 barely moved `logged` either — so it refutes "fixable by more
of the operator's own demos," not "by any high-quality signal.")

## 4. Cross-campaign synthesis — why timing helps and quality hurts

The two AQI objectives pull in **opposite directions** on operator-dependence:

| | **Active trigger (timing)** | **Quality gate (filtering)** |
|---|---|---|
| what it does | asks for help when the policy is uncertain | keeps good corrections, drops bad ones |
| effect on the weak operator | gets the *same* kind of (modest) help, just less of it | its help is *filtered out* → policy starves |
| effect on dependence | **reduces** it (everyone converges to a similar band; A: ICC→0) | **amplifies** it (good operators → ceiling, weak → collapse; C: ICC→0.93) |
| at the ceiling | σ_Q timing ≈ chance (A: p=0.30) — the *rate* matters, not the picking | oracle is worst (C) — perfect picking maximally amplifies skill |

The deep reason: **operator-independence requires helping the weak operator
MORE, while quality filtering does the opposite** — it trusts the strong and
abandons the weak. A *perfect* quality score is therefore the *most* skill-
amplifying, which is exactly backwards for the dependence goal. Active timing,
by contrast, is dependence-reducing precisely because it is *indiscriminate*
about correction quality — it just modulates *how often* help arrives.

This also reconciles A's weak timing result: if σ_Q's per-moment picking carried
real value, you would expect quality (per-moment value picking) to help — but it
is catastrophic. Both point to the same thing: **per-moment value signals from
this critic are not the useful lever; help *volume/rate* is.**

## 5. Thesis-level conclusions

1. **Keep the active trigger, drop the quality gate.** `active` (timing) is the
   useful AQI component; `aqi` (timing + quality) is strictly dominated.
   `active-q55` is the best single configuration (87% SR, ICC 0.01, 42% cost).
2. **The operator-independence win is real but comes from efficient help volume,
   not smart timing or smart correction-selection.** State this honestly.
3. **A perfect quality gate is counterproductive for operator-independence** — a
   novel, mechanistically-explained negative that subsumes the RQ2–RQ4 quality
   failures (it is the paradigm, not the score).
4. **Caveats across the arc:** n = 9 operators × 2 seeds (ICC noisy, but effects
   are large, mechanisms explicit in per-cell data, and consistent across oracle
   AND counterfactual); `active-q10` unstable at very low help; `aqi-oracle`
   converged 0/18 (best-window eval of non-converged policies); `logged`'s own
   ICC is run-noisy (0.00 in A at 35k, 0.18 in C at 40k) — both low.

## 6. Recommendations / how to proceed

- **Write-up ready:** A + C + the demo probe form a complete, publishable arc
  (timing helps; quality hurts even at the oracle ceiling; robust to demos).
- **Optional — campaign D (`active_mild_novice`, `tier_s_exec 0.25`):** does a
  weaker-but-*informative* novice (corrections good enough that the oracle keeps
  them) avoid the collapse? Would **bound** the active harm to operators weak
  enough that the gate filters their help. ~2 days, config ready.
- **Optional — +2 seeds** on the key arms to firm up A's timing/baseline p-values
  and C's ICC estimates.
- **If salvaging quality:** redesign the objective to be *dependence-aware* (help
  the weak operator MORE when its help is poor) — a different paradigm, not a
  tuning fix.
- **Not worth running as headline:** campaign B (`active_trigger_loudness`,
  σ_Q-ranking at matched rate) — A already shows per-moment σ_Q timing ≈ chance,
  so a louder ranking is unlikely to help; keep it only as a ranking control.

## 7. Artifact + document index

| campaign | run dir | detailed doc |
|---|---|---|
| A — active frontier | `runs/active_ratematch/` | [active_ratematch_results_and_qa_log.md](active_ratematch_results_and_qa_log.md) |
| A calibration probe | `runs/active_ratematch_probe/` | A doc §3 |
| C — oracle quality | `runs/aqi_oracle_quality/` | [aqi_oracle_quality_results_and_qa_log.md](aqi_oracle_quality_results_and_qa_log.md) |
| C oracle smoke test | `runs/aqi_oracle_smoke/` | C doc §0, §7 |
| demo mechanism probe | `runs/quality_starvation_probe/` | C doc §8 |

Each run dir has `report/dependency.md` (+ `_baseline_*` variants for A),
`report/plots/`, `eval_runs/<cell>/cell.json` (per-cell SR), and
`aqi_logs/<cell>_actor/{steps,segments,episodes}.jsonl` (per-step/segment data).
