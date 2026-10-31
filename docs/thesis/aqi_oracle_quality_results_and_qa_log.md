# Oracle-quality ceiling (campaign C) — results & QA log

Companion to [active_ratematch_results_and_qa_log.md](active_ratematch_results_and_qa_log.md)
(campaign A) and the RQ2–RQ4 quality logs. This is the **quality half** of AQI:
does *selecting which corrections to keep* — with a **perfect, simulator-ground-
truth oracle** score — reduce operator-dependence on top of *when to ask*?

- Campaign: `runs/aqi_oracle_quality` (90 cells, finished 2026-10-29, run in two
  parts across travel; no crashes bar 1 cell).
- Config: [`examples/campaigns/aqi_oracle_quality.yaml`](../../examples/campaigns/aqi_oracle_quality.yaml).
- Report: `runs/aqi_oracle_quality/report/dependency.md`.

---

## 0. Headline (a decisive NEGATIVE, cleanly explained)

**A quality gate — even a PERFECT oracle one — does NOT reduce operator-dependence;
it dramatically AMPLIFIES it.** ICC rises from `logged` 0.18 to **0.63 (quality-
oracle), 0.93 (aqi-oracle), 0.97 (aqi-cf)**. The mechanism is unambiguous:
quality filtering keeps good operators' corrections (expert/intermediate → 100%)
and **correctly discards bad novice corrections, which STARVES the novice-trained
policy** → the novice tier collapses (65% → 9–36%) → dependence explodes.

This is not a broken-score problem: the **oracle proves it is intrinsic to the
quality-gate paradigm.** It deepens RQ2–RQ4 ("the score is anti-predictive,
AUROC 0.175") to "**even a perfect score makes dependence worse**."

Verified real, not the 2026-08 OOM artifact: the collapsed novice cells each
trained **130+ episodes to ~22–23k steps** and genuinely eval at 0–7%.

## 1. Results

| arm | timing | quality | mean SR | ICC | intervention rate | converged |
|---|---|---|---|---|---|---|
| `logged` | engineered | none | 79.0% | **0.183** | 54.6% | 15/18 |
| `active-q30` | σ_Q | none | 82.9% | **0.234** | 28.4% | 9/18 |
| `quality-oracle` | engineered | oracle | 73.3% | **0.625** | 37.9% | 8/18 |
| `aqi-oracle` | σ_Q | oracle | 78.8% | **0.926** | 27.7% | 0/18 |
| `aqi-cf` | σ_Q | counterfactual | 73.4% | **0.965** | 29.1% | 2/18 |

### Skill gradient (mean SR per tier) — the mechanism in one table

| arm | expert | intermediate | novice |
|---|---|---|---|
| `logged` | 93.8% | 77.8% | **65.4%** |
| `active-q30` | 100.0% | 91.4% | 57.4% |
| `quality-oracle` | 96.9% | 91.4% | **31.5%** |
| `aqi-oracle` | 100.0% | 100.0% | **36.4%** |
| `aqi-cf` | 100.0% | 100.0% | **9.6%** |

Quality **lifts expert/intermediate to the ceiling** (100/100 for the aqi arms)
while **collapsing the novice**. The gap between tiers widens → ICC explodes.

### Paired per-cell vs `logged`

| arm | W/L/T | mean d | sign-p |
|---|---|---|---|
| `active-q30` | 10/4/4 | +3.9 | 0.180 |
| `quality-oracle` | 9/6/3 | −5.8 | 0.607 |
| `aqi-oracle` | 10/4/4 | −0.2 | 0.180 |
| `aqi-cf` | 8/5/4 | −6.8 | 0.581 |

No quality arm beats `logged` on SR; all massively raise ICC.

## 2. The mechanism: quality filtering starves the weak operator

In quality/aqi mode the scorer assigns each takeover a replay priority (q_int)
and imitation weight. The oracle (and the `outcome_filter`) correctly give
**q_int = 0 / weight 0 to takeovers from failed episodes**. For a **novice**,
almost every episode fails early in training, so **its corrections are (correctly)
judged bad and discarded** → the novice policy gets no usable intervention signal,
and its per-operator demos are mediocre too → it collapses (eval SR 0–36%).
Under `logged` (no filtering) the novice's mediocre corrections are **all used**,
so the policy still reaches 65%. **Filtering the bad help is worse than using it.**

The two objectives **conflict**:
- *Quality filtering* optimizes "learn from the best corrections" → amplifies
  skill-dependence (good operators → ceiling, weak operators → starved).
- *Operator-independence* needs the opposite: help the weak operator **more**, not
  discard its help.

## 3. The three designed contrasts

1. **`quality-oracle` vs `logged`** (oracle quality at engineered timing): novice
   65→31%, ICC 0.18→0.63, SR −5.8. **Quality hurts**, even at matched engineered
   timing.
2. **`aqi-oracle` vs `active-q30`** (adding oracle quality to σ_Q timing): ICC
   0.23→**0.93** (4×), novice 57→36%. **Quality DESTROYS active's dependence
   gain** from campaign A.
3. **`aqi-oracle` vs `aqi-cf`** (perfect vs broken score): oracle novice 36% / ICC
   0.93 vs cf 9.6% / 0.97. **The oracle IS the better score** (as hypothesized —
   less destructive), but **both collapse the novice.** So "oracle beats
   counterfactual" holds, yet "quality helps the dependence goal" fails even with
   the oracle.

## 4. What it means for the thesis

- **The quality component of AQI is counterproductive for operator-independence**,
  and this is now proven at the ceiling (a perfect score). It is a *paradigm*
  problem, not an implementation one.
- **Combined with campaign A:** σ_Q *timing* (A) is the useful AQI lever —
  efficient, lifts, reduces dependence modestly; *quality filtering* (C) is
  counterproductive. **Recommendation: use `active` (timing only); `aqi`
  (timing + quality) is dominated.** `active-q55` (A) remains the best single arm
  overall (87% SR, ICC 0.01).
- This is a strong, novel **negative result with a clear mechanism** — publishable
  as the quality-side contribution.

## 5. Caveats

- n = 9 operators × 2 seeds → ICC point estimates are noisy, but the effect is
  huge (0.18 → 0.93), the mechanism is explicit in the per-cell data, and it is
  **consistent across both the oracle AND the counterfactual** score.
- `aqi-oracle` converged 0/18 (best-window eval of non-converged policies); the SR
  is still the frozen-ruler autonomous SR.
- 1 `aqi-cf` cell (`nov02_s1`) failed to eval (17/18 pairs); does not change the
  conclusion.
- Novice collapse is compounded by per-operator demos also being weak.

## 6. How to proceed

1. **Frame the negative (recommended).** A + C together form a complete story:
   timing helps, quality filtering hurts (even at the oracle ceiling). The thesis
   quality chapter can be written now.
2. **Mechanism probes (cheap, high-value follow-ups) — D and E are now the
   interesting ones:**
   - **E `active_demos_sweep`** (more demos): if the collapse is *starvation*, more
     novice demos should rescue the novice under quality. Direct test of the
     mechanism.
   - **D `active_mild_novice`** (weaker-but-informative novice, `tier_s_exec 0.25`):
     a less-catastrophic novice gives *usable* corrections the oracle keeps → the
     collapse should shrink. Bounds the mechanism to very-weak operators.
3. **If salvaging quality:** redesign the objective to be *dependence-aware* — help
   the weak operator MORE when its help is poor (not a keep/drop filter). This is a
   different paradigm and a research pivot, not a tuning fix.
4. **Optional robustness:** +2 seeds on the key arms; re-eval the 1 missing cell.
   Not required — the effect is decisive.

## 7. Artifacts

| what | path |
|---|---|
| report | `runs/aqi_oracle_quality/report/dependency.md` + `plots/` |
| per-cell eval SR | `runs/aqi_oracle_quality/eval_runs/<cell>/cell.json` (`best_window_sr`) |
| per-segment oracle scores | `runs/aqi_oracle_quality/aqi_logs/<cell>_actor/segments.jsonl` (`q_int_mean`) |
| oracle smoke test (de-risk) | `runs/aqi_oracle_smoke/` |
| config | `examples/campaigns/aqi_oracle_quality.yaml` |

Oracle signal at scale (confirmed non-degenerate): `aqi-oracle` 66.7% of segments
q_int>0, `quality-oracle` 45.2% — tracking episode success, exactly as designed.

## 8. Follow-up: the demo mechanism probe (2026-10-31)

Tests whether the novice collapse is demo-STARVATION (fixable with more data) or
quality doing ACTIVE HARM. `runs/quality_starvation_probe` reruns the novice tier
at **20 demos** (vs C's 6) for `logged` / `active-q30` / `aqi-oracle`, using C's
exact novice roster (frozen θ copied in) so each cell pairs with C's 6-demo cell.
18 cells, no crashes. Config: `examples/campaigns/quality_starvation_probe.yaml`.

| arm | novice @ 6 demos (C) | novice @ 20 demos (probe) | Δ |
|---|---|---|---|
| `logged` | 65.4% | 60.5% | −4.9 |
| `active-q30` | 57.4% | 64.2% | +6.8 |
| `aqi-oracle` | 36.4% | 27.2% | **−9.3** |

**More demos do NOT rescue the quality-collapsed novice** — it stays ~27–36% while
`logged` stays ~60–65%; the quality gap persists (29 → 33 pp). So C's "quality
amplifies dependence" is **not a demo-quantity artifact** — it is robust. The
interpretation tilts from "starvation" toward **quality is actively harmful to the
weak operator** (tripling the demo buffer, which bypasses the online quality
filter, doesn't help).

Caveat: this adds more of the novice's OWN mediocre demos — 20 demos barely moved
`logged` either (65→60) — so it refutes "fixable by more of the operator's own
demos," not "fixable by any high-quality signal" (that would need expert demos on
novice cells, which breaks the per-operator design).

NEXT: campaign D (`active_mild_novice`, `tier_s_exec 0.25`) — does a
weaker-but-INFORMATIVE novice (corrections good enough that the oracle KEEPS them)
avoid the collapse? That bounds the active harm to operators weak enough that the
gate filters out their help.
