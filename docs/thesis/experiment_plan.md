# Thesis experiment plan — answering RQ1–RQ3 with this repo

Maps every research question to the concrete elements in this repository — human
operators, synthetic operators, tasks, and the AQI mode-ladder + ablations — with
the exact endpoints and decision rules. The **operational runbook** (exact
commands, where results are saved, how to analyse and plot) is the companion file
[experiment_protocols.md](experiment_protocols.md).

Background: [methods.md](../methods.md) · [dependency_analysis.md](../dependency_analysis.md)
· [critic_ensemble.md](../critic_ensemble.md) · [oracle_and_synthetic_operator.md](../oracle_and_synthetic_operator.md)
· [experimental_protocol.md](../experimental_protocol.md)

---

## 0. The questions and the claim

| RQ | Question | Primary endpoint |
|---|---|---|
| **RQ1 — operator dependency** | Does the method reduce how much the trained policy depends on *which* operator trained it? | ICC(1,1) of autonomous SR across operators; $RG_{\text{rel}}=1-\mathrm{Var_{op}(AQI)}/\mathrm{Var_{op}(vanilla)}$ |
| **RQ2 — intervention quality (usage)** | What makes an intervention useful, and does the counterfactual advantage $\hat A$ (Option B) beat hand-weighted correlates (Option A)? | AUROC of the usefulness fit; $\mathrm{corr}(q_{\text{int}},\text{outcome})$; interaction decomposition |
| **RQ3 — trigger calibration** | Does $A(s_t)$ / $\sigma_Q$ / $P_{\text{fail}}$ predict imminent failure well enough to time requests? | AUROC + ECE vs "failure within $H$ steps" |

**Hypothesis.** AQI makes the policy *operator-robust* (RQ1 ↓ dependence) **because**
its trigger fires on genuine risk (RQ3) and its usage rewards genuinely good
corrections (RQ2), **without** sacrificing mean SR (non-inferiority guardrail).

---

## 1. Element inventory

| element | role in the study |
|---|---|
| **Synthetic operators** (graded θ tiers + jittered *populations* + `op4` skilled-but-jumpy) | statistical power: many operators × seeds, ground-truth skill labels; drive demos (Role 1) + interventions (Role 2) |
| **Human operators** (O1–O4, skill-typed) | external validity on real people |
| **Tasks** (difficulty ladder) | toy 2-D (validation) → **cylinder** (geometric oracle) → **hex/cuboid** (policy oracle) → **RAM/real** (human) |
| **Methods = ablation ladder** | `hilserl`, `logged`, `active`, `quality`, `aqi` + finer knobs |
| **Frozen ruler** (`eval_ruler.py`) | operator-independent measuring stick, hash-locked grid |
| **Analyses** | `run_campaign.py analyze`, `experiment_pipeline.py evaluate`, `aqi_analysis.py`, `operator_skill.py` |

---

## 2. Four experiments

### E0 — Validate the synthetic operators *(prerequisite / Milestone-1.2)*
Legitimises synthetic operators as human proxies. Toy env, jittered populations,
`operator_skill.py --mode validate-synthetic`. **Gate:** accuracy / macro-F1 / QWK /
Spearman-ρ / test-retest ICC above the S8/S11 targets. Must pass before E1.

### E1 — Main operator-dependency campaign (synthetic, automated) → RQ1 (+ feeds RQ2/RQ3)
`run_campaign.py` with a study config.

| axis | setting |
|---|---|
| methods | `hilserl`, `logged`, `active`, `quality`, `aqi` |
| operators | populations: 5–8 ops/tier × 3 tiers (+ `op4`) |
| seeds | 3–5 (independent: `--seed=--synthetic_seed=--env_seed=r`) |
| task | **cylinder** primary; **hex** headroom (policy oracle) |
| demos | modest (6 filtered) so *online* interventions matter |
| budget | convergence stop + step/time cap; **frozen ruler** |

Outputs → `report/dependency.{md,json}` + plots: ICC, $\sigma^2_{\text{between/within}}$,
CV, permutation-$p$, **$RG_{\text{rel}}$**, and the effort axis **steps-to-convergence ICC**.

### E2 — Ablation decomposition (synthetic) → RQ2 + RQ3 mechanisms
The **5-mode ladder is the core ablation** (isolates AQI's two mechanisms):

| arm | isolates |
|---|---|
| `hilserl` | plain 2-critic baseline |
| `logged` | ensemble only (same-arch vanilla) |
| `active` | + trigger only (RQ3 contribution) |
| `quality` | + usage only (RQ2 contribution) |
| `aqi` | both → tests synergy $\text{aqi}\overset{?}{\approx}\text{active}\oplus\text{quality}$ |

Finer arms: **RQ3** — `trigger_terms` subsets, `trigger_combiner` sum/max, ensemble-diversity
control (`bootstrap_p=1.0 --per_member_targets=false`). **RQ2** — Option B vs A,
`quality_disagreement_gate` on/off, `quality_filters` on/off, replay-only
(`bc_loss_weight=0`) vs BC-only (`priority_sampling=false`).

### E3 — Human confirmation (interactive) → RQ1 external validity + human RQ2/RQ3
`experiment_pipeline.py init → session → evaluate` (pre-registered). Methods
`{logged, aqi}` (± `active`,`quality`) × O1–O4 × 1–2 seeds × RAM (or sim),
counterbalanced, same **frozen ruler**.

---

## 3. Per-RQ answer (metric → script → decision rule)

### RQ1 — operator dependency
- **Metrics:** `icc_oneway`, `operator_spread` (CV/range/worst-op), `permutation_sigma2` ($p$),
  `bootstrap_icc` (CI), `robustness_gain`/`paired_bootstrap_rg_rel` (**$RG_{\text{rel}}$, primary**),
  steps-to-convergence ICC.
- **Scripts:** E1 `run_campaign.py analyze`; E3 `experiment_pipeline.py evaluate`.
- **Rule:** Exp-1 = ICC point + permutation-$p$; Exp-2 = $RG_{\text{rel}}>0$ (paired-bootstrap CI)
  **and** non-inferiority guardrail $\bar d$ vs $\delta$ on mean SR.
- **Supported iff** vanilla shows a real skill gradient (expert ≫ novice) **and** AQI flattens it
  (ICC ↓, $RG_{\text{rel}}>0$) **without** mean-SR loss.

### RQ2 — intervention quality
- **Metrics:** logistic fit of post-intervention success on Option-A features + **AUROC**;
  **$\mathrm{corr}(q_{\text{int}},\text{outcome})$** (Option B beats A); `interaction_decomposition`.
- **Scripts:** `aqi_analysis.py` (RQ2) on E1/E2/E3 logs; `experiment_pipeline.py evaluate`.
- **Ablation evidence:** `quality` vs `logged`; gate on/off; replay-only vs BC-only vs both.

### RQ3 — trigger calibration
- **Metrics:** **AUROC + ECE** of $P_{\text{fail}}$ and $A(s_t)$ vs "failure within $H$ steps".
- **Scripts:** `aqi_analysis.py` (RQ3); `experiment_pipeline.py evaluate` (D.3).
- **Ablation evidence:** per-term subsets + combiner; **shared-data control** shows $\sigma_Q$'s
  AUROC drops when the ensemble isn't bootstrapped → its power is epistemic.

---

## 4. Run matrix (starting numbers)

| exp | methods | operators | seeds | task(s) | driver |
|---|---|---|---|---|---|
| E0 | — | 10/tier (toy) | — | toy | `operator_skill.py` |
| E1 | 5 | 15 (+`op4`) | 3 | cylinder (+hex) | `run_campaign.py` |
| E2 | ladder + ~6 finer arms | 5/tier | 2–3 | cylinder | `run_campaign.py` |
| E3 | 2–4 | O1–O4 | 1–2 | RAM / sim | `experiment_pipeline.py` |

Validate the loop first (`--methods hilserl,aqi --ops_per_tier 2 --seeds 2`), then scale.
Campaigns are idempotent/resumable.

---

## 5. Controls & threats to validity

1. **Frozen, hash-locked ruler** for every cell.
2. **Demo-strength regime:** keep demos modest; **confirm the vanilla baseline shows a skill
   gradient before trusting any AQI delta** (saturated/floored baseline ⇒ RQ1 uncomputable —
   [dependency_analysis.md](../dependency_analysis.md) §8).
3. **OOM artifact:** run `active`/`quality`/`aqi` actors with the chunked/OOM-guarded
   counterfactual (or CPU actor) so **novice cells don't crash and manufacture a skill-graded ICC**.
4. **Small-K stats:** report ICC point + permutation-$p$, not the CI lower bound.
5. **Contaminated human trials** excluded (`intervened=true`) and counted.
6. **Same-architecture vanilla = `logged`** for the AQI delta; keep `hilserl` as the plain reference.

---

## 6. Thesis deliverables produced

| figure/table | source | RQ |
|---|---|---|
| money plot (paired per-op SR arrows) | `experiment_pipeline evaluate` | RQ1 |
| interaction plot (per-op lines converge under AQI) | `evaluate` / campaign | RQ1 |
| ICC / $RG_{\text{rel}}$ table + decision outcomes | `evaluate` / `analyze` | RQ1 |
| ablation-ladder table + interaction decomposition | E2 | RQ2/RQ3 |
| $q_{\text{int}}$-vs-outcome; Option A vs B | `aqi_analysis.py` | RQ2 |
| calibration curves + per-term AUROC | `aqi_analysis.py` | RQ3 |
| steps-to-convergence dependency | `analyze` | RQ1 (cost) |
| synthetic-operator validation table | E0 | legitimises E1 |

---

**Next:** [experiment_protocols.md](experiment_protocols.md) — the exact commands, output
locations, and analysis/plot steps for E0–E3.
