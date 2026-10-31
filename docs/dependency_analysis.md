# Operator-dependency: the variance decomposition in full

The study's central question is a **variance-decomposition** over training
outcomes: *how much of the variation in a trained policy's success rate is
explained by **which operator** trained it, versus random run-to-run noise* — and
does AQI **reduce** that operator-explained share versus plain HIL-SERL?

This document is the mathematical reference for that decomposition: every
estimator, its exact formula as implemented, and a fully worked decode of a real
report. It complements the conceptual treatment in
[methods.md](methods.md#3-the-dependency-assessment) §3 and the protocol
metric→code map in [experimental_protocol.md](experimental_protocol.md).

Where it lives in code:

| concern | file |
|---|---|
| estimators (ICC, spread, permutation, bootstrap) | [`serl_launcher/serl_launcher/aqi/stats.py`](../serl_launcher/serl_launcher/aqi/stats.py) |
| campaign-level report construction | [`examples/run_campaign.py`](../examples/run_campaign.py) |
| your numbers | `report/dependency.json`, `report/dependency.md` |

---

## 1. The statistical unit and hierarchy

- One **cell** = one full training run, indexed by `(method, operator, seed)`.
- Each cell is scored by its **final autonomous success rate (SR)** from the fixed
  evaluation ruler (frozen policy, no operator — see
  [eval_ruler.md](eval_ruler.md)).
- For each **method**, operators form **groups**; each group has **seed
  replicates**.

So the hierarchy is two-level:

$$
\underbrace{y_{o,s}}_{\text{seed replicate}}
\;\in\;
\underbrace{\text{operator } o}_{\text{group}}
\;\in\;
\underbrace{\text{method } m}_{\text{analysed separately}}
$$

with, for a fixed method $m$:

- $y_{o,s}$ = SR for operator $o$, seed $s$;
- $\bar y_o = \frac{1}{n_o}\sum_s y_{o,s}$ = operator mean;
- $\bar y$ = global method mean.

**Dependency** is fundamentally: *how much variance is attributable to changing
operator identity, versus random replicate noise.*

---

## 2. The core within-method decomposition

Computed **per method** ([`icc_oneway`](../serl_launcher/serl_launcher/aqi/stats.py)).
A one-way random-effects ANOVA splits the total SR variance into a
**between-operator** and a **within-operator** component.

### 2.1 Sums of squares and mean squares

$$
SS_B=\sum_{o} n_o(\bar y_o-\bar y)^2,
\qquad
SS_W=\sum_{o}\sum_{s}(y_{o,s}-\bar y_o)^2,
$$

$$
MS_B=\frac{SS_B}{K-1},
\qquad
MS_W=\frac{SS_W}{N-K},
$$

where $K$ = number of operators (groups) in the method and $N=\sum_o n_o$ =
total cells.

### 2.2 Variance components (Searle's correction)

For possibly-unbalanced groups the effective group size is Searle's $k_0$:

$$
k_0=\frac{N-\dfrac{\sum_o n_o^2}{N}}{K-1}
$$

(if every group has the same size $n$, then $k_0=n$). The variance components are

$$
\sigma^2_{\text{between}}=\max\!\left(\frac{MS_B-MS_W}{k_0},\,0\right),
\qquad
\sigma^2_{\text{within}}=MS_W .
$$

The truncation at $0$ handles the case where sampling noise makes
$MS_B<MS_W$ (a negative variance estimate is set to zero).

### 2.3 The intraclass correlation ICC(1,1)

$$
\boxed{\;\mathrm{ICC}=\frac{\sigma^2_{\text{between}}}
                           {\sigma^2_{\text{between}}+\sigma^2_{\text{within}}}\;}
$$

Interpretation:

- **ICC → 1** — almost all variation is *between operators*: the outcome is
  **highly operator-dependent** (who trained it dominates).
- **ICC → 0** — the operator barely matters; runs differ mostly by seed noise:
  **operator-independent**.

The implementation returns `icc`, `ms_between`, `ms_within`, `k0`, `sigma2_op`
($=\sigma^2_{\text{between}}$) and `sigma2_seed` ($=\sigma^2_{\text{within}}=MS_W$);
if fewer than 2 operators or no replication, all fields are `nan`.

---

## 3. Reinforcing within-method diagnostics

These make the story robust at small $K$, and are computed from the per-operator
means $\{\bar y_o\}$ ([`operator_spread`](../serl_launcher/serl_launcher/aqi/stats.py)).

$$
\sigma_{op}=\operatorname{sd}\big(\{\bar y_o\}\big),\qquad
CV=\frac{\sigma_{op}}{\bar y_{op}},\qquad
\text{range}=\max_o\bar y_o-\min_o\bar y_o,
$$

plus the **worst operator** $\arg\min_o\bar y_o$ and its SR. $CV$ is the unit-free
"how far apart do operators land" number; the worst-operator SR is the
worst-case-robustness read.

### 3.1 The "operator matters" permutation test

([`permutation_sigma2`](../serl_launcher/serl_launcher/aqi/stats.py)) — a
label-exchangeability test that needs no distributional assumptions:

1. **Null:** operator labels are irrelevant (cells are exchangeable across
   operators).
2. Shuffle the operator labels across cells $B$ times (default $B=10{,}000$),
   recomputing the between-operator variance of the group means each time.
3. Observed statistic $\sigma^2_{\text{obs}}$ from the true labels.
4. p-value with the $+1$ correction (never zero):

$$
p=\frac{\#\{\sigma^2_{\text{perm}}\ge\sigma^2_{\text{obs}}\}+1}{B+1}.
$$

**Low $p$** ⇒ the observed operator spread is unlikely under label-exchange, so
**operator identity matters**. **High $p$** ⇒ the spread is consistent with noise.

### 3.2 Hierarchical bootstrap CI

([`bootstrap_icc`](../serl_launcher/serl_launcher/aqi/stats.py)) — a 95 %
confidence interval for ICC and $\sigma_{op}$ by **two-level** resampling:

1. resample operators **with replacement**;
2. within each sampled operator, resample its seeds **with replacement**;
3. recompute ICC; take the 2.5 / 97.5 percentiles.

At small $K$ (4–6 operators) this CI is **wide** — the lower tail is dominated by
low-diversity resamples (few distinct operators drawn). That is honest
uncertainty about small $K$, and it is why the pre-registered rule reads the ICC
**point estimate together with the permutation $p$**, rather than trusting the CI
lower bound alone.

---

## 4. Between-method comparison — the thesis-claim axis

After each method has its own within-method profile, methods are compared against
the baseline $b$ (default `hilserl`). For method $m$:

$$
\Delta\mathrm{ICC}_{\text{vs base}}=\mathrm{ICC}_b-\mathrm{ICC}_m,
\qquad
\Delta\sigma^2_{\text{between, vs base}}=\sigma^2_{\text{between},b}-\sigma^2_{\text{between},m}.
$$

**Sign convention:** a **positive** delta means a **reduction** vs baseline (good
for the robustness claim); negative means an increase. The primary
pre-registered endpoint in [experiment_pipeline.py](../examples/experiment_pipeline.py)
`evaluate` is the closely-related **robustness gain**
$RG=\operatorname{var}_{op}(\text{baseline})-\operatorname{var}_{op}(\text{treat})$
and its relative form $RG_{\text{rel}}=1-\operatorname{var}_{op}(\text{treat})/\operatorname{var}_{op}(\text{baseline})$.

The target shape of the result:

| method | mean SR | ICC | $\sigma^2_{\text{between}}$ | $CV$ | $p$(operator matters) |
|---|---|---|---|---|---|
| `hilserl` (baseline) | 55 % | **0.99** | 0.12 | **0.45** | **0.000** |
| `aqi` | 80 % | **0.16** | 0.000 | **0.02** | 0.035 |

Baseline: expert-trained win, novice-trained lose (strong dependency). AQI:
everyone lands near the same SR (dependency largely removed).

---

## 5. The skill-gradient sanity block

Per method, tier means are simple averages over all cells in each tier:

$$
\mu_t=\frac{1}{N_t}\sum_{o\in t,\,s} y_{o,s},\qquad t\in\{\text{expert, intermediate, novice}\}.
$$

Purpose: check the **ordinal** effect (expert should usually exceed novice unless
the method equalizes strongly or the task saturates). It is a guard against
drawing dependency conclusions from a ceiling/floor-confounded run (see §8).

---

## 6. The second axis — steps-to-convergence

Dependency is not only about the *final* SR; it is also about the **cost** of
getting there. The same decomposition is applied with the outcome = **steps to
convergence**:

- a cell records its convergence step if it hit the sustained autonomous-success
  criterion (see [methods.md §3.6](methods.md)
  and [campaign_runner.md](campaign_runner.md));
- if it never converged by the cap, the value is **null** and is excluded from
  the variance decomposition.

Per method the report gives the converged fraction, mean/median steps among
converged cells, and the **ICC of steps-to-convergence across operators** (does
training *cost* depend on the operator? does AQI flatten it?). A method can reduce
final-SR dependency, effort dependency, or both.

> **Edge case:** if *no* cell converged, all step-based statistics are undefined
> (`nan`), because there is no numeric sample to decompose.

---

## 7. Worked decode of a real report

From a real `report/dependency.json` (cylinder, physics-oracle, `hilserl` vs
`quality`, 9 operators × 2 seeds = 18 cells per method, balanced $n_o=2$).

### 7.1 `hilserl` — degenerate ceiling

All 18 SR values are `1.0`. Therefore $MS_B=MS_W=0$,
$\sigma^2_{\text{between}}=\sigma^2_{\text{within}}=0$, ICC reported as `0` by the
$\varepsilon$-stabilized convention (both numerator and denominator zero), all
spread metrics `0`, and permutation $p=1.0$ (no observed spread at all).

> **This is complete ceiling saturation.** Dependency is *not measurable* from SR
> here because there is no variance to attribute — not evidence of independence.

### 7.2 `quality` — strong operator signal

Operator means: experts $\{1,1,1\}$, intermediate $\{0.5,1,1\}$, novices
$\{0,0,0\}$. Only `int00` has seed disagreement ($0$ and $1$), creating the
within component; the tier separation creates the strong between component:

$$
\sigma^2_{\text{between}}=0.20833,\quad
\sigma^2_{\text{within}}=0.05556,\quad
\mathrm{ICC}=0.78947,
$$
$$
\sigma_{op}=0.4859,\quad CV=0.7951,\quad \text{range}=1.0,\quad p=0.0334.
$$

Steep tier gradient (expert 100 %, intermediate 83.3 %, novice 0 %) and low $p$:
a **strong, real operator-dependency signal within `quality`**.

### 7.3 The method comparison here is structurally misleading

$$
\Delta\mathrm{ICC}_{\text{quality vs hilserl}}=0-0.78947=-0.789.
$$

Because the baseline is **saturated at ICC 0**, this delta is negative — it
*looks like* "quality increases operator-dependence," the opposite of the thesis.
It is a special-case artifact: a baseline with degenerate zero variance makes
"reduction vs baseline" **structurally impossible** on SR-ICC. (In this specific
campaign the 0 % novice `quality` cells were also a GPU-OOM artifact in the
counterfactual batch, since fixed — see
[aqi_analysis.md](aqi_analysis.md) and the runtime notes — a second reason not to
read the headline at face value.)

---

## 7.4 Reconciling the metrics — a second worked example

The metrics in a report answer **different questions**, and they can appear to
disagree. From the `active_effect` 4-arm run (cuboid, 9 operators × 2 seeds,
`logged` vs `active` vs `random` vs `active-loud`):

| method | mean SR | ICC | $\sigma^2_{\text{between}}$ | $\sigma^2_{\text{within}}$ | $CV$ | $p$(op matters) |
|---|---|---|---|---|---|---|
| `logged` | 78.2 % | **0.841** | 0.072 | 0.014 | 0.360 | **0.018** |
| `active` | 70.8 % | **0.000** | 0.000 | **0.128** | 0.212 | 0.922 |
| `random` | 56.2 % | 0.510 | 0.064 | 0.062 | 0.549 | 0.062 |
| `active-loud` | 63.6 % | **0.000** | 0.000 | **0.126** | 0.253 | 0.885 |

**(a) ICC and the permutation $p$ agree — they are two views of one quantity.**
ICC is the *variance ratio* $\sigma^2_{\text{between}}/(\sigma^2_{\text{between}}+\sigma^2_{\text{within}})$;
the permutation $p$ is the *significance* that $\sigma^2_{\text{between}}$ exceeds
label-shuffled chance. High ICC ⇔ low $p$. Here `logged` (0.841 / 0.018) is
dependent, `active`/`active-loud` (0.000 / 0.92, 0.88) are independent, `random`
(0.510 / 0.062) is borderline — consistent on both. Use $p$ as the significance
guard on the ICC **point** estimate; expect them to move together.

**(b) ICC and $CV$ can disagree — because only ICC removes seed noise.**
`active` has ICC $=0.000$ but $CV=0.212$. $CV$ is the raw spread of the
per-operator means $\{\bar y_o\}$; ICC *decomposes* it. When $\sigma^2_{\text{within}}$
is large, the ANOVA gives $MS_B<MS_W$, so

$$
\sigma^2_{\text{between}}=\max\!\Big(\tfrac{MS_B-MS_W}{k_0},0\Big)=0
\;\Rightarrow\;\mathrm{ICC}=0,
$$

**even though $CV>0$.** Reading: the operators *do* land at different mean SRs
($CV\approx0.21$), but that spread is **statistically indistinguishable from
run-to-run noise**. This is *noise-dominated independence / instability*, which is
**not** the same as clean uniformity — a saturated run drives *both* ICC **and**
$CV$ to 0 (§7.1), whereas instability drives ICC→0 while $CV$ stays positive.

**(c) Dependency (ICC) and performance (SR) are orthogonal axes.** The lowest-ICC
methods (`active`/`active-loud`, ICC 0) are **not** the best (`logged`, SR 78 %).
ICC $=0$ was reached by (i) **flattening** the skill gradient toward the middle
(expert $93\%\!\to\!76\%$, novice $52\%\!\to\!63\%$) and (ii) **seed instability**
(§7.5) — never by lifting everyone to a uniform high. *Operator-independent* can
mean *uniformly mediocre* or *uniformly noisy*. Always read ICC **with** mean SR
and the tier gradient; "most operator-independent" alone is not "best".

## 7.5 The $\sigma^2_{\text{within}}$ "surprising values"

$\sigma^2_{\text{within}}=MS_W$ is the mean squared **seed** deviation around each
operator mean. Two things make it look strange, for different reasons:

- **Final-SR table — a large within that *exceeds* between** (`active`
  $\sigma^2_{\text{within}}=0.128$ vs $\sigma^2_{\text{between}}=0.000$). This is
  **training instability**: the *same* operator on two seeds can land far apart —
  e.g. `exp00` scored $100\%$ on one seed and $0\%$ on the other (a full $1.0$-SR
  swing), which is why `active`'s *worst* operator is an **expert**. A handful of
  $\sim\!0.5$–$1.0$ within-operator swings pushes $MS_W$ to $\sim0.12$. It is the
  **mechanical cause** of ICC$\to$0 (seed noise swamps the operator signal), and a
  genuine finding about the method, not a defect.
- **Steps-to-convergence table — values in the millions** (e.g.
  $15{,}127{,}748$). This is purely **units**: the outcome is env-steps,
  $O(3\times10^4)$, so its variance is $O((3\times10^4)^2)=O(10^9)$. Interpret via
  $\sqrt{\sigma^2_{\text{within}}}\approx3{,}000$–$6{,}000$ steps, or via the
  unit-free steps-ICC. Only **converged** cells enter this table, so sparse
  samples (a tier with few converged cells, or `n/a`) make these estimates noisy.

> **Takeaway.** A big $\sigma^2_{\text{within}}$ on SR is a red flag that the
> method is **seed-unstable**, and that its low ICC is "noise $\approx$ operator",
> not robustness. A big $\sigma^2_{\text{within}}$ on steps is just the squared
> units of a large-magnitude outcome.

---

## 8. Why this happens — the ceiling/floor caveat

Dependency conclusions on SR require the **baseline** to have **non-trivial
variance and a real tier gradient**:

- **Saturated baseline** (all ones): no between-operator spread is observable on
  SR.
- **Floor baseline** (all zeros): equally uninformative.

In either case, **retune** (more difficult task — hex/cuboid, tighter clearance;
fewer/weaker demos; smaller online budget) to restore a non-trivial gradient in
the baseline *before* drawing robustness conclusions from SR dependency. When SR
saturates, fall back on the **steps-to-convergence** axis (§6), which still varies.

---

## 9. Practical checklist for reading any dependency report

**Within each method**

1. Check **mean SR and spread first** — are you near ceiling/floor?
2. Read $\sigma^2_{\text{between}}$, $\sigma^2_{\text{within}}$ and ICC **together**.
   A large $\sigma^2_{\text{within}}$ (bigger than $\sigma^2_{\text{between}}$) is a
   **seed-instability** flag: the low ICC is "noise $\approx$ operator", not
   robustness (§7.4b, §7.5).
3. Confirm with $\sigma_{op}$, $CV$, range, worst-operator. If ICC$=0$ but
   $CV>0$, the per-operator spread is being attributed to noise, **not** removed —
   that is instability, not clean uniformity (§7.4b).
4. Use the permutation $p$ as the "operator matters" evidence; it should **agree**
   with ICC (high ICC ⇔ low $p$) (§7.4a).
5. **Dependency $\neq$ performance.** The lowest-ICC method is not necessarily the
   best — ICC$=0$ can mean uniformly mediocre or uniformly noisy. Always pair ICC
   with mean SR and the tier gradient (§7.4c).

**Between methods**

6. Compare $\Delta\mathrm{ICC}$ and $\Delta\sigma^2_{\text{between}}$ vs baseline.
7. Trust conclusions only when the **baseline has non-trivial variance and a tier
   gradient**.
8. Use **steps-to-convergence** dependency as a second axis when final SR
   saturates.

**Reliability**

9. Inspect the ICC bootstrap CI width (especially at small $K$) and keep the
   permutation $p$ in view.

---

## 10. Equation → report-column legend

| `dependency.json` field | symbol / formula |
|---|---|
| `mean` | $\bar y$ |
| `sigma2_between` / `sigma2_op` | $\sigma^2_{\text{between}}=\max((MS_B-MS_W)/k_0,0)$ |
| `sigma2_within` / `sigma2_seed` | $\sigma^2_{\text{within}}=MS_W$ |
| `icc` | $\sigma^2_{\text{between}}/(\sigma^2_{\text{between}}+\sigma^2_{\text{within}})$ |
| `k0` | $\big(N-\sum_o n_o^2/N\big)/(K-1)$ |
| `sigma_op`, `cv`, `range`, `worst` | operator-mean spread (§3) |
| `p_value` / operator-matters | $(\#\{\sigma^2_{\text{perm}}\ge\sigma^2_{\text{obs}}\}+1)/(B+1)$ |
| `icc_ci95` | hierarchical bootstrap percentiles (§3.2) |
| `delta_icc_vs_base` | $\mathrm{ICC}_{\text{base}}-\mathrm{ICC}_m$ |
| tier means | $\mu_t$ (§5) |

### See also

- [methods.md](methods.md#3-the-dependency-assessment) — the conceptual version.
- [experimental_protocol.md](experimental_protocol.md) — the pre-registered B.1–B.4
  suite and the full metric→code map.
- [campaign_runner.md](campaign_runner.md) — how the cells are produced and how the
  report is assembled.
