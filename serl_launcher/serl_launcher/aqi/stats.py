"""Statistics for the AQI-HIL-SERL experimental protocol.

Implements the metric catalogue and statistical machinery of the thesis
document "Experimental Design - Measuring Operator Dependency and
Comparing AQI-HIL-SERL to Vanilla" (section references B.1-B.4, C.2,
D.2 refer to that document). Every quantity here is defined there; the
worked numerical examples in the document double as this module's test
vectors.

Dependencies: numpy always; scipy (repo requirement) for the classical
tests (Brown-Forsythe, Wilcoxon); statsmodels+pandas OPTIONALLY for the
trial-level Binomial GLMM (D.2 "model view") - glmm_success() degrades
to an explanatory note when they are absent, and the exact algebraic
interaction decomposition (B.4 measure 5) is always available.

Conventions:
    * "cells" are (method, operator, seed) units; per-cell values are
      usually success rates SR in [0,1].
    * operator-mean dicts map operator id -> ybar_o (mean over seeds).
    * All bootstrap/permutation routines take an explicit numpy
      Generator so results are reproducible and pre-registerable.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_B = 10_000


# ===========================================================================
# B.1 helpers - Wilson interval for success rates
# ===========================================================================
def wilson_interval(k: int, n: int, z: float = 1.959963985
                    ) -> Tuple[float, float]:
    """95% Wilson score interval for k successes out of n trials.

    Stays inside [0,1], asymmetric near the edges, accurate at small n -
    the pre-registered error bar for every per-cell success rate (B.1).
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def intervals_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def cell_outcomes(trials: List[dict], f_max: Optional[float] = None) -> dict:
    """B.1 outcome metrics for one cell from the fixed-ruler trial
    records (A.2). Each trial dict: success, time_s (successful trials),
    f_peak, f_mean, and optionally violation / intervened.

    Contaminated trials (a human intervened during the frozen eval) are
    excluded from every statistic and counted separately - the ruler is
    defined as autonomous."""
    clean = [t for t in trials if not t.get("intervened", False)]
    n = len(clean)
    k = int(sum(bool(t.get("success", False)) for t in clean))
    times = [float(t["time_s"]) for t in clean
             if t.get("success", False) and t.get("time_s") is not None]
    f_peaks = [float(t.get("f_peak", 0.0)) for t in clean]
    if f_max is not None:
        viol = [float(t.get("f_peak", 0.0)) > f_max for t in clean]
    else:
        viol = [bool(t.get("violation", False)) for t in clean]
    lo, hi = wilson_interval(k, n)
    return dict(
        n_trials=n, k_success=k,
        n_contaminated=len(trials) - n,
        sr=(k / n) if n else float("nan"),
        sr_wilson95=[lo, hi],
        time_to_success_mean=float(np.mean(times)) if times else float("nan"),
        time_to_success_sd=float(np.std(times, ddof=1)) if len(times) > 1
        else float("nan"),
        f_peak_mean=float(np.mean(f_peaks)) if f_peaks else float("nan"),
        f_mean_mean=float(np.mean([t.get("f_mean", 0.0) for t in clean]))
        if clean else float("nan"),
        violation_rate=float(np.mean(viol)) if clean else float("nan"),
        f_max=f_max,
    )


# ===========================================================================
# B.2 - process metrics from per-episode training records
# ===========================================================================
def rolling_success(successes: Sequence[float], window: int) -> np.ndarray:
    """Rolling mean over the trailing `window` episodes (shorter head)."""
    s = np.asarray(successes, dtype=np.float64)
    out = np.empty_like(s)
    for i in range(len(s)):
        out[i] = s[max(0, i - window + 1): i + 1].mean()
    return out


def s80(successes: Sequence[float], lengths: Sequence[int],
        window: int = 20, threshold: float = 0.8) -> float:
    """Sample efficiency S80 (B.2): env-steps until the rolling success
    over the last `window` episodes first reaches `threshold`.
    Returns math.inf if never reached. Only windows with `window`
    episodes count (a 2-episode run cannot certify 80%)."""
    s = np.asarray(successes, dtype=np.float64)
    cum = np.cumsum(np.asarray(lengths, dtype=np.float64))
    roll = rolling_success(s, window)
    for i in range(len(s)):
        if i + 1 >= window and roll[i] >= threshold:
            return float(cum[i])
    return math.inf


def intervention_summary(intervention_steps: Sequence[int],
                         lengths: Sequence[int],
                         n_segments: Optional[Sequence[int]] = None,
                         dt: float = 0.1) -> dict:
    """Intervention count / time (B.2). Two counts are reported:
    episode-level events (episodes with any intervention - the document's
    simple version) and distinct takeover events (sum of segments, when
    segment counts are available)."""
    iv = np.asarray(intervention_steps, dtype=np.float64)
    total_steps = float(np.sum(lengths))
    steps = float(iv.sum())
    return dict(
        events_episode_level=int(np.sum(iv > 0)),
        events_takeovers=(int(np.sum(n_segments))
                          if n_segments is not None else None),
        intervention_steps=int(steps),
        intervention_time_s=steps * dt,
        intervention_step_frac=steps / max(total_steps, 1.0),
    )


def intervention_rate_curve(intervention_steps: Sequence[int],
                            lengths: Sequence[int],
                            bin_steps: int = 1000) -> dict:
    """r(t) (B.2): per time-bin, intervention steps / total steps.
    Episodes are attributed to bins by their cumulative step spans
    (an episode crossing a bin edge is split proportionally)."""
    iv = np.asarray(intervention_steps, dtype=np.float64)
    ln = np.asarray(lengths, dtype=np.float64)
    total = ln.sum()
    n_bins = int(math.ceil(total / bin_steps)) if total > 0 else 0
    num = np.zeros(n_bins)
    den = np.zeros(n_bins)
    t = 0.0
    for i in range(len(ln)):
        remaining, iv_rate = ln[i], (iv[i] / ln[i] if ln[i] > 0 else 0.0)
        pos = t
        while remaining > 0:
            b = min(int(pos // bin_steps), n_bins - 1)
            take = min(remaining, (b + 1) * bin_steps - pos)
            den[b] += take
            num[b] += take * iv_rate
            pos += take
            remaining -= take
        t += ln[i]
    r = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return dict(
        bin_steps=bin_steps,
        bin_ends=[float((b + 1) * bin_steps) for b in range(n_bins)],
        r=[float(x) for x in r],
        final_autonomy=float(1.0 - r[-1]) if n_bins else float("nan"),
    )


def learning_curve_auc(successes: Sequence[float]) -> float:
    """Learning-curve AUC (B.2): the mean of the per-episode success
    values, i.e. the whole-run average success in [0,1]."""
    s = np.asarray(successes, dtype=np.float64)
    return float(s.mean()) if len(s) else float("nan")


def steps_to_threshold(successes: Sequence[float],
                       cost_per_episode: Sequence[float],
                       window: int = 20, threshold: float = 0.8) -> float:
    """Generic 'cost until rolling success >= threshold' - used for the
    B.4 intervention-efficiency measure with cost = interventions or
    human-seconds per episode. math.inf if never reached."""
    s = np.asarray(successes, dtype=np.float64)
    cum = np.cumsum(np.asarray(cost_per_episode, dtype=np.float64))
    roll = rolling_success(s, window)
    for i in range(len(s)):
        if i + 1 >= window and roll[i] >= threshold:
            return float(cum[i])
    return math.inf


# ===========================================================================
# B.3 - operator-dependency measures (within a method)
# ===========================================================================
def operator_spread(op_means: Dict[str, float]) -> dict:
    """sigma_op, CV, range, worst case over the operator means (B.3)."""
    ops = sorted(op_means)
    y = np.array([op_means[o] for o in ops], dtype=np.float64)
    out = dict(n_operators=len(y), mean=float(y.mean()))
    if len(y) > 1:
        out.update(sigma_op=float(y.std(ddof=1)),
                   var_op=float(y.var(ddof=1)),
                   cv=float(y.std(ddof=1) / (y.mean() + 1e-12)))
    else:
        out.update(sigma_op=float("nan"), var_op=float("nan"),
                   cv=float("nan"))
    out.update(range=float(y.max() - y.min()),
               worst=float(y.min()), best=float(y.max()),
               worst_operator=ops[int(np.argmin(y))],
               best_operator=ops[int(np.argmax(y))])
    return out


def icc_oneway(groups: Dict[str, Sequence[float]]) -> dict:
    """ICC(1,1): one-way random-effects over operator groups of per-seed
    values (B.3). Returns the ICC, the ANOVA mean squares, the implied
    variance components sigma2_op / sigma2_seed, and k0 (mean group size,
    Searle's approximation for unbalanced designs)."""
    groups = {g: np.asarray(v, dtype=np.float64)
              for g, v in groups.items() if len(v) > 0}
    n_groups = len(groups)
    sizes = np.array([len(v) for v in groups.values()], dtype=np.float64)
    n_total = sizes.sum()
    nanres = dict(icc=float("nan"), ms_between=float("nan"),
                  ms_within=float("nan"), k0=float("nan"),
                  sigma2_op=float("nan"), sigma2_seed=float("nan"))
    if n_groups < 2 or n_total <= n_groups:
        return nanres
    grand = np.concatenate(list(groups.values())).mean()
    ss_b = sum(len(v) * (v.mean() - grand) ** 2 for v in groups.values())
    ss_w = sum(((v - v.mean()) ** 2).sum() for v in groups.values())
    ms_b = ss_b / (n_groups - 1)
    ms_w = ss_w / (n_total - n_groups)
    k0 = (n_total - (sizes ** 2).sum() / n_total) / (n_groups - 1)
    sigma2_op = max((ms_b - ms_w) / k0, 0.0)
    icc = sigma2_op / (sigma2_op + ms_w + 1e-12)
    return dict(icc=float(icc), ms_between=float(ms_b),
                ms_within=float(ms_w), k0=float(k0),
                sigma2_op=float(sigma2_op), sigma2_seed=float(ms_w))


def bootstrap_icc(groups: Dict[str, Sequence[float]], B: int = DEFAULT_B,
                  rng: Optional[np.random.Generator] = None) -> dict:
    """95% bootstrap CI for ICC and sigma_op (C.2): hierarchical
    resampling - operators with replacement, then each sampled
    operator's seeds with replacement.

    At small K (4-6 operators) this CI is wide: the lower tail is
    dominated by low-diversity resamples (few distinct operators drawn).
    That is honest uncertainty about K=4, and it is why the
    pre-registered Exp-1 rule (C.3) tests the ICC point estimate
    together with the permutation p, not the CI bound."""
    rng = rng or np.random.default_rng(0)
    ops = sorted(groups)
    iccs, sigmas = [], []
    for _ in range(B):
        pick = rng.choice(len(ops), size=len(ops), replace=True)
        g = {}
        for j, gi in enumerate(pick):
            v = np.asarray(groups[ops[gi]], dtype=np.float64)
            g[f"b{j}"] = v[rng.integers(0, len(v), size=len(v))]
        res = icc_oneway(g)
        if np.isfinite(res["icc"]):
            iccs.append(res["icc"])
        means = np.array([np.mean(v) for v in g.values()])
        if len(means) > 1:
            sigmas.append(means.std(ddof=1))
    def ci(x):
        return ([float(v) for v in np.percentile(x, [2.5, 97.5])]
                if x else [float("nan")] * 2)
    return dict(icc_ci95=ci(iccs), sigma_op_ci95=ci(sigmas),
                n_boot=len(iccs))


def permutation_sigma2(values: Sequence[float], operators: Sequence[str],
                       B: int = DEFAULT_B,
                       rng: Optional[np.random.Generator] = None) -> dict:
    """Permutation test for 'operator matters' (C.2): shuffle operator
    labels across cells, recompute the between-operator variance of the
    group means; p = (#permutations >= observed + 1) / (B + 1)."""
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values, dtype=np.float64)
    operators = np.asarray(operators)

    def sig2(labels):
        means = [values[labels == o].mean() for o in np.unique(labels)]
        return np.var(means, ddof=1) if len(means) > 1 else np.nan

    obs = sig2(operators)
    count = 0
    for _ in range(B):
        count += sig2(rng.permutation(operators)) >= obs
    return dict(sigma2_obs=float(obs),
                p_value=float((count + 1) / (B + 1)), B=B)


# ===========================================================================
# B.4 - comparison measures (between methods, paired on operator)
# ===========================================================================
def _paired(yv: Dict[str, float], ya: Dict[str, float]):
    ops = sorted(set(yv) & set(ya))
    return (ops, np.array([yv[o] for o in ops]),
            np.array([ya[o] for o in ops]))


def robustness_gain(y_vanilla: Dict[str, float],
                    y_treat: Dict[str, float]) -> dict:
    """RG = var_op(vanilla) - var_op(treat); RG_rel = 1 - ratio (B.4-1,
    the primary endpoint)."""
    ops, v, a = _paired(y_vanilla, y_treat)
    if len(ops) < 2:
        return dict(rg=float("nan"), rg_rel=float("nan"), operators=ops)
    var_v, var_a = v.var(ddof=1), a.var(ddof=1)
    return dict(rg=float(var_v - var_a),
                rg_rel=float(1.0 - var_a / (var_v + 1e-12)),
                var_vanilla=float(var_v), var_treat=float(var_a),
                operators=ops)


def paired_bootstrap_rg_rel(y_vanilla: Dict[str, float],
                            y_treat: Dict[str, float],
                            B: int = DEFAULT_B,
                            rng: Optional[np.random.Generator] = None
                            ) -> dict:
    """D.2 primary: resample OPERATORS with replacement (keeping each
    operator's vanilla and treat values together), recompute RG_rel each
    draw, report the 95% percentile CI. Degenerate draws (zero vanilla
    variance, e.g. one operator picked n times) are dropped and
    counted."""
    rng = rng or np.random.default_rng(0)
    ops, v, a = _paired(y_vanilla, y_treat)
    if len(ops) < 2:
        return dict(rg_rel=float("nan"), ci95=[float("nan")] * 2)
    boot, dropped = [], 0
    for _ in range(B):
        idx = rng.integers(0, len(ops), size=len(ops))
        vv, aa = v[idx], a[idx]
        var_v = vv.var(ddof=1)
        if var_v < 1e-12:
            dropped += 1
            continue
        boot.append(1.0 - aa.var(ddof=1) / var_v)
    lo, hi = (np.percentile(boot, [2.5, 97.5])
              if boot else (float("nan"), float("nan")))
    return dict(rg_rel=float(1.0 - a.var(ddof=1) / (v.var(ddof=1) + 1e-12)),
                ci95=[float(lo), float(hi)], n_boot=len(boot),
                n_dropped_degenerate=dropped, B=B)


def mean_diff_guardrail(y_vanilla: Dict[str, float],
                        y_treat: Dict[str, float],
                        delta: float = 0.05, B: int = DEFAULT_B,
                        rng: Optional[np.random.Generator] = None) -> dict:
    """Non-inferiority guardrail (B.4-2 / D.3): dbar = mean paired
    difference; PASS iff the 95% CI lower bound of dbar > -delta.
    Bootstrap CI over operators (primary at small K) plus the classical
    paired-t interval for reference."""
    rng = rng or np.random.default_rng(0)
    ops, v, a = _paired(y_vanilla, y_treat)
    d = a - v
    if len(d) == 0:
        return dict(dbar=float("nan"), passed=False)
    boot = [d[rng.integers(0, len(d), size=len(d))].mean()
            for _ in range(B)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    dbar = float(d.mean())
    t_ci = [float("nan")] * 2
    if len(d) > 1 and d.std(ddof=1) > 0:
        try:
            from scipy import stats as sps
            se = d.std(ddof=1) / math.sqrt(len(d))
            tq = sps.t.ppf(0.975, len(d) - 1)
            t_ci = [dbar - tq * se, dbar + tq * se]
        except ImportError:
            pass
    return dict(dbar=dbar, ci95_bootstrap=[float(lo), float(hi)],
                ci95_t=t_ci, delta=delta,
                per_operator={o: float(x) for o, x in zip(ops, d)},
                passed=bool(lo > -delta))


def worst_operator_gain(y_vanilla: Dict[str, float],
                        y_treat: Dict[str, float]) -> dict:
    """B.4-3: min_o ybar_o(treat) - min_o ybar_o(vanilla)."""
    ops, v, a = _paired(y_vanilla, y_treat)
    if not ops:
        return dict(gain=float("nan"))
    return dict(gain=float(a.min() - v.min()),
                worst_vanilla=float(v.min()), worst_treat=float(a.min()))


def interaction_decomposition(table: Dict[str, Dict[str, float]]) -> dict:
    """B.4-5, exact algebra on the balanced methods x operators table of
    operator means:  y_mo = mu + alpha_m + b_o + gamma_mo.

    Returns every component plus two variance readouts:
      * var_gamma - variance of the interaction residuals (with two
        methods these are antisymmetric per operator, so this is the
        overall interaction variance);
      * var_operator_effects[m] - Var over operators of (y_mo - column
        mean) = Var(b_o + gamma_mo | m): the per-method operator spread
        whose vanilla -> AQI shrinkage is the robustness story.
    The reconstruction max-error certifies the decomposition is exact.
    """
    methods = sorted(table)
    ops = sorted(set.intersection(*[set(table[m]) for m in methods]))
    Y = np.array([[table[m][o] for o in ops] for m in methods])
    mu = float(Y.mean())
    alpha = {m: float(Y[i].mean() - mu) for i, m in enumerate(methods)}
    b = {o: float(Y[:, j].mean() - mu) for j, o in enumerate(ops)}
    gamma = {m: {o: float(Y[i, j] - mu - alpha[m] - b[o])
                 for j, o in enumerate(ops)} for i, m in enumerate(methods)}
    recon_err = max(abs(Y[i, j] - (mu + alpha[m] + b[o] + gamma[m][o]))
                    for i, m in enumerate(methods)
                    for j, o in enumerate(ops))
    g_all = np.array([gamma[m][o] for m in methods for o in ops])
    var_op_eff = {m: float(np.var([table[m][o] for o in ops], ddof=1))
                  for m in methods}
    return dict(methods=methods, operators=ops, mu=mu, alpha=alpha, b=b,
                gamma=gamma,
                var_gamma=float(g_all.var(ddof=1)),
                var_operator_effects=var_op_eff,
                reconstruction_max_err=float(recon_err))


def brown_forsythe(groups: Dict[str, Sequence[float]]) -> dict:
    """D.2 classical check on the variances: median-centred Levene
    (Brown-Forsythe) on the operator means between methods."""
    try:
        from scipy import stats as sps
    except ImportError:
        return dict(note="scipy not installed")
    arrays = [np.asarray(v, dtype=np.float64) for v in groups.values()]
    stat, p = sps.levene(*arrays, center="median")
    return dict(W=float(stat), p_value=float(p))


def wilcoxon_paired(y_vanilla: Dict[str, float],
                    y_treat: Dict[str, float]) -> dict:
    """D.2 means (secondary): paired Wilcoxon signed-rank on the
    operator means, with matched-pairs rank-biserial correlation and
    Cohen's d_z effect sizes."""
    ops, v, a = _paired(y_vanilla, y_treat)
    d = a - v
    nz = d[d != 0]
    out = dict(n_pairs=len(d), n_nonzero=len(nz))
    if len(nz) == 0:
        out.update(note="all paired differences are zero")
        return out
    try:
        from scipy import stats as sps
        res = sps.wilcoxon(a, v, zero_method="wilcox")
        out.update(statistic=float(res.statistic), p_value=float(res.pvalue))
    except ImportError:
        out.update(note="scipy not installed; test skipped")
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    w_pos = float(ranks[nz > 0].sum())
    w_neg = float(ranks[nz < 0].sum())
    out["rank_biserial"] = (w_pos - w_neg) / (w_pos + w_neg)
    if len(d) > 1 and d.std(ddof=1) > 0:
        out["cohen_dz"] = float(d.mean() / d.std(ddof=1))
    return out


def glmm_success(cells: List[dict], vanilla_method: str) -> dict:
    """D.2 'model view' at the statistically-correct trial level: expand
    each cell's (k, n) into n Bernoulli rows and fit a Binomial mixed
    GLMM with a fixed method effect, a random operator intercept, and a
    random per-operator method slope (the gamma term). Requires
    statsmodels + pandas; degrades to a note otherwise.

    cells: dicts with method, operator, k (successes), n (trials)."""
    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        return dict(note="statsmodels/pandas not installed - trial-level "
                         "GLMM skipped (pip install statsmodels pandas); "
                         "the exact interaction decomposition above is "
                         "the assumption-light substitute.")
    rows = []
    for c in cells:
        t = 1.0 if c["method"] != vanilla_method else 0.0
        rows += [dict(success=1, treat=t, operator=str(c["operator"]))] * int(c["k"])
        rows += [dict(success=0, treat=t, operator=str(c["operator"]))] * int(c["n"] - c["k"])
    df = pd.DataFrame(rows)
    try:
        md = BinomialBayesMixedGLM.from_formula(
            "success ~ treat",
            {"op": "0 + C(operator)", "op_slope": "0 + C(operator):treat"},
            df)
        res = md.fit_vb()
        names = list(res.model.exog_names)
        i = names.index("treat")
        beta, sd = float(res.fe_mean[i]), float(res.fe_sd[i])
        vcp = {n: float(np.exp(m)) for n, m in
               zip(res.model.vcp_names, res.vcp_mean)}
        return dict(n_rows=len(df),
                    method_coef_logodds=beta, method_coef_sd=sd,
                    method_coef_ci95=[beta - 1.96 * sd, beta + 1.96 * sd],
                    random_sd=vcp,
                    note="vcp values are posterior-mean SDs of the random "
                         "effects; 'op_slope' is the operator-specific "
                         "method response (the gamma term) - small means "
                         "operator-robust.")
    except Exception as e:  # numerical failure on tiny/degenerate data
        return dict(note=f"GLMM fit failed: {e}")


# ===========================================================================
# C.3 / D.3 - pre-registered decision rules
# ===========================================================================
def exp1_decision(icc: float, perm_p: float,
                  range_pp: Optional[float] = None,
                  extremes_nonoverlap: Optional[bool] = None,
                  icc_min: float = 0.3, alpha: float = 0.05,
                  range_min_pp: float = 0.15) -> dict:
    """C.3: H1 supported (sim) iff ICC >= 0.3 AND permutation p < 0.05;
    (real) iff operator range >= 15 pp with non-overlapping extreme
    Wilson CIs."""
    sim = bool(np.isfinite(icc) and icc >= icc_min and perm_p < alpha)
    out = dict(sim_supported=sim,
               sim_rule=f"ICC>={icc_min} and p<{alpha}",
               icc=icc, perm_p=perm_p)
    if range_pp is not None:
        out.update(real_supported=bool(range_pp >= range_min_pp
                                       and bool(extremes_nonoverlap)),
                   real_rule=f"range>={range_min_pp} and extreme Wilson "
                             f"CIs non-overlapping",
                   range_pp=range_pp,
                   extremes_nonoverlap=extremes_nonoverlap)
    return out


def exp2_decision(rg_rel_ci: Sequence[float], guardrail_passed: bool,
                  worst_gain: float,
                  interventions_to_80_reduced: Optional[bool],
                  violation_rate_not_increased: Optional[bool],
                  ablation_ordering_ok: Optional[bool]) -> dict:
    """D.3 checklist. PRIMARY: RG_rel 95% CI excludes 0 (lower bound
    > 0). GUARDRAIL: non-inferiority must pass regardless of RG.
    SECONDARY items are reported, not accept/reject."""
    primary = bool(np.isfinite(rg_rel_ci[0]) and rg_rel_ci[0] > 0)
    return dict(
        primary_rg_rel_ci_excludes_0=primary,
        guardrail_noninferiority=bool(guardrail_passed),
        overall_pass=bool(primary and guardrail_passed),
        secondary=dict(
            worst_operator_gain_positive=bool(worst_gain > 0)
            if np.isfinite(worst_gain) else None,
            interventions_to_80_reduced=interventions_to_80_reduced,
            violation_rate_not_increased=violation_rate_not_increased,
            ablation_ordering_ok=ablation_ordering_ok,
        ),
    )
