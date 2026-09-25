"""Automatic operator skill assessment (methodology doc Part B, S9-S11).

Feature catalogue (doc Table 7) -> per-operator aggregation and
normalization (S10.1) -> four assignment methods (S10.2-10.5) ->
validation metrics (S11). Everything computes from the standard log
tuple; the counterfactual-advantage aggregates re-use the AQI quality
numerator already logged per intervention segment.

Censoring discipline (the doc's Table-5 lesson): any per-episode
statistic is computed over ALL episodes with failures capped at the
budget and flagged; time-to-success additionally reports the
Kaplan-Meier median, which is the honest scalar under censoring.

Dependencies: numpy; scipy for Savitzky-Golay filtering and Beta
quantiles (repo requirement); statsmodels optional for the ordinal
logistic variant of Method C (a built-in multinomial Newton solver is
the default).
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from serl_launcher.aqi import stats as aqi_stats

CONTROL_HZ = 10.0

# feature -> +1 if larger is better, -1 if smaller is better (Table 7)
FEATURE_SIGNS = dict(
    sr=+1, t_mean_capped=-1, t_km_median=-1, attempts=-1, plr=-1,
    idle_frac=-1, ldj=+1, sparc=+1, vel_peaks=-1, reversals=-1,
    f_peak=-1, f_impulse=-1, jams=-1, scrape_frac=-1, commit_err=-1,
    cv_t=-1, dtw=-1, lapse_freq=-1,
    intervention_rate=-1, takeover_dur=-1, over_assist=-1,
    onset_margin=-1, recovery=+1, dv_rec=+1, a_hat_mean=+1,
    a_hat_frac_pos=+1, unnecessary_frac=-1,
)
# the seven-feature set of the doc's worked example (S10.2)
CORE7 = ("sr", "t_mean_capped", "plr", "ldj", "jams", "idle_frac",
         "commit_err")


# ===========================================================================
# S9 - per-episode features
# ===========================================================================
def _savgol(x, window=7, poly=3):
    try:
        from scipy.signal import savgol_filter
        w = min(window if window % 2 else window + 1,
                len(x) - (1 - len(x) % 2))
        if w <= poly + 1:
            return np.asarray(x, dtype=np.float64)
        return savgol_filter(np.asarray(x, dtype=np.float64), w, poly,
                             axis=0)
    except Exception:
        return np.asarray(x, dtype=np.float64)


def ldj(positions: np.ndarray, dt: float) -> float:
    """Log dimensionless jerk (doc S9.2, Balasubramanian et al.):
    LDJ = -ln( (T^5 / L^2) * integral ||jerk||^2 dt ); HIGHER = smoother.
    Positions are Savitzky-Golay smoothed before differentiating."""
    p = _savgol(positions)
    if len(p) < 5:
        return float("nan")
    v = np.gradient(p, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    T = len(p) * dt
    L = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    if L <= 1e-9:
        return float("nan")
    integ = float(np.sum(np.sum(j * j, axis=1)) * dt)
    return -math.log(max((T ** 5 / L ** 2) * integ, 1e-300))


def sparc(speed: np.ndarray, fs: float, fc: float = 10.0,
          amp_th: float = 0.05) -> float:
    """Spectral arc length of the speed profile (doc S9.2; the
    noise-robust smoothness choice for real data). Standard definition:
    arc length of the normalized magnitude spectrum up to the adaptive
    cutoff; HIGHER (closer to 0, less negative) = smoother."""
    v = np.asarray(speed, dtype=np.float64)
    if len(v) < 6 or np.all(v < 1e-9):
        return float("nan")
    n = max(512, 2 ** int(math.ceil(math.log2(len(v)))))
    mag = np.abs(np.fft.rfft(v, n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    mag = mag / max(mag[0], 1e-12)
    sel = f <= fc
    idx = np.where(sel & (mag >= amp_th))[0]
    cutoff = f[idx[-1]] if len(idx) else fc
    sel = f <= max(cutoff, f[1])
    fn = f[sel] / max(f[sel][-1], 1e-12)
    dm = np.diff(mag[sel])
    dfn = np.diff(fn)
    return -float(np.sum(np.sqrt(dfn ** 2 + dm ** 2)))


def _count_peaks(speed, frac=0.25):
    v = np.asarray(speed)
    if len(v) < 3:
        return 0
    th = frac * float(v.max()) if v.max() > 0 else 0.0
    return int(np.sum((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])
                      & (v[1:-1] > th)))


def episode_features(traj: dict, dt: float = 1.0 / CONTROL_HZ,
                     budget_s: Optional[float] = None,
                     goal: Optional[Sequence[float]] = None) -> dict:
    """Per-episode feature row (doc Table 7, execution block).

    traj keys: positions (T, D) [mm]; success (bool); optional: forces
    (T,) [N-ish], contact (T,) bool, jam_events int, commit_err float,
    attempts int, vertical_dim int (default last), start/goal for L_ref.
    """
    p = np.asarray(traj["positions"], dtype=np.float64)
    T = len(p) * dt
    success = bool(traj.get("success", False))
    budget = budget_s if budget_s is not None else T
    v = np.linalg.norm(np.gradient(_savgol(p), dt, axis=0), axis=1) \
        if len(p) >= 3 else np.zeros(len(p))
    path_len = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    g = np.asarray(goal, dtype=np.float64) if goal is not None else p[-1]
    l_ref = max(float(np.linalg.norm(p[0] - g)), 1e-6)
    vd = int(traj.get("vertical_dim", p.shape[1] - 1))
    vz = np.gradient(_savgol(p[:, vd]), dt) if len(p) >= 3 else np.zeros(1)
    forces = np.asarray(traj.get("forces", np.zeros(len(p))),
                        dtype=np.float64)
    contact = np.asarray(traj.get("contact",
                                  forces > 0.05), dtype=bool)
    n_contact = max(int(contact.sum()), 1)
    return dict(
        success=success,
        t=float(T), t_capped=float(min(T, budget)), censored=not success,
        attempts=float(traj.get("attempts", 1)),
        plr=path_len / l_ref,
        idle_frac=float(np.mean(v < 2.0)),
        ldj=ldj(p, dt),
        sparc=sparc(v, 1.0 / dt),
        vel_peaks=float(_count_peaks(v)),
        reversals=float(np.sum(np.diff(np.sign(vz + 1e-12)) != 0)),
        f_peak=float(forces.max()) if len(forces) else 0.0,
        f_impulse=float(np.sum(forces) * dt),
        jams=float(traj.get("jam_events", 0)),
        scrape_frac=float(np.sum(forces[contact] > 0.5) / n_contact)
        if contact.any() else 0.0,
        commit_err=float(traj["commit_err"])
        if traj.get("commit_err") is not None else np.nan,
        positions=p,     # kept for DTW; dropped at aggregation
    )


def intervention_features(segments: List[dict],
                          episode_success: Optional[Sequence[bool]] = None,
                          matched_phase: bool = True) -> dict:
    """Per-operator Role-2 aggregates (doc S9.5) from AQI segment
    records (segments.jsonl rows or equivalent dicts). Uses: length,
    requested, a_hat_mean / a_hat_at_trigger, sigma_at_trigger,
    episode_success, and - when the step logs carry mu_q - the onset
    margin/recovered value computed upstream and passed per segment as
    dv_on / dv_rec."""
    out = dict(n_takeovers=float(len(segments)))
    if not segments:
        return out
    durs = [s.get("length", 0) for s in segments]
    a_means = [s["a_hat_mean"] for s in segments
               if s.get("a_hat_mean") is not None]
    out.update(
        takeover_dur=float(np.mean(durs)) / CONTROL_HZ,
        over_assist=float(np.mean([max(0, d - 20) for d in durs]))
        / CONTROL_HZ,
        a_hat_mean=float(np.mean(a_means)) if a_means else np.nan,
        a_hat_frac_pos=float(np.mean([a > 0 for a in a_means]))
        if a_means else np.nan,
    )
    dvo = [s["dv_on"] for s in segments if s.get("dv_on") is not None]
    dvr = [s["dv_rec"] for s in segments if s.get("dv_rec") is not None]
    if dvo:
        out["onset_margin"] = float(np.mean(dvo))
    if dvr:
        out["dv_rec"] = float(np.mean(dvr))
    if episode_success is not None and len(episode_success):
        out["recovery"] = float(np.mean(episode_success))
    unnecessary = [s["unnecessary"] for s in segments
                   if s.get("unnecessary") is not None]
    if unnecessary:
        out["unnecessary_frac"] = float(np.mean(unnecessary))
    return out


# ===========================================================================
# S10.1 - aggregation and normalization
# ===========================================================================
def km_median(times: Sequence[float], censored: Sequence[bool]) -> float:
    """Kaplan-Meier median time-to-success with right censoring at the
    budget (the honest scalar under censoring, doc Table-5 lesson)."""
    order = np.argsort(times)
    t = np.asarray(times, dtype=np.float64)[order]
    c = np.asarray(censored, dtype=bool)[order]
    n = len(t)
    surv, at_risk = 1.0, n
    for i in range(n):
        if not c[i]:
            surv *= (at_risk - 1) / at_risk
        at_risk -= 1
        if surv <= 0.5:
            return float(t[i])
    return float("inf")


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def _resample_path(p: np.ndarray, k: int = 40) -> np.ndarray:
    if len(p) < 2:
        return np.repeat(p, k, axis=0)[:k]
    d = np.concatenate([[0], np.cumsum(
        np.linalg.norm(np.diff(p, axis=0), axis=1))])
    d = d / max(d[-1], 1e-9)
    out = np.stack([np.interp(np.linspace(0, 1, k), d, p[:, j])
                    for j in range(p.shape[1])], axis=1)
    return out


def aggregate_operator(episode_rows: List[dict],
                       budget_s: float,
                       interventions: Optional[dict] = None) -> dict:
    """One feature vector per operator (doc S10.1): level = median over
    episodes; censoring handled (capped mean + KM median); consistency =
    CV of capped time, mean pairwise DTW of length-normalized paths,
    lapse frequency (episodes beyond 2 MAD of the operator's own median
    capped time)."""
    rows = episode_rows
    n = len(rows)
    k = int(sum(r["success"] for r in rows))
    tc = np.array([r["t_capped"] for r in rows])
    med = float(np.median(tc))
    mad = float(np.median(np.abs(tc - med))) + 1e-9
    paths = [_resample_path(r["positions"]) for r in rows]
    dtws = [_dtw(paths[i], paths[j])
            for i in range(min(n, 10)) for j in range(i + 1, min(n, 10))]
    out = dict(
        n_episodes=n, k_success=k, sr=k / n,
        sr_wilson=aqi_stats.wilson_interval(k, n),
        t_mean_capped=float(tc.mean()),
        t_km_median=km_median([r["t"] for r in rows],
                              [r["censored"] for r in rows]),
        cv_t=float(tc.std(ddof=1) / max(tc.mean(), 1e-9)) if n > 1
        else np.nan,
        lapse_freq=float(np.mean(np.abs(tc - med) > 2 * mad)),
        dtw=float(np.mean(dtws)) if dtws else np.nan,
    )
    for key in ("attempts", "plr", "idle_frac", "ldj", "sparc",
                "vel_peaks", "reversals", "f_peak", "f_impulse", "jams",
                "scrape_frac", "commit_err"):
        vals = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        out[key] = float(np.median(vals)) if vals else np.nan
    if interventions:
        out.update(interventions)
    return out


def normalize_features(op_vectors: Dict[str, dict],
                       features: Sequence[str],
                       robust: bool = False) -> Dict[str, dict]:
    """Per-task z-scores across the cohort (doc S10.1); robust=True uses
    median/MAD. Features missing for an operator z-score to 0 (no
    evidence either way)."""
    ops = sorted(op_vectors)
    Z = {o: {} for o in ops}
    for f in features:
        vals = np.array([op_vectors[o].get(f, np.nan) for o in ops],
                        dtype=np.float64)
        ok = np.isfinite(vals)
        if ok.sum() < 2:
            for o in ops:
                Z[o][f] = 0.0
            continue
        if robust:
            c = np.median(vals[ok])
            s = 1.4826 * np.median(np.abs(vals[ok] - c)) + 1e-9
        else:
            c, s = vals[ok].mean(), vals[ok].std(ddof=1) + 1e-9
        for o, v, good in zip(ops, vals, ok):
            Z[o][f] = float((v - c) / s) if good else 0.0
    return Z


# ===========================================================================
# S10.2 - Method A: transparent composite index
# ===========================================================================
def composite_score(z: dict, features: Sequence[str] = CORE7) -> float:
    """S_i = mean_k sign_k * z_ik with the Table-7 direction signs."""
    vals = [FEATURE_SIGNS[f] * z[f] for f in features if f in z]
    return float(np.mean(vals)) if vals else float("nan")


def composite_tiers(scores: Dict[str, float],
                    cuts: Optional[Sequence[float]] = None) -> dict:
    """Tertile cuts by default (balanced cohorts); pass anchor-derived
    cuts for skewed cohorts (doc S10.2 caveat)."""
    s = np.array(list(scores.values()))
    if cuts is None:
        cuts = (float(np.quantile(s, 1 / 3)), float(np.quantile(s, 2 / 3)))
    tier = {o: ("novice" if v < cuts[0] else
                "expert" if v >= cuts[1] else "intermediate")
            for o, v in scores.items()}
    return dict(cuts=tuple(cuts), tiers=tier)


# ===========================================================================
# S10.3 - Method B: unsupervised clustering
# ===========================================================================
def _kmeans(X, k, rng, n_init=10, iters=100):
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        # k-means++ init
        centers = [X[rng.integers(len(X))]]
        for _ in range(k - 1):
            d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers],
                        axis=0)
            centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(centers)
        for _ in range(iters):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), axis=1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any()
                             else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if inertia < best_inertia:
            best, best_inertia = (lab, C), inertia
    return best


def cluster_operators(z_vectors: Dict[str, dict],
                      features: Sequence[str] = CORE7, k: int = 3,
                      sr: Optional[Dict[str, float]] = None,
                      rng: Optional[np.random.Generator] = None,
                      n_boot: int = 100) -> dict:
    """k-means on the z-vectors, clusters ORDERED by mean success rate
    and only then read as novice/intermediate/expert (doc S10.3), with
    bootstrap label stability. Refuses nothing, but flags cohorts under
    15 operators as unstable per the doc."""
    rng = rng or np.random.default_rng(0)
    ops = sorted(z_vectors)
    X = np.array([[z_vectors[o][f] for f in features] for o in ops])
    lab, C = _kmeans(X, k, rng)
    order_key = (np.array([np.mean([sr[o] for o, l in zip(ops, lab)
                                    if l == j]) for j in range(k)])
                 if sr else -np.array([np.mean(X[lab == j, 0])
                                       for j in range(k)]))
    order = np.argsort(order_key)   # ascending -> novice..expert
    names = ["novice", "intermediate", "expert"] + \
        [f"tier{j}" for j in range(3, k)]
    remap = {int(order[j]): names[j] for j in range(k)}
    tiers = {o: remap[int(l)] for o, l in zip(ops, lab)}
    # bootstrap stability: perturb X with resampled ops
    agree = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ops), len(ops))
        lb, _ = _kmeans(X[idx], k, rng, n_init=3)
        # majority label per original op among its bootstrap copies
        m = defaultdict(list)
        for i, l in zip(idx, lb):
            m[i].append(l)
        stable = [len(set(v)) == 1 for v in m.values() if len(v) > 1]
        if stable:
            agree.append(np.mean(stable))
    return dict(tiers=tiers, labels={o: int(l) for o, l in zip(ops, lab)},
                stability=float(np.mean(agree)) if agree else np.nan,
                small_cohort_warning=len(ops) < 15)


def adjusted_rand_index(a: Sequence, b: Sequence) -> float:
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    ua, ub = np.unique(a), np.unique(b)
    M = np.array([[np.sum((a == x) & (b == y)) for y in ub] for x in ua])
    def c2(x):
        return x * (x - 1) / 2
    sum_ij = c2(M).sum()
    sa, sb = c2(M.sum(1)).sum(), c2(M.sum(0)).sum()
    exp = sa * sb / c2(n)
    mx = 0.5 * (sa + sb)
    return float((sum_ij - exp) / (mx - exp + 1e-12))


# ===========================================================================
# S10.4 - Method C: (ordinal) classification with LOSO CV
# ===========================================================================
def _softmax_fit(X, y, k, l2=1e-2, iters=300, lr=0.5):
    n, d = X.shape
    W = np.zeros((k, d + 1))
    Xb = np.hstack([X, np.ones((n, 1))])
    Y = np.eye(k)[y]
    for _ in range(iters):
        P = np.exp(Xb @ W.T - (Xb @ W.T).max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        G = (P - Y).T @ Xb / n + l2 * W
        W -= lr * G
    return W


def _softmax_predict(W, X):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return np.argmax(Xb @ W.T, axis=1)


def macro_f1(y_true, y_pred, k=3) -> float:
    f1s = []
    for c in range(k):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1s.append(2 * p * r / max(p + r, 1e-12))
    return float(np.mean(f1s))


def qwk(y_true, y_pred, k=3) -> float:
    """Quadratic-weighted Cohen's kappa (doc S10.4): the correct ordinal
    loss - confusing novice with expert costs 4x an adjacent slip."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    O = np.array([[np.sum((y_true == i) & (y_pred == j))
                   for j in range(k)] for i in range(k)], dtype=np.float64)
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)],
                 dtype=np.float64) / (k - 1) ** 2
    hist_t, hist_p = O.sum(1), O.sum(0)
    E = np.outer(hist_t, hist_p) / max(O.sum(), 1)
    denom = float((W * E).sum())
    return float(1.0 - (W * O).sum() / denom) if denom > 1e-12 else 0.0


def loso_classify(z_vectors: Dict[str, dict], labels: Dict[str, int],
                  features: Sequence[str] = CORE7, k: int = 3) -> dict:
    """Leave-one-OPERATOR-out CV (never episode splits - doc S10.4) of a
    multinomial logistic model on the per-operator z-vectors; reports
    accuracy, macro-F1 and QWK. statsmodels' ordinal (proportional-odds)
    model is used instead when installed."""
    ops = sorted(z_vectors)
    X = np.array([[z_vectors[o][f] for f in features] for o in ops])
    y = np.array([labels[o] for o in ops])
    preds = np.zeros_like(y)
    use_ordinal = False
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        use_ordinal = True
    except ImportError:
        pass
    for i in range(len(ops)):
        tr = np.arange(len(ops)) != i
        if use_ordinal:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = OrderedModel(y[tr], X[tr], distr="logit").fit(
                        method="bfgs", disp=False)
                preds[i] = int(np.argmax(
                    m.predict(X[i:i + 1])[0]))
                continue
            except Exception:
                pass
        W = _softmax_fit(X[tr], y[tr], k)
        preds[i] = _softmax_predict(W, X[i:i + 1])[0]
    return dict(accuracy=float(np.mean(preds == y)),
                macro_f1=macro_f1(y, preds, k), qwk=qwk(y, preds, k),
                predictions={o: int(p) for o, p in zip(ops, preds)},
                model="ordinal_logit" if use_ordinal else
                "multinomial_logit")


# ===========================================================================
# S10.5 - Method D: online estimators
# ===========================================================================
class BetaBernoulliTracker:
    """Beta(alpha, beta) posterior on per-episode success (doc worked
    example: 7/10 -> Beta(8,4), mean 0.667, 90% CI [0.44, 0.86])."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.a, self.b = alpha, beta

    def update(self, success: bool):
        self.a += bool(success)
        self.b += not bool(success)
        return self

    @property
    def mean(self):
        return self.a / (self.a + self.b)

    def ci(self, level: float = 0.90):
        from scipy.stats import beta as _beta
        lo = (1 - level) / 2
        return (float(_beta.ppf(lo, self.a, self.b)),
                float(_beta.ppf(1 - lo, self.a, self.b)))

    def tier(self, bands=(0.55, 0.90)):
        m = self.mean
        return ("novice" if m < bands[0] else
                "expert" if m >= bands[1] else "intermediate")


class EWMAComposite:
    """S-hat <- (1-lambda) S-hat + lambda S_episode; lambda sets the
    forgetting horizon and doubles as the drift handling (doc S10.5)."""

    def __init__(self, lam: float = 0.25):
        self.lam, self.s = lam, None

    def update(self, s_episode: float):
        self.s = (s_episode if self.s is None
                  else (1 - self.lam) * self.s + self.lam * s_episode)
        return self.s


def _logistic_newton(X, y, iters=50, l2=1e-6):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + l2 * w
        Wd = p * (1 - p) + 1e-9
        H = (X * Wd[:, None]).T @ X + l2 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w


def rasch_fit(trials: List[dict]) -> dict:
    """Difficulty-adjusted ability (doc S10.5, IRT/Rasch):
    P(success) = sigmoid(theta_op - b_task), fitted as a logistic
    regression with operator and task one-hot effects (first task's
    difficulty pinned to its given value for identifiability when task
    difficulties are supplied, else b_task[0] = 0).

    trials: dicts with operator, task, k, n (or success 0/1 with n=1);
    optionally b (known task difficulty) to fix task parameters and fit
    abilities only - the doc's worked example: 9/10 at b=-1 and 5/10 at
    b=+1 gives theta-hat = 1.05 with fitted P 0.89 / 0.51."""
    ops = sorted({t["operator"] for t in trials})
    tasks = sorted({t["task"] for t in trials})
    known_b = {t["task"]: t["b"] for t in trials if "b" in t}
    rows_X, rows_y = [], []
    for t in trials:
        k = int(t.get("k", t.get("success", 0)))
        n = int(t.get("n", 1))
        for succ in [1] * k + [0] * (n - k):
            x = np.zeros(len(ops) + (0 if known_b else len(tasks)))
            x[ops.index(t["operator"])] = 1.0
            off = 0.0
            if known_b:
                off = -known_b[t["task"]]
            else:
                x[len(ops) + tasks.index(t["task"])] = -1.0
            rows_X.append((x, off))
            rows_y.append(succ)
    X = np.array([r[0] for r in rows_X])
    off = np.array([r[1] for r in rows_X])
    y = np.array(rows_y, dtype=np.float64)
    # absorb offsets: P = sigmoid(Xw + off) -> augment
    Xa = np.hstack([X, off[:, None]])
    w = _logistic_newton_offset(Xa, y)
    theta = {o: float(w[i]) for i, o in enumerate(ops)}
    if known_b:
        b = dict(known_b)
    else:
        b = {t: float(w[len(ops) + j]) for j, t in enumerate(tasks)}
        shift = b[tasks[0]]
        b = {t: v - shift for t, v in b.items()}
        theta = {o: v - shift for o, v in theta.items()}
    def p(o, task):
        return 1.0 / (1.0 + math.exp(-(theta[o] - b[task])))
    return dict(theta=theta, b=b, predict=p)


def _logistic_newton_offset(Xa, y, iters=60, l2=1e-6):
    """Newton with the last column as a FIXED-coefficient offset."""
    d = Xa.shape[1] - 1
    w = np.zeros(d)
    for _ in range(iters):
        eta = Xa[:, :d] @ w + Xa[:, d]
        p = 1.0 / (1.0 + np.exp(-eta))
        g = Xa[:, :d].T @ (p - y) + l2 * w
        Wd = p * (1 - p) + 1e-9
        H = (Xa[:, :d] * Wd[:, None]).T @ Xa[:, :d] + l2 * np.eye(d)
        w -= np.linalg.solve(H, g)
    return w


# ===========================================================================
# S11 - validation utilities
# ===========================================================================
def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def test_retest_icc(scores_a: Dict[str, float],
                    scores_b: Dict[str, float]) -> float:
    """Split-battery reliability: the same ICC(1,1) machinery the
    operator-variance analysis uses, pointed at the classifier (two
    disjoint episode batches per operator)."""
    groups = {o: [scores_a[o], scores_b[o]]
              for o in scores_a if o in scores_b}
    return aqi_stats.icc_oneway(groups)["icc"]


# ===========================================================================
# real AQI-HIL-SERL training logs -> the same pipeline (doc S12 schema)
# ===========================================================================
def features_from_aqi_logs(run_dir: str, pos_slice=(0, 3),
                           dt: float = 1.0 / CONTROL_HZ,
                           budget_s: Optional[float] = None) -> dict:
    """Consume one aqi_logs run (steps/segments/episodes.jsonl) and
    return dict(episode_rows, interventions, meta) ready for
    aggregate_operator. Execution features come from the logged proprio
    positions (pos_slice into `state`) and f_norm; Role-2 features from
    the segment records, including the mean-A-hat aggregates (the AQI
    quality numerator re-aggregated per operator, doc S9.5) and - via
    the logged mu_q - the onset margin dV_on = median(mu_q pre-onset) -
    mu_q(onset) and the recovered value dV_rec."""
    import json as _json
    import os as _os

    def _rd(name):
        path = _os.path.join(run_dir, name)
        if not _os.path.exists(path):
            return []
        with open(path) as f:
            return [_json.loads(l) for l in f if l.strip()]

    steps, segs, eps = _rd("steps.jsonl"), _rd("segments.jsonl"), \
        _rd("episodes.jsonl")
    meta = {}
    mp = _os.path.join(run_dir, "meta.json")
    if _os.path.exists(mp):
        with open(mp) as f:
            meta = _json.load(f)

    ep_ids = sorted({s.get("episode", 0) for s in steps})
    by_ep = {e: [s for s in steps if s.get("episode", 0) == e]
             for e in ep_ids}
    succ_by_ep = {e.get("episode", i): bool(e.get("success", False))
                  for i, e in enumerate(eps)}

    ep_rows = []
    for e in ep_ids:
        est = by_ep[e]
        if not est or "state" not in est[0]:
            continue
        P = np.array([s["state"][pos_slice[0]:pos_slice[1]]
                      for s in est], dtype=np.float64)
        F = np.array([s.get("f_norm", 0.0) for s in est])
        ep_rows.append(episode_features(
            dict(positions=P, forces=F, contact=F > 0.1,
                 success=succ_by_ep.get(e, False), attempts=1),
            dt=dt, budget_s=budget_s or len(est) * dt))

    seg_records, succ_given_takeover = [], []
    for g in segs:
        rec = dict(g)
        e = g.get("episode")
        est = by_ep.get(e, [])
        if est and "mu_q" in est[0]:
            mus = [s["mu_q"] for s in est]
            onset = int(g.get("start_step", 0))
            onset = min(max(onset, 0), len(mus) - 1)
            pre = mus[max(0, onset - 10):onset] or mus[:1]
            rec["dv_on"] = float(np.median(pre) - mus[onset])
            hb = min(onset + int(g.get("length", 1)), len(mus) - 1)
            rec["dv_rec"] = float(mus[hb] - mus[onset])
        seg_records.append(rec)
        if g.get("episode_success") is not None:
            succ_given_takeover.append(bool(g["episode_success"]))
    interv = intervention_features(seg_records, succ_given_takeover)
    total = max(len(steps), 1)
    interv["intervention_rate"] = \
        sum(int(s.get("intervening", False)) for s in steps) / total
    return dict(episode_rows=ep_rows, interventions=interv, meta=meta)
