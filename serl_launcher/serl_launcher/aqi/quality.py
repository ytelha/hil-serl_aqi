"""Intervention-quality scoring (S5.5) and its mapping to replay priority
and imitation-loss weights (S5.6).

Layered design from the proposal ("Decision and resulting design"):
    (1) Option B - counterfactual advantage - is the primary graded signal:
            A_hat(s, a_H) = mu_Q(s, a_H) - mu_Q(s, a_robot)
            q_signed      = A_hat * g(sigma_Q)
            q_int         = clip(q_signed, 0, inf)
        with the disagreement gate g(sigma) = 1 / (1 + (sigma/sigma_ref)^2).
    (2) Option G - outcome filter: segments from clearly-failed rollouts are
        excluded from the intervention (imitation) buffer before grading.
    (3) Force/torque (and optionally jerk) are HARD SAFETY VETOES, not
        quality terms: a vetoed segment is excluded from imitation but its
        transitions remain ordinary RL experience.
    (4) Option A - the hand-weighted feature score q_feat - is computed and
        logged as the interpretable baseline for the offline analysis (RQ2)
        but does not drive the online loop.

Composability: layer (1)'s A_hat is the irreducible core; the disagreement
gate is switchable (cfg.quality_disagreement_gate) and layers (2)-(3) are
named filter sections selected by cfg.quality_filters (see SEGMENT_FILTERS
below). With quality_disagreement_gate=False and quality_filters=() the
quality signal is the counterfactual advantage alone.

Mapping to training signals (S5.6):
    replay priority   p_i = (eps + |delta_i| + eta * max(q_int_i, 0))^alpha
                            (implemented in QualityAwareReplayBuffer)
    imitation weight  w_i = 0                          if A_hat_i < adv_exclude
                          = clip(sigmoid(gamma * q~_i), w_min, w_max) otherwise
    actor loss        L_actor = L_RL + lambda_BC * mean_i(w_i * L_int_i)
                            (implemented in the SAC agents)

q~_i is q_signed standardized by a running scale of |A_hat| so `gamma` is
dimension-free. Note sigmoid(0) = 0.5: a redundant-but-harmless correction
keeps a moderate weight, a harmful one (A_hat < adv_exclude) is hard-zeroed,
exactly the asymmetric treatment the proposal specifies.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from serl_launcher.aqi.force_risk import action_jerk
from serl_launcher.aqi.normalization import RunningZScore


def disagreement_gate(sigma: np.ndarray, sigma_ref: float) -> np.ndarray:
    """g(sigma_Q): 1 where the ensemble agrees, -> 0 as it disagrees."""
    sigma = np.asarray(sigma, dtype=np.float64)
    return 1.0 / (1.0 + (sigma / max(sigma_ref, 1e-8)) ** 2)


@dataclass
class InterventionSegment:
    """One contiguous human takeover, buffered until episode end."""
    operator_id: str
    start_step: int                      # env step index at segment start
    requested: bool                      # was it preceded by an active query?
    transitions: List[dict] = field(default_factory=list)
    obs_list: List = field(default_factory=list)       # obs at each step
    human_actions: List = field(default_factory=list)  # a_H
    robot_actions: List = field(default_factory=list)  # a_robot at same s
    forces: List = field(default_factory=list)         # |F| per step
    torques: List = field(default_factory=list)        # |tau| per step
    hard_violation: bool = False
    # simulator ground truth per step (captured only when quality_cf_diag is
    # on; consumed by quality_score_mode="oracle"):
    gt_z_task: List = field(default_factory=list)      # true task trouble
    gt_jammed: List = field(default_factory=list)      # physical jam flag
    gt_height: List = field(default_factory=list)      # mm above the hole
    # filled by score_segment():
    a_hat: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    q_signed: Optional[np.ndarray] = None
    q_int: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    excluded: bool = False
    exclusion_reason: str = ""
    # inconsistent operator: hidden condition at the takeover onset (label)
    condition: Optional[str] = None
    # discard filters: the score the segment was ranked by, and whether the
    # quality warm-up still held the filter off
    drop_score: Optional[float] = None
    drop_warmup: Optional[bool] = None

    def __len__(self):
        return len(self.transitions)


# --------------------------------------------------------------------------
# Segment filters (S5.5 items 2-3): each is an independent, removable
# section returning True when the segment must be EXCLUDED from the
# imitation buffer. cfg.quality_filters selects which run; they combine by
# short-circuit OR (first hit names the exclusion_reason). None of them
# touches the graded signal itself.
# --------------------------------------------------------------------------
def _filter_safety_veto(seg, cfg, ctx) -> bool:
    """Hard force/torque limit breached during the segment (S5.4/S5.5-3)."""
    return bool(seg.hard_violation)


def _filter_jerk_veto(seg, cfg, ctx) -> bool:
    """Operator action jerk above cfg.jerk_hard (no-op when None)."""
    if cfg.jerk_hard is None or len(seg.human_actions) < 3:
        return False
    jerk = action_jerk(np.asarray(seg.human_actions))
    return float(np.max(jerk)) > cfg.jerk_hard


def _filter_outcome(seg, cfg, ctx) -> bool:
    """Option G: drop segments from clearly-failed episodes."""
    return ctx.get("episode_success") is False


def _filter_unsolicited(seg, cfg, ctx) -> bool:
    """Drop non-requested takeovers when the gate is live and
    unsolicited_policy == "rl_only" (the transitions stay RL data)."""
    return bool(ctx.get("active_on")) and not seg.requested


def _filter_after_scoring(seg, cfg, ctx) -> bool:
    """Discard filters decide after the final score (AQIRuntime.end_episode)."""
    return False


SEGMENT_FILTERS = {
    "safety_veto": _filter_safety_veto,
    "jerk_veto": _filter_jerk_veto,
    "outcome_filter": _filter_outcome,
    "unsolicited": _filter_unsolicited,
    "score_filter": _filter_after_scoring,
    "oracle_filter": _filter_after_scoring,
    "random_filter": _filter_after_scoring,
}


class QualityScorer:
    """Stateful scorer around the counterfactual-advantage core signal:
    tracks the running scales used by the optional disagreement gate and
    the weight sigmoid, runs the configured filter sections, and produces
    per-transition q/w."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._sigma_scale = RunningZScore(center=False)   # running |sigma_Q|
        self._adv_scale = RunningZScore(center=False)     # running |A_hat|
        # behavioral-score running standardisers (quality_score_mode="behavioral")
        self._z_tint = RunningZScore()
        self._z_dval = RunningZScore()
        self._z_fpeak = RunningZScore()
        # running |x| scales of the sign-carrying segment scores
        # (quality_score_mode="hindsight" / "progress"): dividing by the mean
        # magnitude keeps quality_gamma dimension-free WITHOUT centring, so a
        # negative hindsight advantage / lost depth stays negative.
        self._hind_scale = RunningZScore(center=False)
        self._prog_scale = RunningZScore(center=False)
        self._oracle_scale = RunningZScore(center=False)

    def behavioral_q(self, t_int: float, d_value: float, f_peak: float) -> float:
        """Critic-free intervention score: short + value-increasing + force-
        controlled takeover -> high. z(d_value) - z(t_int) - z(f_peak), each
        standardised online, restricted to cfg.quality_behavioral_terms
        (default: all three). A term's z-scorer returns 0 until it has seen
        min_count segments."""
        terms = getattr(self.cfg, "quality_behavioral_terms",
                        ("d_value", "t_int", "f_peak"))
        q = 0.0
        if "d_value" in terms:
            q += self._z_dval.z(float(d_value))
        if "t_int" in terms:
            q -= self._z_tint.z(float(t_int))
        if "f_peak" in terms:
            q -= self._z_fpeak.z(float(f_peak))
        return q

    @staticmethod
    def _scaled(scale: RunningZScore, x: float) -> float:
        """x / running mean |x| (sign kept). Non-finite x (e.g. a missing
        value estimate) scores 0 = neutral and does not touch the scale."""
        x = float(x)
        if not np.isfinite(x):
            return 0.0
        scale.update(abs(x))
        return x / max(scale.mean, 1e-6)

    def hindsight_q(self, a_hind: float) -> float:
        """quality_score_mode="hindsight": A_hind / running mean |A_hind|."""
        return self._scaled(self._hind_scale, a_hind)

    def progress_q(self, d_progress_post: float) -> float:
        """quality_score_mode="progress": the post-release progress change
        over its running mean magnitude."""
        return self._scaled(self._prog_scale, d_progress_post)

    @staticmethod
    def outcome_q(episode_success: bool) -> float:
        """quality_score_mode="outcome": +1 success, -1 failure."""
        return 1.0 if episode_success else -1.0

    def oracle_q(self, seg: "InterventionSegment",
                 episode_success: Optional[bool]) -> float:
        """quality_score_mode="oracle": a simulator-GROUND-TRUTH ceiling.

        Scores a takeover by how much it actually resolved task trouble, using
        the per-step ground truth captured under quality_cf_diag:
            resolved = op_z_task(onset) - op_z_task(handback)   (>= 0 = helped)
            q_raw    = resolved - jam_end + 0.5 * (+1 success / -1 fail)
        then standardised by the running mean |q_raw| so quality_gamma stays
        dimension-free. This is the "perfect quality score" the counterfactual
        and behavioral scores only approximate -- not deployable, a ceiling."""
        z = np.asarray(seg.gt_z_task, dtype=np.float64)
        z = z[np.isfinite(z)]
        resolved = float(z[0] - z[-1]) if z.size >= 1 else 0.0
        jam_end = 1.0 if (seg.gt_jammed and bool(seg.gt_jammed[-1])) else 0.0
        outcome = 1.0 if episode_success else -1.0
        q_raw = resolved - jam_end + 0.5 * outcome
        return self._scaled(self._oracle_scale, q_raw)

    def apply_segment_score(self, seg: InterventionSegment,
                            q_seg: float) -> InterventionSegment:
        """Replace the counterfactual q/w of a scored segment by ONE signed
        segment-level score q_seg (every mode except "counterfactual"):
            q_signed = q_seg,  q_int = max(q_seg, 0)          (replay priority)
            w        = clip(sigmoid(gamma * q_seg), w_min, w_max)   (BC weight)
        Excluded segments keep the zero weights of the filter pipeline."""
        n = len(seg)
        seg.q_signed = np.full(n, q_seg, dtype=np.float64)
        seg.q_int = np.clip(seg.q_signed, 0.0, None)
        if not seg.excluded:
            w = 1.0 / (1.0 + np.exp(
                -np.clip(self.cfg.quality_gamma * q_seg, -30, 30)))
            w = float(np.clip(w, self.cfg.quality_w_min, self.cfg.quality_w_max))
            seg.weights = np.full(n, w, dtype=np.float64)
        return seg

    # -- called every control step so scales track the current critic -----
    def observe_sigma(self, sigma_q: float) -> None:
        self._sigma_scale.update(abs(float(sigma_q)))

    @property
    def sigma_ref(self) -> float:
        if self.cfg.quality_sigma_ref is not None:
            return self.cfg.quality_sigma_ref
        # running mean of sigma_Q; std of a center=False RunningZScore is
        # sqrt(E[(x - mean)^2]) so use mean directly:
        return max(self._sigma_scale.mean, 1e-6)

    # ---------------------------------------------------------------------
    def score_segment(self, seg: InterventionSegment,
                      a_hat: np.ndarray, sigma: np.ndarray,
                      episode_success: Optional[bool],
                      active_on: bool = False) -> InterventionSegment:
        """Attach q/w arrays and the exclusion decision to a segment.

        Args:
            a_hat, sigma: outputs of counterfactual_advantage_batch for the
                segment's (obs, a_H, a_robot) triples.
            episode_success: outcome of the episode containing the segment
                (None -> unknown, treated as success for filtering).
            active_on: whether the query gate is live (context for the
                "unsolicited" filter section).
        """
        cfg = self.cfg
        a_hat = np.asarray(a_hat, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        for a in np.abs(a_hat):
            self._adv_scale.update(a)

        # ---- core graded signal: the counterfactual advantage -----------
        # With quality_disagreement_gate=False this is EXACTLY q = A_hat
        # (pure Option B); the gate g(sigma_Q) is an optional modifier.
        if cfg.quality_disagreement_gate:
            g = disagreement_gate(sigma, self.sigma_ref)
        else:
            g = np.ones_like(a_hat)
        q_signed = a_hat * g
        q_int = np.clip(q_signed, 0.0, None)

        scale = max(self._adv_scale.mean, 1e-6)
        q_tilde = q_signed / scale
        w = 1.0 / (1.0 + np.exp(-np.clip(cfg.quality_gamma * q_tilde,
                                         -30, 30)))
        w = np.clip(w, cfg.quality_w_min, cfg.quality_w_max)
        # hard-zero harmful corrections: derived from A_hat itself, so it
        # belongs to the core signal (disable via adv_exclude=-inf)
        w = np.where(a_hat < cfg.adv_exclude, 0.0, w)

        seg.a_hat, seg.sigma = a_hat, sigma
        seg.q_signed, seg.q_int, seg.weights = q_signed, q_int, w

        # ---- filter pipeline: named, individually removable sections ----
        ctx = dict(episode_success=episode_success, active_on=active_on)
        for name in cfg.effective_quality_filters:
            if SEGMENT_FILTERS[name](seg, cfg, ctx):
                seg.excluded, seg.exclusion_reason = True, name
                break
        if seg.excluded:
            seg.weights = np.zeros_like(w)
        return seg


# --------------------------------------------------------------------------
# Option A: hand-weighted heuristic feature score (baseline / RQ2 logging)
#   q_feat = a1 z(dprog) + a2 z(dV) + a3 success
#          - a4 z(F_peak) - a5 z(jerk) - a6 z(T_int)
# z() here is per-run standardization applied offline in the analysis
# scripts; online we log the RAW features so the offline fit (Phase 2)
# can calibrate the alphas.
# --------------------------------------------------------------------------
FEATURE_KEYS = ("d_progress", "d_value", "success", "f_peak", "jerk_mean",
                "t_int")


def segment_features(seg: InterventionSegment, d_progress: float,
                     d_value: float, post_success: bool) -> dict:
    """Raw Option-A features for one segment (logged, not used online)."""
    human_actions = np.asarray(seg.human_actions, dtype=np.float64)
    jerk = action_jerk(human_actions) if len(human_actions) else np.zeros(1)
    return dict(
        d_progress=float(d_progress),
        d_value=float(d_value),
        success=bool(post_success),
        f_peak=float(np.max(seg.forces)) if seg.forces else 0.0,
        tau_peak=float(np.max(seg.torques)) if seg.torques else 0.0,
        jerk_mean=float(np.mean(jerk)),
        jerk_max=float(np.max(jerk)),
        t_int=int(len(seg)),
    )


# --------------------------------------------------------------------------
# Post-release window: "hindsight" and "progress" judge a takeover by what
# the ROBOT reached once it had control back, i.e. over [onset, handback+K].
# Pure functions of per-step episode data so the online runtime and the
# offline audit (examples/quality_signal_audit.py) share one definition.
# --------------------------------------------------------------------------
def post_release_window(t0: int, length: int, n_steps: int, horizon: int,
                        terminal: bool):
    """End index t_e of the window [t0, t0 + length + horizon] clipped to an
    episode of n_steps transitions (steps 0..n_steps-1), and whether the
    return is bootstrapped from V(s_te).

    t0 + length is the handback step (first autonomous step). If the window
    reaches the episode end:
      - terminated (mask 0 on the last step): t_e = n_steps, no bootstrap
        (nothing follows a terminal state);
      - truncated (time limit): t_e = n_steps - 1, bootstrap from the last
        state that has a value estimate.
    """
    t_e = int(t0) + int(length) + int(horizon)
    if t_e <= n_steps - 1:
        return t_e, True
    if terminal:
        return int(n_steps), False
    return max(int(n_steps) - 1, int(t0)), True


# KNOWN LIMITATION (cluster method test 2026-09-26, handoff FINDINGS F4):
# before the critic has seen any reward, r = 0 in the window and V is nearly
# flat, so A_hind ~ (g^(n+K) - 1) * V(s_t0): negative, and set by the
# takeover length n and the onset value rather than by what the takeover
# achieved (corr +0.96 with that expression over 28 segments). Treat
# "hindsight" scores from early training as a timing signal.
def hindsight_advantage(values, rewards, t0: int, length: int, horizon: int,
                        discount: float, terminal: bool):
    """n-step advantage of the takeover measured on the states actually
    visited (AWR / CRR / GAE lineage):

        A_hind = sum_{t=t0}^{t_e-1} g^(t-t0) r_t + g^(t_e-t0) V(s_te) - V(s_t0)

    values[t] = V(s_t) (here mu_Q(s_t, a_robot_t), NaN where unknown),
    rewards[t] = r_t, both over the whole episode. The critic is only ever
    evaluated at the policy's own action, never at a_H. Returns
    (A_hind, t_e); A_hind is NaN if a needed value is missing."""
    values = np.asarray(values, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    t_e, bootstrap = post_release_window(t0, length, len(rewards), horizon,
                                         terminal)
    disc = float(discount) ** np.arange(t_e - t0)
    ret = float(np.sum(disc * rewards[t0:t_e]))
    if bootstrap:
        ret += float(discount) ** (t_e - t0) * values[t_e]
    return float(ret - values[t0]), int(t_e)


def heuristic_quality(z_features: np.ndarray, alphas: np.ndarray) -> float:
    """q_feat given standardized features [dprog, dV, success, F_peak,
    jerk, T_int] and weights (a1..a6). Sign convention per S5.5."""
    z = np.asarray(z_features, dtype=np.float64).reshape(6)
    a = np.asarray(alphas, dtype=np.float64).reshape(6)
    signs = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    return float(np.sum(a * signs * z))
