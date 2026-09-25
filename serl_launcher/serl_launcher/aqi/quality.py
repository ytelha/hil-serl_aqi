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
    # filled by score_segment():
    a_hat: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    q_signed: Optional[np.ndarray] = None
    q_int: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    excluded: bool = False
    exclusion_reason: str = ""

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


SEGMENT_FILTERS = {
    "safety_veto": _filter_safety_veto,
    "jerk_veto": _filter_jerk_veto,
    "outcome_filter": _filter_outcome,
    "unsolicited": _filter_unsolicited,
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


def heuristic_quality(z_features: np.ndarray, alphas: np.ndarray) -> float:
    """q_feat given standardized features [dprog, dV, success, F_peak,
    jerk, T_int] and weights (a1..a6). Sign convention per S5.5."""
    z = np.asarray(z_features, dtype=np.float64).reshape(6)
    a = np.asarray(alphas, dtype=np.float64).reshape(6)
    signs = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    return float(np.sum(a * signs * z))
