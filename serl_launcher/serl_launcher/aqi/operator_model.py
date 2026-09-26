"""Synthetic operator model for HIL-SERL (methodology doc Part A / S3-S6).

Implements the perception -> decision -> motor operator of the thesis
document "Synthetic Operator Behaviour and Automatic Skill
Classification" on top of AQI-HIL-SERL. Design requirements R1-R6 of
that document govern this module:

    R1 ordinal validity    higher skill s => stochastically better
                           outcomes (validated by the calibration test
                           against the document's Table-5 pattern);
    R2 ground truth        every operator carries its true s and theta;
    R3 human-plausible     errors enter through delayed perception,
                           per-approach misalignment, signal-dependent
                           motor noise, OU tremor, hesitation, recovery
                           style and lapses - never i.i.d. action noise;
    R4 both roles          the same theta drives demonstrations (Role 1)
                           and interventions (Role 2, operator_gate.py);
    R5 non-invasiveness    integration point == the SpaceMouse wrapper
                           (info["intervene_action"]), nothing
                           downstream can tell it from a human;
    R6 determinism         same seed => byte-identical episodes.

Two skill axes are kept separate throughout (doc S1.2): execution
(motor) parameters and monitoring (supervisory) parameters; mixed
profiles such as "skilled-but-jumpy" compose them independently.

Units follow the document's tables: millimetres, mm/s, control steps at
10 Hz, seconds. Environments in other units supply an `action_scale`
(env action units per mm/s) at the wrapper.
"""

import dataclasses
import json
import math
from collections import deque
from typing import Optional

import numpy as np

CONTROL_HZ = 10.0
LAPSE_NOISE_MULT = 2.5      # doc S3.3: lapse episodes inflate noise x2.5
JITTER_STD = 0.15           # doc S3.5: per-operator log-normal jitter


# ===========================================================================
# theta: all knobs of both skill axes in one place (doc Tables 2 + 4)
# ===========================================================================
@dataclasses.dataclass
class OperatorTheta:
    # ---- execution (motor/perception) skill, doc Table 2 -----------------
    sigma0: float = 5.0        # baseline motor noise (mm/s)
    k_sdn: float = 0.05        # signal-dependent noise gain (-)
    sigma_tr: float = 5.0      # OU tremor intensity (mm/s)
    latency: int = 2           # perception-action delay (steps @10 Hz)
    lag: float = 0.55          # first-order motor lag gamma (-)
    v_max: float = 70.0        # speed cap (mm/s)
    sigma_perc: float = 1.0    # per-APPROACH perceptual bias std (mm)
    p_pause: float = 0.04      # hesitation probability (/step)
    pause_len_mean: float = 0.6  # mean pause duration (s)
    q_rec: float = 0.30        # retract-and-realign prob. (/jam step)
    p_early: float = 0.05      # premature-commit prob. (/attempt, doc-2 M3)
    p_lapse: float = 0.06      # lapse probability (/episode)
    # ---- monitoring (supervisory) skill, doc Table 4 ---------------------
    sigma_mon: float = 0.12    # monitor noise on z in [0,1]
    delta_in: float = 0.60     # takeover trigger criterion
    m_in: int = 3              # consecutive steps above delta_in
    l_react: int = 5           # reaction delay after detection (steps)
    p_miss: float = 0.15       # per-true-event miss probability
    rho_fa: float = 0.5        # false alarms (/minute)
    delta_out: float = 0.25    # handback threshold
    k_out: int = 5             # consecutive steps below delta_out
    d_max: float = 5.0         # max takeover duration (s)
    # ---- ground truth (R2) ------------------------------------------------
    s_exec: float = 0.5
    s_mon: float = 0.5
    tier: str = "intermediate"
    operator_id: str = "op"

    def as_dict(self):
        return dataclasses.asdict(self)


# Tier anchors: doc Table 2 (execution) and Table 4 (monitoring) point
# values; expert == s=1, novice == s=0. theta_from_scale reads ONLY the
# ``expert`` and ``novice`` entries; the ``intermediate`` entry is
# REFERENCE-ONLY (the intermediate tier is auto-derived at s=0.5) and is kept
# in sync for readability.
#
# RECALIBRATED 2026-09-06, WIDENED 2026-09-07 to a realistic-but-separated
# skill range. The original expert (sigma0=0.25 mm/s, delta_in=0.45) was a
# super-human teleoperator whose near-perfect, hair-trigger interventions
# caused the policy to FREE-RIDE (operator finishes the hard insertion ->
# policy never learns it -> low autonomous SR). The 2026-09-06 pass set the
# expert to the old s=0.6 operator, which removed free-riding but OVER-
# COMPRESSED the novice->expert range (RQ1 operator spread too small at n=3:
# CV~0.09-0.29, ICC 0). The expert anchor is now the old s=0.75 operator:
# clearly better than the old intermediate yet still short of the free-riding
# super-human. novice is unchanged; the intermediate tier falls out as the
# midpoint of novice and the new expert.
EXEC_ANCHORS = dict(
    expert=dict(sigma0=0.629, k_sdn=0.03, sigma_tr=1.732, latency=1, lag=0.641,
                v_max=79.574, sigma_perc=0.217, p_pause=0.028, p_early=0.052,
                pause_len_mean=0.186, q_rec=0.17, p_lapse=0.06),
    intermediate=dict(sigma0=2.508, k_sdn=0.055, sigma_tr=3.948, latency=2,
                      lag=0.506, v_max=66.156, sigma_perc=0.691, p_pause=0.054, p_early=0.116,
                      pause_len_mean=0.472, q_rec=0.125, p_lapse=0.105),
    novice=dict(sigma0=10.0, k_sdn=0.10, sigma_tr=9.0, latency=4, lag=0.40,
                v_max=55.0, sigma_perc=2.2, p_pause=0.08, p_early=0.18,
                pause_len_mean=1.2, q_rec=0.08, p_lapse=0.15),
)
MON_ANCHORS = dict(
    expert=dict(sigma_mon=0.072, delta_in=0.538, m_in=2, l_react=4,
                p_miss=0.125, rho_fa=0.313, delta_out=0.2, k_out=4,
                d_max=3.834),
    intermediate=dict(sigma_mon=0.126, delta_in=0.669, m_in=3, l_react=6,
                      p_miss=0.237, rho_fa=0.613, delta_out=0.275, k_out=6,
                      d_max=5.538),
    novice=dict(sigma_mon=0.22, delta_in=0.80, m_in=4, l_react=8,
                p_miss=0.35, rho_fa=1.2, delta_out=0.35, k_out=8,
                d_max=8.0),
)

# interpolation classes (doc S3.5): geometric keeps scale-like params
# positive and makes equal steps in s equal RATIOS; probabilities and
# criterion placements interpolate linearly; integers are rounded.
_GEOMETRIC = {"sigma0", "k_sdn", "sigma_tr", "v_max", "sigma_perc",
              "pause_len_mean", "lag", "sigma_mon", "rho_fa", "d_max",
              "latency", "l_react"}
_LINEAR = {"p_pause", "q_rec", "p_lapse", "p_early", "p_miss", "delta_in",
           "delta_out", "m_in", "k_out"}
_INTEGER = {"latency", "m_in", "l_react", "k_out"}
_SCALE_JITTERED = {"sigma0", "k_sdn", "sigma_tr", "sigma_perc",
                   "sigma_mon", "rho_fa", "pause_len_mean"}

TIER_S = dict(novice=0.0, intermediate=0.5, expert=1.0)


def _interp(name, lo, hi, s):
    if name in _GEOMETRIC:
        v = (lo ** (1.0 - s)) * (hi ** s)
    else:
        v = (1.0 - s) * lo + s * hi
    if name in _INTEGER:
        v = int(round(v))
    return v


def theta_from_scale(s_exec: float, s_mon: Optional[float] = None
                     ) -> OperatorTheta:
    """Continuous skill scale theta(s) (doc S3.5): s=0 at the novice
    anchor, s=1 at the expert anchor; geometric interpolation for
    scale-like parameters, linear for probabilities/criteria."""
    if s_mon is None:
        s_mon = s_exec
    kw = {}
    for k in EXEC_ANCHORS["novice"]:
        kw[k] = _interp(k, EXEC_ANCHORS["novice"][k],
                        EXEC_ANCHORS["expert"][k], float(s_exec))
    for k in MON_ANCHORS["novice"]:
        kw[k] = _interp(k, MON_ANCHORS["novice"][k],
                        MON_ANCHORS["expert"][k], float(s_mon))
    tier = min(TIER_S, key=lambda t: abs(TIER_S[t] - s_exec))
    return OperatorTheta(s_exec=float(s_exec), s_mon=float(s_mon),
                         tier=tier, **kw)


def jitter_theta(theta: OperatorTheta, rng: np.random.Generator,
                 strength: float = JITTER_STD,
                 keys: Optional[set] = None) -> OperatorTheta:
    """Per-operator jitter (doc S3.5): multiply each scale-like parameter
    by exp(N(0, strength^2)) so tiers become overlapping clouds rather
    than three identical robots. ``keys`` restricts jitter to a subset of
    the scale-like parameters (e.g. execution-only, to keep a unified
    monitoring signal identical across operators).

    The keys are visited in sorted order: iterating the set directly follows
    Python's per-process string hashing (PYTHONHASHSEED), so the same rng
    draws landed on different parameters in every process."""
    jkeys = _SCALE_JITTERED if keys is None else (_SCALE_JITTERED & set(keys))
    kw = theta.as_dict()
    for k in sorted(jkeys):
        kw[k] = kw[k] * float(np.exp(rng.normal(0.0, strength)))
    return OperatorTheta(**kw)


def make_operator_theta(profile: str, rng: Optional[np.random.Generator]
                        = None, jitter: bool = False,
                        operator_id: Optional[str] = None) -> OperatorTheta:
    """Build a theta from a profile string. Accepted forms:
        "expert" | "intermediate" | "novice"           (both axes)
        "s=0.7" | "s_exec=0.9,s_mon=0.2"               (continuous)
        "exec=expert,mon=novice"                        (mixed tiers)
        "op1".."op4"                                    (Exp-1 set, doc
                                                         Table 4 of the
                                                         assessment doc)
    op4 is the skilled-but-jumpy profile: expert hands, trigger-happy
    supervision (low delta_in, high rho_fa, short reaction)."""
    rng = rng or np.random.default_rng(0)
    p = profile.strip().lower()
    if p in ("op1", "op2", "op3"):
        tier = {"op1": "expert", "op2": "intermediate", "op3": "novice"}[p]
        th = theta_from_scale(TIER_S[tier], TIER_S[tier])
    elif p == "op4":  # skilled-but-jumpy
        th = theta_from_scale(TIER_S["expert"], TIER_S["expert"])
        th = dataclasses.replace(
            th, delta_in=0.35, rho_fa=1.4, p_miss=0.05, l_react=3,
            delta_out=0.30, k_out=8, d_max=6.0, s_mon=0.15,
            sigma_mon=0.15)
    elif p in TIER_S:
        th = theta_from_scale(TIER_S[p], TIER_S[p])
    elif "=" in p:
        kv = dict(item.split("=") for item in p.split(","))
        if "exec" in kv or "mon" in kv:
            se = TIER_S[kv.get("exec", "intermediate")]
            sm = TIER_S[kv.get("mon", kv.get("exec", "intermediate"))]
        else:
            se = float(kv.get("s_exec", kv.get("s", 0.5)))
            sm = float(kv.get("s_mon", kv.get("s", se)))
        th = theta_from_scale(se, sm)
    else:
        raise ValueError(f"unknown operator profile '{profile}'")
    if jitter:
        th = jitter_theta(th, rng)
    th.tier = (p if p.startswith("op") else th.tier)
    th.operator_id = operator_id or p
    return th


def theta_to_json(theta: OperatorTheta, path: str):
    """Write an operator theta to JSON so runs can be reproduced exactly."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(theta.as_dict(), f, indent=2, sort_keys=True)
        f.write("\n")


def theta_from_json(path: str) -> OperatorTheta:
    """Load an exact operator theta written by theta_to_json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"theta JSON must be an object: {path}")
    valid = {field.name for field in dataclasses.fields(OperatorTheta)}
    unknown = sorted(set(data.keys()) - valid)
    if unknown:
        raise ValueError(
            f"unknown theta fields in {path}: {', '.join(unknown)}")
    return OperatorTheta(**data)


# ===========================================================================
# Oracle interfaces: the competent core pi_or the model degrades (doc S4)
# ===========================================================================
class PhaseOracle:
    """Task-primitive oracle (doc S4 Option A): exposes the geometric
    primitives the operator's generic phase machine consumes. Implement
    per environment (the toy env ships one; a real cell would supply the
    known hole pose)."""

    lateral_dims = (0,)      # action dims commanding lateral motion
    vertical_dim = 1         # action dim commanding descent (+ = up)

    def reset(self):
        pass

    def lateral_error(self, obs) -> np.ndarray:
        """True lateral error of the tool w.r.t. the insertion axis (mm);
        the operator perceives this plus its per-approach bias."""
        raise NotImplementedError

    def height(self, obs) -> float:
        """Tool height above the insertion surface (mm)."""
        raise NotImplementedError

    def in_contact(self, obs) -> bool:
        raise NotImplementedError

    def jammed(self, obs) -> bool:
        """Contact without insertion progress (drives q_rec recovery)."""
        raise NotImplementedError

    def commit_threshold(self) -> float:
        """Perceived |error| below which the operator commits to
        descend (mm); ~0.8 x clearance is a sensible default."""
        raise NotImplementedError

    def approach_height(self) -> float:
        return 8.0

    def action_dim(self) -> int:
        return 2


class PolicyOracle:
    """Doc S4 Option B: a trained policy as the competent core; the
    decision block degrades it with temporally correlated wrong-intent
    segments (doc S3.2) instead of the phase machine. `policy_fn(obs) ->
    action (env units)`; `action_scale` converts env action units to
    mm/s so the motor-channel parameters keep their table meaning."""

    def __init__(self, policy_fn, action_dim: int, action_scale: float = 1.0):
        self.policy_fn = policy_fn
        self._action_dim = action_dim
        self.action_scale = action_scale

    def reset(self):
        pass

    def action_dim(self) -> int:
        return self._action_dim

    def intent(self, obs) -> np.ndarray:
        return np.asarray(self.policy_fn(obs), dtype=np.float64) \
            * self.action_scale


# ===========================================================================
# the operator itself: perception -> decision -> motor (doc S3)
# ===========================================================================
class _OU:
    """Ornstein-Uhlenbeck tremor: band-limited, temporally correlated
    wobble (white noise looks wrong in trajectories and smoothness
    metrics, doc S3.3)."""

    def __init__(self, dim, sigma, rng, tau=0.3, dt=1.0 / CONTROL_HZ):
        self.x = np.zeros(dim)
        self.sigma, self.tau, self.dt, self.rng = sigma, tau, dt, rng

    def step(self):
        a = self.dt / self.tau
        self.x = (1 - a) * self.x + self.sigma * math.sqrt(2 * a) \
            * self.rng.standard_normal(self.x.shape)
        return self.x


class SyntheticOperator:
    """One operator = one frozen theta + one seeded RNG (R6), usable for
    Role 1 (full-episode teleoperation via `act`) and as the correction
    engine of Role 2 (operator_gate.SyntheticInterventionGate).

    Perception (doc S3.1): acts on the state delayed by `latency` steps;
    the alignment estimate carries a per-APPROACH bias eta ~
    N(0, sigma_perc^2), resampled only when the scene is re-examined
    (after a retract) - never per step, which would average the
    mis-estimate away and erase tier separation (the documented
    pitfall).

    Decision (doc S3.2): with a PhaseOracle, a phase machine
    approach -> align -> descend, committing when the PERCEIVED |error|
    is inside the threshold; outright mistakes with p_miss (premature
    commit); after a jam, retract-and-realign with prob q_rec per step
    (else push through). With a PolicyOracle, wrong-intent segments of
    0.5-1.5 s (frozen / scaled / rotated intent) at rate p_miss,
    preferentially triggered at contact.

    Motor (doc S3.3):  a_t = (1-lag)a_{t-1} + lag (u + eps + tremor),
    eps ~ N(0, (sigma0 + k_sdn ||u||)^2), ||a|| <= v_max, geometric
    pauses with prob p_pause, whole-episode lapses with prob p_lapse
    inflating all noise x2.5.
    """

    RETRACT_SPEED = 40.0  # mm/s used by the recovery macro

    def __init__(self, oracle, theta: OperatorTheta,
                 rng: Optional[np.random.Generator] = None,
                 action_scale: float = 1.0, dt: float = 1.0 / CONTROL_HZ):
        self.oracle = oracle
        self.th = theta
        self.rng = rng or np.random.default_rng(0)
        self.action_scale = action_scale  # env units per mm/s
        self.dt = dt
        self.phase_based = isinstance(oracle, PhaseOracle)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        th = self.th
        self.oracle.reset()
        self.lapse = self.rng.random() < th.p_lapse
        self._noise_mult = LAPSE_NOISE_MULT if self.lapse else 1.0
        self.bias = self._draw_bias()
        self._delay = deque(maxlen=max(th.latency, 0) + 1)
        self.a_prev = np.zeros(self.oracle.action_dim())
        self.ou = _OU(self.oracle.action_dim(),
                      th.sigma_tr * self._noise_mult, self.rng, dt=self.dt)
        self.pause_left = 0
        self.phase = "approach"
        self.committed = False
        self.n_approaches = 1
        self._premature = self.rng.random() < th.p_early
        self._premature_after = int(self.rng.uniform(0.2, 0.7) / self.dt)
        self._align_steps = 0
        self._ehat_s = None
        self.commit_events = []   # (perceived_err, true_err) at commits
        # PolicyOracle wrong-intent machinery
        self._wrong_left = 0
        self._wrong_mode = None

    def _draw_bias(self):
        dim = len(getattr(self.oracle, "lateral_dims", (0,))) \
            if self.phase_based else 1
        return self.rng.normal(0.0, self.th.sigma_perc, size=dim)

    # ------------------------------------------------------------------
    def _perceive(self, obs):
        """Latency: act on the observation from `latency` steps ago."""
        self._delay.append(obs)
        return self._delay[0]

    # ---- decision block ----------------------------------------------
    def _intent_phase(self, obs_d):
        th, orc = self.th, self.oracle
        u = np.zeros(orc.action_dim())
        err_true = np.asarray(orc.lateral_error(obs_d), dtype=np.float64)
        err_hat = err_true + self.bias           # per-approach mis-estimate
        h = orc.height(obs_d)
        appr_h = orc.approach_height()

        if self.phase == "retract":
            u[orc.vertical_dim] = self.RETRACT_SPEED
            if h >= appr_h * 0.9:
                # scene re-examined: resample the bias (doc S3.1)
                self.bias = self._draw_bias()
                self.n_approaches += 1
                self.phase, self.committed = "align", False
                self._premature = self.rng.random() < th.p_early
                self._premature_after = int(self.rng.uniform(0.2, 0.7)
                                            / self.dt)
                self._align_steps = 0
                self._ehat_s = None
            return u

        if orc.jammed(obs_d):
            if self.rng.random() < th.q_rec:
                self.phase = "retract"           # expert-style recovery
                u[orc.vertical_dim] = self.RETRACT_SPEED
                return u
            # push through (novice-style): keep pressing, wiggle a bit
            u[orc.vertical_dim] = -0.6 * th.v_max
            for i, d in enumerate(orc.lateral_dims):
                u[d] = np.clip(-4.0 * err_hat[i], -th.v_max, th.v_max)
            return u

        if self.phase == "approach":
            u[orc.vertical_dim] = -th.v_max if h > appr_h else 0.0
            for i, d in enumerate(orc.lateral_dims):
                u[d] = np.clip(-3.0 * err_hat[i], -th.v_max, th.v_max)
            if h <= appr_h * 1.1:
                self.phase = "align"
            return u

        if self.phase == "align":
            for i, d in enumerate(orc.lateral_dims):
                u[d] = np.clip(-4.0 * err_hat[i], -th.v_max, th.v_max)
            u[orc.vertical_dim] = -0.15 * th.v_max  # slow creep down
            self._align_steps += 1
            # commit on a lightly smoothed percept (one clean reading of
            # the smoothed estimate, not a raw one-sample spike)
            prev_s = self._ehat_s if self._ehat_s is not None else err_hat
            self._ehat_s = 0.5 * prev_s + 0.5 * err_hat
            aligned = (float(np.linalg.norm(self._ehat_s))
                       < orc.commit_threshold())
            premature = (self._premature
                         and self._align_steps >= self._premature_after)
            if aligned or premature:
                self.phase, self.committed = "descend", True
                self.commit_events.append(
                    (float(np.linalg.norm(err_hat)),
                     float(np.linalg.norm(err_true))))
            return u

        # descend / insert
        u[orc.vertical_dim] = -th.v_max
        for i, d in enumerate(orc.lateral_dims):
            u[d] = np.clip(-2.0 * err_hat[i], -th.v_max, th.v_max)
        return u

    def _intent_policy(self, obs_d, contact=False):
        u = self.oracle.intent(obs_d)
        th = self.th
        if self._wrong_left <= 0:
            p = th.p_miss * self.dt * 10 * (3.0 if contact else 1.0)
            if self.rng.random() < p:
                self._wrong_left = int(self.rng.uniform(0.5, 1.5) / self.dt)
                self._wrong_mode = self.rng.choice(["freeze", "scale",
                                                    "rotate"])
                self._frozen = u.copy()
        if self._wrong_left > 0:
            self._wrong_left -= 1
            if self._wrong_mode == "freeze":
                u = self._frozen
            elif self._wrong_mode == "scale":
                u = 0.2 * u
            else:  # rotate the first two dims
                c, s = math.cos(0.8), math.sin(0.8)
                u = u.copy()
                if len(u) >= 2:
                    u[0], u[1] = c * u[0] - s * u[1], s * u[0] + c * u[1]
        return u

    # ---- motor block --------------------------------------------------
    def _motor(self, u):
        th = self.th
        if self.pause_left > 0:
            self.pause_left -= 1
            u = np.zeros_like(u)
        elif self.rng.random() < th.p_pause:
            self.pause_left = max(1, int(self.rng.geometric(
                min(self.dt / max(th.pause_len_mean, 1e-3), 0.999))))
            u = np.zeros_like(u)
        sd = (th.sigma0 + th.k_sdn * float(np.linalg.norm(u))) \
            * self._noise_mult
        eps = self.rng.normal(0.0, sd, size=u.shape)
        a = (1 - th.lag) * self.a_prev + th.lag * (u + eps + self.ou.step())
        n = float(np.linalg.norm(a))
        if n > th.v_max:
            a = a * (th.v_max / n)
        self.a_prev = a
        return a

    # ------------------------------------------------------------------
    def begin_takeover(self, obs):
        """Role-2 entry (doc S6.3): called by the intervention gate at
        takeover onset. The operator has been WATCHING (latency applies
        to their reaction via the gate l_react), so the delay queue is
        seeded with the current scene; they take a fresh look (bias
        resampled) and run the recovery macro stabilize -> retract ->
        re-align -> re-attempt through their own motor model -
        push-through instead of retract emerges from low q_rec exactly
        as in solo operation."""
        self._delay.clear()
        for _ in range(self._delay.maxlen):
            self._delay.append(obs)
        self.bias = self._draw_bias()
        self.n_approaches += 1
        self.a_prev = np.zeros(self.oracle.action_dim())
        self._ehat_s = None
        self._align_steps = 0
        self.committed = False
        if self.phase_based:
            in_trouble = (self.oracle.jammed(obs)
                          or self.oracle.in_contact(obs))
            self.phase = "retract" if in_trouble else "align"

    def act(self, obs, contact: bool = False) -> np.ndarray:
        """One control step: returns the executed action in ENV units."""
        obs_d = self._perceive(obs)
        if self.phase_based:
            u = self._intent_phase(obs_d)
        else:
            u = self._intent_policy(obs_d, contact=contact)
        return self._motor(u) * self.action_scale
