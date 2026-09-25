"""Configuration for AQI-HIL-SERL (Active and Quality-Aware Intervention HIL-SERL).

Every field maps to an equation or design decision in the research proposal
"Active and Quality-Aware Human-in-the-Loop Reinforcement Learning for
Operator-Robust Robotic Insertion". Section references (S5.2, S5.6, ...) refer
to that document.

Modes (S6.4 baselines/ablations):
    "hilserl"  : plain HIL-SERL behaviour (gate off, quality off, logging off).
    "logged"   : baseline + full intervention/trigger logging (Phase 2).
    "active"   : active query gate A(s_t) on, replay/loss unchanged (Phase 3).
    "quality"  : manual interventions, quality-aware replay + loss (Phase 4).
    "aqi"      : active gate + quality-aware replay/loss (Phase 5, full system).
    "e2hil"    : EXTERNAL COMPARISON BASELINE, not part of AQI. Plain
                 HIL-SERL (hilserl's exact 2-critic architecture, gate off,
                 quality off, logging off) plus the entropy-bounded actor
                 loss of E2HiL (Deng et al., IEEE RA-L 2026, arXiv:2601.19969):
                 the actor update drops batch states whose estimated
                 influence on policy entropy |c(s)| lies outside the batch
                 [e2hil_q_low, e2hil_q_high] percentile band. See
                 serl_launcher/agents/continuous/e2hil.py.

Ablations from S6.4 are compositions, not code changes: `trigger_terms`
selects which term sections feed A(s_t) (("uncertainty",) = ensemble
epistemic uncertainty alone) and `trigger_combiner` the combining logic;
`quality_disagreement_gate` and `quality_filters` select the layers around
the counterfactual-advantage core (gate off + no filters = q is A_hat
alone). Zeroing individual lambdas remains equivalent for the
weighted-sum combiner; mode="active" == "no q_int".
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

MODES = ("hilserl", "logged", "active", "quality", "aqi", "e2hil")

# Modes that train the plain 2-critic HIL-SERL architecture (no ensemble,
# no bootstrap, no AQI runtime). Checkpoints are interchangeable between them.
BASELINE_ARCH_MODES = ("hilserl", "e2hil")

# E2HiL: how the critic ensemble is reduced to one Q in the actor objective
# and in the influence estimate. "min" matches the official E2HiL code
# (cov_actor_loss.py); "mean" matches this repo's hilserl actor loss, so with
# it the entropy-bounded mask is the ONLY difference from hilserl.
E2HIL_Q_AGGS = ("min", "mean")

# Composable trigger sections (S5.2-S5.4). Each name is an independent code
# section; `trigger_terms` selects which ones feed A(s_t):
#   "uncertainty" : critic-ensemble epistemic uncertainty sigma_Q   (S5.3)
#   "failure"     : online logistic P_fail                          (S5.4)
#   "force"       : force/torque contact risk R_force               (S5.4)
#   "ood"         : k-NN novelty vs D_success                       (S5.4)
#   "cost"        : human workload penalty C_human (subtracts)      (S5.2) [OPTIONAL]
# The core minimal trigger is ("uncertainty",): ensemble alone.
# Include "cost" to penalize requests when the human is already overloaded.
# Exclude "cost" for trigger_terms that focus only on task risk.
TRIGGER_TERMS = ("uncertainty", "failure", "force", "ood", "cost")

# How the selected z-scored terms combine into A(s_t):
#   "weighted_sum" : A = sum(lambda_i * z_i) - lambda_cost * z_cost (proposal)
#   "max"          : A = max_i(lambda_i * z_i) - lambda_cost * z_cost, i.e.
#                    OR-logic: request when ANY single term exceeds tau_A.
TRIGGER_COMBINERS = ("weighted_sum", "max")

# Composable quality filters (S5.5 items 2-3). The core graded signal is
# ALWAYS the counterfactual advantage A_hat (Option B); these named
# sections are binary keep/drop layers applied around it, each removable:
#   "safety_veto"    : hard force/torque limit breached in segment
#   "jerk_veto"      : action jerk above jerk_hard (no-op if jerk_hard=None)
#   "outcome_filter" : Option G, drop segments from failed episodes
#   "unsolicited"    : drop non-requested takeovers (only if
#                      unsolicited_policy == "rl_only" and gate active)
QUALITY_FILTERS = ("safety_veto", "jerk_veto", "outcome_filter", "unsolicited")


@dataclass
class AQIConfig:
    # ------------------------------------------------------------------
    # Global
    # ------------------------------------------------------------------
    mode: str = "aqi"

    # ------------------------------------------------------------------
    # Critic ensemble (S5.3). K critics; subsample for the TD target keeps
    # the conservative min-Q backup (REDQ-style) while sigma is computed
    # over the full ensemble.
    # ------------------------------------------------------------------
    critic_ensemble_size: int = 10
    critic_subsample_size: Optional[int] = 2
    lcb_beta: float = 1.0                 # LCB_Q = mu_Q - beta * sigma_Q

    # ------------------------------------------------------------------
    # Active query function A(s_t) (S5.2):
    # A = l1*z(U_Q) + l2*z(P_fail) + l3*z(R_force) + l4*z(OOD) - l5*z(C_human)
    # request_human(s_t) = 1 if A(s_t) > tau_A
    # ------------------------------------------------------------------
    lambda_uq: float = 1.0
    lambda_fail: float = 1.0
    lambda_force: float = 1.0
    lambda_ood: float = 0.5
    lambda_cost: float = 0.5
    tau_A: float = 0.5                    # threshold on the z-combined score

    # Which term sections feed A(s_t) and how they combine (see module
    # docstring). ("uncertainty",) gives the pure ensemble-epistemic
    # trigger; inactive sections are not computed at all (their raw values
    # are absent from the step logs). Add "cost" to include the human
    # workload penalty (subtracts from A to avoid overloading the operator).
    trigger_terms: Tuple[str, ...] = ("uncertainty", "failure", "force", "ood")
    trigger_combiner: str = "weighted_sum"

    # Bootstrapped ensemble training (S5.3). The K critics must see
    # DIFFERENT data for their disagreement sigma_Q to be an epistemic
    # signal: critic_bootstrap_p is the Bernoulli keep-probability of the
    # persistent per-(transition, member) masks stored in the replay
    # buffers (1.0 = shared data, the deep-ensembles init-only ablation),
    # and critic_per_member_targets gives each member its own independent
    # min-subsample target draw instead of the single shared REDQ target
    # (without subsampling: its own target head, bootstrapped-DQN style).
    # hilserl mode ignores both (exact 2-critic baseline).
    critic_bootstrap_p: float = 0.8
    critic_per_member_targets: bool = True

    query_warmup_steps: int = 1000        # collect z-stats before triggering
    query_min_on_steps: int = 10          # hysteresis: keep request raised
    query_cooldown_steps: int = 20        # steps between distinct requests
    lcb_request_threshold: Optional[float] = None  # optional: request if LCB < thr

    # ------------------------------------------------------------------
    # Force / torque risk (S5.4):
    # R_force = max(0,|F|-F_warn) + rho*max(0,|tau|-tau_warn) + kappa*impulse_H
    # Hard safety limits act as an override for the gate AND as the hard
    # safety veto for the quality module (S5.5 "decision and resulting
    # design", item 3: force/jerk are safety filters, not quality terms).
    # ------------------------------------------------------------------
    force_warn: float = 15.0              # [N]
    torque_warn: float = 1.5              # [Nm]
    force_hard: float = 30.0              # [N]  hard veto / safety override
    torque_hard: float = 3.0              # [Nm] hard veto / safety override
    force_rho: float = 1.0                # weight on torque excess
    force_kappa: float = 0.1              # weight on windowed impulse
    impulse_horizon: int = 10             # H steps for the impulse window
    jerk_hard: Optional[float] = None     # optional hard veto on action jerk
    # Observation keys holding force/torque inside the SERL proprio state.
    force_obs_key: str = "tcp_force"
    torque_obs_key: str = "tcp_torque"

    # ------------------------------------------------------------------
    # OOD / novelty (S5.4): mean k-NN distance to D_success in phi(s).
    # phi(s) = per-dim standardized proprio state by default.
    # ------------------------------------------------------------------
    ood_k: int = 10
    ood_ref_capacity: int = 20000
    ood_min_ref: int = 200                # score 0 until this many refs
    ood_subsample_success: int = 2        # keep every n-th step of a success

    # ------------------------------------------------------------------
    # Failure probability P_fail (S5.4): online logistic model over trigger
    # features, labels = failure event within `fail_horizon` steps.
    # ------------------------------------------------------------------
    fail_horizon: int = 20
    fail_lr: float = 0.05
    fail_l2: float = 1e-4
    fail_min_episodes: int = 5            # predictions used only after this

    # ------------------------------------------------------------------
    # Human cost C_human (S5.2): penalizes recent intervention frequency
    # and duration (EMA of intervention indicator).
    # ------------------------------------------------------------------
    cost_ema_horizon: int = 200

    # ------------------------------------------------------------------
    # Intervention quality (S5.5, Option B primary):
    #   A_hat(s,a_H)   = mu_Q(s,a_H) - mu_Q(s,a_robot)
    #   q_signed       = A_hat * g(sigma_Q)
    #   q_int          = clip(q_signed, 0, inf)          (priority, S5.6)
    #   w_i            = 0                    if A_hat < adv_exclude
    #                    clip(sigmoid(gamma * q_signed / scale), wmin, wmax)
    # g(sigma) = 1 / (1 + (sigma/sigma_ref)^2) suppresses the signal where
    # the ensemble disagrees; sigma_ref tracks the running mean of sigma_Q
    # unless overridden.
    # ------------------------------------------------------------------
    quality_sigma_ref: Optional[float] = None
    quality_gamma: float = 2.0
    quality_w_min: float = 0.05
    quality_w_max: float = 1.0
    adv_exclude: float = 0.0              # A_hat below this -> excluded from BC
    quality_warmup_steps: int = 0         # interventions get BASELINE replay
                                          # priority (q_int->0 contribution, and
                                          # zero BC weight) until this many env
                                          # steps, so the immature critic
                                          # ensemble's noisy q does not drive
                                          # replay/loss early. Raw q_int is still
                                          # logged. 0 = no warm-up (legacy).
    outcome_filter: bool = True           # Option G: drop failed-episode segments
    demo_weight: float = 0.0              # w for pre-recorded demos in BC loss

    # Quality composition: the graded signal is always the counterfactual
    # advantage A_hat. The disagreement gate g(sigma_Q) is an optional
    # multiplicative modifier; `quality_filters` selects the binary
    # keep/drop sections (see QUALITY_FILTERS). Pure counterfactual-only
    # quality: quality_disagreement_gate=False, quality_filters=().
    # (adv_exclude's hard-zero is derived from A_hat itself and therefore
    # stays with the core signal; disable via adv_exclude=-inf.)
    quality_disagreement_gate: bool = True
    quality_filters: Tuple[str, ...] = QUALITY_FILTERS

    # Scoring mode for the graded intervention signal:
    #   "counterfactual" (default) = A_hat = mu_Q(a_H) - mu_Q(a_robot) (S5.5).
    #   "behavioral" = a critic-FREE composite of the takeover's behaviour,
    #     z(d_value) - z(t_int) - z(f_peak): a short, value-increasing,
    #     force-controlled takeover scores high. Avoids the off-distribution
    #     critic evaluation of a_H that makes A_hat anti-aligned with outcome
    #     (diagnostic AUROC 0.72 vs 0.18). Standardised online per run.
    quality_score_mode: str = "counterfactual"

    # ------------------------------------------------------------------
    # Quality-aware replay (S5.6): p_i = (eps + |delta_i| + eta*max(q,0))^alpha
    # ------------------------------------------------------------------
    per_eps: float = 0.05
    per_eta: float = 1.0
    per_alpha: float = 0.6
    per_delta_init: float = 1.0           # optimistic init in place of |delta|
    use_td_priority: bool = False         # learner streams |delta| back if True
    per_standardize: bool = False         # divide q_int by its running scale
                                          # before the priority so `per_eta` is
                                          # dimension-free (q_int is a tiny Q-
                                          # advantage; raw it is swamped by
                                          # |delta|). Off = raw q_int (legacy).
    per_scale_decay: float = 0.0          # EMA decay for the per_standardize
                                          # scale (0 = cumulative, never-forget
                                          # mean; e.g. 0.01 = forgetting EMA that
                                          # keeps the scale stationary as the
                                          # critic drifts).
    per_refresh_period: int = 0           # recompute ALL stored priorities every
                                          # N inserts against the CURRENT scale
                                          # (0 = never). Needed with an EMA scale
                                          # so old transitions' priorities stay
                                          # consistent with new ones.
    per_is_beta: float = 0.0              # importance-sampling exponent for the
                                          # prioritised replay (0 = OFF = no IS
                                          # correction; ~0.4-1.0 corrects the
                                          # sampling bias the priority injects
                                          # into the critic target).

    # ------------------------------------------------------------------
    # Auxiliary intervention/imitation loss (S5.6):
    # L_actor = L_RL + lambda_bc * mean(w_i * L_intervention)
    # ------------------------------------------------------------------
    bc_loss_weight: float = 0.0

    # ------------------------------------------------------------------
    # Unsolicited interventions in "active"/"aqi" modes:
    #   "allow"   : treated exactly like requested ones (default; operators
    #               keep full authority, source is logged)
    #   "rl_only" : kept as RL data but excluded from the intervention buffer
    # ------------------------------------------------------------------
    unsolicited_policy: str = "allow"

    # ------------------------------------------------------------------
    # E2HiL comparison baseline (mode="e2hil" only; inert otherwise).
    # Per actor update, for every batch state s:
    #   c(s) = -Cov_k( log pi(a_k|s), pi(a_k|s) * A_soft(s,a_k) ),
    #          a_k ~ pi(.|s), k = 1..e2hil_K   (paper Eq. 9-10)
    #   I(s) = 1[ |c(s)| in [P_q_low, P_q_high] of the batch's |c| ]  (Eq. 11)
    #   L_actor = -sum_s I(s) l(s) / (sum_s I(s) + eps)               (Eq. 12)
    # Defaults are the paper's / official code's (K=8, 5th-90th pct).
    # e2hil_K=1 makes every covariance 0, i.e. no selection (as in the
    # official code).
    # ------------------------------------------------------------------
    e2hil_K: int = 8
    e2hil_q_low: float = 0.05
    e2hil_q_high: float = 0.90
    e2hil_q_agg: str = "min"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_dir: str = "aqi_logs"
    log_every_step: bool = True           # per-step trigger timeline
    log_proprio: bool = True              # store proprio (not images) per step
    log_actions: bool = True              # store a_robot / a_executed / a_h
                                          # per step (logging schema E.3)

    def __post_init__(self):
        assert self.mode in MODES, f"mode must be one of {MODES}, got {self.mode}"
        if self.critic_subsample_size is not None:
            assert self.critic_subsample_size <= self.critic_ensemble_size
        self.trigger_terms = tuple(self.trigger_terms)
        self.quality_filters = tuple(self.quality_filters)
        bad = set(self.trigger_terms) - set(TRIGGER_TERMS)
        assert not bad, f"unknown trigger_terms {bad}; valid: {TRIGGER_TERMS}"
        assert self.trigger_combiner in TRIGGER_COMBINERS, (
            f"trigger_combiner must be one of {TRIGGER_COMBINERS}")
        bad = set(self.quality_filters) - set(QUALITY_FILTERS)
        assert not bad, f"unknown quality_filters {bad}; valid: {QUALITY_FILTERS}"
        assert 0.0 < self.critic_bootstrap_p <= 1.0, (
            "critic_bootstrap_p must be in (0, 1]")
        if self.active_on:
            assert set(self.trigger_terms) - {"cost"}, (
                "active gating needs at least one signal term besides 'cost'")
        assert int(self.e2hil_K) >= 1, "e2hil_K must be >= 1"
        assert 0.0 <= self.e2hil_q_low < self.e2hil_q_high <= 1.0, (
            "need 0 <= e2hil_q_low < e2hil_q_high <= 1")
        assert self.e2hil_q_agg in E2HIL_Q_AGGS, (
            f"e2hil_q_agg must be one of {E2HIL_Q_AGGS}")

    # Composition helpers ---------------------------------------------------
    def uses_term(self, name: str) -> bool:
        """Is this trigger section part of A(s_t)?"""
        return name in self.trigger_terms

    @property
    def needs_ood(self) -> bool:
        """OOD section runs if it feeds the gate or the P_fail features."""
        return self.uses_term("ood") or self.uses_term("failure")

    @property
    def needs_failure(self) -> bool:
        return self.uses_term("failure")

    @property
    def effective_quality_filters(self) -> Tuple[str, ...]:
        """quality_filters reconciled with the legacy switches: the
        `outcome_filter` bool and `unsolicited_policy` string can each
        disable their section without touching the tuple."""
        out = []
        for name in self.quality_filters:
            if name == "outcome_filter" and not self.outcome_filter:
                continue
            if name == "unsolicited" and self.unsolicited_policy != "rl_only":
                continue
            out.append(name)
        return tuple(out)

    # Convenience switches -------------------------------------------------
    @property
    def active_on(self) -> bool:
        return self.mode in ("active", "aqi")

    @property
    def quality_on(self) -> bool:
        return self.mode in ("quality", "aqi")

    @property
    def logging_on(self) -> bool:
        return not self.baseline_arch

    @property
    def baseline_arch(self) -> bool:
        """Plain 2-critic HIL-SERL architecture (hilserl, e2hil)."""
        return self.mode in BASELINE_ARCH_MODES

    @property
    def e2hil_on(self) -> bool:
        return self.mode == "e2hil"

    def trigger_lambdas(self) -> Tuple[float, float, float, float, float]:
        return (
            self.lambda_uq,
            self.lambda_fail,
            self.lambda_force,
            self.lambda_ood,
            self.lambda_cost,
        )
