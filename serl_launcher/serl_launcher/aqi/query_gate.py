"""Active intervention query gate (S5.2), composed from independent term
sections.

Each trigger term is its own code section elsewhere in this package
(uncertainty.py, failure.py, force_risk.py, ood.py, the cost EMA here);
the gate only (a) z-normalizes the raw values of the terms selected in
``cfg.trigger_terms`` and (b) combines them with the logic selected in
``cfg.trigger_combiner``:

    weighted_sum (proposal S5.2, default):
        A(s_t) = sum_i lambda_i * z(term_i)  -  lambda_cost * z(C_human)
    max (OR-logic):
        A(s_t) = max_i lambda_i * z(term_i)  -  lambda_cost * z(C_human)
        i.e. request when ANY single weighted term exceeds tau_A.

    request_human(s_t) = 1  if A(s_t) > tau_A

The minimal configuration ``trigger_terms=("uncertainty",)`` gates on the
critic-ensemble epistemic uncertainty alone; no other section is computed.

Practical additions for a real control loop:
    * warmup: no requests until the z-statistics have seen enough samples;
    * hysteresis: once raised, a request stays on for `min_on` steps so the
      operator gets a stable cue;
    * cooldown: after a request window closes, no new request for
      `cooldown` steps (part of C_human's role of limiting workload).
Hard force/torque violations ALWAYS bypass A(s_t) (safety override, reason
"safety"), independent of which terms are selected: the hard-limit check
is a safety mechanism, not a trigger term.

C_human is an exponential moving average of the intervention indicator
(recent human workload).
"""

import numpy as np

from collections import deque

from serl_launcher.aqi.normalization import RunningZScore

# legacy flat log keys per term: (raw_key, z_key)
_TERM_KEYS = {
    "uncertainty": ("sigma_q", "z_uq"),
    "failure": ("p_fail", "z_fail"),
    "force": ("r_force", "z_force"),
    "ood": ("ood", "z_ood"),
    "cost": ("c_human", "z_cost"),
}


class ActiveQueryGate:
    def __init__(self, cfg):
        """cfg: AQIConfig."""
        self.cfg = cfg
        self.terms = tuple(cfg.trigger_terms)
        self.lambdas = dict(
            uncertainty=cfg.lambda_uq, failure=cfg.lambda_fail,
            force=cfg.lambda_force, ood=cfg.lambda_ood, cost=cfg.lambda_cost)
        # per-active-term running z; 0 is meaningful for force (no contact
        # risk) so it is scaled but not centered.
        self.z = {t: RunningZScore(center=(t != "force")) for t in self.terms}
        self._cost_ema = 0.0
        self._cost_alpha = 1.0 / max(cfg.cost_ema_horizon, 1)
        self.total_steps = 0
        self._on_steps_left = 0
        self._cooldown_left = 0
        self.n_requests = 0
        self._rng = np.random.default_rng(int(getattr(cfg, "random_request_seed", 0)))
        # running window of A(s_t) behind the optional target-rate quantile
        self._score_window = deque(
            maxlen=int(getattr(cfg, "query_quantile_window", 2000)))

    # ------------------------------------------------------------------
    def update_cost(self, intervening: bool) -> None:
        """Track recent operator workload C_human(t)."""
        self._cost_ema = ((1 - self._cost_alpha) * self._cost_ema
                          + self._cost_alpha * float(intervening))

    @property
    def cost(self) -> float:
        return self._cost_ema

    # ------------------------------------------------------------------
    def _combine(self, weighted_z: dict) -> float:
        """Combination logic over the active terms' lambda*z values."""
        penalty = weighted_z.get("cost", 0.0)
        signals = [v for t, v in weighted_z.items() if t != "cost"]
        if self.cfg.trigger_combiner == "max":
            return (max(signals) if signals else 0.0) - penalty
        return sum(signals) - penalty          # "weighted_sum"

    def step(self, terms: dict, hard_violation: bool) -> dict:
        """One control step: z-update the active terms, combine, decide.

        Args:
            terms: raw values for the ACTIVE non-cost terms, keyed by
                section name (e.g. {"uncertainty": sigma_q}). The cost
                term is supplied internally from the workload EMA.
            hard_violation: hard force/torque limit breached (safety
                override; independent of trigger_terms).

        Returns a flat record with the raw terms, z-terms, a_score,
        request flag and reason ("score" | "lcb" | "safety" | "").
        Keys of inactive terms are absent so downstream consumers can
        distinguish "not computed" from a value.
        """
        self.total_steps += 1
        raw = dict(terms)
        if "cost" in self.terms:
            raw["cost"] = self._cost_ema

        weighted_z, record = {}, {}
        for name in self.terms:
            if name not in raw or raw[name] is None:
                continue
            z = self.z[name].z(float(raw[name]))
            weighted_z[name] = self.lambdas[name] * z
            raw_key, z_key = _TERM_KEYS[name]
            record[raw_key] = float(raw[name])
            record[z_key] = float(z)
        a_score = self._combine(weighted_z)
        self._score_window.append(float(a_score))

        request = False
        reason = ""
        tau_eff = float(self.cfg.tau_A)
        if hard_violation:
            request = True
            reason = "safety"
            self._on_steps_left = max(self._on_steps_left,
                                      self.cfg.query_min_on_steps)
        elif self._on_steps_left > 0:
            request = True
            reason = "score"
            self._on_steps_left -= 1
            if self._on_steps_left == 0:
                self._cooldown_left = self.cfg.query_cooldown_steps
        elif (self.total_steps > self.cfg.query_warmup_steps
              and self._cooldown_left == 0):
            # explicit target-rate quantile overrides the fixed tau_A: request
            # the top `query_target_rate` fraction of recent scores (scale-free
            # rate control), else fall back to the absolute threshold tau_A.
            target = getattr(self.cfg, "query_target_rate", None)
            if target is not None and len(self._score_window) >= 10:
                tau_eff = float(np.quantile(self._score_window, 1.0 - target))
            above = a_score > tau_eff
            lcb_trip = (self.cfg.lcb_request_threshold is not None
                        and hasattr(self, "_last_lcb")
                        and self._last_lcb < self.cfg.lcb_request_threshold)
            rand_trip = (self.cfg.random_request_rate > 0.0
                         and self._rng.random() < self.cfg.random_request_rate)
            if above or lcb_trip or rand_trip:
                request = True
                reason = "score" if above else ("lcb" if lcb_trip else "random")
                self.n_requests += 1
                self._on_steps_left = self.cfg.query_min_on_steps - 1
        if self._cooldown_left > 0 and not request:
            self._cooldown_left -= 1

        record.update(a_score=float(a_score), tau_eff=float(tau_eff),
                      request=request, reason=reason,
                      hard_violation=bool(hard_violation))
        return record

    def set_lcb(self, lcb_value: float) -> None:
        self._last_lcb = lcb_value

    def state_dict(self) -> dict:
        return dict(
            total_steps=self.total_steps, n_requests=self.n_requests,
            trigger_terms=list(self.terms),
            trigger_combiner=self.cfg.trigger_combiner,
            z={name: zs.state_dict() for name, zs in self.z.items()},
            cost_ema=self._cost_ema,
        )
