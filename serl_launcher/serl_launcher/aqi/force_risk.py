"""Force/torque contact-risk score and hard safety limits (S5.4, S5.5).

    R_force(t) = max(0, |F_t| - F_warn)
               + rho * max(0, |tau_t| - tau_warn)
               + kappa * impulse_H

impulse_H is a discrete windowed force impulse, sum over the last H steps of
|F_t| * dt (dt folded into kappa, so impulse_H = mean-free running sum of
|F_t| over the window; we use the sum of the above-warn excess to keep the
term zero during nominal contact).

Hard limits (force_hard / torque_hard) are *not* part of the quality score.
They serve two roles (proposal S5.5, "Decision and resulting design", (3)):
    1. safety override for the active gate: request help immediately;
    2. hard safety veto for the quality module: the intervention segment is
       excluded from imitation regardless of its counterfactual advantage.
"""

from collections import deque

import numpy as np


class ForceRiskMonitor:
    def __init__(self, force_warn: float, torque_warn: float,
                 force_hard: float, torque_hard: float,
                 rho: float = 1.0, kappa: float = 0.1,
                 impulse_horizon: int = 10):
        self.force_warn = force_warn
        self.torque_warn = torque_warn
        self.force_hard = force_hard
        self.torque_hard = torque_hard
        self.rho = rho
        self.kappa = kappa
        self._window = deque(maxlen=max(int(impulse_horizon), 1))

    def reset(self) -> None:
        self._window.clear()

    def update(self, force_vec, torque_vec) -> dict:
        """Consume one step of F/T readings, return risk terms.

        Args:
            force_vec:  iterable of 3 (or None if unavailable)
            torque_vec: iterable of 3 (or None if unavailable)
        Returns dict with keys:
            f_norm, tau_norm, r_force, hard_violation (bool)
        """
        if force_vec is None:
            return dict(f_norm=0.0, tau_norm=0.0, r_force=0.0,
                        hard_violation=False)
        f = float(np.linalg.norm(np.asarray(force_vec, dtype=np.float64)))
        tau = 0.0
        if torque_vec is not None:
            tau = float(np.linalg.norm(np.asarray(torque_vec,
                                                  dtype=np.float64)))
        f_excess = max(0.0, f - self.force_warn)
        tau_excess = max(0.0, tau - self.torque_warn)
        self._window.append(f_excess)
        impulse = float(np.sum(self._window))
        r = f_excess + self.rho * tau_excess + self.kappa * impulse
        hard = (f > self.force_hard) or (tau > self.torque_hard)
        return dict(f_norm=f, tau_norm=tau, r_force=r, hard_violation=hard)


def action_jerk(actions: np.ndarray) -> np.ndarray:
    """Second-order difference of commanded actions (S5.5, Option A table).

    actions: (T, A). Returns per-step jerk magnitude (T,), zero-padded at
    the start. Used for logging / the Option-A baseline score and for the
    optional jerk hard veto; NOT part of the primary quality signal.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions[:, None]
    t = actions.shape[0]
    jerk = np.zeros(t)
    if t >= 3:
        dd = actions[2:] - 2 * actions[1:-1] + actions[:-2]
        jerk[2:] = np.linalg.norm(dd, axis=-1)
    return jerk
