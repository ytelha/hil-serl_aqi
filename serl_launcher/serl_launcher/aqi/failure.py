"""Predicted failure probability P_fail (S5.4).

    P_fail(s_t, a_t) = P( jamming, high-force violation, timeout, or
                          insertion failure within horizon H | s_t, a_t )

Implemented as a small online logistic-regression model over the (already
scale-free) trigger features. Labels are constructed retroactively at
episode end: step t is positive iff a failure event occurs in (t, t+H]:
    - a hard force/torque violation, or
    - the episode terminates unsuccessfully (timeout / failed insertion)
      within H steps of t.
The model is refit incrementally (SGD) on each finished episode. Its output
enters the acquisition function only after `min_episodes` episodes so a
cold, uncalibrated head cannot drive queries.

Features (fixed order):
    [ z(sigma_Q), z(R_force), z(OOD), z(f_norm), episode_progress ]
"""

import numpy as np

FEATURE_NAMES = ("z_sigma_q", "z_r_force", "z_ood", "z_f_norm", "ep_progress")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class FailureRiskModel:
    def __init__(self, horizon: int = 20, lr: float = 0.05,
                 l2: float = 1e-4, min_episodes: int = 5,
                 epochs_per_fit: int = 3):
        self.horizon = horizon
        self.lr = lr
        self.l2 = l2
        self.min_episodes = min_episodes
        self.epochs_per_fit = epochs_per_fit
        self.dim = len(FEATURE_NAMES)
        self.w = np.zeros(self.dim, dtype=np.float64)
        self.b = 0.0
        self.n_episodes = 0
        self.n_pos = 0
        self.n_neg = 0

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.n_episodes >= self.min_episodes and self.n_pos > 0

    def predict(self, feats: np.ndarray) -> float:
        """P_fail for a single feature vector; 0.5 before the model is fit
        (the gate z-normalizes, so a constant contributes nothing)."""
        if not self.ready:
            return 0.5
        feats = np.asarray(feats, dtype=np.float64).reshape(self.dim)
        return float(_sigmoid(feats @ self.w + self.b))

    # ------------------------------------------------------------------
    @staticmethod
    def make_labels(num_steps: int, horizon: int, success: bool,
                    hard_violation_steps) -> np.ndarray:
        """Per-step binary labels for one finished episode."""
        labels = np.zeros(num_steps, dtype=np.float64)
        events = list(hard_violation_steps)
        if not success:
            events.append(num_steps - 1)  # terminal failure event
        for e in events:
            lo = max(0, e - horizon)
            labels[lo: e + 1] = 1.0
        return labels

    def fit_episode(self, feats: np.ndarray, success: bool,
                    hard_violation_steps=()) -> None:
        """SGD update on one finished episode's per-step features."""
        feats = np.asarray(feats, dtype=np.float64)
        if feats.ndim != 2 or feats.shape[0] == 0:
            return
        assert feats.shape[1] == self.dim, (
            f"expected {self.dim} features, got {feats.shape[1]}")
        y = self.make_labels(feats.shape[0], self.horizon, success,
                             hard_violation_steps)
        self.n_pos += int(y.sum())
        self.n_neg += int((1 - y).sum())
        n = feats.shape[0]
        rng = np.random.default_rng(self.n_episodes)
        for _ in range(self.epochs_per_fit):
            order = rng.permutation(n)
            for i in order:
                p = _sigmoid(feats[i] @ self.w + self.b)
                g = p - y[i]
                self.w -= self.lr * (g * feats[i] + self.l2 * self.w)
                self.b -= self.lr * g
        self.n_episodes += 1

    def state_dict(self) -> dict:
        return dict(w=self.w.tolist(), b=self.b, n_episodes=self.n_episodes,
                    n_pos=self.n_pos, n_neg=self.n_neg)
