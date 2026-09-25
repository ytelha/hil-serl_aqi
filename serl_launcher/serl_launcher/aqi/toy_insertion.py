"""2-D toy insertion environment (methodology doc S7 worked example).

The smallest environment with the right physics of failure: a peg tip
p = (x, z) under velocity control at 10 Hz; a hole at x = 0 in the
surface z = 0 with radial clearance 1 mm and insertion depth 20 mm;
contact with friction (lateral correction impeded while pressing),
wedging drift, a jam budget of 25 contact events beyond which the part
counts as damaged; 15 s episode budget; starts x0 ~ U(-30, 30) mm at
z0 = 60 mm.

Deliberately trivial - what transfers to a real simulator is the wrapper
structure, the calibration protocol, the censoring caveat, and the
classification pipeline (doc S7 "what transfers"). Used here to
(a) validate the synthetic-operator generator against the document's
Table-5 separation pattern and (b) manufacture ground-truth-labelled
populations for validating the skill classifier (doc S11).

Gymnasium API. Observation: {"state": [x, z, vx, vz, F_lat, F_z,
jam_frac, t_frac]} (mm, mm/s, N-ish force units). info carries the
ground truths the oracle and the feature extractor need.
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

from serl_launcher.aqi.operator_model import PhaseOracle

DT = 0.1                     # 10 Hz
CLEARANCE = 1.0              # mm
DEPTH = 20.0                 # mm
JAM_BUDGET = 25              # contact events -> damaged
BUDGET_S = 15.0
X0_RANGE = 30.0
Z0 = 60.0
A_MAX = 100.0                # |commanded velocity| cap (mm/s)
FRICTION = 0.30              # lateral effectiveness while pressing
WEDGE_DRIFT = 1.0            # mm/s drift outward while jammed
K_FORCE = 0.12               # force per commanded mm/s while blocked


class ToyInsertionEnv(gym.Env):
    """See module docstring. Action: desired (vx, vz) in mm/s."""

    metadata = {"render_modes": []}

    def __init__(self, budget_s: float = BUDGET_S, seed: int = 0,
                 fixed_x0=None):
        self.observation_space = spaces.Dict(
            {"state": spaces.Box(-np.inf, np.inf, (8,), np.float64)})
        self.action_space = spaces.Box(-A_MAX, A_MAX, (2,), np.float64)
        self.max_steps = int(round(budget_s / DT))
        self.rng = np.random.default_rng(seed)
        self.fixed_x0 = fixed_x0
        self.reset()

    # ------------------------------------------------------------------
    def _obs(self):
        s = np.array([self.x, self.z, self.vx, self.vz,
                      self.f_lat, self.f_z,
                      self.jams / JAM_BUDGET, self.t / self.max_steps])
        return {"state": s}

    def _info(self, jam_event=False):
        return dict(x=self.x, z=self.z, f_lat=self.f_lat, f_z=self.f_z,
                    contact=self.contact, jammed=self.jammed,
                    jam_event=jam_event, jams=self.jams,
                    succeeded=self.succeeded, t=self.t * DT)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x = (float(self.fixed_x0) if self.fixed_x0 is not None
                  else float(self.rng.uniform(-X0_RANGE, X0_RANGE)))
        self.z = Z0
        self.vx = self.vz = 0.0
        self.f_lat = self.f_z = 0.0
        self.t = 0
        self.jams = 0
        self.contact = self.jammed = False
        self._grind = 0
        self.succeeded = False
        return self._obs(), self._info()

    # ------------------------------------------------------------------
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -A_MAX, A_MAX)
        vx_c, vz_c = float(a[0]), float(a[1])
        self.f_lat = self.f_z = 0.0
        jam_event = False

        inside = self.z < 0.0
        at_surface = (not inside) and self.z <= 1e-9
        misaligned = abs(self.x) > CLEARANCE

        if inside:
            # in the hole: descend freely, walls constrain |x| <= clearance
            self.x += vx_c * DT
            if abs(self.x) > CLEARANCE:
                self.f_lat = K_FORCE * abs(vx_c) + 0.5
                self.x = np.clip(self.x, -CLEARANCE, CLEARANCE)
            self.z += vz_c * DT
            self.z = min(self.z, 0.0)      # can retract out through the top
            self.contact, self.jammed = True, False
        elif at_surface and misaligned and vz_c < 0.0:
            # pressing on the surface off the hole: JAM. A "contact
            # event" (doc S7 jam budget) is the rising edge of a jam
            # plus one further event per 0.5 s of sustained grinding -
            # continuous pressing damages the part progressively.
            rising = not self.jammed
            self.contact = self.jammed = True
            self._grind = 0 if rising else self._grind + 1
            jam_event = bool(rising or (self._grind % 5 == 0))
            self.jams += int(jam_event)
            self.f_z = K_FORCE * abs(vz_c) + 1.0
            self.f_lat = 0.3 * self.f_z
            # friction impedes lateral correction; wedging drifts outward
            self.x += FRICTION * vx_c * DT \
                + WEDGE_DRIFT * np.sign(self.x) * DT
            self.z = 0.0                    # cannot penetrate the surface
            if vz_c > 0:                    # (unreachable branch, clarity)
                self.z += vz_c * DT
        else:
            # free space (or moving up off the surface)
            self.contact = self.jammed = False
            self.x += vx_c * DT
            self.z += vz_c * DT
            if self.z <= 0.0 and abs(self.x) <= CLEARANCE:
                self.contact = True         # entering the hole
            elif self.z < 0.0:
                self.z = 0.0                # blocked by the surface

        self.vx, self.vz = vx_c, vz_c
        self.t += 1
        self.succeeded = self.z <= -DEPTH
        damaged = self.jams > JAM_BUDGET
        terminated = bool(self.succeeded or damaged)
        truncated = bool(self.t >= self.max_steps and not terminated)
        reward = 1.0 if self.succeeded else 0.0
        return self._obs(), reward, terminated, truncated, \
            self._info(jam_event)


class ToyInsertionOracle(PhaseOracle):
    """PhaseOracle for the toy env: the primitives the operator's phase
    machine consumes. Reads the (delayed, operator-side) observation."""

    lateral_dims = (0,)
    vertical_dim = 1

    def lateral_error(self, obs):
        return np.array([obs["state"][0]])   # hole is at x = 0

    def height(self, obs):
        return float(obs["state"][1])

    def in_contact(self, obs):
        return bool(obs["state"][5] > 0.05 or obs["state"][1] < 0.0)

    def jammed(self, obs):
        # pressing at the surface with lateral misalignment: F_z present
        return bool(obs["state"][5] > 0.05)

    def commit_threshold(self):
        return 0.8 * CLEARANCE

    def approach_height(self):
        return 8.0

    def action_dim(self):
        return 2


# ===========================================================================
# population harness (doc S7 step 3 / S11): rollouts with the feature-
# extractor's trajectory schema and ground-truth labels attached (R2)
# ===========================================================================
def rollout_episode(env, op, seed):
    """One closed-loop episode -> the trajectory dict episode_features
    consumes, plus ground truth."""
    obs, info = env.reset(seed=seed)
    op.reset()
    P, F, C = [], [], []
    done = False
    jam = 0
    while not done:
        a = op.act(obs)
        obs, r, term, trunc, info = env.step(a)
        P.append([obs["state"][0], obs["state"][1]])
        F.append(abs(info["f_lat"]) + abs(info["f_z"]))
        C.append(bool(info["contact"]))
        jam += int(info["jam_event"])
        done = term or trunc
    return dict(positions=np.asarray(P), forces=np.asarray(F),
                contact=np.asarray(C), success=bool(info["succeeded"]),
                jam_events=jam,
                commit_err=(op.commit_events[0][1]
                            if op.commit_events else None),
                attempts=op.n_approaches, vertical_dim=1)


def generate_population(op_thetas, n_episodes, seed0=0, budget_s=BUDGET_S):
    """dict operator_id -> list of trajectory dicts; each theta is run
    on its own seeded env/operator stream (R6)."""
    from serl_launcher.aqi.operator_model import SyntheticOperator
    out = {}
    for i, th in enumerate(op_thetas):
        env = ToyInsertionEnv(budget_s=budget_s, seed=seed0 + 1000 * i)
        op = SyntheticOperator(ToyInsertionOracle(), th,
                               rng=np.random.default_rng(seed0 + 77 * i))
        out[th.operator_id] = [
            rollout_episode(env, op, seed0 * 100000 + 1000 * i + e)
            for e in range(n_episodes)]
    return out
