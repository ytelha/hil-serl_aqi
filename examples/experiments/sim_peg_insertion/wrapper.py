"""Environment wrapper classes for the MuJoCo peg-insertion task.

`SimPegEnv` adapts the MuJoCo `PandaPegInsertGymEnv` to the constructor
signature the HIL-SERL experiment configs use (`fake_env`, `save_video`, ...),
and `SimPegRewardWrapper` exposes the simulator's ground-truth success as the
task reward -- this replaces the real tasks' learned reward classifier, since in
simulation we know exactly when the peg is seated.
"""

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from franka_sim.envs.orientation_box import parse_rpy_box
from franka_sim.envs.panda_peg_insert_gym_env import PandaPegInsertGymEnv
from serl_launcher.aqi.operator_model import PhaseOracle


class SimPegEnv(PandaPegInsertGymEnv):
    """Thin adapter so the experiment config can construct the sim env with the
    same keyword arguments (`fake_env`, `save_video`) as the real Franka envs.

    Both flags are accepted for API compatibility. `fake_env` builds the same
    MuJoCo env (used by the learner only for its observation/action spaces);
    rendering happens lazily so no GL context is created unless the env is
    actually stepped.
    """

    def __init__(self, fake_env: bool = False, save_video: bool = False, **kwargs):
        self.fake_env = fake_env
        self.save_video = save_video
        super().__init__(**kwargs)
        # Pace interactive/actor rollouts to real time so a human teleoperator
        # has the full episode duration to react; the headless learner
        # (fake_env=True) keeps running as fast as possible.
        self.realtime = not fake_env


class SimPegRewardWrapper(gym.Wrapper):
    """Use the MuJoCo ground-truth insertion success as the sparse task reward.

    The underlying env already computes success from the peg-tip / socket
    geometry and returns it in `info["succeed"]`; this wrapper simply mirrors
    the real tasks' reward-classifier wrapper interface (setting `reward`,
    `done`, and `info["succeed"]`) so the rest of the pipeline is identical.
    """

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        succeed = bool(info.get("succeed", rew > 0))
        rew = int(succeed)
        done = done or succeed
        info["succeed"] = succeed
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return obs, info


# ===========================================================================
# Synthetic-operator support (methodology doc Part A, option A): a geometric
# PhaseOracle for the sim so a SyntheticOperator can TELEOPERATE the real sim
# task (not just the toy env) and produce training-format demonstrations.
# ===========================================================================

# True radial bore clearances measured from the arena XMLs (peg vs bore):
# cylinder peg r=20 mm / bore r=28 mm -> 8 mm; cuboid ~5 mm; hex ~3 mm. This is
# the physical "how far off-axis can the tip be and still enter" margin, and it
# drives both the operator's commit decision (commit_threshold = 0.8 x this)
# and the monitor's corridor risk (saturates at this error).
_CLEARANCE_MM = {"cylinder": 8.0, "cuboid": 5.0, "hex": 3.0}

# Orientation tolerances (rad) of the operator monitor's optional orientation
# channel: (tilt, yaw about the bore axis); yaw None = free (round bore). The
# channel is 1.0 at these errors near the surface, like the corridor at the
# bore clearance. Yaw: a centred cuboid (36 mm peg, 46 mm bore) touches the
# wall at 19.6 deg (0.34 rad), so 0.25 leaves room for a lateral offset. Tilt
# 0.10 and the hex yaw are placeholders below the geometric limits; replace
# them with the measured ones (examples/oracle_insertion_test.py --sweep).
_ORIENTATION_TOL = {"cylinder": (0.10, None), "cuboid": (0.10, 0.25),
                    "hex": (0.10, 0.12)}


def sim_rpy_box_from_env():
    """The optional orientation box of the sim env from $SIM_RPY_BOX:
    'roll,pitch,yaw' half-widths in rad around the aligned home orientation;
    unset or empty = no box (the original, unbounded orientation). The
    campaign runner sets it for every process of a campaign (demos, learner,
    actor, ruler), so all of them see the same task."""
    import os
    return parse_rpy_box(os.environ.get("SIM_RPY_BOX", ""))


class SimPegPhaseOracle(PhaseOracle):
    """Geometric oracle (doc S4 option A) for the MuJoCo peg env.

    Exposes the tip/hole primitives the synthetic operator's phase machine
    consumes, read LIVE from the simulator's ground-truth sensors
    (``peg_tip_pos`` / ``hole_top_pos`` / ``hole_bottom_pos``). Lateral error
    is returned in the END-EFFECTOR frame so that after ``RelativeFrame``
    rotates the operator's action back to the world frame the correction points
    at the bore (the R_ee round-trips exactly for the lateral subspace); the
    vertical action sign is fixed once per episode by ``_VerticalSignAdapter``.

    Because it reads live sensors rather than the (delayed) observation, the
    operator's perception LATENCY does not apply to the geometric channel -- a
    deliberate v1 simplification; the dominant tier signals (per-approach
    perceptual bias, signal-dependent motor noise, recovery style, premature
    commits) are unaffected.
    """

    lateral_dims = (0, 1)     # EE x, y translation action dims
    vertical_dim = 2          # EE z translation action dim
    rotation_dims = (3, 4, 5)  # EE rx, ry, rz rotation action dims
    # rotational symmetry of the bore about its axis (rad); None = round bore
    _SYMMETRY = {"cylinder": None, "cuboid": np.pi / 2, "hex": np.pi / 3}

    def __init__(self, base_env, peg_shape="cylinder",
                 approach_h_mm=25.0, contact_h_mm=4.0,
                 force_baseline=12.0):
        self.env = base_env
        self.peg_shape = peg_shape
        self.symmetry = self._SYMMETRY.get(peg_shape)
        self.clearance_mm = _CLEARANCE_MM.get(peg_shape, 8.0)
        self.approach_h_mm = float(approach_h_mm)
        self.contact_h_mm = float(contact_h_mm)
        # Wrist-force sensor baseline: the static peg+gripper load registers
        # ~12 sim units even in free air (measured); the monitor uses the
        # force EXCESS above this (a jam adds ~10-14).
        self.force_baseline = float(force_baseline)

    def reset(self):
        pass

    def _geom(self):
        tip = self.env._sensor("peg_tip_pos")
        htop = self.env._sensor("hole_top_pos")
        hbot = self.env._sensor("hole_bottom_pos")
        _, quat_xyzw = self.env._pinch_pose()
        return tip, htop, hbot, Rotation.from_quat(quat_xyzw)

    def lateral_error(self, obs):
        tip, _htop, hbot, R_ee = self._geom()
        e_world = np.array([tip[0] - hbot[0], tip[1] - hbot[1], 0.0])
        e_ee = R_ee.inv().apply(e_world)
        return e_ee[:2] * 1000.0            # mm, end-effector frame

    def height(self, obs):
        tip, htop, _hbot, _R = self._geom()
        return float((tip[2] - htop[2]) * 1000.0)   # mm above the opening

    def in_contact(self, obs):
        return bool(self.height(obs) <= self.contact_h_mm)

    def wrist_force_mag(self, obs=None):
        """Live wrist force magnitude (sim units; ~12 baseline in free air)."""
        return float(np.linalg.norm(self.env._sensor("panda/wrist_force")))

    def jammed(self, obs):
        lat = float(np.linalg.norm(self.lateral_error(obs)))
        return bool(self.height(obs) <= self.contact_h_mm
                    and lat > self.clearance_mm)

    def commit_threshold(self):
        return 0.8 * self.clearance_mm

    def approach_height(self):
        return self.approach_h_mm

    def action_dim(self):
        return 6

    def orientation_error(self, obs):
        """Rotation vector (rad, END-EFFECTOR frame) from the current tool
        orientation to the nearest insertable one: the aligned home
        orientation of the reset (``_reset_quat``, before the random yaw),
        turned by any multiple of the bore's symmetry angle; for the round
        bore only the tilt counts. Expressed in the EE frame like
        ``lateral_error``, because ``RelativeFrame`` rotates the operator's
        rotation command back to the world frame with the same R_ee."""
        _tip, _htop, _hbot, R_ee = self._geom()
        return R_ee.inv().apply(self._orientation_error_world(R_ee))

    def tilt_yaw_error(self, obs):
        """(tilt, yaw) in rad: the same error as ``orientation_error``,
        split about the WORLD axes into the tilt of the tool axis and the
        yaw about the bore axis (folded by the bore symmetry; 0 for the
        round bore). Read by the monitor's optional orientation channel."""
        _tip, _htop, _hbot, R_ee = self._geom()
        rv = self._orientation_error_world(R_ee)
        return float(np.linalg.norm(rv[:2])), float(abs(rv[2]))

    def orientation_tolerance(self):
        """Default (tilt, yaw) limits of the orientation channel (rad)."""
        return _ORIENTATION_TOL.get(self.peg_shape, (0.10, None))

    def _orientation_error_world(self, R_ee):
        """World-frame rotation vector from the tool to the nearest
        insertable orientation."""
        R_tgt = Rotation.from_quat(np.asarray(self.env._reset_quat,
                                              dtype=np.float64))
        if self.symmetry is None:
            rv = (R_tgt * R_ee.inv()).as_rotvec()
            rv[2] = 0.0                   # yaw is free in a round bore
        else:
            n = int(round(2 * np.pi / self.symmetry))
            cands = [(Rotation.from_rotvec([0.0, 0.0, k * self.symmetry])
                      * R_tgt * R_ee.inv()).as_rotvec() for k in range(n)]
            rv = min(cands, key=lambda v: float(np.linalg.norm(v)))
        return rv

    def rot_rad_per_env_unit(self):
        return float(self.env._action_scale[1])


class _VerticalSignAdapter(gym.ActionWrapper):
    """Flip the operator's EE-frame vertical action so that 'descend' (the
    phase machine's ``-v_max`` on ``vertical_dim``) moves the tool DOWN in the
    world after ``RelativeFrame`` rotates it. The sign is ``sign(R_ee[2, 2])``
    -- the world-z component of the EE z-axis -- read from the base env at each
    reset (nearly constant during an episode). Must sit OUTSIDE RelativeFrame
    so it conditions the operator's raw EE-frame action before the rotation.
    """

    def __init__(self, env, base_env):
        super().__init__(env)
        self._base = base_env
        self._s = 1.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        _, quat_xyzw = self._base._pinch_pose()
        quat = np.asarray(quat_xyzw, dtype=np.float64)
        if np.linalg.norm(quat) < 1e-12:
            self._s = 1.0
            return obs, info
        r22 = float(Rotation.from_quat(quat).as_matrix()[2, 2])
        self._s = 1.0 if r22 >= 0.0 else -1.0
        return obs, info

    def action(self, action):
        action = np.array(action, dtype=np.float32)
        if action.shape[0] > self._VERT:
            action[self._VERT] = action[self._VERT] * self._s
        return action

    _VERT = 2


class VerticalSignConditioner:
    """Condition synthetic operator actions so a negative EE-z command means
    world-frame descent after RelativeFrame processing.

    The sign is measured once per episode from the base env end-effector pose
    as sign(R_ee[2, 2]) and then applied to the configured vertical action
    dimension.
    """

    def __init__(self, base_env, vertical_dim: int = 2):
        self._base = base_env
        self._vert = int(vertical_dim)
        self._s = 1.0

    def reset(self):
        _, quat_xyzw = self._base._pinch_pose()
        quat = np.asarray(quat_xyzw, dtype=np.float64)
        if np.linalg.norm(quat) < 1e-12:
            self._s = 1.0
            return
        r22 = float(Rotation.from_quat(quat).as_matrix()[2, 2])
        self._s = 1.0 if r22 >= 0.0 else -1.0

    def __call__(self, action):
        out = np.array(action, dtype=np.float32, copy=True)
        if out.shape[0] > self._vert:
            out[self._vert] = out[self._vert] * self._s
        return out


def make_synthetic_sim_env(peg_shape: str = None, image_obs: bool = True,
                           max_episode_length: int = 300,
                           rpy_box="from_env"):
    """Build the sim_peg_insertion training env stack WITHOUT the SpaceMouse
    wrapper and attach a `phase_oracle` so `collect_synthetic_demos.py --env
    experiments.sim_peg_insertion.wrapper:make_synthetic_sim_env` can drive a
    SyntheticOperator and record TRAINING-FORMAT demonstrations. The env also
    advertises `operator_action_scale` (env action units per mm/s) and
    `budget_s` for the collector.

    `--env module:factory` calls this with no arguments, so the peg geometry is
    taken from $SIM_SYN_PEG_SHAPE (default 'cylinder'). By default the
    synthetic operator commands translation only (no yaw), so 'cylinder' (no
    yaw alignment) is the supported geometry; 'cuboid'/'hex' need yaw and
    insert less reliably unless the operator runs with rotation control
    (SyntheticOperator(..., rotation_control=True), which uses
    SimPegPhaseOracle.orientation_error).

    `rpy_box` is the env's optional orientation box ('roll,pitch,yaw' rad or a
    3-sequence); the default takes it from $SIM_RPY_BOX, None = no box.
    """
    import os
    peg_shape = peg_shape or os.environ.get("SIM_SYN_PEG_SHAPE", "cylinder")
    from franka_env.envs.wrappers import GripperCloseEnv, Quat2EulerWrapper
    from franka_env.envs.relative_env import RelativeFrame
    from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
    from serl_launcher.wrappers.chunking import ChunkingWrapper

    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque",
                    "gripper_pose"]
    # fake_env=True keeps the env non-realtime (fast batch collection) while
    # still stepping full MuJoCo physics; it also means no SpaceMouse wrapper.
    if isinstance(rpy_box, str) and rpy_box == "from_env":
        rpy_box = sim_rpy_box_from_env()
    base = SimPegEnv(fake_env=True, image_obs=image_obs,
                     image_keys=("wrist_1", "wrist_2"),
                     max_episode_length=max_episode_length,
                     peg_shape=peg_shape, rpy_box=rpy_box)
    oracle = SimPegPhaseOracle(base, peg_shape=peg_shape)

    env = GripperCloseEnv(base)
    env = RelativeFrame(env)
    env = _VerticalSignAdapter(env, base)     # OUTSIDE RelativeFrame
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = SimPegRewardWrapper(env)

    env.phase_oracle = oracle
    env.operator_action_scale = 0.01          # 1 action unit ~= 100 mm/s
    env.budget_s = max_episode_length * 0.1
    return env


def make_synthetic_monitor(oracle: "SimPegPhaseOracle", budget_steps: int,
                           force_norm: float = 20.0,
                           force_baseline: float = 20.0):
    """Calibrated risk signal z_t (doc S6.1) for the sim actor's synthetic
    operator. Channels (combined by max):
      * corridor  : |lateral error| / bore clearance (saturates at the edge of
                    the bore), active only near the surface;
      * stagnation: no descent progress for ~1.2 s -- the PRIMARY jam detector;
      * timeout   : fraction of the episode budget used;
      * force     : wrist-force EXCESS above `force_baseline`, normalized by
                    `force_norm`. LOOSENED (baseline/norm 12 -> 20): insertion
                    ITSELF requires a firm press, so a small force excess must
                    NOT trip a takeover or the policy never learns to insert
                    (it collapses to no-press). A genuine jam is caught by the
                    stagnation channel (stuck = no descent); only a large,
                    sustained force excess drives THIS channel toward 1.
    Every distance is millimetres and every force is in the sim's wrist-sensor
    units, measured against THIS environment (not assumed)."""
    from serl_launcher.aqi.operator_gate import MonitoringSignal

    def force_fn(obs):
        return max(0.0, oracle.wrist_force_mag(obs) - force_baseline) \
            / force_norm

    return MonitoringSignal(oracle, corridor_mm=oracle.clearance_mm,
                            stall_s=1.2, budget_steps=int(budget_steps),
                            corridor_height_mm=12.0,
                            hover_corridor_mm=max(12.0,
                                                  2.5 * oracle.clearance_mm),
                            force_fn=force_fn)
