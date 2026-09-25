"""MuJoCo peg-in-hole (peg insertion) environment for HIL-SERL / AQI-HIL-SERL.

This env exposes the *same* observation/action contract as the real-robot
``franka_env.envs.franka_env.FrankaEnv`` so that the entire HIL-SERL /
AQI-HIL-SERL wrapper and training stack (SpaceMouse intervention, RelativeFrame,
Quat2EulerWrapper, SERLObsWrapper, ChunkingWrapper, RLPD, AQI) runs unchanged in
simulation.

Interface (matches FrankaEnv):
    action : Box(7,)  ->  [dx, dy, dz, drx, dry, drz, gripper]  (delta pose)
    obs    : Dict{
                "state": Dict{
                    "tcp_pose"   : (7,)  xyz + quat(xyzw),
                    "tcp_vel"    : (6,)  linear + angular,
                    "gripper_pose": (1,),
                    "tcp_force"  : (3,),
                    "tcp_torque" : (3,),
                },
                "images": Dict{cam: (128,128,3) uint8}   # empty in state-only mode
             }
    step   -> (obs, int(reward), done, False, info)   with info["succeed"]

Unlike the real robot, the simulator knows ground-truth success, so the reward
is computed directly from the peg-tip / socket geometry -- no reward classifier
is required (see SimPegRewardWrapper in the experiment wrapper).
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from franka_sim.controllers import opspace

_HERE = Path(__file__).parent
# Arena XMLs keyed by peg_shape. "hex" is the exact reference task; "cylinder"
# is an easier round-peg/round-bore variant (no yaw alignment, larger clearance);
# "cuboid" is a square peg/square socket of intermediate difficulty (needs yaw
# alignment to within +/-45 deg, moderate clearance).
_ARENA_XMLS = {
    "hex": _HERE / "xmls" / "arena_peg.xml",
    "cylinder": _HERE / "xmls" / "arena_peg_cylinder.xml",
    "cuboid": _HERE / "xmls" / "arena_peg_cuboid.xml",
}

# Panda home joint configuration (arm pointing down).
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))

# Cartesian safety box for the TCP (pinch site), world frame [x, y, z].
_CARTESIAN_BOUNDS = np.asarray([[0.35, -0.15, 0.10], [0.65, 0.15, 0.46]])


class PandaPegInsertGymEnv(gym.Env):
    """Franka Panda peg-in-hole task in MuJoCo with the FrankaEnv interface."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        image_obs: bool = True,
        image_keys: Tuple[str, ...] = ("wrist_1", "wrist_2"),
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        # socket/peg geometry: "hex" (exact reference), "cylinder" (easier round
        # peg) or "cuboid" (square peg, intermediate difficulty)
        peg_shape: str = "hex",
        render_width: int = 128,
        render_height: int = 128,
        # resolution of the operator monitoring view returned by render()
        # (kept separate from the 128x128 policy observation images so the
        # human-facing video is large and crisp).
        view_width: int = 640,
        view_height: int = 480,
        max_episode_length: int = 100,
        randomize_reset: bool = True,
        random_xy_range: float = 0.015,
        random_rz_range: float = 0.08,
        # success thresholds (see reward): the exact hex peg self-centers once
        # started, so success = tip within success_lateral of the bore axis AND
        # inserted deeper than success_depth below the opening.
        success_lateral: float = 0.012,
        success_depth: float = 0.05,
        # action scaling: [pos (m), ori (rad)] per unit action
        action_scale: Tuple[float, float] = (0.01, 0.05),
        seed: int = 0,
        render_mode: str = "rgb_array",
        **kwargs,
    ):
        super().__init__()
        self.image_obs = image_obs
        self.image_keys = tuple(image_keys) if image_obs else ()
        self._action_scale = np.asarray(action_scale, dtype=np.float64)
        self.max_episode_length = max_episode_length
        self.randomize_reset = randomize_reset
        self.random_xy_range = random_xy_range
        self.random_rz_range = random_rz_range
        self._success_lateral = success_lateral
        self._success_depth = success_depth
        self._random = np.random.RandomState(seed)
        self.render_mode = render_mode

        # ---- MuJoCo model / data ---------------------------------------
        if peg_shape not in _ARENA_XMLS:
            raise ValueError(
                f"peg_shape must be one of {sorted(_ARENA_XMLS)}, got {peg_shape!r}"
            )
        self.peg_shape = peg_shape
        self._model = mujoco.MjModel.from_xml_path(_ARENA_XMLS[peg_shape].as_posix())
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = max(1, int(round(control_dt / physics_dt)))
        # When True, step() sleeps so each control step takes ~control_dt of
        # wall-clock time (real-time pacing). The real robot runs in real time
        # naturally; the sim otherwise blitzes through an episode in a couple of
        # seconds, giving a human teleoperator no time to react. Enabled for
        # interactive/actor rollouts, disabled for the (headless) learner.
        self.realtime = False
        self._next_step_wall: Optional[float] = None
        self._data = mujoco.MjData(self._model)
        self._render_width = render_width
        self._render_height = render_height
        self._view_width = view_width
        self._view_height = view_height
        self._renderer: Optional[mujoco.Renderer] = None
        self._view_renderer: Optional[mujoco.Renderer] = None

        # ---- cached ids ------------------------------------------------
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._pinch_site_id = self._model.site("pinch").id
        self._peg_qpos_adr = self._model.jnt("peg").qposadr[0]

        # sensor address helpers
        self._sensor_adr = {
            self._model.sensor(i).name: (
                self._model.sensor_adr[i],
                self._model.sensor_dim[i],
            )
            for i in range(self._model.nsensor)
        }

        # ---- spaces ----------------------------------------------------
        self.action_space = spaces.Box(
            low=-np.ones(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )
        state_space = spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "gripper_pose": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                "tcp_force": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "tcp_torque": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )
        image_space = spaces.Dict(
            {
                key: spaces.Box(0, 255, shape=(render_height, render_width, 3), dtype=np.uint8)
                for key in self.image_keys
            }
        )
        self.observation_space = spaces.Dict(
            {"state": state_space, "images": image_space}
        )

        # runtime state
        self._des_pos = np.zeros(3)
        self._des_quat = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw
        self._reset_quat = np.array([0.0, 0.0, 0.0, 1.0])
        # Fixed evaluation pose for the ruler (eval_ruler.py): when set to
        # (dx_m, dyaw_rad) the next reset(s) use that exact offset from the
        # socket with randomization OFF, so every campaign cell is measured on
        # the identical pose grid. None -> normal (randomized) reset.
        self._eval_pose = None
        self.curr_path_length = 0
        self.success = False

    # ------------------------------------------------------------------
    # sensor helpers
    # ------------------------------------------------------------------
    def _sensor(self, name: str) -> np.ndarray:
        adr, dim = self._sensor_adr[name]
        return self._data.sensordata[adr : adr + dim].copy()

    def _pinch_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self._sensor("2f85/pinch_pos")
        quat_wxyz = self._sensor("2f85/pinch_quat")
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        )
        return pos, quat_xyzw

    def set_eval_reset_pose(self, dx_m: float, dyaw_rad: float) -> None:
        """Ruler hook (eval_ruler.py): pin the next reset(s) to a fixed grid
        offset -- lateral dx (m) along +x from the socket and yaw dyaw (rad)
        about z -- with reset randomization disabled, so every experimental
        cell is measured on the identical, hash-locked pose grid. Pass None to
        restore normal randomized resets."""
        self._eval_pose = None if dx_m is None else (float(dx_m),
                                                     float(dyaw_rad))

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self._random = np.random.RandomState(seed)

        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        self._data.ctrl[self._panda_ctrl_ids] = 0.0
        mujoco.mj_forward(self._model, self._data)

        # capture the natural downward orientation of the home pose
        _, self._reset_quat = self._pinch_pose()

        # place the peg at its welded location so the weld does not "snap"
        self._settle_peg_to_weld()

        # target reset pose: above the socket opening, optionally randomized
        hole_top = self._sensor("hole_top_pos")
        reset_xy = hole_top[:2].copy()
        reset_quat = self._reset_quat.copy()
        if self._eval_pose is not None:
            # Fixed-grid eval pose (ruler): deterministic lateral offset dx
            # along +x and yaw offset dyaw about z, randomization OFF, so every
            # campaign cell is measured on the identical grid.
            dx, dyaw = self._eval_pose
            reset_xy[0] += float(dx)
            reset_quat = (
                Rotation.from_rotvec([0, 0, float(dyaw)])
                * Rotation.from_quat(reset_quat)
            ).as_quat()
        elif self.randomize_reset:
            reset_xy += self._random.uniform(
                -self.random_xy_range, self.random_xy_range, size=2
            )
            dz = self._random.uniform(-self.random_rz_range, self.random_rz_range)
            reset_quat = (
                Rotation.from_rotvec([0, 0, dz]) * Rotation.from_quat(reset_quat)
            ).as_quat()
        # pinch sits ~0.10 m above the opening so the peg tip clears it
        reset_pos = np.array([reset_xy[0], reset_xy[1], hole_top[2] + 0.10])

        self._des_pos = np.clip(reset_pos, _CARTESIAN_BOUNDS[0], _CARTESIAN_BOUNDS[1])
        self._des_quat = reset_quat
        # drive the arm to the reset pose. The long welded peg forms a slow
        # pendulum, so the opspace controller needs several hundred substeps to
        # converge the tip to <1 mm lateral error (120 leaves it ~3.5 cm off).
        self._run_controller(settle_steps=400)

        self.curr_path_length = 0
        self.success = False
        self._next_step_wall = None
        obs = self._get_obs()
        return obs, {"succeed": False}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], int, bool, bool, Dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # position delta (world frame)
        self._des_pos = self._des_pos + action[:3] * self._action_scale[0]
        self._des_pos = np.clip(
            self._des_pos, _CARTESIAN_BOUNDS[0], _CARTESIAN_BOUNDS[1]
        )
        # orientation delta (world frame, applied on the left)
        drot = Rotation.from_rotvec(action[3:6] * self._action_scale[1])
        self._des_quat = (drot * Rotation.from_quat(self._des_quat)).as_quat()

        # action[6] (gripper) is ignored: the peg is pregrasped/welded.
        self._run_controller(settle_steps=self._n_substeps)

        self.curr_path_length += 1
        obs = self._get_obs()
        reward, succeed = self._compute_reward()
        self.success = succeed
        done = bool(succeed) or self.curr_path_length >= self.max_episode_length
        info = {"succeed": bool(succeed)}

        # real-time pacing: hold each control step to ~control_dt of wall time
        # so a human teleoperator experiences the episode at real-world speed.
        if self.realtime:
            now = time.monotonic()
            if self._next_step_wall is None:
                self._next_step_wall = now
            self._next_step_wall += self._control_dt
            sleep_for = self._next_step_wall - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # fell behind (slow render/compute) -> resync, don't accumulate lag
                self._next_step_wall = time.monotonic()
        return obs, int(reward), done, False, info

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def _run_controller(self, settle_steps: int):
        for _ in range(settle_steps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._des_pos,
                ori=self._quat_xyzw_to_wxyz(self._des_quat),
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

    def _settle_peg_to_weld(self):
        """Set the peg free-joint qpos to its welded pose in the base frame so the
        weld constraint does not violently snap it into place on reset."""
        base_id = self._model.body("base").id
        base_pos = self._data.xpos[base_id].copy()
        base_quat_wxyz = self._data.xquat[base_id].copy()  # wxyz
        base_rot = Rotation.from_quat(
            [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]]
        )
        # weld relpose in base frame: pos (0,0,0.12), identity orientation
        peg_pos = base_pos + base_rot.apply([0.0, 0.0, 0.12])
        peg_quat_xyzw = base_rot.as_quat()
        adr = self._peg_qpos_adr
        self._data.qpos[adr : adr + 3] = peg_pos
        self._data.qpos[adr + 3 : adr + 7] = np.array(
            [peg_quat_xyzw[3], peg_quat_xyzw[0], peg_quat_xyzw[1], peg_quat_xyzw[2]]
        )
        self._data.qvel[
            self._model.jnt("peg").dofadr[0] : self._model.jnt("peg").dofadr[0] + 6
        ] = 0.0
        mujoco.mj_forward(self._model, self._data)

    @staticmethod
    def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
        return np.array([q[3], q[0], q[1], q[2]])

    # ------------------------------------------------------------------
    # observation / reward
    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict[str, Any]:
        pos, quat_xyzw = self._pinch_pose()
        tcp_pose = np.concatenate([pos, quat_xyzw]).astype(np.float32)
        tcp_vel = np.concatenate(
            [self._sensor("2f85/pinch_vel"), self._sensor("2f85/pinch_angvel")]
        ).astype(np.float32)
        state = {
            "tcp_pose": tcp_pose,
            "tcp_vel": tcp_vel,
            "gripper_pose": np.zeros(1, dtype=np.float32),
            "tcp_force": self._sensor("panda/wrist_force").astype(np.float32),
            "tcp_torque": self._sensor("panda/wrist_torque").astype(np.float32),
        }
        images = self._render_images() if self.image_obs else {}
        return {"state": state, "images": images}

    def _render_images(self) -> Dict[str, np.ndarray]:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self._model, height=self._render_height, width=self._render_width
            )
        images = {}
        for key in self.image_keys:
            self._renderer.update_scene(self._data, camera=key)
            images[key] = self._renderer.render().copy()
        return images

    def _compute_reward(self) -> Tuple[int, bool]:
        tip = self._sensor("peg_tip_pos")
        hole_top = self._sensor("hole_top_pos")
        hole_bottom = self._sensor("hole_bottom_pos")
        lateral = np.linalg.norm(tip[:2] - hole_bottom[:2])
        depth = hole_top[2] - tip[2]  # how far below the opening the tip is
        succeed = bool(lateral < self._success_lateral and depth > self._success_depth)
        return int(succeed), succeed

    def render(self):
        # dedicated higher-resolution renderer for the operator monitoring view
        # (the "front" 3/4 scene camera), independent of the small policy images.
        if self._view_renderer is None:
            self._view_renderer = mujoco.Renderer(
                self._model, height=self._view_height, width=self._view_width
            )
        self._view_renderer.update_scene(self._data, camera="front")
        return self._view_renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._view_renderer is not None:
            self._view_renderer.close()
            self._view_renderer = None
