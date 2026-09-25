# Intro:
This package provide a simple Franka arm and Robotiq Gripper simulator written in Mujoco.
It includes a state-based and a vision-based Franka lift cube task environment.

---

# Peg-insertion sim (`sim_peg_insertion`): files & tunable parameters

This is the reference for the MuJoCo peg-insertion task used by HIL-SERL /
AQI-HIL-SERL. It tells you **which files define the environment** and **where to
change the tolerances, the episode/reset time, the clearance, and the other
simulation parameters**. For the end-to-end run guide (collect demos → train →
evaluate) see [`docs/sim_walkthrough.md`](../docs/sim_walkthrough.md).

## Files that define the simulation environment

| File | What it defines |
| --- | --- |
| `franka_sim/franka_sim/envs/panda_peg_insert_gym_env.py` | The physics env `PandaPegInsertGymEnv`: reset logic, reward/success, action scaling, randomization, real-time pacing, cameras/render sizes, safety bounds. **Most numeric knobs live here** (constructor defaults). |
| `franka_sim/franka_sim/envs/xmls/arena_peg.xml` | **Hex** peg + hex socket scene geometry (exact reference meshes). Clearance is set by the peg mesh `scale`. |
| `franka_sim/franka_sim/envs/xmls/arena_peg_cylinder.xml` | **Cylinder** (round) peg + round bore scene geometry. Clearance is set by the peg radius vs. the bore staves. |
| `franka_sim/franka_sim/envs/xmls/meshes_peg/` | Hex peg/socket meshes (`peg1.obj`, `framing1-6.obj`). |
| `franka_sim/franka_sim/envs/xmls/panda.xml` (+ `franka_emika_panda/`, `robotiq_2f85/`) | The Panda arm + gripper model included by both arenas. |
| `examples/experiments/sim_peg_insertion/config.py` | Experiment `TrainConfig`: geometry choice, episode length, image/proprio keys, AQI thresholds, training periods. |
| `examples/experiments/sim_peg_insertion/wrapper.py` | `SimPegEnv` adapter (+ ground-truth reward wrapper). Enables real-time pacing for interactive rollouts. |
| `examples/experiments/sim_peg_insertion/sim_env.sh` | Shell env setup (venv, `MUJOCO_GL`, `LD_LIBRARY_PATH` for the SpaceMouse, `PYTHONPATH`). |

## Where to change the common parameters

Unless noted, edit the **constructor defaults** of `PandaPegInsertGymEnv.__init__`
in `panda_peg_insert_gym_env.py`.

| Parameter | Default | Where | Meaning / how to tune |
| --- | --- | --- | --- |
| **Geometry** `peg_shape` | `"cylinder"` | `config.py` (`TrainConfig.peg_shape`) | `"cylinder"` = easy round peg (no yaw alignment, 8 mm clearance); `"hex"` = exact tight hex task (needs xy **and** yaw). |
| **Episode / reset time** `max_traj_length` | `300` | `config.py` (`TrainConfig.max_traj_length`) | Steps per episode before it resets. At 10 Hz **real-time** control this is ~30 s. Increase for more time per teleop attempt. |
| **Real-time pacing** `realtime` | on for teleop | `wrapper.py` (`SimPegEnv`, `= not fake_env`) | When on, each step sleeps to `control_dt` wall-time so the episode runs at real speed. The headless learner (`fake_env=True`) runs full speed. |
| **Success tolerance (lateral)** `success_lateral` | `0.012` m | env constructor | Max tip-to-bore-axis error to count as inserted. Larger = easier. |
| **Success tolerance (depth)** `success_depth` | `0.05` m | env constructor | Min insertion depth below the opening to count as inserted. Smaller = easier. |
| **Clearance — cylinder** | peg r `0.02`, bore r `~0.028` | `arena_peg_cylinder.xml` | Widen the gap by shrinking the peg `<geom name="peg" ... size="0.02 0.10">` radius or moving the `stave*` boxes outward. |
| **Clearance — hex** | mesh `scale="0.9 0.9 1"` | `arena_peg.xml` (`peg_mesh`) | Smaller x/y scale = looser fit (`"1 1 1"` = exact reference). |
| **Reset randomization (xy)** `random_xy_range` | `0.015` m | env constructor | Half-range of the random start offset in x/y. Smaller = peg starts closer to centered. |
| **Reset randomization (yaw)** `random_rz_range` | `0.08` rad | env constructor | Half-range of the random start yaw. Irrelevant for the round peg. |
| **Action scale** `action_scale` | `(0.01, 0.05)` | env constructor | Meters/radians of motion per unit action `[pos, ori]`. Lower = finer/slower control. |
| **Control / physics rate** `control_dt`, `physics_dt` | `0.1`, `0.002` s | env constructor | Control period and sim timestep (`n_substeps = control_dt/physics_dt`). |
| **Reset settle steps** | `400` | env `reset()` (`_run_controller(settle_steps=400)`) | Substeps to let the welded peg pendulum converge over the hole (<1 mm) before an episode starts. |
| **Monitoring view size** `view_width`, `view_height` | `640`, `480` | env constructor | Resolution of `env.render()` (the operator/video `front` camera). |
| **Policy image size** `render_width`, `render_height` | `128`, `128` | env constructor | Resolution of the `wrist_1`/`wrist_2` policy-observation cameras. |
| **TCP safety box** `_CARTESIAN_BOUNDS` | `[[0.35,-0.15,0.10],[0.65,0.15,0.46]]` | module constant in env file | Clamps the commanded TCP position (world frame). |
| **Cameras** | `wrist_1`, `wrist_2`, `front` | both arena XMLs (`<camera .../>`) | Reposition/retarget the policy and monitoring views. |

> Tip: the tolerances (`success_lateral`, `success_depth`), randomization and
> `action_scale` are plain constructor arguments, so you can also override them
> per-call from `config.py`'s `get_environment()` (e.g. pass
> `success_lateral=0.02`) without editing the env file.

---

# Installation:
- From `serl` folder, cd into `franka_sim`.
- In your `serl` conda environment, run `pip install -e .` to install this package.
- run `pip install -r requirements.txt` to install sim dependencies.

# Explore the Environments
- Run `python franka_sim/test/test_gym_env_human.py` to launch a display window and visualize the task.

# Credits:
- This simulation is initially built by [Kevin Zakka](https://kzakka.com/).
- Under Kevin's permission, we adopted a Gymnasium environment based on it.

# Notes:
- Error due to `egl` when running on a CPU machine:
```bash
export MUJOCO_GL=egl
conda install -c conda-forge libstdcxx-ng
```
