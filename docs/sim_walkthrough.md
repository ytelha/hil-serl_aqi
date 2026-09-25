# MuJoCo simulation walkthrough (`sim_peg_insertion`)

This guide runs the whole HIL-SERL / AQI-HIL-SERL pipeline **in simulation** —
no Franka, no ROS, no reward classifier. A MuJoCo peg-insertion environment
(`franka_sim`) is exposed through the exact same interface as the real robot
tasks, so `record_demos.py`, `train_rlpd.py`, `train_aqi_rlpd.py` and
`aqi_analysis.py` all work unchanged with `--exp_name=sim_peg_insertion`.

You can therefore:

1. collect demonstrations with a SpaceMouse,
2. train a vision policy with human takeovers (HIL-SERL),
3. run the multi-operator study to measure **operator dependency**, and
4. evaluate the **AQI-HIL-SERL** extension (active + quality-aware interventions).

---

## 0. What was added

| Path | What it is |
| --- | --- |
| `franka_sim/` | Ported MuJoCo Panda package (from rail-berkeley/serl) |
| `franka_sim/franka_sim/envs/panda_peg_insert_gym_env.py` | `PandaPegInsertGymEnv` — the peg-in-hole physics env |
| `franka_sim/franka_sim/envs/xmls/arena_peg.xml` | Hex peg + hex socket scene (exact reference mesh geometry), two `wrist_1`/`wrist_2` policy cams + a `front` 3/4 monitoring cam |
| `franka_sim/franka_sim/envs/xmls/meshes_peg/` | Peg/socket meshes (`peg1.obj`, `framing1-6.obj`) from the reference env |
| `examples/experiments/sim_peg_insertion/` | The experiment: `config.py`, `wrapper.py`, `sim_env.sh`, `run_*.sh` |
| `examples/experiments/mappings.py` | Registers `sim_peg_insertion` in `CONFIG_MAPPING` |

The environment reports **ground-truth success** from the simulator, so no
reward classifier needs to be trained. Observations are two 128×128 wrist camera
images (`wrist_1`, `wrist_2`, one on each side of the peg) plus a 19-D proprio
state; the action is a 6-D
Cartesian twist (gripper is held closed — fixed-gripper setup).

The peg and socket use the **exact hexagonal mesh geometry** from the reference
`assets/peg_in_hole.xml` (a ~65 mm hex prism into a matching 6-wall socket). The
peg is welded to the gripper. Because the fit is a tight hex, a successful
insertion requires aligning the peg **both in XY and in yaw** before pushing
down — pushing while misaligned jams the peg on the rim (this is the intended
task difficulty). Contacts are tuned so there is **no tunneling** (the tip never
passes the socket floor) and the wrist force/torque sensors register real
contact (~12 N nominal). Success = lateral error < 12 mm **and** insertion depth
> 50 mm, reported as ground truth by the simulator.

The `front` camera is a **3/4 isometric monitoring view** (front-right, angled
down at the socket) rendered at 640×480 by `env.render()`. It is what you watch
during teleoperation / training videos (`--save_video`) — not a policy input —
and is framed so both the lateral alignment and the insertion depth are easy to
read. The two `wrist_1`/`wrist_2` cameras (128×128) are the actual policy
observations.

---

## 1. One-time setup

Everything is already installed in the repo virtualenv at `.venv/`. The key
facts (in case you rebuild the environment):

- **Python** 3.12, venv at `<repo>/.venv`.
- **MuJoCo** 3.10.0, rendered headless with `MUJOCO_GL=egl`.
- **JAX** 0.4.35 + a coherent pin set: `flax==0.8.5 optax==0.2.3 chex==0.1.87
  distrax==0.1.5 orbax-checkpoint==0.5.16 ml_dtypes==0.4.1 numpy==1.26.4
  tensorflow==2.17.1 tensorflow-probability==0.24.0 tf-keras==2.17.0
  wandb>=0.30.0` (required for current API-key formats).
- In-repo packages installed editable: `serl_launcher`, `serl_robot_infra`,
  plus `franka_sim` on `PYTHONPATH`; `agentlace` from git.
- **SpaceMouse**: needs the `libhidapi` C library. If it is not installed
  system-wide, the extracted copy under
  `/tmp/hidapi_deb/extracted/usr/lib/x86_64-linux-gnu` is put on
  `LD_LIBRARY_PATH` by `sim_env.sh` (override with `HIDAPI_LIB_DIR`).

All of the above environment wiring is done for you by **`sim_env.sh`**, which
every `run_*.sh` script sources. To set up a shell manually:

```bash
cd <repo>/examples/experiments/sim_peg_insertion
source sim_env.sh          # activates venv + MUJOCO_GL + LD_LIBRARY_PATH + PYTHONPATH
```

### Sanity check

```bash
cd <repo>/examples/experiments/sim_peg_insertion
source sim_env.sh
cd ../..                   # -> examples/
python -c "
from experiments.mappings import CONFIG_MAPPING
env = CONFIG_MAPPING['sim_peg_insertion']().get_environment(fake_env=True)
print(env.observation_space)   # wrist_1/wrist_2 (1,128,128,3) + state (1,19)
print(env.action_space)        # Box(6)
env.reset(); env.step(env.action_space.sample()*0)
print('OK')
"
```

`fake_env=True` builds everything **except** the SpaceMouse wrapper, so it runs
without a device. `fake_env=False` (used by the actor and by `record_demos.py`)
opens the SpaceMouse and will raise `No device connected/supported!` if none is
plugged in.

---

## 2. Collect demonstrations (SpaceMouse)

Plug in the 3Dconnexion SpaceMouse, then:

```bash
cd <repo>/examples/experiments/sim_peg_insertion
source sim_env.sh
cd ../..                                   # run from examples/
python record_demos.py --exp_name=sim_peg_insertion --successes_needed=20
```

Drive the peg into the socket with the SpaceMouse. The zero base action means
the robot only moves while you intervene, so every recorded action is your
teleop. After 20 successful episodes a pickle is written to
`examples/demo_data/cylinder/human/20_demos_<timestamp>.pkl`.

Point the learner at that file via the `DEMO_PATH` variable in the next step
(the run scripts run from the task dir, so the path is
`../../demo_data/cylinder/human/<file>.pkl`).

---

## 3. Baseline HIL-SERL training (with takeovers)

HIL-SERL runs an **actor** and a **learner** as two processes that talk over
`agentlace`. Open two terminals.

**Terminal A — learner** (GPU/CPU, no SpaceMouse needed):

```bash
cd <repo>/examples/experiments/sim_peg_insertion
DEMO_PATH=../../demo_data/cylinder/human/20_demos_<timestamp>.pkl \
    bash run_learner.sh
```

**Terminal B — actor** (needs the SpaceMouse for takeovers):

```bash
cd <repo>/examples/experiments/sim_peg_insertion
bash run_actor.sh
```

The actor rolls out the current policy. **Grab control at any time by moving
the SpaceMouse** — the wrapper detects the deflection, hands control to you,
and records the intervention as expert data. Release to return control to the
policy. The learner updates the policy from the growing buffer and checkpoints
into `runs/cylinder_hilserl_op0_s42/`.

---

## 4. AQI-HIL-SERL training

`train_aqi_rlpd.py` is the extended trainer. `--aqi_mode` selects the method:

| mode | what it does |
| --- | --- |
| `hilserl` | plain HIL-SERL, byte-identical to the baseline (primary control) |
| `logged` | same training + K-critic ensemble + full trigger/intervention logging |
| `active` | adds the active query gate A(sₜ) that *asks* for help |
| `quality` | manual takeovers, but scored/filtered/weighted by intervention quality |
| `aqi` | full system = active + quality |

Two terminals again (example uses the full `aqi` mode):

**Learner:**

```bash
cd <repo>/examples/experiments/sim_peg_insertion
DEMO_PATH=../../demo_data/cylinder/human/20_demos_<timestamp>.pkl \
    bash run_learner_aqi.sh --aqi_mode=aqi
```

**Actor:**

```bash
cd <repo>/examples/experiments/sim_peg_insertion
bash run_actor_aqi.sh --aqi_mode=aqi --operator_id=op0
```

In `active`/`aqi` mode the actor will **prompt for a takeover** when the query
gate fires (e.g. rising contact force or high policy uncertainty). Per-episode
diagnostics are written to `aqi_logs/<run_name>/episodes.jsonl`.

Learner metrics can also be streamed to Weights & Biases. See
[wandb.md](wandb.md) for account setup, online/offline modes, available metrics,
and data export.

> **Calibration note.** The force/torque thresholds in
> `config.get_aqi_config()` (`force_warn=15`, `force_hard=30`, …) are
> **placeholder** values — MuJoCo wrist forces are not calibrated to real
> newtons. Run `--aqi_mode=logged` first, inspect the logged `f_norm`
> distribution, then set `force_warn`/`force_hard` before enabling the active
> gate. See the comment in
> `examples/experiments/sim_peg_insertion/config.py`.

---

## 5. Operator-dependency study

The goal is to measure how much final performance varies **across operators**
(and to show AQI reduces that variance). Run the same mode once per operator,
tagging each with a distinct `--operator_id` (and vary `--seed` for repeats):

```bash
# Operator 1, plain HIL-SERL baseline (via the diagnostic 'logged' control)
bash run_actor_aqi.sh --aqi_mode=logged --operator_id=alice --seed=0
# Operator 2
bash run_actor_aqi.sh --aqi_mode=logged --operator_id=bob   --seed=0
# ... then repeat all operators with --aqi_mode=aqi to compare
```

Each run logs under `aqi_logs/<exp>_<mode>_<operator>_s<seed>_.../`. The
`operator_id` is stored with every record, which is what the cross-operator
variance analysis consumes.

---

## 6. Offline analysis (RQ1–RQ3)

After collecting runs, analyze operator variance, intervention-quality
features, and trigger calibration:

```bash
cd <repo>/examples/experiments/sim_peg_insertion
source sim_env.sh
cd ../..
python aqi_analysis.py --log_root aqi_logs --baseline_mode logged \
    --metric final_success_rate --out aqi_analysis_out
```

Outputs `aqi_analysis_out/summary.md`, `results.json` and figures:
- **RQ1** operator-variance reduction (AQI vs baseline),
- **RQ2** which intervention features predict quality,
- **RQ3** trigger calibration vs actual failures.

`examples/experiment_pipeline.py` and `examples/eval_ruler.py` provide the
higher-level orchestration / evaluation entry points if you want to script the
whole study.

---

## 7. GPU note (important)

The pinned `jax==0.4.35` CUDA build targets older GPU architectures. The
machine's **RTX PRO 2000 (Blackwell, sm_120)** likely needs a newer JAX + CUDA
12.8. The integration above is validated on **CPU** (`JAX_PLATFORMS=cpu`);
real training throughput needs a working GPU JAX. If GPU training fails with a
`no kernel image is available` / `ptxas` / sm_120 error, upgrade the JAX stack
(e.g. `pip install -U "jax[cuda12]"`) — note that a newer JAX may require small
code updates (e.g. `jax.tree_map` → `jax.tree_util.tree_map`).

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No device connected/supported!` | SpaceMouse not plugged in, or `libhidapi` not on `LD_LIBRARY_PATH`. Only the **actor** / `record_demos.py` need it. |
| `hid_enumerate not found` | `libhidapi` C library missing — set `HIDAPI_LIB_DIR` (see `sim_env.sh`). |
| Blank/failed rendering | Ensure `MUJOCO_GL=egl` (headless) or `glfw` (with display) is exported. |
| `ImportError: pyrealsense2 / gym` | Run from a shell that sourced `sim_env.sh` (sets `PYTHONPATH`). |
| GPU kernel/sm_120 error | See section 7 — upgrade JAX for Blackwell. |
