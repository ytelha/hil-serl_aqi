# 01 — Repository map: what lives where

Start with [`getting_started.md`](../getting_started.md) (plain words, 20
minutes) and [`architecture.md`](../architecture.md) (the five roles of the
scripts). This page adds the view from the code: layers, directories,
**where each algorithm is implemented**, the wrapper chain around the
simulator, and one training step traced end to end.

## 1. The layers, bottom to top

```
  analysis        examples/aqi_analysis.py, quality_signal_audit.py, cf_score_analysis.py,
                  roster_audit.py, campaign_plots.py, operator_skill.py, q_summary.py ...
  orchestration   examples/run_campaign.py (synthetic campaigns)
                  examples/experiment_pipeline.py (human study)   cluster/claix/ (Slurm)
  evaluation      examples/eval_ruler.py (frozen policy on a fixed pose grid)
  data producers  examples/record_demos.py (SpaceMouse), collect_synthetic_demos.py
  training        examples/train_rlpd.py (original HIL-SERL)
                  examples/train_aqi_rlpd.py (the thesis: AQI modes, E2HiL)
                  examples/train_hiril.py (HiRIL baseline)   train_bc.py, train_hgdagger.py
  task configs    examples/experiments/<task>/config.py + wrapper.py  (sim_peg_insertion)
  thesis library  serl_launcher/serl_launcher/aqi/   (gate, quality, operators, stats)
                  serl_launcher/serl_launcher/hiril/ (HiRIL agent)
  RL library      serl_launcher/serl_launcher/  agents/ networks/ vision/ data/ common/ utils/ wrappers/
  environments    franka_sim/ (MuJoCo Panda sim)   serl_robot_infra/franka_env/ (real robot + wrappers)
```

Three packages are installed into the environment (`pip install -e`, see
each `setup.py`): `serl_launcher`, `franka_sim` and `serl_robot_infra`
(which provides `franka_env`). The `examples/` scripts import them like any
library. That's why `from serl_launcher.aqi.config import AQIConfig` works
from anywhere.

## 2. Directory tour

| path | what it contains | origin |
|---|---|---|
| `franka_sim/franka_sim/envs/panda_peg_insert_gym_env.py` | `PandaPegInsertGymEnv`: the MuJoCo peg-insertion task (`reset`, `step`, `_get_obs`, `_render_images`, `_compute_reward`) | SERL, extended |
| `franka_sim/franka_sim/mujoco_gym_env.py` | `MujocoGymEnv` base class (model loading, rendering spec) | SERL |
| `serl_robot_infra/franka_env/envs/` | the real-robot env (`franka_env.py`) and the generic wrappers (`wrappers.py`, `relative_env.py`) | HIL-SERL |
| `serl_robot_infra/robot_servers/` | servers that talk to the real Franka/gripper (not used in sim) | HIL-SERL |
| `serl_launcher/serl_launcher/agents/continuous/` | `sac.py` (SACAgent), `bc.py`, `sac_hybrid_*.py` (learned gripper), `e2hil.py` | SERL + thesis |
| `serl_launcher/serl_launcher/networks/` | `mlp.py`, `actor_critic_nets.py` (Critic, Policy, `ensemblize`, tanh-Gaussian) | SERL |
| `serl_launcher/serl_launcher/vision/` | ResNet encoders (`resnet_v1.py`, pretrained ResNet-10), spatial embeddings, image augmentations | SERL |
| `serl_launcher/serl_launcher/common/` | `JaxRLTrainState` (params + target params + optimisers + RNG), `encoding.py` (images + proprio → features), typing | SERL |
| `serl_launcher/serl_launcher/data/` | replay buffers, and the data stores that agentlace ships between processes, incl. the quality-aware and HiRIL PER variants | SERL + thesis |
| `serl_launcher/serl_launcher/wrappers/` | `SERLObsWrapper`, `ChunkingWrapper`, video wrappers | SERL |
| `serl_launcher/serl_launcher/utils/launcher.py` | factory functions: `make_sac_pixel_agent`, `make_hiril_pixel_agent`, `make_trainer_config`, `make_wandb_logger` | SERL + thesis |
| `serl_launcher/serl_launcher/aqi/` | **the thesis method**: config, runtime, gate, uncertainty, failure, force, OOD, quality, consistency, synthetic operators, logger, statistics | thesis |
| `serl_launcher/serl_launcher/hiril/` | HiRIL agent, config, priorities | thesis (baseline) |
| `examples/experiments/sim_peg_insertion/` | the task config (`config.py`: env + wrappers, image keys, training defaults) and task wrappers + oracle (`wrapper.py`) | thesis |
| `examples/campaigns/*.yaml` | every campaign that was or can be run | thesis |
| `tests/` | unit and integration tests (plain `python tests/<file>.py` or pytest) | thesis |
| `docs/`, `docs/thesis/` | design docs and the RQ logs with results | thesis |
| `cluster/claix/` | running on RWTH CLAIX: env setup, Slurm scripts, stubs | thesis |
| `handoff/` | patches + instructions for the training machine (not in git) | thesis |

## 3. Where each algorithm lives

| algorithm / concept | file → function or class |
|---|---|
| **SAC critic loss** (TD target, min over subsampled critics, per-member targets) | `agents/continuous/sac.py` → `SACAgent.critic_loss_fn` |
| **SAC actor loss** (Q − α log π, mean over critics) + weighted BC term | `sac.py` → `SACAgent.policy_loss_fn` (BC part: `aqi_bc_weight`, `quality_weight`) |
| **SAC temperature** (automatic entropy tuning) | `sac.py` → `temperature_loss_fn`, `forward_temperature` |
| one gradient step for chosen networks, target update | `sac.py` → `SACAgent.update` (jitted) |
| building the pixel agent (encoder, network sizes, K, M, LR) | `utils/launcher.py` → `make_sac_pixel_agent`; `sac.py` → `SACAgent.create_pixels` |
| **critic ensemble** (K critics as one vmapped module) | `networks/actor_critic_nets.py` → `ensemblize`, `Critic` |
| tanh-squashed Gaussian policy | `actor_critic_nets.py` → `Policy`, `TanhMultivariateNormalDiag` |
| image encoder (frozen pretrained ResNet-10) | `vision/resnet_v1.py` → `PreTrainedResNetEncoder`; `common/encoding.py` → `EncodingWrapper` |
| image augmentation (random crop, DrQ-style) | `vision/data_augmentations.py` → `batched_random_crop`; used via `utils/launcher.py` → `make_batch_augmentation_func` |
| **RLPD / HIL-SERL learner** (50/50 online + demo/intervention batches, `cta_ratio` critic steps per actor step) | `examples/train_aqi_rlpd.py` → `learner` (and the original `train_rlpd.py`) |
| **actor loop** (act, step env, route interventions, send data) | `train_aqi_rlpd.py` → `actor` |
| replay buffers; per-member bootstrap masks | `data/replay_buffer.py`, `data/memory_efficient_replay_buffer.py` |
| **quality-aware prioritised replay** (priority from TD error + q_int) | `data/data_store.py` → `QualityAwareReplayBufferDataStore` |
| **AQI runtime** (ties all AQI sections into the actor loop) | `aqi/runtime.py` → `AQIRuntime.observe_step`, `record_transition`, `end_episode` |
| **active query gate** A(s) | `aqi/query_gate.py` → `ActiveQueryGate.step`; normaliser `aqi/normalization.py` → `RunningZScore` |
| ensemble uncertainty σ_Q, counterfactual Â | `aqi/uncertainty.py` → `q_stats_single`, `counterfactual_advantage_batch` (+ diagnostics, patch 0010) |
| failure predictor P_fail | `aqi/failure.py` (online logistic model) |
| force risk, hard limits, jerk | `aqi/force_risk.py` |
| novelty (OOD) | `aqi/ood.py` → `KnnOODScorer` |
| **quality scores** (counterfactual, behavioral, hindsight, progress, outcome, uniform), filters, q → w | `aqi/quality.py` → `QualityScorer`, `SEGMENT_FILTERS`, `hindsight_advantage`; mode switch in `runtime._segment_score` |
| operator-consistency weight (σβ) | `aqi/consistency.py` |
| all AQI knobs and run modes | `aqi/config.py` → `AQIConfig` (`active_on`, `quality_on`, `logging_on`, `baseline_arch`) |
| **E2HiL** entropy-guided state selection | `agents/continuous/e2hil.py` → `influence_per_state`, `entropy_bounded_mask`, `e2hil_actor_loss` (called from `sac.policy_loss_fn`) |
| **HiRIL** (Bayesian IQN critic, dual uncertainty, TD3-style actor) | `hiril/agent.py`; `hiril/config.py`; `hiril/priorities.py`; loop in `examples/train_hiril.py`; buffer `data_store.RiskAwarePERDataStore` |
| **synthetic operator** (perception → decision → motor) | `aqi/operator_model.py` → `OperatorTheta`, `SyntheticOperator`, `PhaseOracle`, `jitter_theta` |
| when the operator takes over; the SpaceMouse stand-in | `aqi/operator_gate.py` → `MonitoringSignal`, `SyntheticInterventionGate`, `SyntheticOperatorWrapper` |
| geometric oracle for the sim task | `examples/experiments/sim_peg_insertion/wrapper.py` → `SimPegPhaseOracle` |
| skill assessment from behaviour | `aqi/skill.py`, `examples/operator_skill.py` |
| **operator-dependency statistics** (ICC, permutation test, bootstrap, RG_rel, Wilson) | `aqi/stats.py` → `icc_oneway`, `permutation_sigma2`, `bootstrap_icc`, `robustness_gain`, … |
| the frozen evaluation ruler | `examples/eval_ruler.py` (+ `aqi/stats.cell_outcomes`) |
| **campaigns** (roster, demos, train, eval, analyze) | `examples/run_campaign.py` → `sample_operators` / `freeze_roster` / `build_grid`, `collect_demos_job`, `train_cell`, `eval_cell`, `analyze`; stages run from `main` (search `STAGE`) |

## 4. The environment, wrapper by wrapper

A Gymnasium **wrapper** takes an environment and changes one aspect of it
(its actions, observations or rewards) without touching the rest. Wrappers
are stacked like layers of an onion. For `sim_peg_insertion`
(`examples/experiments/sim_peg_insertion/config.py` → `get_environment`),
from the outside in:

| # | wrapper | file | what it does |
|---|---|---|---|
| 1 | `SyntheticOperatorWrapper` | `aqi/operator_gate.py` | added by `train_aqi_rlpd.py` when a synthetic operator is used. Decides takeovers; when in control, replaces the action and puts it in `info["intervene_action"]`, exactly where the SpaceMouse would |
| 2 | `RecordEpisodeStatistics` | gymnasium | episode return/length in `info["episode"]` |
| 3 | `SimPegRewardWrapper` | `experiments/sim_peg_insertion/wrapper.py` | the task's success/reward definition |
| 4 | `ChunkingWrapper` | `serl_launcher/wrappers/chunking.py` | stacks observations over time (`obs_horizon=1` here → a leading axis of size 1) |
| 5 | `SERLObsWrapper` | `serl_launcher/wrappers/serl_obs_wrappers.py` | flattens the proprio keys into one `state` vector; keeps images |
| 6 | `Quat2EulerWrapper` | `franka_env/envs/wrappers.py` | quaternion → Euler angles in the observation |
| 7 | `RelativeFrame` | `franka_env/envs/relative_env.py` | actions and poses relative to the end-effector frame / episode start |
| 8 | `SpacemouseIntervention` | `franka_env/envs/wrappers.py` | real human takeover (**skipped in sim**, `SIM_DISABLE_SPACEMOUSE=1`) |
| 9 | `GripperCloseEnv` | `franka_env/envs/wrappers.py` | keeps the gripper closed (the peg is held) |
| 10 | `SimPegEnv` → `PandaPegInsertGymEnv` | `wrapper.py` / `franka_sim/.../panda_peg_insert_gym_env.py` | the physics |

What comes out at the top: `obs = {"state": (1, 19), "wrist_1": (1, 128,
128, 3), "wrist_2": (1, 128, 128, 3)}`, plus 6-D actions in [−1, 1].

You can see this stack in any crash traceback. The one in
`handoff/FINDINGS_cluster_method_test.md` F5 goes
`operator_gate.step → record_episode_statistics → wrapper.py → chunking → … → panda_peg_insert_gym_env.step`.

## 5. Actor and learner: two processes that talk

HIL-SERL splits training into two processes, usually on one GPU:

```
 ACTOR (train_aqi_rlpd.py --actor)                LEARNER (train_aqi_rlpd.py --learner)
 ─────────────────────────────────                ─────────────────────────────────────
 env (+ synthetic operator) at 10 Hz             replay_buffer   (online transitions)
 agent.sample_actions(obs)                        demo_buffer     (demos + interventions)
 AQIRuntime: gate, logging, quality scoring       loop: sample 1/2 batch from each buffer,
 data_store / intvn_data_store (QueuedDataStore)        agent.update(...)  (jitted SAC step)
        │   transitions (agentlace, ZeroMQ)             │
        └──────────── TrainerClient ─────────────► TrainerServer.register_data_store(...)
        ◄──────────── new network params ────────  server.publish_network(params) every
                      client.recv_network_callback        `steps_per_update` steps
```

- The actor never computes gradients. It only acts, and ships transitions.
- The learner never touches the environment. It trains as fast as the GPU
  allows. That's why updates per env step differ by method and GPU
  (finding F3).
- They talk over TCP ports (`make_trainer_config`, per-job ports on the
  cluster via `AQI_TRAINER_PORT`). When the learner publishes new
  parameters, the actor's callback swaps them into its (immutable) agent:
  `agent = agent.replace(state=agent.state.replace(params=params))`.

## 6. One step, traced end to end

**Actor, `train_aqi_rlpd.py` → `actor`, inside `for step in pbar`:**
1. `agent.sample_actions(obs, seed=key)` → the policy's action `a_robot`
   (random during `random_steps`).
2. `aqi.observe_step(agent, obs, a_robot, ...)` (`aqi/runtime.py`):
   - computes σ_Q, P_fail, force risk, novelty;
   - calls the gate → `request`;
   - stores a pending step record.
3. `env.notify_request(...)`: in `active`/`aqi` the synthetic operator
   hears the request.
4. `env.step(action)` runs down the wrapper stack. If the operator took
   over, `info["intervene_action"]` holds its action, which becomes the
   stored action.
5. A transition `dict(observations, actions, next_observations, rewards,
   masks, dones)` is built:
   - `aqi.record_transition(...)` completes the step record (written to
     `steps.jsonl`) and collects takeover steps into a segment;
   - `data_store.insert(transition)` ships every transition to the online
     buffer;
   - in non-quality modes, intervention transitions also go straight to
     `intvn_data_store`.
6. At the episode end, `aqi.end_episode(...)` scores each takeover segment
   (quality modes), writes `segments.jsonl` / `episodes.jsonl`, and in
   quality modes sends the scored intervention transitions (with `q_int`,
   `quality_weight`) to the intervention buffer.
7. Every few steps `client.update()` flushes the queues to the learner.

**Learner, `train_aqi_rlpd.py` → `learner`:**
1. Waits until `training_starts` transitions exist.
2. Each learner step:
   - `cta_ratio − 1` critic-only updates, then one full update (critic +
     actor + temperature);
   - each on a batch that is half online (`replay_iterator`) and half
     demo/intervention (`demo_iterator`);
   - with quality on, the second half is sampled by priority, with
     importance weights.
3. Every `steps_per_update` steps: `server.publish_network(agent.state.params)`.
4. Saves checkpoints to `checkpoint_path`; the campaign runner stops it with a `STOP` file.

**Exercise:** put breakpoints (or `print`s) at each numbered point, run the
smallest test campaign (`examples/campaigns/fast_test.yaml`, or a dry run
first), and check you see them in this order.

## 7. Where results end up

The campaign folder layout (`runs/<name>/`: `operators/`, `demos/`,
`checkpoints/<cell>/`, `aqi_logs/<cell>_actor/`, `eval_runs/<cell>/`,
`logs/`, `report/`) is explained in
[`campaign_runner.md`](../campaign_runner.md). Every folder has a
`README.md` and `manifest.json` saying what produced it. Learn to read
them; that's how you trace any number in the thesis back to its run
([`architecture.md`](../architecture.md) §4).
