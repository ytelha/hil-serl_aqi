# 03 — The libraries: concepts, where they're used, tiny examples

The exact versions of the training environment (Python 3.12, JAX, Flax,
MuJoCo 3, …) are pinned in
[`cluster/claix/requirements-lock.txt`](../../cluster/claix/requirements-lock.txt).

Run the examples in a Python shell or notebook inside that environment
(`./.venv/bin/python`, or the Jupyter extension in VS Code with the venv
selected). **Type them, don't copy them.**

---

## 1. NumPy: arrays for everything outside the neural networks

- **Concepts:**
  - `ndarray`, shapes and `axis`;
  - broadcasting;
  - vectorised operations instead of Python loops;
  - boolean masks;
  - `np.random.default_rng(seed)`: a seeded generator, the modern API.
- **Used in:** all of `aqi/` except `uncertainty.py` is plain NumPy (gate,
  quality, OOD, failure model, operator model, statistics), and so are the
  analysis scripts.
- **Try:**

  ```python
  import numpy as np
  rng = np.random.default_rng(0)
  x = rng.normal(size=(1000, 6))
  x.mean(axis=0).shape, (x > 1).mean(), np.linalg.norm(x, axis=1)[:3]
  ```

- **Pitfall:** integer vs float division, and views vs copies (`a[1:]` is a
  view). The repo uses `dtype=np.float64` for statistics and `float32` for
  network inputs.

## 2. JAX: NumPy that compiles, differentiates and vectorises

JAX is the engine of every network in this repo. Five ideas cover 90% of
it:

| idea | what it does | where here |
|---|---|---|
| `jax.numpy` | NumPy API on accelerator arrays | everywhere in `agents/`, `networks/` |
| `jax.jit` | traces a function once per input shape/type and compiles it with XLA | `SACAgent.update`, `sample_actions` (`@partial(jax.jit, static_argnames=...)`), `aqi/uncertainty.py` |
| `jax.grad` / `value_and_grad` | derivatives of a scalar function | inside `JaxRLTrainState.apply_loss_fns` (`common/common.py`) |
| `jax.vmap` | vectorise a function over a batch axis | Flax's `nn.vmap` in `ensemblize` (the K critics) |
| explicit random keys | no hidden global state: split a key for every use | the actor loop `sampling_rng, key = jax.random.split(sampling_rng)`; `fold_in` in the diagnostics |

**Pytrees:** any nested structure of dicts, lists, tuples and registered
classes whose leaves are arrays. The network parameters are a pytree.
`SACAgent` is one too (`flax.struct.PyTreeNode`), which is why it can be
passed straight into a jitted function. `jax.tree_util.tree_map(f, tree)`
applies `f` to every leaf (see `stack_obs` in `aqi/uncertainty.py`).

**Try:**

```python
import jax, jax.numpy as jnp
def loss(w, x, y):
    return jnp.mean((x @ w - y) ** 2)
grad_fn = jax.jit(jax.grad(loss))
w, x, y = jnp.zeros(3), jnp.ones((5, 3)), jnp.ones(5)
print(grad_fn(w, x, y))                        # compiled on first call
print(jax.vmap(lambda row: row.sum())(x))      # one call per row, vectorised
key = jax.random.PRNGKey(0)
key, sub = jax.random.split(key)
print(jax.random.normal(sub, (2,)))
```

**The sharp bits** (read the JAX docs page with that name):
- inside `jit`, Python `print` runs once, at trace time. Use
  `jax.debug.print`;
- Python `if` on a traced value fails. Use `jnp.where` / `lax.cond`, or
  mark the argument static (`static_argnames`, e.g. `argmax` in
  `sample_actions`);
- **every new input shape means a recompile.** That's why
  `counterfactual_advantage_batch` pads every chunk to the same size (32):
  it compiles once instead of once per segment length;
- reusing a random key gives identical "random" numbers: always `split`;
- default precision is float32;
- JAX preallocates GPU memory. `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
  `XLA_PYTHON_CLIENT_MEM_FRACTION` (set by the campaign runner and
  `cluster/claix/env.sh`) let learner and actor share one GPU.

**Docs:** https://docs.jax.dev: "Quickstart", "Thinking in JAX", "The
Sharp Bits", "Pytrees".

## 3. Flax (Linen): neural networks on top of JAX

- **Modules are dataclasses that describe computation.** They don't hold
  the parameters: the parameters live in a separate pytree.
  - `params = model.init(key, example_input)` creates them;
  - `model.apply(params, x)` runs the network.
- `@nn.compact` lets you declare sub-layers inline in `__call__`, as the
  repo's `MLP`, `Critic` and `Policy` do.
- `nn.vmap` turns one module into K independent copies (a leading
  parameter axis). `ensemblize` in `networks/actor_critic_nets.py` is
  exactly that.
- `flax.struct.PyTreeNode` makes immutable dataclasses that are pytrees:
  - `SACAgent` and `JaxRLTrainState` are two;
  - fields marked `struct.field(pytree_node=False)` (like `apply_fn`,
    `txs`) are static, not traced;
  - you never modify them. `.replace(...)` returns a new object (see
    `update_params` in the actor).
- Checkpoints: `flax.training.checkpoints.save_checkpoint` /
  `restore_checkpoint` (Orbax underneath: the `orbax` lines in the logs),
  used by the learner and by `eval_ruler.py`.

**Try:**

```python
import jax, jax.numpy as jnp, flax.linen as nn
class MLP(nn.Module):
    hidden: int = 32
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(nn.relu(nn.Dense(self.hidden)(x)))
m = MLP()
params = m.init(jax.random.PRNGKey(0), jnp.ones((4, 3)))
print(jax.tree_util.tree_map(jnp.shape, params))   # the parameter pytree
print(m.apply(params, jnp.ones((4, 3))).shape)      # (4, 1)

from serl_launcher.networks.actor_critic_nets import ensemblize
Ens = ensemblize(MLP, 5)                  # 5 independent MLPs, one module
p5 = Ens().init(jax.random.PRNGKey(1), jnp.ones((4, 3)))
print(Ens().apply(p5, jnp.ones((4, 3))).shape)       # (5, 4, 1): K first
```

Verified output: the parameter tree shows `Dense_0` with a `(32, 1)`
kernel. That's the **output** layer: Flax names sub-layers in the order
they're created, and Python creates `nn.Dense(1)` before it evaluates the
argument `nn.relu(nn.Dense(32)(x))`. Small details like this are why you
should print the parameter tree of every network you build.

**Docs:** https://flax.readthedocs.io (Linen basics; "Flax struct").

## 4. Optax (optimisers), distrax (distributions), chex (checks)

- **Optax:** an optimiser is a `GradientTransformation`:

  ```python
  import optax
  tx = optax.adam(3e-4)
  opt_state = tx.init(params)
  updates, opt_state = tx.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  ```

  Here, `common/optimizers.py` builds one per network (actor, critic,
  temperature); `JaxRLTrainState.apply_gradients` applies them.
- **distrax:** probability distributions in JAX.
  `TanhMultivariateNormalDiag` (`networks/actor_critic_nets.py`) is a
  Gaussian pushed through `tanh`. `sample_and_log_prob(seed=key)` gives
  actions and their log-probability, including the tanh correction.
  `.mode()` is the deterministic action used for evaluation
  (`argmax=True`).
- **chex:** `chex.assert_shape(x, (batch,))` catches shape bugs inside
  jitted code (see `sac.policy_loss_fn`).

## 5. Gymnasium: the environment interface

- `obs, info = env.reset(seed=...)`
- `obs, reward, terminated, truncated, info = env.step(action)`
- **`terminated`:** the task ended (success or failure);
  **`truncated`:** a time limit cut it off.
  - In this codebase the time limit sets `done` (mask 0); see
    `docs/quality_scores_v2.md` §2.1.
  - That's why the hindsight score and the audits treat episode ends the
    way they do.
- **Spaces:** `env.observation_space` (a `Dict` of `Box`es here),
  `env.action_space` (`Box(-1, 1, (6,))`).
- **Wrappers:** subclass `gym.Wrapper` (anything), `ObservationWrapper`
  (override `observation`), `ActionWrapper` (override `action`) or
  `RewardWrapper` (override `reward`). The whole stack is in part 01 §4.
- **Try it on the toy environment** (no MuJoCo needed):

  ```python
  import numpy as np, gymnasium as gym
  from serl_launcher.aqi.toy_insertion import ToyInsertionEnv
  class SlowDown(gym.ActionWrapper):
      def action(self, a):
          return 0.5 * np.asarray(a)
  env = SlowDown(ToyInsertionEnv(seed=0))
  obs, info = env.reset()
  obs, r, term, trunc, info = env.step(np.array([10.0, -20.0]))
  print(obs["state"][:4], r, term, trunc)
  ```

- **Docs:** https://gymnasium.farama.org (Basic usage, Wrappers).

## 6. MuJoCo: the physics

- `MjModel` is the robot and scene (from XML); `MjData` is the state;
  `mj_step` advances it; `Renderer` makes camera images.
- On headless machines the rendering goes through **EGL**
  (`MUJOCO_GL=egl`, set in `sim_env.sh`).
- `franka_sim/.../panda_peg_insert_gym_env.py` builds the task. Its
  sensors (`peg_tip_pos`, `hole_top_pos`, the wrist force) are what the
  geometric oracle reads (`SimPegPhaseOracle`).
- **Try:**

  ```python
  import mujoco
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><light pos="0 0 1"/><geom size="0.1"/></worldbody></mujoco>')
  d = mujoco.MjData(m)
  mujoco.mj_step(m, d)
  print(d.time)
  ```

  `cluster/claix/egl_probe.py` is a complete rendering example.
- **Docs:** https://mujoco.readthedocs.io (Overview, Python bindings).

## 7. agentlace: moving data and weights between processes

- The learner runs a `TrainerServer`:
  - it registers the replay buffers as named data stores
    (`register_data_store("actor_env", replay_buffer)`);
  - it answers requests (`request_callback`, e.g. the actor's
    `"send-stats"`);
  - it publishes new weights (`publish_network`).
- The actor runs a `TrainerClient`:
  - it fills local `QueuedDataStore`s;
  - `client.update()` ships them;
  - `recv_network_callback(fn)` is called with new weights.
- Underneath it's ZeroMQ (sockets), hence the two TCP ports per cell.
- Where: `examples/train_aqi_rlpd.py` (`actor`, `learner`);
  `utils/launcher.py` → `make_trainer_config`.
- Source and docs: `pip show agentlace` gives the project URL. Reading its
  `trainer.py` is a good exercise in distributed-systems code.

## 8. Configuration and command-line: absl flags, argparse, PyYAML

- **absl** (the training scripts):
  - `flags.DEFINE_integer("max_steps", ...)` at module level;
  - `FLAGS.max_steps` to read;
  - `FLAGS["x"].present` tells whether the user set it (used to build
    `AQIConfig` overrides);
  - `app.run(main)` parses the command line.
- **argparse** (`run_campaign.py`, the audits): `p.add_argument(...)`,
  `BooleanOptionalAction` for `--x` / `--no-x`.
- **PyYAML:** `yaml.safe_load(open(path))` gives nested dicts;
  `run_campaign.apply_config` maps them onto the argparse namespace.
  **Pitfall:** YAML 1.1 reads bare `off`, `no`, `on`, `yes` as booleans.
  `apply_config` handles that for `logging.wandb`.
- **Try:** read `train_aqi_rlpd.py` from the top: all `flags.DEFINE_*`,
  then `main` building `aqi_cfg`, then `app.run(main)` at the bottom.

## 9. Everything else you'll meet

| library | purpose | where |
|---|---|---|
| `tqdm` | progress bars; its `it/s` is how the cluster slowdown was diagnosed (F5) | actor/learner loops |
| `wandb` | experiment tracking (`off` / `offline` / `online`) | `utils/launcher.make_wandb_logger`, `docs/wandb.md` |
| `pickle` | demos (`demos.pkl`), the pretrained ResNet weights | `collect_synthetic_demos.py`, `vision/` |
| `json` / JSON Lines | one JSON object per line: easy to append, stream and grep | `aqi/logger.py` (`steps.jsonl`, `segments.jsonl`, `episodes.jsonl`) |
| `subprocess` | the campaign runner launches learner, actor, demos and eval as child processes | `run_campaign.py` → `_launch` |
| `pathlib` | file paths as objects (`campaign / "operators" / f"{op}.json"`) | `run_campaign.py` |
| `scipy`, `statsmodels` (optional) | classical tests, GLMM | `aqi/stats.py` |
| `opencv` (`cv2`) | optional live view windows; headless build on the cluster | `train_aqi_rlpd.py --render` |
| `matplotlib` | figures | `campaign_plots.py`, the audits |
