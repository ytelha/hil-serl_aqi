# 04 — Python and programming paradigms, learned from this code

Part A lists the Python features this repository uses, each with a real
place to see it. Part B covers the **ways of structuring programs**
(paradigms and design patterns) the codebase is built from. Knowing the
names lets you explain design decisions in the defence ("the wrappers are
the decorator pattern") and reuse them later.

Good books to keep next to this page:
- Luciano Ramalho, *Fluent Python* (2nd ed.): the best deep Python book;
- Brett Slatkin, *Effective Python*: short, practical items;
- the official Python tutorial (docs.python.org/3/tutorial), for gaps.

---

## Part A — Python features, with where to see them

### A1. Data and control flow basics

| feature | example here | note |
|---|---|---|
| tuple unpacking | `obs, reward, terminated, truncated, info = env.step(a)` (every actor loop) | also `for i, r in enumerate(rows)`, `for a, b in zip(x, y)` |
| comprehensions | `{os.path.basename(p)[:-5]: _load(p) for p in sorted(glob.glob(...))}` in `examples/roster_audit.py` | list, dict and set comprehensions build a new collection in one expression |
| generator expressions | `any(not (...) for k, v in b.items())` in `run_campaign.freeze_roster` | lazy: stops at the first `True` |
| f-strings with format specs | `f"{sigma_q:.4f}"`, `f"{x:+.2f}"`, `f"{n:3d}"` | almost every print |
| `sorted(..., key=lambda ...)` | `sorted(glob.glob(...), key=lambda p: int(...))` | `key` decides the order |
| truthiness, `or` defaults | `ep.get("length") or 0`; `x if x is not None else default` | careful: `0` and `""` are falsy too |
| exceptions | `try: ... except (OSError, ValueError): return None` in `roster_audit._load`; `raise SystemExit("[config] ...")` in `run_campaign` | catch only what you expect; `SystemExit` stops a CLI with a message |

### A2. Functions

| feature | example here | note |
|---|---|---|
| default and keyword arguments | `def audit(camp, restore=False)` (`roster_audit.py`) | callers can write `audit(c, restore=True)` |
| keyword-only arguments (`*`) | `def random_crop(img, rng, *, padding)` (`vision/data_augmentations.py`) | after `*`, arguments must be named: prevents mix-ups |
| `*args, **kwargs` | `def __call__(self, *args, train=False, **kwargs)` in `ensemblize` | forwards whatever the wrapped module accepts |
| `lambda` and functions as values | `lambda obs, acts, key: self.forward_critic(obs, acts, rng=key)` passed into `e2hil_actor_loss` | higher-order functions: functions taking functions |
| closures and `nonlocal` | `def update_params(params): nonlocal agent; agent = agent.replace(...)` in the actor | the inner function rebinds a variable of the enclosing function; it's the callback agentlace calls |
| `functools.partial` | `@partial(jax.jit, static_argnames=("argmax",))` in `sac.py` | pre-fills arguments of a function (here: of `jax.jit`) |
| type hints | `def jitter_theta(theta: OperatorTheta, rng: np.random.Generator, ...) -> OperatorTheta`; `Optional[float]`, `Tuple[str, ...]` in `aqi/config.py` | documentation checked by editors (and optionally mypy); not enforced at run time |

### A3. Classes

| feature | example here | note |
|---|---|---|
| `@dataclass` | `AQIConfig` (`aqi/config.py`), `OperatorTheta`, `InterventionSegment` (`aqi/quality.py`), `HiRILConfig` | generates `__init__`, `__repr__`, `__eq__` from the annotated fields |
| `field(default_factory=list)` | `InterventionSegment.transitions` | **never** `= []` as a default: it would be shared by all instances (the classic mutable-default bug) |
| `__post_init__` validation | `AQIConfig.__post_init__` (asserts on modes, terms, ranges) | checks a config the moment it's built: fail fast |
| `dataclasses.replace` | `aqi_cfg = dataclasses.replace(aqi_cfg, **overrides)` in `train_aqi_rlpd.main` | copy with some fields changed; the original stays untouched |
| `@property` | `AQIConfig.active_on`, `quality_on`, `logging_on`, `baseline_arch` | computed attributes: `cfg.active_on`, no parentheses |
| `@staticmethod`, `@classmethod` | `FailureRiskModel.make_labels` (static); `SACAgent.create`, `create_pixels` (class methods = alternative constructors) | class methods receive the class (`cls`), static methods receive nothing |
| dunder methods | `__len__` in `InterventionSegment`, `KnnOODScorer`, `ReplayBuffer`; `__call__` in Flax modules and `MonitoringSignal` | `len(seg)` works; objects become callable |
| inheritance | `MujocoGymEnv → PandaPegInsertGymEnv → SimPegEnv`; `ReplayBuffer → MemoryEfficientReplayBuffer → ... → QualityAwareReplayBufferDataStore` | subclass = "is a kind of" |
| multiple inheritance (mixins) | `class ReplayBufferDataStore(ReplayBuffer, DataStoreBase)` in `data/data_store.py` | combines a buffer with agentlace's data-store interface; the order sets the method resolution order (`Cls.__mro__`) |
| informal interfaces | `PhaseOracle` methods `raise NotImplementedError`; `ToyInsertionOracle` and `SimPegPhaseOracle` implement them | "duck typing": anything with these methods works |
| class attributes | `PhaseOracle.lateral_dims = (0,)`, overridden by `SimPegPhaseOracle.lateral_dims = (0, 1)` | shared by all instances unless overridden |

### A4. Resources, modules, scripts

| feature | example here | note |
|---|---|---|
| context managers (`with`) | `with open(path) as f:`; `with timer.context("step_env"):` (`utils/timer_utils.py`) | setup/teardown guaranteed, even on errors; you can write your own |
| `if __name__ == "__main__":` | the bottom of every script; `app.run(main)` in the training scripts | the file works both as a script and as an importable module |
| packages and imports | `serl_launcher/serl_launcher/aqi/__init__.py`; `from serl_launcher.aqi.config import AQIConfig` | installed with `pip install -e` (editable: edits apply without reinstalling) |
| `pathlib.Path` | `campaign / "operators" / f"{op}.json"` (`run_campaign.py`) | the `/` operator is overloaded to join paths |
| `subprocess` | `run_campaign._launch` starts learner and actor as child processes; `proc.wait(timeout=...)` | how one Python program orchestrates others |
| JSON / JSON Lines | `aqi/logger.py` writes one JSON object per line | robust to crashes (every line is complete), easy to stream |

### A5. The Python pitfall that became finding F1

In `aqi/operator_model.jitter_theta`, the loop `for k in jkeys:` iterated a
**set of strings**. String hashing is randomised per process
(`PYTHONHASHSEED`), so a set's iteration order can differ between runs.
The seeded random draws were therefore assigned to different parameters
in every process. The fix is `for k in sorted(jkeys):`.

**Lesson:** never let the iteration order of a `set`, or a hash, decide
anything that must be reproducible. Dicts keep insertion order (since
Python 3.7), sets don't. `tests/test_roster_freeze.py` shows how to test
it: run the code in subprocesses with different `PYTHONHASHSEED` values.

---

## Part B — Paradigms and design patterns in this codebase

### B1. Functional core, imperative shell

The **core** is made of pure functions:
- the losses (`critic_loss_fn`, `policy_loss_fn`), `update`, the network
  forward passes;
- no side effects, same inputs → same outputs, explicit random keys;
- that's what JAX needs to `jit`, `grad` and `vmap` them.

The **shell** is ordinary imperative Python:
- the actor and learner loops;
- environment stepping, sockets, logging and files.

This split is the most important architectural idea in the repo. When you
add something, decide first whether it belongs in the pure core (then it
must be side-effect free and shape-stable) or in the shell.

### B2. Immutability

- `SACAgent` and `JaxRLTrainState` are frozen pytrees. An update returns a
  **new** agent: `new_agent, info = agent.update(batch)`.
- JAX arrays can't be modified in place (`x.at[i].set(v)` returns a copy).
- **Benefits:** no hidden aliasing bugs, safe to pass into compiled code,
  easy to keep old versions (the actor swaps params atomically).
- **Cost:** you must always use the returned value.

### B3. Object-oriented design where state is natural

Environments, wrappers, replay buffers, the AQI runtime and the synthetic
operator are stateful objects (an episode, a buffer, running statistics).

Two relationships to name in the defence:
- **inheritance:** "is a kind of" (a `SimPegEnv` is a `PandaPegInsertGymEnv`);
- **composition:** "has a". `AQIRuntime` *has* a gate, a force monitor, a
  scorer, a logger, an OOD scorer and a failure model. Each is a separate
  "section" that can be switched on or off (`aqi/runtime.py`, and the
  "composable" docs in `aqi_walkthrough.md`).

Composition is why the ablations are configuration, not code.

### B4. Design patterns you can name

| pattern | where | in one sentence |
|---|---|---|
| **Decorator** | Gymnasium wrappers (part 01 §4) | each wrapper adds one behaviour around the same interface; they stack |
| **Strategy** | `quality_score_mode`, `trigger_combiner`, `e2hil_q_agg` | the algorithm is chosen by a setting; the caller doesn't change |
| **Registry** (a dict of functions or classes) | `SEGMENT_FILTERS` (`aqi/quality.py`), `PIPELINES` and `ARM_KEYS` (`run_campaign.py`), `_TERM_KEYS` (`query_gate.py`) | adding an option means adding a dict entry |
| **Factory** | `utils/launcher.make_sac_pixel_agent`, `SACAgent.create_pixels` | builds a complex, correctly configured object from a few parameters |
| **Callback / Observer** | `client.recv_network_callback(update_params)`, `TrainerServer(request_callback=...)` | "call me when X happens" |
| **Producer–consumer** | actor → `QueuedDataStore` → learner's replay buffer | the two run at their own pace; a queue decouples them |
| **Publish–subscribe** | `server.publish_network(params)` | the learner broadcasts; any actor listening receives |
| **Null object** | `AQIRuntime.observe_step` returns a neutral record when disabled | callers don't need `if enabled:` everywhere |
| **Template method** | `gym.ObservationWrapper.step` calls your `observation()` | the base class fixes the algorithm, you fill in one step |

### B5. Configuration-driven experiments

There is one source of truth per setting, with a clear override order:
- the defaults live in dataclasses (`AQIConfig`, `DefaultTrainingConfig`,
  the task `config.py`);
- training scripts override them with absl flags;
- campaigns set them from YAML, and the campaign runner turns YAML into
  command lines.

The runner also documents **inert** settings (a knob that doesn't apply to
a mode) in every cell's README. Lesson: experiments you can't reproduce
from their config aren't experiments. The whole `provenance.py` /
`manifest.json` machinery exists for that.

### B6. Distributed systems, in miniature

The actor–learner split is a real distributed system:
- **message passing** instead of shared memory;
- **stale reads:** the actor acts with slightly old weights, until the
  next `publish_network`;
- **throughput mismatch:** the learner is free-running, so updates per env
  step vary (F3);
- **ports** as shared resources: clashes on a shared node, hence
  `AQI_TRAINER_PORT`;
- **failure handling:** the `STOP` sentinel file, timeouts and graceful
  shutdown in `run_campaign.train_cell`.

The same ideas scale up to large RL systems. Being able to name them is
valuable.

### B7. Pipelines, idempotency and reproducibility engineering

- **Idempotent stages:** a campaign's stages each skip work whose output
  already exists (`.done`, `cell.json`), so a crash just means "run it
  again" (resumability). `--shard i/n` splits the same pipeline across
  cluster jobs.
- **Frozen inputs:** the operator roster (since patch 0007) and the ruler
  (hash-locked) are frozen on first write, so every cell is measured the
  same way.
- **Provenance:** every artifact records what produced it.
- **Pre-registration:** decision rules are committed before results
  (`docs/counterfactual_score_study.md`).

These are research-engineering skills that transfer directly to any
data-heavy job.

### B8. Testing strategies used here

- **Unit tests of pure functions** with known answers, e.g. the quality
  window arithmetic in `tests/test_quality_scores_v2.py`.
- **Fakes and stubs:**
  - a fake critic with an analytic Q (`tests/test_cf_diag.py` →
    `FakeAgent`);
  - fake hardware modules (`cluster/claix/stubs/`);
  - a local stub runner for machines without JAX.

  They test the logic without the expensive or unavailable part.
- **Regression by equality:** after a change to `run_campaign.py`, every
  existing campaign must emit byte-identical commands. For comment-only
  patches, the Python parse tree (`ast.dump`) must be identical.
- **Tests as documentation:** the test names say what's guaranteed
  (`test_audit_never_restores_when_demo_seeds_disagree`).

### B9. Numerical habits worth copying

- **Running statistics** (Welford) instead of storing everything. But know
  their failure mode: all-time statistics never forget (F6).
- **Standardise before combining** signals with different units (z-scores
  in the gate).
- **Clip and guard:** `np.clip`, `max(x, 1e-6)`, treating NaN as missing,
  log-prob corrections for squashed distributions.
- **Pad to fixed shapes** for compiled code (`uncertainty.py`).
