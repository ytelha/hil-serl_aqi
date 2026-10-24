# Synthetic operators, operator‑dependency, and the campaign

*The methods in detail.* This document explains **how the synthetic operators
are established**, **how operator‑dependency is measured** (both *within‑operator*
and *within‑method*), and **how the automated campaign ties it together**. If
you just want to run the code, start with [getting_started.md](getting_started.md)
and come back here for the "why".

---

## 0. The question this machinery answers

HIL‑SERL trains a robot policy while a human can take over (intervene) whenever
they judge the robot is about to fail. The trained policy therefore depends, in
part, on **who** supervised it — a careful expert and a jumpy novice will not
teach the same thing. This project asks:

> **How much does the final policy depend on which operator trained it, and does
> AQI (Active + Quality‑aware Intervention) reduce that dependence versus plain
> HIL‑SERL?**

To answer it rigorously we need (a) **operators we can grade and reproduce**,
(b) **a measuring stick that does not itself depend on the operator**, and
(c) **a variance decomposition** that separates "the operator effect" from
"run‑to‑run noise". Those are the three parts below.

---

## 1. How the synthetic operators are established

A *synthetic operator* is a small, seeded program that stands in for a human
teleoperator. It plugs in at exactly the point a SpaceMouse would
(`info["intervene_action"]`), so nothing downstream can tell it from a person.
Everything about it is governed by six design requirements (implemented in
[`serl_launcher/serl_launcher/aqi/operator_model.py`](../serl_launcher/serl_launcher/aqi/operator_model.py)):

| # | requirement | meaning |
|---|---|---|
| R1 | ordinal validity | a higher skill `s` yields *stochastically better* outcomes |
| R2 | ground truth | every operator carries its true skill `s` and parameters `θ` |
| R3 | human‑plausible errors | mistakes come from delayed perception, mis‑alignment, motor noise, tremor, hesitation and lapses — **never** i.i.d. action noise |
| R4 | both roles | the **same** `θ` drives demonstrations *and* live interventions |
| R5 | non‑invasiveness | it enters at the SpaceMouse hook; the rest of the stack is unchanged |
| R6 | determinism | same seed ⇒ byte‑identical episodes |

> **Deep dive:** [oracle_and_synthetic_operator.md](oracle_and_synthetic_operator.md)
> unpacks the oracle → synthetic-operator design in full — the competent-core
> oracle (geometric vs policy), the perception → decision → motor pipeline, and
> exactly where noise is injected.

### 1.1 Two independent skill axes

Human skill is not one number. We keep two axes separate throughout:

- **Execution (motor / perception)** — how well they *move the tool*: speed,
  steadiness, alignment accuracy, recovery style.
- **Monitoring (supervisory)** — how well they *decide when to take over*:
  detection threshold, reaction delay, miss rate, false‑alarm rate.

This separation lets us build mixed profiles like *"skilled hands but jumpy
supervision"* (great execution, trigger‑happy monitoring) — which turns out to
matter a lot for HIL‑SERL.

### 1.2 `θ` — the parameter vector

All knobs of both axes live in one dataclass, `OperatorTheta`. The important
ones:

**Execution axis**

| param | meaning | units |
|---|---|---|
| `sigma0` | baseline motor noise | mm/s |
| `k_sdn` | signal‑dependent noise gain (noise grows with speed) | — |
| `sigma_tr` | Ornstein–Uhlenbeck tremor intensity (band‑limited wobble) | mm/s |
| `latency` | perception→action delay | control steps @10 Hz |
| `lag` | first‑order motor lag (smoothing of commands) | — |
| `v_max` | speed cap | mm/s |
| `sigma_perc` | per‑**approach** perceptual bias (mis‑estimate of alignment) | mm |
| `p_pause`, `pause_len_mean` | hesitation frequency / duration | /step, s |
| `q_rec` | probability of retract‑and‑realign on a jam (vs push through) | /jam‑step |
| `p_early` | premature‑commit probability | /attempt |
| `p_lapse` | whole‑episode lapse probability (inflates all noise ×2.5) | /episode |

**Monitoring axis**

| param | meaning |
|---|---|
| `sigma_mon` | noise on the risk estimate `z ∈ [0,1]` |
| `delta_in` | take‑over trigger threshold |
| `m_in` | consecutive steps above `delta_in` before acting |
| `l_react` | reaction delay after detection (steps) |
| `p_miss` | probability of missing a true failure event |
| `rho_fa` | false alarms per minute |
| `delta_out`, `k_out` | hand‑back threshold and consecutive steps below it |
| `d_max` | maximum take‑over duration (s) |

Plus the ground truth: `s_exec`, `s_mon`, `tier`, `operator_id`.

### 1.3 Tier anchors

Three reference points anchor the scale. `expert = s 1.0`, `intermediate = 0.5`,
`novice = 0.0`. A representative slice:

| | expert | intermediate | novice |
|---|---|---|---|
| `sigma0` (motor noise, mm/s) | 1.5 | 5.0 | 10.0 |
| `v_max` (mm/s) | 90 | 70 | 55 |
| `latency` (steps) | 1 | 2 | 4 |
| `sigma_perc` (mm) | 0.30 | 1.0 | 2.2 |
| `q_rec` (recover vs shove) | 0.60 | 0.30 | 0.08 |
| `p_lapse` | 0.03 | 0.06 | 0.15 |
| `delta_in` (trigger) | 0.45 | 0.60 | 0.80 |
| `l_react` (steps) | 3 | 5 | 8 |
| `p_miss` | 0.05 | 0.15 | 0.35 |
| `rho_fa` (/min) | 0.2 | 0.5 | 1.2 |

Read it as: the expert is fast, steady, aligns well, recovers gracefully,
detects trouble early and rarely misses; the novice is slow, noisy, mis‑aligns,
shoves on jams, reacts late and misses a third of true failures.

### 1.4 The continuous skill scale `θ(s)`

Between the anchors, `theta_from_scale(s_exec, s_mon)` interpolates:

- **scale‑like** parameters (`sigma0`, `v_max`, `sigma_perc`, `rho_fa`, …) are
  interpolated **geometrically** — equal steps in `s` are equal *ratios*, which
  keeps them positive and perceptually even;
- **probabilities and criteria** (`p_miss`, `delta_in`, …) are interpolated
  **linearly**; integer parameters are rounded.

So you can ask for any `s = 0.7` operator, or a mixed `exec=expert, mon=novice`.

### 1.5 Distinct operators within a tier — jitter

If every "expert" were identical, a tier would be a single robot and there would
be no *between‑operator* variance to measure. `jitter_theta(θ, rng, strength)`
multiplies each scale‑like parameter by `exp(𝒩(0, strength²))` (default
`strength = 0.15`), turning each tier into an **overlapping cloud** of genuinely
different operators. This is exactly how the campaign builds its roster
([`sample_operators`](../examples/run_campaign.py)):

```python
anchor = make_operator_theta(tier)          # expert / intermediate / novice
for i in range(ops_per_tier):
    th = jitter_theta(anchor, rng, strength=jitter)   # a distinct operator
    th.operator_id = f"{abbr}{i:02d}"                 # exp00, exp01, …
```

The **same** jittered `θ` is saved once (`operators/<id>.json`) and reused for
both that operator's demonstrations and its live interventions (R4).

### 1.6 How an operator acts — perception → decision → motor

`SyntheticOperator.act(obs)` runs three blocks every control step:

1. **Perception** — it acts on the observation from `latency` steps ago, and its
   alignment estimate carries a per‑**approach** bias `η ~ 𝒩(0, sigma_perc²)`.
   Crucially the bias is resampled only when the operator *re‑examines the scene*
   (after a retract), never every step — otherwise the mis‑estimate would average
   away and erase tier differences.
2. **Decision** — a phase machine `approach → align → descend`. It commits to
   descend when the **perceived** alignment error is inside a threshold (≈ 0.8 ×
   clearance); it may commit prematurely (`p_early`); after a jam it retracts and
   realigns with probability `q_rec`, else shoves through (the novice failure
   mode). *(With a policy‑oracle core instead of the phase machine, the same
   idea appears as temporally‑correlated "wrong‑intent" segments.)*
3. **Motor** — first‑order lag, signal‑dependent Gaussian noise, OU tremor,
   speed cap `v_max`, geometric hesitation pauses, and whole‑episode lapses:
   `aₜ = (1−lag)·aₜ₋₁ + lag·(u + ε + tremor)`.

The result is trajectories that *look* human in smoothness and error structure
(R3), not a clean path plus white noise.

### 1.7 The two roles

- **Role 1 — demonstrations.** The operator teleoperates full episodes from
  reset to termination; successful (or all) episodes are recorded in
  training format. This is what [`collect_synthetic_demos.py`](../examples/collect_synthetic_demos.py)
  does.
- **Role 2 — interventions.** During RL, a **monitoring gate** watches a risk
  signal `zₜ` and decides *when* to take over, using signal‑detection‑theory
  parameters (`delta_in`, `m_in`, `l_react`, `p_miss`, `rho_fa`, `delta_out`,
  `k_out`, `d_max`). When it fires, the *same* execution model produces the
  correction. Implemented in
  [`operator_gate.py`](../serl_launcher/serl_launcher/aqi/operator_gate.py).

Because both roles share one `θ`, an operator's demonstrations and its rescues
are consistent — a jumpy novice both *teaches* wobbly successes and *intervenes*
late and imperfectly.

### 1.8 How the operator "sees" the simulator

In simulation the operator's geometric primitives come from
`SimPegPhaseOracle` ([`wrapper.py`](../examples/experiments/sim_peg_insertion/wrapper.py)),
which reads **live** sensors (peg‑tip pose, socket top/bottom, wrist force) and
computes the lateral error in the end‑effector frame, the height above the
socket, and a force‑based jam signal. For the cylinder the peg radius is 20 mm
and the bore 28 mm (an **8 mm** radial clearance); one action unit = 0.01 m per
step = 100 mm/s at 10 Hz. This is why an expert inserts in ~13 control steps and
a novice takes ~50 — the same geometry, filtered through different `θ`.

> **Supported geometry.** By default the synthetic operator commands translation
> only, so the **cylinder** (no yaw alignment needed) is the supported task;
> cuboid/hex need yaw and insert less reliably. Its motor noise still reaches the
> rotation dims, so a twist from the policy or from its own tremor is never
> undone. `--synthetic_rotation_control` (YAML `aqi.synthetic_rotation_control`)
> adds orientation control: the operator turns the tool back to the nearest
> insertable orientation (90° symmetry for the cuboid, 60° for the hex, yaw free
> for the cylinder), commits only once position and orientation are aligned, and
> treats a twisted tool in contact like a jam. Off by default, so every earlier
> campaign keeps its operator; see [oracle_and_synthetic_operator.md §6.4](oracle_and_synthetic_operator.md).
> Two further optional pieces (§6.6 there): an orientation box for the sim env
> (`--sim_rpy_box`, YAML `task.rpy_box`), which clips roll, pitch and yaw like the
> real robots' safety box, and an orientation channel in the operator's monitor
> (`--synthetic_orientation_monitor`), which takes over a twisted but centred peg
> before it jams. `examples/oracle_insertion_test.py --sweep tilt,yaw` measures
> the orientation tolerance that sets the channel's limits.

---

## 2. The fixed "ruler" — an operator‑independent measuring stick

To compare operators fairly we must score their trained policies **the same
way**, with no operator in the loop. [`eval_ruler.py`](../examples/eval_ruler.py)
does this:

- the policy is **frozen** and run with `argmax` (deterministic);
- there is **no operator** and **no randomization** — every trial uses a fixed
  start pose from a **hash‑locked grid** (a sweep of lateral offsets `dx` and yaw
  offsets `dyaw`);
- the grid is content‑addressed with SHA‑256 and the runner refuses a grid whose
  hash does not match, so *every* cell in a campaign is measured on the
  byte‑identical ruler.

On the simulator this is enabled by `set_eval_reset_pose(dx_m, dyaw_rad)`, which
pins each reset deterministically (randomization off). Training, by contrast,
uses randomized starts (±15 mm / ±0.08 rad) seeded per replicate — so the policy
sees variety while the exam is fixed.

The output per policy is a `cell.json` with the **success rate (SR)** and force
statistics.

---

## 3. The dependency assessment

The unit of analysis is a **cell** = `(method, operator, seed)`. Each cell is
one full training run scored by the ruler, giving one SR. For each operator we
train several **seed replicates**; across a tier we have several **operators**.
That nesting is what the variance decomposition exploits.

### 3.1 Within‑operator variance (`σ²_within`)

Fix one operator and one method. Train it several times with different seeds
(different reset sequences, exploration noise, and the operator's own stochastic
interventions). The spread of SR across those seeds is the **within‑operator
variance** — the *irreducible run‑to‑run noise* when nothing changes but the
seed. It is the "noise floor" against which any operator effect must stand out.

### 3.2 Between‑operator variance (`σ²_between`)

Now look across operators (within a method): each operator has a mean SR. The
spread of those per‑operator means is the **between‑operator variance** — the
*operator effect*, i.e. how much the outcome moves when you change **who**
trained the policy. **This is the dependency we care about.**

### 3.3 Putting them together — ICC(1,1)

A one‑way random‑effects ANOVA (`icc_oneway` in
[`stats.py`](../serl_launcher/serl_launcher/aqi/stats.py)) splits the total SR
variance into these two components and forms the **intraclass correlation**:

$$\mathrm{ICC} = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{between}} + \sigma^2_{\text{within}}}$$

with `σ²_between = max((MS_b − MS_w)/k₀, 0)` and `σ²_within = MS_w` (`k₀` is the
mean group size, Searle's correction for unequal replicate counts).

Interpretation:

- **ICC → 1**: almost all the variation is *between operators* — the outcome is
  **highly operator‑dependent** (who trained it dominates).
- **ICC → 0**: the operator barely matters; runs differ only by seed noise —
  **operator‑independent**.

### 3.4 Reinforcing measures

Two complementary numbers make the story robust at small sample sizes:

- **Operator spread** (`operator_spread`): the standard deviation `σ_op`,
  **coefficient of variation** `CV = σ_op / mean`, the **range**, and the
  **worst‑case operator** — a direct, unit‑free feel for how far apart operators
  land within a method.
- **Permutation test** (`permutation_sigma2`): shuffle the operator labels many
  times and recompute the between‑operator variance; the **p‑value** is the
  chance of seeing this much operator spread if operators didn't actually matter.
  Low `p` = "operators matter"; high `p` = "the spread is just noise".

### 3.5 *Within‑method* vs *between‑method*

Everything in 3.1–3.4 is computed **within each method** — `hilserl`, `active`,
`quality`, `aqi` each get their own `σ²_between`, `σ²_within`, ICC, CV and
permutation `p`. That is the **within‑method** view: *for this method, how
operator‑dependent is the outcome?*

The study's claim is a **between‑method** comparison: AQI should have a **lower
ICC / lower CV / higher permutation p** than baseline HIL‑SERL — i.e. it makes
the outcome depend *less* on who trained it. The report prints, per method, the
change versus the baseline (`Δ ICC`, `Δ σ²_between`).

An illustrative shape (from a validation run on fabricated cells) makes the
target concrete:

| method | mean SR | ICC | σ²_between | σ²_within | CV | p(operator matters) |
|---|---|---|---|---|---|---|
| `hilserl` (baseline) | 55 % | **0.99** | 0.12 | 0.001 | **0.45** | **0.000** |
| `aqi` | 80 % | **0.16** | 0.000 | 0.001 | **0.02** | 0.035 |

Baseline: expert‑trained policies win, novice‑trained lose (strong dependency).
AQI: everyone lands near the same SR (dependency largely removed).

### 3.6 A second axis — steps‑to‑convergence

Dependency is not only about the *final* SR; it is also about the *cost* of
getting there. Each cell stops when it **converges** (a sustained streak of
autonomous successes) or hits a step cap, and the step at convergence is
recorded. The report therefore also decomposes **steps‑to‑convergence** the same
way (converged fraction, mean/median steps, and the **ICC of steps‑to‑
convergence across operators**). A method can reduce final‑SR dependency, effort
dependency, or both.

> **The convergence criterion in detail.** The check runs at the end of every
> **actor** episode and declares convergence only when **all three** hold
> ([`train_aqi_rlpd.py`](../examples/train_aqi_rlpd.py)):
>
> 1. **enough episodes** — at least `--converge_window` (default 20) have
>    finished;
> 2. **past warm-up** — `step ≥ --converge_min_steps`;
> 3. **rolling success high enough** — the success rate over the last
>    `--converge_window` episodes is `≥ --converge_success`.
>
> An episode counts as `1.0` in that window only if it **succeeded** *and* — with
> `--converge_require_autonomous` (default on, and the campaign passes it) — the
> operator **never took over** (`intervention_count == 0`). So the criterion is on
> the **autonomous** success rate — the policy solving the task unaided — which
> matches the ruler's definition of "the policy no longer needs the operator."
> The actor stops at whichever comes first: convergence, the `--max_steps` cap, or
> `--max_minutes`.

> **Convergence by the mini-eval (`--converge_mode eval`, off by default).**
> The rolling rule only counts training episodes the operator never touched, so
> an operator who keeps stepping in delays it even when the policy would already
> succeed alone. With `--converge_mode eval` (YAML `budget.converge_mode`) the
> autonomous fixed-pose mini-eval decides instead: once the policy has succeeded
> without help at least once and `step ≥ --converge_min_steps`, the actor runs
> the mini-eval (`--converge_eval_poses`, without real-time pacing) every
> `--converge_eval_every` env steps (default 2000) at an episode end and stops
> after `--converge_eval_consecutive` (default 2) passes of
> `--converge_eval_threshold` in a row (which must be > 0). Each evaluation is
> appended to `<checkpoint>/convergence_evals.jsonl`; `CONVERGED.json` records
> `mode: eval` and the evaluation history. The rolling rule is not used then.
>
> **Fixed update ratio (`--utd_ratio`, off by default).** The learner normally
> runs as fast as its GPU allows while the actor is paced at 10 Hz, so the
> learner steps per env step depend on the hardware (≈ 0.45 on the training
> machine, 1.3–2.7 on an H100, finding F3). With `--utd_ratio U > 0` (YAML
> `budget.utd_ratio`) the two processes of a cell hold it at `U`
> ([`update_ratio.py`](../serl_launcher/serl_launcher/utils/update_ratio.py)):
> the learner writes its update count to `<checkpoint>/learner_progress.json`
> and waits while it already has `U` updates per received env step; the actor
> waits before each env step until the learner has done `U` updates per step of
> every episode but the one it is collecting, and the env runs without
> real-time pacing. Nothing waits while the learner fills its buffer, and a
> learner silent for `--utd_timeout_s` (600 s) releases the actor with a
> warning. A cell then takes `env steps × U / learner steps per second`:
> about 100 min for 30 000 steps at `U = 1` on the training machine
> (≈ 5 learner steps/s) unless it converges earlier.

### 3.7 Reading the report

`report/dependency.md` (and `.json`) contains four blocks:

1. **Operator dependency per method** — mean SR, ICC, `σ²_between`, `σ²_within`,
   `Δ ICC` vs baseline.
2. **Operator spread + "operator matters" test** — `σ_op`, CV, range,
   worst‑operator, permutation `p`.
3. **Skill gradient** — mean SR per tier (does expert ≫ novice?).
4. **Training effort** — steps‑to‑convergence and its ICC.

> The lean in‑runner analysis covers SR‑ICC, spread, permutation and
> convergence. The **full** pre‑registered suite (robustness gain, interaction
> decomposition, decision rules — protocol B.4) lives in
> [`experiment_pipeline.py`](../examples/experiment_pipeline.py) `evaluate`,
> which can ingest the same per‑cell `cell.json` files.
>
> For the **full estimator mathematics** (the ANOVA sums of squares, Searle's
> `k₀`, the permutation and bootstrap procedures, and a worked decode of a real
> report) see [dependency_analysis.md](dependency_analysis.md).

---

## 4. The campaign — putting it on rails

[`run_campaign.py`](../examples/run_campaign.py) automates the whole study.
See [campaign_runner.md](campaign_runner.md) for the option‑by‑option reference;
the essentials:

### 4.1 Five idempotent stages

| stage | what it does |
|---|---|
| `sample` | draw the jittered operator roster (§1.5) and freeze the ruler grid (§2) |
| `demos` | once per `(operator, seed)`: record that operator's demos (reused across all methods) |
| `train` | for every cell `(method × operator × seed)`: run the distributed learner+actor, stopping on convergence |
| `eval` | score each trained policy on the fixed ruler |
| `analyze` | compute the dependency assessment (§3) and write the report |

Each stage **skips work whose output already exists**, so an interrupted
campaign resumes cleanly.

### 4.2 One cell, start to finish

For each cell the runner launches the **learner** (loads the operator's demos,
trains from scratch, waits in the background) and, once it prints `LEARNER
READY`, the **actor** (the synthetic operator standing in for the SpaceMouse,
same `θ` as the demos). The learner and actor talk over the ordinary,
unchanged, synchronous loop. The actor stops as soon as the policy converges
(or hits the step cap), the runner tells the learner to save via a `STOP`
sentinel, and the cell is done. Seeds are wired so replicates are genuinely
independent (`--seed = --synthetic_seed = --env_seed = r`).

### 4.3 Why demos are collected once per (operator, seed)

All methods for a given operator+seed **share the same demonstrations**. That is
deliberate: it holds the demonstrations fixed so the **method** is the only thing
that changes between cells — otherwise a method difference could be a demo
difference in disguise.

### 4.4 The demo‑strength regime (important)

Operator skill and AQI act mostly during the **online interventions**. If you
give too many or too strong demonstrations, every method solves the task from the
demos alone, the methods converge, and there is *no dependency left to reduce* —
the study shows nothing. So keep `--demos` modest (default 6, filtered
successes) and give a meaningful online budget, then **check that the baseline
still shows a real skill gradient** (expert ≫ novice) before trusting the AQI
comparison.

### 4.5 Self‑describing storage

Every artifact the campaign writes is a folder carrying a `manifest.json` and a
`README.md`: a top‑level campaign README, one per `demos/<op>_s<seed>/`, one per
`checkpoints/<cell>/`, and one per `eval_runs/<cell>/`. Each explains the task,
the conditions, and — derived from the live `AQIConfig` — **which choices were
inert** for that cell (e.g. a `hilserl` cell shows its trigger/quality/ensemble
switches as *inert, with the reason*, because the baseline never consults them).
See [campaign_runner.md](campaign_runner.md#self-describing-artifacts-readme--manifest).

---

## 5. Where each piece lives

| concern | file |
|---|---|
| operator model (`θ`, tiers, jitter, perception→decision→motor) | [`serl_launcher/serl_launcher/aqi/operator_model.py`](../serl_launcher/serl_launcher/aqi/operator_model.py) |
| monitoring / intervention gate (SDT, Role 2) | [`serl_launcher/serl_launcher/aqi/operator_gate.py`](../serl_launcher/serl_launcher/aqi/operator_gate.py) |
| sim geometric oracle | [`examples/experiments/sim_peg_insertion/wrapper.py`](../examples/experiments/sim_peg_insertion/wrapper.py) |
| demo collection (Role 1) | [`examples/collect_synthetic_demos.py`](../examples/collect_synthetic_demos.py) |
| training (learner + actor) | [`examples/train_aqi_rlpd.py`](../examples/train_aqi_rlpd.py) |
| fixed ruler evaluation | [`examples/eval_ruler.py`](../examples/eval_ruler.py) |
| variance stats (ICC, spread, permutation) | [`serl_launcher/serl_launcher/aqi/stats.py`](../serl_launcher/serl_launcher/aqi/stats.py) |
| campaign runner | [`examples/run_campaign.py`](../examples/run_campaign.py) |
| provenance / READMEs | [`examples/provenance.py`](../examples/provenance.py) |

### Further reading

- [getting_started.md](getting_started.md) — a hands-on, no-background walkthrough.
- [oracle_and_synthetic_operator.md](oracle_and_synthetic_operator.md) — the
  oracle → synthetic-operator design in depth (§1 companion).
- [operator_skill_assessment.md](operator_skill_assessment.md) — automatic
  operator **skill typing** from interaction logs (methodology Part B).
- [dependency_analysis.md](dependency_analysis.md) — the variance-decomposition
  mathematics in full (§3 companion).
- [critic_ensemble.md](critic_ensemble.md) — how the K critics are aggregated for
  the backup, the actor, and the AQI trigger.
- [architecture.md](architecture.md) — the `examples/` scripts, what each writes
  and reads, and how they call each other.
