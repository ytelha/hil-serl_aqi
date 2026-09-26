# Automated campaign runner — `examples/run_campaign.py`

End-to-end orchestrator for the thesis' core question: **how much does a
HIL-SERL training outcome depend on *which* operator trained it, and does AQI
reduce that dependency versus the baseline?**

It runs, unattended, five idempotent stages:

| stage | what it does |
|---|---|
| `sample` | draws 5–10 *distinct* synthetic operators per skill tier (jittered θ) and freezes the fixed evaluation ruler (a hash‑locked pose grid) |
| `demos` | once per `(operator, seed)`: the operator teleoperates the sim via its geometric oracle and records training‑format demos (reused across every method) |
| `train` | for every cell `(method × operator × seed)`: launches the distributed learner+actor exactly as a human run would, with the synthetic operator standing in for the SpaceMouse; **each cell trains until the policy converges** (a sustained autonomous‑success streak), capped by `--steps` |
| `eval` | scores each trained policy on the identical fixed grid with `eval_ruler.py` (policy frozen, no operator, argmax) → `cell.json` |
| `analyze` | per method computes **within‑operator** variance (across seeds), **between‑operator** variance (across operators) and the one‑way **ICC**, then compares methods → `report/dependency.{json,md}` |

A *cell* is one `(method, operator, seed)`; its id is `<method>_<operator>_s<seed>`.

## Design guarantees

- **Learner ↔ actor stay distributed and synchronous.** The runner only
  *starts* and *stops* them (learner via the `<ckpt>/STOP` sentinel after the
  actor stops); it never changes how they train.
- **Each cell ends on convergence, not a fixed budget.** The actor watches a
  rolling window of episode outcomes and stops as soon as the policy solves the
  task reliably *without* operator takeover (`--converge_success` over
  `--converge_window` episodes). `--steps` is only the hard cap for cells that
  never converge. The stopping step is written to
  `checkpoints/<cell>/CONVERGED.json`, so **steps‑to‑convergence** becomes an
  operator‑dependency metric in its own right.
- **The same jittered operator drives demos *and* interventions.** Each operator
  is sampled once (`operators/<id>.json`) and passed by `--theta_json` /
  `--synthetic_theta` to both the demo collector and the actor.
- **The ruler is truly fixed on sim.** `set_eval_reset_pose(dx_m, dyaw_rad)` on
  the sim env pins each eval pose deterministically (randomization off), so
  every cell is measured on the byte‑identical, SHA‑256‑locked grid.
- **Idempotent / resumable.** Every unit skips if its output exists
  (`operators.csv`, a demo `.pkl`, a checkpoint `.done` marker, a `cell.json`),
  so an interrupted campaign resumes where it stopped.
- **Shared‑GPU safe.** A pre‑flight aborts if other processes already occupy the
  GPU (unless `--force`); demo collection runs on CPU (`JAX_PLATFORMS=cpu`).

## Layout produced under `--campaign`

```
README.md                    top-level campaign README (what/how)
campaign.json                frozen manifest (params + grid sha256)
eval_grid.json               the fixed ruler grid (dx × dyaw poses)
operators.csv                the sampled operator roster
operators/<op>.json          per‑operator ground‑truth θ
demos/<op>_s<seed>/          per‑operator demos folder: demos.pkl +
                             demos_groundtruth.json + README.md + manifest.json
checkpoints/<cell>/          trained policy per cell (+ .done, README.md, manifest.json)
eval_runs/<cell>/            ruler outcomes per cell: cell.json + README.md + manifest.json
logs/<...>.log               every subprocess' stdout+stderr
report/dependency.{json,md}  the operator‑dependency assessment
```

Every artifact folder is **self-describing**: alongside its payload it carries a
`manifest.json` (machine) and a `README.md` (human) explaining the **task**, the
**conditions**, and — importantly — **which choices were inert** for that cell.
The conditions table is derived from the live `AQIConfig` (`active_on` /
`quality_on` / `logging_on`), so a `hilserl` cell automatically shows its
trigger, quality and ensemble‑epistemic switches as *inert (with the reason)*
rather than pretending they mattered. The standalone scripts
(`train_aqi_rlpd.py`, `collect_synthetic_demos.py`, `eval_ruler.py`) write the
same docs for ad‑hoc runs, and `examples/backfill_readmes.py` generates them for
pre‑existing `run_*/`, `runs/*/`, `aqi_logs/*/` and `demo_data/` artifacts.

## Typical use

Always dry‑preview the plan first:

```bash
./.venv/bin/python examples/run_campaign.py \
    --campaign runs/study1 --ops_per_tier 8 --seeds 3 --plan
```

Print the exact subprocess commands without running anything heavy:

```bash
./.venv/bin/python examples/run_campaign.py --campaign runs/study1 --dry_run
```

Run the whole thing (uses the shared GPU — check `nvidia-smi` first):

```bash
./.venv/bin/python examples/run_campaign.py \
    --campaign runs/study1 \
    --ops_per_tier 8 --seeds 3 \
    --methods hilserl,active,quality,aqi \
    --demos 6 --steps 4000
```

Re‑run only a later stage (e.g. after all cells trained):

```bash
./.venv/bin/python examples/run_campaign.py --campaign runs/study1 --stages eval,analyze
```

## Key options (see `--help` for all)

| option | default | meaning |
|---|---|---|
| `--ops_per_tier` | 5 | distinct (jittered) operators per tier (thesis: 5–10) |
| `--tiers` | `expert,intermediate,novice` | skill tiers to sample |
| `--seeds` | 3 | replicate seeds per operator (≥2 for a within‑operator variance estimate) |
| `--methods` | `hilserl,active,quality,aqi` | ARMS to compare: plain AQI modes and/or names defined under YAML `arms:` (see below) |
| `--baseline` | `hilserl` | arm the others are Δ‑compared against |
| `--demos` | 6 | demos per `(operator, seed)` — keep **modest** (see below) |
| `--steps` | 20000 | per‑cell **actor** env‑step **hard cap** (bounds only non‑converging cells) |
| `--converge_success` | 0.9 | stop a cell at this rolling success rate (0 disables → `--steps` is the only stop) |
| `--converge_window` | 20 | episodes in the rolling success window |
| `--converge_min_steps` | 500 | do not converge before this many env steps |
| `--converge_allow_assisted` | off | count operator‑assisted successes too (default: autonomous only) |
| `--eval_dx_mm` | `-16…16` (9) | lateral offsets swept by the ruler |
| `--jax_platform` | `cuda` | `cuda` or `cpu` |
| `--gpu_mem_learner` / `--gpu_mem_actor` | 0.6 / 0.2 | VRAM fraction ceilings |
| `--plan` / `--dry_run` / `--stages` / `--retrain` / `--force` | — | flow control |

## Arms: several variants of one mode in the same campaign

Cells are named `<arm>_<operator>_s<seed>`. A plain mode name (`hilserl`,
`logged`, `quality`, …) is an arm with no overrides. To compare **variants of
the same mode**, for example two quality scores or several ensemble settings,
declare named arms. Each arm is a base `mode` plus overrides of any
learning-method knob:

```yaml
methods: [logged, quality-cf, quality-beh]     # arm names, in report order
arms:
  quality-cf:  {mode: quality, quality_score_mode: counterfactual}
  quality-beh: {mode: quality, quality_score_mode: behavioral}
analysis:
  baseline: logged
```

- **Resolution** for each knob of an arm:
  `arms[arm][key]` > `method_overrides[arm][key]` > campaign-wide value
  (CLI > YAML > default). `method_overrides` is the older per-mode block and
  keeps working unchanged. An arm entry named after a mode (e.g. `hilserl:`)
  must keep that mode.
- **Per-arm knobs** (`ARM_KEYS` in `run_campaign.py`):
  - learner: `demo_pretrain`, `demo_pretrain_steps`,
    `demo_pretrain_batch_size`, `learning_rate`;
  - ensemble: `critic_ensemble_size`, `critic_subsample_size`,
    `per_member_targets`;
  - active gate: `trigger_terms`, `trigger_combiner`;
  - quality: `quality_filters`, `quality_disagreement_gate`,
    `bc_loss_weight`, `quality_score_mode`, `quality_behavioral_terms`,
    `quality_post_release_steps`, `quality_hindsight_discount`,
    `quality_consistency_*` (see `docs/quality_scores_v2.md`), `per_*`,
    `quality_warmup_steps`;
  - E2HiL: `e2hil_*`.

  Operators, demos, budget, the convergence rule and the ruler are
  deliberately **not** per-arm: the arm must be the only difference between
  paired cells.
- **Names:** lowercase letters, digits, `-` and `.` only, **no `_`**, because
  the analysis scripts parse cell ids back into arm / operator / seed.
- **Checks:** everything is validated before any job starts: unknown mode,
  unknown knob, bad value, bad name.
- **Where the arm shows up:**
  - `--plan` prints each arm's mode and overrides;
  - `campaign.json` and every cell's `manifest.json`/`README.md` record the
    arm definition, and the cell README's reproduce block is the exact
    learner/actor command that ran;
  - the actor gets `--arm <name>` (recorded in `aqi_logs/<run>/meta.json` and
    the wandb run name), so `aqi_analysis.py`, `q_summary.py` and
    `intervention_curves.py` group by arm, not by base mode.
- **Resume safety:** changing an arm's overrides after cells finished does not
  retrain them (`.done` skips). Give the new variant a new arm name, or use
  `--retrain`.

Worked example: [campaigns/rq4_attribution_sweep.yaml](../examples/campaigns/rq4_attribution_sweep.yaml)
(the one-factor attribution of RQ4's four ensemble fixes).

## Choosing the regime (important)

The operator‑dependency signal lives in the **online interventions**, where
operator skill (monitoring timing, execution quality) and AQI's gating actually
act. If you give **too many / too strong demos**, every method solves the task
from the demos alone and the methods converge — there is no dependency left to
reduce, and the study shows nothing. So:

- keep `--demos` **small** (default 6, filtered successes) and
- give a **meaningful** `--steps` online budget (default 4000),

then confirm from `report/dependency.md` that the **baseline** `hilserl` shows a
real skill gradient (expert ≫ novice) and non‑trivial ICC before reading the AQI
comparison. Tune `--demos` / `--steps` if the baseline is already saturated
(no gradient) or hopeless (all zero).

## What the report tells you

`report/dependency.md` gives two complementary views of operator dependency:

1. **Final autonomous SR** — per method: mean success rate, the one‑way **ICC**,
   **σ²_between** (operator effect) and **σ²_within** (seed/replicate effect),
   the Δ‑ICC versus baseline, and the per‑tier skill gradient.
2. **Training effort: steps‑to‑convergence** — per method: the converged
   fraction, mean/median steps to converge, and the **ICC of
   steps‑to‑convergence** across operators (does *how long* training takes
   depend on who ran it?), plus the per‑tier step gradient.

Lower ICC / lower σ²_between on **either** axis = the outcome (or the cost of
reaching it) depends **less** on who trained it — the thesis claim is that AQI
shrinks these relative to baseline HIL‑SERL.

> **The maths behind the report.** For every estimator in `dependency.md` — the
> ANOVA sums of squares, Searle's `k₀`, ICC(1,1), the operator-spread metrics, the
> permutation test and the hierarchical bootstrap CI — with a worked decode of a
> real report, see [dependency_analysis.md](dependency_analysis.md). The exact
> three-part convergence criterion is in [methods.md §3.6](methods.md).

## Cost / scale

`#cells = methods × (tiers × ops_per_tier) × seeds`. The default
`4 × 15 × 3 = 180` cells each run one learner+actor training of `--steps` env
steps plus a ruler eval — this is a **large GPU job**. Start small
(`--methods hilserl,aqi --ops_per_tier 2 --seeds 2`) to validate the loop, then
scale up. Cells run sequentially (no port clash) and resume on restart.
