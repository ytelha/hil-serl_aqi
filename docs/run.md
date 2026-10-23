# Run the campaign — quick reference

This project trains a peg‑insertion policy with a **synthetic operator** standing
in for a human, then measures **how much the result depends on which operator
trained it** (and whether AQI reduces that dependence). No robot, no human, no
ROS — everything runs in the MuJoCo simulator on this machine, unattended.

*New to the project?* Read [getting_started.md](getting_started.md) (no
background) or [methods.md](methods.md) (the science). This page is the commands.

---

## 1. Setup

The environment ships in `.venv/`, so there's nothing to install. Run from the
project root, and check the shared GPU is free before a real run:

```bash
cd /home/user/Downloads/hil-serl_aqi
nvidia-smi        # make sure the card isn't already full
```

---

## 2. Define the study in one file

Copy [examples/campaigns/study.yaml](../examples/campaigns/study.yaml) and edit
it. It captures the whole study:

```yaml
task:
  exp_name: sim_peg_insertion
  peg_shape: cylinder

operators:              # operator population PER TIER (counts may differ)
  expert: 8
  intermediate: 8
  novice: 8
  jitter: 0.15
  sample_seed: 0

demos:
  per_operator: 6       # keep modest (see the note below)
  filtered: true

methods:                # the ablations to compare
  - hilserl             # baseline
  - active
  - quality
  - aqi                 # full system

replicates:
  seeds: 5              # repeats per operator

budget:
  converge_success: 0.9 # stop each cell once it reliably succeeds on its own
  steps_cap: 20000      # hard cap for cells that never converge
  minutes_per_cell: 60  # fixed wall-clock budget per cell (0 = no time limit)
  total_hours: 0        # fixed wall-clock budget for the WHOLE campaign
                        # (0 = unlimited; e.g. 10 = stop after 10 h, resumable)

analysis:
  baseline: hilserl
  plots: true
```

Any command‑line flag overrides the matching field, e.g. add
`--methods hilserl,aqi` to run just those two.

---

## 3. Preview (no GPU, trains nothing)

```bash
./.venv/bin/python examples/run_campaign.py \
    --config examples/campaigns/study.yaml \
    --campaign runs/study1 --plan
```

Writes `runs/study1/` with `README.md`, `operators.csv` and `eval_grid.json`, and
prints the plan (24 operators, 480 cells, …). Use `--dry_run` instead to print
the exact per‑cell commands.

---

## 4. Run it (unattended, resumable)

Drop `--plan` to execute; launch detached so it survives logout:

```bash
nohup ./.venv/bin/python examples/run_campaign.py \
    --config examples/campaigns/study.yaml \
    --campaign runs/study1 > runs/study1.out 2>&1 &

tail -f runs/study1.out          # watch progress
```

Sample operators → collect demos (CPU) → train every `(method × operator ×
seed)` cell until it converges → score each on the fixed ruler → analyze and
**plot**. **If it stops** (GPU needed, reboot, Ctrl‑C), run the **same command
again** — finished work is skipped.

**Fixed training budget.** Each cell stops at whichever comes first: convergence,
the `steps_cap`, or `minutes_per_cell` (the wall‑clock budget). With
`minutes_per_cell: 60` and 480 cells the whole study is bounded at ~480 GPU‑hours
even if nothing converges early, so the run has a predictable duration.
Set `total_hours` to also cap the **whole campaign** — once that wall‑clock
budget elapses the runner stops launching new cells and evaluates/analyzes what
finished (re‑run later to continue the rest).

**Fixed ruler.** The evaluation grid is frozen on first write and hash‑locked;
**every** cell — across the whole campaign and any resumed runs — is scored on the
byte‑identical exam. A later run that asks for a different grid keeps the frozen
one (and warns); use a fresh `--campaign` dir to change the ruler.

> Keep `demos.per_operator` small: operator skill and AQI act mainly during the
> online interventions; too many demos let every method solve the task from the
> demos alone and hide the dependency the study measures.

---

## 5. Results

Under `runs/study1/report/`:

- **`dependency.md`** — tables: success rate, the dependency number **ICC**,
  σ²_between / σ²_within, operator‑spread CV and the "operator matters" p‑value,
  and steps‑to‑convergence — per method, with the change vs the baseline.
- **`dependency.json`** — the same, machine‑readable.
- **`plots/`** — seven figures:
  `icc_by_method.png` (the headline), `sr_by_operator.png` (per‑operator spread
  = the dependency; error bars = within‑operator noise), `sr_by_tier.png` (skill
  gradient), `variance_decomposition.png`, `operator_spread.png`,
  `steps_to_convergence.png`, `frontier.png` (success against human effort).

Every artifact folder (`checkpoints/<cell>/`, `eval_runs/<cell>/`,
`demos/<op>_s<seed>/`) also carries its own `README.md` explaining its settings,
including which options were *inert* for that method.

Re‑build just the report + plots from already‑trained cells at any time:

```bash
./.venv/bin/python examples/run_campaign.py --config examples/campaigns/study.yaml \
    --campaign runs/study1 --stages analyze
# or plots only, from an existing report:
./.venv/bin/python examples/campaign_plots.py --campaign runs/study1
```

---

## 6. Smoke test first

Validate the whole loop with a tiny, fast study before committing to a big one:

```bash
./.venv/bin/python examples/run_campaign.py \
    --campaign runs/smoke --methods hilserl,aqi \
    --ops_per_tier 2 --seeds 2 --demos 4 --steps 1500
```

---

## 7. Command‑line only (no config file)

Every field has an equivalent flag if you prefer not to use a config:

```bash
./.venv/bin/python examples/run_campaign.py \
    --campaign runs/study1 \
    --tiers expert,intermediate,novice --ops_per_tier 8 --seeds 5 \
    --methods hilserl,active,quality,aqi --demos 6 --steps 20000
```

Run `./.venv/bin/python examples/run_campaign.py --help` for the full list.
