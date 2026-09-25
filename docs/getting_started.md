# Getting started — no background needed

This guide assumes **no prior knowledge** of robotics, reinforcement learning,
or this codebase. By the end you'll understand what the project does and be able
to run it yourself. For the science and the maths, read
[methods.md](methods.md) afterwards.

---

## 1. What is this, in plain words?

Imagine teaching a robot to **push a peg into a hole**. It's fiddly: the peg has
to line up and go straight in. To teach it, a **helper** (a person with a little
joystick) watches the robot try, and **takes over** whenever the robot is about
to mess up. Over many tries, the robot learns to do it alone.

Here's the catch: **the robot's final skill depends on who helped it.** A careful
expert helper and a clumsy, panicky beginner will teach it differently. That's a
problem if you want reliable robots — you don't want the result to hinge on which
person happened to be on shift.

This project studies exactly that, and tests a smarter helping method called
**AQI** that is supposed to make the result depend **less** on who helped.

To study it carefully we do three things:

1. **Build fake "helpers" we can control.** Instead of real people, we use
   *synthetic operators* — little programs that behave like an expert, an
   intermediate, or a novice, in a realistic (not random) way. We can make as
   many as we like and repeat them exactly.
2. **Grade every robot the same way.** After training, we test each robot on an
   identical, fixed "exam" (we call it the **ruler**) — same starting positions
   every time, no helper allowed.
3. **Measure the dependence.** We check how much the exam scores move when we
   change *who* trained the robot. Less movement = less dependence = better.

Everything runs in a **simulator** (a physics sandbox), so no real robot or
joystick is required.

---

## 2. Words you'll see

| word | plain meaning |
|---|---|
| **operator** | the helper (here, a synthetic one) |
| **tier** | the helper's skill level: `expert`, `intermediate`, or `novice` |
| **demonstration (demo)** | a recording of the helper doing the task, used to bootstrap the robot |
| **method** | the *style* of helping: `hilserl` (the plain baseline) up to `aqi` (the smart one) |
| **the ruler** | the fixed exam every trained robot takes (same poses, no helper) |
| **success rate (SR)** | the exam score: fraction of attempts that succeeded |
| **cell** | one training run = one (method, operator, seed) combination |
| **seed** | a random‑number setting; changing it gives an independent repeat |
| **dependency (ICC)** | a 0–1 number: how much the score depends on *who* trained it (1 = a lot, 0 = not at all) |
| **campaign** | the whole automated study: make helpers → collect demos → train → exam → report |

---

## 3. One‑time setup

The project ships a ready Python environment in `.venv/`, so you don't install
anything. You run everything with that environment's Python:

```bash
cd /home/user/Downloads/hil-serl_aqi     # the project folder
./.venv/bin/python --version              # should print Python 3.12.x
```

Two things worth knowing:

- **The simulator draws graphics headlessly.** A helper script,
  `examples/experiments/sim_peg_insertion/sim_env.sh`, sets that up
  automatically; the campaign runner uses it for you.
- **Training uses the GPU, which may be shared.** Before starting a *real*
  training run, check nobody else is using the graphics card:

  ```bash
  nvidia-smi        # look at "Memory-Usage"; if it's already full, wait
  ```

  Planning and analysis (below) do **not** touch the GPU, so they're always safe
  to run.

---

## 4. Your first command — look before you leap (`--plan`)

The safest way to start is to ask the campaign to **describe what it would do**
without doing anything heavy:

```bash
./.venv/bin/python examples/run_campaign.py \
    --campaign runs/first_try \
    --ops_per_tier 2 --seeds 2 --methods hilserl,aqi \
    --plan
```

This prints a plan: how many helpers it will create, how many demo recordings,
how many training runs, and it lists each synthetic operator with its skill
numbers. It also writes a folder `runs/first_try/` with:

- `README.md` — a plain description of this campaign;
- `operators.csv` — the roster of synthetic helpers;
- `eval_grid.json` — the fixed exam.

Nothing was trained yet. Open `runs/first_try/README.md` and `operators.csv` to
see what you're about to run.

> Tip: add `--dry_run` (instead of `--plan`) to print the **exact** commands the
> campaign would run for demos, training and exams — a good way to learn what's
> happening under the hood.

---

## 5. Run a tiny real study (a "smoke test")

This actually trains robots, so it uses the GPU and takes a while. Keep it small
the first time:

```bash
./.venv/bin/python examples/run_campaign.py \
    --campaign runs/first_try \
    --ops_per_tier 2 --seeds 2 --methods hilserl,aqi \
    --demos 4 --steps 1500
```

What happens, in order:

1. **make helpers** — 2 experts, 2 intermediates, 2 novices (jittered so they
   genuinely differ);
2. **collect demos** — each helper records a few successful attempts (this part
   runs on the CPU, off the GPU);
3. **train** — for every (method, helper, seed) it trains a robot, with the
   helper taking over when needed, until the robot reliably succeeds on its own;
4. **exam** — every trained robot takes the identical ruler exam;
5. **report** — it computes how much the scores depend on the helper and writes
   the result.

If it stops halfway (you close the laptop, the GPU was needed), just **run the
same command again** — it skips everything already finished and continues.

---

## 6. Where are the results?

Everything lands under your `--campaign` folder. The two things to open first:

- **`runs/first_try/report/dependency.md`** — the headline. It has a small table
  per method with the **dependency number (ICC)** and the average score. The
  story you're looking for: the baseline `hilserl` has a **high** ICC (score
  depends a lot on the helper) and `aqi` has a **lower** ICC (it depends less).
- **`runs/first_try/README.md`** — a plain description of the whole run.

Every sub‑folder is **self‑describing** — it contains its own `README.md`
explaining what it is and under what settings it was made. For example, open a
trained robot's folder:

```
runs/first_try/checkpoints/hilserl_exp00_s0/README.md
```

It lists the task, the helper's skill numbers, the demos used, when training
stopped, and a **conditions table**. That table is the clever bit: some settings
only matter for the smart methods, so for a baseline run they're marked
**inert** with the reason — e.g. *"the active gate is off in `hilserl`, so this
setting is never consulted."* You never have to guess whether a knob mattered.

---

## 7. Doing one piece at a time

You don't have to run the whole campaign. Each stage can run alone with
`--stages`:

```bash
# just (re)build the report from already-trained cells
./.venv/bin/python examples/run_campaign.py --campaign runs/first_try --stages analyze
```

Or use the individual scripts directly:

- **Record demos from one helper** (CPU only, quick):

  ```bash
  ./.venv/bin/python examples/collect_synthetic_demos.py \
      --env experiments.sim_peg_insertion.wrapper:make_synthetic_sim_env \
      --profile novice --episodes 4 --seed 0 --out /tmp/novice_demos/demos.pkl
  ```

  It writes the recording **and** a `README.md` describing the helper.

- **Grade a trained robot on the ruler**:

  ```bash
  ./.venv/bin/python examples/eval_ruler.py \
      --exp_name sim_peg_insertion \
      --checkpoint_path runs/first_try/checkpoints/aqi_exp00_s0 \
      --grid_file runs/first_try/eval_grid.json \
      --out /tmp/one_eval --argmax
  ```

- **Add READMEs to older results** made before this documentation existed:

  ```bash
  ./.venv/bin/python examples/backfill_readmes.py --dry_run   # preview
  ./.venv/bin/python examples/backfill_readmes.py             # do it
  ```

There are also ready‑made VS Code tasks (Terminal → Run Task → `sim: …`) that
prompt you for the choices if you prefer buttons over commands.

---

## 8. Turning the knobs

The options you'll most likely change (see `--help` for the full list):

| option | plain meaning | default |
|---|---|---|
| `--ops_per_tier` | how many distinct helpers per skill level | 5 |
| `--seeds` | how many repeats per helper (more = steadier numbers) | 3 |
| `--methods` | which helping styles to compare | `hilserl,active,quality,aqi` |
| `--demos` | how many demo recordings per helper | 6 |
| `--steps` | a safety cap on how long each robot may train | 20000 |
| `--converge_success` | stop a robot once it succeeds this often on its own | 0.9 |
| `--plan` / `--dry_run` | preview only / print the commands | off |

**One important caution.** The helper's skill mainly shows up *while the robot is
training* (when they take over). If you hand the robot **too many** demo
recordings, it learns everything from the demos and every method looks the same —
so there's no dependence left to measure. Keep `--demos` small, and check that
the plain baseline still shows a clear expert‑beats‑novice gap before trusting
the comparison. (More on this in [methods.md §4.4](methods.md).)

---

## 9. When something goes wrong

| symptom | what it means / what to do |
|---|---|
| `nvidia-smi` shows the card is full | someone else is using the GPU; wait, or the runner will refuse to start (pass `--force` only if you're sure) |
| "learner never reported READY" | the training helper process didn't start in time; check the log it names under `runs/…/logs/` |
| a cell failed but others ran | fine — re‑run the same campaign command; it retries only the unfinished parts |
| the report says a method "did not converge" for some helpers | that's a **real result**, not a bug: a poor helper may never get the robot reliable within the cap |
| you want to start clean | delete the `runs/<name>/` folder and run again |

Nothing is destructive by default, and re‑running is always safe — finished work
is skipped, not repeated.

---

## 10. The one‑paragraph recap

You create graded fake helpers (expert → novice), let each teach a robot to
insert a peg, give every trained robot the **same fixed exam**, and measure how
much the exam scores depend on **which** helper taught it. A high dependency
number (ICC) for the plain method and a **lower** one for AQI is the result the
project is looking for. Start with `--plan`, then a small `--demos 4 --steps
1500` run, open `report/dependency.md`, and read [methods.md](methods.md) when
you want the details.

---

## 11. Where to go next

| you want… | read |
|---|---|
| the science and the maths of the study | [methods.md](methods.md) |
| how the fake helpers (oracles + operators) are built | [oracle_and_synthetic_operator.md](oracle_and_synthetic_operator.md) |
| how an operator is graded by skill from its logs | [operator_skill_assessment.md](operator_skill_assessment.md) |
| how the K critics become the uncertainty signal | [critic_ensemble.md](critic_ensemble.md) |
| the variance-decomposition (ICC) mathematics in full | [dependency_analysis.md](dependency_analysis.md) |
| which script writes/reads what (the data flow) | [architecture.md](architecture.md) |
| the AQI algorithm (trigger + quality) | [aqi_walkthrough.md](aqi_walkthrough.md) |
| the automated campaign runner | [campaign_runner.md](campaign_runner.md) |
| the fixed evaluation ruler | [eval_ruler.md](eval_ruler.md) |
