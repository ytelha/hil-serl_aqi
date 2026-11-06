# Which part of the ensemble fix mattered? — the ablation on the first training machine

The first ensemble extension lost to plain HIL-SERL; the fixed one matched it. The fix changed four things at once, so
nobody knows which one did the work. This campaign tests each change on its own, on the **first training machine with
the new stack** (the shortened operator, as in the comprehensive study; the standard ruler) and with **experts and intermediates
only** (the synthetic novice is too weak).

- Config: `examples/campaigns/ensemble_fix_ablation.yaml` (new, nothing else changes).
- Campaign folder: `runs/ensemble_fix_ablation`; W&B group `ensemble_fix_ablation`.
- Background: `docs/thesis/rq4_ensemble_convergence_and_qa_log.md` (sections 2–3: the four changes and why).

## 1. The four changes, and what stays the same

| | first version | fixed version |
|---|---|---|
| critics K | 10 | 8 |
| target of a critic (M) | minimum of 2 random critics | 1 random critic (no minimum) — "de-pessimisation" |
| warm start | none | 5 000 learner steps on the demonstrations first |
| learning rate | 3e-4 | 1.5e-4 |
| per-member targets | on | on (unchanged) |
| bootstrap masks | p = 0.8 | p = 0.8 (unchanged) |

Per-member targets and the masks were already on in the first version, so they are not part of the fix.

## 2. The arms (12 runs each: 3 experts + 3 intermediates × 2 seeds)

| arm | mode | K | M | warm start | LR | question |
|---|---|---|---|---|---|---|
| `ens` | ensemble | 10 | 2 | – | 3e-4 | the first version: the base of every comparison |
| `ens-m1` | ensemble | 10 | 1 | – | 3e-4 | de-pessimisation alone |
| `ens-fix` | ensemble | 8 | 1 | 5 000 | 1.5e-4 | all four = the fixed version |
| `hilserl` | HIL-SERL | 2 | – | – | 3e-4 | plain HIL-SERL |
| `ens-wu` | ensemble | 10 | 2 | 5 000 | 3e-4 | warm start alone |
| `ens-wu-lr` | ensemble | 10 | 2 | 5 000 | 1.5e-4 | warm start + halved learning rate |
| `ens-k8` | ensemble | 8 | 2 | – | 3e-4 | fewer critics alone |
| `hilserl-wu` | HIL-SERL | 2 | – | 5 000 | 3e-4 | warm start for plain HIL-SERL |

**The new stack** (patches 0014–0016 of the first-machine bundle): rotation control; orientation box and twist channel
(cuboid limits tilt 0.384, yaw 0.1396 rad from stage 0, no learning involved); the mini-eval stop rule (every 2 000
steps, 2 passes in a row); one learner step per environment step. The ruler is the standard one (same 9 poses) on the
12 kept checkpoints; the shorter ruler (patch 0037) is not needed and not used. 30 000-step cap, 6 demonstrations,
roster seed 0.

Runs are trained arm by arm in this order, so the first four arms (48 runs) already answer the main question.

**Cost:** 96 runs. On this machine with the new stack: about 30 min online training (about 9 000 steps at about 5 per
second, the learner sets the pace), plus about 20 min warm start in the four warm-start arms, plus about 30 min for the
ruler (12 checkpoints): **60–80 min per run, about 5 days in total**, the first four arms about 2.5 days. A run that never
converges trains to the 30 000-step cap (about 100 min plus the ruler), so a failing arm takes longer.

These numbers are not comparable with the old RQ3/RQ4 campaigns (old stack, novices, 0.45 updates per step); only the
arms of this campaign are compared with each other. Whether the first version still loses on the new stack is itself
the first read-out.

## 3. How the result is read (fixed before any run)

Outcomes per arm, from `report/dependency.md`: converged runs of 12 and steps to converge (the main measures: the first
version failed by hovering and not converging), ruler success (mean, expert, intermediate), ICC. Paired comparisons use
the same operator and seed (sign test over the non-tied pairs; with 12 pairs, p < 0.05 needs at least 10 wins).

- **First read-out:** does `ens` still lose to `ens-fix` and `hilserl` on the new stack (fewer converged runs, more
  steps, lower success)? If not, the old failure was tied to the old stack, and the factor attribution below is
  descriptive only.
- **A change explains the fix** if its arm beats `ens` (sign test p < 0.05 on ruler success or steps to converge, or at
  least 3 of 12 more converged runs) **and** is not significantly worse than `ens-fix`.
- **A change helps but does not explain it** if it beats `ens` but stays significantly below `ens-fix`.
- **No effect** if it does not beat `ens`.
- `ens-wu-lr` against `ens-wu`: the halved learning rate on top of the warm start.
- `hilserl-wu` against `hilserl`: the warm start on plain HIL-SERL.
- Single changes may interact; this design does not test all combinations. If no single change explains the fix, the
  next step is leave-one-out from `ens-fix`.
- Sanity check: on the second machine the fixed ensemble converged in about 9 000 steps with experts and intermediates
  at about 93 %. `ens-fix` here should be close; a large gap means something differs and must be found first.

## 4. Steps for the Claude on the first machine

Rules: do not change the config, the seeds or the reading above (the only allowed change is the orientation limits,
see Step 0); never delete or modify anything under `runs/`; never type, paste or read credentials (the user does any
`wandb login` or `sudo`); report with actual output.

### Step 0 — checks (stop and report if any fails)

```bash
cd ~/Downloads/hil-serl_aqi                     # the repo; adjust if it lives elsewhere
git log -1 --oneline; git status --short | head -20
pgrep -af "train_aqi_rlpd|run_campaign" || echo "nothing running"
systemctl --user list-units --type=service --state=running | grep -i -E "campaign|aqi|rq|ens|oracle|novice|suite" || echo "no campaign unit"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
df -h .
ls examples/campaigns/ensemble_fix_ablation.yaml
# the new stack must be installed (patches 0014-0016); the shorter ruler (0037) is NOT needed:
grep -c "synthetic_rotation_control" examples/run_campaign.py      # > 0
grep -c "synthetic_orientation_monitor" examples/run_campaign.py   # > 0
grep -c "converge_mode" examples/run_campaign.py                   # > 0
grep -c "utd_ratio" examples/run_campaign.py                       # > 0
git status --short examples/ruler_select.py                         # see below
```

- Nothing may be running. Free disk: at least 200 GB (96 runs at about 1.5 GB with 12 checkpoints).
- If any grep prints 0, the new stack is not installed: stop and tell the user.
- `select_newest` in `examples/eval_ruler.py` is NOT required (that is the shorter ruler, not used here).
- If `examples/ruler_select.py` shows `??` (copied by hand, not from a patch), move it out of the repo:
  `mv examples/ruler_select.py ~/ruler_select.py.handcopied`. This code does not use it, and a later `git apply` of
  patch 0037 would fail on it. Report that you moved it.
- This machine's code may differ from the copy the expected output below was checked on. What matters is that the
  dry-run flags match exactly; if they do not, stop and report the difference.

**Orientation limits:** if this machine has its own stage-0 result, use it:

```bash
python3 -c "import json; print(json.load(open('runs/synthetic_suite/stage0/limits.json'))['limits'].get('cuboid'))"
```

If it prints tilt and yaw values that differ from 0.384 / 0.1396, add `--synthetic_orientation_tilt_rad <tilt>
--synthetic_orientation_yaw_rad <yaw>` to every command below (dry run and launch) and report the values. If the file
does not exist, keep the config values.

```bash
./.venv/bin/python examples/run_campaign.py --config examples/campaigns/ensemble_fix_ablation.yaml \
    --campaign runs/ensemble_fix_ablation --list_cells | wc -l          # must print 96, no "nov" cells
./.venv/bin/python examples/run_campaign.py --config examples/campaigns/ensemble_fix_ablation.yaml \
    --campaign runs/ensemble_fix_ablation --dry_run > /tmp/ens_abl_dry.txt 2>&1; echo "exit $?"
for a in ens ens-m1 ens-fix hilserl ens-wu ens-wu-lr ens-k8 hilserl-wu; do
  grep "DRY learner" /tmp/ens_abl_dry.txt | grep -m1 "checkpoints/${a}_exp00_s0 " \
   | grep -o -E -- "--aqi_mode [a-z]+|--(no)?demo_pretrain( |$)|--demo_pretrain_steps [0-9]+|--learning_rate [0-9.e-]+|--critic_ensemble_size [0-9]+|--critic_subsample_size [0-9]+" \
   | tr '\n' ' '; echo " <- $a"; done
grep "DRY actor" /tmp/ens_abl_dry.txt | grep -m1 "checkpoints/ens_exp00_s0 " \
   | grep -o -E -- "--max_steps [0-9]+|--synthetic_rotation_control|--synthetic_orientation_monitor|--synthetic_orientation_(tilt|yaw)_rad [0-9.]+|--converge_mode [a-z]+|--converge_eval_every [0-9]+|--utd_ratio [0-9]+" | tr '\n' ' '; echo
head -30 /tmp/ens_abl_dry.txt | grep -E "orientation box|convergence"
```

The method lines must be exactly this (verified on a copy of this machine's code with the bundle up to patch 0021,
without 0037, 2026-11-06):

```
--aqi_mode logged --nodemo_pretrain --demo_pretrain_steps 0 --critic_ensemble_size 10 --critic_subsample_size 2  <- ens
--aqi_mode logged --nodemo_pretrain --demo_pretrain_steps 0 --critic_ensemble_size 10 --critic_subsample_size 1  <- ens-m1
--aqi_mode logged --demo_pretrain --demo_pretrain_steps 5000 --learning_rate 0.00015 --critic_ensemble_size 8 --critic_subsample_size 1  <- ens-fix
--aqi_mode hilserl --nodemo_pretrain --demo_pretrain_steps 0  <- hilserl
--aqi_mode logged --demo_pretrain --demo_pretrain_steps 5000 --critic_ensemble_size 10 --critic_subsample_size 2  <- ens-wu
--aqi_mode logged --demo_pretrain --demo_pretrain_steps 5000 --learning_rate 0.00015 --critic_ensemble_size 10 --critic_subsample_size 2  <- ens-wu-lr
--aqi_mode logged --nodemo_pretrain --demo_pretrain_steps 0 --critic_ensemble_size 8 --critic_subsample_size 2  <- ens-k8
--aqi_mode hilserl --demo_pretrain --demo_pretrain_steps 5000  <- hilserl-wu
```

and the new-stack line:

```
--max_steps 30000 --synthetic_rotation_control --synthetic_orientation_monitor --synthetic_orientation_tilt_rad 0.384 --synthetic_orientation_yaw_rad 0.1396 --converge_mode eval --converge_eval_every 2000 --utd_ratio 1
```

plus the header lines "orientation box roll +/-0.1, pitch +/-0.1, yaw +/-0.4 rad" and "mini-eval SR >= 0.6 in 2
evaluations in a row, every 2000 env steps". The ruler commands show `--select best_window --select_window_k 3` and no
`--select_newest`. (Spacing may differ by a blank; the flags and values may not.) No
`--learning_rate` means the built-in 3e-4; both HIL-SERL arms get no critic flags.

W&B: check `./.venv/bin/wandb status`. If it is not logged in, start with `--wandb offline` instead of `online` and tell
the user (they can sync later). Never enter a key.

### Step 1 — start (in the background, OOM-protected)

```bash
WANDB_RUN_GROUP=ensemble_fix_ablation \
systemd-run --user --unit=ens_ablation --collect \
  -p ManagedOOMPreference=avoid -p WorkingDirectory=$PWD \
  -- bash -c './.venv/bin/python examples/run_campaign.py \
       --config examples/campaigns/ensemble_fix_ablation.yaml \
       --campaign runs/ensemble_fix_ablation \
       --converge_require_autonomous --wandb online \
       >> runs/ensemble_fix_ablation.out 2>&1'
```

Confirm after 15–30 minutes: `systemctl --user status ens_ablation`, `tail -n 40 runs/ensemble_fix_ablation.out`, the
first learner log under `runs/ensemble_fix_ablation/logs/` shows training steps. Note the time per run once the first
run has finished (training + ruler) and report it with the unit name, the W&B mode and the expected end. Then stop. The
run is resumable: if it dies, re-running the same command continues with the missing runs.

### Step 2 — when it is done (or for an early look at finished arms)

```bash
./.venv/bin/python examples/run_campaign.py --config examples/campaigns/ensemble_fix_ablation.yaml \
    --campaign runs/ensemble_fix_ablation --stages eval,analyze --baseline hilserl
cp runs/ensemble_fix_ablation/report/dependency.md   runs/ensemble_fix_ablation/report/dependency_baseline_hilserl.md
cp runs/ensemble_fix_ablation/report/dependency.json runs/ensemble_fix_ablation/report/dependency_baseline_hilserl.json
./.venv/bin/python examples/run_campaign.py --config examples/campaigns/ensemble_fix_ablation.yaml \
    --campaign runs/ensemble_fix_ablation --stages analyze --baseline ens
cp runs/ensemble_fix_ablation/report/dependency.md   runs/ensemble_fix_ablation/report/dependency_baseline_ens.md
cp runs/ensemble_fix_ablation/report/dependency.json runs/ensemble_fix_ablation/report/dependency_baseline_ens.json
```

(Add the orientation-limit flags here too if Step 0 used them.) Pack the results for the user (small text files only,
no checkpoints):

```bash
cd runs && find ensemble_fix_ablation \( -name cell.json -o -name episodes.jsonl -o -name .done -o -name campaign.json \
  -o -name learner_progress.json -o -path '*/report/*' -o -path '*/operators/*' \) -print0 \
  | tar czf ~/ensemble_fix_ablation_results.tgz --null -T - \
  && ls -lh ~/ensemble_fix_ablation_results.tgz && tar tzf ~/ensemble_fix_ablation_results.tgz | grep -c cell.json
```

The last number must be 96 (fewer if the campaign was stopped early: say which arms are complete).
