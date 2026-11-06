# Which part of the ensemble fix mattered? — the ablation on the first training machine

The first ensemble extension lost to plain HIL-SERL; the fixed one matched it. The fix changed four things at once, so
nobody knows which one did the work. This campaign tests each change on its own. It runs on the **first training
machine** (original platform, the code of the RQ3/RQ4 campaigns), so its numbers sit next to those campaigns.

- Config: `examples/campaigns/ensemble_fix_ablation.yaml` (new, nothing else changes).
- Campaign folder: `runs/ensemble_fix_ablation`; W&B group `ensemble_fix_ablation`.
- Background: `docs/thesis/rq4_ensemble_convergence_and_qa_log.md` (sections 2–3: the four changes and why).

## 1. The four changes, and what stays the same

| | first version (RQ3) | fixed version (RQ4) |
|---|---|---|
| critics K | 10 | 8 |
| target of a critic (M) | minimum of 2 random critics | 1 random critic (no minimum) — "de-pessimisation" |
| warm start | none | 5 000 learner steps on the demonstrations first |
| learning rate | 3e-4 | 1.5e-4 |
| per-member targets | on | on (unchanged) |
| bootstrap masks | p = 0.8 | p = 0.8 (unchanged) |

Per-member targets and the masks were already on in the first version, so they are not part of the fix.

## 2. The arms (18 runs each: 3 operators per tier × 3 tiers × 2 seeds)

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

Runs are trained arm by arm in this order, so if the campaign has to stop early, the first four arms (72 runs, about
5 days) already answer the main question. Everything else is copied from `rq4_ensemble_convergence.yaml`: same roster
seed, 6 demonstrations, 40 000-step cap, convergence rule (90 % of the last 20 episodes without help) and ruler. All
arms are trained fresh here; no runs are copied from RQ3/RQ4, whose code predates patches 0001–0010.

**Cost:** 144 runs. On this machine a run took about 1.5–1.7 h, one at a time on the 8 GB GPU: about **9–10 days**.

## 3. How the result is read (fixed before any run)

Outcomes per arm, from `report/dependency.md`: ruler success (mean and per tier), converged runs of 18, steps to
converge, ICC. Paired comparisons use the same operator and seed (sign test over the non-tied pairs).

- **A change explains the fix** if its arm beats `ens` (sign test p < 0.05 on ruler success, or clearly more converged
  runs: at least 4 of 18 more) **and** is not significantly worse than `ens-fix`.
- **A change helps but does not explain it** if it beats `ens` but stays significantly below `ens-fix`.
- **No effect** if it does not beat `ens`.
- `ens-wu-lr` against `ens-wu` gives the effect of the halved learning rate on top of the warm start.
- `hilserl-wu` against `hilserl` gives the warm start's effect on plain HIL-SERL (success, and ICC: an earlier run on
  this machine, `rq1_3seed_pretrain`, found that pre-training flattens the teacher effect to ICC 0.00).
- Single changes may interact; this design does not test all combinations. If no single change explains the fix, the
  next step is leave-one-out from `ens-fix`.
- Reference: RQ4 found the fixed version at 82 % and HIL-SERL at 84 % (16 of 18 converged each); `ens-fix` and
  `hilserl` here should land near those. A large gap means something differs from RQ4 and must be found first.

## 4. Steps for the Claude on the first machine

Rules: do not change the config, the seeds or the reading above; never delete or modify anything under `runs/`; never
type, paste or read credentials (the user does any `wandb login` or `sudo`); report with actual output.

### Step 0 — checks (stop and report if any fails)

```bash
cd ~/Downloads/hil-serl_aqi                     # the repo; adjust if it lives elsewhere
git log -1 --oneline; git status --short | head -20
pgrep -af "train_aqi_rlpd|run_campaign" || echo "nothing running"
systemctl --user list-units --type=service --state=running | grep -i -E "campaign|aqi|rq|ens|oracle|novice" || echo "no campaign unit"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
df -h . ; du -sh runs/rq4_ensemble_convergence      # per-run disk = this / 36
ls examples/campaigns/ensemble_fix_ablation.yaml
```

- Nothing may be running. Free disk must be at least 144 × (RQ4 size / 36) × 1.3.
- The code may or may not have the later first-machine bundle applied; this campaign uses only knobs that exist since
  the RQ4 campaign (`demo_pretrain`, `learning_rate`, `critic_ensemble_size`, `critic_subsample_size`), so either works.

```bash
./.venv/bin/python examples/run_campaign.py --config examples/campaigns/ensemble_fix_ablation.yaml \
    --campaign runs/ensemble_fix_ablation --list_cells | wc -l          # must print 144
./.venv/bin/python examples/run_campaign.py --config examples/campaigns/ensemble_fix_ablation.yaml \
    --campaign runs/ensemble_fix_ablation --dry_run > /tmp/ens_abl_dry.txt 2>&1; echo "exit $?"
for a in ens ens-m1 ens-fix hilserl ens-wu ens-wu-lr ens-k8 hilserl-wu; do
  grep "DRY learner" /tmp/ens_abl_dry.txt | grep -m1 "checkpoints/${a}_exp00_s0 " \
   | grep -o -E -- "--aqi_mode [a-z]+|--(no)?demo_pretrain( |$)|--demo_pretrain_steps [0-9]+|--learning_rate [0-9.e-]+|--critic_ensemble_size [0-9]+|--critic_subsample_size [0-9]+" \
   | tr '\n' ' '; echo " <- $a"; done
```

The output must be exactly this (verified on a copy of this machine's code of 31 October):

```
--aqi_mode logged --nodemo_pretrain  --demo_pretrain_steps 0 --critic_ensemble_size 10 --critic_subsample_size 2  <- ens
--aqi_mode logged --nodemo_pretrain  --demo_pretrain_steps 0 --critic_ensemble_size 10 --critic_subsample_size 1  <- ens-m1
--aqi_mode logged --demo_pretrain  --demo_pretrain_steps 5000 --learning_rate 0.00015 --critic_ensemble_size 8 --critic_subsample_size 1  <- ens-fix
--aqi_mode hilserl --nodemo_pretrain  --demo_pretrain_steps 0  <- hilserl
--aqi_mode logged --demo_pretrain  --demo_pretrain_steps 5000 --critic_ensemble_size 10 --critic_subsample_size 2  <- ens-wu
--aqi_mode logged --demo_pretrain  --demo_pretrain_steps 5000 --learning_rate 0.00015 --critic_ensemble_size 10 --critic_subsample_size 2  <- ens-wu-lr
--aqi_mode logged --nodemo_pretrain  --demo_pretrain_steps 0 --critic_ensemble_size 8 --critic_subsample_size 2  <- ens-k8
--aqi_mode hilserl --demo_pretrain  --demo_pretrain_steps 5000  <- hilserl-wu
```

No `--learning_rate` means the built-in 3e-4. Both HIL-SERL arms get no critic flags (the runner keeps HIL-SERL at its
2 critics).

```bash
./.venv/bin/python examples/run_campaign.py --config examples/campaigns/ensemble_fix_ablation.yaml \
    --campaign runs/ensemble_fix_ablation --plan      # roster and ruler grid only, no training
python3 - <<'EOF'
import json
a = json.load(open("runs/ensemble_fix_ablation/campaign.json"))["grid_sha256"]
b = json.load(open("runs/rq4_ensemble_convergence/campaign.json"))["grid_sha256"]
print("ruler grid identical to RQ4:", a == b)
EOF
diff -r runs/ensemble_fix_ablation/operators runs/rq4_ensemble_convergence/operators && echo "roster identical to RQ4"
```

Both must be identical. If the operator folders differ only in files RQ4 does not have, report and continue only if
the operator parameters match.

W&B: check `./.venv/bin/wandb status`. If it is not logged in, start with `--wandb offline` instead of `online` and tell
the user (they can sync later). Never enter a key.

### Step 1 — start (in the background, OOM-protected, as RQ3/RQ4)

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

Confirm after 10–20 minutes: `systemctl --user status ens_ablation`, `tail -n 40 runs/ensemble_fix_ablation.out`, the
first learner log under `runs/ensemble_fix_ablation/logs/` shows training steps. Then report the unit name, W&B mode,
start time and the expected end (about 9–10 days) to the user and stop. The run is resumable: if it dies, re-running
the same command continues with the missing runs.

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

Pack the results for the user (small text files only, no checkpoints):

```bash
cd runs && find ensemble_fix_ablation \( -name cell.json -o -name episodes.jsonl -o -name .done -o -name campaign.json \
  -o -path '*/report/*' -o -path '*/operators/*' \) -print0 | tar czf ~/ensemble_fix_ablation_results.tgz --null -T - \
  && ls -lh ~/ensemble_fix_ablation_results.tgz && tar tzf ~/ensemble_fix_ablation_results.tgz | grep -c cell.json
```

The last number must be 144 (fewer if the campaign was stopped early: say which arms are complete).
