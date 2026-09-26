# Running AQI-HIL-SERL on the RWTH cluster CLAIX

The code gets its **own copy on the cluster**. The repo on your PC (and on the
training machine) stays where it is and keeps working unchanged. You edit
the cluster copy with VS Code over SSH, and training runs as Slurm jobs on the
H100 GPU nodes.

```
your PC (VS Code, Remote-SSH) --SSH--> login node login23-1/-2
                                         $HOME/hil-serl_aqi   code (backed up)
                                         $HPCWORK/aqi_runs    runs, checkpoints, logs
                                         sbatch ...
                                            |
                                            v
                                       c23g GPU nodes: 1 H100 per cell
```

Everything cluster-specific is in `cluster/claix/`. Outside the cluster
nothing uses it: the only code changes are two opt-in options.
- `AQI_TRAINER_PORT` sets the learner/actor ports; the default is still 5588/5589.
- `run_campaign.py --shard i/n` and `--list_cells`.

## 0. Before you start (done by you, once)

1. **HPC account:** register in the RWTH RegApp with your RWTH account
   ([First steps](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/b7221149cca349e196cc0a28fcba8cc9/)).
   You need the RWTH VPN password and MFA set up in the IdM Selfservice.
2. **Network:** from outside the RWTH network, connect the RWTH VPN first (or
   use eduroam at RWTH).
3. **SSH key** (on your PC, PowerShell):
   ```powershell
   ssh-keygen -t ed25519 -f $HOME\.ssh\id_ed25519_claix -C "claix"
   ```
   - Use a passphrase.
   - In the RegApp: *Index → My SSH Pubkeys → Add SSH Key*, and paste the
     contents of `id_ed25519_claix.pub`, never the private key.
   - Then *Set SSH Key → ADD → usage "interactive"*.
   - With the key, CLAIX still asks for your HPC password + 2FA code once
     every 10 hours
     ([Manage SSH keys](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/14e38a2d1975405d9fbf623a2237e552/)).
4. **SSH config:** append `cluster/claix/ssh_config.example` to
   `C:\Users\<you>\.ssh\config` and put in your HPC username. Test it with
   `ssh claix` (password + 2FA the first time).

## 1. Computing time: what it costs

| | |
|---|---|
| billing | 1 H100 for 1 hour = 24 core-h |
| free student quota | 500 core-h per month ≈ **20 GPU-hours per month** |
| one training cell (40k env steps) | about 1.5–2.5 GPU-h (the actor runs at real-time 10 Hz: 40k steps ≈ 67 min, plus demo pre-training and the ruler). The smoke test shows the real number. |
| one 54-cell campaign | about 80–135 GPU-h ≈ 2,000–3,200 core-h |

The free quota covers the smoke test and a few cells, not a campaign.
- **Thesis project:** 1,500 GPU-h over 6 months. The application is the
  title only, with no review.
- **Who applies:** the applicant (PI) may be an RWTH employee without a
  doctorate, e.g. your supervisor, who then adds you as a member
  ([project catalogue](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/45825b06afb647e194be4a5b9f5b8768/)).
- **Using it:** jobs charge the project with `--account <project>`.
- **Over quota:** you are moved to a low-priority queue, not stopped.

## 2. Make the separate copy on the cluster

Pick **one** source: the code you want the cluster to run.

**A. From your PC with a git bundle**, a single file (no GitHub/GitLab
needed; Git Bash on the PC):
```bash
cd ~/Desktop/ZIP/hil-serl_aqi-neu
git bundle create ../aqi.bundle --all
scp ../aqi.bundle claix:
ssh claix 'git clone -b main ~/aqi.bundle ~/hil-serl_aqi && rm ~/aqi.bundle'
```
Later updates: make a new bundle, `scp` it, then on the cluster
`git pull ~/aqi.bundle main`.

**B. Through a private RWTH GitLab project** (git.rwth-aachen.de). This is the
best choice if the PC, the training machine and the cluster should all stay in
sync: every machine `git push`/`git pull`s the same remote.

**C. From the training machine**, if its code is newer. Copy it without
`.venv` and `runs`:
```bash
rsync -a --exclude .venv --exclude runs --exclude '*.tar' ./ <user>@copy23-1.hpc.itc.rwth-aachen.de:hil-serl_aqi/
```

## 3. Python environment (once, on a login node)

1. **Lock the versions** so the cluster runs exactly what the training machine
   runs (same MuJoCo physics, same JAX). On the **training machine**, from
   the repo root:
   ```bash
   ./.venv/bin/python -m pip freeze > requirements-lock.txt
   ./.venv/bin/python -c 'import platform; print(platform.python_version())' > python-version
   ```
2. Copy both files to `cluster/claix/` on the cluster.
3. Then, on the cluster:
   ```bash
   ssh claix
   cd ~/hil-serl_aqi
   bash cluster/claix/probe.sh 2>&1 | tee ~/probe.txt   # facts about your account/nodes
   bash cluster/claix/setup_env.sh                      # ~10-20 min
   ```

`setup_env.sh` does the following:
- builds `.venv` with `uv`, which downloads the right Python by itself;
- skips the robot-hardware packages: headless stand-ins in `stubs/` replace
  them on the cluster only;
- installs the headless OpenCV;
- links `runs -> $HPCWORK/aqi_runs`;
- writes VS Code settings that keep the file watcher out of `runs/` and
  `.venv/`.

Without a lock, `--fallback` installs from the repo's requirement files. Use
that for testing only: the MuJoCo version may then differ from the training
machine.

## 4. VS Code on your PC

1. Install the extension **Remote - SSH** (Microsoft).
2. In your *local* VS Code settings (JSON), add:
   ```json
   "remote.SSH.showLoginTerminal": true,
   "remote.SSH.connectTimeout": 60,
   "remote.SSH.remotePlatform": {"claix": "linux", "claix2": "linux", "claix-gpu": "linux"}
   ```
   `showLoginTerminal` shows the password / 2FA prompt when CLAIX asks for it.
3. Connect: *F1 → Remote-SSH: Connect to Host → claix*, then open
   `/home/<user>/hil-serl_aqi`.
4. In the VS Code terminal: `source cluster/claix/env.sh` (or add
   `alias aqi='source ~/hil-serl_aqi/cluster/claix/env.sh'` to `~/.bashrc`).

**Login-node rules:**
- VS Code, git, editing and `sbatch` are fine.
- **Training is not:** at most 4 cores per user, processes that use a lot of
  CPU for 20 minutes get killed, and GPU processes on `login23-g-1` get killed
  after 30 minutes.
- All login nodes reboot every **Monday at 6:00**, so VS Code reconnects then.
  Running jobs are not affected.

## 5. Smoke test (one H100, ~45 min, ~20 core-h)

```bash
bash cluster/claix/submit_smoke.sh            # add --account <project> if you have one
squeue --me                                    # PD = waiting, R = running
tail -f $HPCWORK/aqi_runs/slurm/smoke_<jobid>.out
```

It checks:
- JAX on the H100 and MuJoCo EGL rendering inside the job;
- all imports;
- the repo tests, including the JAX tests that never ran on the development
  PC;
- one tiny campaign cell end to end.

It must end with `SMOKE TEST PASSED`.

## 6. Campaigns

```bash
bash cluster/claix/submit_campaign.sh examples/campaigns/rq5_quality_score_ext.yaml c7b --account <project>
```

This submits three chained steps:
1. **prepare:** roster, frozen ruler, demos;
2. **cells:** a Slurm array with one H100 per cell, at most `--parallel` (8)
   at once;
3. **analyze:** the report.

Results land in `runs/c7b/`, i.e. `$HPCWORK/aqi_runs/c7b/`:
- `report/dependency.md` and the plots;
- `logs/<cell>.{learner,actor,eval}.log`;
- `slurm/*.out`.

Re-running the same command resumes and skips finished cells.
`--dry` shows the plan without submitting.

**Monitoring and control:**
- `squeue --me`;
- `sacct -X --starttime today -o JobID,JobName%30,State,Elapsed`;
- `scancel <jobid>` to stop.

**Copying results to your PC** (Git Bash on the PC):
```bash
rsync -a --exclude checkpoints claix-copy:/hpcwork/<user>/aqi_runs/c7b/ ./runs_claix/c7b/
```

## 7. Comparability: important for the thesis

The actor runs at real time (10 Hz) while the learner trains as fast as its
GPU allows. On an H100 the learner therefore makes **more gradient updates
per environment step** than on the training machine's GPU. Consequences:
- **Run every campaign completely on one kind of hardware**, baselines
  included.
- **Do not copy cells between the cluster and the training machine.** The
  "copy the `logged_*` cells from C7" shortcuts in the campaign YAML headers
  apply only within one machine.
- **One cell per GPU** (the default here): packing several cells onto one GPU
  would slow each learner by a varying amount.

## 8. Troubleshooting

| symptom | fix |
|---|---|
| VS Code hangs at "Setting up SSH host" | the password/2FA prompt is waiting: `remote.SSH.showLoginTerminal: true`; or `ssh claix` once in a terminal (valid for 10 h) |
| `Permission denied (publickey)` | key not set to "interactive" in the RegApp, or it expired (12 months) |
| `[env.sh] WARNING: no working EGL device` | `python cluster/claix/egl_probe.py -v` inside a GPU job and send the output |
| learner never READY / `Address already in use` | two cells on one port: check `AQI_TRAINER_PORT` in the log header (`[env.sh] ... port=`); the job scripts pick a free pair per job |
| job pending for long | `squeue --me --start`; GPU nodes are shared by all users; `--parallel` does not make it faster |
| `ImportError` for easyhid / pyrealsense2 / pynput | `env.sh` was not sourced (it puts `cluster/claix/stubs` on the path) |
