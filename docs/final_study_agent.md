# The final study: instructions for the agent on the second training machine

This is for the Claude working on the second training machine (Linux, RTX 5090), which gets this repo by cloning it
from GitHub.

**Nobody looks after the machine while the study runs.** Your job:
1. set it up;
2. start the whole study in one unattended run;
3. make sure it is running;
4. stop.

The design and the commands are in [final_study.md](final_study.md): read it first.

## Rules

- **Start everything with one command, in the background:** `bash tools/run_final_unattended.sh`.
  - It never waits for a person and never stops for a failure.
  - Its decisions and warnings go to `results/final/RUN_STATUS.md`.
- **Do not change:**
  - a config, an arm, the operators, the seeds, the careless share or the asking rate;
  - the ruler, the decision rules or the run's automatic decisions.
- **Results stay on this machine.** Never push results, never push the code branch, and never commit results.
  `results/final/` is the folder the user copies.
- **Credentials:** never type, paste or read the W&B key, a GitHub token or a password. The user runs
  `./.venv/bin/wandb login` themselves, if they want it.
- **Data:** never delete anything in `runs/` or `results/`.
- **Fixes:**
  - only before the run starts, and only for a setup problem that would make every cell fail (a missing system library,
    a wrong driver);
  - one commit per fix, with a test; then `git format-patch -1 HEAD -o results/final/patches/`;
  - tell the user.
- **Never stop or kill** a process you did not start.
- **Report** with the actual output, in plain words.

## Steps

1. **Code version:** `git log -1 --format='%h %s'` must show the commit the user gave you. If it does not,
   `git fetch && git checkout <that commit>`.
2. **Setup (about 15 minutes):** `bash tools/setup_workstation.sh`.
   - Report the GPU and driver line, and the import check (every line `ok`, JAX sees the GPU).
   - If it fails, find the cause and fix the setup.
3. **Tests of this study (about 1 minute):** `./.venv/bin/python tests/test_final_suite.py` must end with
   `16 tests passed`.
   - If not, report the failure.
   - Fix it only if it is a setup problem (see Fixes); never change a config to make a test pass.
4. **Machine check:** `./.venv/bin/python examples/synthetic_suite.py --suite final check --no_export`. Report READY
   or the problem list.
   - **W&B:** a missing key is fine (the run logs offline). Tell the user they can still run
     `./.venv/bin/wandb login` now.
   - **Free disk:** the check wants about 300 GB for `runs/`. If there is less, the run decides on its own, per
     campaign, to keep only the newest checkpoints. Report the free space; never delete anything yourself.
   - **A missing system package or driver** (anything needing `sudo`): give the user the exact command to run; never
     enter a password yourself.
5. **Start:** `bash tools/run_final_unattended.sh`. It prints the process id, the log path and the results path.
6. **Confirm it runs (about 5–15 minutes):**
   - `tail -n 40 runs/final/unattended.out` shows the machine check and then stage 0 starting;
   - `results/final/RUN_STATUS.md` lists the first finished step.
   - If the process died within these minutes, report the end of the log. That is the only case where you fix
     something and start again.
7. **Hand over and stop.** Tell the user:
   - the process id;
   - whether W&B is online or offline;
   - the free disk;
   - that the run takes about 5–6 days: about 2 h of checks and smoke, then `F1_hard`, then `F2_standard`;
   - that `F1_hard`'s results are complete in `results/final/` as soon as it finishes;
   - that the machine must stay on: no suspend and no logout. **Under WSL2 the Windows host must not sleep either;**
     that is a Windows setting the user changes, not you.

   Then end the session. The run carries on without you.

## What comes back

The results folder `results/final/` holds:
- the reports, plots and score audits (`report/inconsistent_audit.*` for operator-decided takeovers,
  `report/audit_robot_asks/` for robot-asked ones);
- the ruler outcomes per kept checkpoint, training records, careful/careless takeover logs and compacted logs;
- stage 0, `machine.json`, and `RUN_STATUS.md` with every step, decision and warning.

It is on this machine only. The user copies it.
