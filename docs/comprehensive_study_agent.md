# The comprehensive study: instructions for the agent on the second training machine

For the Claude working on the second training machine (Linux, RTX 5090), which gets this repo by cloning it from
GitHub. **Nobody looks after the machine while the study runs:** your job is to set it up, start the whole study in
one unattended run, make sure it is running, and stop. The design, the decision rules and the commands are in
[comprehensive_study.md](comprehensive_study.md): read it first.

## Rules

- **Start everything with one command, in the background:** `bash tools/run_study_unattended.sh`. It never waits for a
  person and never stops for a failure; its decisions and warnings go to `results/study/RUN_STATUS.md`.
- **Do not change** a config, an arm, the seeds, the careless share, the ruler, the decision rules or the run's
  automatic decisions.
- **Credentials:** never type, paste or read the W&B key, a GitHub token or a password. The user runs
  `./.venv/bin/wandb login` and any GitHub login (`gh auth login`) themselves, if they want them.
- **Data:** never delete anything in `runs/` or `results/`.
- **Fixes:** only before the run starts, and only for a setup problem that would make every cell fail (a missing system
  library, a wrong driver). One commit per fix with a test, `git format-patch -1 HEAD -o results/study/patches/`;
  tell the user. Do not push the code branch.
- **Report** with the actual output, in plain words.

## Steps

1. **Setup (about 15 minutes):** `bash tools/setup_workstation.sh`. Report the GPU and driver line, and the import
   check (every line `ok`, JAX sees the GPU). If it fails, find the cause and fix the setup.
2. **Machine check:** `./.venv/bin/python examples/synthetic_suite.py --suite study check --no_export`. Report READY or
   the problem list. A missing W&B key is fine (the run logs offline); tell the user they can still run
   `./.venv/bin/wandb login` now. Anything else on the list: fix it if it is a setup problem, else report it.
3. **Start:** `bash tools/run_study_unattended.sh`. It prints the process id and the log path.
4. **Confirm it runs (about 5–15 minutes):** `tail -n 40 runs/study/unattended.out` shows the machine check and then
   stage 0 starting; `results/study/RUN_STATUS.md` lists the first finished step. If the process died within these
   minutes, report the end of the log; that is the only case to fix something and start again.
5. **Hand over and stop:** tell the user:
   - the process id;
   - whether W&B is online or offline;
   - whether the results push to the branch `study-results` worked (`RUN_STATUS.md`, decision "results push", or
     `[push]` lines in the log);
   - that the run takes about 4–7 days and the machine must stay on, without suspend or logout.

   Then end the session. The run carries on without you.

## What comes back

The results folder `results/study/`:
- the reports, plots and score audits;
- the ruler outcomes, training records, careful/careless takeover logs and compacted logs;
- `RUN_STATUS.md`, with every step, decision and warning.

It is on the machine, and on the GitHub branch `study-results` if the push works.
