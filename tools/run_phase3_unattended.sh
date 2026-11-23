#!/usr/bin/env bash
# Phase 3 of the extension study, unattended and in one go (docs/extension_phase3.md):
#   check; stage 0 (tests; the hex orientation limits: copied from phase 2's runs/extension/stage0 if that folder is
#               here, else measured);
#   X0_probe    the NaN probe: P1_hex's diverged keepall run (int00, seed 0) twice, 3 000 steps, on P1_hex's inputs
#               (copied from runs/extension/P1_hex and checked by hash; on a fresh clone sampled again from the same
#               config)                                                    -> runs/phase3/X0_probe/PROBE.md
#   X_smoke     every new code path once (8 runs, 4 000 steps)            -> runs/phase3/X_smoke/SMOKE.md
#   X1_careless the main campaign: sometimes-careless operator, 12 arms, 144 runs, fixed budget of 20 000 steps
#   X2_careful  the careful twin: 6 arms, 72 runs, X1's operators, demonstrations and ruler (copied, checked by hash)
#   audit, the phase-3 analysis (results/phase3/PHASE3_REPORT.md), report, plots, and an archive of the results
#   folder for the transfer: results/phase3_results_<date>.tgz (+ .sha256).
# Runs 2 side by side; each run is scored right after training and pruned to the checkpoints its result needs. A run
# that crashes or diverges (the learner's divergence guard) is retrained once from scratch. Nothing waits for a
# person; every step records its outcome (results/phase3/RUN_STATUS.md) and the next one starts. Reads runs/extension/
# only; never pushes anything (the repository is public, the results are not).
# Resumable: run it again after an interruption; finished runs are kept, an interrupted run is retrained from scratch.
# A lock (runs/phase3/.pipeline.lock) makes a second copy exit at once, so tools/phase3_watchdog.sh can relaunch it
# safely.
#
#   bash tools/run_phase3_unattended.sh             # start in the background and return
#   bash tools/phase3_watchdog.sh                   # optional: relaunch it if it dies before it finished
#   tail -f runs/phase3/pipeline.out                # watch it
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
ROOT="runs/phase3"
RES="results/phase3"
if [ "${1:-}" != "--here" ]; then
    mkdir -p "${ROOT}"
    LOG="${ROOT}/pipeline.out"
    INHIBIT=()
    if command -v systemd-inhibit >/dev/null 2>&1 \
            && systemd-inhibit --what=sleep:idle --who="AQI phase 3" --why="probe" true >/dev/null 2>&1; then
        INHIBIT=(systemd-inhibit --what=sleep:idle --who="AQI phase 3" --why="unattended training run" --mode=block)
    fi
    nohup setsid "${INHIBIT[@]}" bash "$0" --here "$@" >> "${LOG}" 2>&1 < /dev/null &
    echo "started in the background (pid $!), $([ ${#INHIBIT[@]} -gt 0 ] && echo 'sleep blocked while it runs' || echo 'no sleep inhibitor available: keep the machine awake')"
    echo "log:     ${REPO}/${LOG}"
    echo "status:  ${REPO}/${RES}/RUN_STATUS.md"
    exit 0
fi
shift
say() { echo "[phase3] $(date -Is) $*"; }
mkdir -p "${ROOT}"
# one pipeline at a time: a second copy (a relaunch while the first still runs) exits here
exec 9> "${ROOT}/.pipeline.lock"
if ! flock -n 9; then say "another phase-3 pipeline holds ${ROOT}/.pipeline.lock: not starting a second one"; exit 0; fi
say "on $(hostname), commit $(git rev-parse --short HEAD)"
PY="./.venv/bin/python"
if [ ! -x "${PY}" ]; then
    say "no .venv: running tools/setup_workstation.sh"
    bash tools/setup_workstation.sh || say "setup reported a problem; trying to run anyway"
fi
if [ ! -x "${PY}" ]; then say "still no .venv/bin/python: nothing can run"; exit 2; fi
git diff --quiet HEAD -- examples serl_launcher franka_sim || say "WARNING: local changes in the training code (git diff HEAD)"
[ -d runs/extension ] || say "runs/extension (phase 2) is not here: the suite measures the hex limits and samples the probe's inputs afresh"
"${PY}" -u examples/synthetic_suite.py --suite phase3 --root "${ROOT}" --results "${RES}" all --parallel 2 "$@"
RC=$?
say "finished (exit ${RC}; 1 = finished with recorded warnings, see ${RES}/RUN_STATUS.md)"
cp "${ROOT}/pipeline.out" "${RES}/" 2>/dev/null || true
# the results folder as one archive for the transfer (no checkpoints or demonstrations are in it)
ARCH="results/phase3_results_$(date +%Y%m%d_%H%M).tgz"
if tar czf "${ARCH}" -C results phase3 2>/dev/null; then
    sha256sum "${ARCH}" > "${ARCH}.sha256"
    say "archive ${ARCH} ($(du -h "${ARCH}" | cut -f1)), sha256 in ${ARCH}.sha256"
else
    say "WARNING: could not write ${ARCH}"
fi
exit 0
