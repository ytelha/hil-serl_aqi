#!/usr/bin/env bash
# Phase 3 of the extension study, unattended and in one go (docs/extension_phase3.md):
#   check; stage 0 (tests; the hex orientation limits copied from phase 2's runs/extension/stage0);
#   X0_probe    the NaN probe: P1_hex's diverged keepall run (int00, seed 0) twice, 3 000 steps, on its own inputs
#               (copied from runs/extension/P1_hex and checked by hash) -> runs/phase3/X0_probe/PROBE.md
#   X_smoke     every new code path once (8 runs, 4 000 steps)            -> runs/phase3/X_smoke/SMOKE.md
#   X1_careless the main campaign: sometimes-careless operator, 12 arms, 144 runs, fixed budget of 20 000 steps
#   X2_careful  the careful twin: 6 arms, 72 runs, X1's operators, demonstrations and ruler (copied, checked by hash)
#   audit, the phase-3 analysis (results/phase3/PHASE3_REPORT.md), report, plots.
# Runs 2 side by side; each run is scored right after training and pruned to the checkpoints its result needs. A run
# that crashes or diverges (the learner's divergence guard) is retrained once from scratch. Nothing waits for a
# person; every step records its outcome (results/phase3/RUN_STATUS.md) and the next one starts. Reads runs/extension/
# only; never pushes anything (the repository is public, the results are not). Resumable: run it again after an
# interruption and finished runs are kept.
#
#   bash tools/run_phase3_unattended.sh             # start in the background and return
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
PY="./.venv/bin/python"
say "on $(hostname), commit $(git rev-parse --short HEAD)"
if [ ! -x "${PY}" ]; then say "no .venv in ${REPO}: nothing can run"; exit 2; fi
git diff --quiet HEAD -- examples serl_launcher franka_sim || say "WARNING: local changes in the training code (git diff HEAD)"
for f in runs/extension/decision.json runs/extension/stage0/limits.json runs/extension/P1_hex/operators/int00.json; do
    [ -e "${f}" ] || say "WARNING: ${f} missing (phase 2's root); the suite records what it does without it"
done
"${PY}" -u examples/synthetic_suite.py --suite phase3 --root "${ROOT}" --results "${RES}" all --parallel 2 "$@"
RC=$?
say "finished (exit ${RC}; 1 = finished with recorded warnings, see ${RES}/RUN_STATUS.md)"
cp "${ROOT}/pipeline.out" "${RES}/" 2>/dev/null || true
exit 0
