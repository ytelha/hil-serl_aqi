#!/usr/bin/env bash
# Run the final study unattended, in the background, in one go (docs/final_study.md): setup if needed, machine check,
# stage 0, the smoke campaign, then F1_hard and F2_standard, a retry of any crashed cell, the score audit, the report
# and the plots. Nothing waits for a person: every gate is a decision written to results/final/RUN_STATUS.md, and a
# failing step is recorded while the run goes on. All results go to results/final/ only (refreshed after every step
# and every 6 hours); it never pushes anything (never --push): the repository is public, the results are not.
#
#   bash tools/run_final_unattended.sh          # start in the background and return
#   tail -f runs/final/unattended.out           # watch it (also copied to results/final/unattended.out)
#
# Before starting (optional): `./.venv/bin/wandb login` (else W&B logs offline on this machine only).
# Keep the machine on: do not suspend it or log out (lock the screen instead). The script holds a sleep inhibitor
# while it runs where systemd allows it.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
if [ "${1:-}" != "--here" ]; then
    mkdir -p runs/final
    LOG="runs/final/unattended.out"
    INHIBIT=()
    if command -v systemd-inhibit >/dev/null 2>&1 \
            && systemd-inhibit --what=sleep:idle --who="AQI final study" --why="probe" true >/dev/null 2>&1; then
        INHIBIT=(systemd-inhibit --what=sleep:idle --who="AQI final study" --why="unattended training run" --mode=block)
    fi
    nohup setsid "${INHIBIT[@]}" bash "$0" --here "$@" > "${LOG}" 2>&1 < /dev/null &
    echo "started in the background (pid $!), $([ ${#INHIBIT[@]} -gt 0 ] && echo 'sleep blocked while it runs' || echo 'no sleep inhibitor available: keep the machine awake')"
    echo "log:     ${REPO}/${LOG}"
    echo "status:  ${REPO}/results/final/RUN_STATUS.md (after the first step)"
    echo "results: ${REPO}/results/final/  <- copy this folder for the analysis"
    exit 0
fi
shift
echo "[unattended] $(date -Is) on $(hostname), commit $(git rev-parse --short HEAD)"
if [ ! -x .venv/bin/python ]; then
    echo "[unattended] no .venv: running tools/setup_workstation.sh"
    bash tools/setup_workstation.sh || echo "[unattended] setup reported a problem; trying to run anyway"
fi
./.venv/bin/python -u examples/synthetic_suite.py --suite final all --parallel 2 "$@"
rc=$?
mkdir -p results/final
cp runs/final/unattended.out results/final/unattended.out 2>/dev/null || true
echo "[unattended] $(date -Is) finished (exit code ${rc}); results in ${REPO}/results/final/"
exit ${rc}
