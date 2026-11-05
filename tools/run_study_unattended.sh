#!/usr/bin/env bash
# Run the whole comprehensive study unattended, in the background, in one go (docs/comprehensive_study.md):
# setup if needed, machine check, stage 0, smoke, ruler check, the main campaigns, audit, report, plots. Nothing
# waits for a person: every gate is a decision written to results/study/RUN_STATUS.md, and a failing step is
# recorded while the run goes on. The results folder results/study/ is refreshed after every step and every 6 hours,
# and pushed to the branch `study-results` of the GitHub remote when this machine can push.
#
#   bash tools/run_study_unattended.sh             # start in the background and return
#   bash tools/run_study_unattended.sh --no-push   # same, without pushing the results
#   tail -f runs/study/unattended.out              # watch it (also in results/study/unattended.out)
#
# Before starting: `./.venv/bin/wandb login` (else W&B logs offline on this machine only); for the push, git must be
# able to push to the remote without a prompt (e.g. `gh auth login`); without it the results stay in results/study/.
# Keep the machine on: do not suspend it or log out (lock the screen instead). The script holds a sleep inhibitor
# while it runs where systemd allows it.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
if [ "${1:-}" != "--here" ]; then
    mkdir -p runs/study
    LOG="runs/study/unattended.out"
    INHIBIT=()
    if command -v systemd-inhibit >/dev/null 2>&1 \
            && systemd-inhibit --what=sleep:idle --who="AQI study" --why="probe" true >/dev/null 2>&1; then
        INHIBIT=(systemd-inhibit --what=sleep:idle --who="AQI study" --why="unattended training run" --mode=block)
    fi
    nohup setsid "${INHIBIT[@]}" bash "$0" --here "$@" > "${LOG}" 2>&1 < /dev/null &
    echo "started in the background (pid $!), $([ ${#INHIBIT[@]} -gt 0 ] && echo 'sleep blocked while it runs' || echo 'no sleep inhibitor available: keep the machine awake')"
    echo "log:     ${REPO}/${LOG}"
    echo "status:  ${REPO}/results/study/RUN_STATUS.md (after the first step)"
    exit 0
fi
shift
PUSH=(--push)
ARGS=()
for a in "$@"; do
    case "$a" in
        --no-push) PUSH=() ;;
        *) ARGS+=("$a") ;;
    esac
done
echo "[unattended] $(date -Is) on $(hostname), commit $(git rev-parse --short HEAD)"
if [ ! -x .venv/bin/python ]; then
    echo "[unattended] no .venv: running tools/setup_workstation.sh"
    bash tools/setup_workstation.sh || echo "[unattended] setup reported a problem; trying to run anyway"
fi
exec ./.venv/bin/python -u examples/synthetic_suite.py --suite study all --parallel 2 "${PUSH[@]}" "${ARGS[@]}"
