#!/usr/bin/env bash
# Watchdog of phase 3 (docs/extension_phase3.md): every 10 minutes, if the pipeline (tools/run_phase3_unattended.sh)
# is not running and has not finished, start it again. Restarting is safe: the pipeline resumes (finished runs are
# kept, an interrupted run is retrained from scratch) and its lock stops a second copy. Gives up after 6 starts within
# 3 hours (a crash loop needs a person) and records that in runs/phase3/WATCHDOG.md. Ends when RUN_STATUS.json says
# "finished". Survives a logout, not a reboot: after a reboot, start it again (or let an @reboot entry do it).
#
#   bash tools/phase3_watchdog.sh            # detaches and returns
#   tail -f runs/phase3/watchdog.out
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
ROOT="runs/phase3"
mkdir -p "${ROOT}"
if [ "${1:-}" != "--here" ]; then
    nohup setsid bash "$0" --here >> "${ROOT}/watchdog.out" 2>&1 < /dev/null &
    echo "watchdog started in the background (pid $!); log ${REPO}/${ROOT}/watchdog.out"
    exit 0
fi
exec 8> "${ROOT}/.watchdog.lock"
if ! flock -n 8; then echo "[watchdog] $(date -Is) another watchdog runs: exiting"; exit 0; fi
say() { echo "[watchdog] $(date -Is) $*"; }
finished() {
    [ -f "${ROOT}/RUN_STATUS.json" ] && grep -q '"state": "finished' "${ROOT}/RUN_STATUS.json"
}
running() {   # the pipeline holds its lock while it runs
    ! flock -n "${ROOT}/.pipeline.lock" true 2>/dev/null
}
STARTS=()
say "watching the phase-3 pipeline (commit $(git rev-parse --short HEAD))"
while true; do
    if finished; then say "the pipeline has finished: watchdog ends"; exit 0; fi
    if ! running; then
        now=$(date +%s)
        recent=0
        for t in ${STARTS[@]+"${STARTS[@]}"}; do [ $((now - t)) -lt 10800 ] && recent=$((recent + 1)); done
        if [ "${recent}" -ge 6 ]; then
            say "6 starts within 3 hours and it keeps stopping: giving up"
            { echo "# Phase-3 watchdog gave up"; echo; echo "$(date -Is): the pipeline stopped 6 times within 3 hours."
              echo "See runs/phase3/pipeline.out and results/phase3/RUN_STATUS.md; restart with"
              echo "bash tools/run_phase3_unattended.sh once the cause is fixed."; } > "${ROOT}/WATCHDOG.md"
            exit 1
        fi
        say "the pipeline is not running and has not finished: starting it"
        bash tools/run_phase3_unattended.sh
        STARTS+=("${now}")
        sleep 60
    fi
    sleep 600
done
