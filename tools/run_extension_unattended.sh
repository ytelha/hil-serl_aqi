#!/usr/bin/env bash
# The extension study, phases 0-2, unattended and in one go (docs/extension_study.md):
#   CPU, in the background from the start: phase 2's stage 0 (tests, cuboid + hex orientation limits) and the
#        operator screen of the four candidate tasks;
#   GPU: phases 0 + 1, the twins of the final study (tools/run_final_twins_unattended.sh), then phase 2: the learning
#        pilots of the passing candidates, the task decision and the smoke test (synthetic_suite.py --suite extension
#        all, which skips the stage 0 and screen already done).
# Nothing waits for a person; every step records its outcome and the next one starts. Results:
#   results/final_twins/  (TWINS_RUN.md)      results/extension/  (RUN_STATUS.md, SCREEN.md, DECISION.md)
#   results/extension_pipeline/PIPELINE_STATUS.md
# Reads runs/final/ only; never pushes anything (the repository is public, the results are not).
#
#   bash tools/run_extension_unattended.sh             # start in the background and return
#   tail -f runs/extension_pipeline/pipeline.out       # watch it
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
LOGDIR="runs/extension_pipeline"
RES="results/extension_pipeline"
if [ "${1:-}" != "--here" ]; then
    mkdir -p "${LOGDIR}"
    LOG="${LOGDIR}/pipeline.out"
    INHIBIT=()
    if command -v systemd-inhibit >/dev/null 2>&1 \
            && systemd-inhibit --what=sleep:idle --who="AQI extension study" --why="probe" true >/dev/null 2>&1; then
        INHIBIT=(systemd-inhibit --what=sleep:idle --who="AQI extension study" --why="unattended training run" --mode=block)
    fi
    nohup setsid "${INHIBIT[@]}" bash "$0" --here "$@" > "${LOG}" 2>&1 < /dev/null &
    echo "started in the background (pid $!), $([ ${#INHIBIT[@]} -gt 0 ] && echo 'sleep blocked while it runs' || echo 'no sleep inhibitor available: keep the machine awake')"
    echo "log:     ${REPO}/${LOG}"
    echo "status:  ${REPO}/results/final_twins/TWINS_RUN.md, ${REPO}/results/extension/RUN_STATUS.md"
    exit 0
fi
shift
say() { echo "[pipeline] $(date -Is) $*"; }
PY="./.venv/bin/python"
mkdir -p "${LOGDIR}" "${RES}"
T0=$(date -Is)
say "on $(hostname), commit $(git rev-parse --short HEAD)"
if [ ! -x "${PY}" ]; then say "no .venv in ${REPO}: nothing can run"; exit 2; fi

# ---- CPU part of phase 2 in the background: stage 0 and the operator screen
(
    "${PY}" -u examples/synthetic_suite.py --suite extension --no_export stage0 --skip_operator_test
    echo "[cpu] stage0 exit $?"
    "${PY}" -u examples/synthetic_suite.py --suite extension --no_export screen
    echo "[cpu] screen exit $?"
) > "${LOGDIR}/cpu_part.out" 2>&1 &
CPU_PID=$!
say "stage 0 and operator screen started in the background (pid ${CPU_PID}; log ${LOGDIR}/cpu_part.out)"

# ---- phases 0 and 1 on the GPU
say "phases 0 + 1: twins of the final study"
bash tools/run_final_twins_unattended.sh --here; RC_TWINS=$?
say "phases 0 + 1 finished (exit ${RC_TWINS})"

# ---- phase 2 on the GPU once its CPU part is done
wait "${CPU_PID}"; RC_CPU=$?
say "stage 0 + screen finished (exit ${RC_CPU})"
"${PY}" -u examples/synthetic_suite.py --suite extension all --parallel 2; RC_EXT=$?
say "phase 2 finished (exit ${RC_EXT})"

{
    echo "# Extension study, phases 0-2: pipeline record"
    echo
    echo "Machine $(hostname), commit $(git rev-parse --short HEAD); started ${T0}, finished $(date -Is)."
    echo
    echo "| part | exit code | details |"; echo "|---|---|---|"
    echo "| phases 0 + 1 (twins) | ${RC_TWINS} | results/final_twins/TWINS_RUN.md |"
    echo "| phase 2 CPU part (stage 0, screen) | ${RC_CPU} | results/extension/stage0, results/extension/screen/SCREEN.md |"
    echo "| phase 2 (pilots, decision, smoke) | ${RC_EXT} | results/extension/RUN_STATUS.md, DECISION.md |"
    echo
    echo "Exit code 1 of phase 2 means it finished with recorded warnings (see RUN_STATUS.md)."
} > "${RES}/PIPELINE_STATUS.md"
cp "${LOGDIR}/pipeline.out" "${LOGDIR}/cpu_part.out" "${RES}/" 2>/dev/null || true
say "done; ${RES}/PIPELINE_STATUS.md"
exit 0
