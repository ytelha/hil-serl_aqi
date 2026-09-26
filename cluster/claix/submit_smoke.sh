#!/bin/bash
# Submit the GPU smoke test (cluster/claix/smoke.sbatch).
#   bash cluster/claix/submit_smoke.sh [--account <project>]
# Output: $HPCWORK/aqi_runs/slurm/smoke_<jobid>.out
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AQI_REPO_DIR="$(cd "${HERE}/../.." && pwd)"
RUNS="${AQI_RUNS:-${HPCWORK:-$HOME}/aqi_runs}"
mkdir -p "${RUNS}/slurm"
ACCOUNT=()
if [ "${1:-}" = "--account" ] && [ -n "${2:-}" ]; then
    ACCOUNT=(--account "$2")
fi
JOB=$(sbatch --parsable "${ACCOUNT[@]}" \
    --output="${RUNS}/slurm/smoke_%j.out" "${HERE}/smoke.sbatch")
echo "submitted smoke test ${JOB}"
echo "  watch:  squeue --me        log: tail -f ${RUNS}/slurm/smoke_${JOB}.out"
