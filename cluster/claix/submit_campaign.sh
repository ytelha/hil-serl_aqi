#!/bin/bash
# Submit one campaign as three chained Slurm steps:
#   prepare (roster, grid, demos; 1 GPU)
#     -> cells (array, one H100 per cell: train + ruler)
#       -> analyze (CPU; runs even if some cells failed)
#
#   bash cluster/claix/submit_campaign.sh CONFIG NAME [options]
#
#   CONFIG   campaign YAML, e.g. examples/campaigns/rq5_quality_score_ext.yaml
#   NAME     campaign folder under $HPCWORK/aqi_runs (= runs/NAME)
# options:
#   --account P      charge project P (default: your personal monthly quota)
#   --parallel N     at most N cell jobs at once (default 8)
#   --time T         wall time per cell job (default 06:00:00)
#   --cells-only     skip prepare (demos exist); still chains analyze
#   --no-analyze     do not submit the analyze job
#   --extra "ARGS"   extra run_campaign.py arguments for all steps
#   --dry            print what would be submitted, submit nothing
#
# Re-running the same command resumes: finished cells are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
[ $# -ge 2 ] || { sed -n '2,24p' "$0"; exit 2; }
CONFIG="$1"; NAME="$2"; shift 2
ACCOUNT=(); PARALLEL=8; TIME=06:00:00; PREP=1; ANALYZE=1; EXTRA=""; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --account) ACCOUNT=(--account "$2"); shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --cells-only) PREP=0; shift ;;
        --no-analyze) ANALYZE=0; shift ;;
        --extra) EXTRA="$2"; shift 2 ;;
        --dry) DRY=1; shift ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
done
case "${NAME}" in */*|"") echo "NAME must be a plain folder name"; exit 2 ;; esac
cd "${REPO}"
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}"; exit 2; }

RUNS="${AQI_RUNS:-${HPCWORK:-$HOME}/aqi_runs}"
export AQI_REPO_DIR="${REPO}"
export AQI_CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
export AQI_CAMPAIGN="${RUNS}/${NAME}"
export AQI_EXTRA="${EXTRA}"
LOGS="${AQI_CAMPAIGN}/slurm"
mkdir -p "${LOGS}"

# number of cells = number of array tasks (computed without training anything)
# shellcheck disable=SC2086
NCELLS=$( ( unset AQI_TRAINER_PORT; source "${HERE}/env.sh" >/dev/null 2>&1
            JAX_PLATFORMS=cpu python examples/run_campaign.py \
                --config "${AQI_CONFIG}" --campaign "${AQI_CAMPAIGN}" \
                --list_cells ${EXTRA} ) | grep -c . )
[ "${NCELLS}" -gt 0 ] || { echo "no cells found (check the config)"; exit 1; }
export AQI_NCELLS="${NCELLS}"
echo "campaign ${NAME}: ${NCELLS} cells, config ${AQI_CONFIG}"
echo "folder   ${AQI_CAMPAIGN}"

run() {
    if [ "${DRY}" = 1 ]; then
        echo "DRY: sbatch $*" >&2
        echo "DRY$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

DEP=()
if [ "${PREP}" = 1 ]; then
    PREP_ID=$(run "${ACCOUNT[@]}" --job-name="aqi-prep-${NAME}" \
        --output="${LOGS}/prepare_%j.out" "${HERE}/campaign_prepare.sbatch")
    echo "prepare  ${PREP_ID}"
    DEP=(--dependency="afterok:${PREP_ID}")
fi
CELLS_ID=$(run "${ACCOUNT[@]}" "${DEP[@]}" --job-name="aqi-cells-${NAME}" \
    --array="0-$((NCELLS - 1))%${PARALLEL}" --time="${TIME}" \
    --output="${LOGS}/cell_%a_%A.out" "${HERE}/campaign_cells.sbatch")
echo "cells    ${CELLS_ID}  (array 0-$((NCELLS - 1)), at most ${PARALLEL} at once)"
if [ "${ANALYZE}" = 1 ]; then
    AN_ID=$(run "${ACCOUNT[@]}" --dependency="afterany:${CELLS_ID}" \
        --job-name="aqi-analyze-${NAME}" --output="${LOGS}/analyze_%j.out" \
        "${HERE}/campaign_analyze.sbatch")
    echo "analyze  ${AN_ID}"
fi
echo
echo "watch:   squeue --me"
echo "logs:    ${LOGS}/"
echo "cells:   ${AQI_CAMPAIGN}/logs/<cell>.{learner,actor,eval}.log"
echo "report:  ${AQI_CAMPAIGN}/report/dependency.md (after analyze)"
