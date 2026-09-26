#!/bin/bash
# Campaign step 3/3: the operator-dependency report (report/dependency.md,
# plots, quality audit) over all finished cells. CPU only. Submitted by
# submit_campaign.sh as
#   sbatch campaign_analyze.sbatch <repo> <config> <campaign> [run_campaign args...]
# (CLAIX does not pass the submitting shell's environment into jobs.)
# It runs after every cell job ended, even if some failed; re-run it any time,
# it only reads the campaign folder.
#SBATCH --job-name=aqi-analyze
#SBATCH --partition=c23ms
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -uo pipefail
[ $# -ge 3 ] || { echo "usage: sbatch campaign_analyze.sbatch <repo> <config> <campaign> [args]"; exit 2; }
AQI_REPO_DIR="$1"; AQI_CONFIG="$2"; AQI_CAMPAIGN="$3"; shift 3
unset AQI_TRAINER_PORT MUJOCO_EGL_DEVICE_ID
cd "${AQI_REPO_DIR}" || exit 2
export JAX_PLATFORMS=cpu
# shellcheck disable=SC1091
source cluster/claix/env.sh
python examples/run_campaign.py --config "${AQI_CONFIG}" \
    --campaign "${AQI_CAMPAIGN}" --stages analyze --jax_platform cpu "$@"
echo "report: ${AQI_CAMPAIGN}/report/dependency.md"
