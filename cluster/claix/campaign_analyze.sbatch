#!/bin/bash
# Campaign step 3/3: the operator-dependency report (report/dependency.md,
# plots, quality audit) over all finished cells. CPU only. Submitted by
# submit_campaign.sh; runs after every cell job ended (even if some failed).
# Re-run it any time: it only reads the campaign folder.
#SBATCH --job-name=aqi-analyze
#SBATCH --partition=c23ms
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -uo pipefail
unset AQI_TRAINER_PORT MUJOCO_EGL_DEVICE_ID
cd "${AQI_REPO_DIR:?submit with submit_campaign.sh}"
export JAX_PLATFORMS=cpu
# shellcheck disable=SC1091
source cluster/claix/env.sh
# shellcheck disable=SC2086
python examples/run_campaign.py --config "${AQI_CONFIG}" \
    --campaign "${AQI_CAMPAIGN}" --stages analyze --jax_platform cpu ${AQI_EXTRA:-}
echo "report: ${AQI_CAMPAIGN}/report/dependency.md"
