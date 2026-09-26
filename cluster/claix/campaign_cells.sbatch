#!/bin/bash
# Campaign step 2/3: one array task = one cell (train on one H100, then score
# it on the ruler). Task i runs the cell with index i in plan order
# (run_campaign.py --shard i/N). Submitted by submit_campaign.sh.
#
# Re-submitting is safe: finished cells (.done / cell.json) are skipped, and a
# cell whose job died is retrained from scratch.
#SBATCH --job-name=aqi-cells
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00

set -uo pipefail
unset AQI_TRAINER_PORT MUJOCO_EGL_DEVICE_ID
cd "${AQI_REPO_DIR:?submit with submit_campaign.sh}"
# shellcheck disable=SC1091
source cluster/claix/env.sh
nvidia-smi -L
# shellcheck disable=SC2086
python examples/run_campaign.py --config "${AQI_CONFIG}" \
    --campaign "${AQI_CAMPAIGN}" --stages train,eval \
    --shard "${SLURM_ARRAY_TASK_ID}/${AQI_NCELLS:?}" ${AQI_EXTRA:-}
