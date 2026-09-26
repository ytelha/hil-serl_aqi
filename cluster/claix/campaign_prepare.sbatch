#!/bin/bash
# Campaign step 1/3: operator roster, frozen ruler grid, manifest, and the
# synthetic demonstrations of every (operator, seed). Runs once per campaign,
# before the cell jobs. Submitted by submit_campaign.sh.
#
# Demo collection is CPU-bound MuJoCo, but it renders camera images through
# EGL, which needs the GPU node's NVIDIA driver; hence a GPU job.
#SBATCH --job-name=aqi-prep
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -uo pipefail
unset AQI_TRAINER_PORT MUJOCO_EGL_DEVICE_ID
cd "${AQI_REPO_DIR:?submit with submit_campaign.sh}"
# shellcheck disable=SC1091
source cluster/claix/env.sh
# shellcheck disable=SC2086
python examples/run_campaign.py --config "${AQI_CONFIG}" \
    --campaign "${AQI_CAMPAIGN}" --stages sample,demos ${AQI_EXTRA:-}
