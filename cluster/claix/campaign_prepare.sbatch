#!/bin/bash
# Campaign step 1/3: operator roster, frozen ruler grid, manifest, and the
# synthetic demonstrations of every (operator, seed). Runs once per campaign,
# before the cell jobs. Submitted by submit_campaign.sh as
#   sbatch campaign_prepare.sbatch <repo> <config> <campaign dir> [run_campaign args...]
# (CLAIX does not pass the submitting shell's environment into jobs.)
#
# Demo collection is CPU-bound MuJoCo, but it renders camera images through
# EGL, which needs the GPU node's NVIDIA driver; hence a GPU job.
#SBATCH --job-name=aqi-prep
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -uo pipefail
[ $# -ge 3 ] || { echo "usage: sbatch campaign_prepare.sbatch <repo> <config> <campaign> [args]"; exit 2; }
AQI_REPO_DIR="$1"; AQI_CONFIG="$2"; AQI_CAMPAIGN="$3"; shift 3
unset AQI_TRAINER_PORT MUJOCO_EGL_DEVICE_ID
cd "${AQI_REPO_DIR}" || exit 2
# shellcheck disable=SC1091
source cluster/claix/env.sh
python examples/run_campaign.py --config "${AQI_CONFIG}" \
    --campaign "${AQI_CAMPAIGN}" --stages sample,demos "$@"
