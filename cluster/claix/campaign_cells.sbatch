#!/bin/bash
# Campaign step 2/3: one array task = one cell (train on one H100, then score
# it on the ruler). Task i runs the cell with index i in plan order
# (run_campaign.py --shard i/N). Submitted by submit_campaign.sh as
#   sbatch --array=0-(N-1) campaign_cells.sbatch <repo> <config> <campaign> <N> [args...]
# (CLAIX does not pass the submitting shell's environment into jobs.)
#
# Re-submitting is safe: finished cells (.done / cell.json) are skipped, and a
# cell whose job died is retrained from scratch.
#SBATCH --job-name=aqi-cells
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00

set -uo pipefail
[ $# -ge 4 ] || { echo "usage: sbatch --array=... campaign_cells.sbatch <repo> <config> <campaign> <ncells> [args]"; exit 2; }
AQI_REPO_DIR="$1"; AQI_CONFIG="$2"; AQI_CAMPAIGN="$3"; AQI_NCELLS="$4"; shift 4
unset AQI_TRAINER_PORT MUJOCO_EGL_DEVICE_ID
cd "${AQI_REPO_DIR}" || exit 2
# shellcheck disable=SC1091
source cluster/claix/env.sh
nvidia-smi -L
python examples/run_campaign.py --config "${AQI_CONFIG}" \
    --campaign "${AQI_CAMPAIGN}" --stages train,eval \
    --shard "${SLURM_ARRAY_TASK_ID:?submit as an array}/${AQI_NCELLS}" "$@"
