#!/bin/bash
# Training-free check of the synthetic operator in the MuJoCo task (no images,
# no learner, CPU only): can it insert the cuboid / cylinder on its own, with
# and without rotation control? Runs on the free devel partition. Submit with
#   sbatch cluster/claix/oracle_test.sbatch <repo dir> [episodes]
#SBATCH --job-name=aqi-oracle-test
#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -uo pipefail
AQI_REPO_DIR="${1:-}"
EPISODES="${2:-30}"
[ -d "${AQI_REPO_DIR}" ] || { echo "usage: sbatch oracle_test.sbatch <repo dir> [episodes]"; exit 2; }
cd "${AQI_REPO_DIR}"
# shellcheck disable=SC1091
source cluster/claix/env.sh
OUT="${HPCWORK:-$HOME}/aqi_runs/oracle_test"
mkdir -p "${OUT}"
python examples/oracle_insertion_test.py --episodes "${EPISODES}" \
    --out "${OUT}/oracle_test_${SLURM_JOB_ID:-local}.json"
