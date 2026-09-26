#!/bin/bash
# GPU smoke test on one H100 (~30-45 min). Submit with
#   bash cluster/claix/submit_smoke.sh [--account <project>]
#
# Checks, in order:
#   1. the GPU and the JAX CUDA backend;
#   2. MuJoCo EGL rendering inside the job's GPU allocation;
#   3. every entry import;
#   4. the repo tests, including the JAX ones never run on the dev PC;
#   5. one tiny campaign cell end to end: roster, 2 demos, 1500 env steps
#      of learner+actor with a synthetic operator, the ruler.
#SBATCH --job-name=aqi-smoke
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

set -uo pipefail
unset AQI_TRAINER_PORT MUJOCO_EGL_DEVICE_ID      # never inherit the login shell's
cd "${AQI_REPO_DIR:?submit with submit_smoke.sh}"
# shellcheck disable=SC1091
source cluster/claix/env.sh
FAILED=()

step() { printf '\n######## %s ########\n' "$*"; }

step "1. GPU and JAX"
nvidia-smi -L
python - <<'PY' || FAILED+=("jax")
import jax, jax.numpy as jnp
print("jax", jax.__version__, "devices:", jax.devices())
assert jax.default_backend() == "gpu", jax.default_backend()
x = jnp.ones((2048, 2048)); print("matmul ok:", float((x @ x).sum()))
PY

step "2. MuJoCo EGL rendering"
python cluster/claix/egl_probe.py -v || FAILED+=("egl")

step "3. imports"
python - <<'PY' || FAILED+=("imports")
import experiments.mappings as m
print("configs:", sorted(m.CONFIG_MAPPING))
cfg = m.CONFIG_MAPPING["sim_peg_insertion"]()
import os; os.environ["SIM_DISABLE_SPACEMOUSE"] = "1"
env = cfg.get_environment(fake_env=False, save_video=False, classifier=True)
obs, _ = env.reset()
print("sim obs keys:", sorted(obs), {k: getattr(v, "shape", None) for k, v in obs.items()})
PY

step "4. repo tests"
for t in tests/test_*.py; do
    echo "--- ${t}"
    out="$(python "${t}" 2>&1)"
    rc=$?
    printf '%s\n' "${out}" | tail -n 3
    [ "${rc}" -eq 0 ] || FAILED+=("${t}")
done

step "5. one tiny campaign cell"
python examples/run_campaign.py \
    --config examples/campaigns/fast_test.yaml \
    --campaign "${AQI_RUNS}/smoke_${SLURM_JOB_ID}" \
    --methods hilserl --tiers expert --ops_per_tier 1 --seeds 1 --demos 2 \
    --cell_budget_mode steps --steps 1500 \
    --stages sample,demos,train,eval || FAILED+=("campaign")
ls "${AQI_RUNS}/smoke_${SLURM_JOB_ID}/eval_runs/"*/cell.json 2>/dev/null \
    || FAILED+=("no cell.json")

step "summary"
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "SMOKE TEST PASSED"
else
    echo "SMOKE TEST FAILED: ${FAILED[*]}"
    exit 1
fi
