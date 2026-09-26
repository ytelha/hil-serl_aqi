# AQI-HIL-SERL environment on the RWTH cluster CLAIX.
#
#   source cluster/claix/env.sh
#
# Source it in every interactive shell and every Slurm job on CLAIX. It:
#   - finds the repo copy on the cluster and the run folder ($AQI_RUNS on
#     $HPCWORK, 1 TB, fast I/O, no backup);
#   - puts headless-only stand-ins for the robot hardware modules (SpaceMouse,
#     RealSense, keyboard listener) first on PYTHONPATH -- the cluster has no
#     USB devices or display, but the experiment configs import them;
#   - gives this job its own learner/actor ports (AQI_TRAINER_PORT), so several
#     jobs can share one 4-GPU node;
#   - picks a MuJoCo EGL device that renders inside this job's GPU allocation;
#   - then sources the repo's own sim_env.sh (venv, MUJOCO_GL=egl, PYTHONPATH).
# Nothing here is used outside the cluster.

# works when sourced from bash or zsh (the CLAIX default login shell)
_AQI_CLAIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export AQI_REPO="$(cd "${_AQI_CLAIX_DIR}/../.." && pwd)"
export AQI_RUNS="${AQI_RUNS:-${HPCWORK:-$HOME}/aqi_runs}"
mkdir -p "${AQI_RUNS}"

# caches off $HOME (quota: 1 million files)
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HPCWORK:-$HOME}/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"

# wandb: local files unless you export WANDB_MODE=online (+ WANDB_API_KEY)
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${AQI_RUNS}/wandb}"

# headless rendering + JAX memory behaviour (sim_env.sh keeps these if set)
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# robot-hardware stand-ins (cluster only; see stubs/README)
export PYTHONPATH="${_AQI_CLAIX_DIR}/stubs${PYTHONPATH:+:${PYTHONPATH}}"

# ---- per-job agentlace ports --------------------------------------------
# learner and actor of one cell talk over TCP ports p, p+1 (default 5588).
# Jobs sharing a node, or users sharing a login node, need different p.
_aqi_free_port_pair() {
    python3 - "$1" <<'PY'
import socket, sys
base = int(sys.argv[1])
for p in range(base, base + 4000, 2):
    free = True
    for q in (p, p + 1):
        s = socket.socket()
        try:
            s.bind(("0.0.0.0", q))
        except OSError:
            free = False
        finally:
            s.close()
    if free:
        print(p)
        break
PY
}
if [ -z "${AQI_TRAINER_PORT:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        _base=$(( 20000 + (SLURM_JOB_ID % 5000) * 4 ))
    else
        _base=$(( 42000 + ($(id -u) % 5000) * 4 ))
    fi
    AQI_TRAINER_PORT="$(_aqi_free_port_pair "${_base}")"
    export AQI_TRAINER_PORT
    unset _base
fi

# ---- the repo's own environment (venv, PYTHONPATH, MUJOCO_GL) ---------------
# shellcheck disable=SC1091
source "${AQI_REPO}/examples/experiments/sim_peg_insertion/sim_env.sh"
# sim_env.sh prepends the repo packages; keep the stand-ins in front of them
export PYTHONPATH="${_AQI_CLAIX_DIR}/stubs:${PYTHONPATH}"

# ---- EGL device inside a GPU job -----------------------------------------------
# MuJoCo renders through EGL; the EGL device index need not match the GPU
# Slurm gave this job. Probe once per job (a few seconds).
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ] \
        && [ -z "${MUJOCO_EGL_DEVICE_ID+x}" ] && [ "${MUJOCO_GL}" = "egl" ]; then
    _probe="$(python "${_AQI_CLAIX_DIR}/egl_probe.py" --bench 2>/dev/null)"
    _dev="$(printf '%s\n' "${_probe}" | sed -n 1p)"
    _spf="$(printf '%s\n' "${_probe}" | sed -n 2p)"
    if [ "${_dev}" = "none" ]; then
        echo "[env.sh] WARNING: no working EGL device found; MuJoCo image" \
             "rendering will fail. Run: python cluster/claix/egl_probe.py -v" >&2
    elif [ -n "${_dev}" ]; then
        export MUJOCO_EGL_DEVICE_ID="${_dev}"
    fi
    # a broken EGL path can pass the one-frame probe and still take ~10 s
    # per frame; the job scripts stop early on such a node (AQI_EGL_SLOW)
    if [ -n "${_spf}" ] && ! awk -v s="${_spf}" 'BEGIN { exit !(s < 0.5) }'; then
        export AQI_EGL_SLOW="${_spf}"
        echo "[env.sh] WARNING: EGL rendering on $(hostname) takes ${_spf} s" \
             "per frame (healthy: ~0.01 s)" >&2
    fi
    unset _probe _dev _spf
fi

echo "[env.sh] repo=${AQI_REPO} runs=${AQI_RUNS} port=${AQI_TRAINER_PORT}" \
     "MUJOCO_GL=${MUJOCO_GL}${MUJOCO_EGL_DEVICE_ID:+ egl_device=${MUJOCO_EGL_DEVICE_ID}}" >&2
