# Shared environment setup for the MuJoCo sim_peg_insertion task.
# Source this from the run_*.sh scripts (or your shell) before launching.
#
#   source sim_env.sh
#
# It: activates the repo venv, selects headless EGL rendering, puts the
# extracted libhidapi (for the SpaceMouse) on the library path, and adds the
# in-repo packages to PYTHONPATH.

# --- locate the repo root (this file lives in examples/experiments/sim_peg_insertion) ---
_SIM_ENV_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${_SIM_ENV_DIR}/../../.." && pwd )"

# --- python venv ---
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# --- MuJoCo headless rendering (use 'glfw' instead if you have a display) ---
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# --- libhidapi for the 3Dconnexion SpaceMouse ---
# If you installed libhidapi system-wide (apt install libhidapi-hidraw0), you
# can drop this. Otherwise point it at a persistent copy of the library. The
# default below is a user-local copy (survives reboots, no sudo needed); the
# SpaceMouse also needs a udev rule granting access to its hidraw node
# (see docs/sim_walkthrough.md).
HIDAPI_LIB_DIR="${HIDAPI_LIB_DIR:-${HOME}/.local/lib/hidapi}"
if [ -d "${HIDAPI_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${HIDAPI_LIB_DIR}:${LD_LIBRARY_PATH}"
fi

# --- in-repo packages ---
export PYTHONPATH="${REPO_ROOT}/examples:${REPO_ROOT}/franka_sim:${REPO_ROOT}/serl_robot_infra:${REPO_ROOT}/serl_launcher:${PYTHONPATH}"

# --- JAX / XLA memory behaviour ---
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
