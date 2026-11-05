#!/usr/bin/env bash
# Set up a training workstation from scratch (Linux, NVIDIA GPU): the Python environment of the training
# machines, the repo's packages, a runs folder, and the machine check of the comprehensive study.
#
#   bash tools/setup_workstation.sh             # create .venv (or reuse it) and check
#   bash tools/setup_workstation.sh --recreate  # delete .venv first
#
# What it installs: Python 3.12.3 (cluster/claix/python-version) through uv, then the package lock of the training
# machines (cluster/claix/requirements-lock.txt: JAX 0.11.0 with CUDA 12, MuJoCo 3.10.0, wandb 0.30.0, ...) exactly,
# without re-solving. The repo's own packages are not installed: sim_env.sh puts them on PYTHONPATH. Robot-hardware
# drivers (SpaceMouse, RealSense, keyboard) are not installed; small stand-ins from cluster/claix/stubs are added
# behind site-packages, so a real driver wins if it is ever installed.
#
# Needs: an NVIDIA driver recent enough for the GPU (an RTX 5090 needs CUDA 12.8 support), git, curl or pip, and
# internet access. Takes about 10 minutes. Afterwards:
#   ./.venv/bin/wandb login                                         # once, for online W&B logging
#   ./.venv/bin/python examples/synthetic_suite.py --suite study check
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="${REPO}/cluster/claix/requirements-lock.txt"
PYVER="$(tr -d '[:space:]' < "${REPO}/cluster/claix/python-version")"
RECREATE=0
for a in "$@"; do
    case "$a" in
        --recreate) RECREATE=1 ;;
        *) echo "unknown option $a"; exit 2 ;;
    esac
done
say() { printf '\n\033[1;32m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }

say "1/6 GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    warn "nvidia-smi not found: training needs an NVIDIA GPU and driver"
fi

say "2/6 uv"
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        python3 -m pip install --user --upgrade uv
    fi
fi
uv --version

say "3/6 Python ${PYVER} in ${REPO}/.venv"
if [ "${RECREATE}" = 1 ] && [ -d "${REPO}/.venv" ]; then
    rm -rf "${REPO}/.venv"
fi
if [ ! -x "${REPO}/.venv/bin/python" ]; then
    uv venv --python "${PYVER}" --seed "${REPO}/.venv"
fi
"${REPO}/.venv/bin/python" --version

say "4/6 packages from the lock"
REQ="$(mktemp)"
trap 'rm -f "${REQ}"' EXIT
# the lock is the complete installed set of the training machine: install it as-is (--no-deps), without the repo's
# own packages (on PYTHONPATH via sim_env.sh), editable installs and robot-hardware drivers; OpenCV headless
grep -v -E '^\s*(#|$|-e )' "${LOCK}" \
  | grep -v -i -E '^(serl[-_]launcher|franka[-_]sim|serl[-_]robot[-_]infra|franka[-_]env)\b' \
  | grep -v -i -E '@ file://' \
  | grep -v -i -E '^(pyrealsense2|easyhid|pynput|evdev|python-xlib|hidapi|pyspacemouse|rospkg|pymodbus)\b' \
  | sed -E 's/^opencv-python(==|>=| @)/opencv-python-headless\1/; s/^opencv-contrib-python(==|>=| @)/opencv-contrib-python-headless\1/' \
  > "${REQ}"
echo "installing $(grep -c . "${REQ}") requirement lines"
uv pip install --python "${REPO}/.venv/bin/python" --no-deps -r "${REQ}"
SITE="$("${REPO}/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "${REPO}/cluster/claix/stubs" > "${SITE}/aqi_hardware_stubs.pth"
echo "hardware stand-ins behind site-packages: ${SITE}/aqi_hardware_stubs.pth"

say "5/6 runs folder"
mkdir -p "${REPO}/runs"
echo "runs -> ${REPO}/runs (git-ignored)"

say "6/6 imports and GPU"
(
    # shellcheck disable=SC1091
    source "${REPO}/examples/experiments/sim_peg_insertion/sim_env.sh"
    python - <<'PY'
import importlib
bad = []
for m in ["serl_launcher", "jax", "flax", "optax", "distrax", "mujoco", "gymnasium", "agentlace", "cv2", "yaml",
          "wandb", "matplotlib", "experiments.mappings", "serl_launcher.aqi", "serl_launcher.hiril"]:
    try:
        mod = importlib.import_module(m)
        print(f"  ok   {m:24s} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        bad.append(m)
        print(f"  FAIL {m:24s} {type(exc).__name__}: {exc}")
import jax
print("  jax devices:", jax.devices())
if jax.default_backend() != "gpu":
    print("  WARNING: JAX does not see the GPU (driver / CUDA)")
    bad.append("jax-gpu")
raise SystemExit(1 if bad else 0)
PY
)
say "done. Next:"
echo "  ./.venv/bin/wandb login                                          # once"
echo "  ./.venv/bin/python examples/synthetic_suite.py --suite study check"
echo "  ./.venv/bin/python examples/synthetic_suite.py --suite study stage0"
echo "  (then docs/comprehensive_study.md, section 'Run it')"
