#!/bin/bash
# One-time Python environment for AQI-HIL-SERL on CLAIX. Run on a login node
# (login23-1/-2/-4; they have internet access), from the repo copy on the
# cluster:
#
#   bash cluster/claix/setup_env.sh              # use the lock (recommended)
#   bash cluster/claix/setup_env.sh --recreate   # delete and rebuild .venv
#   bash cluster/claix/setup_env.sh --fallback   # no lock available (see below)
#
# The lock is cluster/claix/requirements-lock.txt = `pip freeze` of the
# training machine's .venv, and cluster/claix/python-version its Python
# version. Using it makes the cluster run the SAME package versions,
# including MuJoCo (physics) and JAX. --fallback installs from the repo's
# requirement files instead; versions (MuJoCo!) may then differ from the
# training machine, so fallback results are for testing only.
#
# What it does:
#   1. installs `uv` (fast pip replacement) to ~/.local/bin if missing;
#   2. creates <repo>/.venv with the pinned Python (uv downloads it; no module needed);
#   3. installs the packages, minus the robot-hardware ones (stand-ins in
#      cluster/claix/stubs) and with opencv-python -> opencv-python-headless;
#   4. links <repo>/runs -> $HPCWORK/aqi_runs and adds VS Code settings;
#   5. checks the imports.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
LOCK="${HERE}/requirements-lock.txt"
FALLBACK_REQ="${HERE}/requirements-fallback.txt"
RECREATE=0
FALLBACK=0
for a in "$@"; do
    case "$a" in
        --recreate) RECREATE=1 ;;
        --fallback) FALLBACK=1 ;;
        *) echo "unknown option $a"; exit 2 ;;
    esac
done

say() { printf '\n\033[1;32m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }

if [ -n "${SLURM_JOB_ID:-}" ]; then
    warn "running inside a Slurm job; compute nodes may lack internet access."
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HPCWORK:-$HOME}/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}"
export UV_LINK_MODE=copy          # cache and venv are on different file systems

# ---- 1. uv ------------------------------------------------------------------
say "1/5 uv"
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user --upgrade uv
fi
uv --version

# ---- 2. venv ----------------------------------------------------------------
say "2/5 virtual environment ${REPO}/.venv"
if [ -f "${HERE}/python-version" ]; then
    PYVER="$(tr -d '[:space:]' < "${HERE}/python-version")"
else
    PYVER="${AQI_PYTHON_VERSION:-3.12}"
    if [ "${FALLBACK}" = 0 ]; then
        warn "no ${HERE}/python-version; using Python ${PYVER}"
    fi
fi
if [ "${RECREATE}" = 1 ] && [ -d "${REPO}/.venv" ]; then
    rm -rf "${REPO}/.venv"
fi
if [ ! -x "${REPO}/.venv/bin/python" ]; then
    uv venv --python "${PYVER}" --seed "${REPO}/.venv"
fi
"${REPO}/.venv/bin/python" --version

# ---- 3. packages -------------------------------------------------------------
say "3/5 packages"
REQ="$(mktemp)"
trap 'rm -f "${REQ}"' EXIT
NODEPS=()
if [ "${FALLBACK}" = 1 ]; then
    cp "${FALLBACK_REQ}" "${REQ}"
    warn "fallback requirements: versions may differ from the training machine"
elif [ -f "${LOCK}" ]; then
    # The lock is the COMPLETE installed set of the training machine, so it
    # is installed as-is without re-solving (--no-deps): pip there accepted
    # combinations a strict resolver rejects (dm-robotics-transformations
    # 0.9.0 declares numpy<2 but runs with numpy 2.1.3).
    NODEPS=(--no-deps)
    # drop: the repo's own packages (on PYTHONPATH via sim_env.sh), editable
    # installs, and robot-hardware packages (headless stand-ins); swap OpenCV
    # for its headless build (no libGL on compute nodes).
    grep -v -E '^\s*(#|$|-e )' "${LOCK}" \
      | grep -v -i -E '^(serl[-_]launcher|franka[-_]sim|serl[-_]robot[-_]infra|franka[-_]env)\b' \
      | grep -v -i -E '@ file://' \
      | grep -v -i -E '^(pyrealsense2|easyhid|pynput|evdev|python-xlib|hidapi|pyspacemouse|rospkg|pymodbus)\b' \
      | sed -E 's/^opencv-python(==|>=| @)/opencv-python-headless\1/; s/^opencv-contrib-python(==|>=| @)/opencv-contrib-python-headless\1/' \
      > "${REQ}"
else
    echo "No lock file at ${LOCK}."
    echo "Create it on the training machine (repo root):"
    echo "  ./.venv/bin/python -m pip freeze > requirements-lock.txt"
    echo "  ./.venv/bin/python -c 'import platform; print(platform.python_version())' > python-version"
    echo "copy both into cluster/claix/ here, and re-run. Or use --fallback."
    exit 1
fi
echo "installing $(grep -c . "${REQ}") requirement lines"
uv pip install --python "${REPO}/.venv/bin/python" "${NODEPS[@]}" -r "${REQ}"

# ---- 4. runs folder + VS Code settings ---------------------------------------
say "4/5 runs folder and VS Code settings"
RUNS="${HPCWORK:-$HOME}/aqi_runs"
mkdir -p "${RUNS}"
if [ ! -e "${REPO}/runs" ]; then
    ln -s "${RUNS}" "${REPO}/runs"
    echo "runs -> ${RUNS}"
elif [ -L "${REPO}/runs" ]; then
    echo "runs -> $(readlink "${REPO}/runs") (kept)"
else
    warn "${REPO}/runs is a real folder; leaving it. New campaigns go to ${RUNS}."
fi
mkdir -p "${REPO}/.vscode"
if [ ! -f "${REPO}/.vscode/settings.json" ]; then
    cp "${HERE}/vscode-settings.json" "${REPO}/.vscode/settings.json"
    echo "wrote .vscode/settings.json"
fi
# keep cluster-only files out of `git status`
EXCL="${REPO}/.git/info/exclude"
if [ -f "${EXCL}" ]; then
    for p in "/runs" "/.venv/" "/.vscode/settings.json"; do
        grep -qxF "${p}" "${EXCL}" || echo "${p}" >> "${EXCL}"
    done
fi

# ---- 5. check -----------------------------------------------------------------
say "5/5 import check (CPU; the GPU is checked by the smoke job)"
(
    # shellcheck disable=SC1091
    source "${HERE}/env.sh"
    JAX_PLATFORMS=cpu python - <<'PY'
import importlib
# serl_launcher first: its __init__ patches JAX for distrax (as in training)
mods = ["jax", "flax", "optax", "serl_launcher", "distrax", "mujoco",
        "gymnasium", "agentlace", "cv2", "yaml", "experiments.mappings",
        "serl_launcher.aqi"]
optional = ["serl_launcher.hiril"]          # only with patch 0004
bad = []
for m in mods + optional:
    try:
        mod = importlib.import_module(m)
        print(f"  ok  {m:24s} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        if m in mods:
            bad.append(m)
        print(f"  {'FAIL' if m in mods else 'skip'} {m:24s} "
              f"{type(exc).__name__}: {exc}")
import jax
print("  jax backend on this node:", jax.default_backend())
raise SystemExit(1 if bad else 0)
PY
)
say "done. Next: bash cluster/claix/submit_smoke.sh   (see cluster/claix/README.md)"
