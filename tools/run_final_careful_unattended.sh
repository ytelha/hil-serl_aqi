#!/usr/bin/env bash
# Careful-operator control of the final study (docs/final_careful_control.md), unattended, in the background:
#   1. copies the final study's operators, demonstrations, ruler (eval_grid.json) and stage-0 limits from runs/final/
#      into runs/final_careful/ and checks them by hash, so both studies train from byte-identical inputs;
#   2. runs F3_careful_hard, then F4_careful_standard (the keepall arm only, careless share 0, 12 cells each,
#      2 side by side as in the final study);
#   3. retrains any crashed cell once (run_campaign retrains exactly the cells without .done);
#   4. report, plots, export to results/final_careful/.
# runs/final/ and results/final/ are only read, never written. It never pushes anything: the repository is public,
# the results are not. It stops (exit 2) before training if an input differs, the disk is short, or the final study's
# folders are missing; it never falls back to keeping only the newest checkpoints (that would lose the budget curves).
#
#   bash tools/run_final_careful_unattended.sh          # start in the background and return
#   tail -f runs/final_careful/unattended.out           # watch it (also copied to results/final_careful/)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
SRC="runs/final"
DST="runs/final_careful"
RES="results/final_careful"
PAIRS=("F1_hard:F3_careful_hard" "F2_standard:F4_careful_standard")
CAMPAIGNS=(F3_careful_hard F4_careful_standard)
if [ "${1:-}" != "--here" ]; then
    mkdir -p "${DST}"
    LOG="${DST}/unattended.out"
    INHIBIT=()
    if command -v systemd-inhibit >/dev/null 2>&1 \
            && systemd-inhibit --what=sleep:idle --who="AQI careful control" --why="probe" true >/dev/null 2>&1; then
        INHIBIT=(systemd-inhibit --what=sleep:idle --who="AQI careful control" --why="unattended training run" --mode=block)
    fi
    nohup setsid "${INHIBIT[@]}" bash "$0" --here "$@" > "${LOG}" 2>&1 < /dev/null &
    echo "started in the background (pid $!), $([ ${#INHIBIT[@]} -gt 0 ] && echo 'sleep blocked while it runs' || echo 'no sleep inhibitor available: keep the machine awake')"
    echo "log:     ${REPO}/${LOG}"
    echo "results: ${REPO}/${RES}/  <- copy this folder for the analysis"
    exit 0
fi
shift
say() { echo "[careful] $(date -Is) $*"; }
stop() { say "STOP before training: $*"; mkdir -p "${RES}"; cp "${DST}/unattended.out" "${RES}/unattended.out" 2>/dev/null; exit 2; }
PY="./.venv/bin/python"
say "on $(hostname), commit $(git rev-parse --short HEAD)"
[ -x "${PY}" ] || stop "no .venv here: run it in the repo that ran the final study"
git diff --quiet HEAD -- examples serl_launcher franka_sim || stop "local changes in the training code (git diff HEAD)"

# ---- 1. inputs from the final study, copied once and checked by hash on every start
[ -f "${SRC}/stage0/limits.json" ] || stop "${SRC}/stage0/limits.json is missing"
mkdir -p "${DST}/stage0"
[ -f "${DST}/stage0/limits.json" ] || cp -p "${SRC}/stage0/limits.json" "${DST}/stage0/limits.json"
cmp -s "${SRC}/stage0/limits.json" "${DST}/stage0/limits.json" || stop "stage-0 limits differ from the final study's"
hashes() { (cd "$1" && find operators demos eval_grid.json -type f -print0 | sort -z | xargs -0 sha256sum); }
for pair in "${PAIRS[@]}"; do
    s="${pair%%:*}"; d="${pair##*:}"
    for item in operators demos eval_grid.json; do
        [ -e "${SRC}/${s}/${item}" ] || stop "${SRC}/${s}/${item} is missing"
    done
    mkdir -p "${DST}/${d}"
    for item in operators demos eval_grid.json; do
        [ -e "${DST}/${d}/${item}" ] || cp -a "${SRC}/${s}/${item}" "${DST}/${d}/${item}"
    done
    n_src=$(hashes "${SRC}/${s}" | wc -l)
    if ! diff <(hashes "${SRC}/${s}") <(hashes "${DST}/${d}") > /dev/null; then
        stop "${DST}/${d}: operators, demonstrations or ruler differ from ${SRC}/${s}"
    fi
    say "${d}: ${n_src} input files identical to ${s} (operators, demonstrations, ruler)"
done

# ---- 2. disk: early checkpoints must stay (budget curves), so no fallback
need=$("${PY}" -c "import sys; sys.path.insert(0, 'examples'); import synthetic_suite as s; s.configure('final'); print(s.disk_needed_gb(sys.argv[1:]))" "${CAMPAIGNS[@]}")
free=$(df -BG --output=avail "${DST}" | tail -1 | tr -dc '0-9')
say "disk: ${free} GB free, about ${need} GB needed"
"${PY}" -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "${free}" "${need}" \
    || stop "only ${free} GB free under ${DST}, about ${need} GB needed; free space first (never by deleting runs/final)"

# ---- 3. train, 2 side by side, W&B as in the final study (online with a key, else offline)
mode=$("${PY}" -c "import sys; sys.path.insert(0, 'examples'); import synthetic_suite as s; print('online' if s.wandb_logged_in() else 'offline')")
SUITE=("${PY}" -u examples/synthetic_suite.py --suite final --root "${DST}" --results "${RES}")
RUN=("${SUITE[@]}" run "${CAMPAIGNS[@]}" --parallel 2 --keep_going --wandb "${mode}")
say "W&B ${mode}; training: ${RUN[*]}"
"${RUN[@]}"; rc_run=$?
crashed() { for c in "${CAMPAIGNS[@]}"; do for f in "${DST}/${c}"/checkpoints/*/.crashed; do
    [ -e "${f}" ] && [ ! -e "$(dirname "${f}")/.done" ] && echo "${c}/$(basename "$(dirname "${f}")")"; done; done; }
left=$(crashed)
rc_retry=-
if [ -n "${left}" ]; then
    say "crashed, retrained once: $(echo ${left})"
    "${RUN[@]}"; rc_retry=$?
    left=$(crashed)
fi

# ---- 4. analysis files and export
"${SUITE[@]}" report "${CAMPAIGNS[@]}"; rc_report=$?
"${SUITE[@]}" plots "${CAMPAIGNS[@]}"; rc_plots=$?
"${SUITE[@]}" export; rc_export=$?
{
    echo "# Careful-operator control: run record"
    echo
    echo "Machine $(hostname), commit $(git rev-parse --short HEAD), finished $(date -Is)."
    echo
    echo "| step | exit code |"; echo "|---|---|"
    echo "| run | ${rc_run} |"; echo "| retry of crashed cells | ${rc_retry} |"
    echo "| report | ${rc_report} |"; echo "| plots | ${rc_plots} |"; echo "| export | ${rc_export} |"
    echo
    echo "| campaign | cells with .done | cells started |"; echo "|---|---|---|"
    for c in "${CAMPAIGNS[@]}"; do
        echo "| ${c} | $(ls -d "${DST}/${c}"/checkpoints/*/.done 2>/dev/null | wc -l) | $(ls -d "${DST}/${c}"/checkpoints/*/ 2>/dev/null | wc -l) |"
    done
    echo
    echo "Still crashed after one retry: ${left:-none}"
} > "${RES}/CAREFUL_RUN.md"
cp "${DST}/unattended.out" "${RES}/unattended.out" 2>/dev/null || true
say "finished; results in ${REPO}/${RES}/ (CAREFUL_RUN.md first)"
[ "${rc_run}" = 0 ] && [ -z "${left}" ]
