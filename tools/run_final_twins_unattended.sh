#!/usr/bin/env bash
# Phases 0 and 1 of the extension study (docs/extension_study.md): twins of the final study, unattended.
#   F3_careful_hard      F1_hard's keepall, careful operators                 (phase 0)
#   F4_careful_standard  F2_standard's keepall, careful operators             (phase 0)
#   F5_bc_standard       F2_standard with imitation: bc, bc-discard, bc-random, bc-perfect, careless operators (phase 1)
#   F6_bc_careful_standard  F2_standard with imitation (bc), careful operators  (phase 1)
# For each twin: copy the final study's operators, demonstrations and ruler (eval_grid.json) from runs/final/<source>/
# and check them by hash (a twin whose inputs differ is skipped, never trained on other inputs), train 2 side by side,
# retrain crashed cells once, then delete the checkpoints the scored runs no longer need (best window, last and newest
# kept). Then report, plots, export to results/final_twins/ and the run record TWINS_RUN.md.
# Never stops for a person: problems are recorded and the next twin starts. Reads runs/final/ only; never pushes.
#
#   bash tools/run_final_twins_unattended.sh          # start in the background and return
#   bash tools/run_extension_unattended.sh            # phases 0-2 in one go (calls this with --here)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
SRC="runs/final"
DST="runs/final_twins"
RES="results/final_twins"
PAIRS=("F1_hard:F3_careful_hard" "F2_standard:F4_careful_standard"
       "F2_standard:F5_bc_standard" "F2_standard:F6_bc_careful_standard")
# phase 0 also ran on its own (commit a290c12, 19-20 Nov) into this root: a twin finished there is not repeated
EARLIER="runs/final_careful"
if [ "${1:-}" != "--here" ]; then
    mkdir -p "${DST}"
    LOG="${DST}/unattended.out"
    INHIBIT=()
    if command -v systemd-inhibit >/dev/null 2>&1 \
            && systemd-inhibit --what=sleep:idle --who="AQI final twins" --why="probe" true >/dev/null 2>&1; then
        INHIBIT=(systemd-inhibit --what=sleep:idle --who="AQI final twins" --why="unattended training run" --mode=block)
    fi
    nohup setsid "${INHIBIT[@]}" bash "$0" --here "$@" > "${LOG}" 2>&1 < /dev/null &
    echo "started in the background (pid $!); log ${REPO}/${LOG}; results ${REPO}/${RES}/"
    exit 0
fi
shift
say() { echo "[twins] $(date -Is) $*"; }
NOTES=()
note() { say "$*"; NOTES+=("$*"); }
PY="./.venv/bin/python"
mkdir -p "${DST}/stage0" "${RES}"
say "on $(hostname), commit $(git rev-parse --short HEAD)"
if [ ! -x "${PY}" ]; then note "no .venv in ${REPO}: nothing can run"; exit 2; fi
git diff --quiet HEAD -- examples serl_launcher franka_sim || note "WARNING: local changes in the training code (git diff HEAD)"

# ---- orientation limits of the final study's stage 0 (the channel limits the final study trained with)
LIMITS_FLAG=()
if [ -f "${SRC}/stage0/limits.json" ]; then
    [ -f "${DST}/stage0/limits.json" ] || cp -p "${SRC}/stage0/limits.json" "${DST}/stage0/limits.json"
    cmp -s "${SRC}/stage0/limits.json" "${DST}/stage0/limits.json" \
        || note "WARNING: ${DST}/stage0/limits.json differs from the final study's"
else
    note "WARNING: ${SRC}/stage0/limits.json missing: shape-default orientation limits"
    LIMITS_FLAG=(--default_limits)
fi

hashes() { (cd "$1" && find operators demos eval_grid.json -type f -print0 | sort -z | xargs -0 sha256sum); }
crashed() { for f in "${DST}/$1"/checkpoints/*/.crashed; do
    [ -e "${f}" ] && [ ! -e "$(dirname "${f}")/.done" ] && echo "$(basename "$(dirname "${f}")")"; done; }
mode=$("${PY}" -c "import sys; sys.path.insert(0, 'examples'); import synthetic_suite as s; print('online' if s.wandb_logged_in() else 'offline')")
SUITE=("${PY}" -u examples/synthetic_suite.py --suite final --root "${DST}" --results "${RES}")
say "W&B ${mode}"
ROWS=()
DONE_CAMPAIGNS=()
for pair in "${PAIRS[@]}"; do
    s="${pair%%:*}"; d="${pair##*:}"
    # ---- 0. finished before in the earlier root: not repeated
    if [ -d "${EARLIER}/${d}/checkpoints" ]; then
        n_e=$(ls -d "${EARLIER}/${d}"/checkpoints/*/.done 2>/dev/null | wc -l)
        n_plan=$("${PY}" -c "import sys; sys.path.insert(0, 'examples'); import synthetic_suite as s; s.configure('final'); print(s.planned_cells(sys.argv[1]))" "${d}")
        if [ "${n_e}" -ge "${n_plan}" ] && [ "${n_plan}" -gt 0 ]; then
            note "${d}: finished before in ${EARLIER} (${n_e} of ${n_plan} runs); not repeated"
            ROWS+=("| ${d} | finished before in ${EARLIER} | ${n_e} / ${n_plan} | - |"); continue
        fi
    fi
    # ---- 1. inputs copied once from the final study and checked by hash on every start
    missing=""
    for item in operators demos eval_grid.json; do [ -e "${SRC}/${s}/${item}" ] || missing="${missing} ${item}"; done
    if [ -n "${missing}" ]; then
        note "SKIPPED ${d}: ${SRC}/${s} lacks${missing}"; ROWS+=("| ${d} | skipped: source lacks${missing} | - | - |"); continue
    fi
    mkdir -p "${DST}/${d}"
    for item in operators demos eval_grid.json; do
        [ -e "${DST}/${d}/${item}" ] || cp -a "${SRC}/${s}/${item}" "${DST}/${d}/${item}"
    done
    if ! diff <(hashes "${SRC}/${s}") <(hashes "${DST}/${d}") > /dev/null; then
        note "SKIPPED ${d}: operators, demonstrations or ruler differ from ${SRC}/${s}"
        ROWS+=("| ${d} | skipped: inputs differ from ${s} | - | - |"); continue
    fi
    say "${d}: $(hashes "${SRC}/${s}" | wc -l) input files identical to ${s}"
    # ---- 2. disk: early checkpoints for the budget curves if there is room, else newest only (recorded)
    EXTRA=()
    need=$("${PY}" -c "import sys; sys.path.insert(0, 'examples'); import synthetic_suite as s; s.configure('final'); print(s.disk_needed_gb([sys.argv[1]]))" "${d}")
    free=$(df -BG --output=avail "${DST}" | tail -1 | tr -dc '0-9')
    if ! "${PY}" -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "${free}" "${need}"; then
        note "WARNING ${d}: only ${free} GB free, about ${need} GB needed: newest checkpoints only (no budget curves)"
        EXTRA=(--extra --checkpoint_keep_every 0)
    fi
    # ---- 3. train, retry crashed cells once, prune the scored checkpoints
    say "training ${d} (${free} GB free, about ${need} GB needed)"
    "${SUITE[@]}" run "${d}" --parallel 2 --keep_going --wandb "${mode}" "${LIMITS_FLAG[@]}" "${EXTRA[@]}"; rc=$?
    left=$(crashed "${d}")
    if [ -n "${left}" ]; then
        say "${d}: crashed, retrained once: $(echo ${left})"
        "${SUITE[@]}" run "${d}" --parallel 2 --keep_going --wandb "${mode}" "${LIMITS_FLAG[@]}" "${EXTRA[@]}"; rc=$?
        left=$(crashed "${d}")
        [ -n "${left}" ] && note "${d}: still crashed after one retry: $(echo ${left})"
    fi
    "${SUITE[@]}" prune "${d}"
    ndone=$(ls -d "${DST}/${d}"/checkpoints/*/.done 2>/dev/null | wc -l)
    nall=$(ls -d "${DST}/${d}"/checkpoints/*/ 2>/dev/null | wc -l)
    ROWS+=("| ${d} | trained (exit ${rc}) | ${ndone} / ${nall} | $(echo ${left:-none}) |")
    DONE_CAMPAIGNS+=("${d}")
done

# ---- 4. analysis files and export
if [ ${#DONE_CAMPAIGNS[@]} -gt 0 ]; then
    "${SUITE[@]}" report "${DONE_CAMPAIGNS[@]}" || note "report: non-zero exit"
    "${SUITE[@]}" plots "${DONE_CAMPAIGNS[@]}" || note "plots: non-zero exit"
fi
"${PY}" -u examples/synthetic_suite.py --suite final --root "${DST}" --results "${RES}" export || note "export: non-zero exit"
{
    echo "# Phases 0 and 1 (twins of the final study): run record"
    echo
    echo "Machine $(hostname), commit $(git rev-parse --short HEAD), finished $(date -Is). Design: docs/extension_study.md."
    echo
    echo "| campaign | status | cells with .done | still crashed |"; echo "|---|---|---|---|"
    for r in "${ROWS[@]}"; do echo "${r}"; done
    echo
    echo "## Notes"
    echo
    if [ ${#NOTES[@]} -eq 0 ]; then echo "- none"; else for n in "${NOTES[@]}"; do echo "- ${n}"; done; fi
} > "${RES}/TWINS_RUN.md"
cp "${DST}/unattended.out" "${RES}/unattended.out" 2>/dev/null || true
say "finished; ${RES}/TWINS_RUN.md"
exit 0
