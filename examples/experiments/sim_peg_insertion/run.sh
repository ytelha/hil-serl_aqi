#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Unified launcher for the MuJoCo `sim_peg_insertion` task.
#
# One entry point for BOTH demo collection and AQI-HIL-SERL training, with every
# choice exposed as an environment variable so it can be driven from the VS Code
# "Run Task" dropdowns (see .vscode/tasks.json) or straight from a shell.
#
# Examples (shell):
#   SIM_ACTION=record  SIM_TASK=cylinder SIM_SUCCESSES=10             bash run.sh
#   # choose the method directly with SIM_MODE (hilserl|logged|active|quality|aqi):
#   SIM_ACTION=learner SIM_MODE=hilserl                              bash run.sh   # baseline
#   SIM_ACTION=learner SIM_MODE=active SIM_TRIGGER_TERMS=uncertainty bash run.sh   # active + custom trigger
#   SIM_ACTION=actor   SIM_MODE=aqi SIM_OPERATOR=op1                 bash run.sh   # combined (full AQI)
#   SIM_ACTION=eval    SIM_MODE=aqi SIM_EVAL_TRAJS=20                bash run.sh
#   # (advanced) instead of SIM_MODE you may set SIM_ACTIVE/SIM_QUALITY/SIM_LOGGING switches.
#
# Variables (all optional, sensible defaults):
#   SIM_ACTION            record | learner | actor | eval | collect-synth | collect-policy
#   SIM_SYN_PROFILE       collect-synth: synthetic operator profile
#                         (expert|intermediate|novice|op1..op4)
#   SIM_SYN_EPISODES      collect-synth: episodes to keep (filtered) / run
#   SIM_SYN_FILTERED      collect-synth: filtered | nofiltered
#   SIM_POLICY_ORACLE_CHECKPOINT  collect-policy: frozen checkpoint dir for
#                         the competent-core policy (required)
#   SIM_POLICY_ORACLE_MODE  collect-policy: checkpoint architecture mode
#                         (hilserl|logged|active|quality|aqi; default hilserl)
#   SIM_POLICY_ORACLE_STEP  collect-policy: checkpoint step (0 = latest)
#   SIM_POLICY_MM_PER_UNIT  collect-policy: policy action scaling to mm/s
#   SIM_TASK              cylinder | hex | cuboid          (insertion geometry)
#   SIM_ACTIVE            on | off                         (active query gate A(s_t))
#   SIM_QUALITY           on | off                         (quality-aware replay+loss)
#   SIM_LOGGING           on | off                         (baseline logging: off=hilserl, on=logged)
#   SIM_MODE              auto | hilserl | logged | active | quality | aqi (auto = derive from switches)
#   SIM_TRIGGER_COMBINER  weighted_sum | max              (active-gate logic)
#   SIM_TRIGGER_TERMS     csv of uncertainty,failure,force,ood,cost
#   SIM_QUALITY_GATE      true | false                    (g(sigma_Q) on/off)
#   SIM_QUALITY_FILTERS   csv of safety_veto,jerk_veto,outcome_filter,unsolicited | none
#   SIM_OPERATOR          operator id tag (actor logs)
#   SIM_SEED              rng seed
#   SIM_SUCCESSES         demos to collect (record)
#   SIM_DEMO_PATH         learner demos .pkl, or "auto" = newest matching file
#   SIM_RENDER            1|0  (record/collect-synth: show the live scene window)
#   SIM_RESUME            continue | scratch  (learner: continue = carry on from
#                         the last checkpoint; scratch = DELETE previous
#                         checkpoints for this task+method and train from zero)
#   SIM_SYNTHETIC_OPERATOR  actor: none (human SpaceMouse) | op1..op4 |
#                         expert|intermediate|novice  (synthetic operator, doc
#                         Part A; replaces the human, no SpaceMouse input needed)
#   SIM_ORACLE_CHECKPOINT   synthetic actor: competent-core policy checkpoint
#                         ('auto' = this run's own checkpoint dir; must share
#                         this run's task+method+K architecture)
#   SIM_SYNTHETIC_SEED      synthetic operator RNG seed (default = SIM_SEED)
#   SIM_SYNTHETIC_INTERVENTION_MODE  actor synthetic-only:
#                         signal | always_latch | latch_until_autonomous_success
#                         (default always_latch)
#   SIM_SYNTHETIC_LATCH_UNTIL_SUCCESS  legacy actor synthetic-only bool kept
#                         for backward compatibility; ignored when the mode
#                         above is set explicitly.
#   SIM_DEBUG             1|0  (train: 1 = --debug / no wandb, 0 = enable wandb)
#   SIM_EVAL_STEP         checkpoint step to evaluate (eval; 'auto' = latest)
#   SIM_EVAL_TRAJS        number of eval episodes (eval)
# ---------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# sourced BEFORE strict mode: sim_env.sh references may-be-unset vars
source "${HERE}/sim_env.sh"
set -eo pipefail

EXAMPLES="$(cd "${HERE}/../.." && pwd)"
cd "${EXAMPLES}"

EXP="sim_peg_insertion"
ACTION="${SIM_ACTION:-record}"
TASK="${SIM_TASK:-cylinder}"
# Method as INDEPENDENT switches (baseline = both gates off). The canonical
# research mode (hilserl|logged|active|quality|aqi) is DERIVED from them below;
# set SIM_MODE explicitly to pin a mode and bypass the derivation.
ACTIVE="${SIM_ACTIVE:-off}"          # active query gate A(s_t)
QUALITY="${SIM_QUALITY:-off}"        # quality-aware replay + loss
LOGGING="${SIM_LOGGING:-off}"        # baseline logging: off=hilserl, on=logged
MODE="${SIM_MODE:-auto}"
TRIGGER_COMBINER="${SIM_TRIGGER_COMBINER:-weighted_sum}"
TRIGGER_TERMS="${SIM_TRIGGER_TERMS:-uncertainty,failure,force,ood,cost}"
QUALITY_GATE="${SIM_QUALITY_GATE:-true}"
QUALITY_FILTERS="${SIM_QUALITY_FILTERS:-safety_veto,jerk_veto,outcome_filter,unsolicited}"
OPERATOR="${SIM_OPERATOR:-op0}"
SEED="${SIM_SEED:-42}"
SUCCESSES="${SIM_SUCCESSES:-20}"
DEMO_PATH="${SIM_DEMO_PATH:-auto}"
RENDER="${SIM_RENDER:-1}"
QUERY_BEHAVIOR="${SIM_QUERY_BEHAVIOR:-continue}"   # actor on a query: continue | pause
ENSEMBLE_SIZE="${SIM_ENSEMBLE_SIZE:-}"             # critics K (empty = config default)
RESUME="${SIM_RESUME:-continue}"                   # learner: continue | scratch
SYNTHETIC_OPERATOR="${SIM_SYNTHETIC_OPERATOR:-none}"  # actor: none=human; op1..op4/expert/intermediate/novice=synthetic
ORACLE_CHECKPOINT="${SIM_ORACLE_CHECKPOINT:-auto}"    # synthetic actor: competent-core ckpt (auto=this run's dir)
SYNTHETIC_SEED="${SIM_SYNTHETIC_SEED:-${SEED}}"       # synthetic operator RNG (R6 determinism)
SYNTHETIC_INTERVENTION_MODE="${SIM_SYNTHETIC_INTERVENTION_MODE:-always_latch}"
SYNTHETIC_LATCH_UNTIL_SUCCESS="${SIM_SYNTHETIC_LATCH_UNTIL_SUCCESS:-0}"
POLICY_ORACLE_CHECKPOINT="${SIM_POLICY_ORACLE_CHECKPOINT:-}"
POLICY_ORACLE_MODE="${SIM_POLICY_ORACLE_MODE:-hilserl}"
POLICY_ORACLE_STEP="${SIM_POLICY_ORACLE_STEP:-0}"
POLICY_MM_PER_UNIT="${SIM_POLICY_MM_PER_UNIT:-100}"
DEBUG="${SIM_DEBUG:-1}"

# JAX device selection. Default: use the GPU when one is present (the learner is
# far too slow on CPU), else fall back to CPU. Override with SIM_JAX_PLATFORM
# (e.g. SIM_JAX_PLATFORM=cpu) or by exporting JAX_PLATFORMS yourself.
if [[ -n "${SIM_JAX_PLATFORM:-}" ]]; then
  export JAX_PLATFORMS="${SIM_JAX_PLATFORM}"
elif [[ -z "${JAX_PLATFORMS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    export JAX_PLATFORMS="cuda"
  else
    export JAX_PLATFORMS="cpu"
  fi
fi

_is_on() { case "${1,,}" in on|true|1|yes|y) return 0 ;; *) return 1 ;; esac; }

# Derive the canonical --aqi_mode from the independent switches unless pinned.
if [[ "${MODE}" == "auto" || -z "${MODE}" ]]; then
  if   _is_on "${ACTIVE}" && _is_on "${QUALITY}"; then MODE="aqi"
  elif _is_on "${ACTIVE}";                        then MODE="active"
  elif _is_on "${QUALITY}";                       then MODE="quality"
  elif _is_on "${LOGGING}";                       then MODE="logged"
  else                                                 MODE="hilserl"
  fi
fi

# AQI composition flags shared by learner + actor. Passing them in every mode is
# safe: hilserl/logged simply ignore the gate/quality settings at runtime.
AQI_FLAGS=(
  --aqi_mode="${MODE}"
  --trigger_combiner="${TRIGGER_COMBINER}"
  --trigger_terms="${TRIGGER_TERMS}"
  --quality_disagreement_gate="${QUALITY_GATE}"
  --quality_filters="${QUALITY_FILTERS}"
  --peg_shape="${TASK}"
)
# Optional ensemble size (K critics); empty keeps the experiment config default.
[[ -n "${ENSEMBLE_SIZE}" ]] && AQI_FLAGS+=(--critic_ensemble_size="${ENSEMBLE_SIZE}")
DEBUG_FLAG=(); [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" ]] && DEBUG_FLAG=(--debug)
CKPT="runs/${TASK}_${MODE}_${OPERATOR}_s${SEED}"

echo "== sim_peg_insertion launcher =="
echo "   action=${ACTION}  task=${TASK}  method=${MODE}  jax=${JAX_PLATFORMS}"
[[ "${ACTION}" != "record" ]] && \
  echo "   trigger=${TRIGGER_COMBINER}[${TRIGGER_TERMS}]  quality_gate=${QUALITY_GATE}  filters=${QUALITY_FILTERS}"

case "${ACTION}" in
  record)
    RENDER_FLAG=(); [[ "${RENDER}" == "1" || "${RENDER}" == "true" ]] && RENDER_FLAG=(--render)
    RECORD_FILTER_FLAG=(--filtered)
    [[ "${SIM_RECORD_FILTERED:-filtered}" == "nofiltered" ]] && RECORD_FILTER_FLAG=(--nofiltered)
    set -x
    exec python record_demos.py --exp_name="${EXP}" --peg_shape="${TASK}" \
        --successes_needed="${SUCCESSES}" "${RECORD_FILTER_FLAG[@]}" "${RENDER_FLAG[@]}"
    ;;

  learner)
    if [[ "${DEMO_PATH}" == "auto" || -z "${DEMO_PATH}" ]]; then
      # structured layout: human > synthetic > policy, matching task shape
      DEMO_PATH="$(ls -t "${EXAMPLES}"/demo_data/${TASK}/human/*demos*.pkl 2>/dev/null | head -1 || true)"
      [[ -z "${DEMO_PATH}" ]] && \
        DEMO_PATH="$(ls -t "${EXAMPLES}"/demo_data/${TASK}/synthetic/*demos*.pkl 2>/dev/null | head -1 || true)"
      [[ -z "${DEMO_PATH}" ]] && \
        DEMO_PATH="$(ls -t "${EXAMPLES}"/demo_data/${TASK}/policy/*demos*.pkl 2>/dev/null | head -1 || true)"
      # legacy flat fallback for pre-existing files
      [[ -z "${DEMO_PATH}" ]] && \
        DEMO_PATH="$(ls -t "${EXAMPLES}"/demo_data/${EXP}_${TASK}_*demos*.pkl 2>/dev/null | head -1 || true)"
      [[ -z "${DEMO_PATH}" ]] && \
        DEMO_PATH="$(ls -t "${EXAMPLES}"/demo_data/${EXP}_*demos*.pkl 2>/dev/null | head -1 || true)"
    fi
    if [[ -z "${DEMO_PATH}" ]]; then
      echo "ERROR: no demo .pkl found in demo_data/ for ${EXP} (task=${TASK})." >&2
      echo "       Record demos first (SIM_ACTION=record) or set SIM_DEMO_PATH." >&2
      exit 1
    fi
    echo "   demos=${DEMO_PATH}  seed=${SEED}  ckpt=${CKPT}"
    # GPU memory cap (fraction of total VRAM; XLA_PYTHON_CLIENT_PREALLOCATE=false
    # so this is a ceiling, not a reservation). Defaults are tuned so the learner
    # and actor can share one ~8 GB GPU; raise SIM_GPU_MEM_LEARNER on a bigger card.
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${SIM_GPU_MEM_LEARNER:-0.6}"
    set -x
    exec python train_aqi_rlpd.py --exp_name="${EXP}" --learner \
        --demo_path="${DEMO_PATH}" --checkpoint_path="${CKPT}" \
        --resume="${RESUME}" \
        --seed="${SEED}" "${AQI_FLAGS[@]}" "${DEBUG_FLAG[@]}"
    ;;

  actor)
    # Optional synthetic operator (methodology doc Part A): replaces the human
    # SpaceMouse. The sim env attaches a FULL geometric operator (execution +
    # monitoring), so NO oracle checkpoint is needed -- execution skill (motor
    # noise, bias, recovery) shapes the correction and monitoring skill shapes
    # the takeover timing. Pass SIM_ORACLE_CHECKPOINT only to force the
    # policy-oracle variant (option B) instead of the geometric one.
    SYN_FLAGS=()
    if [[ -n "${SYNTHETIC_OPERATOR}" && "${SYNTHETIC_OPERATOR}" != "none" ]]; then
      SYN_FLAGS=(--synthetic_operator="${SYNTHETIC_OPERATOR}"
                 --synthetic_seed="${SYNTHETIC_SEED}"
                 --synthetic_intervention_mode="${SYNTHETIC_INTERVENTION_MODE}")
      # Legacy compatibility path: only used if callers still pass the old
      # bool and leave the new mode unset.
      if [[ -z "${SIM_SYNTHETIC_INTERVENTION_MODE:-}" ]] && [[ "${SYNTHETIC_LATCH_UNTIL_SUCCESS}" == "1" || "${SYNTHETIC_LATCH_UNTIL_SUCCESS}" == "true" ]]; then
        SYN_FLAGS+=(--synthetic_latch_until_success)
      fi
      if [[ -n "${ORACLE_CHECKPOINT}" && "${ORACLE_CHECKPOINT}" != "auto" \
            && "${ORACLE_CHECKPOINT}" != "none" ]]; then
        SYN_FLAGS+=(--oracle_checkpoint="${ORACLE_CHECKPOINT}")
        echo "   SYNTHETIC operator=${SYNTHETIC_OPERATOR} seed=${SYNTHETIC_SEED} mode=${SYNTHETIC_INTERVENTION_MODE} (policy oracle=${ORACLE_CHECKPOINT})  (no SpaceMouse)"
      else
        echo "   SYNTHETIC operator=${SYNTHETIC_OPERATOR} seed=${SYNTHETIC_SEED} mode=${SYNTHETIC_INTERVENTION_MODE} (geometric oracle; no checkpoint needed)  (no SpaceMouse)"
      fi
    else
      echo "   operator=${OPERATOR}  seed=${SEED}  ckpt=${CKPT}  query_behavior=${QUERY_BEHAVIOR}  (needs a SpaceMouse)"
    fi
    RENDER_FLAG=(); [[ "${RENDER}" == "1" || "${RENDER}" == "true" ]] && RENDER_FLAG=(--render)
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${SIM_GPU_MEM_ACTOR:-0.2}"
    set -x
    exec python train_aqi_rlpd.py --exp_name="${EXP}" --actor \
        --checkpoint_path="${CKPT}" --operator_id="${OPERATOR}" \
        --query_behavior="${QUERY_BEHAVIOR}" \
        --seed="${SEED}" "${AQI_FLAGS[@]}" "${SYN_FLAGS[@]}" "${RENDER_FLAG[@]}" "${DEBUG_FLAG[@]}"
    ;;

  eval)
    STEP="${SIM_EVAL_STEP:-auto}"; NTRAJ="${SIM_EVAL_TRAJS:-10}"
    if [[ "${STEP}" == "auto" || -z "${STEP}" || "${STEP}" == "0" ]]; then
      STEP="$(ls -d "${CKPT}"/checkpoint_* 2>/dev/null | sed 's/.*checkpoint_//' | sort -n | tail -1 || true)"
    fi
    if [[ -z "${STEP}" ]]; then
      echo "ERROR: no checkpoints in ${CKPT}/ (train the same task+method first)." >&2
      echo "       Expected ${CKPT}/checkpoint_<step>/ ; set SIM_EVAL_STEP explicitly." >&2
      exit 1
    fi
    echo "   eval ckpt=${CKPT} step=${STEP} n_trajs=${NTRAJ}  (policy runs itself; a SpaceMouse may be attached)"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${SIM_GPU_MEM_ACTOR:-0.2}"
    set -x
    exec python train_aqi_rlpd.py --exp_name="${EXP}" --actor \
        --checkpoint_path="${CKPT}" \
        --eval_checkpoint_step="${STEP}" --eval_n_trajs="${NTRAJ}" \
        --seed="${SEED}" "${AQI_FLAGS[@]}" "${DEBUG_FLAG[@]}"
    ;;

  collect-synth)
    # Role 1 (methodology doc Part A) on the REAL sim task: a SyntheticOperator
    # teleoperates via the geometric PhaseOracle (no SpaceMouse) and records
    # training-format demos. The operator commands translation only (no yaw),
    # so 'cylinder' is the supported geometry.
    PROFILE="${SIM_SYN_PROFILE:-expert}"
    EPISODES="${SIM_SYN_EPISODES:-20}"
    FILTERED_FLAG="--filtered"
    [[ "${SIM_SYN_FILTERED:-filtered}" == "nofiltered" ]] && FILTERED_FLAG="--nofiltered"
    OUT="${SIM_SYN_OUT:-auto}"
    if [[ -z "${OUT}" || "${OUT}" == "auto" ]]; then
      OUT="demo_data/${TASK}/synthetic/${EXP}_${PROFILE}_s${SEED}_demos.pkl"
    fi
    RENDER_FLAG=(); [[ "${RENDER}" == "1" || "${RENDER}" == "true" ]] && RENDER_FLAG=(--render)
    export SIM_SYN_PEG_SHAPE="${TASK}"
    echo "   synthetic demos: profile=${PROFILE} episodes=${EPISODES} ${FILTERED_FLAG} shape=${TASK}"
    echo "   out=${OUT}   (translation-only operator; 'cylinder' recommended)"
    set -x
    exec python collect_synthetic_demos.py \
        --env="experiments.${EXP}.wrapper:make_synthetic_sim_env" \
        --profile="${PROFILE}" --episodes="${EPISODES}" ${FILTERED_FLAG} \
      --seed="${SEED}" --operator_id="${OPERATOR}" --out="${OUT}" \
      "${RENDER_FLAG[@]}"
    ;;

  collect-policy)
    # Role 1 variant: collect demos from a SyntheticOperator whose competent
    # core is a frozen policy checkpoint (PolicyOracle, doc S4 option B).
    PROFILE="${SIM_SYN_PROFILE:-expert}"
    EPISODES="${SIM_SYN_EPISODES:-20}"
    FILTERED_FLAG="--filtered"
    [[ "${SIM_SYN_FILTERED:-filtered}" == "nofiltered" ]] && FILTERED_FLAG="--nofiltered"
    OUT="${SIM_SYN_OUT:-auto}"
    if [[ -z "${OUT}" || "${OUT}" == "auto" ]]; then
      OUT="demo_data/${TASK}/policy/${EXP}_${PROFILE}_s${SEED}_demos.pkl"
    fi
    RENDER_FLAG=(); [[ "${RENDER}" == "1" || "${RENDER}" == "true" ]] && RENDER_FLAG=(--render)
    if [[ -z "${POLICY_ORACLE_CHECKPOINT}" ]]; then
      echo "ERROR: collect-policy requires SIM_POLICY_ORACLE_CHECKPOINT." >&2
      echo "       Example: SIM_POLICY_ORACLE_CHECKPOINT=./run_cylinder_hilserl" >&2
      exit 1
    fi
    echo "   policy-oracle synthetic demos: profile=${PROFILE} episodes=${EPISODES} ${FILTERED_FLAG} shape=${TASK}"
    echo "   oracle_ckpt=${POLICY_ORACLE_CHECKPOINT} mode=${POLICY_ORACLE_MODE} step=${POLICY_ORACLE_STEP} mm_per_unit=${POLICY_MM_PER_UNIT}"
    echo "   out=${OUT}"
    set -x
    exec python collect_policy_synthetic_demos.py \
      --exp_name="${EXP}" --checkpoint_path="${POLICY_ORACLE_CHECKPOINT}" \
      --aqi_mode="${POLICY_ORACLE_MODE}" --checkpoint_step="${POLICY_ORACLE_STEP}" \
      --profile="${PROFILE}" --episodes="${EPISODES}" ${FILTERED_FLAG} \
      --seed="${SEED}" --operator_id="${OPERATOR}" \
      --policy_env_mm_per_unit="${POLICY_MM_PER_UNIT}" --out="${OUT}" \
      "${RENDER_FLAG[@]}"
    ;;

  *)
    echo "Unknown SIM_ACTION='${ACTION}' (expected: record | learner | actor | eval | collect-synth | collect-policy)" >&2
    exit 2
    ;;
esac
