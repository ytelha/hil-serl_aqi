# AQI-HIL-SERL learner for the MuJoCo peg-insertion task.
# --aqi_mode selects the method: hilserl | logged | active | quality | aqi
# Run from this directory:  bash run_learner_aqi.sh
# Set DEMO_PATH to the .pkl produced by record_demos.py, e.g.:
#   DEMO_PATH=../../demo_data/cylinder/human/20_demos_<timestamp>.pkl bash run_learner_aqi.sh
# Prefer the unified launcher:  SIM_ACTION=learner SIM_MODE=aqi bash run.sh
source "$(dirname "${BASH_SOURCE[0]}")/sim_env.sh"

DEMO_PATH="${DEMO_PATH:-$(ls -t ../../demo_data/cylinder/human/*demos*.pkl 2>/dev/null | head -1)}"

export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_aqi_rlpd.py \
    --exp_name=sim_peg_insertion \
    --checkpoint_path=../../runs/cylinder_aqi_op0_s42 \
    --demo_path="${DEMO_PATH}" \
    --learner \
    --aqi_mode=aqi \
    "$@"
