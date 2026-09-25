# HIL-SERL baseline learner for the MuJoCo peg-insertion task.
# Run from this directory:  bash run_learner.sh
# Set DEMO_PATH to the .pkl produced by record_demos.py, e.g.:
#   DEMO_PATH=../../demo_data/cylinder/human/20_demos_<timestamp>.pkl bash run_learner.sh
# Prefer the unified launcher:  SIM_ACTION=learner SIM_MODE=hilserl bash run.sh
source "$(dirname "${BASH_SOURCE[0]}")/sim_env.sh"

DEMO_PATH="${DEMO_PATH:-$(ls -t ../../demo_data/cylinder/human/*demos*.pkl 2>/dev/null | head -1)}"

export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py \
    --exp_name=sim_peg_insertion \
    --checkpoint_path=../../runs/cylinder_hilserl_op0_s42 \
    --demo_path="${DEMO_PATH}" \
    --learner \
    "$@"
