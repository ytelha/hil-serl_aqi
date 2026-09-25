# HIL-SERL baseline actor for the MuJoCo peg-insertion task.
# Requires a physical 3Dconnexion SpaceMouse for human takeovers.
# Run from this directory:  bash run_actor.sh
# Prefer the unified launcher:  SIM_ACTION=actor SIM_MODE=hilserl bash run.sh
source "$(dirname "${BASH_SOURCE[0]}")/sim_env.sh"

export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py \
    --exp_name=sim_peg_insertion \
    --checkpoint_path=../../runs/cylinder_hilserl_op0_s42 \
    --actor \
    "$@"
