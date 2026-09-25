# AQI-HIL-SERL actor for the MuJoCo peg-insertion task.
# --aqi_mode selects the method: hilserl | logged | active | quality | aqi
# --operator_id tags every logged record for the multi-operator study.
# Requires a physical 3Dconnexion SpaceMouse for human takeovers.
# Run from this directory:  bash run_actor_aqi.sh
# Prefer the unified launcher:  SIM_ACTION=actor SIM_MODE=aqi bash run.sh
source "$(dirname "${BASH_SOURCE[0]}")/sim_env.sh"

export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_aqi_rlpd.py \
    --exp_name=sim_peg_insertion \
    --checkpoint_path=../../runs/cylinder_aqi_op0_s42 \
    --actor \
    --aqi_mode=aqi \
    --operator_id=op0 \
    "$@"
