export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_aqi_rlpd.py \
    --exp_name=ram_insertion \
    --checkpoint_path=aqi_first_run \
    --actor \
    --aqi_mode=aqi \
    --operator_id=op0 \
    "$@"
