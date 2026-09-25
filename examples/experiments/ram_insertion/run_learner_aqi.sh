export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_aqi_rlpd.py \
    --exp_name=ram_insertion \
    --checkpoint_path=aqi_first_run \
    --demo_path=... \
    --learner \
    --aqi_mode=aqi \
    "$@"
