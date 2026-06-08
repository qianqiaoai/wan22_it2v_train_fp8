python -u -m torch.distributed.run \
    --nproc_per_node 8 \
    --nnodes 4 \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    train.py \
    --config_path configs/rf_final.yaml \
    --logdir "logs/${RUN_NAME}" \
    --swanlab-logdir swanlab_logdir \
    --swanlab-experiment-name "${RUN_NAME}"