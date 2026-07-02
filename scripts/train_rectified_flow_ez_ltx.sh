#!/usr/bin/env bash
set -euo pipefail


cd "$(dirname "$0")/.."

RUN_NAME="wan2.2_5B_fp8_audio_$(date +%Y%m%d_%H%M%S)"
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29600}

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    train.py \
    --config_path configs/rf_final_ltx.yaml \
    --logdir "logs/${RUN_NAME}" \
    --swanlab-logdir swanlab_logdir \
    --swanlab-experiment-name "${RUN_NAME}"