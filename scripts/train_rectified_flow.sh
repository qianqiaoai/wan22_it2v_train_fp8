#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-rectified_flow_finetune/configs/rectified_flow_finetune.yaml}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29500}
LOGDIR=${LOGDIR:-logs/rectified_flow_finetune}

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  rectified_flow_finetune/train.py \
  --config_path "${CONFIG_PATH}" \
  --logdir "${LOGDIR}" \
  "$@"
