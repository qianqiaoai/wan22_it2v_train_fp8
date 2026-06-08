  CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/debug_fsdp_init.py \
    --config_path configs/rf_final.yaml \
    --mode real-to-bf16