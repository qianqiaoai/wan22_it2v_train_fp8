# Experiment Record: window3_audio_drop0_smoke

Time: 2026-06-19 15:23:36 UTC

## Goal

Check whether a narrower 20fps audio window improves lip-sync diagnosability, and run a minimal smoke test after the config change.

## Changes From Previous Code/Config

File: `configs/rf_final.yaml`

- Changed `audio_condition.window_size` from `5` to `3`.
  - Reason: at 20fps, a 5-frame window spans about 250ms and may blur phoneme timing. A 3-frame window spans about 150ms and should make audio-lip alignment sharper.
- Changed `audio_condition.p_audio_drop` from `0.1` to `0.0`.
  - Reason: temporarily disable audio dropout while debugging lip-sync, so every training batch carries audio conditioning.
- Kept old values commented next to the new values for traceability.

## Expected Impact

- `AudioProjModel` first-frame input changes from `[5, 12, 768]` to `[3, 12, 768]`.
- Latter latent-frame audio window changes from `4 + 5 - 1 = 8` frames to `4 + 3 - 1 = 6` frames.
- Output audio context shape should remain `[latent_frames, 32, 768]`.
- Lip-sync timing should be easier to diagnose because audio context is less temporally smoothed.

## Validation Plan

- Parse `configs/rf_final.yaml` and assert `window_size == 3`, `p_audio_drop == 0.0`.
- Run an `AudioProjModel` shape check for `audio_window=3`.
- Run a 1-step training smoke using a temporary low-resolution config.

## Validation Results

- Config parse passed:
  - `audio_condition.window_size == 3`
  - `audio_condition.p_audio_drop == 0.0`
- `AudioProjModel(audio_window=3, vae_scale=4, output_dim=768)` shape check passed:
  - output shape: `(1, 21, 32, 768)`
- 1-step smoke passed with temporary config `/tmp/rf_final_window3_smoke.yaml`.
  - Command: `timeout 900 torchrun --standalone --nproc_per_node=1 train.py --config_path /tmp/rf_final_window3_smoke.yaml --no_save --disable-swanlab`
  - Temporary overrides: `height=256`, `width=256`, `max_samples=2`, `num_workers=0`, `gradient_accumulation_steps=1`, `max_steps=1`, `gradient_checkpointing=false`, `sharding_strategy=full`, `logdir=/tmp/rf_final_window3_smoke`, `save_at_end=false`, `disable_swanlab=true`.
  - Trainable filter: `audio_proj`, `audio_cross_attn`.
  - Trainable params reported: `763,497,472`.
  - Step result: `loss=1.273438`, `grad_norm=13.7914`, `lr=5.00e-08`, `timestep=933.3[0.0,980.0]`.
  - NCCL destroy completed cleanly.

## Not Validated

- Full-resolution `832x480` training.
- Multi-GPU `hybrid_full` behavior.
- Long-run convergence or lip-sync quality.
- Inference quality after training with `window_size=3`.
