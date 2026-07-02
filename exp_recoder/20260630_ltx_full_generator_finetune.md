# 2026-06-30 LTX2 VAE full generator finetuning config

## Problem
- The LTX2 config still used `generator_trainable_modules`, so training was adapter-only instead of full generator finetuning.
- `configs/rf_final_ltx.yaml` also had an invalid YAML scalar `lr:5.0e-5` and a base model path that did not exist in the current environment.

## Changes
- Updated `configs/rf_final_ltx.yaml`.
- Set `generator_name` to `/mnt/data/nlp/user/qiaoqian/models/Wan2.2-TI2V-5B`.
- Commented out `generator_trainable_modules` so all generator parameters remain trainable.
- Kept the old adapter-only trainable list as comments for reference.
- Fixed `lr: 5.0e-5` YAML formatting.

## Validation
- Parsed `configs/rf_final_ltx.yaml` with `OmegaConf.load`.
- Confirmed `generator_trainable_modules` is unset.
- Confirmed audio config:
  - `window_size=5`
  - `p_audio_drop=0.1`
  - `vae_scale=8`
- Ran CUDA model-load smoke on `cuda:0`.
- Confirmed LTX2 profile:
  - `patch_size=(1, 1, 1)`
  - `in_dim=128`
  - `out_dim=128`
  - `audio_vae_scale=8`
  - `audio_proj.proj1_latter.in_features=110592` with current `window_size=5`
- Confirmed full generator finetuning:
  - `trainable_numel=5801009280`
  - `total_numel=5801009280`
  - `trainable_tensors=1135`
  - `total_tensors=1135`
  - `full_trainable=True`
- Ran one CUDA small latent `i2v-5B` loss smoke after fixing patch-size-dependent timestep indexing:
  - Fixed training timestep mask from hardcoded `::2, ::2` to current generator `patch_size`.
  - Fixed inference timestep mask the same way.
  - `patch_size=(1, 1, 1)`.
  - `loss=0.90625`.
  - `flow_pred=[1, 2, 128, 2, 2]`, `torch.bfloat16`.
  - `timestep=[1, 8]`, `torch.bfloat16`.
  - finite check passed.

## Notes
- VAE remains frozen; this change is full generator finetuning, not VAE finetuning.
- Current `lr=5.0e-5` is aggressive for full 5B generator finetuning and should be watched closely for instability.
