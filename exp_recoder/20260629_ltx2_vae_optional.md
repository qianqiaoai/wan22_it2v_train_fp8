# 2026-06-29 LTX-2 VAE optional path

## Problem
- Current training/inference code hardcoded the Wan2.2 5B VAE path.
- Inference bypassed the wrapper and called `model.vae.model.decode(...)`, which prevents plugging in a VAE with separate encoder/decoder modules.

## Changes
- Added optional `vae_type` selection:
  - `wan5b`: current Wan2.2 5B VAE, default behavior.
  - `ltx2`: LTX-2 video VAE loaded from local LTX-2 repo and LTX-2.3 safetensors.
- Added `LTX2VideoVAEWrapper` with separated `encoder` and `decoder`.
- Added unified `decode_to_pixel()` for Wan2.2 VAE and changed audio I2V inference to use the wrapper decode path.
- Added an early DiT/VAE latent-channel compatibility check before loading the selected VAE.
- Added VAE-specific DiT latent I/O configuration:
  - `wan5b`: keep pretrained `patch_embedding/head.head`, `latent_channels=48`, `patch_size=(1, 2, 2)`, `audio_vae_scale=4`.
  - `ltx2`: rebuild incompatible `patch_embedding/head.head`, `latent_channels=128`, `patch_size=(1, 1, 1)`, `audio_vae_scale=8`.
- Added checkpoint loading filter for LTX mode so old incompatible Wan patch I/O keys are skipped instead of being loaded.
- Added LTX-safe trainer VAE setup; LTX wrapper does not expose `vae.model`.
- Added metadata:
  - `wan5b latent_patch_size = (1, 1, 1)`
  - `ltx2 latent_patch_size = (1, 2, 2)`
  - `ltx2 latent_channels = 128`

## Smoke
- Ran LTX official loader smoke on CPU because CUDA initialization was unavailable in this shell.
- Input: `[1, 3, 1, 32, 32]`
- Encoded latent: `[1, 128, 1, 1, 1]`
- Decoded video: `[1, 3, 1, 32, 32]`
- Ran integrated wrapper smoke with temporal compression:
  - Input: `[1, 3, 9, 64, 64]`
  - Encoded latent: `[1, 2, 128, 2, 2]`
  - Decoded video: `[1, 9, 3, 64, 64]`
  - Output finite check: passed.
- Ran CUDA wrapper smoke on `cuda:0`:
  - LTX2 VAE bf16 input `[1, 3, 9, 64, 64]`
  - Encoded latent: `[1, 2, 128, 2, 2]`, `torch.bfloat16`, `cuda:0`
  - Decoded video: `[1, 9, 3, 64, 64]`, `torch.float32`, `cuda:0`
  - Output finite check: passed, peak memory about `1.575 GB`.
- Ran CUDA wrapper smoke for default Wan2.2 VAE `decode_to_pixel()`:
  - Encoded latent: `[1, 3, 48, 4, 4]`, `torch.bfloat16`, `cuda:0`
  - Decoded video: `[1, 9, 3, 64, 64]`, `torch.float32`, `cuda:0`
  - Output finite check: passed, peak memory about `1.397 GB`.
- Ran CUDA small `WanModel.reset_patch_io()` smoke:
  - Old patch weight: `[64, 48, 1, 2, 2]`
  - New patch weight: `[64, 128, 1, 1, 1]`
  - New head weight: `[128, 64]`
  - Forward output: `[1, 128, 2, 3, 5]`, finite check passed.
- Ran audio profile smoke:
  - `ltx2` profile: `latent_channels=128`, `patch_size=(1, 1, 1)`, `audio_vae_scale=8`.
  - `audio_condition.vae_scale` and `audio_condition.audio_vae_scale` are set to `8`.
- Ran checkpoint filter smoke:
  - Old Wan `patch_embedding.*` and `head.head.*` were skipped.
  - `audio_proj.proj1.weight` was preserved.
- Ran split VAE training-load smoke:
  - `LTX2VideoVAEWrapper(model_name="vae_ckpts")` resolved encoder to `vae_ckpts/ltx2_video_vae_encoder.safetensors`.
  - It resolved decoder to `vae_ckpts/ltx2_video_vae_decoder.safetensors`.
  - Minimal encode/decode passed: `[1, 3, 1, 32, 32] -> [1, 1, 128, 1, 1] -> [1, 1, 3, 32, 32]`.
  - LTX training profile sets `audio_condition.vae_scale=8` and appends `patch_embedding/head.head` to trainable modules.
- Ran full training-load smoke with `/mnt/data/nlp/user/qiaoqian/models/Wan2.2-TI2V-5B` on `cuda:0`:
  - `vae_type=ltx2`.
  - Encoder path: `vae_ckpts/ltx2_video_vae_encoder.safetensors`.
  - Decoder path: `vae_ckpts/ltx2_video_vae_decoder.safetensors`.
  - DiT `patch_size=(1, 1, 1)`, `in_dim=128`, `out_dim=128`.
  - `patch_embedding.weight=[3072, 128, 1, 1, 1]`, bf16, CUDA.
  - `head.head.weight=[128, 3072]`, bf16, CUDA.
  - `audio_vae_scale=8`, `audio_proj.proj1_latter.in_features=92160`.
  - `patch_embedding` and `head.head` are trainable.
- Created `configs/rf_final_ltx.yaml` from `configs/rf_final.yaml`:
  - `generator_name=/mnt/data/nlp/user/qiaoqian/models/Wan2.2-TI2V-5B`.
  - `vae_type=ltx2`.
  - `vae_name=/mnt/data/nlp/user/qiaoqian/newproject/wan2.2_5B_fp8/vae_ckpts`.
  - Uses split encoder/decoder safetensors under `vae_ckpts`.
  - Sets `audio_condition.vae_scale=8`.
  - Explicitly includes `patch_embedding` and `head.head` in `generator_trainable_modules`.

## Notes
- LTX-2 VAE latent channel is 128. A Wan2.2 5B DiT checkpoint whose `in_dim/out_dim` is 48 cannot directly consume this latent without matching DiT input/output layers or a compatible checkpoint.
- LTX-2 VAE temporal downscale is 8, so audio `vae_scale` should be revisited when training with `vae_type: ltx2`.
