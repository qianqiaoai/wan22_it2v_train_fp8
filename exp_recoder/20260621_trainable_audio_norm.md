# 20260621 Trainable Audio Norm

## Problem

`audio_norm` was not included in `generator_trainable_modules`, so the per-block audio normalization parameters stayed frozen while `audio_proj` and `audio_cross_attn` were trained.

## Change

- Added `audio_norm` to `configs/rf_final.yaml` under `generator_trainable_modules`.

## Expected Effect

- The optimizer will include `model.blocks.*.audio_norm.{weight,bias}`.
- This matches the audio adapter path more closely, allowing the normalization before audio cross attention to adapt with the audio branch.

## Validation

- YAML parse check.
- Field check for `generator_trainable_modules`.

## Risk

- Trainable parameter count increases only slightly.
- Because `audio_norm` gates the audio residual input distribution, it can change audio branch strength; compare with previous runs using the same checkpoint interval.
