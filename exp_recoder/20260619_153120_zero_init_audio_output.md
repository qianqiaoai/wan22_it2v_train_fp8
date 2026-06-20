# Experiment Record: zero_init_audio_output

Time: 2026-06-19 15:31:20 UTC

## Goal

Make the newly added audio residual branch identity-safe at initialization, to test whether previous visual artifacts came from random audio residual injection.

## Changes From Previous Config

File: `configs/rf_final.yaml`

- Changed `audio_condition.zero_init_audio_output` from `false` to `true`.
  - Reason: make `audio_cross_attn.o` output projection start at zero, so the audio branch initially adds nearly no residual to the pretrained DiT hidden states.
- Kept the previous value commented for traceability.

## Expected Impact

- At training step 0, audio cross-attention output residual should be zero at the final output projection.
- The model should start closer to the original Wan2.2 visual distribution.
- This should reduce early grid/noisy visual artifacts while keeping the InfiniteTalk-style audio token path:
  - `AudioProjModel -> [F, 32, 768]`
  - `kv = Linear(768 -> 2 * hidden_dim)`

## Validation Results

- Config parse passed:
  - `audio_condition.zero_init_audio_output == true`
  - `audio_condition.window_size == 3`
  - `audio_condition.p_audio_drop == 0.0`
- Initialization behavior check passed:
  - A small `WanAttentionBlock` with `zero_init_output=True` had `audio_cross_attn.o.weight == 0` and `audio_cross_attn.o.bias == 0`.

## Not Validated

- No training smoke was run for this config-only change.
- Full-resolution training stability and inference quality remain unvalidated.
