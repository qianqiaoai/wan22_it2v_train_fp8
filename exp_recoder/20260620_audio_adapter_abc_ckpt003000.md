# Experiment Record: audio_adapter_abc_ckpt003000

Time: 2026-06-20 UTC

## Goal

Verify whether inference with the latest 3000-step checkpoint actually loads and uses the audio adapter.

## Input

- Input manifest: `eval_outputs/train_fit_ckpt_compare_v1/inference/trainfit_20260619_ckpt002500/manifest.jsonl`
- Sample used: first row, `generated_0001`
- Checkpoint: `logs/wan2.2_5B_fp8_audio_20260619_233623/checkpoint_model_003000/model.pt`
- Base model: `/mnt/data/nlp/user/qiaoqian/models/Wan2.2-TI2V-5B`
- Inference setting: `sample_steps=50`, `cfg_mode=text`, `guide_scale=5.0`, `seed=42`

## Cases

- A: normal audio embedding.
- B: all-zero audio embedding.
- C: audio embedding shifted by `+0.5s` (`10` frames at `20fps`).

For `cfg_mode=text`, the same audio variant is used for both conditional and unconditional branches, so text CFG does not directly amplify audio differences.

## Command

```bash
python /tmp/validate_audio_adapter_abc.py \
  --manifest eval_outputs/train_fit_ckpt_compare_v1/inference/trainfit_20260619_ckpt002500/manifest.jsonl \
  --checkpoint logs/wan2.2_5B_fp8_audio_20260619_233623/checkpoint_model_003000/model.pt \
  --out-dir exp_recoder/20260620_audio_adapter_abc_ckpt003000 \
  --base-model-dir /mnt/data/nlp/user/qiaoqian/models/Wan2.2-TI2V-5B \
  --sample-steps 50 \
  --device cuda:0 \
  --dtype bf16 \
  --cfg-mode text \
  --guide-scale 5.0 \
  --shift-sec 0.5
```

## Load Result

- `load_state_dict(strict=True)` result: `<All keys matched successfully>`
- `audio_proj` exists: `true`
- `blocks[0].audio_cross_attn` exists: `true`

Audio adapter parameter stats after checkpoint load:

- `audio_proj.proj1.weight.abs().mean() = 0.0030059814453125`
- `blocks[0].audio_cross_attn.o.weight.abs().mean() = 0.00028228759765625`
- `blocks[-1].audio_cross_attn.o.weight.abs().mean() = 0.0003662109375`

These are non-zero, so the checkpoint contains trained audio adapter parameters and inference loaded them.

## Outputs

Output directory:

`exp_recoder/20260620_audio_adapter_abc_ckpt003000`

Generated files:

- `A_normal_with_used_audio.mp4`
- `B_zero_with_used_audio.mp4`
- `C_shift_plus_0p5s_with_used_audio.mp4`
- Raw video-only files: `A_normal.mp4`, `B_zero.mp4`, `C_shift_plus_0p5s.mp4`
- Used audio files: `A_normal_audio.wav`, `B_zero_audio.wav`, `C_shift_plus_0p5s_audio.wav`
- Full manifest: `manifest.json`

## Numeric Video Differences

Whole-video pixel-space differences after decode:

- `A_normal_vs_B_zero_mae = 0.0563310869038105`
- `A_normal_vs_B_zero_max = 2.0`
- `A_normal_vs_C_shift_plus_0p5s_mae = 0.04409413784742355`
- `A_normal_vs_C_shift_plus_0p5s_max = 1.95703125`
- `B_zero_vs_C_shift_plus_0p5s_mae = 0.04929758608341217`
- `B_zero_vs_C_shift_plus_0p5s_max = 1.9609375`

## Conclusion

The audio adapter is present, loaded with matching keys, has non-zero trained parameters, and changing the audio input changes the generated video numerically.

Manual mouth-region inspection is still needed to judge whether the audio effect is useful lip-sync or mostly global/visual drift.
