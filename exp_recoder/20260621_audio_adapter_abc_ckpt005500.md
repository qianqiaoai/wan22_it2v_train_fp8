# 20260621 audio adapter ABC validation ckpt005500

## Goal

验证 `checkpoint_model_005500` 推理时 audio adapter 是否真实加载并影响生成结果。

## Scope

- 不修改训练代码。
- 不修改配置文件。
- 只运行 A/B/C 三路推理：
  - A: normal audio
  - B: zero audio
  - C: audio shifted by +0.5s

## Checkpoint

`logs/wan2.2_5B_fp8_audio_20260619_233623/checkpoint_model_005500/model.pt`

## Command

```bash
python /tmp/validate_audio_adapter_abc.py \
  --manifest eval_outputs/train_fit_ckpt_compare_v1/inference/trainfit_20260619_ckpt002500/manifest.jsonl \
  --checkpoint logs/wan2.2_5B_fp8_audio_20260619_233623/checkpoint_model_005500/model.pt \
  --out-dir exp_recoder/20260621_audio_adapter_abc_ckpt005500 \
  --base-model-dir /mnt/data/nlp/user/qiaoqian/models/Wan2.2-TI2V-5B \
  --sample-steps 50 \
  --device cuda:0 \
  --dtype bf16 \
  --cfg-mode text \
  --guide-scale 5.0 \
  --shift-sec 0.5
```

## Load Result

`load_state_dict(strict=True)` succeeded with:

```text
<All keys matched successfully>
```

Adapter stats after loading:

```json
{
  "has_audio_proj": true,
  "has_audio_cross_attn_block0": true,
  "audio_proj.proj1.weight.abs.mean": 0.0030059814453125,
  "blocks.0.audio_cross_attn.o.weight.abs.mean": 0.00041961669921875,
  "blocks.last.audio_cross_attn.o.weight.abs.mean": 0.000537872314453125
}
```

## Outputs

- `exp_recoder/20260621_audio_adapter_abc_ckpt005500/A_normal.mp4`
- `exp_recoder/20260621_audio_adapter_abc_ckpt005500/A_normal_with_used_audio.mp4`
- `exp_recoder/20260621_audio_adapter_abc_ckpt005500/B_zero.mp4`
- `exp_recoder/20260621_audio_adapter_abc_ckpt005500/B_zero_with_used_audio.mp4`
- `exp_recoder/20260621_audio_adapter_abc_ckpt005500/C_shift_plus_0p5s.mp4`
- `exp_recoder/20260621_audio_adapter_abc_ckpt005500/C_shift_plus_0p5s_with_used_audio.mp4`
- `exp_recoder/20260621_audio_adapter_abc_ckpt005500/manifest.json`

## Video Diffs

```json
{
  "A_normal_vs_B_zero_mae": 0.06667102873325348,
  "A_normal_vs_B_zero_max": 2.0,
  "A_normal_vs_C_shift_plus_0p5s_mae": 0.058867063373327255,
  "A_normal_vs_C_shift_plus_0p5s_max": 2.0,
  "B_zero_vs_C_shift_plus_0p5s_mae": 0.055533312261104584,
  "B_zero_vs_C_shift_plus_0p5s_max": 2.0
}
```

## Notes

- `shift_sec=0.5` at 20 fps corresponds to `shift_frames=10`.
- Compared with ckpt003000, audio cross attention output weights are larger, and A/B/C numeric video differences are also larger.
- This confirms inference is loading the audio adapter and audio conditioning changes generated pixels.
- This does not by itself prove lip-sync quality; manual inspection or AV/lip metric is still needed to determine whether the changed pixels correspond to mouth timing.
