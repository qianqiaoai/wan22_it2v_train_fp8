# 2026-06-22 ABC validation for new audio_norm experiment checkpoints

## Goal

验证新实验权重 `logs/wan2.2_5B_fp8_audio_20260621_210556` 的推理是否真的使用 audio adapter。

本次不改训练、不改模型代码、不改配置，只对当前可见的四个 checkpoint 做同一套 ABC 推理：

- A: normal audio
- B: zero audio
- C: audio shifted by +0.5s

如果 A/B/C 输出几乎一样，说明推理 audio branch 可能没生效，或者 checkpoint 没正确加载 audio 参数。

## Scope

- Checkpoints:
  - `logs/wan2.2_5B_fp8_audio_20260621_210556/checkpoint_model_000500/model.pt`
  - `logs/wan2.2_5B_fp8_audio_20260621_210556/checkpoint_model_001000/model.pt`
  - `logs/wan2.2_5B_fp8_audio_20260621_210556/checkpoint_model_001500/model.pt`
  - `logs/wan2.2_5B_fp8_audio_20260621_210556/checkpoint_model_002000/model.pt`
- Sample manifest:
  - `eval_outputs/train_fit_ckpt_compare_v1/inference/trainfit_20260619_ckpt002500/manifest.jsonl`
- Sample:
  - `generated_0001`
  - `raw_clips_25fps_videovideo4328_udtA5ExkgNA-scene34-scene1`
- Inference params:
  - `sample_steps=50`
  - `cfg_mode=text`
  - `guide_scale=5.0`
  - `shift_sec=0.5`
  - `fps=20`
  - `frame_num=81`
  - `width=480`
  - `height=832`

## Command template

```bash
python /tmp/validate_audio_adapter_abc.py \
  --manifest eval_outputs/train_fit_ckpt_compare_v1/inference/trainfit_20260619_ckpt002500/manifest.jsonl \
  --checkpoint logs/wan2.2_5B_fp8_audio_20260621_210556/checkpoint_model_XXXXXX/model.pt \
  --out-dir exp_recoder/20260622_audio_adapter_abc_new_audio_norm_ckptXXXXXX \
  --base-model-dir /mnt/data/nlp/user/qiaoqian/models/Wan2.2-TI2V-5B \
  --sample-steps 50 \
  --device cuda:0 \
  --dtype bf16 \
  --cfg-mode text \
  --guide-scale 5.0 \
  --shift-sec 0.5
```

## Load checks

四个 checkpoint 均显示：

- `Generator checkpoint load result: <All keys matched successfully>`
- `has_audio_proj: true`
- `has_audio_cross_attn_block0: true`
- `Generator trainable filter enabled: patterns=['audio_proj', 'audio_cross_attn', 'audio_norm']`
- `trainable_tensors=250`
- `trainable_numel=763681792`
- `total_numel=5763469504`

这说明推理侧模型结构里存在 audio adapter，并且 checkpoint key 是完整匹配的。

## Results

| ckpt | audio_proj.proj1 abs mean | block0 audio_cross_attn.o abs mean | last audio_cross_attn.o abs mean | A vs B MAE | A vs C MAE | B vs C MAE |
|---|---:|---:|---:|---:|---:|---:|
| 000500 | 0.0030059814 | 0.0000820160 | 0.0001168251 | 0.03046130 | 0.02947628 | 0.02792961 |
| 001000 | 0.0030059814 | 0.0001401901 | 0.0001859665 | 0.02572322 | 0.03534697 | 0.03124670 |
| 001500 | 0.0030059814 | 0.0001840591 | 0.0002393723 | 0.02747240 | 0.02806473 | 0.03380111 |
| 002000 | 0.0030059814 | 0.0002202988 | 0.0002841949 | 0.03817965 | 0.02815262 | 0.03349509 |

## Output dirs

每个目录都有 6 个 mp4：

- `A_normal.mp4`
- `A_normal_with_used_audio.mp4`
- `B_zero.mp4`
- `B_zero_with_used_audio.mp4`
- `C_shift_plus_0p5s.mp4`
- `C_shift_plus_0p5s_with_used_audio.mp4`

输出目录：

- `exp_recoder/20260622_audio_adapter_abc_new_audio_norm_ckpt000500`
- `exp_recoder/20260622_audio_adapter_abc_new_audio_norm_ckpt001000`
- `exp_recoder/20260622_audio_adapter_abc_new_audio_norm_ckpt001500`
- `exp_recoder/20260622_audio_adapter_abc_new_audio_norm_ckpt002000`

每个目录还包含对应的 `*_audio.wav` 和 `manifest.json`。

## Interpretation

1. 新实验权重推理时 audio adapter 是加载上的，不是 missing/unexpected key 问题。
2. A/B/C 的视频 pixel MAE 都明显非零，因此 audio tensor 改变会影响生成结果；不能再按“audio branch 完全没生效”来判断。
3. `audio_cross_attn.o.weight.abs.mean` 从 500 到 2000 step 持续增大，说明 zero-init output branch 正在被训练打开。
4. `audio_proj.proj1.weight.abs.mean` 在这个粗粒度统计下没有变化，不能证明 proj1 完全没学动；它只说明均值统计不敏感。若要确认，需要继续做参数 delta 或逐层 norm delta。
5. 这些 pixel diff 只能证明 audio 影响视频，不能证明嘴形已经对齐。嘴形同步仍需要人工看 `*_with_used_audio.mp4` 或跑 lip-sync metric。

## Not validated

- 未人工逐帧确认嘴形是否跟 normal/shifted audio 对齐。
- 未跑 SyncNet/AV-HuBERT 等客观 lip-sync 指标。
- 未比较同一 checkpoint 在不同 sample 上的稳定性。
- 未做 `audio_proj.proj1/proj2/proj3` 和 `audio_norm` 的真实参数 delta 汇总。
