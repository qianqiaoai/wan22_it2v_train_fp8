# 2026-06-22 ABC validation: old 20260619 run vs new audio_norm run

## Goal

对旧实验 `logs/wan2.2_5B_fp8_audio_20260619_233623` 按照新实验已测试的相同步数做 ABC 推理，并和新实验 `logs/wan2.2_5B_fp8_audio_20260621_210556` 对比。

本次不改训练、不改模型代码、不改配置，只验证推理侧 audio adapter 是否生效，以及相同步数下旧/新 run 的 audio adapter 强度和 A/B/C 差异。

## Checkpoints

旧 run:

- `checkpoint_model_000500`
- `checkpoint_model_001000`
- `checkpoint_model_001500`
- `checkpoint_model_002000`

新 run 对照:

- `checkpoint_model_000500`
- `checkpoint_model_001000`
- `checkpoint_model_001500`
- `checkpoint_model_002000`

## Inference settings

```bash
python /tmp/validate_audio_adapter_abc.py \
  --manifest eval_outputs/train_fit_ckpt_compare_v1/inference/trainfit_20260619_ckpt002500/manifest.jsonl \
  --checkpoint logs/wan2.2_5B_fp8_audio_20260619_233623/checkpoint_model_XXXXXX/model.pt \
  --out-dir exp_recoder/20260622_audio_adapter_abc_old_20260619_ckptXXXXXX \
  --base-model-dir /mnt/data/nlp/user/qiaoqian/models/Wan2.2-TI2V-5B \
  --sample-steps 50 \
  --device cuda:0 \
  --dtype bf16 \
  --cfg-mode text \
  --guide-scale 5.0 \
  --shift-sec 0.5
```

Sample:

- `generated_0001`
- `raw_clips_25fps_videovideo4328_udtA5ExkgNA-scene34-scene1`
- `fps=20`
- `frame_num=81`
- `shift_sec=0.5`, equal to `10` frames at 20 fps

## Load checks

旧 run 四个 checkpoint 均显示：

- `Generator checkpoint load result: <All keys matched successfully>`
- `has_audio_proj: true`
- `has_audio_cross_attn_block0: true`
- `Generator trainable filter enabled: patterns=['audio_proj', 'audio_cross_attn']`
- `trainable_tensors=190`
- `trainable_numel=763497472`
- `total_numel=5763469504`

这说明旧 run 推理时 audio adapter 也正确加载了。旧 run 没有把 `audio_norm` 放进 trainable filter；新 run 的 filter 是 `['audio_proj', 'audio_cross_attn', 'audio_norm']`。

## Results

| run | ckpt | audio_proj.proj1 abs mean | block0 audio_cross_attn.o abs mean | last audio_cross_attn.o abs mean | A vs B MAE | A vs C MAE | B vs C MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| old | 000500 | 0.0030059814 | 0.0000801086 | 0.0001163483 | 0.03627332 | 0.03518644 | 0.02457836 |
| old | 001000 | 0.0030059814 | 0.0001382828 | 0.0001869202 | 0.03397008 | 0.03478467 | 0.03447112 |
| old | 001500 | 0.0030059814 | 0.0001821518 | 0.0002384186 | 0.02990004 | 0.02105806 | 0.03383575 |
| old | 002000 | 0.0030059814 | 0.0002183914 | 0.0002822876 | 0.03282856 | 0.02294927 | 0.03200671 |
| new | 000500 | 0.0030059814 | 0.0000820160 | 0.0001168251 | 0.03046130 | 0.02947628 | 0.02792961 |
| new | 001000 | 0.0030059814 | 0.0001401901 | 0.0001859665 | 0.02572322 | 0.03534697 | 0.03124670 |
| new | 001500 | 0.0030059814 | 0.0001840591 | 0.0002393723 | 0.02747240 | 0.02806473 | 0.03380111 |
| new | 002000 | 0.0030059814 | 0.0002202988 | 0.0002841949 | 0.03817965 | 0.02815262 | 0.03349509 |

## Output dirs

旧 run 输出：

- `exp_recoder/20260622_audio_adapter_abc_old_20260619_ckpt000500`
- `exp_recoder/20260622_audio_adapter_abc_old_20260619_ckpt001000`
- `exp_recoder/20260622_audio_adapter_abc_old_20260619_ckpt001500`
- `exp_recoder/20260622_audio_adapter_abc_old_20260619_ckpt002000`

每个目录均有 6 个 mp4 和 1 个 `manifest.json`：

- `A_normal.mp4`
- `A_normal_with_used_audio.mp4`
- `B_zero.mp4`
- `B_zero_with_used_audio.mp4`
- `C_shift_plus_0p5s.mp4`
- `C_shift_plus_0p5s_with_used_audio.mp4`

## Interpretation

1. 旧 run 的 audio adapter 推理加载正常，不是 missing/unexpected key 问题。
2. 旧 run A/B/C 的 pixel MAE 全部非 0，说明 audio 输入改变会影响生成结果。
3. 旧 run 和新 run 的 `audio_cross_attn.o.weight.abs.mean` 非常接近，并且都随步数增大。这说明加入 `audio_norm` 可训练后，前 2000 step 内从这个粗粒度统计看，adapter output branch 强度没有出现明显量级变化。
4. 旧 run 的 `audio_proj.proj1.weight.abs.mean` 在这个统计下也保持不变，和新 run 一样；这个指标不适合判断 proj1 是否微小更新，需要看真实参数 delta。
5. A/B/C pixel diff 只能证明 audio 影响视频，不能证明嘴形时序对齐。仍需要人工看 `*_with_used_audio.mp4` 或跑 lip-sync metric。

## Notes

旧 checkpoint 的 `torch.load` 多次卡在磁盘读取阶段，状态为 `D (disk sleep)`。对 `000500/001000/001500/002000` 都观察到过这种现象；中断后重跑通常可以继续完成。最终四个 checkpoint 都完整完成 ABC 推理。

## Not validated

- 未人工确认 old/new 同步数下嘴形是否更好。
- 未跑 SyncNet/AV-HuBERT 等客观 lip-sync 指标。
- 未做 old vs new 的逐参数 delta 对比，尤其是 `audio_norm` 和 `audio_proj.proj1/proj2/proj3`。
