# Wan2.2 5B Audio I2V Training Analysis

Last updated: 2026-06-08

## Current Goal

当前训练目标是给 Wan2.2 TI2V 5B 增加 audio-conditioned image-to-video 能力，并让推理时可以使用 text CFG 和 audio CFG。

当前主要配置文件：

```text
configs/rf_final.yaml
```

当前训练数据：

```text
raw_dataset/talkvid0918_zhibo_9_16_132h_with_audio_path.jsonl
```

数据条数：

```text
73124
```

## Effective Batch And Epoch

DLC 训练配置：

```text
world_size = 48
batch_size = 1
gradient_accumulation_steps = 4
```

所以每个 optimizer step 实际消耗：

```text
48 * 1 * 4 = 192 samples
```

真实 epoch 计算方式：

```text
true_epoch = optimizer_step * gradient_accumulation_steps / len(dataloader_per_rank)
```

当前 DLC 日志里：

```text
DATASET SIZE 1523
```

因此：

```text
12000 steps = 12000 * 4 / 1523 = 31.52 true epochs
```

注意：日志中的 `total_epochs=max_steps/1523` 没有乘 `gradient_accumulation_steps`，所以它低估了真实数据遍历轮数。

## Dataset Sampling

当前视频长度配置：

```text
num_frames = 81
target_fps = 20
```

每个训练 clip 的视频时长：

```text
81 / 20 = 4.05 seconds
```

`dataset.py` 中对长视频使用随机窗口：

```text
frame_start = random integer in [0, total_frames - num_frames]
```

所以即使真实 epoch 达到 30 多轮，同一个视频样本每轮也可能截取不同 4.05 秒片段，不是完全重复训练。

当前音频特征会跟随视频 `frame_start/frame_end` 对齐读取；如果音频帧不足，会重复最后一帧 pad 到 `num_frames`。这个逻辑很重要，否则 video/audio 时间窗口不一致会导致模型学不到稳定 audio alignment。

## Previous NCCL Timeout At Step 284

之前多次在 step 284 后等待约 600 秒触发 NCCL watchdog：

```text
Watchdog caught collective operation timeout
OpType=_ALLGATHER_BASE / _REDUCE_SCATTER_BASE / ALLREDUCE
```

结合后续验证：

- `log_iters=20` 时 step 20/40/60 都可以保存成功。
- 因此问题不是“48 卡、batch size 1、log_iters 不能整除”导致的。
- 更像是固定数据位置触发了某个 rank 的 forward/backward 不一致，或某个样本处理极慢/异常，最终导致 FSDP collective 顺序不一致或超时。
- 当前重点应继续确保 video/audio/text shape 对齐，坏样本被提前扫描或在 dataset 阶段明确报错。

## Checkpoint Saving

当前 checkpoint 保存的是：

```text
step
generator
```

保存路径类似：

```text
logs/.../checkpoint_model_008000/model.pt
```

当前不是 sharded checkpoint，而是将 FSDP state dict gather 后保存为 rank0 单文件。它能保存成功，但大模型下会带来：

- 保存慢。
- rank0 内存/显存压力更高。
- 多卡 collective 更重。

更稳的长期方案是：

```text
SHARDED_STATE_DICT + torch.distributed.checkpoint
```

也就是每个 rank 保存自己负责的 shard，不再 gather 成一个 rank0 `model.pt`。

## Resume Status

已验证 `checkpoint_model_008000/model.pt` 可以读取：

```text
step = 8000
generator tensors = 1195
```

本地 2 卡测试中：

```text
Generator checkpoint loaded successfully
Dataloader resumed
```

所以当前 resume 至少能恢复：

- generator 权重
- step
- dataloader skip 位置

但当前不是完整断点续训，因为没有保存/恢复：

- optimizer state
- scheduler state
- RNG state
- sampler state

影响：

- 可以继续训练已有权重。
- AdamW 的历史二阶矩会重置。
- 不能保证与 uninterrupted training 完全一致。

如果后续要严格续训，需要把 optimizer/scheduler/RNG 也纳入 checkpoint。FSDP 下建议和 sharded checkpoint 一起做。

## Audio Drop And CFG

旧 run：

```text
run-20260606_120131-ja2l9jtlq3lwchfnfuads
```

配置中：

```text
audio_condition.drop_prob = 0.0
```

所以旧的 0-8500 checkpoints 没有训练 audio-uncondition 分支，不能可靠使用 audio CFG。

当前 `configs/rf_final.yaml` 已加入：

```text
audio_condition.p_audio_drop = 0.1
```

含义是按 batch/sample 维度独立 drop audio condition：

```text
batch_size=1, world_size=48, grad_accum=4
effective samples per step = 192
p_audio_drop=0.1
```

平均每个 optimizer step 约有：

```text
192 * 0.1 = 19.2
```

个样本使用 zero audio condition。这样训练出来的新 checkpoint 才更适合 audio CFG。

## Caption Embedding Mode

当前支持三种 caption embedding 使用方式：

```text
fixed
fallback
row
```

推荐当前阶段使用：

```text
caption_emb_mode: fixed
```

含义是所有样本都使用统一预生成 prompt embedding：

```text
raw_dataset/text_emb/person_speaking_zh_umt5_xxl_bf16.pt
```

这样能减少 text condition 差异，让训练更集中在 audio/ref-image 到视频动态的学习上。

如果后续数据质量更好，可以切换：

```text
caption_emb_mode: fallback
```

含义是有 row-level caption embedding 的样本使用自己的 embedding，缺失时使用默认 embedding。

## Learning Rate Analysis

旧稳定 run 使用：

```text
lr = 3e-6
```

从 SwanLab 本地记录看：

- 前 500 step loss 均值约 `0.2183`
- 8001-8500 step loss 均值约 `0.2019`
- 8500 step 只下降约 `7.5%`
- grad_norm 长期约 `0.07`

结论：`3e-6` 稳，但偏保守，audio branch 学得慢。

后来直接把全模型 lr 改到：

```text
lr = 5e-5
```

在 step 82-87 附近出现明显不稳定：

```text
step 82 loss=0.4456 grad_norm=1.6060
step 83 loss=0.4419 grad_norm=3.1066
step 84 loss=0.5015 grad_norm=4.5025
step 85 loss=1.3755 grad_norm=16.4961
step 86 loss=1.8184 grad_norm=7.9468
step 87 loss=2.7227 grad_norm=8.3359
```

结论：`5e-5` 作为全模型统一 learning rate 太高，会冲坏 pretrained 分布。

## Current LR Schedule Recommendation

当前建议使用 cosine + warmup：

```yaml
max_steps: 12000
lr: 1.0e-5
lr_scheduler: cosine
warmup_steps: 200
lr_decay_steps: 12000
min_lr_ratio: 0.1
```

对应学习率：

```text
step 1      5e-08
step 50     2.5e-06
step 100    5e-06
step 200    1e-05
step 381    9.99e-06
step 1000   9.90e-06
step 6000   5.62e-06
step 12000  1e-06
```

选择理由：

- `warmup_steps=200` 约半个真实数据 epoch，可以避免前期直接大步更新。
- `lr=1e-5` 比旧 `3e-6` 更积极，但比爆掉的 `5e-5` 小 5 倍。
- `12000 steps` 约 `31.5` 个真实 epoch；考虑随机截取，覆盖的片段多于 31.5 个固定 pass。
- `min_lr_ratio=0.1` 让后期保留 `1e-6`，避免完全停止微调。

## What To Watch In The New Run

新 run 前 300 step 重点观察：

```text
grad_norm mostly < 1.5
loss mostly in 0.15-0.35
no long streak of loss > 0.6
```

如果出现：

```text
grad_norm repeatedly > 3
loss jumps above 1.0 for multiple steps
```

说明 peak lr 仍然偏大，应降到：

```text
lr = 5e-6
```

如果 1000-2000 step 仍然稳定但推理 audio control 很弱，可以考虑更细的参数组：

```text
base Wan params: 3e-6 or 5e-6
audio_proj / audio_cross_attn / audio_norm: 5e-5
```

这个比全模型 `5e-5` 更合理，因为新加 audio 分支需要更大 lr，而 pretrained Wan 主干不应该被大步扰动。

## Recommended Next Steps

1. 停掉已经用全模型 `5e-5` 跑起来的 run，不从它 resume。
2. 用当前 `configs/rf_final.yaml` 从干净初始模型或稳定 checkpoint 重开。
3. 前 300 step 看 loss/grad_norm 是否稳定。
4. 到 step 500 保存一个 checkpoint，固定同一组 ref/audio/text CFG 做推理对比。
5. 如果稳定，继续到 1000/2000/4000/8000/12000 做 checkpoint sweep。
6. 如果 audio CFG 仍弱，再实现参数组学习率，而不是继续提高全模型 lr。

