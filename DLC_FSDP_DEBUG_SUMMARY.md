# DLC FSDP 初始化卡住问题排查总结

日期：2026-06-05

## 背景

训练任务：Wan2.2 TI2V 5B + audio conditioning + FSDP + FP8。

现象：在 DLC 上启动训练后，日志长期停在 FSDP 初始化附近，例如：

```text
fsdp_wrap: entering FSDP constructor device_id=4
```

后续没有：

```text
fsdp_wrap: FSDP constructor returned
```

同时也没有出现：

```text
DATASET SIZE 36562
```

因此最初怀疑过 dataset、num_workers、NCCL、FSDP sharding strategy、模型加载等问题。

## 已排除的问题

### 1. 不是 dataset / num_workers

代码顺序中，`DATASET SIZE` 在 `trainer._build_dataloader()` 里打印，发生在 generator FSDP wrap、VAE 移动到 GPU、optimizer 创建之后。

实际卡住时日志停在：

```text
fsdp_wrap: entering FSDP constructor
```

说明还没有走到 dataloader 创建阶段。因此：

- `num_workers` 不是当前卡点。
- dataset 读取视频、caption embedding、audio embedding 也不是当前卡点。

### 2. 不是 SwanLab

SwanLab 初始化在模型构造之前，且本地 2 卡 smoke test 已经验证 SwanLab 可以创建云端 run。

卡住位置在 FSDP / model.to 之后，与 SwanLab 无关。

### 3. 不是基础 NCCL 通信

运行：

```bash
torchrun --standalone --nproc_per_node=8 scripts/debug_fsdp_init.py \
  --config_path configs/rf_final.yaml \
  --mode nccl
```

DLC 单机 8 卡结果：

```text
NCCL all_reduce result=36.0
barrier after nccl: complete
script complete
```

结论：

- `torchrun --standalone` 正常。
- `init_process_group` 正常。
- 8 卡 NCCL all-reduce 正常。
- rank 间 barrier 正常。

### 4. 不是 PyTorch FSDP API 整体坏了

运行：

```bash
torchrun --standalone --nproc_per_node=8 scripts/debug_fsdp_init.py \
  --config_path configs/rf_final.yaml \
  --mode tiny-fsdp \
  --sharding_strategy full
```

DLC 单机 8 卡结果：

```text
FSDP constructor sharding=full ... complete in 0.0s
tiny fsdp module ready type=FullyShardedDataParallel
```

结论：

- FSDP `FULL_SHARD` 对小模型正常。
- DLC 的 PyTorch FSDP 基础功能不是完全不可用。

### 5. 不是 `hybrid_full` 独有问题

训练配置分别测试过：

```yaml
sharding_strategy: hybrid_full
```

和：

```yaml
sharding_strategy: full
```

都卡在：

```text
fsdp_wrap: entering FSDP constructor
```

因此 `hybrid_full` 不是唯一原因。

### 6. 不是 size auto-wrap 太细

测试过：

```yaml
sharding_strategy: full
generator_fsdp_wrap_strategy: size
generator_fsdp_min_num_params: 100000000000
```

仍然卡住。该配置基本避免递归 auto-wrap 很多子模块。

结论：不是 auto-wrap 阈值太小导致的主要问题。

## 关键定位过程

### 1. FSDP 卡点实际是 `_move_states_to_device`

训练中的 traceback 显示：

```text
torch/distributed/fsdp/_init_utils.py line 1021 in _move_states_to_device
torch/distributed/fsdp/_init_utils.py line 991 in _move_module_to_device
torch/distributed/fsdp/_init_utils.py line 598 in _init_param_handle_from_module
torch/distributed/fsdp/fully_sharded_data_parallel.py line 497 in __init__
```

这说明 FSDP constructor 内部正在把 module states 移动到 GPU。

### 2. 真实 Wan generator 单卡 `.to(cuda)` 也卡

运行：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/debug_fsdp_init.py \
  --config_path configs/rf_final.yaml \
  --mode real-to
```

DLC 结果停在：

```text
plain generator.to(cuda): enter
```

长时间无：

```text
plain generator.to(cuda): complete
```

结论：

- 单卡也会卡。
- 不是 8 卡并发导致。
- 不是 FSDP 本身导致。
- 问题是大 Wan generator 从 CPU 搬到 GPU 极慢。

### 3. 顶层子模块定位

运行：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/debug_fsdp_init.py \
  --config_path configs/rf_final.yaml \
  --mode real-to-children
```

DLC 结果停在：

```text
move top-level child model type=WanModel params=6245028544 cpu_mib=23822.9: enter
```

结论：

- 卡在 `WanDiffusionWrapper.model`，也就是 `WanModel` 本体。
- `WanModel` CPU fp32 参数约 23.8 GiB。

### 4. 逐参数搬运速度异常慢

运行：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/debug_fsdp_init.py \
  --config_path configs/rf_final.yaml \
  --mode real-to-params \
  --max_items 30
```

DLC 关键结果：

```text
48.0 MiB -> complete in 5.3s
36.0 MiB -> complete in 3.8s
216.0 MiB -> complete in 23.5s
```

折算约 9 MiB/s，远低于正常 CPU -> GPU H2D 拷贝速度。

如果按该速度完整搬 23.8 GiB：

```text
23822 MiB / 9 MiB/s ~= 44 分钟
```

这解释了为什么训练看起来像卡死。

### 5. bf16 搬运可以完成

运行：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/debug_fsdp_init.py \
  --config_path configs/rf_final.yaml \
  --mode real-to-bf16
```

DLC 结果：

```text
plain generator.to(cuda:0, dtype=torch.bfloat16): complete in 82.1s
```

结论：

- DLC 不是完全不能搬 GPU。
- fp32 搬运路径极慢或近似卡死。
- bf16 搬运可行且明显更快。

## DSW 对照

在 DSW 上运行相同 `real-to-params --max_items 30`，结果：

```text
WanModel loaded in 0.8s
48.0 MiB -> complete in 0.0s
36.0 MiB -> complete in 0.0s
216.0 MiB -> complete in 0.0s
```

而 DLC：

```text
WanModel loaded in ~21s
216.0 MiB -> complete in 23.5s
```

结论：

- DSW 和 DLC 虽然都使用 `/mnt/data/...` NAS 路径，但实际 I/O、page cache、mount client、网络路径或容器限制不同。
- DSW 很可能有热 page cache 或更快的 NAS 访问路径。
- DLC 的慢点发生在权重数据真正被触碰并搬 GPU 时。

## 最终结论

当前卡住根因不是训练代码主逻辑，也不是 dataset、num_workers、SwanLab、NCCL、FSDP sharding strategy。

根因更准确地说是：

> DLC 环境下，`WanModel` 的 fp32 CPU 参数从 NAS/CPU 内存搬到 GPU 的速度异常慢。FSDP constructor 卡在 `_move_states_to_device` 是该问题的表现。

## 为什么同一个 NAS 路径表现不同

同一个 `/mnt/data/...` 路径不保证同一个性能：

- DSW 和 DLC 可能运行在不同物理节点。
- Linux page cache 是节点本地的，不随容器镜像或 NAS 路径共享。
- DSW 可能已经热缓存模型文件，DLC 是冷缓存。
- DLC 可能使用不同 NAS mount 参数或网络链路。
- DLC 容器可能有 I/O 或内存/cgroup 限制。
- `from_pretrained` 可能先建立 CPU tensor 或读取元信息，真正大量读取权重页发生在 `.to(cuda)` 时。

## 建议修复

### 方案 1：generator 直接 bf16 加载

建议增加配置项：

```yaml
generator_load_dtype: bf16
```

然后在 `WanModel.from_pretrained` 时传入：

```python
WanModel.from_pretrained(path, torch_dtype=torch.bfloat16)
```

这样可以避免：

- 先在 CPU 上保留 23.8 GiB fp32 参数。
- FSDP 初始化时走 fp32 H2D 慢路径。

这和 FP8 训练不冲突：

- `fp8: true` 仍然控制 Transformer Engine 的 FP8 计算。
- FSDP mixed precision 当前本来就是 `param_dtype=torch.bfloat16`。
- bf16 权重加载是合理的训练初始化优化。

### 方案 2：DLC 启动后复制模型到节点本地盘

不要依赖 NAS 冷读性能。DLC 启动后先复制：

```bash
mkdir -p /tmp/qiaoqian_models
cp -a /mnt/data/nlp/user/qiaoqian/models/Wan2.2-TI2V-5B /tmp/qiaoqian_models/
cp -a /mnt/data/nlp/user/qiaoqian/models/chinese-wav2vec2-base /tmp/qiaoqian_models/
```

然后配置改成：

```yaml
generator_name: /tmp/qiaoqian_models/Wan2.2-TI2V-5B
vae_name: /tmp/qiaoqian_models/Wan2.2-TI2V-5B
audio_condition:
  audio_model_name_or_path: /tmp/qiaoqian_models/chinese-wav2vec2-base
```

### 方案 3：不要指望 DSW 保存镜像缓存 NAS page cache

保存镜像通常只保存容器文件系统层，不保存 `/mnt/data/...` 的 Linux page cache。

只有把模型文件真的复制进镜像内部路径，例如 `/opt/models/...`，保存镜像才会包含模型。但模型很大，镜像构建和分发成本高，不一定划算。

## 当前保留的 debug 工具

`scripts/debug_fsdp_init.py` 支持以下模式：

```bash
# 测基础 NCCL
torchrun --standalone --nproc_per_node=8 scripts/debug_fsdp_init.py \
  --config_path configs/rf_final.yaml \
  --mode nccl

# 测小模型 FSDP
torchrun --standalone --nproc_per_node=8 scripts/debug_fsdp_init.py \
  --config_path configs/rf_final.yaml \
  --mode tiny-fsdp \
  --sharding_strategy full

# 测真实模型整体搬 GPU
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/debug_fsdp_init.py \
  --config_path configs/rf_final.yaml \
  --mode real-to

# 测真实模型 bf16 搬 GPU
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/debug_fsdp_init.py \
  --config_path configs/rf_final.yaml \
  --mode real-to-bf16

# 测真实模型顶层子模块搬 GPU
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/debug_fsdp_init.py \
  --config_path configs/rf_final.yaml \
  --mode real-to-children

# 测真实模型逐参数搬 GPU
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/debug_fsdp_init.py \
  --config_path configs/rf_final.yaml \
  --mode real-to-params \
  --max_items 30
```

## 下一步建议

优先级：

1. 加 `generator_load_dtype: bf16`，让 generator 直接 bf16 加载。
2. 在 DLC 上用本地盘模型路径测试。
3. 如果仍慢，再继续检查 DLC 节点的 NAS mount、page cache、cgroup I/O 限制、NUMA/PCIe 拓扑。

