# 2026-06-24 audio QK norm full-dim change

## Goal

把 audio cross attention 的 Q/K norm 改成：

```text
Linear -> RMSNorm(dim) -> split heads
```

而不是旧实现：

```text
Linear -> split heads -> RMSNorm(head_dim)
```

同时在当前训练配置中打开 `audio_qk_norm`。

## Files changed

- `wan2/modules/model.py`
- `configs/rf_final.yaml`

## Details

### `wan2/modules/model.py`

修改 `WanAudioCrossAttention`：

- 旧代码保留为注释。
- `norm_q` / `norm_k` 从 `WanRMSNorm(self.head_dim)` 改成 `WanRMSNorm(dim)`。
- Q path:

```text
q = Linear(x)
q = RMSNorm(dim)(q)
q = view(B, L, num_heads, head_dim)
```

- K/V path:

```text
kv = Linear(audio_context)
k, v = split last dim
k = RMSNorm(dim)(k)
k = view(B, L_audio, num_heads, head_dim)
v = view(B, L_audio, num_heads, head_dim)
```

V 不做 RMSNorm。

### `configs/rf_final.yaml`

打开：

```yaml
audio_qk_norm: true
```

旧值保留为注释：

```yaml
# audio_qk_norm: false
```

注意：`generator_trainable_modules` 中的 `audio_norm` 是前一次实验已经加入的，不是本次新增目标，但当前配置会继续保留。

## Validation

已运行：

```bash
python -m py_compile wan2/modules/model.py
```

结果：通过。

已运行：

```bash
python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('configs/rf_final.yaml'); print(cfg.audio_condition.audio_qk_norm)"
```

结果：

```text
True
```

已运行 CPU fallback forward shape check：

```bash
python -c "import torch; import wan2.modules.model as wm; wm.USE_TE=False; wm.flash_attention=lambda q,k,v,k_lens=None: torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)).transpose(1,2).contiguous(); m=wm.WanAudioCrossAttention(dim=64, context_dim=32, num_heads=4, qk_norm=True).float(); x=torch.randn(2,5,64); c=torch.randn(2,7,32); y=m(x,c); print(tuple(y.shape), y.dtype, torch.isfinite(y).all().item(), tuple(m.norm_q.weight.shape), tuple(m.norm_k.weight.shape))"
```

结果：

```text
(2, 5, 64) torch.float32 True (64,) (64,)
```

这确认 `norm_q` / `norm_k` 是完整 dim 的 RMSNorm，不是 head_dim。

## Not validated

- 没有跑完整训练 smoke。
- 没有在真实 TransformerEngine CUDA path 上跑完整模型 forward；当前沙箱里 CUDA/NVML 初始化不可用，直接实例化 TE Linear 会报 `TransformerEngine needs CUDA`。
- 没有验证新 checkpoint 是否能和旧 checkpoint strict load。这个修改会改变 `audio_cross_attn.norm_q.weight` / `norm_k.weight` 的 shape；旧 `audio_qk_norm=false` checkpoint 不包含有效 RMSNorm 参数，通常新实验应从 base 初始化重新训，不建议直接 strict load 旧 audio_qk_norm=false checkpoint。

## Risk

- 如果用旧 `audio_qk_norm=false` checkpoint 继续训，strict load 可能遇到 Q/K norm 参数 shape 或 missing key 问题，取决于 checkpoint 是否保存了 Identity 对应参数。
- full-dim Q/K RMSNorm 会改变 audio branch 的数值尺度，建议先从小步 smoke 或 500-step sanity 开始观察 loss 和画面稳定性。
