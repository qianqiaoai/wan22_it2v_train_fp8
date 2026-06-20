# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

USE_TE = True
if USE_TE:
    import transformer_engine.pytorch as te
    print("Using transformer_engine")

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# @amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


def patch_te_extra_state(model):
    """精准拦截 TE 模块，保留正常 weight/bias，仅屏蔽 _extra_state"""
    for m in model.modules():
        if isinstance(m, (te.Linear, te.RMSNorm, te.LayerNormLinear, te.DotProductAttention)):        
            # 1. 拦截 state_dict 的生成 (解决 Diffusers from_pretrained 的 missing_keys 报错)
            original_save = m._save_to_state_dict
            # 使用默认参数捕获当前循环中的 original_save，避免闭包陷阱
            def custom_save(destination, prefix, keep_vars, orig_save=original_save):
                # 先执行原版逻辑，把 weight 和 bias 存进去，同时也会存入 _extra_state
                orig_save(destination, prefix, keep_vars)
                # 精准删掉 _extra_state
                extra_state_key = prefix + '_extra_state'
                if extra_state_key in destination:
                    del destination[extra_state_key]
            m._save_to_state_dict = custom_save
            # 2. 拦截 load_state_dict 的检查 (解决 PyTorch 底层 strict=True 的报错)
            original_load = m._load_from_state_dict
            def custom_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs, orig_load=original_load):
                # 先执行原版加载逻辑，正常载入 weight 和 bias
                orig_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
                # 由于我们去掉了 _extra_state，底层机制会认为 "缺失" 了这个键，将其加入了 missing_keys # 我们需要将其从 missing_keys 列表中剔除，假装无事发生
                extra_state_key = prefix + '_extra_state'
                if extra_state_key in missing_keys:
                    missing_keys.remove(extra_state_key)
            m._load_from_state_dict = custom_load


def te_linear_padded(linear, x, multiple=16):
    if not USE_TE or not isinstance(linear, te.Linear) or x.size(1) % multiple == 0:
        return linear(x)

    seq_len = x.size(1)
    pad_len = multiple - seq_len % multiple
    padded = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
    return linear(padded)[:, :seq_len]


def te_mlp_padded(mlp, x, multiple=16):
    if not USE_TE or x.size(1) % multiple == 0:
        return mlp(x)

    seq_len = x.size(1)
    pad_len = multiple - seq_len % multiple
    padded = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
    return mlp(padded)[:, :seq_len]


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        if USE_TE:
            self.q = te.Linear(dim, dim, bias=True)
            self.k = te.Linear(dim, dim, bias=True)
            self.v = te.Linear(dim, dim, bias=True)
            self.o = te.Linear(dim, dim, bias=True)
            self.norm_q = te.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
            self.norm_k = te.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
            self.te_dpa = te.DotProductAttention(
                num_attention_heads=self.num_heads,
                kv_channels=self.head_dim,
                attention_dropout=0.0,
                attn_mask_type="no_mask",
                qkv_format="bshd",
            )
            # for module in [self.q, self.k, self.v, self.o, self.norm_q, self.norm_k, self.te_dpa]:
            #     # 阻止该模块在 model.state_dict() 中生成 _extra_state，绕过 Diffusers 前置校验
            #     module._save_to_state_dict = lambda *args, **kwargs: None
            #     # 阻止 PyTorch 底层在 load_state_dict 时寻找该模块的 key
            #     module._load_from_state_dict = lambda *args, **kwargs: None
            patch_te_extra_state(self)
        else:
            # layers
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)
            self.o = nn.Linear(dim, dim)
            self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
            self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
            

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(te_linear_padded(self.q, x)).view(b, s, n, d)
            k = self.norm_k(te_linear_padded(self.k, x)).view(b, s, n, d)
            v = te_linear_padded(self.v, x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if USE_TE:
            x = self.te_dpa(
                rope_apply(q, grid_sizes, freqs), rope_apply(k, grid_sizes, freqs), v,
                qkv_format="bshd",
                attn_mask_type="no_mask",
                # core_attention_bias_type="post_scale_bias",
                # core_attention_bias=self.attn_mask,
            )
        else:
            x = flash_attention(
                q=rope_apply(q, grid_sizes, freqs),
                k=rope_apply(k, grid_sizes, freqs),
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size)
            

        # output
        x = x.flatten(2)
        x = te_linear_padded(self.o, x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(te_linear_padded(self.q, x)).view(b, -1, n, d)
        k = self.norm_k(te_linear_padded(self.k, context)).view(b, -1, n, d)
        v = te_linear_padded(self.v, context).view(b, -1, n, d)

        # compute attention
        if USE_TE:
            x = self.te_dpa(q, k, v, qkv_format="bshd", attn_mask_type="no_mask")
        else:
            x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = te_linear_padded(self.o, x)
        return x


class WanAudioCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        context_dim,
        num_heads,
        qk_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        if USE_TE:
            self.q = te.Linear(dim, dim, bias=True)
            self.kv = te.Linear(context_dim, dim * 2, bias=True)
            self.o = te.Linear(dim, dim, bias=True)
            self.te_dpa = te.DotProductAttention(
                num_attention_heads=self.num_heads,
                kv_channels=self.head_dim,
                attention_dropout=0.0,
                attn_mask_type="no_mask",
                qkv_format="bshd",
            )
            patch_te_extra_state(self)
        else:
            self.q = nn.Linear(dim, dim)
            self.kv = nn.Linear(context_dim, dim * 2)
            self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, audio_context_dim]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = te_linear_padded(self.q, x).view(b, -1, n, d)
        kv = te_linear_padded(self.kv, context).view(b, -1, 2, n, d)
        k, v = kv.unbind(dim=2)
        q = self.norm_q(q)
        k = self.norm_k(k)

        if USE_TE:
            x = self.te_dpa(q, k, v, qkv_format="bshd", attn_mask_type="no_mask")
        else:
            x = flash_attention(q, k, v, k_lens=context_lens)

        x = x.flatten(2)
        x = te_linear_padded(self.o, x)
        return x


class AudioProjModel(nn.Module):
    def __init__(
        self,
        audio_window=5,
        vae_scale=4,
        audio_layers=12,
        audio_dim=768,
        intermediate_dim=512,
        output_dim=3072,
        context_tokens=32,
        norm_output_audio=True,
    ):
        super().__init__()
        self.audio_window = int(audio_window)
        self.vae_scale = int(vae_scale)
        self.audio_layers = int(audio_layers)
        self.audio_dim = int(audio_dim)
        self.latter_window = self.vae_scale + self.audio_window - 1
        self.intermediate_dim = int(intermediate_dim)
        self.output_dim = int(output_dim)
        self.context_tokens = int(context_tokens)

        self.proj1 = nn.Linear(self.audio_window * self.audio_layers * self.audio_dim, self.intermediate_dim)
        self.proj1_latter = nn.Linear(self.latter_window * self.audio_layers * self.audio_dim, self.intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, first_frame_audio, latter_frame_audio):
        batch_size = first_frame_audio.shape[0]
        first = first_frame_audio.reshape(batch_size, first_frame_audio.shape[1], -1)
        first = torch.relu(self.proj1(first.reshape(-1, first.shape[-1]))).view(
            batch_size, first_frame_audio.shape[1], -1)

        if latter_frame_audio.shape[1] > 0:
            latter = latter_frame_audio.reshape(batch_size, latter_frame_audio.shape[1], -1)
            latter = torch.relu(self.proj1_latter(latter.reshape(-1, latter.shape[-1]))).view(
                batch_size, latter_frame_audio.shape[1], -1)
            audio = torch.cat([first, latter], dim=1)
        else:
            audio = first

        batch_size, frames, channels = audio.shape
        audio = torch.relu(self.proj2(audio.reshape(batch_size * frames, channels)))
        context_tokens = self.proj3(audio).reshape(
            batch_size * frames, self.context_tokens, self.output_dim)
        if isinstance(self.norm, nn.LayerNorm):
            context_tokens = F.layer_norm(
                context_tokens.float(),
                self.norm.normalized_shape,
                self.norm.weight.float() if self.norm.weight is not None else None,
                self.norm.bias.float() if self.norm.bias is not None else None,
                self.norm.eps,
            ).to(context_tokens.dtype)
        else:
            context_tokens = self.norm(context_tokens)
        return context_tokens.reshape(batch_size, frames, self.context_tokens, self.output_dim)


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.audio_norm = None
        self.audio_cross_attn = None
        self.audio_context_dim = None
        self.norm2 = WanLayerNorm(dim, eps)
        if USE_TE:
            self.ffn = nn.Sequential(
                te.Linear(dim, ffn_dim, bias=True), nn.GELU(approximate='tanh'),
                te.Linear(ffn_dim, dim, bias=True))
            patch_te_extra_state(self.ffn)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
                nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def enable_audio_conditioning(self, audio_qk_norm=None, zero_init_output=False):
        if self.audio_cross_attn is not None:
            return
        # Old implementation inherited visual/text attention qk_norm:
        # qk_norm = self.qk_norm if audio_qk_norm is None else bool(audio_qk_norm)
        qk_norm = False if audio_qk_norm is None else bool(audio_qk_norm)
        self.audio_norm = WanLayerNorm(self.dim, self.eps, elementwise_affine=True)
        if self.audio_context_dim is None:
            raise ValueError("audio_context_dim must be set before enabling audio conditioning")
        # Old implementation expected audio tokens to already be in hidden dim:
        # self.audio_cross_attn = WanCrossAttention(
        #     self.dim, self.num_heads, (-1, -1), qk_norm, self.eps)
        self.audio_cross_attn = WanAudioCrossAttention(
            self.dim, self.audio_context_dim, self.num_heads, qk_norm, self.eps)
        # Old default zero-initialized the audio attention output projection:
        # zero_init_output=True
        if zero_init_output:
            nn.init.zeros_(self.audio_cross_attn.o.weight)
            if self.audio_cross_attn.o.bias is not None:
                nn.init.zeros_(self.audio_cross_attn.o.bias)

    def _audio_cross_attention(self, x, audio_context, grid_sizes):
        if audio_context is None or self.audio_cross_attn is None or self.audio_norm is None:
            return x.new_zeros(x.shape)

        residuals = []
        for sample_idx, (frames, height, width) in enumerate(grid_sizes.tolist()):
            frames, height, width = int(frames), int(height), int(width)
            seq_len = frames * height * width
            if seq_len == 0:
                residuals.append(x.new_zeros(1, x.shape[1], self.dim))
                continue

            audio = audio_context[sample_idx]
            if audio.shape[0] < frames:
                pad = audio[-1:].expand(frames - audio.shape[0], -1, -1)
                audio = torch.cat([audio, pad], dim=0)
            # Old implementation reshaped to self.dim because AudioProjModel emitted hidden-dim tokens:
            # audio = audio[:frames].reshape(frames, -1, self.dim).to(device=x.device, dtype=x.dtype)
            audio = audio[:frames].reshape(frames, -1, self.audio_context_dim).to(device=x.device, dtype=x.dtype)

            visual = self.audio_norm(x[sample_idx:sample_idx + 1, :seq_len])
            visual = visual.reshape(1, frames, height * width, self.dim).reshape(frames, height * width, self.dim)
            audio_residual = self.audio_cross_attn(visual, audio, None).reshape(1, seq_len, self.dim)
            if seq_len < x.shape[1]:
                audio_residual = torch.cat(
                    [audio_residual, audio_residual.new_zeros(1, x.shape[1] - seq_len, self.dim)],
                    dim=1)
            residuals.append(audio_residual)
        return torch.cat(residuals, dim=0)

    @staticmethod
    def _match_audio_frames(audio, target_frames):
        current_frames = audio.shape[0]
        if current_frames == target_frames:
            return audio
        if current_frames > target_frames:
            indices = torch.linspace(
                0,
                current_frames - 1,
                target_frames,
                device=audio.device,
            ).round().long()
            return audio.index_select(0, indices)
        pad = audio[-1:].repeat(target_frames - current_frames, *([1] * (audio.ndim - 1)))
        return torch.cat([audio, pad], dim=0)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        audio_context=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, audio_context):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            if audio_context is not None and self.audio_cross_attn is not None:
                x = x + self._audio_cross_attention(x, audio_context, grid_sizes)
            y = te_mlp_padded(self.ffn, 
                self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            # with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, audio_context)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        # assert e.dtype == torch.float32
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        x = (
            self.head(
                self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        self.audio_proj = None
        self.audio_window = 5
        self.audio_vae_scale = 4
        self.audio_layers = 12
        self.audio_dim = 768
        self.audio_context_tokens = 32

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _checkpoint_context_fn(self):
        fp8_context = getattr(self, "fp8_context", None)
        if fp8_context is None or not getattr(fp8_context, "enabled", False):
            return nullcontext(), nullcontext()
        return nullcontext(), fp8_context.context()

    def enable_audio_conditioning(
        self,
        audio_window=5,
        vae_scale=4,
        audio_layers=12,
        audio_dim=768,
        intermediate_dim=512,
        context_tokens=32,
        norm_output_audio=True,
        zero_init_audio_output=False,
        audio_qk_norm=None,
    ):
        self.audio_window = int(audio_window)
        self.audio_vae_scale = int(vae_scale)
        self.audio_layers = int(audio_layers)
        self.audio_dim = int(audio_dim)
        self.audio_context_tokens = int(context_tokens)
        self.audio_context_dim = self.audio_dim
        self.audio_proj = AudioProjModel(
            audio_window=self.audio_window,
            vae_scale=self.audio_vae_scale,
            audio_layers=self.audio_layers,
            audio_dim=self.audio_dim,
            intermediate_dim=intermediate_dim,
            # Old implementation projected audio tokens directly to DiT hidden dim:
            # output_dim=self.dim,
            output_dim=self.audio_context_dim,
            context_tokens=self.audio_context_tokens,
            norm_output_audio=norm_output_audio,
        )
        for block in self.blocks:
            block.audio_context_dim = self.audio_context_dim
            block.enable_audio_conditioning(
                audio_qk_norm=audio_qk_norm,
                zero_init_output=zero_init_audio_output)

        device = self.patch_embedding.weight.device
        dtype = self.patch_embedding.weight.dtype
        self.audio_proj.to(device=device, dtype=dtype)
        for block in self.blocks:
            block.audio_norm.to(device=device, dtype=dtype)
            block.audio_cross_attn.to(device=device, dtype=dtype)

    def forward(
        self,
        x,
        t,
        context,
        seq_len=None,
        y=None,
        audio_emb=None,
        mask_clean_first_audio=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if seq_len:
            assert seq_lens.max() <= seq_len
        else:
            seq_len = max([u.size(1) for u in x])
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).to(dtype=torch.bfloat16))
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        audio_context = self._prepare_audio_context(
            audio_emb,
            grid_sizes,
            mask_clean_first_audio=mask_clean_first_audio,
        )

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            audio_context=audio_context)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    e0,
                    seq_lens,
                    grid_sizes,
                    self.freqs,
                    context,
                    context_lens,
                    audio_context,
                    context_fn=self._checkpoint_context_fn,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _prepare_audio_context(self, audio_emb, grid_sizes, mask_clean_first_audio=False):
        if audio_emb is None:
            return None
        if self.audio_proj is None:
            raise ValueError("audio_emb was provided but audio conditioning is not enabled")

        if torch.is_tensor(audio_emb):
            audio_items = [audio_emb] if audio_emb.dim() == 3 else [item for item in audio_emb]
        else:
            audio_items = list(audio_emb)
        if len(audio_items) != grid_sizes.shape[0]:
            raise ValueError(
                f"audio_emb batch size {len(audio_items)} does not match video batch size {grid_sizes.shape[0]}")

        contexts = []
        for raw_audio, grid_size in zip(audio_items, grid_sizes.tolist()):
            latent_frames = int(grid_size[0])
            target_video_frames = 1 + max(0, latent_frames - 1) * self.audio_vae_scale
            raw_audio = self._normalize_audio_tensor(raw_audio, target_video_frames)
            audio_windows = self._audio_sliding_windows(raw_audio)
            contexts.append(self._project_audio_windows(audio_windows, latent_frames))

        max_frames = max(context.shape[0] for context in contexts)
        padded = []
        for context in contexts:
            if context.shape[0] < max_frames:
                pad = context.new_zeros(max_frames - context.shape[0], context.shape[1], context.shape[2])
                context = torch.cat([context, pad], dim=0)
            padded.append(context)
        audio_context = torch.stack(padded, dim=0)
        if mask_clean_first_audio and audio_context.shape[1] > 0:
            audio_context = audio_context.clone()
            audio_context[:, 0] = 0
        return audio_context

    def _normalize_audio_tensor(self, audio, target_video_frames):
        audio = audio.to(device=self.patch_embedding.weight.device, dtype=self.patch_embedding.weight.dtype)
        if audio.dim() == 4 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        if audio.dim() != 3:
            raise ValueError(
                f"audio_emb must have shape [T, {self.audio_layers}, {self.audio_dim}], got {tuple(audio.shape)}")
        if audio.shape[1] != self.audio_layers and audio.shape[0] == self.audio_layers:
            audio = audio.transpose(0, 1).contiguous()
        if audio.shape[1] != self.audio_layers or audio.shape[2] != self.audio_dim:
            raise ValueError(
                f"audio_emb must have shape [T, {self.audio_layers}, {self.audio_dim}], got {tuple(audio.shape)}")
        if audio.shape[0] == 0:
            raise ValueError("audio_emb must not be empty")

        if audio.shape[0] < target_video_frames:
            pad = audio[-1:].expand(target_video_frames - audio.shape[0], -1, -1)
            audio = torch.cat([audio, pad], dim=0)
        return audio[:target_video_frames].contiguous()

    def _audio_sliding_windows(self, audio):
        radius = self.audio_window // 2
        offsets = torch.arange(self.audio_window, device=audio.device) - radius
        indices = torch.arange(audio.shape[0], device=audio.device).unsqueeze(1) + offsets.unsqueeze(0)
        indices = indices.clamp_(0, audio.shape[0] - 1).long()
        return audio[indices]

    def _project_audio_windows(self, audio_windows, latent_frames):
        first_frame_audio = audio_windows[:1].unsqueeze(0)
        if latent_frames > 1:
            required_latter = (latent_frames - 1) * self.audio_vae_scale
            latter = audio_windows[1:1 + required_latter].reshape(
                1,
                latent_frames - 1,
                self.audio_vae_scale,
                self.audio_window,
                self.audio_layers,
                self.audio_dim,
            )
            mid_idx = self.audio_window // 2
            first_of_group = latter[:, :, :1, :mid_idx + 1]
            middle_of_group = latter[:, :, 1:-1, mid_idx:mid_idx + 1]
            last_of_group = latter[:, :, -1:, mid_idx:]
            latter_frame_audio = torch.cat([
                first_of_group.reshape(1, latent_frames - 1, -1, self.audio_layers, self.audio_dim),
                middle_of_group.reshape(1, latent_frames - 1, -1, self.audio_layers, self.audio_dim),
                last_of_group.reshape(1, latent_frames - 1, -1, self.audio_layers, self.audio_dim),
            ], dim=2)
        else:
            latter_frame_audio = audio_windows.new_empty(
                1,
                0,
                self.audio_vae_scale + self.audio_window - 1,
                self.audio_layers,
                self.audio_dim,
            )
        return self.audio_proj(first_frame_audio, latter_frame_audio).squeeze(0)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
