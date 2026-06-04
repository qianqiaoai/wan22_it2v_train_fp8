from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from utils.wan_wrapper import WanDiffusionWrapper, Wan2_1_VAEWrapper, Wan2_2_VAEWrapper
from wan2.utils.utils import masks_like


class RectifiedFlowFineTuneModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.generator_type = getattr(config, "generator_type", "bidirectional")
        self.num_train_timestep = int(getattr(config, "num_train_timestep", 1000))
        audio_condition = getattr(config, "audio_condition", None)
        self.audio_condition_enabled = bool(getattr(audio_condition, "enabled", False)) if audio_condition is not None else False
        self.audio_dropout_prob = float(
            getattr(audio_condition, "drop_prob", getattr(audio_condition, "audio_dropout_prob", getattr(audio_condition, "dropout_prob", 0.0)))
        ) if audio_condition is not None else 0.0
        self.mask_clean_first_audio = bool(
            getattr(audio_condition, "mask_clean_first_frame", True)
        ) if audio_condition is not None else True

        model_name = getattr(config, "generator_name", getattr(config, "model_name", "Wan2.1-T2V-1.3B"))
        vae_name = getattr(config, "vae_name", model_name)

        self.generator = WanDiffusionWrapper(
            **getattr(config, "model_kwargs", {}),
            model_name=model_name,
            audio_condition=audio_condition,
            # is_causal=self.generator_type == "causal",
        )
        self._set_generator_trainable(True)

        if getattr(config, "gradient_checkpointing", False):
            self.generator.enable_gradient_checkpointing()

        # Text conditions are loaded from precomputed caption_emb files by the dataset.
        self.text_encoder = None

        WanVAEWrapper = Wan2_2_VAEWrapper
        self.vae = WanVAEWrapper(model_name=vae_name)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _set_generator_trainable(self, trainable: bool):
        if getattr(self.generator, "dual_exp", False):
            self.generator.model0.requires_grad_(trainable)
            self.generator.model1.requires_grad_(trainable)
        else:
            self.generator.model.requires_grad_(trainable)

    @torch.no_grad()
    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        return self.vae.encode_to_latent(frames).to(device=self.device, dtype=self.dtype)

    def rectified_flow_loss(
        self,
        clean_latent: torch.Tensor,
        conditional_dict: dict,
        task: str = None,
        audio_embeds: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size, num_frames = clean_latent.shape[:2]
        noise = torch.randn_like(clean_latent)
        if self.audio_condition_enabled:
            if audio_embeds is None:
                raise ValueError("audio_condition.enabled is true, but batch does not contain audio_embeds")
            audio_embeds = audio_embeds.to(device=self.device, dtype=self.dtype)
            if self.training and self.audio_dropout_prob > 0.0:
                keep_shape = (audio_embeds.shape[0],) + (1,) * (audio_embeds.ndim - 1)
                keep = torch.rand(keep_shape, device=audio_embeds.device) >= self.audio_dropout_prob
                audio_embeds = audio_embeds * keep.to(dtype=audio_embeds.dtype)
        else:
            audio_embeds = None

        index = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            [batch_size, 1],
            device=self.device,
            dtype=torch.long,
        ).repeat(1, num_frames)

        timestep = self.scheduler.timesteps[index].to(device=self.device, dtype=self.dtype)
        noisy_latent = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frames))
        if task=='i2v-5B':
            noisy_latent[:, :1] = clean_latent[:, :1]
            mask1, mask2 = masks_like([noise], zero=True)
            temp_ts = (mask2[0][0][:, 0, ::2, ::2] * timestep[0][0]).flatten()
            timestep = temp_ts.unsqueeze(0)

        target = self.scheduler.training_target(clean_latent, noise, timestep)

        flow_pred = self.generator(
            noisy_image_or_video=noisy_latent,
            conditional_dict=conditional_dict,
            timestep=timestep,
            audio_embeds=audio_embeds,
            mask_clean_first_audio=self.audio_condition_enabled and task == 'i2v-5B' and self.mask_clean_first_audio,
        )

        # per_frame_loss = F.mse_loss(flow_pred.float(), target.float(), reduction="none").mean(dim=(2, 3, 4))
        # weights = self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frames)).float()
        # loss = (per_frame_loss * weights).mean()
        if task=='i2v-5B':
            loss =  F.mse_loss(flow_pred[:, 1:], target[:, 1:], reduction="mean")
        else:
            loss =  F.mse_loss(flow_pred, target, reduction="mean")

        return loss, {
            "flow_pred": flow_pred.detach(),
            "timestep": timestep.detach(),
        }
