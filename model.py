from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from utils.wan_wrapper import LTX2VideoVAEWrapper, WanDiffusionWrapper, Wan2_1_VAEWrapper, Wan2_2_VAEWrapper
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
            getattr(
                audio_condition,
                "p_audio_drop",
                getattr(
                    audio_condition,
                    "drop_prob",
                    getattr(audio_condition, "audio_dropout_prob", getattr(audio_condition, "dropout_prob", 0.0)),
                ),
            )
        ) if audio_condition is not None else 0.0
        if not 0.0 <= self.audio_dropout_prob <= 1.0:
            raise ValueError(f"audio dropout probability must be in [0, 1], got {self.audio_dropout_prob}")
        self.mask_clean_first_audio = bool(
            getattr(audio_condition, "mask_clean_first_frame", True)
        ) if audio_condition is not None else True

        model_name = getattr(config, "generator_name", getattr(config, "model_name", "Wan2.1-T2V-1.3B"))
        vae_name = getattr(config, "vae_name", model_name)
        vae_type = str(getattr(config, "vae_type", "wan5b")).lower()
        vae_profile = self._vae_profile(vae_type)
        self.vae_type = vae_type
        self._configure_audio_for_vae(audio_condition, vae_profile)

        self.generator = WanDiffusionWrapper(
            **getattr(config, "model_kwargs", {}),
            model_name=model_name,
            audio_condition=audio_condition,
            # is_causal=self.generator_type == "causal",
        )
        self._configure_generator_for_vae(vae_profile)
        self._set_generator_trainable(True)
        self._ensure_vae_trainable_modules(vae_type)
        self._apply_generator_trainable_filter()

        if getattr(config, "gradient_checkpointing", False):
            self.generator.enable_gradient_checkpointing()

        # Text conditions are loaded from precomputed caption_emb files by the dataset.
        self.text_encoder = None

        self.vae = self._build_vae(vae_type=vae_type, vae_name=vae_name)
        self.vae.requires_grad_(False)
        self._validate_vae_generator_compatibility()

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _build_vae(self, vae_type: str, vae_name: str):
        if vae_type in {"wan5b", "wan2.2", "wan2_2", "wan"}:
            return Wan2_2_VAEWrapper(model_name=vae_name)
        if vae_type in {"ltx2", "ltx2vae", "ltx-2", "ltx"}:
            return LTX2VideoVAEWrapper(
                model_name=vae_name,
                checkpoint_name=getattr(self.config, "ltx2_checkpoint_name", None),
                encoder_checkpoint_name=getattr(self.config, "ltx2_encoder_checkpoint_name", None),
                decoder_checkpoint_name=getattr(self.config, "ltx2_decoder_checkpoint_name", None),
                repo_path=getattr(self.config, "ltx2_repo_path", None),
            )
        if vae_type in {"wan2.1", "wan2_1"}:
            return Wan2_1_VAEWrapper(model_name=vae_name)
        raise ValueError(f"Unsupported vae_type={vae_type!r}. Expected one of: wan5b, ltx2, wan2_1")

    @staticmethod
    def _vae_profile(vae_type: str):
        if vae_type in {"wan5b", "wan2.2", "wan2_2", "wan"}:
            return {"latent_channels": 48, "patch_size": (1, 2, 2), "audio_vae_scale": 4}
        if vae_type in {"ltx2", "ltx2vae", "ltx-2", "ltx"}:
            return {"latent_channels": 128, "patch_size": (1, 1, 1), "audio_vae_scale": 8}
        if vae_type in {"wan2.1", "wan2_1"}:
            return {"latent_channels": 16, "patch_size": (1, 2, 2), "audio_vae_scale": 4}
        return {"latent_channels": None, "patch_size": None, "audio_vae_scale": None}

    @staticmethod
    def _uses_ltx2_vae(vae_type: str) -> bool:
        return vae_type in {"ltx2", "ltx2vae", "ltx-2", "ltx"}

    def _configure_audio_for_vae(self, audio_condition, vae_profile):
        if audio_condition is None or vae_profile.get("audio_vae_scale") is None:
            return
        target_scale = int(vae_profile["audio_vae_scale"])
        audio_condition.vae_scale = target_scale
        audio_condition.audio_vae_scale = target_scale

    def _configure_generator_for_vae(self, vae_profile):
        latent_channels = vae_profile.get("latent_channels")
        patch_size = vae_profile.get("patch_size")
        if latent_channels is None or patch_size is None:
            return
        generator_model = self.generator.model0 if getattr(self.generator, "dual_exp", False) else self.generator.model
        current_patch_size = tuple(getattr(generator_model, "patch_size", ()))
        if (
            int(getattr(generator_model, "in_dim", -1)) == int(latent_channels)
            and int(getattr(generator_model, "out_dim", -1)) == int(latent_channels)
            and current_patch_size == tuple(patch_size)
        ):
            return
        self.generator.reset_patch_io(
            in_dim=int(latent_channels),
            out_dim=int(latent_channels),
            patch_size=patch_size,
        )

    def _ensure_vae_trainable_modules(self, vae_type: str):
        if not self._uses_ltx2_vae(vae_type):
            return
        patterns = getattr(self.config, "generator_trainable_modules", None)
        if patterns is None:
            patterns = getattr(self.config, "trainable_modules", None)
        if patterns is None:
            return
        if isinstance(patterns, str):
            patterns = [patterns]
        patterns = [str(pattern) for pattern in patterns if str(pattern)]
        for pattern in ("patch_embedding", "head.head"):
            if not any(pattern == existing for existing in patterns):
                patterns.append(pattern)
        self.config.generator_trainable_modules = patterns

    def _validate_vae_generator_compatibility(self):
        latent_channels = getattr(self.vae, "latent_channels", None)
        if latent_channels is None:
            return
        self._validate_generator_latent_channels(latent_channels)

    def _validate_generator_latent_channels(self, latent_channels):
        if latent_channels is None:
            return
        generator_model = self.generator.model0 if getattr(self.generator, "dual_exp", False) else self.generator.model
        in_dim = getattr(generator_model, "in_dim", None)
        out_dim = getattr(generator_model, "out_dim", None)
        mismatches = []
        if in_dim is not None and int(in_dim) != int(latent_channels):
            mismatches.append(f"generator.in_dim={in_dim}")
        if out_dim is not None and int(out_dim) != int(latent_channels):
            mismatches.append(f"generator.out_dim={out_dim}")
        if mismatches:
            raise ValueError(
                f"VAE latent_channels={latent_channels} is incompatible with "
                + ", ".join(mismatches)
                + ". Use a DiT checkpoint/config with matching latent channels before training or inference."
            )

    def _generator_patch_size(self):
        generator_model = self.generator.model0 if getattr(self.generator, "dual_exp", False) else self.generator.model
        return tuple(int(v) for v in getattr(generator_model, "patch_size", (1, 2, 2)))

    def filter_generator_state_dict_for_current_vae(self, state_dict):
        if not self._uses_ltx2_vae(self.vae_type):
            return state_dict, []
        current_state = self.generator.state_dict()
        skip_prefixes = set()
        check_suffixes = ("patch_embedding.weight", "head.head.weight")
        for key, value in state_dict.items():
            if not torch.is_tensor(value) or not key.endswith(check_suffixes):
                continue
            current_value = current_state.get(key)
            if current_value is None or tuple(current_value.shape) != tuple(value.shape):
                skip_prefixes.add(key.rsplit(".", 1)[0] + ".")
        if not skip_prefixes:
            return state_dict, []
        filtered = {}
        skipped = []
        for key, value in state_dict.items():
            if any(key.startswith(prefix) for prefix in skip_prefixes):
                skipped.append(key)
            else:
                filtered[key] = value
        return filtered, skipped

    def load_generator_state_dict_for_current_vae(self, state_dict, strict=True):
        filtered, skipped = self.filter_generator_state_dict_for_current_vae(state_dict)
        load_result = self.generator.load_state_dict(filtered, strict=False if skipped else strict)
        if skipped:
            allowed_missing = set(skipped)
            allowed_suffixes = (
                "patch_embedding.weight",
                "patch_embedding.bias",
                "head.head.weight",
                "head.head.bias",
            )
            missing = [
                key for key in load_result.missing_keys
                if key not in allowed_missing and not key.endswith(allowed_suffixes)
            ]
            unexpected = list(load_result.unexpected_keys)
            if missing or unexpected:
                raise RuntimeError(
                    "Unexpected checkpoint incompatibility after skipping LTX VAE patch I/O keys: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            print("Skipped incompatible LTX VAE patch I/O checkpoint keys: " + ", ".join(skipped))
        return load_result

    def _set_generator_trainable(self, trainable: bool):
        if getattr(self.generator, "dual_exp", False):
            self.generator.model0.requires_grad_(trainable)
            self.generator.model1.requires_grad_(trainable)
        else:
            self.generator.model.requires_grad_(trainable)

    def _apply_generator_trainable_filter(self):
        patterns = getattr(self.config, "generator_trainable_modules", None)
        if patterns is None:
            patterns = getattr(self.config, "trainable_modules", None)
        if patterns is None:
            return

        if isinstance(patterns, str):
            patterns = [patterns]
        patterns = [str(pattern) for pattern in patterns if str(pattern)]
        if not patterns:
            return

        self.generator.requires_grad_(False)
        matched = {pattern: 0 for pattern in patterns}
        trainable_count = 0
        trainable_numel = 0
        total_numel = 0
        for name, param in self.generator.named_parameters():
            total_numel += param.numel()
            should_train = any(pattern in name for pattern in patterns)
            param.requires_grad_(should_train)
            if not should_train:
                continue
            trainable_count += 1
            trainable_numel += param.numel()
            for pattern in patterns:
                if pattern in name:
                    matched[pattern] += 1

        missing = [pattern for pattern, count in matched.items() if count == 0]
        if missing:
            raise ValueError(
                "generator_trainable_modules did not match any parameters: "
                + ", ".join(missing)
            )

        print(
            "Generator trainable filter enabled: "
            f"patterns={patterns}, trainable_tensors={trainable_count}, "
            f"trainable_numel={trainable_numel}, total_numel={total_numel}"
        )

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
                p_audio_drop = self.audio_dropout_prob
                is_drop = (torch.rand(audio_embeds.shape[0], device=audio_embeds.device) < p_audio_drop)
                is_drop = is_drop.to(audio_embeds.dtype)
                audio_embeds = audio_embeds * (1 - is_drop).reshape(
                    (-1,) + (1,) * (audio_embeds.ndim - 1)
                ).type_as(audio_embeds)
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
            pt, ph, pw = self._generator_patch_size()
            temp_ts = (mask2[0][0][::pt, 0, ::ph, ::pw] * timestep[0][0]).flatten()
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
