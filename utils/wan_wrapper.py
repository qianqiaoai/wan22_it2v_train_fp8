import os
import types
from contextlib import nullcontext
from typing import List, Optional, Any, Tuple
import torch
from torch import nn

from utils.scheduler import FlowMatchScheduler
from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan2.modules.tokenizers import HuggingfaceTokenizer
from wan2.modules.model import WanModel
from wan2.modules.vae2_1 import _video_vae as _video_vae2_1
from wan2.modules.vae2_2 import _video_vae as _video_vae2_2
from wan2.modules.t5 import umt5_xxl


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_name="Wan2.1-T2V-14B") -> None:
        super().__init__()
        self.model_name = model_name

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(f"{self.model_name}/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=f"{self.model_name}/google/umt5-xxl/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class Wan2_2_VAEWrapper(torch.nn.Module):
    def __init__(self, model_name="Wan2___2-TI2V-5B"):
        super().__init__()
        self.model_name = model_name
        self.mean = torch.tensor([
            -0.2289,-0.0052,-0.1323,-0.2339,-0.2799,0.0174,0.1838,0.1557,
            -0.1382,0.0542,0.2813,0.0891,0.1570,-0.0098,0.0375,-0.1825,
            -0.2246,-0.1207,-0.0698,0.5109,0.2665,-0.2108,-0.2158,0.2502,
            -0.2055,-0.0322,0.1109,0.1567,-0.0729,0.0899,-0.2799,-0.1230,
            -0.0313,-0.1649,0.0117,0.0723,-0.2839,-0.2083,-0.0520,0.3748,
            0.0152,0.1957,0.1433,-0.2944,0.3573,-0.0548,-0.1681,-0.0667,
        ], dtype=torch.float32)
        self.std = torch.tensor([
            0.4765,1.0364,0.4514,1.1677,0.5313,0.4990,0.4818,0.5013,
            0.8158,1.0344,0.5894,1.0901,0.6885,0.6165,0.8454,0.4978,
            0.5759,0.3523,0.7135,0.6804,0.5833,1.4146,0.8986,0.5659,
            0.7069,0.5338,0.4889,0.4917,0.4069,0.4999,0.6866,0.4093,
            0.5709,0.6065,0.6415,0.4944,0.5726,1.2042,0.5458,1.6887,
            0.3971,1.0600,0.3943,0.5537,0.5444,0.4089,0.7468,0.7744,
        ], dtype=torch.float32)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae2_2(
                pretrained_path=f"{self.model_name}/Wan2.2_VAE.pth",
                z_dim=48,
                dim=160,
                dim_mult=[1, 2, 4, 4],
                temperal_downsample=[False, True, True],
        ).eval().requires_grad_(False)

        self.dtype = torch.bfloat16

        self.vae_stride = (4, 8, 8)
        self.target_video_length = 81

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        self.scale = [_.to(device=device, dtype=dtype) for _ in self.scale]
        output = [
            self.model.encode(u.unsqueeze(0), self.scale).squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class Wan2_1_VAEWrapper(torch.nn.Module):
    def __init__(self, model_name="Wan2.1-T2V-14B"):
        super().__init__()
        self.model_name = model_name
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae2_1(
            pretrained_path=f"{self.model_name}/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)

        self.dtype = torch.bfloat16

        self.vae_stride = (4, 8, 8)
        self.target_video_length = 81

    def encode(self, pixel):
        device, dtype = pixel[0].device, self.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = [
            self.model.encode(u.to(self.dtype).unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        return output

    def run_vae_encoder(self, img):
        # img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
        img = img.to(torch.bfloat16).cuda()
        h, w = img.shape[1:]
        lat_h = h // self.vae_stride[1]
        lat_w = w // self.vae_stride[2]

        msk = torch.ones(
            1,
            self.target_video_length,
            lat_h,
            lat_w,
            device=torch.device("cuda"),
        )
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        vae_encode_out = self.encode(
            [
                torch.concat(
                    [
                        torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                        torch.zeros(3, self.target_video_length - 1, h, w),
                    ],
                    dim=1,
                ).cuda()
            ],
        )[0]
        vae_encode_out = torch.concat([msk, vae_encode_out]).to(torch.bfloat16)
        return [vae_encode_out]

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-14B",
            dual_exp=False,
            timestep_shift=8.0,
            uniform_timestep=False,
            local_attn_size=-1,
            sink_size=0
    ):
        super().__init__()
        self.model_name = model_name
        self.dual_exp = dual_exp

        if not self.dual_exp:
            self.model = WanModel.from_pretrained(f"{model_name}/")
            self.model.eval()
        else:
            self.model0 = WanModel.from_pretrained(f"{model_name}/")
            self.model1 = WanModel.from_pretrained(f"{model_name}/")
            self.model0.eval()
            self.model1.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = uniform_timestep

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        if not self.dual_exp:
            self.model.enable_gradient_checkpointing()
        else:
            self.model0.enable_gradient_checkpointing()
            self.model1.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        exp: str = None,
    ) -> Tuple[Any, Any]:
        prompt_embeds = conditional_dict["prompt_embeds"]
        if self.dual_exp and exp=='low_snr':
            model = self.model0
        elif self.dual_exp and exp=='high_snr':
            model = self.model1
        else:
            model = self.model

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        fp8_context = getattr(self, "fp8_context", None)
        fp8_autocast = fp8_context.context() if fp8_context is not None else nullcontext()
        # X0 prediction
        with fp8_autocast:
            flow_pred = model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=getattr(self, "seq_len", None),
                # clip_fea=clip_fea,
                y=y
            ).permute(0, 2, 1, 3, 4)

        # pred_x0 = self._convert_flow_pred_to_x0(
        #     flow_pred=flow_pred.flatten(0, 1),
        #     xt=noisy_image_or_video.flatten(0, 1),
        #     timestep=timestep.flatten(0, 1)
        # ).unflatten(0, flow_pred.shape[:2])

        return flow_pred#, pred_x0

    def get_scheduler(self) -> FlowMatchScheduler:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
