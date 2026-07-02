import gc
import logging
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import swanlab
from omegaconf import OmegaConf

from fp8 import FP8Context
from utils.distributed import barrier, fsdp_state_dict, fsdp_wrap, launch_distributed_job
from utils.misc import set_seed

from dataset import JsonlVideoDataset, cycle, worker_init_fn
from model import RectifiedFlowFineTuneModel


class RectifiedFlowTrainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.is_main_process = self.global_rank == 0
        self.disable_swanlab = bool(getattr(config, "disable_swanlab", False))
        self.output_path = config.logdir
        self.max_grad_norm = float(getattr(config, "max_grad_norm", 10.0))
        self.grad_accum = int(getattr(config, "gradient_accumulation_steps", 1))
        self.max_steps = getattr(config, "max_steps", None)

        self.fp8 = FP8Context(config)

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = int(random_seed.item())
        set_seed(config.seed + self.global_rank)

        self._init_logging()
        self.model = RectifiedFlowFineTuneModel(config, device=self.device)
        self._attach_fp8(self.model.generator)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            min_num_params=int(getattr(config, "generator_fsdp_min_num_params", 5e7)),
        )
        if self.model.text_encoder is not None:
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
                min_num_params=int(getattr(config, "text_encoder_fsdp_min_num_params", 5e7)),
                cpu_offload=bool(getattr(config, "text_encoder_cpu_offload", False)),
        )

        self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype)
        if hasattr(self.model.vae, "model"):
            self.model.vae.model.eval()
            self.model.vae.model.encode = torch.compile(self.model.vae.model.encode)
        else:
            self.model.vae.eval()
        # compile_mode = getattr(config, "compile_vae_mode", "reduce-overhead")
        # print("----Compiling VAE with torch.compile(mode=%s).", compile_mode)
        # self.model.vae.model = torch.compile(self.model.vae.model, mode=compile_mode)

        trainable_params = [p for p in self.model.generator.parameters() if p.requires_grad]
        optimizer_type = str(getattr(config, "optimizer_type", "adamw")).lower()
        if optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay,
            )
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                trainable_params,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=float(getattr(config, "sgd_momentum", 0.0)),
                foreach=False,
            )
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")
        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]

        self.dataloader = self._build_dataloader()

        ##############################################################################################################
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        # EMA is disabled for this run; regular generator checkpoints are enough.

        self._maybe_resume()
        torch.cuda.empty_cache()

    def _lr_factor_for_step(self, step):
        scheduler = str(getattr(self.config, "lr_scheduler", "constant")).lower()
        warmup_steps = int(getattr(self.config, "warmup_steps", 0))
        min_lr_ratio = float(getattr(self.config, "min_lr_ratio", 0.0))
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

        if warmup_steps > 0 and step <= warmup_steps:
            return max(float(step), 1.0) / float(warmup_steps)

        if scheduler in {"constant", "none"}:
            return 1.0
        if scheduler != "cosine":
            raise ValueError(f"Unsupported lr_scheduler: {scheduler}")

        decay_steps = int(getattr(self.config, "lr_decay_steps", self.max_steps or warmup_steps))
        if decay_steps <= warmup_steps:
            return 1.0
        progress = (float(step) - warmup_steps) / float(decay_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    def _set_optimizer_lr_for_step(self, step):
        factor = self._lr_factor_for_step(step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def _init_logging(self):
        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
        barrier()

        self.tb_writer = None
        if self.is_main_process and not self.disable_swanlab:
            api_key = str(getattr(self.config, "swanlab_api_key", "") or "")
            if not api_key:
                api_key = os.environ.get(str(getattr(self.config, "swanlab_api_key_env", "SWANLAB_API_KEY")), "")
            if api_key:
                swanlab.login(api_key=api_key, save=bool(getattr(self.config, "swanlab_save_api_key", False)))
            logged_config = OmegaConf.to_container(self.config, resolve=True)
            if "swanlab_api_key" in logged_config:
                logged_config["swanlab_api_key"] = "***"
            swanlab.init(
                config=logged_config,
                experiment_name=getattr(self.config, "swanlab_experiment_name", self.config.config_name),
                mode=getattr(self.config, "swanlab_mode", "cloud"),
                project=getattr(self.config, "swanlab_project", "wan2.2_5B_fp8"),
                workspace=getattr(self.config, "swanlab_workspace", None),
                logdir=getattr(self.config, "swanlab_logdir", None) or self.output_path,
            )
        else:
            if self.is_main_process:
                import socket
                from datetime import datetime
                hostname = socket.gethostname()
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                tb_dir = getattr(self.config, "tensorboard_logdir", None)
                if tb_dir is None:
                    tb_dir = os.path.join(self.config.logdir, "tensorboard", f"{hostname}_{run_id}")
                os.makedirs(tb_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                self.tb_writer.add_text("config", f"<pre>{OmegaConf.to_yaml(self.config)}</pre>", global_step=0)

    def _attach_fp8(self, generator):
        target = generator.module if hasattr(generator, "module") else generator
        target.fp8_context = self.fp8
        if getattr(target, "dual_exp", False):
            target.model0.fp8_context = self.fp8
            target.model1.fp8_context = self.fp8
        else:
            target.model.fp8_context = self.fp8

    def _num_consumed_dataloader_batches(self, optimizer_step):
        return int(optimizer_step) * self.grad_accum

    def _resume_dataloader(self, dataloader=None):
        if dataloader is None:
            dataloader = getattr(self, "train_dataloader", None)
        if dataloader is None:
            raise ValueError("Cannot resume dataloader before it has been built.")
        batches_per_epoch = len(dataloader)
        if batches_per_epoch <= 0:
            raise ValueError("Cannot resume dataloader because it has no batches.")

        consumed_batches = self._num_consumed_dataloader_batches(self.step)
        resume_epoch, batches_to_skip = divmod(consumed_batches, batches_per_epoch)
        self.dataloader = cycle(dataloader, start_epoch=resume_epoch)

        for _ in range(batches_to_skip):
            next(self.dataloader)

        if self.is_main_process:
            print(
                "Dataloader resumed: "
                f"consumed_batches={consumed_batches}, "
                f"epoch={resume_epoch}, "
                f"batch_offset={batches_to_skip}"
            )

    def _maybe_resume(self):
        resume_ckpt = getattr(self.config, "resume_ckpt", None)
        if not resume_ckpt:
            return

        # EMA checkpoints are intentionally ignored in this training setup.

        # Load generator checkpoint
        generator_path = os.path.join(self.config.resume_ckpt, "model.pt")
        if os.path.exists(generator_path):
            print(f"Loading generator from {generator_path}")
            generator_state_dict = torch.load(generator_path, map_location="cpu")
            # FSDP will automatically handle dtype conversion
            self.model.load_generator_state_dict_for_current_vae(generator_state_dict["generator"], strict=True)
            self.step = int(generator_state_dict.get("step", 0))
            print("Generator checkpoint loaded successfully")
        else:
            print(f"Warning: Generator checkpoint not found at {generator_path}")

        if self.step > 0:
            self._resume_dataloader()

    def _build_dataloader(self):
        audio_condition = getattr(self.config, "audio_condition", None)
        load_audio_emb = bool(getattr(audio_condition, "enabled", False)) if audio_condition is not None else False
        target_fps = getattr(self.config, "target_fps", None)
        target_fps = int(target_fps) if target_fps is not None else None
        dataset = JsonlVideoDataset(
            jsonl_path=self.config.data_path,
            video_root=getattr(self.config, "video_root", None),
            num_frames=int(getattr(self.config, "num_frames", 81)),
            height=int(getattr(self.config, "height", 480)),
            width=int(getattr(self.config, "width", 832)),
            short_video_policy=getattr(self.config, "short_video_policy", "error"),
            target_fps=target_fps,
            reader=getattr(self.config, "video_reader", "auto"),
            max_samples=getattr(self.config, "max_samples", None),
            load_caption_emb=bool(getattr(self.config, "load_caption_emb", True)),
            caption_emb_key=getattr(self.config, "caption_emb_key", "caption_emb"),
            caption_emb_root=getattr(self.config, "caption_emb_root", None),
            default_caption_emb_path=getattr(self.config, "default_caption_emb_path", None),
            caption_emb_mode=getattr(self.config, "caption_emb_mode", "fallback"),
            text_len=int(getattr(self.config, "text_len", 512)),
            load_audio_emb=load_audio_emb,
            audio_emb_mode=getattr(audio_condition, "audio_emb_mode", "offline") if audio_condition is not None else "offline",
            audio_emb_key=getattr(audio_condition, "audio_emb_key", "vocals_emb_base_all") if audio_condition is not None else "vocals_emb_base_all",
            audio_emb_root=getattr(audio_condition, "audio_emb_root", None) if audio_condition is not None else None,
            audio_path_key=getattr(audio_condition, "audio_path_key", "audio_path") if audio_condition is not None else "audio_path",
            audio_path_root=getattr(audio_condition, "audio_path_root", None) if audio_condition is not None else None,
            audio_model_name_or_path=getattr(audio_condition, "audio_model_name_or_path", "TencentGameMate/chinese-wav2vec2-base") if audio_condition is not None else "TencentGameMate/chinese-wav2vec2-base",
            audio_sample_rate=int(getattr(audio_condition, "audio_sample_rate", 16000)) if audio_condition is not None else 16000,
            audio_encoder_device=getattr(audio_condition, "audio_encoder_device", "cpu") if audio_condition is not None else "cpu",
            audio_model_local_files_only=bool(getattr(audio_condition, "audio_model_local_files_only", False)) if audio_condition is not None else False,
            audio_only_last_features=bool(getattr(audio_condition, "audio_only_last_features", False)) if audio_condition is not None else False,
        )
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        num_workers = int(getattr(self.config, "num_workers", 8))
        dataloader_kwargs = dict(
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        if num_workers > 0:
            dataloader_kwargs.update(
                prefetch_factor=4,
                persistent_workers=True,
                worker_init_fn=worker_init_fn,
            )
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        self.batches_per_epoch = len(dataloader)
        self.train_dataloader = dataloader
        if self.is_main_process:
            print(f"DATASET SIZE {self.batches_per_epoch}")
        return cycle(dataloader)

    def save(self):
        if self.is_main_process:
            print("Start gathering distributed model states...")
        state = {
            "step": self.step,
            "generator": fsdp_state_dict(self.model.generator),
        }
        if self.is_main_process:
            # state["optimizer"] = self.optimizer.state_dict()
            ckpt_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(state, os.path.join(ckpt_dir, "model.pt"))
            OmegaConf.save(self.config, os.path.join(ckpt_dir, "config.yaml"))
            print("Model saved to", os.path.join(ckpt_dir, "model.pt"))

    def train_one_step(self):
        # self.model.generator.train()
        # self.model.vae.eval()
        self._set_optimizer_lr_for_step(self.step + 1)
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        last_log_dict = None
        metric_dict = None
        for micro_step in range(self.grad_accum):
            batch = next(self.dataloader)
            frames = batch["frames"].to(device=self.device, dtype=self.dtype, non_blocking=True)
            prompt_embeds = batch["prompt_embeds"].to(device=self.device, dtype=self.dtype, non_blocking=True)
            audio_embeds = batch.get("audio_embeds")
            if audio_embeds is not None:
                audio_embeds = audio_embeds.to(device=self.device, dtype=self.dtype, non_blocking=True)
            # if self.is_main_process:
            #     start = time.time()
            with torch.no_grad():
                clean_latent = self.model.encode_video(frames)
                # if self.is_main_process:
                #     torch.cuda.synchronize()
                #     print('----vae cost:', time.time()-start)
                #     start = time.time()
                conditional_dict = {"prompt_embeds": prompt_embeds}
            
            sync_context = (
                self.model.generator.no_sync()
                if micro_step < self.grad_accum - 1 and hasattr(self.model.generator, "no_sync")
                else nullcontext()
            )
            with sync_context:
                loss, log_dict = self.model.rectified_flow_loss(clean_latent, conditional_dict,
                                                                getattr(self.config, "task", None),
                                                                audio_embeds=audio_embeds)
                (loss / self.grad_accum).backward()
                # if self.is_main_process:
                #     torch.cuda.synchronize()
                #     print('----rf cost:', time.time()-start)
                #     start = time.time()

            total_loss += float(loss.detach().item())
            last_log_dict = log_dict

        grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        self.optimizer.step()
        self.step += 1

        if self.is_main_process:
            timestep = last_log_dict["timestep"].float()
            metric_dict = {
                "train/loss": total_loss / self.grad_accum,
                "train/grad_norm": float(grad_norm.detach().item()),
                "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                "train/timestep_mean": float(timestep.mean().item()),
                "train/timestep_min": float(timestep.min().item()),
                "train/timestep_max": float(timestep.max().item()),
            }
            if not self.disable_swanlab:
                swanlab.log(metric_dict, step=self.step)
            else:
                for k, v in metric_dict.items():
                    self.tb_writer.add_scalar(f"train/{k}", v, self.step)

        if getattr(self.config, "gc_interval", None) and self.step % int(getattr(self.config, "gc_interval", None)) == 0:
            if self.is_main_process:
                logging.info("DistGarbageCollector: Running GC.")
            torch.cuda.empty_cache()
            gc.collect()

        return metric_dict

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
        if minutes:
            return f"{minutes:d}m{seconds:02d}s"
        return f"{seconds:d}s"

    def _format_train_log(self, metrics, iter_time):
        step_text = str(self.step)

        if self.batches_per_epoch:
            epoch_idx, epoch_step = divmod(self.step, self.batches_per_epoch)
            epoch_text = str(epoch_idx)
            epoch_step_text = f"{epoch_step}/{self.batches_per_epoch}"
        else:
            epoch_text = "--"
            epoch_step_text = "--"
        total_epochs_text = "--"
        if self.max_steps is not None and self.batches_per_epoch:
            total_epochs = int(self.max_steps) / self.batches_per_epoch
            total_epochs_text = f"{total_epochs:.2f}"

        samples_per_iter = self.world_size * int(getattr(self.config, "batch_size", 1)) * self.grad_accum
        samples_per_sec = samples_per_iter / iter_time if iter_time > 0 else 0.0
        lr = self.optimizer.param_groups[0]["lr"]

        return (
            f"epoch={epoch_text} "
            f"total_epochs={total_epochs_text} "
            f"step={step_text} "
            f"epoch_step={epoch_step_text} "
            f"loss={metrics['train/loss']:.6f} "
            f"grad_norm={metrics['train/grad_norm']:.4f} "
            f"lr={lr:.2e} "
            f"timestep={metrics['train/timestep_mean']:.1f}"
            f"[{metrics['train/timestep_min']:.1f},{metrics['train/timestep_max']:.1f}] "
            f"iter={iter_time:.2f}s "
            f"samples/s={samples_per_sec:.3f}"
        )

    def train(self):
        while self.max_steps is None or self.step < self.max_steps:
            step_start_time = time.time()
            metric_dict = self.train_one_step()
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                iter_time = time.time() - step_start_time
                if not self.disable_swanlab:
                    swanlab.log({"train/iter_time": iter_time}, step=self.step)
                else:
                    self.tb_writer.add_scalar("perf/per_iteration_time", iter_time, self.step)
                if metric_dict is not None:
                    print(self._format_train_log(metric_dict, iter_time), flush=True)

        if (not self.config.no_save) and bool(getattr(self.config, "save_at_end", True)):
            self.save()

    def close(self):
        if self.is_main_process and not self.disable_swanlab:
            swanlab.finish()
        if getattr(self, "tb_writer", None) is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        if dist.is_initialized():
            barrier()
            dist.destroy_process_group()
