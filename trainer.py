import gc
import logging
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import wandb
from omegaconf import OmegaConf

from fp8 import FP8Context
from utils.distributed import EMA_FSDP, barrier, fsdp_state_dict, fsdp_wrap, launch_distributed_job
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
        self.disable_wandb = bool(getattr(config, "disable_wandb", False))
        self.output_path = config.logdir
        self.max_grad_norm = float(getattr(config, "max_grad_norm", 10.0))
        self.grad_accum = int(getattr(config, "gradient_accumulation_steps", 1))
        self.max_steps = getattr(config, "max_steps", None)
        self.previous_time = None

        self.fp8 = FP8Context(config)
        print(f"FP8 enabled: {self.fp8.enabled}")

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
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            min_num_params=int(getattr(config, "text_encoder_fsdp_min_num_params", 5e7)),
            cpu_offload=bool(getattr(config, "text_encoder_cpu_offload", False)),
        )
        self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype)
        self.model.vae.model.eval()
        self.model.vae.model.encode = torch.compile(self.model.vae.model.encode)
        # compile_mode = getattr(config, "compile_vae_mode", "reduce-overhead")
        # print("----Compiling VAE with torch.compile(mode=%s).", compile_mode)
        # self.model.vae.model = torch.compile(self.model.vae.model, mode=compile_mode)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

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
        self.ema_weight = config.get("ema_weight", -1.0)
        self.ema_start_step = config.get("ema_start_step", 0)
        self.generator_ema = None
        if (self.ema_weight > 0.0) and (self.step >= self.ema_start_step):
            print(f"Setting up EMA with weight {self.ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)

        self._maybe_resume()

        torch.cuda.empty_cache()

    def _init_logging(self):
        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
        barrier()

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=self.config.wandb_host, key=self.config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(self.config, resolve=True),
                name=self.config.config_name,
                mode="online",
                entity=self.config.wandb_entity,
                project=self.config.wandb_project,
                dir=self.config.wandb_save_dir,
            )
        else:
            self.tb_writer = None
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

    def _resume_dataloader(self, dataloader):
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

        # Load generator_ema checkpoint (if exists)
        generator_ema_path = os.path.join(self.config.resume_ckpt, "model_ema.pt")
        if os.path.exists(generator_ema_path):
            # Initialize EMA if not already initialized (needed for loading state)
            if self.generator_ema is None and self.ema_weight > 0.0:
                print("Initializing EMA for resume...")
                generator_state_dict = torch.load(generator_ema_path, map_location="cpu")
                generator_state_dict = {k.replace("_fsdp_wrapped_module.", ""). \
                                            replace("_checkpoint_wrapped_module.", ""). \
                                            replace("_orig_mod.", ""): v for k, v in generator_state_dict.items()}
                # FSDP will automatically handle dtype conversion
                self.model.generator.load_state_dict(generator_state_dict, strict=True)
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)
                print("Generator EMA checkpoint loaded successfully")
        else:
            print(f"Info: Generator EMA checkpoint not found at {generator_ema_path}")

        # Load generator checkpoint
        generator_path = os.path.join(self.config.resume_ckpt, "model.pt")
        if os.path.exists(generator_path):
            print(f"Loading generator from {generator_path}")
            generator_state_dict = torch.load(generator_path, map_location="cpu")
            # FSDP will automatically handle dtype conversion
            self.model.generator.load_state_dict(generator_state_dict["generator"], strict=True)
            self.step = int(generator_state_dict["generator"])
            print("Generator checkpoint loaded successfully")
        else:
            print(f"Warning: Generator checkpoint not found at {generator_path}")

        if self.step > 0:
            self._resume_dataloader(self.dataloader)

    def _build_dataloader(self):
        dataset = JsonlVideoDataset(
            jsonl_path=self.config.data_path,
            video_root=getattr(self.config, "video_root", None),
            num_frames=int(getattr(self.config, "num_frames", 81)),
            height=int(getattr(self.config, "height", 480)),
            width=int(getattr(self.config, "width", 832)),
            target_fps=int(getattr(self.config, "target_fps", None)),
            reader=getattr(self.config, "video_reader", "auto"),
            max_samples=getattr(self.config, "max_samples", None),
        )
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=int(getattr(self.config, "num_workers", 8)),
            prefetch_factor=4,         # Each worker keeps 2 batches ready (8 total)
            persistent_workers=True,   # Keeps worker processes alive across epochs
            pin_memory=True,            # Faster transfer from CPU RAM to GPU VRAM
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
        self.batches_per_epoch = len(dataloader)
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
            if (self.ema_weight > 0.0) and (self.ema_start_step < self.step):
                torch.save(self.generator_ema.state_dict(), os.path.join(ckpt_dir, "model_ema.pt"))
            print("EMA Model saved to", os.path.join(ckpt_dir, "model_ema.pt"))

    def train_one_step(self):
        # self.model.generator.train()
        # self.model.text_encoder.eval()
        # self.model.vae.eval()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        last_log_dict = None
        for micro_step in range(self.grad_accum):
            batch = next(self.dataloader)
            frames = batch["frames"].to(device=self.device, dtype=self.dtype, non_blocking=True)
            prompts = list(batch["prompts"])
            # if self.is_main_process:
            #     start = time.time()
            with torch.no_grad():
                clean_latent = self.model.encode_video(frames)
                # if self.is_main_process:
                #     torch.cuda.synchronize()
                #     print('----vae cost:', time.time()-start)
                #     start = time.time()
                conditional_dict = self.model.text_encoder(text_prompts=prompts)
                # if self.is_main_process:
                #     torch.cuda.synchronize()
                #     print('----t5 cost:', time.time()-start)
                #     start = time.time()
            
            sync_context = (
                self.model.generator.no_sync()
                if micro_step < self.grad_accum - 1 and hasattr(self.model.generator, "no_sync")
                else nullcontext()
            )
            with sync_context:
                loss, log_dict = self.model.rectified_flow_loss(clean_latent, conditional_dict, 
                                                                getattr(self.config, "task", None))
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

        # Create EMA params (if not already created)
        if (self.step >= self.ema_start_step) and \
                (self.generator_ema is None) and (self.ema_weight > 0):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)

        if self.is_main_process:
            timestep = last_log_dict["timestep"].float()
            wandb_loss_dict = {
                "train/loss": total_loss / self.grad_accum,
                "train/grad_norm": float(grad_norm.detach().item()),
                "train/timestep_mean": float(timestep.mean().item()),
                "train/timestep_min": float(timestep.min().item()),
                "train/timestep_max": float(timestep.max().item()),
            }
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)
            else:
                for k, v in wandb_loss_dict.items():
                    self.tb_writer.add_scalar(f"train/{k}", v, self.step)

        if getattr(self.config, "gc_interval", None) and self.step % int(getattr(self.config, "gc_interval", None)) == 0:
            if self.is_main_process:
                logging.info("DistGarbageCollector: Running GC.")
            torch.cuda.empty_cache()
            gc.collect()

    def train(self):
        while self.max_steps is None or self.step < self.max_steps:
            loss_dict = self.train_one_step()
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is not None:
                    if not self.disable_wandb:
                        wandb.log({"train/iter_time": current_time - self.previous_time}, step=self.step)
                    else:
                        self.tb_writer.add_scalar("perf/per_iteration_time", current_time - self.previous_time, self.step)
                    print(f"step {self.step} perf/per_iteration_time: ", current_time - self.previous_time, 
                          f'process: epoch {self.step / self.batches_per_epoch:.3f}')
                          # ', loss:', loss_dict['train/loss'])

                self.previous_time = current_time

        if (not self.config.no_save) and bool(getattr(self.config, "save_at_end", True)):
            self.save()

    def close(self):
        if getattr(self, "tb_writer", None) is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        if dist.is_initialized():
            barrier()
            dist.destroy_process_group()
