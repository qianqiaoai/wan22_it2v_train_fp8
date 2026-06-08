import argparse
import os
import sys
import time
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.wan_wrapper import WanDiffusionWrapper


def log(message):
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(f"[rank{rank}/local{local_rank}] debug_fsdp_init: {message}", flush=True)


def timed(label):
    class _Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            log(f"{label}: enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                log(f"{label}: complete in {time.perf_counter() - self.start:.1f}s")
            else:
                log(f"{label}: failed after {time.perf_counter() - self.start:.1f}s {exc_type.__name__}: {exc}")

    return _Timer()


def init_dist():
    local_rank = int(os.environ["LOCAL_RANK"])
    with timed(f"set cuda device local_rank={local_rank}"):
        torch.cuda.set_device(local_rank)
    with timed("init_process_group"):
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    log(
        "distributed ready "
        f"rank={dist.get_rank()} world_size={dist.get_world_size()} "
        f"device={torch.cuda.current_device()}"
    )


def barrier(label):
    with timed(f"barrier {label}"):
        dist.barrier()


def nccl_smoke():
    with timed("NCCL all_reduce smoke"):
        tensor = torch.ones(1, device=torch.cuda.current_device()) * (dist.get_rank() + 1)
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        log(f"NCCL all_reduce result={tensor.item()}")


def make_tiny_model():
    with timed("create tiny CPU model"):
        return nn.Sequential(
            nn.Linear(4096, 4096, bias=False),
            nn.GELU(),
            nn.Linear(4096, 4096, bias=False),
        )


def make_real_generator(config):
    audio_condition = getattr(config, "audio_condition", None)
    model_name = getattr(config, "generator_name", getattr(config, "model_name", "Wan2.1-T2V-1.3B"))
    with timed(f"create WanDiffusionWrapper from {model_name}"):
        generator = WanDiffusionWrapper(
            **getattr(config, "model_kwargs", {}),
            model_name=model_name,
            audio_condition=audio_condition,
        )
    with timed("set generator trainable"):
        if getattr(generator, "dual_exp", False):
            generator.model0.requires_grad_(True)
            generator.model1.requires_grad_(True)
        else:
            generator.model.requires_grad_(True)
    if bool(getattr(config, "gradient_checkpointing", False)):
        with timed("enable gradient checkpointing"):
            generator.enable_gradient_checkpointing()
    return generator


def tensor_mib(tensor):
    return tensor.numel() * tensor.element_size() / (1024 ** 2)


def move_parameter_tree(module, device, dtype=None, max_items=None):
    count = 0
    for name, param in module.named_parameters(recurse=True):
        count += 1
        if max_items is not None and count > max_items:
            log(f"parameter move reached max_items={max_items}, stopping")
            return
        with timed(
            "move parameter "
            f"{name} shape={tuple(param.shape)} dtype={param.dtype} mib={tensor_mib(param):.1f}"
        ):
            moved = param.detach().to(device=device, dtype=dtype or param.dtype)
            torch.cuda.synchronize()
            del moved
    log(f"parameter move complete count={count}")


def move_top_level_modules(module, device, dtype=None):
    children = list(module.named_children())
    if not children:
        log("module has no top-level children; using direct module.to")
        with timed("module.to(cuda) no children"):
            module.to(device=device, dtype=dtype)
            torch.cuda.synchronize()
        return

    for name, child in children:
        param_count = sum(p.numel() for p in child.parameters(recurse=True))
        param_mib = sum(tensor_mib(p) for p in child.parameters(recurse=True))
        with timed(
            "move top-level child "
            f"{name} type={child.__class__.__name__} params={param_count} cpu_mib={param_mib:.1f}"
        ):
            child.to(device=device, dtype=dtype)
            torch.cuda.synchronize()
    log(f"top-level child move complete count={len(children)}")


def wrap_fsdp(module, sharding_strategy, mixed_precision, wrap_policy, min_num_params):
    strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mp_policy = None

    if wrap_policy == "none":
        auto_wrap_policy = None
    elif wrap_policy == "size":
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
    else:
        raise ValueError(f"Unsupported wrap_policy: {wrap_policy}")

    with timed(
        "FSDP constructor "
        f"sharding={sharding_strategy} wrap_policy={wrap_policy} "
        f"min_num_params={min_num_params} mixed_precision={mixed_precision}"
    ):
        wrapped = FSDP(
            module,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=strategy,
            mixed_precision=mp_policy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
            sync_module_states=False,
        )
    return wrapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs/rf_final.yaml")
    parser.add_argument(
        "--mode",
        choices=(
            "nccl",
            "tiny-fsdp",
            "real-to",
            "real-to-bf16",
            "real-to-children",
            "real-to-params",
            "real-fsdp-none",
            "real-fsdp-size",
        ),
        default="real-fsdp-none",
    )
    parser.add_argument("--sharding_strategy", default="full", choices=("full", "hybrid_full", "hybrid_zero2", "no_shard"))
    parser.add_argument("--min_num_params", type=int, default=100000000000)
    parser.add_argument("--max_items", type=int, default=None)
    args = parser.parse_args()

    log(f"script enter mode={args.mode}")
    init_dist()
    nccl_smoke()
    barrier("after nccl")

    config = OmegaConf.load(args.config_path)

    if args.mode == "nccl":
        barrier("final")
        log("script complete")
        return

    if args.mode == "tiny-fsdp":
        module = make_tiny_model()
        module = wrap_fsdp(
            module,
            sharding_strategy=args.sharding_strategy,
            mixed_precision=bool(getattr(config, "mixed_precision", False)),
            wrap_policy="none",
            min_num_params=args.min_num_params,
        )
        barrier("after tiny fsdp")
        log(f"tiny fsdp module ready type={module.__class__.__name__}")
        return

    module = make_real_generator(config)
    barrier("after real generator create")

    device = torch.device("cuda", torch.cuda.current_device())
    if args.mode == "real-to":
        with timed(f"plain generator.to({device})"):
            module = module.to(device=device)
            torch.cuda.synchronize()
        barrier("after real generator to cuda")
        log(f"real generator to cuda ready type={module.__class__.__name__}")
        return

    if args.mode == "real-to-bf16":
        with timed(f"plain generator.to({device}, dtype=torch.bfloat16)"):
            module = module.to(device=device, dtype=torch.bfloat16)
            torch.cuda.synchronize()
        barrier("after real generator to cuda bf16")
        log(f"real generator to cuda bf16 ready type={module.__class__.__name__}")
        return

    if args.mode == "real-to-children":
        move_top_level_modules(module, device=device, dtype=None)
        barrier("after real generator top-level child moves")
        log(f"real generator top-level child moves ready type={module.__class__.__name__}")
        return

    if args.mode == "real-to-params":
        move_parameter_tree(module, device=device, dtype=None, max_items=args.max_items)
        barrier("after real generator parameter moves")
        log(f"real generator parameter moves ready type={module.__class__.__name__}")
        return

    wrap_policy = "none" if args.mode == "real-fsdp-none" else "size"
    module = wrap_fsdp(
        module,
        sharding_strategy=args.sharding_strategy,
        mixed_precision=bool(getattr(config, "mixed_precision", False)),
        wrap_policy=wrap_policy,
        min_num_params=args.min_num_params,
    )
    barrier("after real fsdp")
    log(f"real fsdp module ready type={module.__class__.__name__}")


if __name__ == "__main__":
    main()
