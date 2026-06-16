import argparse
import os
import sys

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import JsonlVideoDataset, worker_init_fn


def maybe_init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def build_dataset(config):
    audio_condition = getattr(config, "audio_condition", None)
    load_audio_emb = bool(getattr(audio_condition, "enabled", False)) if audio_condition is not None else False
    target_fps = getattr(config, "target_fps", None)
    target_fps = int(target_fps) if target_fps is not None else None
    return JsonlVideoDataset(
        jsonl_path=config.data_path,
        video_root=getattr(config, "video_root", None),
        num_frames=int(getattr(config, "num_frames", 81)),
        height=int(getattr(config, "height", 480)),
        width=int(getattr(config, "width", 832)),
        short_video_policy=getattr(config, "short_video_policy", "error"),
        target_fps=target_fps,
        reader=getattr(config, "video_reader", "auto"),
        max_samples=getattr(config, "max_samples", None),
        load_caption_emb=bool(getattr(config, "load_caption_emb", True)),
        caption_emb_key=getattr(config, "caption_emb_key", "caption_emb"),
        caption_emb_root=getattr(config, "caption_emb_root", None),
        default_caption_emb_path=getattr(config, "default_caption_emb_path", None),
        text_len=int(getattr(config, "text_len", 512)),
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


def tensor_meta(batch, key):
    value = batch.get(key)
    if value is None:
        return None
    return {
        "shape": tuple(value.shape),
        "dtype": str(value.dtype),
    }


def batch_meta(rank, step, batch):
    return {
        "rank": rank,
        "step": step,
        "idx": batch.get("idx").tolist() if torch.is_tensor(batch.get("idx")) else batch.get("idx"),
        "video_path": batch.get("video_path"),
        "frames": tensor_meta(batch, "frames"),
        "prompt_embeds": tensor_meta(batch, "prompt_embeds"),
        "audio_embeds": tensor_meta(batch, "audio_embeds"),
    }


def comparable(meta):
    return {
        "frames": meta["frames"],
        "prompt_embeds": meta["prompt_embeds"],
        "audio_embeds": meta["audio_embeds"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    rank, world_size, _ = maybe_init_distributed()
    config = OmegaConf.load(args.config_path)
    dataset = build_dataset(config)
    batch_size = args.batch_size if args.batch_size is not None else int(getattr(config, "batch_size", 1))

    shuffle = not args.no_shuffle
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
    ) if world_size > 1 else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else shuffle,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    if rank == 0:
        per_rank = len(sampler) if sampler is not None else len(dataset)
        print(
            f"dataset_len={len(dataset)} world_size={world_size} "
            f"per_rank_samples={per_rank} dataloader_len={len(dataloader)} "
            f"batch_size={batch_size}",
            flush=True,
        )

    for step, batch in enumerate(dataloader, start=1):
        meta = batch_meta(rank, step, batch)
        gathered = [None for _ in range(world_size)]
        if world_size > 1:
            dist.all_gather_object(gathered, meta)
        else:
            gathered[0] = meta

        if rank == 0:
            expected = comparable(gathered[0])
            mismatches = [item for item in gathered if comparable(item) != expected]
            if mismatches:
                print(f"shape mismatch at dataloader step {step}", flush=True)
                for item in gathered:
                    print(item, flush=True)
                raise SystemExit(2)
            if step == 1 or step % 20 == 0:
                print(f"step={step} shapes={expected}", flush=True)

        if step >= args.steps:
            break

    if rank == 0:
        print(f"checked_steps={min(args.steps, len(dataloader))} status=ok", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
