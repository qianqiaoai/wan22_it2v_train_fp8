import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import JsonlVideoDataset
from scripts.check_dataloader_shapes import build_dataset


def shape_of(value):
    if torch.is_tensor(value):
        return tuple(value.shape), str(value.dtype)
    return None, type(value).__name__


def validate_sample(sample, config):
    expected = {
        "frames": (3, int(config.num_frames), int(config.height), int(config.width)),
        "prompt_embeds": (int(config.text_len), 4096),
        "audio_embeds": (int(config.num_frames), 12, 768),
    }
    for key, expected_shape in expected.items():
        actual_shape, actual_dtype = shape_of(sample.get(key))
        if actual_shape != expected_shape:
            raise ValueError(
                f"{key} has shape={actual_shape} dtype={actual_dtype}, expected shape={expected_shape}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--world-size", type=int, default=48)
    parser.add_argument("--next-step", type=int, required=True, help="Optimizer step that would be printed if this train step completes.")
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None, help="Only inspect one global rank.")
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    grad_accum = args.grad_accum if args.grad_accum is not None else int(getattr(config, "gradient_accumulation_steps", 1))
    start_pos = (int(args.next_step) - 1) * grad_accum
    ranks = [args.rank] if args.rank is not None else list(range(args.world_size))
    shuffle = not args.no_shuffle

    dataset = build_dataset(config)
    print(
        f"dataset_len={len(dataset)} world_size={args.world_size} "
        f"per_rank_samples={len(dataset) // args.world_size} "
        f"next_step={args.next_step} grad_accum={grad_accum} "
        f"dataloader_positions={start_pos}..{start_pos + grad_accum - 1}",
        flush=True,
    )

    failures = []
    slowest = []
    for rank in ranks:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )
        indices = list(iter(sampler))
        for micro_step in range(grad_accum):
            pos = start_pos + micro_step
            if pos >= len(indices):
                failures.append((rank, micro_step, pos, None, "position out of range"))
                continue
            sample_idx = int(indices[pos])
            started = time.time()
            try:
                sample = dataset[sample_idx]
                validate_sample(sample, config)
            except Exception as exc:
                failures.append((rank, micro_step, pos, sample_idx, repr(exc)))
                print(
                    f"FAIL rank={rank} micro={micro_step} pos={pos} "
                    f"sample_idx={sample_idx} error={exc!r}",
                    flush=True,
                )
                continue

            elapsed = time.time() - started
            slowest.append((elapsed, rank, micro_step, pos, sample_idx, sample.get("video_path"), sample.get("audio_path")))
            print(
                f"OK rank={rank:02d} micro={micro_step} pos={pos} sample_idx={sample_idx} "
                f"time={elapsed:.2f}s video={sample.get('video_path')} audio={sample.get('audio_path')}",
                flush=True,
            )

    slowest.sort(reverse=True)
    print("slowest_samples:", flush=True)
    for elapsed, rank, micro_step, pos, sample_idx, video_path, audio_path in slowest[:10]:
        print(
            f"time={elapsed:.2f}s rank={rank} micro={micro_step} pos={pos} "
            f"sample_idx={sample_idx} video={video_path} audio={audio_path}",
            flush=True,
        )

    if failures:
        print(f"status=failed failures={len(failures)}", flush=True)
        raise SystemExit(2)
    print(f"status=ok checked={len(slowest)}", flush=True)


if __name__ == "__main__":
    main()
