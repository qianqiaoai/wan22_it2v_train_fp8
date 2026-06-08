#!/usr/bin/env python3
"""Encode and save the official Wan2.2 negative prompt embedding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BASE_MODEL_DIR = Path("/mnt/data/nlp/user/qiaoqian/models/Wan2.2-TI2V-5B")
DEFAULT_OUTPUT = PROJECT_ROOT / "raw_dataset/text_emb/wan2_2_negative_prompt_umt5_xxl_bf16.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def default_negative_prompt() -> str:
    try:
        from wan2.configs.shared_config import wan_shared_cfg

        return str(wan_shared_cfg.sample_neg_prompt)
    except Exception:
        return (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )


def main() -> None:
    args = parse_args()
    prompt = args.negative_prompt if args.negative_prompt is not None else default_negative_prompt()

    from utils.wan_wrapper import WanTextEncoder

    device_name = args.device
    if device_name == "cuda":
        device_name = "cuda:0"
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    encoder = WanTextEncoder(model_name=str(args.base_model_dir)).eval()
    encoder.text_encoder.to(device)
    with torch.no_grad():
        prompt_embeds = encoder([prompt])["prompt_embeds"].detach().cpu().to(torch.bfloat16)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "caption_emb": prompt_embeds.squeeze(0),
            "prompt_embeds": prompt_embeds,
            "prompt": prompt,
            "source": "Wan2.2 sample_neg_prompt",
        },
        args.output_path,
    )
    print(f"saved {args.output_path}")
    print(f"shape {tuple(prompt_embeds.shape)} dtype {prompt_embeds.dtype}")


if __name__ == "__main__":
    main()
