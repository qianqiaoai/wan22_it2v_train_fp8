#!/usr/bin/env python3
"""Audio-conditioned Wan2.2 TI2V inference for the locally fine-tuned model."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import JsonlVideoDataset, OnlineWav2VecAudioExtractor
from wan2.utils.utils import masks_like, merge_video_audio, save_video

if TYPE_CHECKING:
    from model import RectifiedFlowFineTuneModel


DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "logs/wan2.2_5B_fp8_audio_20260606_120055"
DEFAULT_INFERENCE_DIR = PROJECT_ROOT / "inference_data"
DEFAULT_TEXT_EMB = PROJECT_ROOT / "raw_dataset/text_emb/person_speaking_zh_umt5_xxl_bf16.pt"
DEFAULT_NEGATIVE_TEXT_EMB = PROJECT_ROOT / "raw_dataset/text_emb/wan2_2_negative_prompt_umt5_xxl_bf16.pt"
DEFAULT_BASE_MODEL_DIR = Path("/mnt/data/nlp/user/qiaoqian/models/Wan2.2-TI2V-5B")


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
TEXT_KEYS = ("caption_embeds", "caption_embed", "prompt_embeds", "context", "hidden_states", "last_hidden_state")

DEFAULT_ARGS: Dict[str, Any] = {
    "checkpoint_dir": DEFAULT_CHECKPOINT_DIR,
    "checkpoint_path": None,
    "base_model_dir": None,
    "inference_dir": DEFAULT_INFERENCE_DIR,
    "audio_dir": None,
    "audio_source_path": None,
    "job_manifest": None,
    "audio_segment_mode": "per_job",
    "audio_segment_start": 0.0,
    "audio_segment_stride": None,
    "skip_incomplete_audio_segments": True,
    "ref_dir": None,
    "prompt_file": None,
    "text_emb_path": DEFAULT_TEXT_EMB,
    "negative_text_emb_path": DEFAULT_NEGATIVE_TEXT_EMB,
    "output_dir": None,
    "width": 480,
    "height": 832,
    "frame_num": 81,
    "fps": 20,
    "sample_steps": 50,
    "shift": None,
    "guide_scale": 1.0,
    "cfg_mode": "off",
    "seed": 42,
    "max_jobs": None,
    "device": "cuda",
    "dtype": "bf16",
    "pairing": "match",
    "merge_audio": True,
    "fp8": True,
    "dry_run": False,
    "local_files_only": True,
    "allow_remote_audio_model": False,
}

CONFIG_ALIASES = {
    "checkpoint.checkpoint_dir": "checkpoint_dir",
    "checkpoint.checkpoint_path": "checkpoint_path",
    "checkpoint.base_model_dir": "base_model_dir",
    "model.base_model_dir": "base_model_dir",
    "model.fp8": "fp8",
    "data.inference_dir": "inference_dir",
    "data.audio_dir": "audio_dir",
    "data.audio_source_path": "audio_source_path",
    "data.job_manifest": "job_manifest",
    "data.ref_dir": "ref_dir",
    "data.prompt_file": "prompt_file",
    "data.text_emb_path": "text_emb_path",
    "data.negative_text_emb_path": "negative_text_emb_path",
    "output.output_dir": "output_dir",
    "generation.width": "width",
    "generation.height": "height",
    "generation.frame_num": "frame_num",
    "generation.fps": "fps",
    "generation.sample_steps": "sample_steps",
    "generation.shift": "shift",
    "generation.guide_scale": "guide_scale",
    "generation.cfg_mode": "cfg_mode",
    "generation.seed": "seed",
    "generation.pairing": "pairing",
    "generation.merge_audio": "merge_audio",
    "guidance.guide_scale": "guide_scale",
    "guidance.cfg_mode": "cfg_mode",
    "guidance.negative_text_emb_path": "negative_text_emb_path",
    "audio.source_path": "audio_source_path",
    "audio.segment_mode": "audio_segment_mode",
    "audio.start_time": "audio_segment_start",
    "audio.stride": "audio_segment_stride",
    "audio.skip_incomplete_segments": "skip_incomplete_audio_segments",
    "runtime.max_jobs": "max_jobs",
    "runtime.device": "device",
    "runtime.dtype": "dtype",
    "runtime.dry_run": "dry_run",
    "runtime.local_files_only": "local_files_only",
    "runtime.allow_remote_audio_model": "allow_remote_audio_model",
}
PATH_KEYS = {
    "checkpoint_dir",
    "checkpoint_path",
    "base_model_dir",
    "inference_dir",
    "audio_dir",
    "audio_source_path",
    "job_manifest",
    "ref_dir",
    "prompt_file",
    "text_emb_path",
    "negative_text_emb_path",
    "output_dir",
}


def _flatten_config(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_config(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def _coerce_config_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in PATH_KEYS:
        return Path(str(value))
    return value


def load_infer_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Inference config does not exist: {config_path}")

    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"Inference config must be a mapping: {config_path}")

    values: Dict[str, Any] = {}
    for raw_key, raw_value in _flatten_config(raw).items():
        key = CONFIG_ALIASES.get(raw_key, raw_key)
        if key not in DEFAULT_ARGS:
            raise KeyError(f"Unsupported inference config key: {raw_key}")
        values[key] = _coerce_config_value(key, raw_value)
    return values


def _set_parser_defaults(parser: argparse.ArgumentParser, defaults: Dict[str, Any]) -> None:
    parser.set_defaults(**{key: value for key, value in defaults.items() if value is not None})


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config_path", type=Path, default=None)
    pre_args, remaining = pre_parser.parse_known_args()

    defaults = dict(DEFAULT_ARGS)
    defaults.update(load_infer_config(pre_args.config_path))

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--base_model_dir", type=Path)
    parser.add_argument("--inference_dir", type=Path)
    parser.add_argument("--audio_dir", type=Path)
    parser.add_argument("--audio_source_path", type=Path)
    parser.add_argument(
        "--job_manifest",
        type=Path,
        help="Optional key-preserving materialized benchmark manifest. Overrides audio/ref directory pairing.",
    )
    parser.add_argument("--audio_segment_mode", choices=("per_job", "sequential", "fixed"))
    parser.add_argument("--audio_segment_start", type=float)
    parser.add_argument("--audio_segment_stride", type=float)
    parser.add_argument("--skip_incomplete_audio_segments", action=argparse.BooleanOptionalAction)
    parser.add_argument("--ref_dir", type=Path)
    parser.add_argument("--prompt_file", type=Path)
    parser.add_argument("--text_emb_path", type=Path)
    parser.add_argument("--negative_text_emb_path", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--frame_num", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--sample_steps", type=int)
    parser.add_argument("--shift", type=float)
    parser.add_argument("--guide_scale", type=float)
    parser.add_argument("--cfg_mode", choices=("off", "text", "audio", "both", "zero"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_jobs", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--dtype", choices=("bf16", "fp32"))
    parser.add_argument("--pairing", choices=("match", "cartesian"))
    parser.add_argument("--merge_audio", action=argparse.BooleanOptionalAction)
    parser.add_argument("--no_merge_audio", dest="merge_audio", action="store_false")
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction)
    parser.add_argument("--disable_fp8", dest="fp8", action="store_false")
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, help="Validate inputs and pairing without loading the model.")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction)
    parser.add_argument("--allow_remote_audio_model", action=argparse.BooleanOptionalAction)
    _set_parser_defaults(parser, defaults)
    args = parser.parse_args(remaining, namespace=pre_args)
    for key, value in DEFAULT_ARGS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def natural_key(path: Path) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.stem)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def sample_key(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    return match.group(1).zfill(4) if match else path.stem.lower()


def list_files(directory: Path, suffixes: Iterable[str]) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    suffixes = {s.lower() for s in suffixes}
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes],
        key=natural_key,
    )


def preferred_ref(paths: Sequence[Path]) -> Path:
    def score(path: Path) -> Tuple[int, int, Tuple[Any, ...]]:
        stem = path.stem.lower()
        starts_ref = 0 if stem.startswith("ref_") else 1
        suffix_rank = {".png": 0, ".jpg": 1, ".jpeg": 2}.get(path.suffix.lower(), 3)
        return (starts_ref, suffix_rank, natural_key(path))

    return sorted(paths, key=score)[0]


def build_jobs(
    audio_dir: Path,
    ref_dir: Path,
    prompt_lines: Sequence[str],
    pairing: str,
    max_jobs: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    audio_files = list_files(audio_dir, AUDIO_SUFFIXES)
    ref_files = list_files(ref_dir, IMAGE_SUFFIXES)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")
    if not ref_files:
        raise FileNotFoundError(f"No reference images found in {ref_dir}")

    refs_by_key: Dict[str, List[Path]] = {}
    for ref in ref_files:
        refs_by_key.setdefault(sample_key(ref), []).append(ref)

    jobs: List[Dict[str, Any]] = []
    unmatched_audio: List[str] = []
    if pairing == "cartesian":
        preferred_refs = [preferred_ref(paths) for _, paths in sorted(refs_by_key.items())]
        for audio in audio_files:
            for ref in preferred_refs:
                jobs.append({"key": f"{audio.stem}_{ref.stem}", "audio": audio, "ref": ref})
    else:
        for audio in audio_files:
            key = sample_key(audio)
            if key not in refs_by_key:
                unmatched_audio.append(str(audio))
                continue
            jobs.append({"key": key, "audio": audio, "ref": preferred_ref(refs_by_key[key])})

    if max_jobs is not None:
        jobs = jobs[: max(0, int(max_jobs))]

    for index, job in enumerate(jobs):
        prompt = prompt_lines[index] if index < len(prompt_lines) else (prompt_lines[0] if prompt_lines else "")
        job["prompt"] = prompt

    selected_ref_paths = {str(job["ref"]) for job in jobs}
    unmatched_refs = [str(p) for p in ref_files if str(p) not in selected_ref_paths]
    warnings = {"unmatched_audio": unmatched_audio, "unused_refs": unmatched_refs}
    return jobs, warnings


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _job_manifest_path(row: Dict[str, Any], *keys: str) -> Optional[Path]:
    for key in keys:
        value = row.get(key)
        if value:
            return Path(str(value))
    return None


def build_jobs_from_manifest(job_manifest: Path, max_jobs: Optional[int]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    if not job_manifest.exists():
        raise FileNotFoundError(f"Job manifest does not exist: {job_manifest}")
    rows = _load_jsonl(job_manifest)
    jobs: List[Dict[str, Any]] = []
    invalid_rows: List[str] = []
    for index, row in enumerate(rows):
        benchmark_key = row.get("benchmark_key") or row.get("key")
        ref = _job_manifest_path(row, "materialized_ref", "prepared_ref", "ref")
        audio = _job_manifest_path(row, "materialized_audio", "audio_segment", "audio")
        if not benchmark_key or ref is None or audio is None:
            invalid_rows.append(json.dumps({"row_index": index, "benchmark_key": benchmark_key}, ensure_ascii=False))
            continue
        generated_sample_id = row.get("generated_sample_id") or f"generated_{index + 1:04d}"
        job = {
            "key": str(generated_sample_id),
            "generated_sample_id": str(generated_sample_id),
            "benchmark_key": str(benchmark_key),
            "benchmark_row_index": row.get("benchmark_row_index"),
            "benchmark_row_hash": row.get("benchmark_row_hash"),
            "audio": audio,
            "ref": ref,
            "prompt": row.get("prompt", ""),
            "materialized_ref": str(ref),
            "materialized_audio": str(audio),
            "materialized_target_clip": row.get("materialized_target_clip"),
            "original_imgpath": row.get("original_imgpath"),
            "original_videopath": row.get("original_videopath"),
            "original_wav_path": row.get("original_wav_path"),
            "original_posepath": row.get("original_posepath"),
            "conditioning": row.get("conditioning") or {
                "reference_mode": "materialized_ref",
                "audio_mode": "materialized_audio",
            },
            "generation": {
                "width": row.get("generation_width"),
                "height": row.get("generation_height"),
                "fps": row.get("generation_fps"),
                "frame_num": row.get("generation_frame_num"),
            },
            "source_manifest_row": row,
        }
        jobs.append(job)
    if max_jobs is not None:
        jobs = jobs[: max(0, int(max_jobs))]
    return jobs, {"unmatched_audio": invalid_rows, "unused_refs": []}


def audio_duration_seconds(path: Path) -> Optional[float]:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                return handle.getnframes() / float(handle.getframerate())
        except wave.Error:
            pass
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def audio_segment_for_job(args: argparse.Namespace, job: Dict[str, Any], job_index: int) -> Tuple[Path, float, float]:
    duration = args.frame_num / float(args.fps)
    source = args.audio_source_path or job["audio"]
    if args.audio_source_path is None or args.audio_segment_mode == "per_job":
        start_time = 0.0
    elif args.audio_segment_mode == "fixed":
        start_time = float(args.audio_segment_start)
    elif args.audio_segment_mode == "sequential":
        stride = duration if args.audio_segment_stride is None else float(args.audio_segment_stride)
        start_time = float(args.audio_segment_start) + job_index * stride
    else:
        raise ValueError(f"Unsupported audio_segment_mode: {args.audio_segment_mode}")
    return Path(source), start_time, duration


def apply_audio_segments(args: argparse.Namespace, jobs: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    duration_cache: Dict[Path, Optional[float]] = {}
    segmented_jobs: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for job_index, job in enumerate(jobs):
        audio_path, start_time, duration = audio_segment_for_job(args, job, job_index)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio source does not exist: {audio_path}")
        if audio_path not in duration_cache:
            duration_cache[audio_path] = audio_duration_seconds(audio_path)
        source_duration = duration_cache[audio_path]
        end_time = start_time + duration
        if args.skip_incomplete_audio_segments and source_duration is not None and end_time > source_duration + 1e-3:
            skipped.append(
                {
                    "key": job["key"],
                    "audio_source": str(audio_path),
                    "audio_start_time": start_time,
                    "audio_end_time": end_time,
                    "audio_source_duration": source_duration,
                }
            )
            continue
        new_job = dict(job)
        new_job["audio_source"] = audio_path
        new_job["audio_start_time"] = start_time
        new_job["audio_duration"] = duration
        new_job["audio_end_time"] = end_time
        new_job["audio_source_duration"] = source_duration
        segmented_jobs.append(new_job)
    return segmented_jobs, {"skipped_incomplete_audio": skipped}


def extract_audio_segment(audio_path: Path, start_time: float, duration: float, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start_time:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def read_prompts(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    if checkpoint_dir.is_file():
        return checkpoint_dir
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    candidates = sorted(
        checkpoint_dir.glob("checkpoint_model_*/model.pt"),
        key=lambda p: natural_key(p.parent),
    )
    if not candidates:
        direct = checkpoint_dir / "model.pt"
        if direct.exists():
            return direct
        raise FileNotFoundError(f"No model.pt checkpoint found under {checkpoint_dir}")
    return candidates[-1]


def load_config(checkpoint_path: Path):
    from omegaconf import OmegaConf

    config_path = checkpoint_path.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint_path.parent.parent / "config.yaml"
    if not config_path.exists():
        sibling_configs = sorted(
            checkpoint_path.parent.parent.glob("checkpoint_model_*/config.yaml"),
            key=lambda p: natural_key(p.parent),
        )
        if sibling_configs:
            config_path = sibling_configs[-1]
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot find config.yaml next to checkpoint: {checkpoint_path}")
    return OmegaConf.load(config_path), config_path


def resolve_base_model_dir(config, requested: Optional[Path]) -> Path:
    if requested is not None:
        return requested
    configured = getattr(config, "generator_name", None) or getattr(config, "model_name", None)
    if configured and Path(str(configured)).exists():
        return Path(str(configured))
    if DEFAULT_BASE_MODEL_DIR.exists():
        return DEFAULT_BASE_MODEL_DIR
    if configured:
        return Path(str(configured))
    return DEFAULT_BASE_MODEL_DIR


def prepare_ref_image(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    new_w = int(math.ceil(src_w * scale))
    new_h = int(math.ceil(src_h * scale))
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = max(0, (new_w - width) // 2)
    top = max(0, (new_h - height) // 2)
    return image.crop((left, top, left + width, top + height))


def ref_to_video_tensor(image: Image.Image, frame_num: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    data = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float()
    first = data.div(127.5).sub(1.0)
    frames = torch.zeros((3, frame_num, image.height, image.width), dtype=torch.float32)
    frames[:, 0] = first
    return frames.unsqueeze(0).to(device=device, dtype=dtype)


def load_text_embedding(path: Path, text_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Text embedding does not exist: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    tensor = JsonlVideoDataset._extract_tensor(obj, str(path), preferred_keys=TEXT_KEYS)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Unsupported text embedding shape {tuple(tensor.shape)} in {path}; expected [T, C]")
    tensor = JsonlVideoDataset._pad_or_truncate_caption_embedding(tensor, text_len)
    return tensor.unsqueeze(0).to(device=device, dtype=dtype)


def build_audio_extractor(config, device: torch.device, local_files_only: bool) -> OnlineWav2VecAudioExtractor:
    audio_cfg = getattr(config, "audio_condition", None)
    if audio_cfg is None:
        raise ValueError("config.audio_condition is required for audio-conditioned inference")
    model_name = (
        getattr(audio_cfg, "model_name_or_path", None)
        or getattr(audio_cfg, "model_name", None)
        or getattr(audio_cfg, "audio_model_name_or_path", None)
        or getattr(audio_cfg, "audio_model_name", None)
    )
    if model_name is None:
        raise ValueError("config.audio_condition audio model path is required")
    extractor_device = str(getattr(audio_cfg, "audio_encoder_device", device))
    cfg_local_files_only = bool(getattr(audio_cfg, "audio_model_local_files_only", local_files_only))
    return OnlineWav2VecAudioExtractor(
        model_name_or_path=str(model_name),
        sample_rate=int(getattr(audio_cfg, "sample_rate", getattr(audio_cfg, "audio_sample_rate", 16000))),
        device=extractor_device,
        local_files_only=local_files_only or cfg_local_files_only,
        only_last_features=bool(getattr(audio_cfg, "only_last_features", getattr(audio_cfg, "audio_only_last_features", False))),
    )


def load_audio_embedding(
    extractor: OnlineWav2VecAudioExtractor,
    audio_path: Path,
    start_time: float,
    duration: float,
    frame_num: int,
    fps: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    audio = extractor.extract(str(audio_path), start_time=start_time, duration=duration, fps=fps, num_frames=frame_num)
    audio = JsonlVideoDataset._normalize_audio_embedding(audio, str(audio_path))
    audio = JsonlVideoDataset._sample_audio_frames(audio, frame_num).contiguous()
    return audio.unsqueeze(0).to(device=device, dtype=dtype)


def attach_fp8(config, generator) -> None:
    from fp8 import FP8Context

    fp8_context = FP8Context(config)
    if not fp8_context.enabled:
        return
    targets = [generator.model0, generator.model1] if getattr(generator, "dual_exp", False) else [generator]
    for target in targets:
        target.fp8_context = fp8_context


def state_dict_fingerprint(state_dict: Dict[str, Any], max_tensors: int = 3, sample_numel: int = 1024) -> str:
    parts: List[str] = []
    for name, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        flat = value.detach().reshape(-1)
        sample = flat[: min(sample_numel, flat.numel())].float()
        parts.append(
            f"{name}:shape={tuple(value.shape)} dtype={value.dtype} "
            f"sum{sample.numel()}={sample.sum().item():.6g}"
        )
        if len(parts) >= max_tensors:
            break
    return "; ".join(parts) if parts else "no tensor entries"


def load_model(args: argparse.Namespace, checkpoint_path: Path, config, device: torch.device) -> "RectifiedFlowFineTuneModel":
    from omegaconf import OmegaConf
    from model import RectifiedFlowFineTuneModel

    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    base_model_dir = resolve_base_model_dir(config, args.base_model_dir)
    config.generator_name = str(base_model_dir)
    config.vae_name = str(base_model_dir)
    config.gradient_checkpointing = False
    config.mixed_precision = args.dtype == "bf16"
    if not args.fp8:
        config.fp8 = False

    model = RectifiedFlowFineTuneModel(config, device=device)
    model.eval().requires_grad_(False)
    target_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model.generator.to(device=device, dtype=target_dtype)
    model.vae.to(device=device, dtype=target_dtype)
    model.vae.model.to(device=device, dtype=target_dtype)

    if args.fp8:
        attach_fp8(config, model.generator)

    logging.info("Loading generator checkpoint: %s", checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    generator_state = state["generator"] if isinstance(state, dict) and "generator" in state else state
    load_result = model.generator.load_state_dict(generator_state, strict=True)
    logging.info("Generator checkpoint load result: %s", load_result)
    logging.info(
        "Generator checkpoint tensors: %d; fingerprint: %s",
        len(generator_state),
        state_dict_fingerprint(generator_state),
    )
    del state, generator_state
    return model


def make_i2v_timestep(mask2: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    if mask2.ndim == 5:
        mask2 = mask2[0]
    return (mask2[:, 0, ::2, ::2] * timestep).flatten().unsqueeze(0)


@torch.no_grad()
def run_sampling(
    model: "RectifiedFlowFineTuneModel",
    ref_image: Image.Image,
    text_embeds: torch.Tensor,
    negative_text_embeds: Optional[torch.Tensor],
    audio_embeds: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    ref_video = ref_to_video_tensor(ref_image, args.frame_num, dtype, device)
    clean_latent = model.encode_video(ref_video)
    noise = torch.randn(clean_latent.shape, generator=generator, device=device, dtype=dtype)
    mask1, mask2_list = masks_like([noise], zero=True)
    del mask1
    mask2 = mask2_list[0].to(device=device, dtype=dtype)
    latent = (1.0 - mask2) * clean_latent + mask2 * noise

    scheduler = model.generator.get_scheduler()
    if args.shift is not None:
        scheduler.shift = float(args.shift)
    scheduler.set_timesteps(args.sample_steps)
    scheduler.timesteps = scheduler.timesteps.to(device=device, dtype=dtype)
    scheduler.sigmas = scheduler.sigmas.to(device=device, dtype=dtype)

    cond = {"prompt_embeds": text_embeds}
    if args.cfg_mode == "text":
        if negative_text_embeds is None:
            raise ValueError("cfg_mode=text requires negative_text_embeds")
        uncond = {"prompt_embeds": negative_text_embeds}
        uncond_audio = audio_embeds
    elif args.cfg_mode == "audio":
        uncond = {"prompt_embeds": text_embeds}
        uncond_audio = torch.zeros_like(audio_embeds)
    elif args.cfg_mode == "both":
        if negative_text_embeds is None:
            raise ValueError("cfg_mode=both requires negative_text_embeds")
        uncond = {"prompt_embeds": negative_text_embeds}
        uncond_audio = torch.zeros_like(audio_embeds)
    elif args.cfg_mode == "zero":
        uncond = {"prompt_embeds": torch.zeros_like(text_embeds)}
        uncond_audio = torch.zeros_like(audio_embeds)
    else:
        uncond = None
        uncond_audio = None
    batch_size, latent_frames = latent.shape[:2]

    for step_index, timestep_value in enumerate(scheduler.timesteps):
        timestep_value = timestep_value.to(device=device, dtype=dtype)
        timestep = make_i2v_timestep(mask2, timestep_value)
        flow_pred = model.generator(
            latent,
            cond,
            timestep,
            audio_embeds=audio_embeds,
            mask_clean_first_audio=True,
        )
        if args.guide_scale != 1.0 and args.cfg_mode != "off":
            uncond_flow = model.generator(
                latent,
                uncond,
                timestep,
                audio_embeds=uncond_audio,
                mask_clean_first_audio=True,
            )
            flow_pred = uncond_flow + float(args.guide_scale) * (flow_pred - uncond_flow)

        flat_timestep = torch.full((batch_size * latent_frames,), timestep_value.item(), device=device, dtype=dtype)
        latent = scheduler.step(
            flow_pred.flatten(0, 1),
            flat_timestep,
            latent.flatten(0, 1),
        ).unflatten(0, (batch_size, latent_frames))
        latent = (1.0 - mask2) * clean_latent + mask2 * latent
        logging.info("Sampling step %d/%d done", step_index + 1, len(scheduler.timesteps))

    z = latent[0].permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype)
    model.vae.scale = [item.to(device=device, dtype=dtype) for item in model.vae.scale]
    video = model.vae.model.decode(z, model.vae.scale).clamp(-1, 1).float().cpu()
    return video


def write_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def key_preserving_manifest_fields(job: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    conditioning = dict(job.get("conditioning") or {})
    generation = dict(job.get("generation") or {})
    return {
        "generated_sample_id": job.get("generated_sample_id") or job.get("key"),
        "benchmark_key": job.get("benchmark_key"),
        "benchmark_row_index": job.get("benchmark_row_index"),
        "benchmark_row_hash": job.get("benchmark_row_hash"),
        "materialized_ref": job.get("materialized_ref") or str(job.get("ref")),
        "materialized_audio": job.get("materialized_audio") or str(job.get("audio")),
        "materialized_target_clip": job.get("materialized_target_clip"),
        "original_imgpath": job.get("original_imgpath"),
        "original_videopath": job.get("original_videopath"),
        "original_wav_path": job.get("original_wav_path"),
        "original_posepath": job.get("original_posepath"),
        "conditioning": {
            "reference_mode": conditioning.get("reference_mode", "materialized_ref"),
            "audio_mode": conditioning.get("audio_mode", "materialized_audio"),
        },
        "generation": {
            "width": generation.get("width") or args.width,
            "height": generation.get("height") or args.height,
            "fps": generation.get("fps") or args.fps,
            "frame_num": generation.get("frame_num") or args.frame_num,
        },
        "generation_width": generation.get("width") or args.width,
        "generation_height": generation.get("height") or args.height,
        "generation_fps": generation.get("fps") or args.fps,
        "generation_frame_num": generation.get("frame_num") or args.frame_num,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    inference_dir = args.inference_dir
    audio_dir = args.audio_dir or inference_dir / "audio"
    ref_dir = args.ref_dir or inference_dir / "ref"
    prompt_file = args.prompt_file or inference_dir / "prompt.txt"
    checkpoint_path = args.checkpoint_path or latest_checkpoint(args.checkpoint_dir)
    config, config_path = load_config(checkpoint_path)
    base_model_dir = resolve_base_model_dir(config, args.base_model_dir)
    output_dir = args.output_dir or PROJECT_ROOT / "inference_outputs" / f"audio_i2v_{time.strftime('%Y%m%d_%H%M%S')}"
    ref_out_dir = output_dir / "refs_480p"
    video_out_dir = output_dir / "videos"
    audio_segment_out_dir = output_dir / "audio_segments"

    prompts = read_prompts(prompt_file if prompt_file.exists() else None)
    if args.job_manifest is not None:
        jobs, warnings = build_jobs_from_manifest(args.job_manifest, args.max_jobs)
        # A key-preserving job manifest already carries per-sample materialized audio.
        # Do not let a legacy global audio_source_path from the YAML override it.
        args.audio_source_path = None
        args.audio_segment_mode = "per_job"
    else:
        jobs, warnings = build_jobs(audio_dir, ref_dir, prompts, args.pairing, args.max_jobs)
    jobs, audio_warnings = apply_audio_segments(args, jobs)

    logging.info("Checkpoint: %s", checkpoint_path)
    logging.info("Config: %s", config_path)
    logging.info("Base model: %s", base_model_dir)
    logging.info("Text embedding: %s", args.text_emb_path)
    if args.audio_source_path is not None:
        logging.info("Audio source: %s", args.audio_source_path)
        logging.info("Audio segment mode: %s", args.audio_segment_mode)
    logging.info("Jobs: %d", len(jobs))
    if args.guide_scale != 1.0 and args.cfg_mode == "off":
        logging.warning("guide_scale=%s is ignored because cfg_mode=off.", args.guide_scale)
    if args.cfg_mode == "both":
        logging.warning("cfg_mode=both enables text CFG and audio CFG by using negative text and zero audio as uncond.")
    if args.cfg_mode == "audio":
        logging.warning("cfg_mode=audio enables audio CFG only by using the same text and zero audio as uncond.")
    if args.cfg_mode == "zero":
        logging.warning("cfg_mode=zero uses an untrained zero-text/zero-audio branch; cfg_mode=text is preferred.")
    if warnings["unmatched_audio"]:
        logging.warning("Audio files without matched ref: %d", len(warnings["unmatched_audio"]))
    if warnings["unused_refs"]:
        logging.warning("Reference images not selected: %d", len(warnings["unused_refs"]))
    if audio_warnings["skipped_incomplete_audio"]:
        logging.warning("Skipped incomplete audio segments: %d", len(audio_warnings["skipped_incomplete_audio"]))

    dry_rows: List[Dict[str, Any]] = []
    for job in jobs[: max(0, min(len(jobs), args.max_jobs or len(jobs)))]:
        ref_image = prepare_ref_image(job["ref"], args.width, args.height)
        planned_video_path = video_out_dir / f"{job['key']}.mp4"
        planned_audio_segment_path = audio_segment_out_dir / f"{job['key']}.wav"
        planned_ref_path = ref_out_dir / f"{job['key']}.png"
        row = {
            "key": job["key"],
            **key_preserving_manifest_fields(job, args),
            "audio": str(job["audio"]),
            "audio_source": str(job["audio_source"]),
            "audio_start_time": job["audio_start_time"],
            "audio_end_time": job["audio_end_time"],
            "audio_duration": job["audio_duration"],
            "audio_segment": str(planned_audio_segment_path),
            "ref": str(job["ref"]),
            "prepared_ref": str(planned_ref_path),
            "prompt": job.get("prompt", ""),
            "prepared_ref_size": list(ref_image.size),
            "checkpoint": str(checkpoint_path),
            "video": str(planned_video_path),
            "generated_video": str(planned_video_path),
            "dry_run": True,
            "generated_video_exists": False,
        }
        dry_rows.append(row)

    if args.dry_run:
        logging.info("Dry run only; model is not loaded.")
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_out_dir.mkdir(parents=True, exist_ok=True)
        video_out_dir.mkdir(parents=True, exist_ok=True)
        audio_segment_out_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(output_dir / "manifest.jsonl", dry_rows)
        for row in dry_rows:
            logging.info(
                "Planned job %s: audio_source=%s audio=[%.3f, %.3f] ref=%s prepared_ref_size=%s",
                row["key"],
                row["audio_source"],
                row["audio_start_time"],
                row["audio_end_time"],
                row["ref"],
                row["prepared_ref_size"],
            )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    ref_out_dir.mkdir(parents=True, exist_ok=True)
    video_out_dir.mkdir(parents=True, exist_ok=True)
    audio_segment_out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    local_files_only = bool(args.local_files_only and not args.allow_remote_audio_model)

    model = load_model(args, checkpoint_path, config, device)
    text_embeds = load_text_embedding(args.text_emb_path, text_len=512, dtype=dtype, device=device)
    negative_text_embeds = None
    if args.cfg_mode in {"text", "both"} and args.guide_scale != 1.0:
        negative_text_embeds = load_text_embedding(args.negative_text_emb_path, text_len=512, dtype=dtype, device=device)
    audio_extractor = build_audio_extractor(config, device, local_files_only=local_files_only)

    manifest_rows: List[Dict[str, Any]] = []
    for job_index, job in enumerate(jobs):
        key = str(job["key"])
        ref_image = prepare_ref_image(job["ref"], args.width, args.height)
        ref_save_path = ref_out_dir / f"{key}.png"
        ref_image.save(ref_save_path)
        audio_embeds = load_audio_embedding(
            audio_extractor,
            job["audio_source"],
            job["audio_start_time"],
            job["audio_duration"],
            args.frame_num,
            args.fps,
            dtype,
            device,
        )

        job_seed = int(args.seed) + job_index
        old_seed = args.seed
        args.seed = job_seed
        try:
            video = run_sampling(model, ref_image, text_embeds, negative_text_embeds, audio_embeds, args, device, dtype)
        finally:
            args.seed = old_seed

        video_path = video_out_dir / f"{key}.mp4"
        save_video(video, save_file=str(video_path), fps=args.fps, nrow=1, normalize=True, value_range=(-1, 1))
        audio_segment_path = audio_segment_out_dir / f"{key}.wav"
        extract_audio_segment(job["audio_source"], job["audio_start_time"], job["audio_duration"], audio_segment_path)
        if args.merge_audio:
            merge_video_audio(str(video_path), str(audio_segment_path))

        row = {
            "key": key,
            **key_preserving_manifest_fields(job, args),
            "audio": str(job["audio"]),
            "audio_source": str(job["audio_source"]),
            "audio_start_time": job["audio_start_time"],
            "audio_end_time": job["audio_end_time"],
            "audio_duration": job["audio_duration"],
            "audio_segment": str(audio_segment_path),
            "ref": str(job["ref"]),
            "prepared_ref": str(ref_save_path),
            "prompt": job.get("prompt", ""),
            "text_emb_path": str(args.text_emb_path),
            "checkpoint": str(checkpoint_path),
            "seed": job_seed,
            "cfg_mode": args.cfg_mode,
            "guide_scale": args.guide_scale,
            "video": str(video_path),
            "generated_video": str(video_path),
            "dry_run": False,
            "generated_video_exists": video_path.exists(),
        }
        manifest_rows.append(row)
        write_manifest(output_dir / "manifest.jsonl", manifest_rows)
        logging.info("Saved %s", video_path)

    logging.info("Done. Outputs: %s", output_dir)


if __name__ == "__main__":
    main()
