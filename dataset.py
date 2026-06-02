import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


class JsonlVideoDataset(Dataset):
    """
    JSONL dataset for rows like:
        {"video": "XXXX.mp4", "caption": "XXXXX"}

    Returns video tensors in [C, T, H, W], normalized to [-1, 1].
    """

    def __init__(
        self,
        jsonl_path: str,
        video_root: Optional[str] = None,
        target_fps: int = None,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        reader: str = "auto",
        max_samples: Optional[int] = None,
        load_audio_emb: bool = False,
        audio_emb_key: str = "vocals_emb_base_all",
        audio_emb_root: Optional[str] = None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.video_root = Path(video_root) if video_root else None
        self.audio_emb_root = Path(audio_emb_root) if audio_emb_root else None
        self.target_fps = target_fps
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.reader = reader
        self.load_audio_emb = bool(load_audio_emb)
        self.audio_emb_key = audio_emb_key
        self.samples = self._load_jsonl(max_samples=max_samples)

    def _load_jsonl(self, max_samples: Optional[int]) -> List[Dict[str, Any]]:
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

        samples = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.jsonl_path}:{line_no}: {exc}") from exc

                if "video" not in row or "caption" not in row:
                    raise ValueError(
                        f"Expected keys 'video' and 'caption' at {self.jsonl_path}:{line_no}, got {list(row.keys())}"
                    )

                video_path = self._resolve_path(row["video"], self.video_root)
                # if not video_path.exists():
                #     raise FileNotFoundError(f"Video not found at {self.jsonl_path}:{line_no}: {video_path}")

                sample = {"video": str(video_path), "caption": str(row["caption"]), "idx": len(samples)}
                if self.load_audio_emb:
                    if self.audio_emb_key not in row or not row[self.audio_emb_key]:
                        raise ValueError(
                            f"Expected audio embedding key '{self.audio_emb_key}' at "
                            f"{self.jsonl_path}:{line_no}"
                        )
                    sample["audio_emb_path"] = str(self._resolve_path(row[self.audio_emb_key], self.audio_emb_root))

                samples.append(sample)
                if max_samples is not None and len(samples) >= max_samples:
                    break

        if not samples:
            raise ValueError(f"No samples found in {self.jsonl_path}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        video = self._read_video(item["video"], self.target_fps)
        video = self._sample_frames(video, self.num_frames)
        video = self._resize_center_crop(video, self.height, self.width)
        video = video.float().div_(127.5).sub_(1.0)
        output = {
            "frames": video.permute(1, 0, 2, 3).contiguous(),  # [C, T, H, W]
            "prompts": item["caption"],
            "video_path": item["video"],
            "idx": item["idx"],
        }
        if self.load_audio_emb:
            output["audio_embeds"] = self._load_audio_embedding(item["audio_emb_path"])
            output["audio_emb_path"] = item["audio_emb_path"]
        return output

    @staticmethod
    def _resolve_path(path_value: str, root: Optional[Path]) -> Path:
        path = Path(path_value)
        if root and not path.is_absolute():
            path = root / path
        return path

    def _load_audio_embedding(self, audio_emb_path: str) -> torch.Tensor:
        obj = torch.load(audio_emb_path, map_location="cpu")
        audio = self._extract_tensor(obj, audio_emb_path)
        audio = torch.as_tensor(audio).float()
        audio = self._normalize_audio_embedding(audio, audio_emb_path)
        return self._sample_audio_frames(audio, self.num_frames).contiguous()

    @staticmethod
    def _extract_tensor(obj: Any, source: str) -> torch.Tensor:
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        if isinstance(obj, dict):
            preferred_keys = (
                "audio_embeds",
                "audio_embed",
                "audio_embedding",
                "vocals_emb_base_all",
                "hidden_states",
                "last_hidden_state",
                "features",
                "feat",
                "embedding",
                "emb",
            )
            for key in preferred_keys:
                if key in obj:
                    return JsonlVideoDataset._extract_tensor(obj[key], source)
            for value in obj.values():
                try:
                    return JsonlVideoDataset._extract_tensor(value, source)
                except TypeError:
                    continue
        raise TypeError(f"Cannot find a tensor audio embedding in {source}")

    @staticmethod
    def _normalize_audio_embedding(audio: torch.Tensor, source: str) -> torch.Tensor:
        if audio.ndim == 5 and audio.shape[0] == 1:
            audio = audio.squeeze(0)

        if audio.ndim == 2:
            # [F, C] -> [F, 1, C]
            audio = audio.unsqueeze(1)
        elif audio.ndim == 3:
            # Accept both [F, num_layers, C] and [num_layers, F, C].
            if audio.shape[0] <= 32 and audio.shape[1] > 32:
                audio = audio.permute(1, 0, 2)
        elif audio.ndim == 4:
            # [F, window, num_layers, C], already windowed.
            pass
        else:
            raise ValueError(
                f"Unsupported audio embedding shape {tuple(audio.shape)} in {source}; "
                "expected [F, C], [F, L, C], [L, F, C], or [F, W, L, C]."
            )
        return audio

    @staticmethod
    def _sample_audio_frames(audio: torch.Tensor, num_frames: int) -> torch.Tensor:
        total = audio.shape[0]
        if total <= 0:
            raise ValueError("Audio embedding has no frames")
        if total >= num_frames:
            return audio[:num_frames]
        pad = audio[-1:].repeat(num_frames - total, *([1] * (audio.ndim - 1)))
        return torch.cat([audio, pad], dim=0)

    def _read_video(self, video_path: str, target_fps:int) -> torch.Tensor:
        if self.reader == "decord" or (self.reader == "auto" and _is_decord_available()):
            return self._read_video_decord(video_path, target_fps)
        # if self.reader in ("auto", "torchvision"):
        #     return self._read_video_torchvision(video_path)
        raise ValueError(f"Unsupported video reader: {self.reader}")

    @staticmethod
    def _read_video_fps(vr, target_fps):
        src_fps = vr.get_avg_fps()
        num_frames = len(vr)
        duration = num_frames / src_fps
        timestamps = np.arange(0, duration, 1.0 / target_fps)
        indices = np.round(timestamps * src_fps).astype(np.int64)
        indices = np.clip(indices, 0, num_frames - 1)
        frames = vr.get_batch(indices).asnumpy()
        return frames

    @staticmethod
    def _read_video_decord(video_path: str, target_fps:int) -> torch.Tensor:
        import decord
        vr = decord.VideoReader(video_path)
        if len(vr) <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        if target_fps:
            src_fps = vr.get_avg_fps()
            num_frames = len(vr)
            duration = num_frames / src_fps
            timestamps = np.arange(0, duration, 1.0 / target_fps)
            indices = np.round(timestamps * src_fps).astype(np.int64)
            indices = np.clip(indices, 0, num_frames - 1)
            frames = vr.get_batch(indices).asnumpy()
        else:
            frames = vr.get_batch(list(range(len(vr)))).asnumpy()
        return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()

    # @staticmethod
    # def _read_video_torchvision(video_path: str) -> torch.Tensor:
    #     from torchvision.io import read_video
    #     video, _, _ = read_video(video_path, pts_unit="sec", output_format="TCHW")
    #     if video.numel() == 0:
    #         raise ValueError(f"Video has no frames: {video_path}")
    #     return video.contiguous()

    @staticmethod
    def _sample_frames(video: torch.Tensor, num_frames: int) -> torch.Tensor:
        total = video.shape[0]
        # if total == num_frames:
        #     return video
        # if total > num_frames:
        #     indices = torch.linspace(0, total - 1, num_frames).round().long()
        #     return video.index_select(0, indices)
        # pad = video[-1:].repeat(num_frames - total, 1, 1, 1)
        # return torch.cat([video, pad], dim=0)
        if total >= num_frames:
            return video[:num_frames]
        else:
            raise ValueError(f"Video has not enough frames: {video_path}")

    @staticmethod
    def _resize_center_crop(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
        _, _, src_h, src_w = video.shape
        scale = max(height / src_h, width / src_w)
        resized_h = max(height, int(round(src_h * scale)))
        resized_w = max(width, int(round(src_w * scale)))
        
        video = F.interpolate(video.float(), size=(resized_h, resized_w), mode="bicubic", align_corners=False)
        top = (resized_h - height) // 2
        left = (resized_w - width) // 2
        return video[:, :, top:top + height, left:left + width].clamp_(0, 255).to(torch.uint8)


def _is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def worker_init_fn(_worker_id: int):
    # Avoid decord inheriting a stale bridge state in forked dataloader workers.
    if _is_decord_available():
        try:
            import decord

            decord.bridge.set_bridge("native")
        except Exception:
            pass
