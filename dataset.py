import json
import math
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
        {"video": "XXXX.mp4", "caption": "XXXXX", "caption_emb": "XXXX.pt"}

    If video_root is set, it is only applied to relative video paths.
    Returns video tensors in [C, T, H, W], normalized to [-1, 1].
    Caption embeddings are padded/truncated to text_len and returned as
    prompt_embeds, so training does not need to run the T5 text encoder.
    If default_caption_emb_path is set, rows without caption_emb use that
    fallback embedding.
    """

    def __init__(
        self,
        jsonl_path: str,
        video_root: Optional[str] = None,
        target_fps: int = None,
        num_frames: int = 81,
        height: int = 832,
        width: int = 480,
        reader: str = "auto",
        max_samples: Optional[int] = None,
        load_caption_emb: bool = True,
        caption_emb_key: str = "caption_emb",
        caption_emb_root: Optional[str] = None,
        default_caption_emb_path: Optional[str] = None,
        text_len: int = 512,
        load_audio_emb: bool = False,
        audio_emb_mode: str = "offline",
        audio_emb_key: str = "vocals_emb_base_all",
        audio_emb_root: Optional[str] = None,
        audio_path_key: str = "audio_path",
        audio_path_root: Optional[str] = None,
        audio_model_name_or_path: str = "TencentGameMate/chinese-wav2vec2-base",
        audio_sample_rate: int = 16000,
        audio_encoder_device: str = "cpu",
        audio_model_local_files_only: bool = False,
        audio_only_last_features: bool = False,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.video_root = Path(video_root) if video_root else None
        self.caption_emb_root = Path(caption_emb_root) if caption_emb_root else None
        self.audio_emb_root = Path(audio_emb_root) if audio_emb_root else None
        self.audio_path_root = Path(audio_path_root) if audio_path_root else None
        self.target_fps = target_fps
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.reader = reader
        self.load_caption_emb = bool(load_caption_emb)
        self.caption_emb_key = caption_emb_key
        self.default_caption_emb_path = None
        if default_caption_emb_path:
            self.default_caption_emb_path = str(self._resolve_path(default_caption_emb_path, self.caption_emb_root))
            if not Path(self.default_caption_emb_path).exists():
                raise FileNotFoundError(f"default_caption_emb_path not found: {self.default_caption_emb_path}")
        self.text_len = int(text_len)
        if self.text_len <= 0:
            raise ValueError(f"text_len must be positive, got {text_len}")
        self.load_audio_emb = bool(load_audio_emb)
        self.audio_emb_mode = str(audio_emb_mode).lower()
        if self.audio_emb_mode not in {"offline", "online"}:
            raise ValueError(f"Unsupported audio_emb_mode: {audio_emb_mode}")
        self.audio_emb_key = audio_emb_key
        self.audio_path_key = audio_path_key
        self.audio_model_name_or_path = audio_model_name_or_path
        self.audio_sample_rate = int(audio_sample_rate)
        self.audio_encoder_device = audio_encoder_device
        self.audio_model_local_files_only = bool(audio_model_local_files_only)
        self.audio_only_last_features = bool(audio_only_last_features)
        self._audio_extractor = None
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

                video_value = row.get("video") or row.get("video_path")
                if not video_value:
                    raise ValueError(
                        f"Expected key 'video' or 'video_path' at "
                        f"{self.jsonl_path}:{line_no}, got {list(row.keys())}"
                    )

                video_path = self._resolve_path(video_value, self.video_root)
                # if not video_path.exists():
                #     raise FileNotFoundError(f"Video not found at {self.jsonl_path}:{line_no}: {video_path}")

                sample = {"video": str(video_path), "caption": str(row.get("caption", "")), "idx": len(samples)}
                if self.load_caption_emb:
                    caption_emb_value = row.get(self.caption_emb_key)
                    if caption_emb_value:
                        sample["caption_emb_path"] = str(self._resolve_path(caption_emb_value, self.caption_emb_root))
                    elif self.default_caption_emb_path:
                        sample["caption_emb_path"] = self.default_caption_emb_path
                    else:
                        raise ValueError(
                            f"Expected caption embedding key '{self.caption_emb_key}' at "
                            f"{self.jsonl_path}:{line_no}"
                        )
                if self.load_audio_emb and self.audio_emb_mode == "online":
                    if self.audio_path_key not in row or not row[self.audio_path_key]:
                        continue
                    sample["audio_path"] = str(self._resolve_path(row[self.audio_path_key], self.audio_path_root))
                elif self.load_audio_emb:
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
        video, clip_fps = self._read_video(item["video"], self.target_fps)
        video, frame_start, frame_end = self._sample_frames(video, self.num_frames, item["video"])
        video = self._resize_center_crop(video, self.height, self.width)
        video = video.float().div_(127.5).sub_(1.0)
        output = {
            "frames": video.permute(1, 0, 2, 3).contiguous(),  # [C, T, H, W]
            "prompts": item["caption"],
            "video_path": item["video"],
            "idx": item["idx"],
        }
        if self.load_audio_emb:
            if self.audio_emb_mode == "online":
                output["audio_embeds"] = self._load_online_audio_embedding(
                    item["audio_path"],
                    frame_start=frame_start,
                    frame_end=frame_end,
                    fps=clip_fps,
                )
                output["audio_path"] = item["audio_path"]
            else:
                output["audio_embeds"] = self._load_audio_embedding(item["audio_emb_path"], frame_start=frame_start)
                output["audio_emb_path"] = item["audio_emb_path"]
        if self.load_caption_emb:
            output["prompt_embeds"] = self._load_caption_embedding(item["caption_emb_path"])
            output["caption_emb_path"] = item["caption_emb_path"]
        return output

    @staticmethod
    def _resolve_path(path_value: str, root: Optional[Path]) -> Path:
        path = Path(path_value)
        if root and not path.is_absolute():
            path = root / path
        return path

    def _load_audio_embedding(self, audio_emb_path: str, frame_start: int = 0) -> torch.Tensor:
        obj = torch.load(audio_emb_path, map_location="cpu")
        audio = self._extract_tensor(obj, audio_emb_path)
        audio = torch.as_tensor(audio).float()
        audio = self._normalize_audio_embedding(audio, audio_emb_path)
        return self._sample_audio_frames(audio, self.num_frames, frame_start=frame_start).contiguous()

    def _load_online_audio_embedding(
        self,
        audio_path: str,
        frame_start: int,
        frame_end: int,
        fps: float,
    ) -> torch.Tensor:
        if fps <= 0:
            raise ValueError(f"Invalid fps for online audio extraction: {fps}")
        start_time = frame_start / fps
        duration = (frame_end - frame_start) / fps
        audio = self._get_audio_extractor().extract(
            audio_path=audio_path,
            start_time=start_time,
            duration=duration,
            fps=fps,
            num_frames=self.num_frames,
        )
        audio = self._normalize_audio_embedding(audio, audio_path)
        return self._sample_audio_frames(audio, self.num_frames).contiguous()

    def _get_audio_extractor(self):
        if self._audio_extractor is None:
            self._audio_extractor = OnlineWav2VecAudioExtractor(
                model_name_or_path=self.audio_model_name_or_path,
                sample_rate=self.audio_sample_rate,
                device=self.audio_encoder_device,
                local_files_only=self.audio_model_local_files_only,
                only_last_features=self.audio_only_last_features,
            )
        return self._audio_extractor

    def _load_caption_embedding(self, caption_emb_path: str) -> torch.Tensor:
        obj = torch.load(caption_emb_path, map_location="cpu")
        caption = self._extract_tensor(
            obj,
            caption_emb_path,
            preferred_keys=(
                "caption_emb",
                "caption_embed",
                "caption_embedding",
                "prompt_embeds",
                "prompt_embed",
                "text_embeds",
                "text_embed",
                "text_embedding",
                "hidden_states",
                "last_hidden_state",
                "context",
                "embedding",
                "emb",
            ),
        )
        caption = torch.as_tensor(caption)
        if not torch.is_floating_point(caption):
            caption = caption.float()
        if caption.ndim == 3 and caption.shape[0] == 1:
            caption = caption.squeeze(0)
        if caption.ndim != 2:
            raise ValueError(
                f"Unsupported caption embedding shape {tuple(caption.shape)} in {caption_emb_path}; "
                "expected [L, C] or [1, L, C]."
            )
        return self._pad_or_truncate_caption_embedding(caption, self.text_len).contiguous()

    @staticmethod
    def _extract_tensor(obj: Any, source: str, preferred_keys: Optional[tuple] = None) -> torch.Tensor:
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        if isinstance(obj, dict):
            if preferred_keys is None:
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
                    return JsonlVideoDataset._extract_tensor(obj[key], source, preferred_keys=preferred_keys)
            for value in obj.values():
                try:
                    return JsonlVideoDataset._extract_tensor(value, source, preferred_keys=preferred_keys)
                except TypeError:
                    continue
        raise TypeError(f"Cannot find a tensor embedding in {source}")

    @staticmethod
    def _pad_or_truncate_caption_embedding(caption: torch.Tensor, text_len: int) -> torch.Tensor:
        if caption.shape[0] >= text_len:
            return caption[:text_len]
        pad = caption.new_zeros((text_len - caption.shape[0], caption.shape[1]))
        return torch.cat([caption, pad], dim=0)

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
    def _sample_audio_frames(audio: torch.Tensor, num_frames: int, frame_start: int = 0) -> torch.Tensor:
        total = audio.shape[0]
        if total <= 0:
            raise ValueError("Audio embedding has no frames")
        if frame_start < 0:
            raise ValueError(f"frame_start must be non-negative, got {frame_start}")
        if frame_start > 0 and total > frame_start:
            audio = audio[frame_start:]
            total = audio.shape[0]
        elif frame_start > 0:
            audio = audio[-1:]
            total = audio.shape[0]
        if total >= num_frames:
            return audio[:num_frames]
        pad = audio[-1:].repeat(num_frames - total, *([1] * (audio.ndim - 1)))
        return torch.cat([audio, pad], dim=0)

    def _read_video(self, video_path: str, target_fps:int):
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
    def _read_video_decord(video_path: str, target_fps:int):
        import decord
        vr = decord.VideoReader(video_path)
        if len(vr) <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        src_fps = float(vr.get_avg_fps())
        if target_fps:
            num_frames = len(vr)
            duration = num_frames / src_fps
            timestamps = np.arange(0, duration, 1.0 / target_fps)
            indices = np.round(timestamps * src_fps).astype(np.int64)
            indices = np.clip(indices, 0, num_frames - 1)
            frames = vr.get_batch(indices).asnumpy()
            clip_fps = float(target_fps)
        else:
            frames = vr.get_batch(list(range(len(vr)))).asnumpy()
            clip_fps = src_fps
        return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous(), clip_fps

    # @staticmethod
    # def _read_video_torchvision(video_path: str) -> torch.Tensor:
    #     from torchvision.io import read_video
    #     video, _, _ = read_video(video_path, pts_unit="sec", output_format="TCHW")
    #     if video.numel() == 0:
    #         raise ValueError(f"Video has no frames: {video_path}")
    #     return video.contiguous()

    @staticmethod
    def _sample_frames(video: torch.Tensor, num_frames: int, video_path: str):
        total = video.shape[0]
        if total >= num_frames:
            max_start = total - num_frames
            frame_start = int(torch.randint(0, max_start + 1, ()).item()) if max_start > 0 else 0
            frame_end = frame_start + num_frames
            return video[frame_start:frame_end], frame_start, frame_end
        else:
            raise ValueError(f"Video has not enough frames ({total} < {num_frames}): {video_path}")

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


def _linear_interpolation(features: torch.Tensor, seq_len: int) -> torch.Tensor:
    features = features.transpose(1, 2)
    features = F.interpolate(features, size=seq_len, align_corners=True, mode="linear")
    return features.transpose(1, 2)


def _build_wav2vec_model_cls():
    from transformers import Wav2Vec2Model
    from transformers.modeling_outputs import BaseModelOutput

    class ReferenceAlignedWav2VecModel(Wav2Vec2Model):
        def forward(
            self,
            input_values,
            seq_len,
            attention_mask=None,
            mask_time_indices=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            extract_features = self.feature_extractor(input_values).transpose(1, 2)
            extract_features = _linear_interpolation(extract_features, seq_len=seq_len)

            if attention_mask is not None:
                attention_mask = self._get_feature_vector_attention_mask(
                    extract_features.shape[1],
                    attention_mask,
                    add_adapter=False,
                )

            hidden_states, extract_features = self.feature_projection(extract_features)
            hidden_states = self._mask_hidden_states(
                hidden_states,
                mask_time_indices=mask_time_indices,
                attention_mask=attention_mask,
            )

            encoder_outputs = self.encoder(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = encoder_outputs[0]

            if self.adapter is not None:
                hidden_states = self.adapter(hidden_states)

            if not return_dict:
                return (hidden_states,) + encoder_outputs[1:]
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

    return ReferenceAlignedWav2VecModel


class OnlineWav2VecAudioExtractor:
    def __init__(
        self,
        model_name_or_path: str,
        sample_rate: int,
        device: str,
        local_files_only: bool,
        only_last_features: bool,
    ):
        from transformers import Wav2Vec2FeatureExtractor

        model_cls = _build_wav2vec_model_cls()
        self.sample_rate = int(sample_rate)
        self.device = torch.device(device)
        self.only_last_features = bool(only_last_features)
        self.audio_encoder = model_cls.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        ).to(device=self.device)
        self.audio_encoder.eval()
        self.audio_encoder.feature_extractor._freeze_parameters()
        for param in self.audio_encoder.parameters():
            param.requires_grad_(False)
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )

    def extract(
        self,
        audio_path: str,
        start_time: float,
        duration: float,
        fps: float,
        num_frames: int,
    ) -> torch.Tensor:
        import os

        os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
        import librosa

        if duration <= 0:
            raise ValueError(f"Audio duration must be positive, got {duration}")
        speech_array, sampling_rate = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True,
            offset=float(start_time),
            duration=float(duration),
        )
        target_samples = max(1, int(math.ceil(duration * self.sample_rate)))
        if speech_array.shape[0] < target_samples:
            speech_array = np.pad(speech_array, (0, target_samples - speech_array.shape[0]))
        elif speech_array.shape[0] > target_samples:
            speech_array = speech_array[:target_samples]

        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(
                speech_array,
                sampling_rate=sampling_rate,
            ).input_values
        )
        seq_len = max(num_frames, int(math.ceil(len(audio_feature) / self.sample_rate * fps)))
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device).unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(
                audio_feature,
                seq_len=seq_len,
                output_hidden_states=True,
            )

        if self.only_last_features:
            audio_emb = embeddings.last_hidden_state.squeeze(0)
        else:
            hidden_states = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = hidden_states.permute(1, 0, 2).contiguous()
        return audio_emb.cpu().detach()
