# Metric Reference Map

This document records read-only reference discovery for the offline digital human video eval agent.

Realtime eval remains pending only. Do not implement TTFF, latency, deadline miss, streaming trace, or chunk boundary in the current MVP.

## 1. Reference Accessibility

| Source | Status | Checked |
| --- | --- | --- |
| Current project | accessible | `OFFLINE_DIGITAL_HUMAN_EVAL_AGENT.md`, `docs/REFERENCE_SOURCES.md` |
| `eval_agent/` | not present | `find eval_agent` returned no such directory |
| Hallo-Live | partially accessible | GitHub raw files accessible; GitHub API hit anonymous rate limit |
| Local Baselines | accessible | `/mnt/data/cv/yutan/Baselines/evaluation` |
| TalkVid dataprocess | accessible | `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess` |

## 2. Checked Key Files

| Source | Files |
| --- | --- |
| Baselines | `/mnt/data/cv/yutan/Baselines/evaluation/sync_calculation_extracted.py`; `/mnt/data/cv/yutan/Baselines/evaluation/utils/SyncNetInstance.py`; `/mnt/data/cv/yutan/Baselines/evaluation/iqa_calculation_extracted.py`; `/mnt/data/cv/yutan/Baselines/evaluation/fid_fvd_calculation_extracted.py` |
| TalkVid | `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/syncnet_python/SyncNetInstance.py`; `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/spot_audiosyncfilter.py`; `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/face_handler.py`; `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/iqa_handler.py`; `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/dwpose_handler.py`; `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/dwpose/wholebody.py` |
| Hallo-Live | `third_party/sync/eval_sync_conf.py`; `third_party/sync/syncnet_detect.py`; `third_party/sync/syncnet/syncnet_eval.py`; `third_party/videoalign/inference.py`; `third_party/audiobox_aesthetics/infer.py`; `tools/download_models.sh` |

Public references:

- https://github.com/fudan-generative-vision/Hallo-Live
- https://raw.githubusercontent.com/fudan-generative-vision/Hallo-Live/main/tools/download_models.sh
- https://raw.githubusercontent.com/fudan-generative-vision/Hallo-Live/main/third_party/sync/eval_sync_conf.py

## 3. Candidate Metric Map

| Metric | Candidate implementation | Signature / I/O | Deps / GPU / weights | Adopt |
| --- | --- | --- | --- | --- |
| Sanity | no direct reference implementation found | input: generated video, expected profile; output: fps, frames, size, duration, audio info, black/frozen rates, coverage, status/error | ffmpeg/ffprobe, cv2/decord/torchaudio; CPU; no weights | reimplement from formula |
| Face detection / tracking | TalkVid InsightFace + CPUFaceHandler; Hallo-Live S3FD tracking reference | TalkVid `FaceHandler.inference(image)->dict`; CPU `detect(image)->bboxs,scores` | InsightFace antelopev2 ONNX exists locally; GPU preferred | thin wrapper around InsightFace, not direct FaceHandler because it rejects multi-face |
| FaceSim | TalkVid InsightFace embeddings | input: ref image, selected face track frames; output: mean/min/std, detection rate, per-frame scores, track id, switch count, coverage | `insightface`, `onnxruntime`, local antelopev2 weights | thin wrapper |
| Sync-C / Sync-D | TalkVid/Baselines SyncNet; Hallo-Live SyncNetEval + S3FD crop pipeline | `SyncNetInstance.evaluate(opt, videofile)->offset,minval,conf`; Hallo `SyncNetEval.evaluate(video_path,...)->offset,min_dist,conf` | torch, ffmpeg, cv2, scipy, python_speech_features; GPU preferred; `syncnet_v2.model` exists locally | thin wrapper; use selected face track crop; record audio source/crop |
| Sliding Sync | same SyncNet backend over windows | input: video windows + audio windows; output: window scores, valid window coverage, curve artifact | same as SyncNet | wrapper over global Sync |
| IQA | Baselines Q-Align style; TalkVid TopIQ | Baselines `average_score_of_video(model,path)->quality,aesthetics`; TalkVid `IQAHandler.inference(image)->float` | Q-Align model path missing; TopIQ weight exists but handler relies on `pyiqa` | use backend-aware variants; TopIQ for IQA candidate, Q-Align only if model path provided |
| AES | Baselines Q-Align aesthetics candidate | output: aesthetics score | `/mnt/workspace/AIGC/model_zoo/one-align` not found | defer until valid video AES model path |
| AKD / PCK | TalkVid DWPose ONNX | `DWPoseHandler(video_frames,dwpose_pt=None)->list[{pose,scores}]` | onnxruntime, controlnet_aux, cv2, torch; CPU/GPU; DWPose ONNX weights exist | thin wrapper, optional |
| FID / FVD | Baselines extracted script | dataset-level FID/FVD | external Diffsynth/FVD paths appear stale/missing | skip for MVP |
| TA / VideoAlign | Hallo-Live VideoAlign reward | `VideoVLMRewardInference.reward(video_paths,prompts)->VQ/MQ/TA/Overall` | heavy VLM reward checkpoint required | skip MVP |
| AudioBox AES | Hallo-Live AudioBox aesthetics | audio CE/CU/PC/PQ, not video AES | HF/local checkpoint required | skip for video quality MVP |

## 4. Weight Paths Found

| Model | Path / source |
| --- | --- |
| SyncNet | `/mnt/data/cv/yutan/Baselines/evaluation/pretrained_models/syncnet_v2.model`; `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/syncnet_python/detectors/syncnet_v2.model` |
| InsightFace antelopev2 | `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/models/insightface/models/antelopev2/` |
| DWPose | `/mnt/data/cv/zhangziwei/DataProject/DWPose/ckpts/yolox_l.onnx`; `/mnt/data/cv/zhangziwei/DataProject/DWPose/ckpts/dw-ll_ucoco_384.onnx` |
| TopIQ | `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/models/model/topiq_nr_gfiqa_res50-d76bf1ae.pth` |
| Q-Align / One-Align | default `/mnt/workspace/AIGC/model_zoo/one-align` does not exist |
| Hallo-Live reward | HF: `ByteDance/LatentSync-1.6`, `facebook/audiobox-aesthetics`, `KlingTeam/VideoReward`, `Qwen/Qwen2-VL-2B-Instruct` |

## 5. License / Copy Risk

| Source | Risk |
| --- | --- |
| TalkVid SyncNet | MIT-like license present under `syncnet_python/LICENSE.md`; small wrapper/import ok |
| TalkVid DWPose / Face / IQA | no top-level license found; prefer import/wrapper, avoid copying |
| Baselines evaluation | no top-level Baselines license found; VBench Apache-2.0 and StableAvatar MIT exist, but extracted eval files have unclear provenance |
| Hallo-Live | root MIT; third-party sync has Apache headers and adapted SyncNet code; AudioBox files reference Meta license; avoid copying large code |

## 6. MVP Implementation Order

1. Sanity check: local formula, no model dependency.
2. Face detection/tracking: InsightFace detections, stable selected track, artifacts.
3. FaceSim sampled: cosine similarity on selected track, coverage/degraded rules.
4. Sync global: SyncNet on selected face/mouth crop, explicit audio source and crop policy.
5. Benchmark target/audio sync validation: same Sync backend before trusting generated Sync.
6. Sliding-window Sync: after global Sync is stable.
7. IQA backend: start with `iqa_topiq_sampled` if `pyiqa` works; add `iqa_qalign_sampled` / `aes_qalign_sampled` only after valid model path.
8. AKD/PCK optional: DWPose extraction, normalized-image AKD/PCK@0.05, only when target pose is available.
9. Skip FID/FVD, VideoAlign TA, AudioBox AES for MVP.

## 7. Uncertain Points

- Q-Align / One-Align video IQA/AES model path is missing.
- Hallo-Live GitHub API was rate-limited; raw key files were checked, but full tree listing was not verified.
- Hallo-Live `SyncNetDetector` signature requires `ckpt_path`, while `eval_sync_conf.py` instantiates without it; needs validation before reuse.
- `pyiqa` may download/cache weights unless explicitly configured.
- DWPose paths are hardcoded in TalkVid; eval agent should make them config fields.
- FaceSim should not reuse TalkVid `FaceHandler` directly because it returns empty for multi-face frames.
- AKD/PCK requires confirming `posepath` format and exact GT/generated frame alignment.

## 8. Needed From User

- Valid Q-Align / One-Align model path if video AES is required in MVP.
- Whether TopIQ-only IQA is acceptable before AES is ready.
- Confirmation that local InsightFace/DWPose/SyncNet weights are approved for eval use.
- Example `inference_manifest.jsonl` or generated output dir for the first smoke run.
- If downloading missing HF reward/aesthetic models is desired, provide HF access/token if private or rate-limited.

## 9. Next Minimal Plan

Implement no metrics yet. First code milestone should create wrappers that always return:

`value/null`, `status`, `error`, `coverage`, `warnings`, `metric_provenance`, and artifact paths.

Start with `sanity`, `face_tracking`, `facesim_sampled`, and `sync_global`; keep realtime metrics only in backlog.
