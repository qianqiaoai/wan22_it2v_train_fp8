# dwpose_sampled_extractor_cpu_v1

## Scope

`dwpose_sampled_extractor_cpu_v1` is a sampled DWPose extraction readiness diagnostic. It is not AKD, PCK, pose similarity, temporal pose jitter, hand/object contact, or a paper pose-accuracy metric.

It is not included in `quick` or `full` presets and must be requested explicitly as `--metrics dwpose_sampled`.

## Backend

- Implementation source: TalkVid low-level DWPose adapter
- Reused read-only files:
  - `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/dwpose/onnxdet.py`
  - `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/dwpose/onnxpose.py`
- High-level `DWPoseHandler` / `DWposeDetector` is not imported because the current environment lacks `controlnet_aux`.
- Video decoding uses ffmpeg sampled-frame extraction, not `cv2.VideoCapture`.

## Local Models

- Detector: `yolox_l`
  - Path: `/mnt/data/cv/zhangziwei/DataProject/DWPose/ckpts/yolox_l.onnx`
  - SHA256: `7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411`
- Pose model: `dw-ll_ucoco_384`
  - Path: `/mnt/data/cv/zhangziwei/DataProject/DWPose/ckpts/dw-ll_ucoco_384.onnx`
  - SHA256: `724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843`

## Provider

- Default requested provider: `cpu`
- Selected provider: `CPUExecutionProvider`
- Provider policy: `explicit_cpu_default`
- CUDA fallback: disabled
- If `--dwpose_provider cuda` is requested but `CUDAExecutionProvider` is unavailable, the metric fails with null scores.

Current environment discovery showed:

- `onnxruntime==1.18.1`
- available providers: `AzureExecutionProvider`, `CPUExecutionProvider`
- `torch.cuda.is_available=false`

## Sampling and Artifacts

- Default sampled frames: `8`
- Sampling policy: uniform by frame index
- Artifacts:
  - `frames/`
  - `sampled_frames_contact_sheet.jpg`
  - `pose_overlay_contact_sheet.jpg`
  - `dwpose_per_frame.jsonl`
  - `keypoints_sampled.json`
  - `frame_extract_command.txt`
  - `frame_extract_stdout.txt`
  - `frame_extract_stderr.txt`
  - `sampled_frame_paths.json`
  - `dwpose_error.json` on failure

## Keypoint Format

`keypoints_sampled.json` stores both pixel and normalized coordinates:

- `keypoints_xy_pixel`
- `keypoints_xy_norm`
- `keypoint_scores`
- `selected_person_bbox`
- `raw_keypoint_count`
- `body_keypoint_count`
- `hand_keypoint_count`
- `face_keypoint_count`
- `body18_indices`
- `body18_mapping_status`

The current body-18 mapping is `provisional`; therefore:

- `akd_pck_ready=false`
- blockers:
  - `target_pose_extraction_not_implemented`
  - `frame_alignment_not_implemented`
  - `body18_mapping_needs_validation`

## Status

- `failed`: unreadable video, frame extraction failure, missing models, missing ONNXRuntime, unavailable requested provider, or no detected person in any sampled frame.
- `degraded`: some frames have pose, but detection coverage is below `min_pose_detection_rate`.
- `ok`: sampled frames meet pose coverage threshold.

Failed/skipped results keep all numeric score fields as `null`.
