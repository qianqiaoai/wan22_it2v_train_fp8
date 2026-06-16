# Third-Party Code Notices

This repository vendors small runtime glue files for offline evaluation only. Model
weights are not vendored and must be prepared separately under an external
`eval_agent_pretrained` directory.

## DWPose ONNX Glue

- adapter: `dwpose_onnx_adapter_v1`
- vendored files:
  - `eval_agent/vendor/dwpose_onnx/onnxdet.py`
  - `eval_agent/vendor/dwpose_onnx/onnxpose.py`
- reference source: TalkVid dataprocess DWPose handler
- upstream reference noted in source: `https://github.com/IDEA-Research/DWPose`
- modification summary:
  - kept low-level ONNX detector/pose inference glue local to the eval agent
  - moved runtime import away from external TalkVid/DWPose directories
  - model paths are resolved through ResourceConfig, not source-tree constants
- license_status: unknown from local reference
- redistribution_status: needs human/license review before Skill publishing
- action_required_before_skill_publishing:
  - confirm DWPose upstream license and any TalkVid modifications
  - include the correct license text if redistribution is allowed

## SyncNet v2 Runtime Glue

- adapter: `syncnet_v2_adapter_v1`
- vendored files:
  - `eval_agent/vendor/syncnet_v2/SyncNetInstance.py`
  - `eval_agent/vendor/syncnet_v2/SyncNetModel.py`
  - `eval_agent/vendor/syncnet_v2/python_speech_features/__init__.py`
  - `eval_agent/vendor/syncnet_v2/python_speech_features/base.py`
  - `eval_agent/vendor/syncnet_v2/python_speech_features/sigproc.py`
  - `eval_agent/vendor/syncnet_v2/video_face_crop.py`
- reference source: TalkVid dataprocess SyncNet runtime
- license_status:
  - `syncnet_python`: MIT license in local `syncnet_python/LICENSE.md`
  - `video_face_crop.py`: unknown from local reference
  - embedded `python_speech_features` files: needs license confirmation
- modification summary:
  - worker subprocess now imports vendored SyncNet modules from this repo
  - checkpoint path is resolved through ResourceConfig
  - mp4/frame/audio IO remains eval-agent controlled with ffmpeg paths
- redistribution_status: partial; SyncNet MIT appears redistributable, but
  `video_face_crop.py` and embedded feature extraction files need review before
  packaging.
- action_required_before_skill_publishing:
  - confirm license for `video_face_crop.py`
  - confirm license for vendored `python_speech_features` files
  - include required notices and copyright text

## Non-Vendored Model Weights

The following weights are external runtime resources and must not be committed or
bundled into the Skill package:

- DWPose ONNX detector and pose models
- SyncNet v2 checkpoint
- InsightFace antelopev2 model pack
- VBench / CLIP / MUSIQ / aesthetic checkpoints

Use `scripts/prepare_eval_agent_pretrained.py` with an explicit local source
manifest to prepare these assets outside the repository.
