# FaceSim Sampled InsightFace AntelopeV2 CPU V1

## Metric Variant

- Name: `facesim_sampled`
- Variant: `facesim_sampled_insightface_antelopev2_cpu_v1`
- Cost level: `medium`
- Scope: sampled identity consistency between generated frames and the actual generation reference image.

## Backend

- Implementation: pip `insightface`
- Backend model pack: InsightFace `antelopev2`
- Provider/device: `CPUExecutionProvider`, `cpu`
- Model root: `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/models/insightface`
- Detector: `scrfd_10g_bnkps.onnx`
- Embedding model: `glintr100.onnx`

Required files are checked before model initialization:

```text
models/antelopev2/scrfd_10g_bnkps.onnx
models/antelopev2/glintr100.onnx
```

The metric must not implicitly download model packs or switch to another InsightFace model.

## Policies

- Reference policy default: `auto_generation_ref`
- Allowed automatic reference sources:
  - `manifest.prepared_ref`
  - `manifest.materialized_ref`
  - `manifest.ref` only when the manifest or resolved config explicitly marks it as the generation reference
- `benchmark imgpath` is allowed only with `facesim_reference_policy=benchmark_img` and `reference_mode=benchmark_img`
- Frame extraction: ffmpeg sampled frame extraction to `artifacts/<sample_id>/facesim/frames/`
- OpenCV video IO: not used
- Frame sampling: uniform sampled frames, configured by `--facesim_num_frames`
- Selected face policy: `largest_center_face`
- `allow_per_frame_best_face=false`

## Outputs

Scores contain only identity similarity values:

```json
{
  "facesim_mean": "float|null",
  "facesim_min": "float|null",
  "facesim_std": "float|null",
  "worst_frame_facesim": "float|null"
}
```

Coverage contains detection coverage:

```json
{
  "valid": "int",
  "total": "int",
  "rate": "float",
  "min_required": "float",
  "min_valid_frames": "int",
  "sampled_frame_count": "int",
  "valid_frame_count": "int"
}
```

Diagnostics include:

- `identity_drift_slope`
- `identity_drift_direction=stable|degrading|improving|unknown`
- `face_detection_rate`
- `reference_face_count`
- `generated_multi_face_detected_count`
- `face_track_status=not_implemented_mvp`

`identity_drift_slope` is a diagnostic field and is not part of the main paper table.

## Artifacts

Each sample writes:

```text
artifacts/<sample_id>/facesim/
  frames/
  reference_face_debug.png
  sampled_frames_contact_sheet.jpg
  face_boxes_debug.jpg
  facesim_per_frame.jsonl
  facesim_error.json
  frame_extract_command.txt
  frame_extract_stdout.txt
  frame_extract_stderr.txt
  sampled_frame_paths.json
  input_video_probe.json
```

## Known Limitations

- No stable face track yet.
- Multi-face cases only record warnings and use `largest_center_face`.
- Sampled metric, not dense per-frame evaluation.
- CPU-only backend in the current environment, so runtime cost is medium.
- Identity drift uses sampled FaceSim values and should be treated as a diagnostic trend, not a standalone quality score.

## Validation Summary

Validation runs performed during MVP and quick-admission hardening:

| Case | Expected | Result |
| --- | --- | --- |
| Missing reference | `failed`, null scores, `conditioning_mismatch` | Passed |
| No-face reference | `failed`, null scores, `reference_face_detection` | Passed |
| Black generated video | `failed`, null scores, `generated_face_detection` | Passed |
| Partial no-face generated video | `degraded`, numeric scores retained, metric coverage warning | Passed |
| Wrong reference sanity | Correct reference should exceed wrong reference by > 0.3 on sampled checks | Passed on validation cases |
| Real manifest smoke | `ok` or clear failure reason | Passed on 3-sample smoke |

## Quick Admission Result

Admission run:

```text
manifest: inference_outputs/audio_i2v_batch_20260608_1610/manifest.jsonl
requested max_samples: 20
available samples: 11
metrics: sanity, facesim_sampled
facesim_num_frames: 8
reference_policy: auto_generation_ref
output_dir: /tmp/eval_facesim_admission
```

Summary:

| Field | Value |
| --- | --- |
| ok_count | 11 |
| degraded_count | 0 |
| failed_count | 0 |
| skipped_count | 0 |
| mean `facesim_mean` over ok-only | 0.8226608001 |
| mean `facesim_min` over ok-only | 0.6738264508 |
| mean coverage over ok+degraded | 1.0 |
| reference source distribution | `manifest.prepared_ref`: 11 |
| generated multi-face detected count total | 0 |
| aggregate provenance status | `ok_single_provenance` |

Wrong-reference sanity:

```text
output: /tmp/eval_facesim_wrong_ref_sanity/wrong_ref_sanity.csv
threshold: delta > 0.3
passed: 3/3
```

Decision:

- `facesim_sampled` is admitted to the default `quick` preset.
- `sync_global` remains excluded from default `quick` and stays in `quick_lipsync` / `full`.
