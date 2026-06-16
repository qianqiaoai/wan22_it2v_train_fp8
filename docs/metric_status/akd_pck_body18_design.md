# AKD/PCK Body18 Design

## Scope

This document defines the planned AKD/PCK protocol for body18 pose accuracy. It is a design document only. No AKD/PCK implementation or numeric reporting is enabled until the readiness conditions at the end of this document are satisfied.

Current prerequisite status:

- DWPose generated-video extractor: `dwpose_sampled_extractor_cpu_v1`
- Extractor admission: passed on 11 generated samples
- `raw_keypoint_count`: 134
- `body_keypoint_count`: 18
- `hand_keypoint_count`: 42
- `face_keypoint_count`: 68
- `body18_mapping_status`: `provisional`
- Strict coordinate bounds check: `false`
- `akd_pck_ready`: `false`

## 1. Task Applicability

AKD/PCK may run only when all of the following are true:

- Generated video exists and is readable.
- A target video or target pose exists.
- Target pose and generated video have a defined frame alignment.
- `keypoint_set` is frozen.
- `coordinate_space` is frozen.
- Body18 mapping is `validated`.

If a task has no pose condition but has a target video, AKD/PCK should be labeled as a paired reconstruction or reenactment diagnostic, not as a universal generation-quality metric. It should not be used to penalize open-ended generation where the target pose is not intended to be reproduced.

Recommended task labels:

| Task context | AKD/PCK interpretation |
|---|---|
| audio + ref image + pose + text -> video | Pose-control accuracy metric |
| audio + ref image + target reenactment -> video | Paired reenactment diagnostic |
| audio + ref image + text only -> video with no pose condition | Optional paired diagnostic only if target video is explicitly meaningful |
| unpaired generated videos | Not applicable |

## 2. Target Pose Source Policy

Supported `target_pose_source` values:

| Source | Input fields | Needs ffmpeg frames | Needs DWPose extraction | Coordinate space | Comparable with generated pose | Failure/status rule |
|---|---|---:|---:|---|---|---|
| `materialized_target_clip_dwpose` | `materialized_target_clip`, `target_clip_path`, or benchmark materialized target clip path | Yes | Yes | Pixel + normalized xy from the same extractor | Yes, preferred if generated and target are materialized to the same eval profile or compared in normalized xy | If target clip missing/unreadable: `failed` for pose-conditioned tasks, `skipped` for non-pose tasks if policy allows |
| `benchmark_posepath_materialized` | `posepath` or `materialized_posepath` | No if pose file already contains comparable keypoints; yes only if pose file references video frames | No if trusted pose is already in body18 normalized xy; otherwise conversion required | Must declare source coordinate space | Comparable only after schema, mapping, and normalization are validated | If schema/mapping unknown: `failed` or `blocked`, no scores |
| `manifest_target_pose` | `target_pose`, `target_pose_path`, `manifest_target_pose` | No if precomputed pose artifact; yes if it points to target video | Depends on manifest artifact type | Must be declared in manifest | Comparable only if provenance-compatible with generated DWPose | If provenance/mapping incompatible: `failed`, no scores |
| `disabled` | None | No | No | None | Not applicable | Metric is `skipped` |

MVP recommendation:

```json
{
  "target_pose_source": "materialized_target_clip_dwpose"
}
```

Reason: using the same DWPose extractor for generated and target clips minimizes detector/backend mismatch and makes provenance comparison straightforward.

## 3. Generated Pose Source Policy

Supported `generated_pose_source` values:

| Source | Meaning | Requirements |
|---|---|---|
| `generated_video_dwpose` | Run the DWPose extractor on the generated video | Default. Uses ffmpeg sampled frames and `dwpose_sampled_extractor_cpu_v1` provenance. |
| `existing_dwpose_artifact` | Reuse a prior generated-side DWPose artifact | Allowed only when provenance is compatible. |

Default:

```json
{
  "generated_pose_source": "generated_video_dwpose"
}
```

Reusing `existing_dwpose_artifact` requires all of the following:

- Same detector model hash.
- Same pose model hash.
- Same ONNXRuntime provider or explicitly compatible provider policy.
- Same frame sample policy.
- Same `num_frames`.
- Same coordinate space.
- Same body18 mapping.
- Same `provenance_fingerprint`, or a compatible fingerprint declared by a compatibility policy.

If compatibility cannot be proven, the metric must recompute DWPose or fail with a concrete provenance mismatch error.

## 4. Frame Alignment Policy

Supported `frame_alignment` values:

| Policy | Meaning | Use case |
|---|---|---|
| `sampled_index_match` | Generated and target use the same sampled positions in their own materialized clips | MVP for uniform 8-frame sampled diagnostics |
| `timestamp_nearest` | Match each generated sampled timestamp to the nearest target timestamp | Useful when fps differs but durations are aligned |
| `materialized_frame_index` | Match by frame index after both clips are materialized to the same fps/frame count | Preferred for paper-ready protocol |
| `full_sequence_resampled` | Resample both pose sequences to a common timeline before scoring | Future dense protocol |

MVP recommendation:

```json
{
  "frame_alignment": "sampled_index_match"
}
```

Requirements:

- Record `generated_frame_indices`.
- Record `target_frame_indices`.
- Record `generated_timestamps_sec`.
- Record `target_timestamps_sec`.
- Record `max_timestamp_diff_sec`.
- Record `frame_alignment_status`.

If generated and target fps/duration differ, the agent must first use a materialized target clip aligned to the generation profile. It must not directly compare generated `480x832 / 20fps / 81 frames` against an original benchmark video with different fps/duration unless an explicit alignment policy materializes both to a common timeline.

MVP frame-alignment failure conditions:

- Missing target sampled frame.
- Missing generated sampled frame.
- `max_timestamp_diff_sec` exceeds configured tolerance.
- Sampled indices are generated under different policies.
- Generated and target materialization profiles are incompatible and no normalized/timestamp fallback is configured.

## 5. Coordinate Policy

Supported `coordinate_space` values:

| Coordinate space | Meaning | Use |
|---|---|---|
| `pixel_xy` | Absolute pixel coordinates in each frame | Visualization and debugging; only comparable if frame sizes match |
| `normalized_xy` | `x / image_width`, `y / image_height` | MVP primary AKD/PCK coordinate space |

MVP primary coordinate space:

```json
{
  "coordinate_space": "normalized_xy",
  "normalization_mode": "normalized_image"
}
```

Frame sizes:

- Generated image size is the generated video eval resolution.
- Target image size is the materialized target clip eval resolution.

If generated and target resolutions differ, either:

- materialize both to the same eval resolution; or
- compute primary AKD/PCK only in `normalized_xy`.

Pixel-space AKD/PCK must not be reported across mismatched resolutions.

## 6. Out-of-Frame Policy

Current DWPose admission found normalized points outside `[0,1]`. These may come from predicted keypoints outside the image, low-confidence wholebody points, or detector bbox extending outside image bounds.

Supported `out_of_frame_policy` values:

| Policy | Meaning |
|---|---|
| `exclude_from_primary_metric` | Exclude out-of-bounds keypoints from AKD/PCK computation |
| `clip_for_visualization_only` | Clip points only when drawing overlays |
| `report_oob_rate` | Report out-of-bounds rates as diagnostics |

MVP recommendation:

```json
{
  "out_of_frame_policy": "exclude_from_primary_metric",
  "visualization_policy": "clip_for_visualization_only",
  "report_oob_rate": true
}
```

Primary AKD/PCK must not silently clip keypoints before scoring.

A keypoint can participate in primary AKD/PCK only when:

- generated keypoint is in bounds;
- target keypoint is in bounds;
- generated score passes the visibility threshold;
- target score passes the visibility threshold.

Diagnostics must report:

- `generated_oob_rate`
- `target_oob_rate`
- `matched_oob_excluded_count`
- `generated_oob_keypoint_count`
- `target_oob_keypoint_count`

## 7. Visibility And Missing Keypoint Policy

Default:

```json
{
  "visibility_score_min": 0.3,
  "missing_keypoint_policy": "exclude_unmatched"
}
```

A keypoint participates in AKD/PCK only if all conditions hold:

- Generated keypoint exists.
- Target keypoint exists.
- Generated score `>= visibility_score_min`.
- Target score `>= visibility_score_min`.
- Generated keypoint is in bounds.
- Target keypoint is in bounds.

Coverage thresholds:

```json
{
  "min_valid_keypoints_per_frame": 6,
  "min_valid_frames": 1,
  "min_frame_coverage": 0.5
}
```

If matched valid keypoints are too few:

- `degraded` if there are enough valid frames to compute a diagnostic score but coverage is below threshold.
- `failed` if no valid aligned frame has enough keypoints.

Failed/skipped results must write numeric scores as `null`.

## 8. Body18 Mapping Validation

Current status:

```json
{
  "body18_mapping_status": "provisional",
  "akd_pck_ready": false
}
```

Upgrade path to `validated`:

1. Read low-level DWPose keypoint order in `onnxpose.py` and `wholebody.py`.
2. Confirm neck insertion:
   - inserted neck is mean of shoulder keypoints 5 and 6 in raw wholebody order.
3. Confirm reorder:
   - `mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]`
   - `openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]`
4. Confirm overlay skeleton edges match OpenPose body18.
5. Produce body18 names and indices.
6. Use overlay examples for manual inspection on both generated and target clips.
7. Validate target pose extraction uses the same mapping and coordinate policy.

Body18 names:

| Index | Name |
|---:|---|
| 0 | nose |
| 1 | neck |
| 2 | right_shoulder |
| 3 | right_elbow |
| 4 | right_wrist |
| 5 | left_shoulder |
| 6 | left_elbow |
| 7 | left_wrist |
| 8 | right_hip |
| 9 | right_knee |
| 10 | right_ankle |
| 11 | left_hip |
| 12 | left_knee |
| 13 | left_ankle |
| 14 | right_eye |
| 15 | left_eye |
| 16 | right_ear |
| 17 | left_ear |

AKD/PCK precondition:

```json
{
  "body18_mapping_status": "validated"
}
```

If mapping remains `provisional`, AKD/PCK implementation must remain blocked and no numeric pose accuracy should be reported.

## 9. AKD Formula

Metric variant:

```text
akd_body18_norm_img_inbounds_v1
```

For each valid matched keypoint:

```text
distance = sqrt((x_gen - x_tgt)^2 + (y_gen - y_tgt)^2)
```

Coordinates are normalized image coordinates.

Per frame:

```text
akd_frame = mean(distance over valid matched keypoints)
```

Per sample:

```text
akd = mean(akd_frame over valid frames)
```

Additional outputs:

- `akd_std`
- `valid_frame_count`
- `valid_keypoint_count`
- `matched_keypoint_rate`

Direction:

- AKD lower is better.

## 10. PCK Formula

Metric variant:

```text
pck_body18_norm_img_0p05_inbounds_v1
```

Threshold:

```json
{
  "pck_threshold": 0.05,
  "pck_threshold_units": "normalized_image"
}
```

For each valid matched keypoint:

```text
correct = distance < 0.05
```

Per frame:

```text
pck_frame = mean(correct over valid matched keypoints)
```

Per sample:

```text
pck = mean(pck_frame over valid frames)
```

Additional outputs:

- `pck_threshold`
- `pck_threshold_units = normalized_image`
- `valid_frame_count`
- `valid_keypoint_count`
- `matched_keypoint_rate`

Direction:

- PCK higher is better.

## 11. Scores / Coverage / Details Schema

Scores:

```json
{
  "akd_body18_norm_img": null,
  "akd_body18_norm_img_std": null,
  "pck_body18_norm_img_0p05": null,
  "valid_frame_pose_error_mean": null,
  "matched_keypoint_rate": null
}
```

Coverage:

```json
{
  "valid": 0,
  "total": 0,
  "rate": 0.0,
  "min_required": 0.5,
  "valid_frame_count": 0,
  "total_aligned_frame_count": 0,
  "valid_keypoint_count": 0,
  "total_possible_keypoint_count": 0
}
```

Details:

```json
{
  "keypoint_set": "body18",
  "coordinate_space": "normalized_xy",
  "normalization_mode": "normalized_image",
  "pck_threshold": 0.05,
  "pck_threshold_units": "normalized_image",
  "visibility_score_min": 0.3,
  "missing_keypoint_policy": "exclude_unmatched",
  "out_of_frame_policy": "exclude_from_primary_metric",
  "frame_alignment": {
    "policy": "sampled_index_match",
    "generated_frame_indices": [],
    "target_frame_indices": [],
    "generated_timestamps_sec": [],
    "target_timestamps_sec": [],
    "max_timestamp_diff_sec": null,
    "frame_alignment_status": "not_run"
  },
  "body18_mapping_status": "provisional",
  "generated_oob_rate": null,
  "target_oob_rate": null,
  "per_frame_metrics_path": null,
  "worst_frames": [],
  "artifacts": {}
}
```

Provenance:

```json
{
  "metric_name": "pose_accuracy",
  "metric_variant": "akd_pck_body18_norm_img_0p05_inbounds_v1",
  "generated_dwpose_provenance": {},
  "target_dwpose_provenance": {},
  "detector_model_sha256": null,
  "pose_model_sha256": null,
  "frame_alignment_policy": "sampled_index_match",
  "coordinate_policy": {
    "coordinate_space": "normalized_xy",
    "normalization_mode": "normalized_image",
    "out_of_frame_policy": "exclude_from_primary_metric"
  },
  "thresholds": {
    "visibility_score_min": 0.3,
    "pck_threshold": 0.05,
    "min_valid_keypoints_per_frame": 6,
    "min_valid_frames": 1,
    "min_frame_coverage": 0.5
  },
  "provenance_fingerprint": null
}
```

Combined AKD/PCK metric variant:

```text
akd_pck_body18_norm_img_0p05_inbounds_v1
```

## 12. Status Rules

| Condition | Status | Scores |
|---|---|---|
| Target pose unavailable in a pose-conditioned task | `failed` | all numeric scores `null` |
| Target pose unavailable in a non-pose task where paired diagnostic is optional | `skipped` | all numeric scores `null` |
| Generated pose unavailable | `failed` | all numeric scores `null` |
| Body18 mapping not validated | `failed` or `blocked` | all numeric scores `null` |
| Frame alignment failed | `failed` | all numeric scores `null` |
| No valid matched frame | `failed` | all numeric scores `null` |
| Too few valid matched keypoints but some numeric score exists | `degraded` | keep scores with warnings |
| Normal coverage and aligned poses | `ok` | numeric scores present |

Failure/degraded taxonomies:

- target/generated pose missing: `pose_accuracy`
- body18 mapping not validated: `metric_runtime` or `pose_accuracy` with `code=body18_mapping_not_validated`
- frame alignment failed: `conditioning_mismatch` or `pose_accuracy`
- too few valid keypoints: `metric_coverage`
- artifact/provenance mismatch: `metric_runtime`

## 13. Artifacts

Planned artifacts:

```text
artifacts/<sample_id>/pose_accuracy/
  generated_pose_overlay_contact_sheet.jpg
  target_pose_overlay_contact_sheet.jpg
  side_by_side_pose_overlay.jpg
  pose_error_per_frame.jsonl
  worst_pose_error_frames.jpg
  akd_pck_error.json if failed
```

`pose_error_per_frame.jsonl` should include:

- aligned generated frame index
- aligned target frame index
- timestamps
- valid body keypoint count
- excluded low-confidence count
- excluded out-of-frame count
- `akd_frame`
- `pck_frame`
- per-keypoint distance/correct flags

## 14. Validation Plan

Before implementation, require:

1. Body18 mapping validation changes from `provisional` to `validated`.
2. One generated sample and its target clip are materialized to the same fps, resolution, duration, and frame count.
3. Synthetic identical-video test:
   - generated = target
   - expected AKD close to `0`
   - expected PCK close to `1`
4. Shifted frame alignment test:
   - intentionally wrong alignment should worsen AKD/PCK.
5. Black/no-person target or generated test:
   - expected `failed` or `degraded` with null scores when no valid matched frames exist.
6. Out-of-frame keypoint test:
   - out-of-frame keypoints are excluded from primary metrics.
   - `generated_oob_rate`, `target_oob_rate`, and excluded counts are reported.
7. Provenance compatibility test:
   - generated and target DWPose artifacts must use compatible model hashes, provider policy, sampling policy, body18 mapping, and coordinate policy.

## 15. Readiness Decision

Can implement AKD/PCK now?

```text
No.
```

Exact blockers:

1. `body18_mapping_status` is still `provisional`.
2. Target pose extraction using the same DWPose extractor is not implemented.
3. Frame alignment between generated and target clips is not implemented.
4. Target/generated materialization to a common eval profile is not enforced for pose accuracy.
5. Out-of-frame handling is designed but not validated in an AKD/PCK computation path.

Can start AKD/PCK design?

```text
Yes.
```

First implementation scope after blockers are resolved:

1. Implement target-side DWPose extraction on `materialized_target_clip_dwpose`.
2. Implement `sampled_index_match` frame alignment for generated and target pose artifacts.
3. Implement body18 mapping validation gate.
4. Implement AKD/PCK computation only for in-bounds, visible, matched body18 keypoints.
5. Add synthetic identical-video and shifted-alignment tests before any real benchmark reporting.
