# AKD/PCK Body18 Metric Status

## Variant

- Metric name: `akd_pck_body18`
- Metric variant: `akd_pck_body18_norm_img_0p05_inbounds_v1`
- Implementation status: implemented
- Regression status: passed
- Admission status: admission_passed_for_dev_50samples
- Promotion status: full_pose_ready
- Preset status: available_in_full_pose
- Preset membership: `full_pose` only
- Not in existing `full`: true
- Paper table status: excluded
- Paper ready: false
- Full pose enabled after 50-sample admission: true
- Next gate: paper_standardized_profile

`full_pose_ready` means the metric passed the current 50-sample development admission and is available through the explicit `full_pose` preset. It does not make AKD/PCK part of the existing `full` preset and does not make the metric paper-table ready.

## Inputs

- Generated pose source: `generated_video_dwpose`
- Target pose source: `materialized_target_clip_dwpose`
- Required binding: strict `benchmark_key` plus `benchmark_row_hash` and existing `materialized_target_clip`
- Forbidden binding fallback: numeric row index, basename guessing, row-order fallback
- Applicability: paired/key-preserving/materialized target-clip tasks where generated video and target pose are comparable.

## Pose Backend

- Upstream extractor: `dwpose_sampled_extractor_cpu_v1`
- Keypoint set: body18
- Body18 status: validated
- Compatibility: `compatible_with_documented_adapter_transform`
- Mapping evidence: `docs/metric_status/dwpose_body18_mapping_final_validation.md`

## Admission Evidence

- Sample count: 50 key-preserving actual generated samples
- Strict binding status: 50/50 validated
- Generated videos: 50/50 exists
- Sanity status: 50/50 ok
- FaceSim status: 50/50 ok
- DWPose status: 50/50 ok
- AKD/PCK status: 50/50 ok
- Identical-video regression: passed
- Wrong-alignment regression: passed
- Mean AKD: 0.10535038797210045
- Mean PCK@0.05: 0.37024537609332775
- Mean matched keypoint rate: 0.7673611111111112
- Mean generated OOB rate: 0.0225
- Mean target OOB rate: 0.02694444444444444

## Promotion Gates

- Current status: full_pose_ready
- Existing full preset: unchanged; AKD/PCK is not included
- Full-pose preset: available as `full_pose`
- Paper table: blocked
- Paper table blockers:
  - Requires `paper_standardized` profile
  - Requires official/versioned split and split hash
  - Requires task applicability statement for paired target-video / pose-comparable evaluation

## Protocol

- Frame alignment: `sampled_index_match`
- Coordinate space: `normalized_xy`
- Normalization: normalized image width/height
- Visibility threshold: `0.3`
- PCK threshold: `0.05`
- PCK threshold units: normalized image
- Missing keypoint policy: `exclude_unmatched`
- Out-of-frame policy: `exclude_from_primary_metric`

## Scores

- `akd_body18_norm_img`: lower is better
- `akd_body18_norm_img_std`
- `pck_body18_norm_img_0p05`: higher is better
- `pck_body18_norm_img_0p05_std`
- `valid_frame_pose_error_mean`
- `matched_keypoint_rate`
- `worst_frame_akd`
- `worst_frame_pck`

## Coverage Thresholds

- `min_valid_keypoints_per_frame = 6`
- `min_valid_frame_rate = 0.5`
- `min_matched_keypoint_rate = 0.3`

Below-threshold nonzero matches return `degraded` with numeric scores. Zero valid matched keypoints returns `failed` with null scores.

## Artifacts

Per sample:

- `artifacts/<sample_id>/pose_accuracy/generated_pose_overlay_contact_sheet.jpg`
- `artifacts/<sample_id>/pose_accuracy/target_pose_overlay_contact_sheet.jpg`
- `artifacts/<sample_id>/pose_accuracy/side_by_side_pose_overlay.jpg`
- `artifacts/<sample_id>/pose_accuracy/pose_error_per_frame.jsonl`
- `artifacts/<sample_id>/pose_accuracy/worst_pose_error_frames.jpg`
- `artifacts/<sample_id>/pose_accuracy/pose_accuracy_policy.json`
- `artifacts/<sample_id>/pose_accuracy/alignment_table.csv`
- `artifacts/<sample_id>/pose_accuracy/compatibility_check.json`
- `artifacts/<sample_id>/pose_accuracy/akd_pck_error.json` if failed

## Known Limits

- Only body18 is evaluated.
- No hand, face, wholebody PCK, object contact, temporal pose jitter, or realtime metrics.
- The metric is available in `full_pose` after 50-sample development admission.
- The metric remains excluded from `quick`, `quick_lipsync`, the existing `full`, and `paper_table.csv`.
