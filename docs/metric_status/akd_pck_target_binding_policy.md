# AKD/PCK Target Binding Policy

## Scope

This policy defines when a generated video can be paired with a target video or target pose for AKD/PCK. It applies before any formal `pose_accuracy` metric is allowed to emit AKD/PCK scores.

AKD/PCK is a paired pose metric. It must not infer a target row silently from file order, basename similarity, or row index.

## Formal Binding Priority

Formal target binding must use the first available validated key source:

1. `inference_manifest.key` / `sample_id`
2. Generated manifest row field that explicitly records the benchmark key, such as `benchmark_key`, `source_benchmark_key`, `benchmark_sample_id`, or `dataset_key`
3. `benchmark_materialized/manifest.jsonl` key, when it records both materialized target paths and the original benchmark key
4. Original `benchmark.jsonl` key, when the benchmark file has an explicit stable key field
5. Derived `benchmark_key` from a saved `benchmark_index.json`, when the original benchmark lacks explicit keys and the derived key follows `docs/metric_status/benchmark_key_policy.md`

The selected key must resolve to exactly one target record. If more than one target row matches, binding is blocked.

For the current benchmark, original rows do not include explicit keys. The accepted formal sidecar identity is therefore:

```text
benchmark_key = zero_padded_row_index_6 + "_" + short_row_hash_8
```

This derived key is formal only when it is persisted in `benchmark_index.json`, propagated into `selected_samples.jsonl`, copied into `benchmark_materialized/manifest.jsonl`, and then copied into generated inference manifests as `benchmark_key`. The row index part alone is never a valid identity.

## Forbidden Defaults

The following are forbidden for formal AKD/PCK:

- Default `numeric_sample_id_to_benchmark_row`
- Default basename guessing, such as mapping `0001.mp4` to `000001.mp4`
- Silent fallback to the N-th benchmark row
- Using raw benchmark `videopath` without materializing it to the active eval configuration

Numeric binding is allowed only when the user explicitly sets:

```yaml
pose_accuracy:
  target_binding_policy: numeric_sample_id_to_benchmark_row_debug
  allow_debug_binding: true
```

When this debug policy is used, binding status must be `debug_not_for_paper`, and the resulting AKD/PCK values must be treated as experimental debug scores only.

## Binding Status

- `validated`: formal key chain is present, unique, and target materialization matches the active eval config.
- `blocked`: no formal key chain exists, or the key chain is ambiguous or inconsistent.
- `debug_only`: only explicit debug numeric binding can resolve the target.
- `debug_not_for_paper`: same as `debug_only`, but used inside metric details if experimental AKD/PCK is computed.

## Required Target Materialization

Formal AKD/PCK must compare generated and target pose under the same eval configuration:

- resolution
- fps
- frame count
- duration
- crop policy

For the current dev-native configuration, target video clips must be materialized to:

- `480x832`
- `20fps`
- `81 frames`
- `4.05s`
- `cover_center_crop`

Raw benchmark video paths are not directly comparable to generated video unless this materialization step is performed and recorded.

## Required Resolved Config

When formal or experimental AKD/PCK is implemented, the final resolved config and metric plan must include:

```yaml
pose_accuracy:
  target_binding_policy: explicit_key
  target_pose_source: materialized_target_clip_dwpose
  generated_pose_source: generated_video_dwpose
  frame_alignment: sampled_index_match
  coordinate_space: normalized_xy
  out_of_frame_policy: exclude_from_primary_metric
  visibility_score_min: 0.3
  missing_keypoint_policy: exclude_unmatched
  pck_threshold: 0.05
  pck_threshold_units: normalized_image
  min_valid_keypoints_per_frame: 6
  min_valid_frame_rate: 0.5
  min_matched_keypoint_rate: 0.3
  require_binding_validated: true
  allow_debug_binding: false
```

Experimental debug runs may override:

```yaml
pose_accuracy:
  target_binding_policy: numeric_sample_id_to_benchmark_row_debug
  require_binding_validated: false
  allow_debug_binding: true
```

Those runs must set `score_interpretation=experimental`, `not_for_paper=true`, and must not enter `paper_table`, `quick`, `full`, or model-ranking aggregates.

## Metric Gate

Formal `akd_pck_body18_norm_img_0p05_inbounds_v1` is allowed only if:

- target binding is `validated`
- body18 mapping is `validated`
- frame alignment is `validated`
- OOB / visibility / missing keypoint policy is frozen
- generated and target DWPose provenance is compatible

If any condition is not met, only the experimental variant may be considered:

`akd_pck_body18_norm_img_0p05_inbounds_exp_v1`
