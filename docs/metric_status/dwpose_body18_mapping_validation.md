# DWPose Body18 Mapping Validation

## Reference Files

- `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/dwpose/wholebody.py`
- `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/dwpose/onnxpose.py`
- `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/draw_dwpose.py`

## Raw Keypoint Count

Admission outputs show `raw_keypoint_count=134` for valid samples. This is consistent with COCO-WholeBody style output after inserting a neck keypoint.

## Body18 Names

```json
[
  "nose",
  "neck",
  "right_shoulder",
  "right_elbow",
  "right_wrist",
  "left_shoulder",
  "left_elbow",
  "left_wrist",
  "right_hip",
  "right_knee",
  "right_ankle",
  "left_hip",
  "left_knee",
  "left_ankle",
  "right_eye",
  "left_eye",
  "right_ear",
  "left_ear"
]
```

## Body18 Mapping

`wholebody.py` inserts a neck keypoint and then reorders a subset of raw MMPose/COCO keypoints into OpenPose-style body18 positions.

```json
{
  "inserted_neck_index": 17,
  "mmpose_idx": [
    17,
    6,
    8,
    10,
    7,
    9,
    12,
    14,
    16,
    13,
    15,
    2,
    1,
    4,
    3
  ],
  "openpose_idx": [
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    10,
    12,
    13,
    14,
    15,
    16,
    17
  ],
  "unchanged_openpose_indices": [
    0,
    5,
    11
  ]
}
```

Interpretation:

- Index `0` remains nose.
- Index `1` is inserted neck.
- Indices `2-4` are right shoulder/elbow/wrist.
- Index `5` remains left shoulder.
- Indices `6-7` are left elbow/wrist.
- Indices `8-10` are right hip/knee/ankle.
- Index `11` remains left hip.
- Indices `12-13` are left knee/ankle.
- Indices `14-17` are right/left eye and right/left ear.

## Skeleton Edges Used In Overlay

```json
[
  [
    1,
    2
  ],
  [
    1,
    5
  ],
  [
    2,
    3
  ],
  [
    3,
    4
  ],
  [
    5,
    6
  ],
  [
    6,
    7
  ],
  [
    1,
    8
  ],
  [
    8,
    9
  ],
  [
    9,
    10
  ],
  [
    1,
    11
  ],
  [
    11,
    12
  ],
  [
    12,
    13
  ],
  [
    1,
    0
  ],
  [
    0,
    14
  ],
  [
    14,
    16
  ],
  [
    0,
    15
  ],
  [
    15,
    17
  ]
]
```

These edges intentionally match the OpenPose body18 convention.

## Coordinate Convention

- `keypoints_xy_pixel`: pixel coordinates in sampled-frame resolution.
- `keypoints_xy_norm`: normalized coordinates, `x / image_width` and `y / image_height`.
- DWPose may produce points outside `[0,1]` when a predicted keypoint lies outside the visible image or when low-confidence wholebody points are present. These are reported by the schema check as warnings.

## Confidence Convention

- `keypoint_scores`: DWPose SimCC confidence scores.
- Body visibility diagnostics use `keypoint_confidence_threshold=0.3`.
- `mean_body_keypoint_confidence` averages visible body keypoints only.

## Validation Method

1. Code review of `wholebody.py` neck insertion and body18 reorder.
2. Code review of `draw_dwpose.py` skeleton edge order.
3. Admission run over sampled generated videos.
4. Schema round-trip validation of `keypoints_sampled.json`.
5. Manual visual review pack generated at `eval_outputs/dwpose_admission_v1/dwpose_overlay_review.html`.

## Validation Examples

- Admission directory: `eval_outputs/dwpose_admission_v1`
- Overlay review: `eval_outputs/dwpose_admission_v1/dwpose_overlay_review.html`
- Schema check structural pass: `True`
- Strict coordinate-bounds pass: `False`
- Schema warning count: `103`

## Conclusion

`provisional`

The code-level mapping is consistent with OpenPose body18, and overlays are generated for visual inspection. However, the mapping should remain `provisional` for AKD/PCK because target pose extraction, frame alignment, and benchmark-side coordinate policy have not been implemented or validated. Do not mark `akd_pck_ready=true` yet.

## AKD/PCK Preflight v1 Appendix


Run directory: `/mnt/data/nlp/user/qiaoqian/newproject/wan2.2_5B_fp8/eval_outputs/akd_pck_preflight_v1`

Evidence added by AKD/PCK preflight:

- Target-side DWPose extraction status: `True`
- Generated-side DWPose reuse/extraction status: `True`
- Frame alignment status: `True`
- Identical-video debug status: `passed`
- Wrong-alignment debug status: `passed`
- OOB/visibility policy status: `ok`

The raw keypoint count remains `134`, with body18 + hands + face. The body18 names and OpenPose-style skeleton edges remain the same as listed above. The preflight verifies that the normalized-coordinate distance path can produce the expected identical-video result and can detect a shifted sampled-index alignment on the tested materialized clips.

Conclusion: `provisional`

Even with successful target-side extraction and debug sanity checks, this document keeps `body18_mapping_status=provisional` and `akd_pck_ready=false` until a human overlay review and task-level target binding policy are accepted for formal paired AKD/PCK reporting.

## AKD/PCK Binding Validation v1 Appendix

Run directory: `eval_outputs/akd_pck_binding_validation_v1`

### Evidence Chain

- Raw keypoint count: `134`
- Body keypoint count: `18`
- Hand keypoint count: `42`
- Face keypoint count: `68`
- Body18 indices: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]`
- Body18 names: `nose`, `neck`, `right_shoulder`, `right_elbow`, `right_wrist`, `left_shoulder`, `left_elbow`, `left_wrist`, `right_hip`, `right_knee`, `right_ankle`, `left_hip`, `left_knee`, `left_ankle`, `right_eye`, `left_eye`, `right_ear`, `left_ear`
- Skeleton edges: OpenPose-style body18 edges listed above.
- Coordinate convention: `keypoints_xy_pixel` plus `keypoints_xy_norm = pixel / image_size`.
- Confidence convention: DWPose SimCC scores; current body visibility threshold is `0.3`.

### Source Code Paths Checked

- `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/dwpose/onnxpose.py`
- `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/dwpose/wholebody.py`
- `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/draw_dwpose.py`
- Local adapter: `eval_agent/third_party_adapters/talkvid_dwpose.py`
- Local extractor: `eval_agent/metrics/dwpose.py`

### Neck Insert / Reorder Evidence

The local adapter mirrors TalkVid's low-level flow:

1. Run YOLOX person detector.
2. Run DWPose pose estimator on the selected person bbox.
3. Insert neck as the mean of left/right shoulder.
4. Reorder the body subset into OpenPose-style body18 positions using the `mmpose_idx` to `openpose_idx` mapping documented above.
5. Keep wholebody keypoints after the body subset, producing `134` raw keypoints.

### Overlay Review Evidence

Labeled overlay review: `eval_outputs/akd_pck_binding_validation_v1/body18_labeled_overlay_review.html`

The review page shows 3 samples with generated and target labeled overlays. Each overlay displays:

- selected sampled frame
- selected person bbox
- body18 skeleton
- body18 index
- inferred body18 joint name
- keypoint confidence
- OOB keypoints clipped for visualization only

### Binding / Gate Context

Binding validation result:

- validated target bindings: `0`
- debug-only target bindings: `3`
- formal target binding: `not validated`

Metric gate result:

- target binding validated: `false`
- body18 mapping validated: `false`
- frame alignment validated: `true`
- OOB policy frozen: `false`
- formal AKD/PCK allowed: `false`

### Conclusion

`provisional`

The body18 code path and labeled overlays are strong evidence that the body18 ordering is OpenPose-style and usable for debugging. However, this document still does not mark the mapping as `validated`, because formal AKD/PCK also requires accepted target binding and final human review of labeled overlays. Current next-step metric, if any, should use the experimental variant `akd_pck_body18_norm_img_0p05_inbounds_exp_v1`, not the formal paper metric.
