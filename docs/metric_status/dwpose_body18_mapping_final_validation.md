# DWPose Body18 Mapping Final Validation

## Final Conclusion

- `body18_mapping_status`: `validated`
- OpenPose compatibility: `compatible_with_documented_adapter_transform`
- Supported next metric family: `akd_pck_body18_norm_img_0p05_inbounds_v1`

The raw DWPose/RTMPose whole-body keypoint order is not directly OpenPose body18. TalkVid applies a deterministic adapter transform in `wholebody.py`: insert a neck joint, then reorder selected MMPose/COCO WholeBody joints into OpenPose body18 slots. The eval-agent adapter mirrors that transform in `eval_agent/third_party_adapters/talkvid_dwpose.py`.

## Counts

- `raw_keypoint_count = 134` after inserting neck.
- `body_keypoint_count = 18`
- `foot_keypoint_count = 6`
- `face_keypoint_count = 68`
- `hand_keypoint_count = 42`

## Evidence Source Files

- `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/dwpose/onnxpose.py`
- `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/dwpose/wholebody.py`
- `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/dwpose/__init__.py`
- `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/src/handler/dwpose/util.py`
- `/mnt/data/nlp/user/qiaoqian/TalkVid/dataprocess/draw_dwpose.py`
- `eval_agent/third_party_adapters/talkvid_dwpose.py`

## Raw Keypoint Order Source

`onnxpose.py` decodes RTMPose SimCC outputs and rescales keypoints back to image pixel coordinates. `wholebody.py` receives those MMPose/COCO-WholeBody-style keypoints and scores from `inference_pose`.

## Neck Insert Rule

`wholebody.py` computes neck as the mean of original shoulder keypoints `[5, 6]`:

```python
neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
neck[:, 2:4] = np.logical_and(
    keypoints_info[:, 5, 2:4] > 0.3,
    keypoints_info[:, 6, 2:4] > 0.3
).astype(int)
new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
```

The neck coordinate is therefore the midpoint of left/right shoulders in pixel coordinate space. Its score is a binary visibility-style confidence derived from both shoulders being above `0.3`.

## Reorder Rule

TalkVid uses:

```python
mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
```

Transform table:

| OpenPose index | Joint | Source |
|---:|---|---|
| `1` | `neck` | inserted/reordered source index `17` |
| `2` | `right_shoulder` | inserted/reordered source index `6` |
| `3` | `right_elbow` | inserted/reordered source index `8` |
| `4` | `right_wrist` | inserted/reordered source index `10` |
| `6` | `left_elbow` | inserted/reordered source index `7` |
| `7` | `left_wrist` | inserted/reordered source index `9` |
| `8` | `right_hip` | inserted/reordered source index `12` |
| `9` | `right_knee` | inserted/reordered source index `14` |
| `10` | `right_ankle` | inserted/reordered source index `16` |
| `12` | `left_knee` | inserted/reordered source index `13` |
| `13` | `left_ankle` | inserted/reordered source index `15` |
| `14` | `right_eye` | inserted/reordered source index `2` |
| `15` | `left_eye` | inserted/reordered source index `1` |
| `16` | `right_ear` | inserted/reordered source index `4` |
| `17` | `left_ear` | inserted/reordered source index `3` |

Indices `0`, `5`, and `11` remain in place from the original order and correspond to `nose`, `left_shoulder`, and `left_hip`.

## Body18 Indices And Names

| Index | Joint name | Storage |
|---:|---|---|
| 0 | `nose` | after TalkVid transform index `0` |
| 1 | `neck` | after TalkVid transform index `1` |
| 2 | `right_shoulder` | after TalkVid transform index `2` |
| 3 | `right_elbow` | after TalkVid transform index `3` |
| 4 | `right_wrist` | after TalkVid transform index `4` |
| 5 | `left_shoulder` | after TalkVid transform index `5` |
| 6 | `left_elbow` | after TalkVid transform index `6` |
| 7 | `left_wrist` | after TalkVid transform index `7` |
| 8 | `right_hip` | after TalkVid transform index `8` |
| 9 | `right_knee` | after TalkVid transform index `9` |
| 10 | `right_ankle` | after TalkVid transform index `10` |
| 11 | `left_hip` | after TalkVid transform index `11` |
| 12 | `left_knee` | after TalkVid transform index `12` |
| 13 | `left_ankle` | after TalkVid transform index `13` |
| 14 | `right_eye` | after TalkVid transform index `14` |
| 15 | `left_eye` | after TalkVid transform index `15` |
| 16 | `right_ear` | after TalkVid transform index `16` |
| 17 | `left_ear` | after TalkVid transform index `17` |

## Skeleton Edges

The overlay skeleton uses the TalkVid/OpenPose body limb sequence from `draw_dwpose.py` and `src/handler/dwpose/util.py`, converted from 1-based to 0-based:

```json
[[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
```

## Coordinate Convention

- `keypoints_xy_pixel`: pixel `(x, y)` coordinates in the extracted frame, `x` rightward and `y` downward.
- `keypoints_xy_norm`: normalized `(x / width, y / height)`.
- Out-of-frame normalized coordinates may be below `0` or above `1`; these must be excluded from primary AKD/PCK, not clipped into the metric.
- Overlays may clip points to the visible canvas for visualization only.

## Confidence Convention

- Scores are per-keypoint confidence values returned by DWPose/RTMPose after TalkVid transform.
- Neck confidence is binary `0/1`, based on both shoulders passing `0.3`.
- Existing extractor and proposed AKD/PCK policy use `visibility_score_min = 0.3`.
- Some TalkVid drawing utilities use `0.5` only for visualization.

## Visual Validation

Overlay review pack:

- `eval_outputs/body18_final_validation_v1/body18_overlay_review.html`
- Generated/target side-by-side included for one key-preserving actual sample.
- Additional DWPose admission samples included for variation.

The labeled overlays show that left/right shoulders, elbows, wrists, hips, knees, ankles, face anchor points, and inserted neck correspond to the documented OpenPose body18 transform.

## Programmatic Consistency Checks

Summary:

```json
{
  "sample_count": 5,
  "samples_with_warnings": 2,
  "aggregate_warnings": [
    "leg_chain_pass_rate below 0.7",
    "oob_body18_rate above 0.35; expected in cropped/half-body samples but AKD/PCK must exclude OOB points"
  ],
  "source_transform_evidence": "deterministic_neck_insert_and_openpose_reorder_in_talkvid_wholebody_py",
  "openpose_compatibility": "compatible_with_documented_adapter_transform",
  "recommended_body18_mapping_status": "validated"
}
```

Warnings from generated artifacts are treated as pose-estimation/content diagnostics, not mapping blockers. The source transform evidence is deterministic and matches the drawing skeleton.

## Final Gate

The mapping can be upgraded from `provisional` to `validated` for the documented TalkVid adapter transform. Formal AKD/PCK implementation can now rely on `body18_mapping_status=validated`, provided strict benchmark binding, frame alignment, visibility, and OOB policies from the design documents are enforced.
