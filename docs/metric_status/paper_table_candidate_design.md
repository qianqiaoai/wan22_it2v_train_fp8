# Paper Table Candidate Design

Status: design frozen, writer not implemented  
Implementation status: schema/design only  
Paper table status: unchanged  
Current date: 2026-06-14

This document defines the future `paper_table_candidate.csv` schema and gate policy for offline digital-human video evaluation. It does not implement the writer, does not generate candidate CSV files, does not modify `paper_table.csv`, and does not change any metric formula or preset.

## 1. Candidate Table Purpose

`paper_table_candidate.csv` is a paper-readiness candidate table.

It is not:

- a replacement for the existing `paper_table.csv`
- a final paper table
- an automatic claim that a run is paper-ready
- valid for ad-hoc sample subsets or legacy runs

It is only allowed under an implemented paper profile such as:

- `paper_model_native_v1`
- future implemented paper profiles

It must use a versioned official split and must carry:

- `split_sha256`
- `benchmark_index_sha256`
- metric provenance fingerprints or a combined `metric_provenance_id`

The table is a structured candidate output for human review, not the final publication artifact.

## 2. Candidate Metrics

The first candidate table may include the following metric columns.

| Metric | Column | Direction | Display | Applicability |
| --- | --- | --- | --- | --- |
| FaceSim | `facesim_mean` | `higher_is_better` | `FaceSim↑` | reference-conditioned generation with actual generation reference |
| IQA | `iqa_mean` | `higher_is_better` | `IQA↑` | generated video quality under frozen VBench local backend |
| AES | `aes_mean` | `higher_is_better` | `AES↑` | generated video aesthetic score under frozen VBench local backend |
| AKD | `akd_body18_norm_img` | `lower_is_better` | `AKD↓` | paired/key-preserving/target-video-comparable tasks only |
| PCK | `pck_body18_norm_img_0p05` | `higher_is_better` | `PCK@0.05↑` | paired/key-preserving/target-video-comparable tasks only |

Diagnostic-only metric:

- `sync_global`

Reason:

- Current `sync_global` has `score_interpretation=uncalibrated`.
- It is not a paper candidate until offset sign, discriminativeness, and benchmark GT audio/target validity are accepted.

Pending metrics:

- TA / text alignment
- VQ / MQ
- temporal flicker / temporal diagnostics
- realtime metrics

These are outside the first candidate table.

## 3. Candidate CSV Schema

The future `paper_table_candidate.csv` should contain one row per evaluated method/checkpoint/run under a compatible paper profile and official split.

### Metadata Columns

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `method` | string | yes | Method or model name shown in a paper table. |
| `run_id` | string | yes | Eval output run id or directory basename. |
| `checkpoint` | string/null | yes | Checkpoint path, checkpoint id, or `null` if unavailable. |
| `eval_profile` | string | yes | Must be an implemented paper profile, initially `paper_model_native_v1`. |
| `split_name` | string | yes | Must start with `official_`. |
| `split_version` | string | yes | Example: `v1`. |
| `split_sha256` | string | yes | Versioned split hash. |
| `benchmark_index_sha256` | string | yes | Benchmark index hash used to resolve `benchmark_key`. |
| `sample_count` | int | yes | Expected official split sample count. |
| `valid_n` | int | yes | Samples included in the primary candidate aggregate. |
| `ok_n` | int | yes | Samples with all candidate metrics used in primary aggregate, or metric-specific ok count if the writer uses per-metric validity. |
| `degraded_n` | int | yes | Samples with degraded candidate metric status. |
| `failed_n` | int | yes | Samples with failed candidate metric status. |
| `skipped_n` | int | yes | Samples with skipped candidate metric status. |
| `comparable_by_default` | bool | yes | True only if all comparability gates are satisfied. |
| `paper_ready` | bool | yes | Remains false until all paper readiness gates pass. |

### Metric Columns

| Column | Type | Direction | Display |
| --- | --- | --- | --- |
| `facesim_mean` | float/null | higher is better | `FaceSim↑` |
| `iqa_mean` | float/null | higher is better | `IQA↑` |
| `aes_mean` | float/null | higher is better | `AES↑` |
| `akd_body18_norm_img` | float/null | lower is better | `AKD↓` |
| `pck_body18_norm_img_0p05` | float/null | higher is better | `PCK@0.05↑` |

### Diagnostic Metadata Columns

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `metric_provenance_id` | string | yes | Combined stable hash over metric variants, model hashes, profile, split, materialization, and aggregation policy. |
| `face_metric_variant` | string | yes | Example: `facesim_sampled_insightface_antelopev2_cpu_v1`. |
| `quality_metric_variant` | string | yes | Example: `iqa_aes_vbench_local_sampled_cpu_v1`. |
| `pose_metric_variant` | string/null | yes | Example: `akd_pck_body18_norm_img_0p05_inbounds_v1`, or `null` if not applicable. |
| `generation_width` | int | yes | Expected generated video width under the profile. |
| `generation_height` | int | yes | Expected generated video height under the profile. |
| `generation_fps` | float | yes | Expected generated FPS under the profile. |
| `generation_frame_num` | int | yes | Expected generated frame count under the profile. |
| `materialization_policy` | string | yes | Target/ref/audio materialization policy id. |
| `target_binding_policy` | string | yes | Must be strict key-based binding for paired metrics. |
| `notes` | string/null | yes | Human-readable warnings or exclusions. |

## 4. Direction Metadata Schema

The future writer must also produce:

```text
paper_table_candidate.schema.json
```

Design:

```json
{
  "schema_version": "v1",
  "table": "paper_table_candidate.csv",
  "metric_columns": [
    {
      "column_name": "facesim_mean",
      "display_name": "FaceSim↑",
      "direction": "higher_is_better",
      "metric_variant": "facesim_sampled_insightface_antelopev2_cpu_v1",
      "eval_profile_required": "paper_model_native_v1",
      "aggregation_policy": "mean_over_ok_samples",
      "missing_value_policy": "null_excluded_not_zero",
      "degraded_policy": "excluded_from_primary_aggregate",
      "paper_ready_requirement": "profile_split_provenance_and_reference_policy_validated"
    },
    {
      "column_name": "iqa_mean",
      "display_name": "IQA↑",
      "direction": "higher_is_better",
      "metric_variant": "iqa_aes_vbench_local_sampled_cpu_v1",
      "eval_profile_required": "paper_model_native_v1",
      "aggregation_policy": "mean_over_ok_samples",
      "missing_value_policy": "null_excluded_not_zero",
      "degraded_policy": "excluded_from_primary_aggregate",
      "paper_ready_requirement": "vbench_local_hashes_sampling_and_profile_validated"
    },
    {
      "column_name": "aes_mean",
      "display_name": "AES↑",
      "direction": "higher_is_better",
      "metric_variant": "iqa_aes_vbench_local_sampled_cpu_v1",
      "eval_profile_required": "paper_model_native_v1",
      "aggregation_policy": "mean_over_ok_samples",
      "missing_value_policy": "null_excluded_not_zero",
      "degraded_policy": "excluded_from_primary_aggregate",
      "paper_ready_requirement": "vbench_local_hashes_sampling_and_profile_validated"
    },
    {
      "column_name": "akd_body18_norm_img",
      "display_name": "AKD↓",
      "direction": "lower_is_better",
      "metric_variant": "akd_pck_body18_norm_img_0p05_inbounds_v1",
      "eval_profile_required": "paper_model_native_v1",
      "aggregation_policy": "mean_over_ok_samples",
      "missing_value_policy": "null_excluded_not_zero",
      "degraded_policy": "excluded_from_primary_aggregate",
      "paper_ready_requirement": "paired_task_strict_binding_body18_mapping_and_pose_policy_validated"
    },
    {
      "column_name": "pck_body18_norm_img_0p05",
      "display_name": "PCK@0.05↑",
      "direction": "higher_is_better",
      "metric_variant": "akd_pck_body18_norm_img_0p05_inbounds_v1",
      "eval_profile_required": "paper_model_native_v1",
      "aggregation_policy": "mean_over_ok_samples",
      "missing_value_policy": "null_excluded_not_zero",
      "degraded_policy": "excluded_from_primary_aggregate",
      "paper_ready_requirement": "paired_task_strict_binding_body18_mapping_and_pose_policy_validated"
    }
  ]
}
```

## 5. Aggregation Policy

Primary paper candidate aggregate:

- Include only samples with metric status `ok`.
- Exclude `failed`.
- Exclude `skipped`.
- Exclude `degraded` from the primary aggregate by default.
- Record `degraded_n` separately.

Optional diagnostic aggregate:

- `ok_plus_degraded_mean`
- This may appear in a report appendix, not in the primary candidate table.

Failed/skipped rules:

- Numeric value remains `null`.
- The value is never counted as `0`.
- Failed/skipped samples must be visible in failure audits and readiness reports.

If `valid_n < sample_count`:

- `paper_ready=false`
- readiness gate must include a warning
- table row can still be emitted as a candidate only if the writer policy explicitly allows incomplete candidates

Default writer policy should be strict for paper candidates:

- incomplete official split aggregate is allowed for smoke/debug only
- incomplete aggregate is not final paper-ready

## 6. Metric-Specific Aggregation

### FaceSim

Primary column:

- `facesim_mean = mean(sample.scores.facesim_mean over ok samples)`

Auxiliary report, not main candidate column:

- `facesim_min_mean`
- `facesim_std_mean`
- reference source distribution
- face detection coverage summary

Requirements:

- Reference must be the actual generation reference.
- Benchmark image fallback is not allowed unless it was explicitly the generation reference and is recorded as such.

### IQA/AES

Primary columns:

- `iqa_mean = mean(sample.scores.iqa_mean over ok samples)`
- `aes_mean = mean(sample.scores.aes_mean over ok samples)`

Rules:

- If IQA backend is missing but AES is available, `iqa_mean=null`, `paper_ready=false`.
- If AES backend is missing but IQA is available, `aes_mean=null`, `paper_ready=false`.
- If both are missing, both columns are `null`, and the candidate row is not paper-ready.
- The local VBench model hashes, score scale, sampling, and resize/crop policy must be frozen.

### AKD/PCK

Primary columns:

- `akd_body18_norm_img = mean(sample.scores.akd_body18_norm_img over ok samples)`
- `pck_body18_norm_img_0p05 = mean(sample.scores.pck_body18_norm_img_0p05 over ok samples)`

Requirements:

- Strict binding validated for every paired sample.
- `body18_mapping_status` validated by final validation document.
- `matched_keypoint_rate` meets the metric policy threshold for status `ok`.
- Generated and target pose sources are comparable under the same eval profile.
- No numeric fallback, basename fallback, or row-index fallback.

If AKD/PCK is not applicable to the task:

- AKD/PCK columns are `null`.
- `notes` records `pose_metric_not_applicable`.
- The candidate row can still be valid for non-pose metrics, but not for paired pose comparison claims.

## 7. Readiness Gate

Candidate generation is allowed only if all preflight gates pass:

- `eval_profile.name = paper_model_native_v1` or another implemented paper profile.
- `split_name` starts with `official_`.
- `split_sha256` is present.
- `benchmark_index_sha256` is present.
- Strict binding is validated for all paired metrics.
- Metric provenance fingerprints are present.
- No `legacy_unknown` profile is present.
- No debug binding is present.
- Metric status distribution is recorded.
- Paper readiness report is generated.

`paper_ready` remains `false` if any of these are missing:

- paper table candidate writer not implemented
- human worst-case review not completed
- limitations report missing
- official split too small for final paper claims
- profile is `paper_model_native_v1` but the target comparison requires a future profile such as `paper_hoivg_720p25fps5s_v1`

Smoke policy:

- `official_smoke_10.v1` can validate the writer and schema.
- It must not set `paper_ready=true`.

Development paper-profile policy:

- `official_dev_50.v1` can validate candidate stability and cost.
- It still does not automatically make the table final paper-ready.

## 8. Comparability Policy

`comparable_by_default=true` only if all conditions match:

- same `eval_profile`
- same `split_name`
- same `split_version`
- same `split_sha256`
- same `benchmark_index_sha256`
- same metric variants
- same metric provenance fingerprints
- same materialization policy
- same target binding policy
- same aggregation policy

Otherwise:

- `comparable_by_default=false`
- comparison requires an explicit override
- report must list the mismatched fields

Examples that are not comparable by default:

- `dev_native_v1` vs `paper_model_native_v1`
- `official_smoke_10.v1` vs `official_dev_50.v1`
- sampled AKD/PCK vs dense AKD/PCK
- VBench local backend with different model hashes
- FaceSim with different reference policy

## 9. Future Writer Output Artifacts

The future implementation should output:

```text
eval_outputs/<run_id>/
  paper_table_candidate.csv
  paper_table_candidate.schema.json
  paper_readiness_gate.md
  paper_candidate_provenance_summary.json
  paper_candidate_failure_audit.md
  paper_candidate_worst_case_review.md
```

This round intentionally does not implement or create these files.

## 10. `official_smoke_10.v1` Limitations

`official_smoke_10.v1` is only a smoke split.

It is useful for:

- validating profile-aware provenance
- validating official split wiring
- validating candidate schema/writer logic in a future step
- checking strict binding on a small fixed set
- checking that `paper_table.csv` remains unchanged

It is not enough for:

- final paper claims
- cross-method leaderboard claims
- final paper-table means
- failure-rate estimation
- robust worst-case analysis

Final paper-table work should use one of:

- `official_dev_50.v1` for development candidate validation
- `official_paper_100.v1` after human approval
- `official_paper_full.v1` if full benchmark cost is acceptable

## 11. Next Implementation Plan

Step 1:

- Implement `paper_table_candidate` writer for `official_smoke_10.v1` only.
- Keep writer opt-in.
- Do not replace `paper_table.csv`.

Step 2:

- Run `paper_model_native_v1 + official_smoke_10.v1` candidate writer smoke.
- Confirm schema JSON, candidate CSV, readiness gate, provenance summary, and failure audit are emitted.

Step 3:

- Generate `paper_readiness_gate.md`.
- Confirm `paper_ready=false` for smoke split.

Step 4:

- Human-review worst cases and limitations.
- Confirm diagnostic-only metrics remain excluded.

Step 5:

- Decide whether to run `official_dev_50.v1` candidate.
- Do not move to paper claims until larger official split and human review pass.
