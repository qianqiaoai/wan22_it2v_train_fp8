# Paper Standardized Profile Design

Status: profile provenance step implemented  
Implementation status: profile registry, profile-aware provenance, and smoke/dev official splits implemented; paper candidate table not implemented  
Paper table status: unchanged  
Current date: 2026-06-13

This document defines profile-aware evaluation for offline digital-human video evaluation. It does not enable paper metrics, does not add paper-table columns, and does not change any metric formula or preset.

## 1. Profile Naming

### dev_native_v1

`dev_native_v1` is the current research-iteration profile.

- Intended use: fast development, checkpoint debugging, failure analysis, and iterative model comparison.
- Current media protocol: `480x832`, `20fps`, `81 frames`, approximately `4.05s`.
- Current presets:
  - `quick = sanity + facesim_sampled`
  - `full = sanity + facesim_sampled + sync_global + iqa_aes_sampled`
  - `full_pose = sanity + facesim_sampled + sync_global + iqa_aes_sampled + akd_pck_body18`
- Status: stable for development, not paper-ready by default.

`dev_native_v1` results must not be treated as paper-ready unless a run explicitly records the paper gates listed below. `dev_native_v1` and paper profiles are not comparable by default.

### paper_model_native_v1

`paper_model_native_v1` is the first implemented paper-candidate profile target. It uses the same model-native media policy as `dev_native_v1`, but profile-aware provenance keeps the aggregates separate.

- Intended use: fixed split, fixed materialization protocol, frozen metric provenance, and paper-table candidate generation.
- Required frozen dimensions:
  - resolution
  - fps
  - duration / frame count
  - crop policy
  - audio clip policy
  - target clip policy
  - official split
  - metric frame sampling / dense sampling policy
- Status: implemented as a profile/provenance option, not paper-ready.

`paper_model_native_v1` must not overwrite or replace `dev_native_v1`. It is an explicit profile in `resolved_config.json`, `metric_plan.json`, and metric provenance.

### paper_hoivg_720p25fps5s_v1

`paper_hoivg_720p25fps5s_v1` is a future HOIVG-style profile.

- Intended use: future 720p portrait / 25fps / 5s reporting.
- Status: planned only, not implemented.
- Current behavior: requesting this profile must fail with a clear `eval_profile_not_implemented` error.

## 2. Paper Media Policy

Three candidate media policies were considered.

### Option A: Model-Native Paper Profile

Use the current model-native protocol:

- `480x832`
- `20fps`
- `81 frames`
- approximately `4.05s`
- target/ref/audio materialized to the same protocol
- crop policy: `cover_center_crop`

Advantages:

- Avoids post-generation resizing, interpolation, padding, or frame-rate conversion artifacts.
- Matches the actual generation process.
- Keeps FaceSim, IQA/AES, and AKD/PCK aligned with the videos that users inspect.
- Lower implementation risk because the current materialization and admissions already use this protocol.

Limitations:

- Not directly comparable to OmniShow-style `5s / 720p / 25fps` reporting.
- Paper must state the model-native media protocol clearly.

### Option B: Post-Standardized 720p / 25fps / 5s

Post-process generated and target videos to a 720p portrait, 25fps, 5s protocol.

Advantages:

- Closer to HOIVG-Bench / OmniShow-style reporting.
- Easier to describe when comparing against a benchmark that expects 5s/720p clips.

Limitations:

- Adds resize/interpolation/padding artifacts.
- Can change IQA/AES and pose keypoint behavior.
- May make scores describe post-processing quality rather than generation quality.
- Requires a separate admission pass for all metrics.

### Option C: Dual Report

Report both:

- `paper_model_native_v1`
- `paper_hoivg_720p25fps5s_v1`

Advantages:

- Most transparent.
- Separates generation-native quality from cross-benchmark standardized quality.

Limitations:

- Doubles protocol surface area.
- Makes the paper table more complex.
- Requires two provenance/fingerprint families and clear non-mixing rules.

### Recommendation

Use Option A as the first implemented paper-candidate profile:

`paper_model_native_v1`

Reason:

- It minimizes artificial post-processing effects.
- It is consistent with current model output and validated admissions.
- It can become a reliable internal paper-table candidate faster.

If HOIVG-Bench-style comparability becomes a hard requirement, use the separate planned profile:

`paper_hoivg_720p25fps5s_v1`

Do not mix these two profiles in one aggregate or table column.

## 3. Official Split Design

Versioned official splits should live under:

```text
benchmark/splits/
  official_smoke_10.v1.json
  official_dev_50.v1.json
  official_paper_100.v1.json
  official_paper_full.v1.json
```

Each split file must contain:

```json
{
  "split_name": "official_dev_50",
  "split_version": "v1",
  "benchmark_keys": ["000001_c70714ce"],
  "split_sha256": "...",
  "benchmark_index_sha256": "...",
  "selection_policy": "stratified_or_first_n_documented",
  "filters": {
    "task": "audio_ref_text_to_digital_human_video",
    "requires_target_video": true
  },
  "seed": 20260613,
  "created_at": "2026-06-13T00:00:00Z",
  "notes": "Human-readable split rationale."
}
```

Rules:

- Paper table candidates must use a versioned split.
- Ad-hoc 50-sample runs are not paper splits.
- `split_sha256` must be recorded in `resolved_config.json`, `metric_plan.json`, and `metrics.json`.
- `benchmark_index_sha256` must be recorded with the split.
- Split membership must be by `benchmark_key`, not row number or basename.
- Split generation must fail on duplicated `benchmark_key`.

Implemented split roles:

- `official_smoke_10.v1`: CI / quick paper-profile smoke.
- `official_dev_50.v1`: development admission for paper-profile mechanics.

Planned split roles:

- `official_paper_100.v1`: minimum candidate paper table.
- `official_paper_full.v1`: full benchmark table if cost allows.

See `docs/metric_status/official_split_policy.md` for versioning, hash, subset, and comparability rules.

## 4. Benchmark Key And Binding Policy

`paper_model_native_v1` and later paper profiles must require:

- `benchmark_key` exists.
- `benchmark_key` collision count is `0`.
- `benchmark_index.json` exists and matches `benchmark_index_sha256`.
- `benchmark_materialized/manifest.jsonl` exists.
- inference manifest carries:
  - `benchmark_key`
  - `benchmark_row_hash`
  - `benchmark_row_index`
  - `materialized_ref`
  - `materialized_audio`
  - `materialized_target_clip`
  - `generated_video`
- strict binding validates the generated row against materialized benchmark metadata.

Forbidden:

- numeric row fallback
- basename fallback
- row-index fallback
- debug-only binding
- silent fallback

Failure policy:

- For development paper-profile smoke, binding failure should fail the sample-level metric with null scores and clear error.
- For paper-table candidate generation, any binding failure should fail the paper candidate run unless an explicit policy says to exclude failed samples and report `failed_n`.
- Debug-only binding must never enter paper metrics.

## 5. Metric Inclusion Policy

### Candidate For `paper_table_candidate.csv`

The following metrics can be candidates after profile-specific validation:

- FaceSim
  - Candidate identity metric.
  - Requires actual generation reference, not benchmark image fallback.
  - Current backend can be reused with profile-aware provenance.
- IQA/AES
  - Candidate visual quality / aesthetic metric.
  - Must be run under `paper_model_native_v1` or a later implemented paper profile.
  - Requires frozen VBench local model hashes and sampling policy.
- AKD/PCK
  - Candidate pose accuracy metric only for paired/key-preserving target-video or pose-comparable tasks.
  - Requires strict binding and materialized target clip.
  - Must use the same paper materialization profile for generated and target videos.

### Diagnostic Only

- `sync_global`
  - Current status: runtime works, but `score_interpretation=uncalibrated`.
  - It must not enter paper table until offset sign, discriminativeness, and benchmark GT audio/target validity are accepted.

### Pending

- TA / Text Alignment
- VQ / MQ / VideoReward
- temporal flicker / temporal diagnostic
- realtime metrics

These remain pending and are outside the current paper profile.

## 6. Metric Variant And Profile Naming

The metric variant may remain stable, but the profile must enter provenance and fingerprint.

### FaceSim

- Metric variant: `facesim_sampled_insightface_antelopev2_cpu_v1`
- Profile field: `paper_model_native_v1`
- Provenance must include:
  - reference policy
  - sample count
  - frame sampling policy
  - profile name
  - profile media policy

### IQA/AES

- Metric variant: `iqa_aes_vbench_local_sampled_cpu_v1`
- Profile field: `paper_model_native_v1`
- `paper_ready=true` only after:
  - local model hashes frozen
  - profile-specific parity/degradation checks accepted
  - versioned split smoke accepted
  - paper-table candidate generation accepted

### AKD/PCK

- Metric variant: `akd_pck_body18_norm_img_0p05_inbounds_v1`
- Profile field: `paper_model_native_v1`
- Applicability: paired/key-preserving/materialized target clip only.

Rules:

- Profile must enter `provenance_fingerprint`.
- `dev_native_v1` and `paper_model_native_v1` results must not aggregate together.
- `paper_table_candidate.csv` only accepts metrics with an implemented paper profile such as `paper_model_native_v1`.
- If a metric lacks a profile field, paper candidate generation must fail.

## 7. AKD/PCK Paper Policy

Frozen policy:

- `keypoint_set = body18`
- `coordinate_space = normalized_xy`
- `normalization_mode = normalized_image`
- `pck_threshold = 0.05`
- `pck_threshold_units = normalized_image`
- `visibility_score_min = 0.3`
- `missing_keypoint_policy = exclude_unmatched`
- `out_of_frame_policy = exclude_from_primary_metric`

Frame alignment options:

- `sampled_index_match`
- future: `paper_dense_alignment`

### Sampled vs Dense

Current AKD/PCK uses sampled 8 frames. This is acceptable for development and may be acceptable for a paper candidate if the table explicitly names the sampling policy. However, sampled 8-frame AKD/PCK is not a dense temporal pose metric.

Recommendation:

- First paper-profile smoke: keep sampled 8 frames.
- Candidate table name or metadata must state sampled policy.
- For final paper table, prefer a second admission with denser sampling:
  - model-native dense: 20fps x 4.05s = 81 frames
  - 720p profile dense, if later added: 25fps x 5s = 125 frames

Estimated cost:

- Current sampled DWPose uses 8 frames.
- Dense model-native extraction is roughly `81 / 8 = 10.125x` DWPose extraction cost.
- Dense 25fps/5s extraction is roughly `125 / 8 = 15.625x` DWPose extraction cost.

Policy:

- If paper table uses sampled AKD/PCK, the table caption and profile report must say so.
- Dense AKD/PCK should use a distinct profile/sampling policy and separate provenance fingerprint.

## 8. IQA/AES Paper Policy

Backend:

- VBench local backend.
- Offline mode only.
- No implicit downloads.
- Model hashes frozen.
- Score scale and direction frozen.

Required frozen fields:

- model root and model hashes
- score scale
- score direction
- frame sampling policy
- resize/crop policy
- input color space
- normalization
- eval profile
- split hash

### Sampled vs More Frames

Current IQA/AES uses sampled 8 frames.

Recommendation:

- First paper-profile smoke: keep sampled 8 frames for cost and continuity.
- Paper-table candidate: use 16 uniformly sampled frames if cost is acceptable, because IQA/AES is frame-based and may miss transient artifacts with 8 frames.
- Dense IQA/AES is optional and should not be introduced until cost and sensitivity are measured.

If a future 720p/25fps/5s profile is added, frames must be re-extracted from the standardized video and must use a distinct profile/fingerprint.

Coverage/status policy:

- Report valid frame coverage.
- High black/frozen frame rate should produce `degraded` with numeric scores if backend scores are still valid.
- Frame extraction failure or backend/model failure should produce `failed` with null scores.
- Failed/skipped results must not be aggregated as zero.

## 9. Report And Paper Table Policy

Do not replace the current `paper_table.csv` yet.

Future paper-profile runs should write:

```text
paper_table_candidate.csv
```

Recommended columns:

```text
method
split_name
split_version
split_sha256
eval_profile
media_profile
facesim_mean
iqa_mean
aes_mean
akd_body18_norm_img
pck_body18_norm_img_0p05
valid_n
failed_n
degraded_n
metric_provenance_id
```

Column direction metadata must be written separately, for example:

```json
{
  "facesim_mean": "higher_is_better",
  "iqa_mean": "higher_is_better",
  "aes_mean": "higher_is_better",
  "akd_body18_norm_img": "lower_is_better",
  "pck_body18_norm_img_0p05": "higher_is_better"
}
```

Rules:

- `sync_global` is excluded from paper-table candidate.
- Failed/skipped results are never treated as zero.
- Degraded results may be included in primary mean only if the policy says `include_degraded=true`; this must be recorded.
- If degraded results are included, `degraded_n` must be reported.
- Every paper candidate row must record metric provenance ID or fingerprint summary.

## 10. Paper Readiness Gate

`paper_ready=true` requires:

- versioned split
- implemented paper profile, initially `paper_model_native_v1`
- strict binding success rate = 100%
- no debug-only binding
- no numeric/basename/row fallback
- benchmark index hash frozen
- split hash frozen
- benchmark materialization manifest hash frozen
- metric provenance frozen
- profile-aware provenance fingerprint
- paper-table candidate generated
- failed/skipped/degraded policy recorded
- human review of worst cases
- limitations report written
- `comparable_by_default` explicitly set

If any gate is missing:

```text
paper_ready=false
```

`comparable_by_default` should remain `false` unless method, split, media profile, metric variants, and provenance fingerprints are compatible.

## 11. Output Artifacts

Future paper-profile runs should output:

```text
eval_outputs/<run_id>/
  resolved_config.json
  metric_plan.json
  selected_samples.jsonl
  benchmark_index.json
  benchmark_materialized/manifest.jsonl
  metrics.json
  paper_table_candidate.csv
  paper_table_candidate_metadata.json
  paper_profile_report.md
  failure_audit.md
  provenance_summary.json
  worst_case_review.html
```

Required config fields:

```json
{
  "eval_profile": "paper_model_native_v1",
  "media_profile": "paper_model_native_v1",
  "split_name": "official_dev_50",
  "split_version": "v1",
  "split_sha256": "...",
  "benchmark_index_sha256": "...",
  "benchmark_materialized_manifest_sha256": "...",
  "paper_candidate": true,
  "paper_ready": false
}
```

## 12. Next Implementation Plan

Do not implement all steps at once. Use the following minimum sequence.

### Step 1: Profile Field And Profile-Aware Provenance

- Add `eval_profile` and `media_profile` to resolved config.
- Pass profile to metric contexts.
- Include profile in each metric provenance fingerprint.
- Add aggregate guard to prevent mixing `dev_native_v1` and `paper_model_native_v1`.

### Step 2: Versioned Official Split Generator

- Generate `benchmark/splits/official_smoke_10.v1.json`.
- Generate `benchmark/splits/official_dev_50.v1.json`.
- Record split hash and benchmark index hash.
- Fail on missing or duplicated `benchmark_key`.

### Step 3: Paper-Standardized Materialization Dry Run

- Materialize or plan target/ref/audio/video paths under the selected media profile.
- Record crop, fps, duration, and resolution policy.
- Validate strict binding without running metrics.

### Step 4: Paper-Standardized Smoke On 3 Samples

- Run:
  - sanity
  - FaceSim
  - IQA/AES
  - AKD/PCK only when paired target clip exists
- Keep `sync_global` diagnostic only.
- Confirm all metrics record profile-aware provenance.

### Step 5: Generate `paper_table_candidate.csv`

- Write candidate table without replacing current `paper_table.csv`.
- Include direction metadata.
- Exclude `sync_global`.
- Exclude failed/skipped from means.
- Record degraded inclusion policy.

## Readiness Decision

Can generate paper table now: no.

Exact blockers:

- Official smoke/dev split support for `paper_model_native_v1` is implemented.
- Official paper split support is not implemented.
- Versioned official splits are not implemented.
- Profile-aware metric provenance/fingerprint is not implemented.
- `paper_table_candidate.csv` writer is not implemented.
- IQA/AES is still `paper_ready=false` until split/provenance gates pass.
- `sync_global` remains diagnostic-only and uncalibrated.

Recommended next step:

Implement Step 1 and Step 2 only, then run a 3-sample paper-profile dry run before adding any candidate table writer.
