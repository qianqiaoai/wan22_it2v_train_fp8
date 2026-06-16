# Offline Digital Human Video Eval Agent

## Goal

Build a natural-language-triggered offline evaluation agent for digital human video generation.

The intended interaction is:

```text
User: 帮我评测一下最新权重
Agent:
1. Resolve the latest checkpoint.
2. Run benchmark inference.
3. Evaluate generated videos with offline multimodal metrics.
4. Write metrics.json, paper_table.csv, and html_report.html.
5. Report the selected checkpoint, result paths, coverage, and failures.
```

The user can also provide an explicit checkpoint:

```text
帮我评测一下 /path/to/checkpoint_model_007500/model.pt
```

Realtime evaluation is intentionally out of scope for the MVP. TTFF, latency, deadline miss, streaming trace, and chunk-boundary diagnostics should only be recorded in `backlog/realtime_eval.md`.

## Current Repository Context

- Benchmark manifest: `benchmark/benchmark.jsonl`
- Benchmark profile: `configs/benchmark_profiles/audio_ref_text2video.current.yaml`
- Existing inference script: `scripts/infer_audio_i2v.py`
- Existing inference config: `configs/infer_audio_i2v.yaml`
- Existing inference output pattern: `manifest.jsonl`, `videos/`, `refs_480p/`, `audio_segments/`
- Recommended evaluation output root: `eval_outputs/<run_id>/`

Benchmark sample fields are interpreted through a profile, not hardcoded in evaluator logic. The current profile maps:

```json
{
  "imgpath": "reference image",
  "videopath": "target video",
  "posepath": "target pose",
  "wav_path": "input audio",
  "prompt": "text prompt",
  "text": "spoken text",
  "language": "zh/en",
  "gender": "male/female",
  "type": "vocal/sing",
  "category": "dance/zhibo",
  "body": "half/full"
}
```

For new benchmarks, add a YAML profile under `configs/benchmark_profiles/` and keep evaluator code unchanged.

The benchmark media should determine the evaluation normalization policy. Current observations:

- Target/reference videos are vertical.
- Main target video shapes include `1088x1920`, `1080x1920`, and `720x1280`.
- Target videos are mostly `25fps`, around 10 seconds.
- Current model inference outputs are `480x832`, `20fps`, `81` frames, around 4.05 seconds.
- Input audio sample rates are mixed, so evaluation modules should resample internally.

## Recommended Directory Layout

```text
eval_agent/
  agent.py                 # Natural language and structured-request orchestration
  config.py                # Resolve, freeze, save, and fingerprint run config
  task_profiles.py         # Task-specific conditioning contracts and metric gating
  generation_profiles.py   # Resolve model/eval output size, fps, frame count, and duration
  checkpoints.py           # Explicit/latest checkpoint resolution
  splits.py                # Official eval split registry and split hash validation
  sample_selection.py      # Flexible sample count/filter/strategy resolution
  metric_registry.py       # Metric definitions, cost levels, inputs, outputs, and variants
  presets.py               # Natural-language presets for quick/full/lip-sync/paper eval
  cost_estimator.py        # Estimate metric cost from selected sample count and variants
  metric_cache.py          # Content-addressed metric cache, invalidation, and hit/miss audit
  face_tracking.py         # Multi-face detection, face-track selection, and switch diagnostics
  manifest.py              # Parse, normalize, fingerprint, and reuse inference manifests
  manifest_eval.py         # Re-evaluate existing outputs from an inference manifest
  video_dir_eval.py        # Evaluate bare video directories via a synthetic manifest
  validation.py            # Benchmark linkage and conditioning validation
  benchmark_normalization.py # Materialize config-aligned ref/video/audio benchmark inputs
  benchmark_infer.py       # Benchmark JSONL to inference jobs
  io.py                    # Benchmark/manifest loading and sample alignment
  media.py                 # ffprobe, frame extraction, audio extraction, resampling
  schema.py                # SampleRecord, MetricResult, Coverage, ErrorInfo
  artifacts.py             # Per-sample visualizations
  comparison.py            # Baseline/previous-run comparison tables and summaries
  report.py                # metrics.json, paper_table.csv, html_report.html

  metrics/
    sanity.py
    facesim.py
    sync.py
    quality.py
    pose.py

scripts/
  eval_digital_human_agent.py

backlog/
  realtime_eval.md
```

Keep the evaluation package separate from `wan2/`, which contains model code. Use `auto_eval/` for Skill-facing eval code; keep `scripts/` for compatibility wrappers and model-specific entrypoints.

Skill packaging note: the Skill-facing implementation now lives under `auto_eval/`, including `auto_eval/eval_digital_human_agent.py` and `auto_eval/eval_agent/`. The top-level `eval_agent/` package is retained only as a compatibility shim for historical scripts.

## Agent Execution Flow

```text
natural language request
        |
        v
resolve run_mode()
        |
        v
resolve_checkpoint()
        |
        v
run_benchmark_inference() or load_existing_outputs()
        |
        v
run_offline_eval()
        |
        v
write_reports()
        |
        v
agent_summary()
```

The natural-language layer should always compile down to a reproducible structured config.

## Natural Language Safety Boundary

Natural-language requests are allowed to select an evaluation intent, checkpoint, split, preset, generation profile, and metric set. They must not directly generate or execute shell commands.

Safety rules:

- Natural language must compile to a whitelisted structured config first.
- The executor may only run known eval-agent actions, not arbitrary shell text from the user request.
- Support `--dry_run` to resolve config, sample selection, metric plan, expected outputs, and estimated cost without running inference or metrics.
- Support `--yes` for non-interactive execution after config resolution. Without `--yes`, destructive or expensive actions should require explicit confirmation.
- Support `allowed_root` / `allowed_roots` to constrain checkpoint, benchmark, manifest, generated video, output, and artifact paths.
- Path fields must resolve under an allowed root unless explicitly allowed by a trusted config.
- Only whitelisted config fields may be set from natural language. Unknown fields must be rejected or placed in `request.unparsed_text`, never silently applied.
- Natural language must not set arbitrary Python code, shell commands, environment variables, import paths, or post-run hooks.
- Any inferred action must appear in `resolved_config.json` before execution.

Suggested safety config:

```json
{
  "safety": {
    "dry_run": false,
    "yes": false,
    "allowed_roots": [
      "/mnt/data/nlp/user/qiaoqian/newproject/wan2.2_5B_fp8",
      "/mnt/data/nlp/user/qiaoqian/newproject"
    ],
    "nl_config_whitelist": [
      "run_mode",
      "checkpoint",
      "benchmark_jsonl",
      "generated_manifest",
      "generated_video_dir",
      "source_eval_dir",
      "sample_selection",
      "task",
      "conditioning",
      "generation",
      "metrics",
      "reports",
      "comparison",
      "validation"
    ],
    "forbid_shell_from_nl": true
  }
}
```

Example structured config:

```json
{
  "run_mode": "infer_eval",
  "checkpoint": ".../model.pt",
  "checkpoint_resolve_mode": "explicit|latest",
  "benchmark_jsonl": "benchmark/benchmark.jsonl",
  "generated_manifest": null,
  "generated_video_dir": null,
  "source_eval_dir": null,
  "sample_selection": {
    "count": null,
    "mode": "all",
    "seed": 20260609,
    "filters": {},
    "group_by": null,
    "ids": []
  },
  "conditioning": {
    "text_mode": "fixed_embedding",
    "text_emb_path": "raw_dataset/text_emb/person_speaking_zh_umt5_xxl_bf16.pt",
    "negative_text_emb_path": "raw_dataset/text_emb/wan2_2_negative_prompt_umt5_xxl_bf16.pt",
    "audio_mode": "benchmark_wav",
    "reference_mode": "materialized_ref",
    "prompt_for_generation": false
  },
  "validation": {
    "benchmark_linkage": "strict",
    "require_sample_id_in_benchmark": true,
    "require_conditioning_match_for_benchmark_modes": true,
    "allow_third_party_outputs": false
  },
  "metric_request": {
    "preset": "quick|full|lip_sync|paper|custom",
    "explicit_metrics": [],
    "max_cost_level": "auto",
    "allow_expensive_metrics": false
  },
  "metrics": {
    "enabled": ["sanity", "facesim_sampled", "sync_global"],
    "excluded": [],
    "metric_plan_path": "eval_outputs/<run_id>/metric_plan.json"
  },
  "output_dir": "eval_outputs/<run_id>"
}
```

## Implementation Milestones

Implement the agent in staged milestones. Do not try to ship every metric and reporting feature in the first implementation.

| Milestone | Goal | Included | Explicitly Deferred |
| --- | --- | --- | --- |
| MVP-0 | Reproducible run skeleton | Natural-language/CLI to `resolved_config.json`, `run_id`, `selected_samples.jsonl`, `metric_plan.json`, `benchmark_index.json`, dry run, path safety checks. | Model inference, learned metrics, cache, comparison. |
| MVP-1 | First executable offline eval | `benchmark_materialized`, inference adapter for selected samples, sanity check, `facesim_sampled`, `sync_global`, `metrics.json`, basic artifacts, `html_report.html` skeleton. | cache beyond single-run reuse, comparison, paper preset, dense metrics, IQA/AES, pose, paired GT diagnostics. |
| MVP-2 | Validation and diagnostics | benchmark target-audio sync validation, face track policy/artifacts, benchmark linkage modes, paired GT diagnostics marked `diagnostics_only`, acceptance criteria. | cross-run cache, formal paper tables, dense quality/sync metrics as default. |
| MVP-3 | Reporting and comparison | baseline/previous-run comparison, `current_run_table.csv`, `comparison_table.csv`, `paper_table.csv`, paper preset, diagnostic summaries and failure taxonomy. | cross-run cache as a default dependency. |
| MVP-4 | Performance and full metric depth | persistent cross-run cache, dense FaceSim/Sync/IQA/AES, DWPose extraction cache, full/paper runs at larger official splits. | realtime metrics; TTFF/latency/deadline miss remain backlog. |

The first implementation target is MVP-1:

- write and reload `resolved_config.json`;
- materialize benchmark inputs under `benchmark_materialized/`;
- run sanity;
- run sampled FaceSim;
- run global Sync-C / Sync-D;
- write `metrics.json`;
- write an `html_report.html` skeleton with run metadata, aggregate table, status summary, and links to per-sample artifacts.

MVP-1 defaults:

- `cache.enabled=false`;
- no cross-run cache;
- no comparison stage;
- no paper preset;
- no dense metric variants;
- no realtime metrics.

## Resolved Config Artifact

`resolved_config.json` is a formal output artifact and must be written for every run mode.

It is the frozen, canonical execution plan after:

- natural-language request parsing;
- CLI/config/default merging;
- run mode resolution;
- checkpoint resolution, if applicable;
- path normalization to absolute paths;
- benchmark path resolution;
- manifest resolution, if applicable;
- sample selection resolution;
- conditioning mode resolution;
- validation level resolution;
- metric list and metric parameter resolution;
- output directory creation.

It should be written before inference or metric computation starts:

```text
eval_outputs/<run_id>/resolved_config.json
```

### Run ID and Latest Symlinks

Each run must use a stable, human-readable `run_id`:

```text
<timestamp>_<mode>_<checkpoint_step>_<split>_<preset>
```

Example:

```text
20260610T143012Z_infer_eval_007500_official_smoke_10_quick
```

Rules:

- `timestamp` should be UTC in `YYYYMMDDTHHMMSSZ`.
- `mode` is the resolved run mode, such as `infer_eval`, `manifest_eval`, `video_dir_eval`, `report_only`, or `compare_only`.
- `checkpoint_step` is a zero-padded step if known, such as `007500`; use `nostep` when unavailable.
- `split` is the resolved split name; use `adhoc` for ad-hoc sample selection.
- `preset` is the resolved metric preset, such as `quick`, `lip_sync`, `full`, `paper`, or `custom`.
- If a generated `run_id` already exists, append a short collision suffix such as `_r2`.

Maintain symlinks under `eval_outputs/`:

```text
eval_outputs/latest -> <latest attempted run_id>
eval_outputs/latest_success -> <latest run_id with overall_status=pass|warn>
```

`latest` may point to failed runs. `latest_success` must only update after reports are written and `overall_status` is not `fail`. On platforms where symlinks are unavailable, write `latest.json` and `latest_success.json` pointer files with the same information.

The executor must use the persisted config, not a separate in-memory-only config. Required sequence:

1. Resolve the final execution config in memory.
2. Write it to `resolved_config.json`.
3. Read `resolved_config.json` back from disk.
4. Compute and record its SHA256.
5. Execute inference/eval/reporting from the reloaded config object.

Once written and reloaded, it should not be mutated. If a later step discovers runtime facts, write them to `metrics.json`, `validation_report.json`, or metric-specific outputs instead of rewriting the resolved config.

No runtime path, metric, sample, conditioning, checkpoint, or validation override may exist only in memory. If a setting affects actual execution, it must appear in `resolved_config.json` before execution starts.

Minimum fields:

```json
{
  "schema_version": 1,
  "created_at": "2026-06-10T00:00:00Z",
  "request": {
    "raw_text": "用10个样本评估最新权重",
    "parsed_intent": {}
  },
  "run": {
    "run_id": "<run_id>",
    "run_mode": "infer_eval",
    "output_dir": "/abs/path/eval_outputs/<run_id>"
  },
  "inputs": {
    "benchmark_jsonl": "/abs/path/benchmark/benchmark.jsonl",
    "generated_manifest": null,
    "generated_video_dir": null,
    "source_eval_dir": null
  },
  "safety": {
    "dry_run": false,
    "yes": false,
    "allowed_roots": ["/mnt/data/nlp/user/qiaoqian/newproject/wan2.2_5B_fp8"],
    "forbid_shell_from_nl": true,
    "nl_config_whitelist": ["run_mode", "checkpoint", "benchmark_jsonl", "sample_selection", "conditioning", "generation", "metrics"]
  },
  "checkpoint": {
    "resolve_mode": "latest",
    "path": "/abs/path/model.pt",
    "step": 7500,
    "config_path": "/abs/path/config.yaml"
  },
  "task": {
    "name": "audio_ref_text2video",
    "version": "v1",
    "required_conditions": ["audio", "reference_image", "text"],
    "optional_conditions": [],
    "output_type": "digital_human_video"
  },
  "sample_selection": {
    "split_name": "official_smoke_10",
    "split_version": "v1",
    "split_file": "/abs/path/benchmark/splits/official_smoke_10.v1.json",
    "split_sha256": "sha256-or-null",
    "requested_count": 10,
    "available_count": 300,
    "selected_count": 10,
    "mode": "official_split",
    "seed": 20260609,
    "filters": {},
    "group_by": null,
    "selected_sample_ids": ["000001", "000002"]
  },
  "conditioning": {
    "text_mode": "fixed_embedding",
    "text_emb_path": "/abs/path/person_speaking_zh_umt5_xxl_bf16.pt",
    "audio_mode": "benchmark_wav",
    "reference_mode": "materialized_ref",
    "pose_mode": "none",
    "prompt_for_generation": false,
    "prompt_for_reporting": true
  },
  "generation": {
    "profile_name": "current_480x832_20fps_81f",
    "source": "model_config",
    "source_path": "/abs/path/configs/infer_audio_i2v.yaml",
    "width": 480,
    "height": 832,
    "fps": 20,
    "frame_num": 81,
    "duration_sec": 4.05,
    "orientation": "portrait",
    "resize_policy": "cover_center_crop",
    "must_match_generated_outputs": true,
    "generated_output_mismatch_policy": "fail"
  },
  "benchmark_normalization": {
    "enabled": true,
    "materialized_dir": "/abs/path/eval_outputs/<run_id>/benchmark_materialized",
    "ref_source_policy": "target_first_frame",
    "resize_policy": "cover_center_crop",
    "target_video_width": 480,
    "target_video_height": 832,
    "target_video_fps": 20,
    "target_video_frames": 81,
    "audio_sample_rate": 16000,
    "audio_duration_sec": 4.05,
    "first_frame_validation": {
      "enabled": true,
      "mae_max": 2.0,
      "psnr_min": 35.0
    }
  },
  "validation": {
    "benchmark_linkage": "strict",
    "require_sample_id_in_benchmark": true,
    "require_conditioning_match_for_benchmark_modes": true,
    "benchmark_av_sync": {
      "enabled": true,
      "target_video_source": "materialized_target_clip",
      "audio_source": "materialized_audio",
      "metric_variant": "sync_global",
      "min_window_coverage": 0.75,
      "min_valid_windows": 4,
      "on_degraded": "degrade_generated_sync",
      "on_failed": "exclude_from_sync_aggregate"
    }
  },
  "metrics": {
    "requested_preset": "quick",
    "resolved_preset": "quick",
    "max_cost_level": "medium",
    "allow_expensive_metrics": false,
    "cost_estimate": {
      "selected_count": 10,
      "total_estimated_cost_units": 45,
      "hardware_profile": "unknown",
      "notes": "Cost units are relative and used for planning, not wall-clock guarantees."
    },
    "metric_plan_path": "/abs/path/eval_outputs/<run_id>/metric_plan.json",
    "enabled": ["sanity", "facesim_sampled", "sync_global"],
    "excluded": [
      {
        "metric": "iqa_qalign_sampled",
        "reason": "cost_level=high exceeds max_cost_level=medium for preset=quick"
      }
    ],
    "params": {
      "sync_window_sec": 2.0,
      "sync_stride_sec": 1.0,
      "sync_audio_source_policy": "generated_audio_track_or_materialized_audio",
      "sync_audio_crop_policy": "align_to_generated_video_duration",
      "frame_sample_policy": "uniform",
      "face_track_policy": {
        "detector": "retinaface|insightface",
        "track_selection": "highest_ref_similarity_track",
        "fallback": "largest_center_face",
        "min_track_length": 8,
        "allow_per_frame_best_face": false,
        "track_switch_penalty": true,
        "degrade_on_track_switch": true
      },
      "pose_metric": {
        "extractor": "DWPose",
        "keypoint_set": "body_18",
        "coordinate_space": "normalized_xy",
        "normalization_mode": "normalized_image",
        "distance_metric": "euclidean",
        "akd_units": "normalized_image",
        "pck_threshold": 0.05,
        "pck_threshold_units": "normalized_image",
        "visibility_score_min": 0.3,
        "missing_keypoint_policy": "exclude_unmatched",
        "frame_alignment": "materialized_frame_index"
      }
    }
  },
  "cache": {
    "enabled": false,
    "lookup_policy": "single_run_reuse_only",
    "cache_root": "/abs/path/eval_cache/v1",
    "run_cache_manifest": "/abs/path/eval_outputs/<run_id>/cache_manifest.json",
    "key_policy": "content_hash_plus_config",
    "require_metric_provenance_match": true,
    "require_benchmark_fingerprint_for_benchmark_bound_cache": true,
    "allow_cross_benchmark_feature_cache": false,
    "allow_cross_run_cache": false,
    "record_cache_events": true
  },
  "reports": {
    "write_metrics_json": true,
    "write_paper_table_csv": true,
    "write_current_run_table_csv": true,
    "write_comparison_table_csv": true,
    "write_html_report": true
  },
  "acceptance_criteria": {
    "enabled": true,
    "rules": [
      {
        "name": "sanity_pass_rate",
        "type": "absolute",
        "metric": "sanity_pass_rate",
        "warn_below": 0.95,
        "fail_below": 0.9
      },
      {
        "name": "facesim_not_regressed",
        "type": "relative_to_baseline",
        "metric": "facesim_mean",
        "baseline_source": "comparison.baseline_eval_dir",
        "warn_delta_below": -0.02,
        "fail_delta_below": -0.05
      }
    ]
  },
  "comparison": {
    "enabled": false,
    "baseline_eval_dir": null,
    "baseline_metrics_json": null,
    "baseline_label": "baseline",
    "current_label": "current",
    "sample_policy": "overlap",
    "require_same_benchmark": true,
    "require_metric_provenance_match": true,
    "allow_degraded_in_primary_delta": false
  }
}
```

`metrics.json` must include:

- `resolved_config_path`;
- `resolved_config_sha256`;
- optionally a compact copy of the resolved config summary.

For `report_only`, the new run should copy or reference the source run's `resolved_config.json`, then write its own report-only `resolved_config.json` that records the source metrics path and report-generation settings. `report_only` must not rewrite the source resolved config.

## Task Profiles and Conditioning Contracts

Different digital-human generation tasks use different benchmark conditions. The eval agent must distinguish them explicitly instead of assuming every benchmark row has the same meaning.

Each run must declare a `task` block in `resolved_config.json`.

Current and planned task profiles:

| Task Name | Required Conditions | Optional Conditions | Typical Metrics |
| --- | --- | --- | --- |
| `audio_ref_text2video` | audio, reference image, text condition | target video, target pose | sanity, FaceSim, Sync, IQA/AES, optional AKD/PCK |
| `audio_ref_pose_text2video` | audio, reference image, pose, text condition | target video | sanity, FaceSim, Sync, AKD/PCK, IQA/AES |
| `audio_ref2video` | audio, reference image | target video, target pose | sanity, FaceSim, Sync, IQA/AES |
| `ref_text2video` | reference image, text condition | target video, target pose | sanity, FaceSim, IQA/AES, optional TA |
| `pose_audio2video` | pose, audio | target video | sanity, Sync, AKD/PCK, IQA/AES |

For the current model, use:

```json
{
  "task": {
    "name": "audio_ref_text2video",
    "version": "v1",
    "required_conditions": ["audio", "reference_image", "text"],
    "optional_conditions": ["target_video", "target_pose"],
    "output_type": "digital_human_video"
  }
}
```

The current text condition is fixed embedding, so the task still includes `text`, but `conditioning.text_mode=fixed_embedding` and `prompt_for_generation=false`.

### Task Contract Fields

Each task profile should define:

```json
{
  "task_name": "audio_ref_pose_text2video",
  "required_benchmark_fields": ["wav_path", "imgpath", "posepath"],
  "optional_benchmark_fields": ["prompt", "text", "videopath"],
  "required_manifest_fields": ["video", "key"],
  "condition_roles": {
    "audio": {"source": "wav_path", "mode_field": "audio_mode"},
    "reference_image": {"source": "imgpath|videopath:first_frame", "mode_field": "reference_mode"},
    "pose": {"source": "posepath", "mode_field": "pose_mode"},
    "text": {"source": "prompt|text_emb_path", "mode_field": "text_mode"}
  },
  "allowed_metrics": ["sanity", "facesim", "sync", "pose", "quality"],
  "default_metrics": ["sanity", "facesim", "sync", "quality"],
  "normalization": {
    "materialize_ref": true,
    "materialize_audio": true,
    "materialize_pose": false,
    "materialize_target_video": true
  }
}
```

### Pose Conditioning Modes

Pose is optional for the current task but required for pose-conditioned tasks.

Supported `pose_mode`:

| Mode | Behavior | Use Case |
| --- | --- | --- |
| `none` | No pose condition. | Current `audio_ref_text2video` model. |
| `benchmark_pose` | Use each sample's `posepath` as pose condition. | `audio_ref_pose_text2video`. |
| `fixed_pose` | Use one pose sequence for all samples. | Controlled pose ablation. |
| `manifest_pose` | Use pose path recorded in an inference manifest. | Re-evaluating pose-conditioned outputs. |
| `generated_pose_extract` | Extract pose from generated video for AKD/PCK evaluation, not as generation input. | Metric-only pose evaluation. |

### Task-Aware Metric Gating

Metric gating must check both actual conditioning and task profile.

Examples:

- If task is `audio_ref_text2video`, missing `posepath` should not fail generation evaluation; AKD/PCK is optional.
- If task is `audio_ref_pose_text2video`, missing `posepath` is a task-contract failure for `infer_eval`.
- If task has no audio condition, Sync metrics should be skipped unless generated video has audio and user explicitly requests output audio sanity.
- If task uses `fixed_embedding`, text alignment should be skipped even though the task has a text condition.
- If task uses `benchmark_prompt`, text alignment can be enabled later.

### Task-Aware Comparisons

Baseline/current comparison requires matching task profile by default:

- same `task.name`;
- same `task.version`;
- compatible required conditions;
- compatible metric definitions.

Comparing different tasks should be marked `not_comparable` for task-dependent metrics unless explicitly requested.

## Config-Aligned Benchmark Normalization

The benchmark media should be materialized into model-compatible inputs according to the resolved generation config.

### Generation / Eval Profile Resolution

Every run must resolve a `generation` profile before benchmark materialization. This profile defines the evaluation output contract:

- output width and height;
- output FPS;
- output frame count;
- derived duration;
- resize/crop policy;
- expected orientation/aspect ratio;
- whether generated outputs must match the profile exactly.

The current default profile is:

```json
{
  "profile_name": "current_480x832_20fps_81f",
  "source": "model_config",
  "width": 480,
  "height": 832,
  "fps": 20,
  "frame_num": 81,
  "duration_sec": 4.05,
  "orientation": "portrait"
}
```

This is derived from `configs/infer_audio_i2v.yaml` and the training configs:

- `generation.width=480`
- `generation.height=832`
- `generation.fps=20`
- `generation.frame_num=81`
- `duration_sec = frame_num / fps = 4.05`

The profile can also be user-specified:

```text
用 720x1280、25fps、121帧评测最新权重
```

```json
{
  "generation": {
    "profile_name": "explicit_720x1280_25fps_121f",
    "source": "user_explicit",
    "width": 720,
    "height": 1280,
    "fps": 25,
    "frame_num": 121,
    "duration_sec": 4.84,
    "orientation": "portrait",
    "resize_policy": "cover_center_crop",
    "must_match_generated_outputs": true,
    "generated_output_mismatch_policy": "fail"
  }
}
```

Use this precedence order:

1. Explicit user request or CLI/config override, such as `--width 720 --height 1280 --fps 25 --frame_num 121`.
2. Explicit eval config file.
3. Checkpoint/inference config associated with the model.
4. In `manifest_eval`, the generation profile recorded in `inference_manifest.jsonl`.
5. In `video_dir_eval`, probe generated videos only as a last resort, and mark `source=generated_probe`.

If the user says "720p", the agent should resolve it to explicit width/height before execution. For vertical digital human videos this is usually `720x1280`, but the resolved config must store the exact dimensions. Do not leave `720p` as an ambiguous string.

The chosen generation profile controls benchmark materialization. If the profile changes from `480x832 / 20fps / 81f` to `720x1280 / 25fps / 121f`, then target clips, reference images, audio segments, pose segments, validation, cache keys, and metric provenance must all reflect the new profile.

Generated-output validation must check:

- generated video width and height;
- generated FPS, within tolerance;
- frame count or duration;
- audio duration when applicable.

If generated outputs do not match `generation` and `must_match_generated_outputs=true`, fail the affected sample or run according to `generated_output_mismatch_policy`. Do not silently resize generated outputs for model-quality metrics unless the config explicitly requests a report-only visualization resize.

The agent should not directly feed arbitrary benchmark media sizes, frame rates, or durations into inference. It should create a per-run materialized benchmark directory:

```text
eval_outputs/<run_id>/benchmark_materialized/
  refs/
    000001.png
  target_clips/
    000001.mp4
  audio_segments/
    000001.wav
  pose_segments/
    000001.pt
  manifest.jsonl
```

### GT Clip Selection Policy

Benchmark videos may have a different FPS from the model and may be much longer than the generated clip. Normalization must first select a GT time window, then derive all conditions and targets from that same window.

For the current default profile:

```text
target_duration_sec = frame_num / fps = 81 / 20 = 4.05s
```

Supported `gt_clip_policy`:

| Policy | Behavior | Use Case |
| --- | --- | --- |
| `start` | Use `[0, duration_sec]`. | Default when no segment metadata exists. |
| `center` | Use a centered window of `duration_sec`. | Generic long-video evaluation. |
| `manifest_segment` | Use `start/end` from an inference or benchmark manifest. | Re-evaluating segmented outputs. |
| `benchmark_fields` | Use explicit benchmark fields such as `start_time`, `end_time`, `frame_start`, `frame_end`. | Datasets with annotated clips. |
| `random_seeded` | Deterministically sample a window with a fixed seed. | Ad-hoc stress tests only, not official splits unless frozen. |
| `fixed_offset` | Use a configured start time. | Debugging specific temporal regions. |

The selected GT clip must be recorded per sample:

```json
{
  "gt_clip_start_sec": 0.0,
  "gt_clip_end_sec": 4.05,
  "gt_clip_duration_sec": 4.05,
  "gt_clip_policy": "start",
  "source_video_fps": 25.0,
  "target_fps": 20,
  "target_frame_num": 81
}
```

If the source video/audio/pose is shorter than the selected window, use explicit policies:

- `pad`: pad last frame / silence / last pose and record warning;
- `fail`: fail sample materialization;
- `skip`: skip sample for affected condition or metric.

### Synchronized Materialization Order

For every sample, materialization order should be:

1. Resolve `sample_id`.
2. Select GT clip window from `videopath` or segment metadata.
3. Materialize target video clip from `videopath` within the selected window.
4. Extract the reference image from the first frame of the selected GT clip.
5. Materialize audio segment from `wav_path` using the same selected time window unless `audio_mode` overrides it.
6. Materialize pose segment from `posepath` using the same selected frame/time window when pose is required or used for metrics.
7. Validate that the materialized target clip and materialized audio segment are synchronized when benchmark audio is used as an evaluation condition.
8. Write per-sample materialization metadata.

This ensures ref image, target video, audio, and pose all correspond to the same temporal region.

### Ref Image Policy

The current model was trained with the reference image as the first video frame. Therefore the eval agent must explicitly control how the reference image is chosen.

Supported `ref_source_policy`:

| Policy | Behavior | Use Case |
| --- | --- | --- |
| `target_first_frame` | Extract the first frame of the selected GT clip, then apply the model resize/crop. | Recommended default for current first-frame-ref model. |
| `benchmark_img` | Use benchmark `imgpath`, then apply the model resize/crop. | Use only when benchmark images are trusted. |
| `benchmark_img_assert_first_frame` | Use benchmark `imgpath` only if it matches the selected GT clip first frame within tolerance; otherwise fail or skip sample. | Strict data validation. |
| `benchmark_img_warn_if_mismatch` | Use benchmark `imgpath`, but record warning if it does not match the selected GT clip first frame. | Exploratory analysis. |

The resize/crop implementation should match `scripts/infer_audio_i2v.py::prepare_ref_image`: scale to cover target `width x height`, then center crop.

### Current Benchmark First-Frame Check

A full check over the current 300-row `benchmark/benchmark.jsonl` using `gt_clip_policy=start` found:

- 300/300 `imgpath` and `videopath` frame 0 shapes match.
- 114/300 have first-frame MAE <= 1.
- 140/300 have first-frame MAE <= 10.
- Worst observed MAE is 89.7.

So the current benchmark cannot be assumed to have `imgpath == selected GT clip first frame` for all samples. For the current first-frame-ref model, default to `ref_source_policy=target_first_frame` unless the user explicitly requests `benchmark_img`.

The validation report should include:

```json
{
  "key": "000001",
  "imgpath_vs_videopath_first_frame": {
    "shape_match": true,
    "mae": 0.42,
    "psnr": 48.0,
    "match": true,
    "threshold": {
      "mae_max": 2.0,
      "psnr_min": 35.0
    }
  }
}
```

Thresholds must be configurable in `resolved_config.json`.

### Target Video Clip Policy

For metrics that compare generated video to target video or create target artifacts, materialize a target clip aligned to the generation config:

- decode `videopath`;
- use the selected GT clip window;
- resample frames to `generation.fps`;
- keep exactly `generation.frame_num` frames;
- use `short_video_policy=pad|fail`;
- resize/crop to `generation.width x generation.height` with the configured resize/crop policy;
- save the materialized target clip under `benchmark_materialized/target_clips/`.

The original benchmark `videopath` must remain recorded. Metrics should declare whether they use the original target video or the materialized target clip.

### Audio Segment Policy

For audio-conditioned generation, materialize audio to the generation duration:

- source: `wav_path` unless `audio_mode` says otherwise;
- start/end: selected GT clip window unless `audio_mode` says otherwise;
- duration: `generation.frame_num / generation.fps`;
- resample to the model/evaluator sample rate, typically `16000`;
- mono output unless a metric explicitly requires stereo;
- save to `benchmark_materialized/audio_segments/`.

If audio is shorter than the required duration:

- `pad`: pad with silence and record warning;
- `fail`: fail audio-dependent metrics for that sample;
- `skip`: skip that sample for audio-dependent metrics.

The policy must be explicit in `resolved_config.json`.

### Benchmark Target-Audio Sync Validation

When `wav_path` or `materialized_audio` is used as the audio condition for Sync evaluation, the agent must first validate that the selected target video clip and selected audio segment are themselves synchronized.

This is benchmark validation, not a model-quality metric. It answers:

```text
Does benchmark_materialized/target_clips/<sample_id>.mp4 match
benchmark_materialized/audio_segments/<sample_id>.wav for the same GT window?
```

Required inputs:

- `materialized_target_clip`;
- `materialized_audio`;
- selected GT clip start/end/duration;
- Sync model/provenance and window policy used for validation.

Recommended validation method:

- run the same Sync-C / Sync-D implementation on `materialized_target_clip` versus `materialized_audio`;
- use the same target duration, audio sample rate, face-crop policy, and window policy that generated Sync metrics will use;
- write results to `validation_report.json`, not only to metric outputs;
- save a per-sample validation artifact such as `artifacts/<sample_id>/benchmark_target_audio_sync_curve.png` when sliding windows are used.

Per-sample validation output:

```json
{
  "key": "000001",
  "benchmark_target_audio_sync": {
    "status": "ok|degraded|failed|skipped",
    "target_video_clip": "benchmark_materialized/target_clips/000001.mp4",
    "audio_segment": "benchmark_materialized/audio_segments/000001.wav",
    "gt_clip_start_sec": 0.0,
    "gt_clip_end_sec": 4.05,
    "sync_c": 6.8,
    "sync_d": 6.4,
    "coverage": {"valid": 8, "total": 8, "rate": 1.0, "min_required": 0.75},
    "warnings": [],
    "error": null,
    "provenance_fingerprint": "sha256"
  }
}
```

Policy:

- If target-audio sync validation is `ok`, generated-video Sync metrics can be interpreted normally.
- If validation is `degraded`, generated-video Sync may still be computed, but it must carry a warning such as `benchmark_target_audio_sync_degraded`; aggregate reporting should keep it out of `ok_only`.
- If validation is `failed`, benchmark-audio-based generated Sync should be `skipped` or excluded from primary Sync aggregates according to `validation.benchmark_av_sync.on_failed`.
- The report must not claim a generated model has bad lip-sync when the benchmark target/audio pair itself is invalid or unverified.
- In `paper` preset, samples with failed benchmark target-audio sync should be excluded from primary Sync aggregates by default and counted in the benchmark validation summary.

This validation is required for `audio_ref_text2video`, `audio_ref_pose_text2video`, and any task where benchmark `wav_path` is treated as the expected speech/audio for the target video.

### Pose Segment Policy

For pose-conditioned generation or AKD/PCK evaluation, pose must be aligned to the same selected GT clip window.

Input `posepath` in the current benchmark is a per-frame list at the source target-video frame rate. Materialization should:

- load `posepath`;
- align pose frames to the selected GT clip window;
- resample or index poses to `generation.frame_num` at `generation.fps`;
- preserve the original pose format;
- save to `benchmark_materialized/pose_segments/`;
- record source pose frame count, selected indices, and any padding.

If pose is required by the task and cannot be materialized, fail before inference. If pose is optional and cannot be materialized, mark pose metrics as `skipped` or `failed` with `null + error`, according to the metric policy.

### Normalization Fields in Resolved Config

`resolved_config.json` should include:

```json
{
  "generation": {
    "profile_name": "current_480x832_20fps_81f",
    "source": "model_config",
    "source_path": "/abs/path/configs/infer_audio_i2v.yaml",
    "width": 480,
    "height": 832,
    "fps": 20,
    "frame_num": 81,
    "duration_sec": 4.05,
    "orientation": "portrait",
    "resize_policy": "cover_center_crop",
    "must_match_generated_outputs": true,
    "generated_output_mismatch_policy": "fail"
  },
  "benchmark_normalization": {
    "enabled": true,
    "materialized_dir": "/abs/path/eval_outputs/<run_id>/benchmark_materialized",
    "gt_clip_policy": "start",
    "gt_clip_start_sec": 0.0,
    "ref_source_policy": "target_first_frame",
    "resize_policy": "cover_center_crop",
    "target_video_width": 480,
    "target_video_height": 832,
    "target_video_fps": 20,
    "target_video_frames": 81,
    "target_video_short_policy": "pad",
    "audio_sample_rate": 16000,
    "audio_channels": 1,
    "audio_duration_sec": 4.05,
    "audio_short_policy": "pad",
    "pose_enabled": false,
    "pose_frame_num": 81,
    "pose_short_policy": "pad",
    "first_frame_validation": {
      "enabled": true,
      "mae_max": 2.0,
      "psnr_min": 35.0
    }
  }
}
```

The materialized benchmark manifest should record, per sample:

```json
{
  "key": "000001",
  "original_imgpath": ".../benchmark/img/000001.jpg",
  "original_videopath": ".../benchmark/video/000001.mp4",
  "original_wav_path": ".../benchmark/audio/...",
  "materialized_ref": "benchmark_materialized/refs/000001.png",
  "materialized_target_clip": "benchmark_materialized/target_clips/000001.mp4",
  "materialized_audio": "benchmark_materialized/audio_segments/000001.wav",
  "materialized_pose": "benchmark_materialized/pose_segments/000001.pt",
  "gt_clip_start_sec": 0.0,
  "gt_clip_end_sec": 4.05,
  "gt_clip_policy": "start",
  "ref_source_policy": "target_first_frame",
  "generation_profile_name": "current_480x832_20fps_81f",
  "generation_source": "model_config",
  "generation_width": 480,
  "generation_height": 832,
  "generation_fps": 20,
  "generation_frame_num": 81,
  "generation_duration_sec": 4.05
}
```

## Run Modes

The agent must support five top-level run modes.

| Mode | Inference | Metrics | Reports | Use Case |
| --- | --- | --- | --- | --- |
| `infer_eval` | Yes | Yes | Yes | Normal "evaluate this checkpoint" workflow. |
| `manifest_eval` | No | Yes | Yes | Recompute metrics for existing outputs with `inference_manifest.jsonl`. |
| `video_dir_eval` | No | Yes | Yes | Evaluate a bare directory of videos by creating a synthetic manifest. |
| `report_only` | No | No | Yes | Regenerate `html_report.html` or `paper_table.csv` from existing `metrics.json`. |
| `compare_only` | No | No | Yes | Compare two existing eval runs without recomputing metrics. |

### `infer_eval`

`infer_eval` is the default mode.

It resolves a checkpoint, runs generation on selected benchmark samples, writes an inference manifest, computes metrics, and writes reports.

Example:

```text
用10个样本评估最新权重
```

### `manifest_eval`

`manifest_eval` is a first-class mode. It skips inference and recomputes metrics from an existing `inference_manifest.jsonl`.

Use it when:

- A batch of videos has already been generated.
- The FaceSim implementation changed and metrics need to be recomputed.
- Outputs from another model need to be evaluated.
- A previous inference run exists but the metric set or metric implementation changed.

Allowed inputs:

1. `generated_manifest`: preferred. A JSONL manifest with `key` and `video`, plus optional `ref`, `prepared_ref`, `audio`, `audio_segment`, `prompt`, and conditioning fields.
2. `source_eval_dir`: an existing `eval_outputs/<run_id>/` directory. The agent should reuse its `inference/inference_manifest.jsonl`.

`manifest_eval` should be preferred over `video_dir_eval` whenever a manifest is available because it preserves the actual generation provenance.

Example structured config:

```json
{
  "run_mode": "manifest_eval",
  "checkpoint": null,
  "benchmark_jsonl": "benchmark/benchmark.jsonl",
  "generated_manifest": "eval_outputs/old_run/inference/inference_manifest.jsonl",
  "metrics": ["facesim"],
  "output_dir": "eval_outputs/rerun_facesim_<timestamp>"
}
```

When `manifest_eval` is used, the report must clearly state:

- no inference was run;
- the source manifest path;
- the source generated videos path from the manifest;
- which metrics were recomputed;
- which previous metrics, if any, were reused.

### `video_dir_eval`

`video_dir_eval` skips inference and evaluates a bare directory of generated videos.

Use it only when no manifest exists, for example when evaluating another model's outputs that are only available as `000001.mp4`, `000002.mp4`, etc.

Allowed input:

1. `generated_video_dir`: a directory of videos named by benchmark sample ID, such as `000001.mp4`.

The agent must create a synthetic manifest before validation and metrics:

```text
eval_outputs/<run_id>/inference/inference_manifest.synthetic.jsonl
```

`video_dir_eval` has weaker provenance than `manifest_eval`. It can validate the generated video key against `benchmark.jsonl`, but it cannot prove which reference image, audio, prompt, text embedding, seed, or checkpoint was used unless those are supplied separately.

Metrics requiring missing provenance must be skipped or downgraded according to `benchmark_linkage` and `conditioning`.

#### `video_dir_eval` Linkage Modes

`video_dir_eval` must declare one linkage mode:

| Linkage Mode | Behavior | Allowed Metrics |
| --- | --- | --- |
| `strict` | Every generated video must map to exactly one benchmark row through explicit mapping, manifest, or unambiguous filename parser. Benchmark-dependent inputs must exist and pass validation. | Full benchmark-dependent metrics allowed after validation. |
| `weak_key_match` | Filenames appear to contain benchmark IDs, but generation provenance is missing. The report must mark provenance as weak. | Benchmark-dependent metrics may run only if required ref/audio/target fields are recovered from benchmark and the user accepts weak linkage. |
| `unpaired` | Videos are not linked to benchmark rows. No target, ref, prompt, or benchmark audio is assumed. | Only no-reference metrics such as sanity and backend-aware IQA/AES. |

Rules:

- `strict` is required for formal internal checkpoint evaluation.
- `weak_key_match` is acceptable for exploratory third-party output analysis, but comparison and paper claims should be disabled by default.
- `unpaired` must not run FaceSim, Sync, AKD/PCK, or paired GT diagnostics unless the user supplies the missing reference/audio/target data through a manifest or explicit mapping.
- In `unpaired`, FaceSim requires user-provided reference images or a manifest with reference provenance.
- In `unpaired`, Sync requires a generated audio track or user-provided audio source. Benchmark `wav_path` must not be inferred.
- The resolved linkage mode must be recorded in `resolved_config.json`, `validation_report.json`, and `metrics.json`.

## Manifest Authority and Reuse

`inference_manifest.jsonl` will be used frequently and should be treated as a first-class artifact.

It is the authority for generated-output provenance:

- generated video path;
- actual reference image or prepared reference used for generation;
- actual audio source or audio segment used for generation;
- actual text embedding or prompt condition used for generation;
- seed, checkpoint, CFG mode, guide scale, and other generation settings when available.

It is not the authority for benchmark ground truth. `benchmark.jsonl` remains the authority for target/reference metadata:

- benchmark sample ID;
- benchmark reference image;
- benchmark target video;
- benchmark pose;
- benchmark audio;
- benchmark prompt/text metadata.

For `manifest_eval`, use this source priority:

1. Explicit `--generated_manifest`.
2. `--source_eval_dir/inference/inference_manifest.jsonl`.

For `video_dir_eval`, build a synthetic manifest from `--generated_video_dir`.

The parsed manifest should be reused by:

- `manifest_eval`;
- `video_dir_eval` after synthetic manifest creation;
- benchmark linkage validation;
- metric gating by actual conditioning;
- per-sample artifact generation;
- report generation;
- rerunning a subset of metrics such as a new FaceSim implementation.

### Trust Model

Trust the manifest for what was actually generated and how it was generated. Do not blindly trust it for benchmark correctness.

The correct rule is:

```text
generated inputs/provenance -> inference_manifest.jsonl
benchmark identity/targets  -> benchmark.jsonl
metric scores              -> metrics.json
```

This means:

- if the manifest says a sample used `audio_segment=...`, Sync should use that generated audio source;
- if the manifest says `text_mode=fixed_embedding`, TA should remain skipped even if benchmark has `prompt`;
- if the manifest says `prepared_ref=...`, FaceSim may compare against that prepared reference or the original benchmark image, but the chosen reference must be recorded;
- if `benchmark_linkage=strict`, the manifest `key` and benchmark row must still validate before benchmark-dependent metrics run.

### Manifest Validation

Every loaded manifest should be normalized and validated once at the start of a run.

Required checks:

- every row has a unique `key`;
- every row has an existing generated `video`;
- paths are resolved relative to the manifest location when needed;
- optional fields such as `ref`, `prepared_ref`, `audio`, and `audio_segment` are checked if present;
- manifest schema version, source path, row count, and file fingerprint are recorded in `metrics.json`.

In `video_dir_eval`, the agent should create a synthetic manifest in the new eval output directory:

```text
eval_outputs/<run_id>/inference/inference_manifest.synthetic.jsonl
```

Synthetic manifests must be marked:

```json
{
  "manifest_type": "synthetic",
  "provenance_confidence": "video_path_only"
}
```

Metrics that require missing provenance, such as exact audio-condition matching, must be skipped or downgraded according to validation level.

### `report_only`

`report_only` regenerates report files from an existing `metrics.json`.

Use it when:

- `html_report.html` needs a new layout;
- `paper_table.csv` columns need to change;
- historical `eval_outputs` need to be viewed again without recomputing metrics.

Example:

```json
{
  "run_mode": "report_only",
  "source_eval_dir": "eval_outputs/old_run",
  "metrics_json": "eval_outputs/old_run/metrics.json",
  "output_dir": "eval_outputs/old_run_report_refresh"
}
```

`report_only` must not change metric values. It may only transform existing metric data into new report artifacts.

### `compare_only`

`compare_only` compares two existing eval runs from their `metrics.json` and `resolved_config.json`.

Use it when:

- asking "最新权重比之前的权重好在哪里，坏在哪里？";
- comparing current run against a previous checkpoint run;
- comparing this model against another model's outputs;
- regenerating comparison tables after report formatting changes.

Example:

```json
{
  "run_mode": "compare_only",
  "baseline_metrics_json": "eval_outputs/baseline/metrics.json",
  "current_metrics_json": "eval_outputs/current/metrics.json",
  "baseline_label": "checkpoint_005000",
  "current_label": "checkpoint_007500",
  "output_dir": "eval_outputs/compare_005000_vs_007500"
}
```

`compare_only` must not recompute metric scores. It may only read existing results, validate comparability, compute deltas, and write comparison reports.

## Baseline / Previous-Run Comparison

Comparison can run as an optional phase after `infer_eval`, `manifest_eval`, or `video_dir_eval`, or as standalone `compare_only`.

Supported baseline sources:

| Source | Behavior |
| --- | --- |
| `baseline_eval_dir` | Use `baseline_eval_dir/metrics.json` and `baseline_eval_dir/resolved_config.json`. |
| `baseline_metrics_json` | Use an explicit metrics file. |
| `previous_run:auto` | Select the most recent comparable eval run before the current run. |
| `previous_checkpoint:auto` | Select the latest eval run for the previous checkpoint step. |

The comparison stage should write:

```text
eval_outputs/<run_id>/
  current_run_table.csv
  comparison_table.csv
  comparison.json
  sample_comparison.jsonl
```

### Current Run Table

`current_run_table.csv` is the experiment-result table for the run itself.

It should contain one row per run or per requested group:

```text
run_id,label,checkpoint,split,n,
sanity_pass_rate,
facesim_mean,facesim_status_ok,facesim_status_degraded,
sync_c,sync_d,
iqa_qalign_sampled,aes_qalign_sampled,
akd,pck,
diagnostic_top_category
```

This table is useful even without a baseline.

IQA/AES columns in this table must use backend-aware metric variant names. If a different backend is used, replace `iqa_qalign_sampled`/`aes_qalign_sampled` with the resolved variants from `metric_plan.json`.

### Comparison Table

`comparison_table.csv` compares baseline and current run.

Suggested columns:

```text
metric,group,baseline_label,current_label,
baseline_value,current_value,
delta,relative_delta,
direction,higher_is_better,
winner,
baseline_status_counts,current_status_counts,
comparable_n,baseline_n,current_n,
provenance_match,comparison_status,warning
```

Metric direction must be explicit:

| Metric | Better Direction |
| --- | --- |
| FaceSim | higher |
| face_detection_rate | higher |
| Sync-C | higher |
| Sync-D | lower |
| IQA / AES | higher |
| AKD | lower |
| PCK | higher |
| frozen_frame_rate / black_frame_rate | lower |
| failure/degraded rate | lower |

### Comparability Rules

Do not compare numbers blindly.

Before computing deltas:

- benchmark identity should match unless explicitly overridden;
- task profile name/version should match by default;
- official split name/version/hash should match by default;
- selected sample sets should match or overlap should be reported;
- metric provenance fingerprints should match by default;
- metric definitions and better-direction metadata must match;
- `ok_only` and `ok_plus_degraded` aggregates should be compared separately;
- failed/skipped values must not be treated as zero.

If sample sets differ, use `sample_policy`:

| Policy | Behavior |
| --- | --- |
| `exact` | Require identical sample IDs. |
| `overlap` | Compare only overlapping sample IDs and report excluded IDs. |
| `aggregate_only` | Compare only aggregate rows; mark weaker evidence. |

If metric provenance differs, mark `comparison_status=not_comparable` unless `allow_cross_provenance=true`. Even when allowed, the warning must say that deltas may reflect metric implementation changes rather than model changes.

For backend-sensitive metrics such as IQA/AES, metric variant names must match exactly before comparison. `iqa_qalign_sampled` and `iqa_custom_aesthetic_v1_sampled` are different metrics and should not be compared under a generic `iqa` heading.

### Comparative Diagnostic Summary

`html_report.html` and `comparison.json` should include a count-grounded comparison summary.

Example:

```text
Compared checkpoint_007500 against checkpoint_005000 on 20 overlapping samples.
The current checkpoint improves identity consistency: FaceSim increases by +0.041
on ok-only samples, and degraded FaceSim cases decrease from 7/20 to 4/20.
The main regression is audio-visual sync: Sync-D worsens by +0.62 and low-window
sync degraded cases increase from 2/20 to 5/20. Visual quality is mostly unchanged.
```

Rules:

- Separate improvements and regressions.
- Mention comparable sample count.
- Mention metrics skipped from comparison due to provenance mismatch.
- Mention if conclusions are based on `ok_only` or include degraded samples.
- Link worst-regressed and most-improved representative sample cards.

## Benchmark Linkage Validation

Evaluation quality depends on knowing which benchmark row each generated video corresponds to.

For normal benchmark evaluation, every generated sample must be linked to exactly one row in `benchmark/benchmark.jsonl`.

The linkage key should be the benchmark sample ID, usually derived from paths like:

```text
benchmark/img/000001.jpg
benchmark/video/000001.mp4
benchmark/pose/000001.pt
generated/videos/000001.mp4
```

## Sample ID Generation and Collision Rules

`sample_id` is a stable benchmark identity, not just a filename basename.

The agent should create a benchmark index before split selection, inference, or validation.

### Benchmark Sample ID Priority

For each benchmark row, resolve `sample_id` in this order:

1. Use explicit `sample_id` if present.
2. Use explicit `id` if present.
3. Derive from path stems only if all available core benchmark paths agree:
   - `imgpath`
   - `videopath`
   - `posepath`
4. If stems disagree, fail benchmark indexing unless a user-provided mapping is supplied.

For the current benchmark, stems like `000001` should be valid only if:

```text
imgpath   -> 000001
videopath -> 000001
posepath  -> 000001
```

`wav_path` should not be used as the primary sample ID source because audio filenames are not aligned to benchmark numeric IDs.

### Canonical ID Format

Canonical IDs should be strings, preserving leading zeros:

```text
000001
000002
```

Do not convert IDs to integers in persisted files.

If a source ID is numeric, zero-pad only when the benchmark itself uses a fixed-width numeric ID scheme. The padding width must be recorded in `resolved_config.json`.

### Collision Detection

The benchmark index must fail fast on:

- duplicate `sample_id`;
- duplicate `imgpath` assigned to multiple sample IDs;
- duplicate `videopath` assigned to multiple sample IDs;
- split files containing IDs not present in benchmark;
- manifest rows with duplicate `key`;
- generated video directory basenames that collide after normalization.

Failure should produce a validation error, not silently suffix IDs.

### Generated Manifest Key Rules

For generated manifests:

- `key` should match benchmark `sample_id`.
- If both `key` and `benchmark_key` exist, `benchmark_key` is used for benchmark linkage and `key` remains the generated-output row ID.
- If `key != benchmark_key`, report this in `validation_report.json`.
- If neither exists, `manifest_eval` should fail unless a mapping file is supplied.

Recommended manifest fields:

```json
{
  "key": "000001",
  "benchmark_key": "000001",
  "sample_id": "000001",
  "video": "videos/000001.mp4"
}
```

### Video Directory ID Rules

For `video_dir_eval`, the agent may infer IDs from video filenames only if:

- each video basename maps to exactly one benchmark `sample_id`;
- normalized basenames are unique;
- the filename parsing rule is recorded in `resolved_config.json`.

If third-party outputs use names like `sample_000001_seed42.mp4`, require a configured regex:

```json
{
  "video_dir_id_parser": {
    "type": "regex",
    "pattern": "sample_(\\d{6})_seed\\d+",
    "group": 1
  }
}
```

If filenames cannot be mapped unambiguously, require a manifest or explicit mapping CSV:

```text
generated_video,sample_id
/path/output_a.mp4,000001
/path/output_b.mp4,000002
```

### Persisted ID Artifacts

Write the benchmark index used by the run:

```text
eval_outputs/<run_id>/benchmark_index.json
```

Each entry should include:

```json
{
  "sample_id": "000001",
  "row_index": 0,
  "id_source": "path_stem_consensus",
  "img_stem": "000001",
  "video_stem": "000001",
  "pose_stem": "000001",
  "wav_stem": "emilia_zh_0010618912"
}
```

`resolved_config.json` and `metrics.json` must record:

- ID generation policy;
- ID parser config;
- benchmark index path;
- benchmark index sha256;
- collision count, which must be zero for `strict` benchmark linkage.

### Validation Levels

| Level | Behavior | Use Case |
| --- | --- | --- |
| `strict` | Sample ID must exist in benchmark; benchmark media must exist; if conditioning mode says `benchmark_img` or `benchmark_wav`, manifest ref/audio must match benchmark row. | Normal internal evaluation. |
| `relaxed` | Sample ID must exist in benchmark, but actual ref/audio may differ if conditioning says `fixed_ref`, `fixed_audio`, `manifest_prepared_ref`, or `manifest_audio_segment`. | Ablations and third-party outputs with known mapping. |
| `metadata_only` | Video can be scored with no benchmark target for metrics that do not need benchmark metadata; benchmark-dependent metrics are skipped. | External videos without benchmark mapping. |

Default for this project should be `strict`.

### Required Checks

For each generated sample:

- `key` must be present and unique.
- `key` must exist in `benchmark.jsonl` unless `benchmark_linkage=metadata_only`.
- generated `video` must exist and be readable.
- benchmark `imgpath`, `videopath`, `posepath`, and `wav_path` must exist for benchmark-dependent metrics.
- benchmark `imgpath` vs selected GT clip first frame should be checked and recorded when `first_frame_validation.enabled=true`.
- if `reference_mode=benchmark_img`, the actual reference source must match benchmark `imgpath`.
- if `reference_mode=target_first_frame` or `materialized_ref`, the actual reference source should trace back to the selected GT clip first frame or the materialized ref created from it.
- if `audio_mode=benchmark_wav`, the actual audio source must match benchmark `wav_path`.
- if `text_mode=benchmark_prompt`, the actual text condition must be benchmark `prompt`.
- if a prepared/cropped reference is used, record both the original benchmark path and prepared path.

Path equality is not enough when files may be copied. The validator should support:

- normalized absolute path comparison;
- optional file size and hash comparison for copied inputs;
- explicit manifest fields such as `benchmark_key`, `source_ref`, `source_audio`, and `source_prompt`.

### Validation Outputs

Write:

```text
eval_outputs/<run_id>/validation_report.json
```

Each sample should include:

```json
{
  "key": "000001",
  "benchmark_linked": true,
  "benchmark_row_index": 0,
  "video_exists": true,
  "ref_match": true,
  "audio_match": true,
  "prompt_match": null,
  "imgpath_first_frame_match": true,
  "conditioning_match": true,
  "warnings": [],
  "errors": []
}
```

If validation fails for a sample:

- metrics that require the failed linkage must be `skipped` or `failed`;
- score fields must be `null`;
- `error.message` must explain the validation failure;
- sample artifacts should still include `error.json`.

Do not silently evaluate a video against the wrong benchmark row.

## Checkpoint Resolution

Supported user intents:

1. Explicit `model.pt`
2. Explicit `checkpoint_model_xxxxxx/`
3. Automatic latest checkpoint

Suggested latest rule:

1. Scan `logs/*/checkpoint_model_*/model.pt`.
2. Prefer the largest numeric checkpoint step.
3. Use modification time only as a tie-breaker.
4. Record the chosen checkpoint path, step, mtime, and config path in `metrics.json`.

If checkpoint resolution fails, stop before inference and report the error. Do not run partial evaluation with an unknown checkpoint.

Checkpoint resolution is only required for `infer_eval`. In `manifest_eval` and `video_dir_eval`, checkpoint information is optional metadata read from the generated manifest, synthetic manifest, or user-supplied metadata if available. In `report_only`, checkpoint information must be copied from the source `metrics.json`.

## Flexible Conditioning

The agent must distinguish benchmark metadata from actual model conditioning.

In the current training setup, `caption_emb_mode: fixed` means the model was trained with a fixed caption embedding. The benchmark `prompt` is useful metadata and may be useful for human inspection, but it should not automatically be used as the generation text condition or for text-alignment metrics.

Current repository evidence:

- `configs/rf_final.yaml` uses `caption_emb_mode: fixed`.
- `scripts/infer_audio_i2v.py` loads a fixed `text_emb_path`.
- `prompt` in inference manifests is currently descriptive metadata, not necessarily the tensor actually used as the model condition.

Therefore, each eval run must record a `conditioning` block and avoid assuming that benchmark fields were used for generation.

### Text Conditioning Modes

| Mode | Behavior | Use Case |
| --- | --- | --- |
| `fixed_embedding` | Use one fixed `text_emb_path` for all samples. Ignore per-sample benchmark `prompt` for generation. | Current model. |
| `benchmark_prompt` | Encode/use each sample's `prompt` as text condition. | Future text-conditioned benchmark. |
| `benchmark_caption_emb` | Use a per-row caption embedding path from the benchmark, if available. | Datasets with precomputed per-sample text embeddings. |
| `manifest_text_emb` | Use the text embedding path recorded in an existing inference manifest. | Evaluating pre-generated outputs. |
| `none` | No text condition. | Audio/reference-only ablation. |

For the current model, default to:

```json
{
  "text_mode": "fixed_embedding",
  "text_emb_path": "raw_dataset/text_emb/person_speaking_zh_umt5_xxl_bf16.pt",
  "prompt_for_generation": false,
  "prompt_for_reporting": true
}
```

### Audio Conditioning Modes

| Mode | Behavior | Use Case |
| --- | --- | --- |
| `benchmark_wav` | Use each sample's `wav_path`. | Normal benchmark evaluation. |
| `fixed_audio` | Use the same audio file for all samples. | Controlled identity/reference stress test. |
| `sequential_long_audio` | Slice a long audio file into per-sample segments. | Podcast/long audio evaluation. |
| `manifest_audio_segment` | Use audio segments recorded in an existing inference manifest. | Re-evaluating generated outputs. |
| `none` | No audio condition. | Audio ablation. |

### Reference Conditioning Modes

| Mode | Behavior | Use Case |
| --- | --- | --- |
| `benchmark_img` | Use each sample's `imgpath`. | Normal benchmark evaluation. |
| `fixed_ref` | Use one reference image for all samples. | Identity consistency stress test. |
| `manifest_prepared_ref` | Use prepared references from an existing inference manifest. | Re-evaluating generated outputs. |
| `target_first_frame` | Use the first frame of `videopath` as reference. | When `imgpath` is absent or inconsistent. |
| `materialized_ref` | Use the config-aligned reference created under `benchmark_materialized/refs/`. | Recommended execution source after normalization. |

### Metric Gating by Conditioning

Metrics should only run when their assumptions match the actual conditioning.

| Metric | Run When | Skip When |
| --- | --- | --- |
| FaceSim | There is a real reference image condition. | `reference_mode=none` or reference missing. |
| Sync-C / Sync-D | There is an audio condition or generated audio track, and benchmark-audio-based Sync has a valid `benchmark_av_sync` validation result when using `materialized_audio` or `benchmark_wav_crop`. | `audio_mode=none` and generated video has no audio; benchmark target-audio sync failed and policy excludes the sample. |
| TA / text alignment | `text_mode=benchmark_prompt` or equivalent per-sample text condition. | `text_mode=fixed_embedding`; do not compare generated video against benchmark `prompt` as if it conditioned generation. |
| IQA / AES | Generated video exists. | Generated video missing/unreadable. |
| AKD / PCK | Target and generated pose tracks are available. | Generated pose unavailable. |

### Similar Flexible Scenarios to Support

- Fixed caption embedding: benchmark `prompt` is reporting metadata, not generation input.
- Fixed negative prompt: `negative_text_emb_path` is shared across all samples; record it once in run metadata.
- Fixed audio source: benchmark `wav_path` should not be used for Sync unless it is actually the generated audio condition.
- Sequential long-audio slicing: each sample's true audio is `(audio_source_path, start_time, end_time)`, not benchmark `wav_path`.
- Fixed reference image: benchmark `imgpath` should not be used for FaceSim unless it was actually the identity/reference input.
- Prepared reference resizing/cropping: FaceSim should record whether it compares against original `imgpath` or prepared/cropped reference.
- CFG/conditioning ablations: `cfg_mode=off|text|audio|both|zero` and `guide_scale` affect interpretation; record them in every run.
- Re-evaluating existing outputs: trust `inference_manifest.jsonl` for actual generated-output provenance, while still validating benchmark linkage before benchmark-dependent metrics run.
- Partial conditioning failures: if one sample's audio segment extraction fails, mark audio-dependent metrics as failed for that sample without fabricating scores.

### Sync Audio Source and Cropping Policy

Sync-C / Sync-D must explicitly record which audio was used and how it was cropped.

Supported `sync_audio_source_policy`:

| Policy | Behavior | Use Case |
| --- | --- | --- |
| `generated_audio_track` | Extract audio embedded in the generated video. | Best when final deliverable video contains merged audio. |
| `manifest_audio_segment` | Use `audio_segment` from inference manifest. | Re-evaluating generated outputs. |
| `materialized_audio` | Use `benchmark_materialized/audio_segments/<id>.wav`. | Benchmark inference with GT-aligned audio. |
| `benchmark_wav_crop` | Crop raw `wav_path` using GT clip window. | When materialized audio was not persisted. |
| `generated_audio_track_or_materialized_audio` | Prefer generated embedded audio; fallback to materialized audio. | Recommended default for current pipeline. |

Supported `sync_audio_crop_policy`:

| Policy | Behavior |
| --- | --- |
| `none` | Use the selected audio as-is. |
| `align_to_generated_video_duration` | Crop or pad audio to generated video duration. |
| `align_to_gt_clip_window` | Crop raw/source audio using `gt_clip_start_sec` and `gt_clip_end_sec`. |
| `manifest_segment_bounds` | Use `audio_start_time`, `audio_end_time`, or `audio_duration` from manifest. |
| `fail_on_mismatch` | Fail Sync if audio/video durations differ beyond tolerance. |

Each Sync result must record:

```json
{
  "sync_audio": {
    "source_policy": "generated_audio_track_or_materialized_audio",
    "resolved_source": "generated_audio_track",
    "source_path": "eval_outputs/<run_id>/inference/videos/000001.mp4",
    "extracted_audio_path": "artifacts/000001/sync/audio.wav",
    "crop_policy": "align_to_generated_video_duration",
    "source_start_sec": 0.0,
    "source_end_sec": 4.05,
    "effective_start_sec": 0.0,
    "effective_duration_sec": 4.05,
    "sample_rate": 16000,
    "channels": 1,
    "duration_mismatch_sec": 0.0
  }
}
```

If the selected audio does not match the intended condition, Sync should be `degraded` or `failed` depending on severity, and the warning must identify the mismatch.

If Sync uses `materialized_audio` or `benchmark_wav_crop`, each generated-video Sync result must also reference benchmark target-audio sync validation:

```json
{
  "benchmark_audio_sync_validation": {
    "status": "ok|degraded|failed|skipped",
    "validation_report_path": "validation_report.json",
    "sample_key": "000001",
    "effect_on_generated_sync": "none|warning|degraded|excluded_from_aggregate"
  }
}
```

Generated-video Sync cannot be `ok` when required benchmark target-audio sync validation is `degraded`, `failed`, or missing. It should be `degraded` if a numeric score exists but benchmark validation is questionable, or `skipped`/excluded when the benchmark target-audio pair is unusable.

### Face Track Policy for FaceSim and Sync

Digital-human videos should usually contain one primary person, but multi-face detections can still happen because of background posters, mirrors, product packaging, reflections, or detector false positives. FaceSim and Sync mouth crops must therefore use a shared selected face track instead of independently choosing faces.

Default policy:

```json
{
  "face_track_policy": {
    "detector": "retinaface|insightface",
    "track_selection": "highest_ref_similarity_track",
    "fallback": "largest_center_face",
    "min_track_length": 8,
    "allow_per_frame_best_face": false,
    "track_switch_penalty": true,
    "degrade_on_track_switch": true
  }
}
```

Rules:

- Build face tracks from detections before running FaceSim or Sync mouth-crop extraction.
- Select one `selected_face_track_id` per sample.
- FaceSim and Sync must use the same `selected_face_track_id`.
- Prefer `track_selection=highest_ref_similarity_track` when a valid reference face embedding is available.
- Use `fallback=largest_center_face` only when reference similarity cannot be computed.
- Do not select the highest-similarity face independently per frame by default. `allow_per_frame_best_face=false` prevents hiding identity drift or track instability.
- If the selected track switches, fragments, or falls below `min_track_length`, record `face_track_switch_count` and mark FaceSim/Sync as `degraded` when numeric scores still exist.
- If no usable face track exists, FaceSim and Sync should be `failed` or `skipped` according to their metric policy, with score fields set to `null`.

Required per-sample face tracking output:

```json
{
  "face_tracking": {
    "status": "ok|degraded|failed|skipped",
    "detector": "insightface",
    "track_selection": "highest_ref_similarity_track",
    "fallback": "largest_center_face",
    "selected_face_track_id": "track_000",
    "num_tracks": 2,
    "selected_track_length": 73,
    "total_frames": 81,
    "face_track_detection_rate": 0.9012,
    "face_track_switch_count": 1,
    "allow_per_frame_best_face": false,
    "track_switch_penalty": true,
    "warnings": ["face_track_switch_count=1"],
    "error": null
  }
}
```

FaceSim output and Sync output must both include:

```json
{
  "selected_face_track_id": "track_000",
  "face_track_switch_count": 1,
  "face_track_status": "degraded"
}
```

Required per-sample artifacts:

```text
artifacts/<sample_id>/
  face_track_overlay.mp4
  selected_face_track.jsonl
  face_track_switches.json
```

`selected_face_track.jsonl` should record frame index, timestamp, selected track ID, bounding box, detection score, optional reference similarity, and whether the frame was interpolated or missing. `face_track_switches.json` should record switch/frame-gap events and their effect on metric status.

## Flexible Sample Selection

Official eval splits should be the default for comparable evaluations. Natural-language sampling should remain available for debugging and exploratory runs, but it should not be the default basis for comparing checkpoints.

## Official Eval Splits

Create versioned split files under:

```text
benchmark/splits/
  official_smoke_10.v1.json
  official_quick_50.v1.json
  official_main_300.v1.json
  official_vocal_zh_50.v1.json
  official_sing_50.v1.json
```

Suggested split tiers:

| Split | Purpose | Size |
| --- | --- | --- |
| `official_smoke_10` | Fast code path validation. | 10 |
| `official_quick_50` | Routine checkpoint comparison. | 50 |
| `official_main_300` | Full benchmark report. | 300 |
| `official_vocal_zh_50` | Chinese speech subset. | 50 |
| `official_sing_50` | Singing subset. | 50 |

Split file schema:

```json
{
  "split_name": "official_quick_50",
  "version": "v1",
  "benchmark_jsonl": "benchmark/benchmark.jsonl",
  "created_at": "2026-06-10T00:00:00Z",
  "selection_policy": "fixed_ids",
  "description": "Fixed 50-sample checkpoint comparison split.",
  "sample_ids": ["000001", "000002"],
  "metadata_counts": {
    "language": {"zh": 25, "en": 25},
    "type": {"vocal": 35, "sing": 15},
    "body": {"half": 30, "full": 20}
  }
}
```

Rules:

- Official split files are versioned and should not be modified in place after use.
- If a split changes, create a new version such as `official_quick_50.v2.json`.
- `resolved_config.json` must record `split_name`, `split_version`, `split_file`, `split_sha256`, and `selected_sample_ids`.
- `metrics.json` must record the same split metadata.
- Baseline/current comparison should require the same split by default.
- If different splits are compared, mark comparison as weaker evidence or `not_comparable` unless explicitly overridden.

Default behavior:

```text
Formal evaluation request -> official_quick_50 or official_main_300
Quick debugging request   -> official_smoke_10
Ad-hoc natural language sample count -> ad_hoc split, not comparable by default
```

Examples:

```text
用 official_quick_50 评估最新权重
```

```text
跑 official_main_300，并和上一个权重对比
```

```text
临时随机抽10个样本做 sanity debug
```

## Ad-Hoc Sample Selection

The agent must still support flexible evaluation size and sample selection from natural language for debugging.

Ad-hoc selections must be marked:

```json
{
  "split_name": "ad_hoc",
  "split_version": null,
  "comparable_by_default": false
}
```

Examples:

```text
用10个样本评估最新权重
```

```json
{
  "checkpoint": "latest",
  "sample_selection": {
    "count": 10,
    "mode": "first",
    "seed": 20260609,
    "filters": {},
    "group_by": null,
    "ids": []
  }
}
```

```text
随机抽10个样本评估最新权重
```

```json
{
  "checkpoint": "latest",
  "sample_selection": {
    "count": 10,
    "mode": "random",
    "seed": 20260609,
    "filters": {},
    "group_by": null,
    "ids": []
  }
}
```

```text
只用中文女声10条评估最新权重
```

```json
{
  "checkpoint": "latest",
  "sample_selection": {
    "count": 10,
    "mode": "first",
    "seed": 20260609,
    "filters": {
      "language": "zh",
      "gender": "female",
      "type": "vocal"
    },
    "group_by": null,
    "ids": []
  }
}
```

```text
每个type各抽5条评估最新权重
```

```json
{
  "checkpoint": "latest",
  "sample_selection": {
    "count": 5,
    "mode": "stratified_random",
    "seed": 20260609,
    "filters": {},
    "group_by": "type",
    "ids": []
  }
}
```

```text
评估000001、000003、000010这几个样本
```

```json
{
  "checkpoint": "latest",
  "sample_selection": {
    "count": 3,
    "mode": "explicit_ids",
    "seed": 20260609,
    "filters": {},
    "group_by": null,
    "ids": ["000001", "000003", "000010"]
  }
}
```

### Supported Selection Modes

| Mode | Behavior |
| --- | --- |
| `official_split` | Use fixed sample IDs from a versioned official split file. |
| `all` | Use all rows after filters. |
| `first` | Use the first `count` rows after filters, preserving benchmark order. |
| `random` | Randomly sample `count` rows after filters with a fixed seed. |
| `stratified_first` | Select `count` rows per group, preserving benchmark order inside each group. |
| `stratified_random` | Randomly select `count` rows per group with a fixed seed. |
| `explicit_ids` | Use exactly the requested sample IDs. |

Supported filter fields should initially match benchmark metadata:

- `language`
- `gender`
- `type`
- `category`
- `body`

Supported `group_by` fields should initially include:

- `language`
- `gender`
- `type`
- `category`
- `body`

### Selection Rules

- If the user says "10个样本" without saying "随机", use `mode=first`.
- If the user says "随机10个样本", use `mode=random` and record the seed.
- If the user says "全量" or "全部", use `mode=all` and `count=null`.
- If the user specifies sample IDs, use `mode=explicit_ids`.
- If the user says "快速评测", resolve both sample policy and metric preset through `presets.py`; do not assume the benchmark has a fixed total size.
- If the user says "完整评测", prefer a versioned full/main official split; if absent, use all filtered samples and record the potentially high cost.
- If the user says "论文格式的评测", prefer a versioned official split and mark results non-comparable by default if only ad-hoc sampling is available.
- If the filtered set has fewer samples than requested, run all available filtered samples and record a warning.
- The selected sample list must be written to `metrics.json`.
- The selected sample list must also be written as `eval_outputs/<run_id>/selected_samples.jsonl`.
- The report must include `requested_count`, `available_count`, `selected_count`, `selection_mode`, `filters`, `group_by`, and `seed`.

## Benchmark Inference Adapter

The current inference script consumes audio/reference directories and prompt files. The eval agent needs a benchmark adapter that converts `benchmark/benchmark.jsonl` into inference jobs.

Inference must consume only the selected sample list produced by `sample_selection.py`, not the full benchmark by default.

For each benchmark sample:

- `sample_id`: resolved from `benchmark_index.json`, e.g. `000001`
- reference image: determined by `conditioning.reference_mode`, usually the materialized ref for `infer_eval`
- input audio: determined by `conditioning.audio_mode`, usually the materialized audio segment for benchmark audio
- pose condition: determined by `conditioning.pose_mode` and `task.required_conditions`
- text condition: determined by `conditioning.text_mode`
- prompt: `prompt`, kept for reporting unless `conditioning.text_mode=benchmark_prompt`
- target video: `videopath`
- target pose: `posepath`

In `infer_eval`, benchmark normalization should run before inference. The inference adapter should consume `benchmark_materialized/manifest.jsonl`, not raw benchmark media paths, unless normalization is explicitly disabled.

Recommended inference output:

```text
eval_outputs/<run_id>/inference/
  videos/
    000001.mp4
  refs/
    000001.png
  audio_segments/
    000001.wav
  inference_manifest.jsonl
```

Each `inference_manifest.jsonl` row should include:

```json
{
  "key": "000001",
  "checkpoint": ".../model.pt",
  "video": "eval_outputs/<run_id>/inference/videos/000001.mp4",
  "ref": ".../benchmark/img/000001.jpg",
  "prepared_ref": "eval_outputs/<run_id>/inference/refs/000001.png",
  "audio": ".../benchmark/audio/...",
  "audio_segment": "eval_outputs/<run_id>/inference/audio_segments/000001.wav",
  "materialized_ref": "eval_outputs/<run_id>/benchmark_materialized/refs/000001.png",
  "materialized_audio": "eval_outputs/<run_id>/benchmark_materialized/audio_segments/000001.wav",
  "materialized_target_clip": "eval_outputs/<run_id>/benchmark_materialized/target_clips/000001.mp4",
  "prompt": "...",
  "prompt_for_generation": false,
  "prompt_for_reporting": true,
  "text_mode": "fixed_embedding",
  "text_emb_path": "raw_dataset/text_emb/person_speaking_zh_umt5_xxl_bf16.pt",
  "audio_mode": "benchmark_wav",
  "reference_mode": "materialized_ref",
  "seed": 42
}
```

## MVP Metrics

The MVP follows an OmniShow-style offline multimodal metric skeleton:

- Reference consistency: FaceSim
- Audio-visual synchronization: Sync-C / Sync-D
- Video quality: IQA / AES
- Pose accuracy: optional AKD / PCK
- Basic media integrity: sanity checks

Do not implement realtime metrics in this phase.

### Metric Set

| Category | Metric |
| --- | --- |
| Sanity | video/audio sanity check |
| Reference consistency | FaceSim mean/min/std + face_detection_rate |
| Audio-visual sync | Sync-C / Sync-D |
| Temporal sync | sliding-window Sync-C / Sync-D |
| Video quality | IQA / AES |
| Pose accuracy | optional AKD / PCK if target pose and generated pose are available |

## Paired GT Diagnostics

Paired GT diagnostics compare generated video against the materialized target clip. They are useful for debugging but must be marked `diagnostics_only`.

These diagnostics are not primary quality metrics because a valid digital-human generation may differ from the GT video while still being acceptable, especially when generation is stochastic or the task allows motion variation.

Initial paired diagnostics:

| Diagnostic | Inputs | Output Fields | Notes |
| --- | --- | --- | --- |
| `lpips_face` | generated selected face crops, target selected face crops | `lpips_face_mean`, `lpips_face_std`, coverage | Requires face track alignment; diagnostics only. |
| `lpips_mouth` | generated mouth crops, target mouth crops | `lpips_mouth_mean`, `lpips_mouth_std`, coverage | Useful for local mouth appearance drift; diagnostics only. |
| `mouth_landmark_l1` | generated mouth landmarks, target mouth landmarks | `mouth_landmark_l1_mean`, coverage | Requires consistent landmark detector/provenance. |
| `psnr` | generated frames, materialized target frames | `psnr_mean`, optional | Optional coarse reconstruction diagnostic. |
| `ssim` | generated frames, materialized target frames | `ssim_mean`, optional | Optional coarse reconstruction diagnostic. |

Rules:

- Every paired GT diagnostic result must include `"diagnostics_only": true`.
- Paired GT diagnostics must not contribute to `overall_status` unless explicitly listed in `acceptance_criteria`.
- Paired GT diagnostics must not be used as primary quality metrics in `paper_table.csv`.
- They require strict benchmark linkage and a materialized target clip.
- In `video_dir_eval` with `unpaired` linkage, paired GT diagnostics must be skipped.
- LPIPS and landmark diagnostics must record backend-aware metric variants and provenance, just like IQA/AES.

Example:

```json
{
  "metric_variant": "lpips_face_alex_sampled",
  "diagnostics_only": true,
  "lpips_face_mean": 0.18,
  "lpips_face_std": 0.04,
  "coverage": {"valid": 64, "total": 81, "rate": 0.79, "min_required": 0.6},
  "warnings": [],
  "error": null
}
```

## Metric Cost Levels and Presets

The agent must support natural-language requests that imply different evaluation depth:

```text
快速评测一下
完整评测一下
只看口型
论文格式的评测
```

Natural language must resolve to a frozen metric plan in `resolved_config.json` and `metric_plan.json`. The plan should record:

- requested preset;
- resolved preset;
- selected sample count and available sample count;
- enabled metric variants;
- excluded metric variants and reasons;
- max cost level;
- estimated relative cost;
- whether the run is comparable by default.

### Cost Level Enum

Use `metric_cost_level` on each metric variant, not only on metric families.

| Level | Name | Intended Meaning |
| --- | --- | --- |
| `free` | metadata/report-only | Reads existing metadata or metrics; no media model inference. |
| `low` | cheap media probe | CPU media checks, ffprobe, light frame sampling. |
| `medium` | routine model metric | Sampled learned metrics or global AV sync on a small/quick split. |
| `high` | heavy model metric | Dense frame/window metrics, IQA/AES models, DWPose extraction. |
| `very_high` | paper/full diagnostic | Multiple heavy metrics, dense windows, large official split, or all available samples. |

Cost levels are planning hints, not wall-clock guarantees. Actual runtime depends on GPU, decoder, metric implementation, frame count, and selected sample count.

### Metric Variant Registry

Suggested MVP registry:

| Metric Variant | Base Metric | Default Cost | Notes |
| --- | --- | --- | --- |
| `sanity` | sanity | `low` | Always enabled unless explicitly disabled. |
| `facesim_sampled` | FaceSim | `medium` | Uniform sampled frames, e.g. 8-16 frames/sample. |
| `facesim_dense` | FaceSim | `high` | More frames or all frames; useful for paper/report diagnostics. |
| `sync_global` | Sync-C / Sync-D | `medium` | One global score or sparse windows. |
| `sync_sliding` | sliding Sync-C / Sync-D | `high` | Sliding-window curve and artifacts. |
| `iqa_qalign_sampled` | IQA | `medium` | Q-Align backend, sampled frames. |
| `aes_qalign_sampled` | AES | `medium` | Q-Align or Q-Align-derived aesthetic backend, sampled frames. |
| `iqa_qalign_dense` | IQA | `high` | Q-Align backend, dense frame quality timeline. |
| `aes_qalign_dense` | AES | `high` | Q-Align or Q-Align-derived aesthetic backend, dense frame aesthetic timeline. |
| `iqa_<backend>_<sampling>` | IQA | varies | Backend-aware naming for future IQA implementations. |
| `aes_<backend>_<sampling>` | AES | varies | Backend-aware naming for future AES implementations. |
| `pose_from_existing` | AKD / PCK | `medium` | Target and generated pose tracks already exist. |
| `pose_extract_dwpose` | AKD / PCK | `high` | Runs DWPose extraction from generated video. |

Each registry item should define:

```json
{
  "metric_variant": "sync_sliding",
  "base_metric": "sync",
  "metric_cost_level": "high",
  "requires_gpu": true,
  "requires_benchmark_linkage": true,
  "required_inputs": ["generated_video", "sync_audio"],
  "default_frame_or_window_policy": {
    "quick": "global_or_sparse",
    "full": "sliding_window",
    "paper": "sliding_window_with_artifacts"
  },
  "outputs": ["sync_c", "sync_d", "sync_curve"],
  "coverage_gates": {}
}
```

### Backend-Aware Metric Variant Naming

Metric variant names must include the backend when different implementations can produce non-equivalent scores.

This is mandatory for IQA/AES. Do not use ambiguous variants such as:

```text
iqa_sampled
aes_sampled
iqa_aes_sampled
```

Use backend-aware names instead:

```text
iqa_qalign_sampled
aes_qalign_sampled
iqa_qalign_dense
aes_qalign_dense
iqa_custom_aesthetic_v1_sampled
aes_custom_aesthetic_v1_sampled
```

Rules:

- `metric_variant` is the table/aggregate identity, not just a human label.
- `paper_table.csv`, `current_run_table.csv`, `comparison_table.csv`, `metrics.json.aggregate`, and HTML metric tables must use backend-aware variant names for IQA/AES columns.
- IQA/AES results from different backends must never share the same column or aggregate, even if both report a field named `iqa_mean` or `aes_mean`.
- Provenance is still required, but provenance is not enough to prevent table-level mixing; the backend must be visible in the metric variant and table column name.
- Comparisons between different IQA/AES backends are `not_comparable` by default.
- If a user asks for "IQA" or "AES" without specifying backend, the preset must resolve to the project default backend and write the resolved variant, for example `iqa_qalign_sampled`.

### Presets

Presets are templates. Explicit user metric choices override the preset, but the override must still be recorded in `resolved_config.json`.

| Preset | Natural-Language Aliases | Default Metrics | Default Split / Sample Policy | Report Mode |
| --- | --- | --- | --- | --- |
| `smoke` | "冒烟", "先试一下", "跑通流程" | `sanity` | `official_smoke_10` if available, else `min(available, auto_smoke_count)` | basic |
| `quick` | "快速评测", "快评一下" | `sanity`, `facesim_sampled` | `official_smoke_10` or explicit count; otherwise auto quick count | basic + summary |
| `lip_sync` | "只看口型", "只测口型", "只看音画同步" | `sanity`, `sync_global`, optional `sync_sliding` if cost allows | explicit split/count if provided; otherwise quick policy | sync-focused |
| `full` | "完整评测", "全指标评测" | all MVP metrics that pass task/input gating, including `sync_sliding`, backend-aware IQA/AES variants such as `iqa_qalign_sampled` and `aes_qalign_sampled`, optional pose | `official_main`/largest official split if available; else all filtered samples | full |
| `paper` | "论文格式", "出论文表格", "正式评测" | full MVP metrics with paper table/report artifacts; prefer dense variants where budget allows | versioned official split required for comparable claims; else mark as non-official | paper |
| `custom` | explicit metrics, e.g. "只跑 FaceSim 和 Sync" | user-specified variants after gating | user-specified sample selection | requested |

### Flexible Sample Policy

Presets must not assume `benchmark.jsonl` always has 300 samples. Sample selection should be computed from the current benchmark index after filters.

Suggested auto-count policy:

```json
{
  "auto_sample_policy": {
    "smoke": {"min": 2, "target": 10, "max": 10},
    "quick": {"min": 5, "fraction": 0.05, "max": 50},
    "lip_sync": {"min": 5, "fraction": 0.05, "max": 50},
    "full": {"mode": "official_or_all"},
    "paper": {"mode": "official_required_for_comparable_claims"}
  }
}
```

Resolution rules:

- If the user gives an explicit count, use that count after filters and record `selection_source=user_explicit_count`.
- If the user gives an official split, use that split and ignore auto-count.
- If a preset has a preferred official split and it exists, use it.
- If no official split exists, compute auto count from `available_count` after filters.
- If the filtered benchmark has fewer samples than the auto target, run all available samples and record a warning.
- `paper` without a versioned official split may still run, but must set `comparable_by_default=false` and report that results are not official benchmark claims.

### Metric Plan Artifact

Each run should write:

```text
eval_outputs/<run_id>/metric_plan.json
```

Example:

```json
{
  "requested_preset": "quick",
  "resolved_preset": "quick",
  "available_count": 300,
  "selected_count": 10,
  "sample_policy": "official_smoke_10",
  "max_cost_level": "medium",
  "enabled_metric_variants": [
    {"name": "sanity", "metric_cost_level": "low"},
    {"name": "facesim_sampled", "metric_cost_level": "medium"}
  ],
  "excluded_metric_variants": [
    {
      "name": "sync_global",
      "metric_cost_level": "medium",
      "reason": "excluded from default quick; use lip_sync or quick_lipsync for audio-visual sync"
    },
    {
      "name": "sync_sliding",
      "metric_cost_level": "high",
      "reason": "excluded by preset=quick and max_cost_level=medium"
    }
  ],
  "estimated_cost": {
    "units": 45,
    "formula": "sum(metric_variant.base_units * selected_count * sampling_multiplier)"
  }
}
```

`metrics.json` must record the metric plan path and a compact copy of the resolved plan.

## Metric Cache and Invalidation

Some offline metrics will become expensive when sample count, frame sampling density, or model-based diagnostics increase. The eval agent should include a cache layer, but cache reuse must be conservative because benchmark rows, clip windows, reference policy, audio crop policy, or metric implementations may change.

### Cache Scopes

Use two cache scopes:

| Scope | Meaning | Default Reuse Rule |
| --- | --- | --- |
| `benchmark_bound` | Depends on `benchmark.jsonl`, sample identity, selected GT window, materialized target/ref/audio/pose, or benchmark linkage. | Invalidate when benchmark fingerprint, benchmark index, sample row hash, or materialization config changes. |
| `input_artifact_bound` | Depends only on the actual generated video/audio/image content and metric implementation. | Reuse only when content hash and metric provenance match. Cross-benchmark reuse is disabled by default. |

MVP cache policy:

- MVP-1 default is `cache.enabled=false`.
- `cache_manifest.json` schema should still be written so later milestones do not change output shape.
- Single-run in-memory or local temp reuse is allowed for duplicated work inside the same process, but it must not be treated as persistent cross-run cache.
- Persistent cross-run cache is deferred to MVP-4.

The default should be safe:

```json
{
  "cache": {
    "enabled": false,
    "lookup_policy": "single_run_reuse_only",
    "cache_root": "eval_cache/v1",
    "allow_cross_run_cache": false,
    "allow_cross_benchmark_feature_cache": false,
    "require_metric_provenance_match": true,
    "require_benchmark_fingerprint_for_benchmark_bound_cache": true
  }
}
```

If a cache key cannot be fully constructed, treat it as a cache miss and recompute.

### What Should Be Cached

Recommended cache targets:

| Component | Cache? | Scope | Why | Key Must Include |
| --- | --- | --- | --- | --- |
| Benchmark index and row fingerprints | Yes | `benchmark_bound` | Avoid repeated ID/index validation and detect benchmark changes. | `benchmark_jsonl_sha256`, sample-id policy, row hash. |
| Benchmark materialization | Yes | `benchmark_bound` | Video clipping, FPS conversion, resize/crop, audio segmenting, pose segmenting are I/O heavy. | source media content hashes, sample ID, row hash, generation profile, GT clip policy/start/end, target size/FPS/frame count, audio sample rate, ref policy, pose policy. |
| Media probe / sanity metadata | Yes | `input_artifact_bound` or `benchmark_bound` | Cheap but shared by many metrics and reports. | media content hash, ffprobe/decoder version, thresholds. |
| Decoded frame/keyframe extraction | Yes | `input_artifact_bound` | Shared by FaceSim, IQA/AES, artifacts, frozen-frame checks. | video hash, frame indices/timestamps, resize/crop/color policy, decoder version. |
| Audio extraction/resampling | Yes | `input_artifact_bound` or `benchmark_bound` | Shared by Sync and sanity. | audio/video hash, audio source policy, crop policy, start/end, sample rate, channels. |
| Benchmark target-audio sync validation | Yes | `benchmark_bound` | Prevent repeated validation of target clip versus materialized audio and detect unusable benchmark pairs. | target clip hash, audio segment hash, benchmark row hash, GT clip policy/window, sync model/provenance, window/stride, audio crop policy. |
| Face detection/tracks | Yes | `input_artifact_bound` | FaceSim and Sync must share the same selected face track. | video/ref image hash, frame policy, detector model/provenance, face track policy, reference source/hash, tracking parameters. |
| Face crops from selected track | Yes | `input_artifact_bound` | FaceSim and Sync both need consistent face/mouth crops. | selected face track hash, crop/alignment policy, output size, interpolation/missing-frame policy. |
| Face embeddings | Yes | `input_artifact_bound` | FaceSim can be slow across many frames. | crop hash, embedding model hash, preprocessing, dtype/device-independent settings, selected face track ID. |
| Sync model features/window scores | Yes | `input_artifact_bound` plus audio policy | Sliding-window Sync is expensive. | video hash, selected face track hash, face-crop policy, audio source/crop policy, sync model hash, window/stride/offset search, sample rate. |
| IQA/AES frame scores/features | Yes | `input_artifact_bound` | Learned quality/aesthetic metrics can be heavy. | metric variant including backend, video/frame hash, model hash, resize/crop, frame sampling policy, score scale. |
| DWPose extraction | Yes | `input_artifact_bound` | Pose extraction is high cost. | video/target clip hash, DWPose model/provenance, frame policy, output coordinate space. |
| AKD/PCK final score | Yes, but separately from DWPose extraction | `benchmark_bound` | Final score depends on target pose, generated pose, temporal alignment, normalization, and threshold. | generated pose hash, target pose hash, benchmark/materialized clip hash, keypoint set, normalization mode, threshold, visibility policy. |
| HTML/report rendering | Usually no | report-only from `metrics.json` | Regenerate from canonical metrics; no need for separate metric cache. | `metrics_json_sha256`, report template version if cached. |
| Model inference outputs | No metric cache | use `manifest_eval` | Generated videos depend on checkpoint/seed/inference config; treat as inference artifacts, not metric cache. | Use inference manifest provenance instead. |

Highest priority caches for MVP:

1. Benchmark materialization.
2. Decoded frames/keyframes.
3. Face detection/tracks, selected track artifacts, and FaceSim embeddings.
4. Sync audio features, selected-track mouth crops, and sliding-window scores.
5. IQA/AES frame scores.
6. DWPose extraction and AKD/PCK final scores.

### Cache Key Requirements

Every cache key must include:

- cache schema version;
- cache scope;
- metric variant or component name;
- input artifact content hash, not only path or basename;
- benchmark fingerprint for `benchmark_bound` entries;
- sample ID and benchmark row hash when sample identity matters;
- materialization config fingerprint when GT clipping, resizing, FPS, audio, or pose segmentation matters;
- generation profile fingerprint when output size, FPS, frame count, duration, or resize policy matters;
- metric provenance fingerprint for metric/model-dependent entries;
- metric parameters that affect outputs, such as frame sampling, Sync window/stride, PCK normalization, or visibility threshold.

Example cache key object:

```json
{
  "cache_schema_version": 1,
  "scope": "benchmark_bound",
  "component": "akd_pck_score",
  "sample_id": "000001",
  "benchmark_jsonl_sha256": "sha256",
  "benchmark_row_sha256": "sha256",
  "materialization_config_sha256": "sha256",
  "generated_pose_sha256": "sha256",
  "target_pose_sha256": "sha256",
  "metric_variant": "dwpose_body18_norm_img_0p05",
  "metric_provenance_fingerprint": "sha256",
  "metric_params_sha256": "sha256"
}
```

### Benchmark Change Invalidation

If `benchmark.jsonl` changes, all `benchmark_bound` cache entries should be treated as invalid by default.

This includes cases where:

- a row's `videopath`, `wav_path`, `posepath`, `imgpath`, `prompt`, or metadata changes;
- sample ID generation policy changes;
- sample IDs collide or are remapped;
- official split file/hash changes;
- generation profile changes, including width, height, FPS, frame count, duration, or resize policy;
- GT clip policy, frame count, FPS, target size, or materialization policy changes;
- ref image source policy changes, such as `benchmark_img` versus `target_first_frame`;
- Sync audio source or crop policy changes;
- AKD/PCK normalization mode, threshold, keypoint set, or visibility policy changes.

`input_artifact_bound` feature caches can theoretically survive a benchmark change if the underlying video/audio/image content hash and metric provenance are identical. However, default `allow_cross_benchmark_feature_cache=false` should prevent silent reuse across benchmark versions. If enabled later, the report must explicitly record `cross_benchmark_cache_hit=true`.

### Cache Manifest

Each run should write:

```text
eval_outputs/<run_id>/cache_manifest.json
```

It should include:

```json
{
  "cache_enabled": false,
  "cache_root": "eval_cache/v1",
  "lookup_policy": "single_run_reuse_only",
  "benchmark_jsonl_sha256": "sha256",
  "benchmark_index_sha256": "sha256",
  "events": [
    {
      "sample_id": "000001",
      "component": "facesim_embedding",
      "scope": "input_artifact_bound",
      "cache_status": "hit|miss|stale|disabled|write",
      "cache_key_sha256": "sha256",
      "reason": "metric_provenance_match"
    }
  ],
  "summary": {
    "hits": 0,
    "misses": 0,
    "stale": 0,
    "writes": 0
  }
}
```

Each per-sample metric result should include a compact cache audit:

```json
{
  "cache": {
    "status": "hit|partial_hit|miss|stale|disabled",
    "components": [
      {"name": "face_detection", "status": "hit"},
      {"name": "selected_face_track", "status": "hit"},
      {"name": "selected_track_crops", "status": "hit"},
      {"name": "face_embedding", "status": "hit"},
      {"name": "facesim_score", "status": "miss"}
    ]
  }
}
```

Cached metric values must still include status, coverage, warnings, errors, and metric provenance. A cached value is not valid unless its cache key proves that all dependencies match.

### Non-MVP Items

- Text Alignment / TA
- NexusScore
- VQ / MQ
- TTFF
- latency
- deadline miss
- streaming trace
- chunk boundary

## Metric Inputs and Outputs

Every metric result must include:

```json
{
  "status": "ok|degraded|failed|skipped",
  "provenance": {
    "metric_name": "facesim",
    "implementation": "insightface_arcface",
    "implementation_version": "unknown",
    "model_name": "buffalo_l",
    "model_path": "/abs/path/model.onnx",
    "model_sha256": "sha256-or-null",
    "preprocess": {},
    "sampling": {},
    "runtime": {}
  },
  "coverage": {
    "valid": 0,
    "total": 0,
    "rate": 0.0,
    "min_required": 0.8
  },
  "warnings": [],
  "error": null
}
```

Status semantics:

| Status | Meaning | Score Fields | Report Handling |
| --- | --- | --- | --- |
| `ok` | Metric completed and coverage/quality gates passed. | Numeric values allowed. | Included in primary aggregates. |
| `degraded` | Metric completed, but coverage or diagnostic gates are below threshold. | Numeric values allowed, with warnings. | Included only in degraded-aware aggregates or flagged separately. |
| `failed` | Metric could not produce a valid score due to an error. | Numeric score fields must be `null`. | Excluded from aggregates. |
| `skipped` | Metric was intentionally not applicable under current config. | Numeric score fields must be `null`. | Excluded from aggregates. |

If a metric fails, all numeric score fields must be `null`, and `error.message` must explain the failure. Never fabricate metric values.

If a metric is degraded, numeric values may be kept, but warnings must explain why the result is not fully reliable.

Example:

```json
{
  "status": "degraded",
  "facesim_mean": 0.72,
  "face_detection_rate": 0.247,
  "coverage": {
    "valid": 20,
    "total": 81,
    "rate": 0.247,
    "min_required": 0.8
  },
  "warnings": [
    "face_detection_rate below min_coverage=0.8"
  ],
  "error": null
}
```

## Metric Coverage and Degraded Policy

Each metric must define coverage gates and diagnostic warning gates in `resolved_config.json`.

Suggested default gates:

```json
{
  "metric_quality_gates": {
    "sanity": {
      "max_frozen_frame_rate": 0.1,
      "max_black_frame_rate": 0.05
    },
    "facesim": {
      "min_coverage": 0.8,
      "min_valid_frames": 8
    },
    "sync": {
      "min_window_coverage": 0.75,
      "min_valid_windows": 4
    },
    "sliding_sync": {
      "min_window_coverage": 0.75,
      "min_valid_windows": 4
    },
    "iqa_aes": {
      "min_frame_coverage": 0.9,
      "max_frozen_frame_rate": 0.1
    },
    "pose": {
      "min_body_coverage": 0.6,
      "min_hand_coverage": 0.3,
      "allow_half_body": true
    }
  }
}
```

Default degraded conditions:

| Metric | Degraded When |
| --- | --- |
| Sanity | Video/audio readable but black/frozen frame rate exceeds warning threshold. |
| FaceSim | Face detection coverage below `min_coverage`, valid frames below `min_valid_frames`, or selected face track switches/fragments. |
| Sync-C / Sync-D | Valid sync windows below `min_window_coverage` or `min_valid_windows`, or Sync mouth crops use a degraded selected face track. |
| Sliding Sync | Too few valid windows, large fraction of windows fail face/audio preprocessing, or selected face track switches inside windows. |
| IQA / AES | Score can be computed, but sanity reports high frozen/black-frame rate or frame coverage is low. |
| AKD / PCK | Body keypoints partially available, but hands/face/key body parts are missing; half-body samples should be degraded rather than failed when configured with `allow_half_body=true`. |

Concrete examples:

- FaceSim detects faces in only `20/81` frames: `status=degraded`, score kept, warning recorded.
- Multi-face tracking switches once across the sampled frames: FaceSim and Sync remain numeric if possible, but `status=degraded` with `face_track_switch_count=1`.
- Sync has only `2/8` valid windows: `status=degraded` if score can be computed; `failed` only if no valid sync score exists.
- Pose on half-body samples has missing hand keypoints: `status=degraded` when body pose is usable; `failed` only if required body keypoints are absent.
- IQA/AES can run but sanity found `30%` frozen frames: IQA/AES may be `degraded` with a frozen-frame warning.

Aggregate reporting must expose:

- `ok_count`;
- `degraded_count`;
- `failed_count`;
- `skipped_count`;
- aggregate over `ok_only`;
- aggregate over `ok_plus_degraded`, clearly labeled.

## Metric Provenance

Global dependency versions are necessary but not sufficient. Every metric must record metric-level provenance so scores remain interpretable when implementations, model weights, preprocessing, or sampling change.

Each metric result, and each aggregate metric block, must include a `provenance` object.

Required fields:

```json
{
  "metric_name": "facesim",
  "metric_version": "1",
  "implementation": "insightface_arcface",
  "implementation_source": "pip|git|local|external_binary",
  "implementation_version": "0.7.3",
  "implementation_commit": "git-sha-or-null",
  "model_name": "buffalo_l",
  "model_version": "det_10g+arcface_r100",
  "model_path": "/abs/path/model.onnx",
  "model_sha256": "sha256-or-null",
  "device": "cuda:0",
  "dtype": "fp32",
  "preprocess": {
    "input_source": "materialized_ref|generated_video|materialized_audio",
    "resize": "112x112",
    "crop": "face_aligned",
    "color_space": "RGB",
    "normalization": "arcface",
    "audio_sample_rate": null
  },
  "sampling": {
    "frame_sample_policy": "uniform",
    "num_frames_requested": 16,
    "window_sec": null,
    "stride_sec": null
  },
  "thresholds": {},
  "failure_policy": "null_with_error",
  "notes": ""
}
```

If a metric has no learned model, use:

```json
{
  "model_name": null,
  "model_path": null,
  "model_sha256": null
}
```

For external tools or binaries, record:

- executable path;
- version command output if available;
- command-line arguments;
- model/checkpoint files and hashes if used.

### Metric-Specific Provenance Requirements

| Metric | Must Record |
| --- | --- |
| Sanity | ffprobe/decoder backend, black-frame threshold, frozen-frame threshold, audio probe method. |
| FaceSim | face detector model, face embedding model, alignment method, selected reference source, frame sampling policy, similarity function, face track policy, selected face track ID, track selection method, fallback method, track switch handling. |
| Sync-C / Sync-D | sync model name/path/hash, face crop policy, selected face track ID, face track policy, audio source policy, resolved audio source, audio crop policy, audio sample rate, window size, stride, offset search range, score definitions. |
| Sliding Sync | all Sync-C/D provenance plus window aggregation method and whether face track switches occur inside windows. |
| IQA | IQA model name/path/hash, image resize/crop, frame sampling policy, score scale. |
| AES | aesthetic model name/path/hash, preprocessing, frame aggregation. |
| AKD / PCK | pose detector/extractor, DWPose keypoint set, coordinate space, normalization mode/scale, PCK threshold semantics, visibility threshold, temporal alignment policy. |

### Aggregate Provenance

`metrics.json.aggregate` must not merge scores from different metric provenance unless explicitly requested.

If two samples were evaluated with different FaceSim implementations or different model hashes, aggregate reporting must either:

- split aggregates by provenance fingerprint; or
- fail aggregation for that metric with a clear error.

Each provenance object should have a stable `provenance_fingerprint`, computed from implementation, model hashes, preprocessing, sampling, thresholds, and metric parameters.

### Sanity

Inputs:

- generated video
- benchmark audio
- benchmark target video, if needed for reference duration/shape checks

Outputs:

```json
{
  "video_ok": true,
  "audio_ok": true,
  "width": 480,
  "height": 832,
  "fps": 20,
  "num_frames": 81,
  "duration": 4.05,
  "audio_duration": 4.05,
  "has_audio": true,
  "black_frame_rate": 0.0,
  "frozen_frame_rate": 0.0,
  "coverage": {"valid": 81, "total": 81, "rate": 1.0, "min_required": 1.0},
  "warnings": [],
  "error": null
}
```

Current MVP backend variant:

- `metric_variant`: `sanity_ffmpeg_probe_sampled_frames_v1`
- Video metadata backend: `ffprobe`.
- Frame extraction backend: `ffmpeg` sampled frames under `artifacts/<sample_id>/sanity/frames/`.
- `cv2.VideoCapture` must not be used for mp4 video reads because the current environment can lack OpenCV FFmpeg support.
- Extracted images may be read with PIL or `cv2.imread`.
- Default sampled-frame policy: uniform frame-index sampling, `frame_sample_count=80`.
- Default near-exact frozen threshold: mean grayscale frame diff `<= 0.1`.
- Default black-frame threshold: mean grayscale luma `<= 5.0`.
- Default degraded thresholds: `max_frozen_frame_rate=0.3`, `max_black_frame_rate=0.3`.
- If ffprobe cannot read the generated video, sanity is `failed`.
- If ffmpeg frame extraction fails, sanity is `failed`.
- If black/frozen rates exceed thresholds, sanity is `degraded`.
- In `eval_profile=dev_native_v1`, fps/resolution/duration profile differences are warning-only unless the run explicitly requests a paper profile such as `paper_model_native_v1`.
- Each sample should save `video_probe.json`, frame extraction command/stdout/stderr, `sampled_frame_paths.json`, `sampled_frames_contact_sheet.jpg`, and `sanity_error.json` when failed.

### FaceSim

Inputs:

- reference image: `imgpath` or `prepared_ref`
- generated video frames

Outputs:

```json
{
  "facesim_mean": 0.82,
  "facesim_min": 0.74,
  "facesim_std": 0.03,
  "face_detection_rate": 0.95,
  "selected_face_track_id": "track_000",
  "face_track_status": "ok",
  "face_track_selection": "highest_ref_similarity_track",
  "face_track_switch_count": 0,
  "allow_per_frame_best_face": false,
  "valid_frames": 77,
  "total_frames": 81,
  "coverage": {"valid": 77, "total": 81, "rate": 0.95, "min_required": 0.8},
  "warnings": [],
  "error": null
}
```

Current MVP implementation:

- Current variant: `facesim_sampled_insightface_antelopev2_cpu_v1`
- Backend: local InsightFace antelopev2 CPU, strict `auto_generation_ref`, ffmpeg sampled-frame extraction, `largest_center_face`, `allow_per_frame_best_face=false`.
- Status document: `docs/metric_status/facesim_sampled_insightface_antelopev2_cpu_v1.md`
- Stable face tracking remains future work; current FaceSim records `face_track_status=not_implemented_mvp`.

### Sync-C / Sync-D

Inputs:

- generated video
- generated audio track, generated `audio_segment`, or benchmark `wav_path`

Outputs:

```json
{
  "sync_c": 7.2,
  "sync_d": 6.4,
  "best_offset_ms": -40,
  "selected_face_track_id": "track_000",
  "face_track_status": "ok",
  "face_track_switch_count": 0,
  "mouth_crop_source": "selected_face_track",
  "sync_audio": {
    "source_policy": "generated_audio_track_or_materialized_audio",
    "resolved_source": "generated_audio_track",
    "source_path": "eval_outputs/<run_id>/inference/videos/000001.mp4",
    "extracted_audio_path": "artifacts/000001/sync/audio.wav",
    "crop_policy": "align_to_generated_video_duration",
    "source_start_sec": 0.0,
    "source_end_sec": 4.05,
    "effective_duration_sec": 4.05,
    "sample_rate": 16000,
    "channels": 1,
    "duration_mismatch_sec": 0.0
  },
  "valid_windows": 1,
  "total_windows": 1,
  "coverage": {"valid": 1, "total": 1, "rate": 1.0, "min_required": 0.75},
  "warnings": [],
  "error": null
}
```

### Sliding-Window Sync-C / Sync-D

Inputs:

- generated video
- generated audio track, generated `audio_segment`, or benchmark `wav_path`
- window size and stride

Outputs:

```json
{
  "sync_c_mean": 6.9,
  "sync_c_min": 5.8,
  "sync_d_mean": 6.6,
  "sync_d_max": 8.1,
  "selected_face_track_id": "track_000",
  "face_track_status": "ok",
  "face_track_switch_count": 0,
  "mouth_crop_source": "selected_face_track",
  "sync_audio": {
    "source_policy": "generated_audio_track_or_materialized_audio",
    "resolved_source": "materialized_audio",
    "source_path": "benchmark_materialized/audio_segments/000001.wav",
    "crop_policy": "none",
    "effective_duration_sec": 4.05,
    "sample_rate": 16000,
    "channels": 1
  },
  "window_scores": [],
  "coverage": {"valid": 8, "total": 8, "rate": 1.0, "min_required": 0.75},
  "warnings": [],
  "error": null
}
```

### IQA / AES

Inputs:

- generated video frames

Outputs:

```json
{
  "metric_variants": ["iqa_qalign_sampled", "aes_qalign_sampled"],
  "iqa_backend": "qalign",
  "iqa_mean": 0.71,
  "iqa_std": 0.05,
  "iqa_metric_variant": "iqa_qalign_sampled",
  "aes_backend": "qalign",
  "aes_mean": 5.8,
  "aes_std": 0.4,
  "aes_metric_variant": "aes_qalign_sampled",
  "valid_frames": 81,
  "total_frames": 81,
  "coverage": {"valid": 81, "total": 81, "rate": 1.0, "min_required": 0.9},
  "warnings": [],
  "error": null
}
```

Current MVP implementation:

- Current full variant: `iqa_aes_vbench_local_sampled_cpu_v1`
- Partial variants: `iqa_vbench_local_sampled_cpu_v1`, `aes_vbench_local_sampled_cpu_v1`
- Backend: local VBench MUSIQ IQA plus LAION aesthetic CLIP head, strict local model preflight, `offline_mode=true`, `allow_download=false`.
- Model root: `/mnt/data/nlp/user/qiaoqian/newproject/eval_agent_pretrained`
- Frame extraction: ffmpeg uniform sampled frames, default `8` frames/sample.
- Status document: `docs/metric_status/iqa_aes_vbench_local_sampled_cpu_v1.md`
- Status: `dev_full_admitted`, `paper_ready=false`, default `eval_profile=dev_native_v1`.
- `quick` remains `sanity + facesim_sampled`; `full` includes `iqa_aes_sampled`.
- `paper_table.csv` and aggregate summaries use backend-aware variant names and must not mix different IQA/AES provenance fingerprints.
- Current dev profile is not directly comparable to the 5s/720p HOIVG-Bench paper protocol; paper readiness requires a fixed official split and protocol-level normalization/sampling.

### DWPose Sampled Extractor

Current diagnostic extractor variant:

- `metric_variant`: `dwpose_sampled_extractor_cpu_v1`
- Scope: sampled DWPose extraction readiness only; not AKD/PCK or pose accuracy.
- Backend: read-only TalkVid low-level `onnxdet.py` + `onnxpose.py` adapter with explicit local ONNX paths.
- Default provider: `CPUExecutionProvider`; CUDA request must fail if `CUDAExecutionProvider` is unavailable.
- Frame extraction: ffmpeg uniform sampled frames, default `8` frames/sample.
- Outputs include body extractor diagnostics, per-frame keypoints in pixel and normalized coordinates, overlays, coverage, and provenance.
- `akd_pck_ready=false` until target pose extraction, frame alignment, and body18 mapping validation are implemented.
- Status document: `docs/metric_status/dwpose_sampled_extractor_cpu_v1.md`
- Not included in `quick` or `full`; request explicitly with `--metrics dwpose_sampled`.

### Optional AKD / PCK

Inputs:

- target pose: `posepath`
- generated pose or generated pose video

Current body18 AKD/PCK design status:

- Formal design document: `docs/metric_status/akd_pck_body18_design.md`
- Formal target binding policy: `docs/metric_status/akd_pck_target_binding_policy.md`
- Benchmark key policy: `docs/metric_status/benchmark_key_policy.md`
- Body18 mapping validation documents: `docs/metric_status/dwpose_body18_mapping_validation.md`, `docs/metric_status/dwpose_body18_mapping_final_validation.md`
- Planned combined variant: `akd_pck_body18_norm_img_0p05_inbounds_v1`
- Planned AKD variant: `akd_body18_norm_img_inbounds_v1`
- Planned PCK variant: `pck_body18_norm_img_0p05_inbounds_v1`
- Recommended target source for MVP: `materialized_target_clip_dwpose`
- Recommended frame alignment for MVP: `sampled_index_match`
- Primary coordinate policy: `normalized_xy`, `normalization_mode=normalized_image`
- Out-of-frame policy: exclude OOB keypoints from primary metrics, clip only for visualization, and report OOB rates.
- Preflight status: `akd_pck_preflight_v1` passed target-side DWPose extraction, generated-side DWPose reuse, sampled-index alignment, identical-video debug sanity, wrong-alignment debug sanity, and OOB policy dry-run.
- Binding hardening status: `akd_pck_binding_validation_v1` found `0/3` validated target bindings and `3/3` debug-only numeric bindings. Formal target binding remains blocked because current manifests do not carry an explicit benchmark key chain.
- Strict linkage hardening status: `binding_hardening_v1` found `benchmark/benchmark.jsonl` has `300/300` rows missing explicit `key/sample_id`, no `benchmark_materialized/manifest.jsonl` exists, and strict binding for the first 3 current generated samples is `0/3` validated and `3/3` blocked. Numeric preflight targets are available only as debug artifacts and are rejected by the strict resolver.
- Benchmark key materialization status: `benchmark_key_materialization_v1` generated derived `benchmark_key` values for `300/300` benchmark rows using `zero_padded_row_index_6 + short_row_hash_8`, with `collision_count=0`. It selected 3 samples, materialized target/ref/audio for the current `480x832 / 20fps / 81f` profile, and strict binding over the key-preserving future manifest is `3/3` validated.
- Key-preserving infer smoke status: `key_preserving_infer_actual_smoke_v1` generated 1 actual sample with `benchmark_key`, `benchmark_row_hash`, materialized ref/audio/target clip paths, and generation profile fields preserved in the inference manifest. Strict binding is `1/1` validated with no numeric or basename fallback, generated video exists, `quick` eval passed (`sanity=ok`, `facesim_sampled=ok`), and explicit `dwpose_sampled` check passed.
- Body18 final validation status: `body18_final_validation_v1` concludes `body18_mapping_status=validated` with OpenPose compatibility `compatible_with_documented_adapter_transform`. Labeled overlay review and consistency checks are stored under `eval_outputs/body18_final_validation_v1/`.
- Current readiness: formal AKD/PCK implementation can proceed in the next round as `akd_pck_body18_norm_img_0p05_inbounds_v1`, provided strict binding, sampled-index alignment, visibility threshold, and OOB exclusion policies are enforced. Do not add AKD/PCK to `quick`, `quick_lipsync`, or `full` until admission passes.

### AKD / PCK Normalization

AKD/PCK must define what "5%" means. Do not report `PCK@5%` without a normalization mode.

OmniShow-style AKD/PCK is based on DWPose. For compatibility, the recommended primary metric is:

```text
DWPose body PCK@0.05 with normalized-image coordinates
```

Default configuration:

```json
{
  "pose_metric": {
    "extractor": "DWPose",
    "keypoint_set": "body_18",
    "coordinate_space": "normalized_xy",
    "normalization_mode": "normalized_image",
    "distance_metric": "euclidean",
    "akd_units": "normalized_image",
    "pck_threshold": 0.05,
    "pck_threshold_units": "normalized_image",
    "visibility_score_min": 0.3,
    "min_valid_keypoints_per_frame": 6,
    "min_valid_frame_rate": 0.5,
    "min_matched_keypoint_rate": 0.3,
    "missing_keypoint_policy": "exclude_unmatched",
    "out_of_frame_policy": "exclude_from_primary_metric",
    "frame_alignment": "sampled_index_match",
    "target_binding_policy": "explicit_key",
    "target_pose_source": "materialized_target_clip_dwpose",
    "generated_pose_source": "generated_video_dwpose",
    "require_binding_validated": true,
    "allow_debug_binding": false
  }
}
```

Interpretation:

- Convert DWPose coordinates to normalized image coordinates: `x_norm=x/W`, `y_norm=y/H`.
- Compute per-keypoint Euclidean distance in normalized coordinates.
- AKD is the mean normalized Euclidean distance over valid matched keypoints.
- PCK@0.05 is the fraction of valid matched keypoints whose normalized Euclidean distance is `<= 0.05`.

Supported normalization modes:

| Mode | Threshold Meaning | Notes |
| --- | --- | --- |
| `normalized_image` | `0.05` in normalized `(x/W, y/H)` coordinate space. | Recommended primary mode for OmniShow-style comparability and current DWPose normalized pose files. |
| `image_diagonal` | Pixel distance `<= 0.05 * sqrt(W^2 + H^2)`. | Easy to interpret but looser on tall videos. |
| `person_bbox_height` | Pixel distance `<= 0.05 * person_bbox_height`. | Scale-aware; useful secondary metric for full/half-body variation. |
| `person_bbox_diagonal` | Pixel distance `<= 0.05 * person_bbox_diagonal`. | Scale-aware and less height-biased. |
| `torso_size` | Pixel distance `<= 0.05 * torso_size`. | Common in pose literature but brittle if torso keypoints are missing. |
| `shoulder_width` | Pixel distance `<= 0.05 * shoulder_width`. | Useful for upper-body but unreliable for side/occluded poses. |

Recommendation:

- Use `normalized_image` as the primary `pck`/`akd` to match DWPose normalized coordinate outputs and keep OmniShow-style comparability.
- Name the primary metric explicitly, for example `pck_norm_img_0p05` and `akd_norm_img`, instead of a bare `pck` when tables may contain multiple pose variants.
- Optionally report secondary metrics such as `pck_bbox_height_0p05` or `pck_bbox_diag_0p05` for scale-aware analysis.
- Avoid `torso_size` and `shoulder_width` as primary normalization modes for this benchmark because half-body, side-view, occlusion, or missing upper-body keypoints can make the scale undefined.
- Do not mix different normalization modes in the same aggregate.
- Store the normalization mode in `resolved_config.json`, metric provenance, every AKD/PCK result, and comparison tables.
- Comparisons between different AKD/PCK normalization modes must be marked `not_comparable` by default.

Visibility and missing-keypoint rules:

- Use only keypoints visible in both target and generated pose above `visibility_score_min`.
- Report `valid_keypoints`, `total_keypoints`, `valid_frames`, and coverage.
- If body keypoint coverage is below the configured threshold but nonzero, mark `status=degraded`.
- If no valid body keypoints exist, mark `status=failed` with `null` scores.
- Hand/face keypoints should be reported separately or as optional sub-metrics; missing hands in half-body clips should not invalidate body PCK.

Outputs:

```json
{
  "metric_variant": "dwpose_body18_norm_img_0p05",
  "akd": 0.043,
  "pck": 0.78,
  "akd_name": "akd_norm_img",
  "pck_name": "pck_norm_img_0p05",
  "pck_threshold": 0.05,
  "pck_threshold_units": "normalized_image",
  "normalization_mode": "normalized_image",
  "keypoint_set": "body_18",
  "valid_keypoints": 1020,
  "total_keypoints": 1458,
  "valid_frames": 70,
  "total_frames": 81,
  "coverage": {"valid": 70, "total": 81, "rate": 0.8642, "min_required": 0.6},
  "warnings": [],
  "error": null
}
```

If generated pose is unavailable, mark the metric as `skipped` with score fields set to `null`.

## Output Files

Each run should write:

```text
eval_outputs/<run_id>/
  resolved_config.json
  metric_plan.json
  cache_manifest.json
  benchmark_index.json
  selected_samples.jsonl
  validation_report.json

  benchmark_materialized/
    manifest.jsonl
    refs/
    target_clips/
    audio_segments/

  inference/
    videos/
    refs/
    audio_segments/
    inference_manifest.jsonl

  artifacts/
    <sample_id>/
      keyframes.jpg
      face_crops.jpg
      face_track_overlay.mp4
      selected_face_track.jsonl
      face_track_switches.json
      sync_curve.png
      quality_timeline.png
      pose_overlay.mp4
      error.json

  metrics.json
  paper_table.csv
  current_run_table.csv
  comparison_table.csv
  comparison.json
  sample_comparison.jsonl
  html_report.html
```

The output root should also maintain:

```text
eval_outputs/latest
eval_outputs/latest_success
```

These are symlinks or JSON pointer files as defined in the run ID rules.

Every sample must have an artifacts directory. If a metric fails, write a basic media probe and `error.json` instead of silently omitting artifacts.

## metrics.json

`metrics.json` is the canonical machine-readable result.

It should contain:

- run metadata
- git commit
- resolved config path and sha256
- checkpoint path, step, and config
- benchmark path
- task profile name/version and condition contract
- generation/eval profile name, source, width, height, FPS, frame count, duration, and generated-output mismatch policy
- sample ID generation policy and benchmark index path/hash
- official eval split name/version/path/hash
- benchmark normalization config and materialized manifest path
- benchmark target-audio sync validation summary
- sample selection config
- requested count, available count, and selected count
- selected sample IDs
- metric preset, max cost level, metric plan path, enabled metric variants, and excluded metric variants with reasons
- cache config, cache manifest path, and cache hit/miss/stale summary
- conditioning config
- actual per-sample conditioning inputs
- validation report path and validation summary
- inference manifest path
- metric config
- face track policy, selected face track summary, and face track switch counts
- dependency versions
- metric provenance registry and provenance fingerprints
- per-sample metric results
- paired GT diagnostics summary, with every such metric marked `diagnostics_only`
- per-sample artifact paths
- aggregate statistics
- status counts for `ok`, `degraded`, `failed`, and `skipped`
- separate `ok_only` and `ok_plus_degraded` aggregates
- failure/degradation taxonomy summary
- natural-language diagnostic summary
- acceptance criteria config and result, including `overall_status`, `failed_criteria`, and `warned_criteria`
- comparison config, comparison status, and comparison artifact paths when enabled
- failure summary
- warning summary
- coverage summary

Do not aggregate failed/null values as zero.

## Acceptance Criteria

Acceptance criteria convert metric results into an operational run status. They are not a replacement for detailed metrics; they are a concise gate for automation.

`metrics.json` must include:

```json
{
  "acceptance": {
    "overall_status": "pass|warn|fail",
    "failed_criteria": [],
    "warned_criteria": [],
    "criteria_results": [
      {
        "name": "sanity_pass_rate",
        "type": "absolute",
        "metric": "sanity_pass_rate",
        "value": 0.98,
        "warn_below": 0.95,
        "fail_below": 0.9,
        "status": "pass"
      },
      {
        "name": "facesim_not_regressed",
        "type": "relative_to_baseline",
        "metric": "facesim_mean",
        "baseline_value": 0.76,
        "current_value": 0.74,
        "delta": -0.02,
        "warn_delta_below": -0.02,
        "fail_delta_below": -0.05,
        "status": "warn"
      }
    ]
  }
}
```

Status rules:

- `fail` if any criterion has `status=fail`.
- `warn` if no criteria fail but one or more criteria have `status=warn`.
- `pass` only if all enabled criteria pass.

Supported criterion types:

| Type | Meaning | Example |
| --- | --- | --- |
| `absolute` | Compare current aggregate to fixed thresholds. | `sanity_pass_rate >= 0.95`, `face_detection_rate >= 0.8`. |
| `relative_to_baseline` | Compare current aggregate to a baseline run. | `facesim_mean` must not drop by more than `0.02`. |

Rules:

- Criteria must declare whether they use `ok_only` or `ok_plus_degraded` aggregates.
- Criteria that depend on a missing metric should be `warn` or `fail` according to `missing_metric_policy`.
- `diagnostics_only` metrics must not affect `overall_status` unless explicitly included in `acceptance_criteria.rules`.
- `overall_status` controls `latest_success`: only `pass` and `warn` may update `latest_success`.
- Acceptance results must be shown in `html_report.html`.

## Failure and Degradation Taxonomy

Reports should not only show metric tables. They should group failures and degraded results into actionable categories, then generate a concise natural-language diagnosis grounded in counts and sample evidence.

Each metric warning/error should map to a taxonomy item.

Suggested taxonomy:

| Category | Trigger Sources | Example Message |
| --- | --- | --- |
| `media_integrity` | sanity failures, unreadable video/audio, black frames, frozen frames | "Video integrity issues affect 6/20 samples; 4 have frozen-frame rate above 10%." |
| `identity_consistency` | FaceSim low score, low face detection coverage, reference mismatch | "Identity consistency is degraded in 8/20 samples, mostly due to low face detection coverage." |
| `face_tracking` | multiple tracks, selected track switches, track too short, inconsistent FaceSim/Sync face source | "Face tracking is unstable in 3/50 samples; FaceSim and Sync are degraded because the selected track switches." |
| `av_sync` | Sync-C/D failed/degraded, low valid sync windows, audio missing | "Audio-visual sync is unreliable in 5/20 samples; 3 have fewer than 4 valid sync windows." |
| `visual_quality` | IQA/AES low/degraded, frozen/black frame warnings | "Visual quality degradation is associated with frozen frames in 6 samples." |
| `pose_accuracy` | AKD/PCK failed/degraded, missing body/hand keypoints | "Pose evaluation is degraded for half-body clips where hand keypoints are mostly missing." |
| `benchmark_linkage` | validation errors, key mismatch, ref/audio mismatch | "2 generated videos could not be linked to benchmark rows and were excluded from benchmark-dependent metrics." |
| `benchmark_av_sync` | target clip and benchmark audio segment are degraded/failed before model evaluation | "4/50 benchmark target/audio pairs are not sufficiently synchronized, so generated Sync scores for those samples are excluded from primary aggregates." |
| `conditioning_mismatch` | manifest/benchmark conditioning mismatch | "3 samples used non-benchmark audio, so benchmark-wav sync was skipped." |
| `metric_coverage` | any metric below min coverage | "Coverage warnings dominate FaceSim and Sync; aggregate scores should be read as degraded." |
| `metric_runtime` | model load errors, dependency/tool failures | "IQA failed because the configured model checkpoint was unavailable." |

Each sample-level metric result should optionally include:

```json
{
  "taxonomy": [
    {
      "category": "identity_consistency",
      "code": "facesim_low_detection_coverage",
      "severity": "medium",
      "message": "face_detection_rate below min_coverage=0.8",
      "evidence": {
        "valid": 20,
        "total": 81,
        "rate": 0.247
      }
    }
  ]
}
```

### Natural-Language Report Summary

`html_report.html` and `metrics.json` should include an automatically generated `diagnostic_summary`.

It must be count-grounded and should not infer causes beyond the measured taxonomy.

Example:

```text
This run evaluated 20 samples. The main degradation source is identity consistency:
8/20 samples have degraded FaceSim, and 6/8 are due to face_detection_rate below 0.8.
Audio-visual sync is the second largest issue: 5/20 samples are degraded, including
3 samples with fewer than 4 valid sync windows. Visual quality is mostly valid, but
2 samples show frozen-frame warnings above 10%. Benchmark linkage passed for all
20 samples.
```

Rules:

- Mention the top 3 taxonomy categories by affected sample count.
- For each category, include affected count and denominator.
- Include the most common warning/error code inside that category.
- Link to representative sample cards, preferably the worst 3 samples by severity.
- If most metrics are `ok`, say so directly and avoid inventing problems.
- If many metrics are `degraded`, explicitly state that aggregate scores should be interpreted cautiously.
- If benchmark linkage fails, mention it before model-quality conclusions.

`metrics.json` should include:

```json
{
  "diagnostic_summary": {
    "text": "...",
    "top_categories": [
      {
        "category": "identity_consistency",
        "affected_samples": 8,
        "total_samples": 20,
        "top_codes": [
          {"code": "facesim_low_detection_coverage", "count": 6}
        ],
        "representative_samples": ["000004", "000041", "000096"]
      }
    ]
  }
}
```

## paper_table.csv

The CSV is for experiment tables.

Suggested columns:

```text
method,checkpoint,split,n,
sanity_pass_rate,
facesim_mean,facesim_min_avg,face_detection_rate,
sync_c,sync_d,sync_c_win,sync_d_win,
iqa_qalign_sampled,aes_qalign_sampled,
akd,pck
```

Empty results should remain empty. Do not write `0` for failed or skipped metrics.

If a different IQA/AES backend is used, the column name must change with the metric variant, for example `iqa_custom_aesthetic_v1_sampled`. Do not write a generic `iqa` or `aes` column when backend-specific scores are present.

## current_run_table.csv and comparison_table.csv

`current_run_table.csv` should always be written when reports are enabled. It summarizes the current run's aggregate experiment results.

`comparison_table.csv` should be written when comparison is enabled or `run_mode=compare_only`. It summarizes baseline/current deltas.

The HTML report should render both tables:

- current experiment results table;
- baseline-vs-current comparison table;
- per-metric deltas with better/worse direction;
- per-sample improvement/regression links when overlapping sample IDs are available.

## html_report.html

The HTML report should include:

- acceptance status and failed/warned criteria
- natural-language diagnostic summary
- natural-language comparative summary, if comparison is enabled
- top failure/degradation taxonomy table
- current run experiment result table
- baseline/current comparison table, if comparison is enabled
- aggregate metric table
- coverage table
- failure summary
- warning summary
- per-sample cards
- reference image
- generated keyframes
- target keyframes
- FaceSim crops
- Sync curve
- IQA/AES timeline
- pose overlay, if available

## Usage

Natural language:

```text
快速评测一下最新权重
```

```text
完整评测一下最新权重
```

```text
只看口型，评测一下最新权重
```

```text
用论文格式评测最新权重，并生成 paper_table.csv 和 html_report.html
```

```text
用 480x832、20fps、81帧评测最新权重
```

```text
用 720x1280、25fps、121帧评测最新权重
```

```text
用 official_quick_50 评估最新权重
```

```text
用 official_main_300 评估最新权重，并和上一次 official_main_300 结果对比
```

```text
评测最新权重，并和上一次评测结果对比
```

```text
最新权重比 checkpoint_005000 好在哪里，坏在哪里？
```

```text
比较 eval_outputs/run_a 和 eval_outputs/run_b，重新生成对比报告
```

```text
我已经有一批生成视频了，只重新算指标
```

```text
我换了一个 FaceSim 实现，重跑一下 FaceSim
```

```text
对别人模型的输出做评估，视频在 /path/to/videos
```

```text
对历史 eval_outputs/xxx 重新生成 html_report
```

```text
用10个样本评估最新权重，caption用固定embedding，不用benchmark里的prompt
```

```text
随机抽10个样本评估最新权重，用每条benchmark音频和图片
```

```text
用10个样本评估最新权重
```

```text
随机抽10个样本评估最新权重
```

```text
只用中文女声10条评估最新权重
```

```text
帮我评测一下最新权重
```

```text
帮我评测一下 checkpoint_model_007500
```

```text
帮我评测一下 /abs/path/to/model.pt
```

```text
用 benchmark/benchmark.jsonl 评测最新权重，先跑 20 条
```

```text
评测最新权重，只跑 sanity、FaceSim、Sync
```

```text
快速评测最新权重，但只用中文女声样本
```

```text
完整评测别人模型的输出，视频在 /path/to/videos
```

Reproducible command shape:

```bash
python auto_eval/eval_digital_human_agent.py \
  --request "帮我评测一下最新权重" \
  --benchmark benchmark/benchmark.jsonl
```

Structured command shape:

```bash
python auto_eval/eval_digital_human_agent.py \
  --checkpoint latest \
  --benchmark benchmark/benchmark.jsonl \
  --eval_split official_quick_50 \
  --text_mode fixed_embedding \
  --text_emb_path raw_dataset/text_emb/person_speaking_zh_umt5_xxl_bf16.pt \
  --audio_mode benchmark_wav \
  --reference_mode materialized_ref
```

Quick preset command:

```bash
python auto_eval/eval_digital_human_agent.py \
  --checkpoint latest \
  --benchmark benchmark/benchmark.jsonl \
  --metric_preset quick
```

Lip-sync-only preset command:

```bash
python auto_eval/eval_digital_human_agent.py \
  --checkpoint latest \
  --benchmark benchmark/benchmark.jsonl \
  --metric_preset lip_sync
```

Paper preset command:

```bash
python auto_eval/eval_digital_human_agent.py \
  --checkpoint latest \
  --benchmark benchmark/benchmark.jsonl \
  --metric_preset paper \
  --eval_split official_main_300
```

Explicit generation profile command:

```bash
python auto_eval/eval_digital_human_agent.py \
  --checkpoint latest \
  --benchmark benchmark/benchmark.jsonl \
  --width 720 \
  --height 1280 \
  --fps 25 \
  --frame_num 121 \
  --metric_preset quick
```

Random sample command:

```bash
python auto_eval/eval_digital_human_agent.py \
  --checkpoint latest \
  --benchmark benchmark/benchmark.jsonl \
  --sample_count 10 \
  --sample_mode random \
  --sample_seed 20260609
```

Manifest-eval command:

```bash
python auto_eval/eval_digital_human_agent.py \
  --run_mode manifest_eval \
  --benchmark benchmark/benchmark.jsonl \
  --generated_manifest eval_outputs/old_run/inference/inference_manifest.jsonl \
  --metrics facesim \
  --benchmark_linkage strict
```

Video-dir-eval command with a third-party video directory:

```bash
python auto_eval/eval_digital_human_agent.py \
  --run_mode video_dir_eval \
  --benchmark benchmark/benchmark.jsonl \
  --generated_video_dir /path/to/third_party/videos \
  --benchmark_linkage strict
```

Report-only command:

```bash
python auto_eval/eval_digital_human_agent.py \
  --run_mode report_only \
  --source_eval_dir eval_outputs/old_run \
  --output_dir eval_outputs/old_run_report_refresh
```

Infer-and-compare command:

```bash
python auto_eval/eval_digital_human_agent.py \
  --run_mode infer_eval \
  --checkpoint latest \
  --benchmark benchmark/benchmark.jsonl \
  --eval_split official_quick_50 \
  --baseline_eval_dir eval_outputs/checkpoint_005000_eval \
  --comparison_enabled true
```

Compare-only command:

```bash
python auto_eval/eval_digital_human_agent.py \
  --run_mode compare_only \
  --baseline_metrics_json eval_outputs/checkpoint_005000_eval/metrics.json \
  --current_metrics_json eval_outputs/checkpoint_007500_eval/metrics.json \
  --output_dir eval_outputs/compare_005000_vs_007500
```

## Minimal Validation Plan

Synthetic metric fixtures should be maintained for repeatable validation:

| Fixture | Purpose |
| --- | --- |
| `perfect_copy` | Generated video is an exact copy of the materialized target clip; sanity should pass and paired diagnostics should be near best possible. |
| `black_video` | All frames are black; sanity should warn/fail and IQA/AES should be degraded or failed according to coverage. |
| `frozen_video` | One frame repeated for the whole clip; frozen-frame diagnostics should degrade quality-related metrics. |
| `wrong_audio` | Correct-looking video with unrelated audio; Sync should degrade/fail while visual metrics may pass. |
| `shifted_audio_200ms` | Correct audio shifted by 200ms; Sync should report offset/degradation. |
| `no_face` | No detectable face; FaceSim and Sync face-crop-dependent metrics should fail or skip with null scores. |
| `two_faces` | Multiple detected faces; face track selection must be stable and shared by FaceSim/Sync. |
| `identity_swap` | Selected identity changes mid-video; face track switch and FaceSim degradation should be reported. |

1. Run `--sample_count 2 --sample_mode first --metrics sanity`.
2. Confirm `resolved_config.json` is written before inference/eval starts.
3. Confirm execution reloads `resolved_config.json` from disk rather than continuing with an unpersisted in-memory config.
4. Confirm `metrics.json` is generated and records `resolved_config_path` plus `resolved_config_sha256`.
5. Confirm `benchmark_index.json` is generated and sample IDs are unique.
6. Inject duplicate/conflicting sample IDs and confirm benchmark indexing fails fast.
7. Confirm every sample has an artifact directory.
8. Test a missing generated video path.
9. Confirm failed metrics write `null + error`.
10. Confirm aggregate statistics only use valid scores.
11. Run every synthetic fixture and confirm expected metric status/warnings are produced.
12. Confirm `perfect_copy`, `black_video`, `frozen_video`, `wrong_audio`, `shifted_audio_200ms`, `no_face`, `two_faces`, and `identity_swap` are all covered by automated validation.
13. Run an official split such as `official_smoke_10` and confirm split name/version/hash are recorded in `resolved_config.json` and `metrics.json`.
14. Run `--sample_count 2 --sample_mode random --sample_seed 1234` twice and confirm the selected sample IDs are identical in `resolved_config.json`, but marked `ad_hoc`.
15. Confirm the resolved generation profile records source, width, height, FPS, frame count, duration, and mismatch policy.
16. Run an explicit `720x1280,25fps,121f` profile and confirm benchmark target clips, refs, audio, and pose segments are materialized to that profile.
17. Run with generated videos that do not match the resolved generation profile and confirm the configured mismatch policy is applied.
18. Confirm changing the generation profile invalidates benchmark materialization cache entries.
19. Confirm `benchmark_materialized/refs`, `target_clips`, `audio_segments`, and pose segments when enabled are generated from the same selected GT clip window.
20. Confirm GT clip window start/end/duration are recorded per sample in the materialized benchmark manifest.
21. Confirm `validation_report.json` records `imgpath` vs selected-GT-clip first-frame match statistics.
22. Confirm `validation_report.json` records target clip versus materialized audio Sync validation for every sample where benchmark audio is used.
23. Force a degraded target/audio benchmark pair and confirm generated-video Sync cannot be `ok` for that sample.
24. Confirm failed benchmark target-audio sync is counted under `benchmark_av_sync` taxonomy and excluded from primary Sync aggregates in `paper` preset.
25. Run current task profile `audio_ref_text2video` and confirm `pose_mode=none` does not fail generation evaluation.
26. Run pose-conditioned task profile `audio_ref_pose_text2video` with missing pose and confirm task-contract validation fails before inference.
27. Run a filtered request such as `language=zh,type=vocal,count=3` and confirm only matching rows are selected.
28. Run with `--text_mode fixed_embedding` and confirm benchmark `prompt` is not used as generation condition.
29. Confirm text-alignment metrics are skipped/null when `text_mode=fixed_embedding`.
30. Run `manifest_eval` on an existing `inference_manifest.jsonl` and confirm no inference is launched.
31. Run `manifest_eval --metrics facesim` and confirm only FaceSim-dependent values are recomputed.
32. Swap the FaceSim implementation or model path and confirm `provenance_fingerprint` changes.
33. Confirm aggregate reporting does not silently mix metric values with different provenance fingerprints.
34. Force a low FaceSim detection coverage case and confirm status is `degraded`, not `ok` or `failed`.
35. Force a low Sync valid-window coverage case and confirm status is `degraded` with warnings.
36. Confirm every Sync result records `sync_audio.source_policy`, `resolved_source`, `crop_policy`, and effective duration.
37. Confirm every benchmark-audio-based Sync result records `benchmark_audio_sync_validation`.
38. Create a multi-face sample and confirm FaceSim and Sync use the same `selected_face_track_id`.
39. Confirm `allow_per_frame_best_face=false` prevents per-frame highest-similarity face selection from changing the selected identity.
40. Force a face track switch and confirm `face_track_switch_count` is recorded and FaceSim/Sync are `degraded` when numeric scores exist.
41. Confirm each sample writes `face_track_overlay.mp4`, `selected_face_track.jsonl`, and `face_track_switches.json`.
42. Force a high frozen-frame-rate video and confirm IQA/AES or sanity-dependent metrics are `degraded`.
43. Confirm paired GT diagnostics write `diagnostics_only=true` and do not enter primary quality aggregates.
44. Confirm acceptance criteria write `overall_status`, `failed_criteria`, and `warned_criteria`.
45. Confirm both absolute and `relative_to_baseline` acceptance criteria are evaluated.
46. Confirm IQA/AES metric variants include backend names, such as `iqa_qalign_sampled` and `aes_qalign_sampled`.
47. Run two IQA backends and confirm their values are written to separate columns/aggregates, not a shared `iqa` column.
48. Compare `iqa_qalign_sampled` against `iqa_custom_aesthetic_v1_sampled` and confirm comparison is `not_comparable` by default.
49. Confirm degraded/failed metrics are mapped into failure/degradation taxonomy categories.
50. Confirm `html_report.html` includes a count-grounded natural-language diagnostic summary.
51. Run comparison against a baseline eval dir and confirm `current_run_table.csv`, `comparison_table.csv`, and `comparison.json` are written.
52. Compare two runs with different task profiles and confirm task-dependent metrics are `not_comparable` by default.
53. Compare two runs with different official split hashes and confirm comparison is `not_comparable` by default.
54. Compare two runs with different sample sets and confirm overlap/excluded sample counts are reported.
55. Compare two runs with different metric provenance and confirm affected metrics are `not_comparable` unless explicitly overridden.
56. Confirm comparative summary separates improvements and regressions with counts.
57. Run `video_dir_eval` with a video key not present in benchmark and confirm strict linkage fails with `null + error`.
58. Run `video_dir_eval --linkage_mode unpaired` and confirm only no-reference metrics run unless ref/audio/manifest inputs are supplied.
59. Run `video_dir_eval --linkage_mode weak_key_match` and confirm provenance is marked weak and paper/comparison claims are disabled by default.
60. Run `report_only` and confirm metric values do not change and the source `resolved_config.json` is not modified.
61. Confirm AKD/PCK results record `metric_variant`, `normalization_mode`, `pck_threshold_units`, `keypoint_set`, `visibility_score_min`, and coverage.
62. Run the same pose samples with `normalization_mode=normalized_image` and `normalization_mode=person_bbox_height`; confirm their provenance fingerprints differ and aggregates are not mixed.
63. Compare two eval runs with different AKD/PCK normalization modes and confirm pose metrics are `not_comparable` by default.
64. Run natural-language "快速评测一下" and confirm it resolves to a quick preset, writes `metric_plan.json`, and excludes high-cost metric variants with reasons.
65. Run natural-language "只看口型" and confirm only sanity plus sync-related metric variants are enabled unless the user explicitly requests more.
66. Run natural-language "论文格式的评测" and confirm a versioned official split is preferred; if none exists, the run is marked `comparable_by_default=false`.
67. Run a benchmark with a different row count and confirm preset sample counts are derived from `available_count`, not hard-coded to 300.
68. Confirm `metrics.json` records requested/resolved preset, max cost level, enabled variants, excluded variants, and metric plan path.
69. Confirm MVP default `cache.enabled=false` but `cache_manifest.json` schema is still written.
70. Confirm single-run reuse does not read or write persistent cross-run cache entries.
71. Change `benchmark.jsonl` or a benchmark row path and confirm `benchmark_bound` cache entries are treated as stale/miss when cross-run cache is enabled in MVP-4.
72. Change FaceSim, Sync, IQA/AES, or DWPose model/provenance and confirm affected metric caches miss because provenance fingerprints differ when cross-run cache is enabled.
73. Change GT clip policy, ref source policy, Sync audio crop policy, AKD/PCK normalization mode, or generation profile and confirm affected caches miss when cross-run cache is enabled.
74. Confirm cached metric results still record status, coverage, warnings/errors, and metric provenance.
75. Confirm `--dry_run` writes no inference outputs or metric scores.
76. Confirm natural language cannot inject shell commands, env vars, or non-whitelisted config fields.
77. Confirm `run_id` follows `<timestamp>_<mode>_<checkpoint_step>_<split>_<preset>` and updates `latest`/`latest_success` correctly.
78. Confirm realtime metrics are only documented in `backlog/realtime_eval.md`.

## Implementation Rules

- Do not fabricate metric values.
- `--dry_run` must resolve and write planned config artifacts only; it must not run inference, compute metric scores, or modify `latest_success`.
- Natural-language requests must never be executed as shell commands and must only set whitelisted config fields.
- All input/output paths must be validated against `allowed_roots` unless explicitly trusted by config.
- `run_id` must follow `<timestamp>_<mode>_<checkpoint_step>_<split>_<preset>`, and `latest` / `latest_success` pointers must be maintained.
- Every run must write `resolved_config.json` before inference or metric computation starts.
- Execution must reload and use `resolved_config.json` from disk; the actual execution config must not exist only in memory.
- Any setting that changes execution must appear in `resolved_config.json` before execution starts.
- Every run must declare a task profile and conditioning contract; do not apply one task's benchmark assumptions to another task.
- Required task conditions must be validated before inference in `infer_eval`.
- `resolved_config.json` is immutable after run start; write runtime discoveries to other artifacts.
- `metrics.json` must record `resolved_config_path` and `resolved_config_sha256`.
- Every run must resolve and record a generation/eval profile before benchmark materialization.
- Benchmark target clips, references, audio segments, and pose segments must be materialized to the resolved generation/eval profile, not to a hard-coded `480x832 / 20fps / 81f` setting.
- Explicit user generation specs such as `720x1280 / 25fps / 121f` override model-config defaults and must be written to `resolved_config.json`.
- Ambiguous terms such as `720p` must be resolved to exact width and height before execution.
- Generated videos must be validated against the resolved generation/eval profile; mismatches must follow `generated_output_mismatch_policy`.
- `infer_eval` should consume config-aligned materialized benchmark inputs by default, not raw benchmark media, unless benchmark normalization is explicitly disabled in `resolved_config.json`.
- Benchmark normalization must first select a GT clip window, then derive ref image, target clip, audio segment, and pose segment from that same window.
- For the current first-frame-ref model, default `ref_source_policy` should be `target_first_frame` of the selected GT clip; do not assume benchmark `imgpath` equals that frame.
- When benchmark audio is used for Sync evaluation, the agent must validate target clip versus materialized audio synchronization before judging generated-video Sync.
- Generated-video Sync cannot be `ok` if required benchmark target-audio sync validation is degraded, failed, or missing.
- Reports must separate benchmark target-audio desynchronization from model-generated audio-visual desynchronization.
- FaceSim and Sync mouth crops must use the same selected face track for a sample.
- Do not select the highest-similarity face independently per frame unless explicitly configured; default `allow_per_frame_best_face=false`.
- If face track switching, fragmentation, or too-short selected tracks occur, record `face_track_switch_count` and mark FaceSim/Sync as `degraded` when numeric scores exist.
- Every sample should write `face_track_overlay.mp4`, `selected_face_track.jsonl`, and `face_track_switches.json` when FaceSim or Sync is enabled.
- On metric failure, write `null` score fields and a concrete error message.
- Every metric must record coverage.
- Paired GT diagnostics must set `diagnostics_only=true` and must not enter primary quality aggregates or paper-table primary metrics.
- Acceptance criteria must write `overall_status`, `failed_criteria`, and `warned_criteria`; both `absolute` and `relative_to_baseline` rules are supported.
- Every metric variant must declare `metric_cost_level` in the metric registry.
- Backend-sensitive metrics must include backend in `metric_variant`; IQA/AES must use names such as `iqa_qalign_sampled` and `aes_qalign_sampled`, not generic `iqa` or `aes`.
- Tables and aggregates must use backend-aware metric variant names for IQA/AES columns and must not merge different IQA/AES backends into one column.
- IQA/AES comparisons across different backend-aware variants are `not_comparable` by default.
- Natural-language metric depth requests such as quick/full/lip-sync/paper must resolve to a frozen metric plan in `resolved_config.json` and `metric_plan.json`.
- Preset sample counts must be derived from the current benchmark's `available_count` after filters or from versioned official splits; do not hard-code assumptions such as 300 samples.
- Explicit user metric choices override preset defaults, but excluded or added metric variants must be recorded with reasons.
- High-cost metric variants must not run under a lower-cost preset unless explicitly requested or allowed by `max_cost_level`.
- MVP default cache is disabled: `cache.enabled=false`; only cache schema and single-run reuse are allowed before MVP-4.
- Persistent cross-run cache must not be required by MVP-1 through MVP-3.
- Cache keys must use content hashes, config fingerprints, benchmark fingerprints, and metric provenance fingerprints; never use basename or path alone.
- Benchmark-bound caches must be invalidated when `benchmark.jsonl`, benchmark row hashes, sample ID policy, official split hash, or materialization policy changes.
- Metric caches must be invalidated when implementation, model/checkpoint hash, preprocessing, sampling, thresholds, or metric parameters change.
- Cached metric results must still record status, coverage, warnings/errors, provenance, and cache hit/miss audit.
- If cache validity cannot be proven, recompute instead of reusing the cache.
- Metrics below configured coverage/quality gates must be `degraded`, not `ok`, when a numeric score still exists.
- Degraded metrics must record warnings and remain distinguishable in reports and aggregates.
- Degraded/failed/skipped results should map to taxonomy codes when possible.
- HTML reports must include a count-grounded diagnostic summary and must not infer unmeasured causes.
- Sync-C / Sync-D must record audio source policy, resolved audio source, crop policy, effective duration, and sample rate; without these fields Sync cannot be `ok`.
- Every metric must record metric provenance, including implementation, model/checkpoint identity, preprocessing, sampling, thresholds, and provenance fingerprint.
- AKD/PCK must never report `PCK@5%` without `normalization_mode`, `pck_threshold_units`, `keypoint_set`, and visibility/missing-keypoint policy.
- `akd_pck_body18_norm_img_0p05_inbounds_v1` is available through the `full_pose` preset after 50-sample development admission; see `docs/metric_status/akd_pck_body18_norm_img_0p05_inbounds_v1.md`.
- `full_pose` expands to `sanity + facesim_sampled + sync_global + iqa_aes_sampled + akd_pck_body18` and requires strict `benchmark_key` target binding plus validated body18 mapping.
- Existing `full`, `quick`, and `quick_lipsync` presets remain unchanged, and AKD/PCK remains excluded from `paper_table.csv`.
- Eval profiles are now first-class provenance fields: `dev_native_v1` for development, `paper_model_native_v1` for model-native paper-candidate runs, and planned-only `paper_hoivg_720p25fps5s_v1`.
- AKD/PCK status: `full_pose_ready`; paper readiness still requires `paper_model_native_v1` plus official/versioned split protocol and `paper_table_candidate.csv` gates.
- Official split generator status: `official_smoke_10.v1` and `official_dev_50.v1` are supported; paper splits and `paper_table_candidate.csv` remain pending.
- The primary pose metric should be DWPose body `PCK@0.05` and `AKD` in `normalized_image` coordinates; secondary pose normalization modes must use distinct metric names and separate aggregates.
- Pose comparisons across different AKD/PCK normalization modes, keypoint sets, or visibility thresholds are `not_comparable` by default.
- Aggregates must not silently mix metric results with different provenance fingerprints.
- Baseline/current comparisons must validate benchmark, sample-set, metric-direction, and metric-provenance compatibility before computing deltas.
- Baseline/current comparisons must validate task profile compatibility before computing task-dependent deltas.
- Formal checkpoint comparisons should use official eval splits, not ad-hoc natural-language sampling.
- Official split name/version/path/hash must be recorded for comparable runs.
- Runs using `split_name=ad_hoc` are not comparable by default.
- Sample IDs must come from the benchmark index; do not rely on basename alone when a manifest/mapping provides stronger identity.
- Sample ID collisions must fail fast; do not silently append suffixes or pick the first match.
- Comparison reports must include both current run tables and baseline-vs-current delta tables.
- Comparison summaries must separate improvements from regressions and must not claim improvement from non-comparable metrics.
- Every sample must save visualization artifacts or an error artifact.
- Realtime metrics stay in backlog and must not enter the MVP execution path.
- The first milestone is to make the OmniShow-style offline multimodal metric skeleton run end to end.
- Temporal diagnostics can be added after the MVP is stable.
- Evaluation must record actual conditioning inputs. Do not infer that benchmark `prompt`, `wav_path`, or `imgpath` were used unless the run configuration says so.
- `manifest_eval` must never launch model inference.
- `video_dir_eval` must never launch model inference.
- `video_dir_eval` must declare `strict`, `weak_key_match`, or `unpaired` linkage mode.
- `video_dir_eval` with `unpaired` linkage may only run no-reference metrics unless the user supplies required refs/audio/targets through a manifest or explicit mapping.
- `report_only` must never recompute metric scores.
- `compare_only` must never recompute metric scores.
- Generated videos must be linked to benchmark rows for benchmark-dependent metrics. If linkage fails, do not guess the target row.
- Portability refactor status: eval resources should resolve through `ResourceConfig` plus `pretrained_root` and relative paths; see `configs/resources.example.yaml`.
- Runtime adapters use capability names (`dwpose_onnx_adapter`, `syncnet_v2_adapter`, `facesim_insightface_adapter`, `quality_vbench_adapter`); TalkVid/VBench paths are reference-source provenance only, not default runtime roots.
- Do not package model weights into the Skill; use `scripts/prepare_eval_agent_pretrained.py` to dry-run and explicitly prepare local assets under `eval_agent_pretrained`.
