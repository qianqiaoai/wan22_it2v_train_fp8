#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auto_eval.eval_agent.metric_registry import registry_summary, resolve_metrics
from auto_eval.eval_agent.metrics.akd_pck import AKD_PCK_SCORE_FIELDS, AKD_PCK_VARIANT
from auto_eval.eval_agent.metrics.dwpose import (
    DEFAULT_BACKEND_UNAVAILABLE_STATUS as DEFAULT_DWPOSE_BACKEND_UNAVAILABLE_STATUS,
    DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD,
    DEFAULT_MIN_DETECTION_RATE as DEFAULT_DWPOSE_MIN_DETECTION_RATE,
    DEFAULT_NUM_FRAMES as DEFAULT_DWPOSE_NUM_FRAMES,
    DEFAULT_PROVIDER as DEFAULT_DWPOSE_PROVIDER,
    DWPOSE_SCORE_FIELDS,
    DWPOSE_VARIANT,
)
from auto_eval.eval_agent.metrics.facesim import FACESIM_SCORE_FIELDS
from auto_eval.eval_agent.metrics.iqa_aes import IQA_AES_SCORE_FIELDS
from auto_eval.eval_agent.metrics.sanity import (
    DEFAULT_BLACK_LUMA_THRESHOLD,
    DEFAULT_FRAME_DIFF_THRESHOLD,
    DEFAULT_FRAME_SAMPLE_COUNT,
    DEFAULT_MAX_BLACK_FRAME_RATE,
    DEFAULT_MAX_FROZEN_FRAME_RATE,
    SANITY_VARIANT,
)
from auto_eval.eval_agent.metrics.sync import SYNC_SCORE_FIELDS
from auto_eval.eval_agent.presets import preset_summary, resolve_preset
from auto_eval.eval_agent.paper_candidate import write_paper_table_candidate
from auto_eval.eval_agent.benchmark import resolve_benchmark_profile
from auto_eval.eval_agent.profiles import (
    PROFILE_NAMES,
    apply_eval_profile_to_metric_result,
    metric_profile_compatibility,
    resolve_eval_profile,
)
from auto_eval.eval_agent.report import write_reports
from auto_eval.eval_agent.resources import ResourceConfigError, doctor_resources_payload, resolve_eval_agent_resources, resource_resolution_block
from auto_eval.eval_agent.schema import MetricResult, RunMetrics, SampleEvalResult, utc_now_iso
from auto_eval.eval_agent.splits import (
    OFFICIAL_SPLITS,
    build_index_from_benchmark,
    build_official_split,
    load_split,
    sample_selection_from_split,
    sha256_file,
    validate_official_split,
    write_json,
)
from auto_eval.eval_agent.third_party_adapters.facesim_insightface_adapter import FACESIM_VARIANT
from auto_eval.eval_agent.workflows.benchmark_workflow import (
    run_index_build,
    run_materialize_dry_run,
    run_profile_validate,
)
from auto_eval.eval_agent.workflows.compare_dashboard_workflow import build_compare_dashboard, parse_run_spec


def parse_metric_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                names.append(item)
    return names or ["sanity"]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def load_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(sample, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            samples.append(sample)
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def resolve_sample_selection(args: argparse.Namespace) -> dict[str, Any]:
    if not args.split:
        return {
            "selection_mode": "manifest_eval_ad_hoc",
            "split_name": None,
            "split_version": None,
            "split_file": None,
            "split_sha256": None,
            "benchmark_index_sha256": None,
            "selected_count": args.max_samples,
            "selected_benchmark_keys": [],
            "split_subset": False,
            "not_paper_comparable": True,
            "warning": "no official split requested",
        }
    validation = validate_official_split(args.split)
    if validation["status"] != "validated":
        raise ValueError(f"official split validation failed before metric execution: {json.dumps(validation, ensure_ascii=False)}")
    split = load_split(args.split)
    selection = sample_selection_from_split(split)
    if args.max_samples is not None:
        if not bool(args.allow_split_subset):
            raise ValueError(
                "--max_samples is not allowed with official --split by default; "
                "set --allow_split_subset true for a debug subset marked not_paper_comparable"
            )
        selection["selected_count"] = min(int(args.max_samples), int(selection.get("selected_count") or args.max_samples))
        selection["selected_benchmark_keys"] = list(selection.get("selected_benchmark_keys") or [])[: selection["selected_count"]]
        selection["split_subset"] = True
        selection["not_paper_comparable"] = True
        selection["subset_reason"] = "--allow_split_subset true with --max_samples"
    return selection


def paper_readiness_gate(eval_profile: dict[str, Any], sample_selection: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    if eval_profile.get("name") != "paper_model_native_v1":
        missing.append("eval_profile is not paper_model_native_v1")
    split_name = sample_selection.get("split_name")
    if not split_name or not str(split_name).startswith("official_"):
        missing.append("official split missing")
    if not sample_selection.get("split_sha256"):
        missing.append("split_sha256 missing")
    if not sample_selection.get("benchmark_index_sha256"):
        missing.append("benchmark_index_sha256 missing")
    if sample_selection.get("split_subset"):
        missing.append("split subset is not paper comparable")
    missing.extend(
        [
            "strict binding validation must pass during paired metric execution",
            "metric provenance must be frozen",
            "paper_table_candidate writer not implemented",
        ]
    )
    return {
        "paper_ready": False,
        "can_generate_paper_table_candidate": False,
        "missing_requirements": missing,
    }


def _maybe_path(value: str | None) -> str | None:
    return str(Path(value).expanduser().resolve()) if value else None


def apply_resource_defaults(args: argparse.Namespace, metric_names: list[str]) -> dict[str, Any]:
    needs_resource = {
        "sync_model_path": "sync_global" in metric_names and not args.sync_model_path,
        "facesim_model_path": "facesim_sampled" in metric_names and not args.facesim_model_path,
        "vbench_model_root": "iqa_aes_sampled" in metric_names and not args.vbench_model_root,
        "dwpose_detector_model_path": (
            ("dwpose_sampled" in metric_names or "akd_pck_body18" in metric_names)
            and not args.dwpose_detector_model_path
        ),
        "dwpose_pose_model_path": (
            ("dwpose_sampled" in metric_names or "akd_pck_body18" in metric_names)
            and not args.dwpose_pose_model_path
        ),
    }
    if not any(needs_resource.values()) and not args.resource_config and not args.pretrained_root:
        return {"configured": False, "resource_resolution": {}}
    payload = resolve_eval_agent_resources(
        resource_config_path=args.resource_config,
        pretrained_root=args.pretrained_root,
    )
    resolved = payload["resolved"]
    if needs_resource["sync_model_path"]:
        args.sync_model_path = str(resolved["syncnet_model"].resolved_path)
    if needs_resource["facesim_model_path"]:
        args.facesim_model_path = str(resolved["facesim_model_root"].resolved_path)
    if needs_resource["vbench_model_root"]:
        args.vbench_model_root = str(resolved["vbench_model_root"].resolved_path)
    if needs_resource["dwpose_detector_model_path"]:
        args.dwpose_detector_model_path = str(resolved["dwpose_detector"].resolved_path)
    if needs_resource["dwpose_pose_model_path"]:
        args.dwpose_pose_model_path = str(resolved["dwpose_pose"].resolved_path)
    return {
        "configured": True,
        "config": payload["config"],
        "resource_resolution": {key: value.as_dict() for key, value in resolved.items()},
    }


def validate_samples_against_split(samples: list[dict[str, Any]], sample_selection: dict[str, Any]) -> None:
    if sample_selection.get("selection_mode") != "official_split":
        return
    expected_keys = [str(key) for key in sample_selection.get("selected_benchmark_keys") or []]
    actual_keys = [str(sample.get("benchmark_key")) for sample in samples if sample.get("benchmark_key") not in (None, "")]
    if len(actual_keys) != len(samples):
        raise ValueError("official split manifest validation failed: at least one sample is missing benchmark_key")
    if len(actual_keys) != len(expected_keys):
        raise ValueError(
            "official split manifest validation failed: manifest sample count does not match split; "
            f"expected={len(expected_keys)} actual={len(actual_keys)}"
        )
    if actual_keys != expected_keys[: len(actual_keys)]:
        raise ValueError(
            "official split manifest validation failed: manifest benchmark_key order does not match split; "
            f"expected_prefix={expected_keys[:len(actual_keys)]} actual={actual_keys}"
        )


def infer_sample_id(sample: dict[str, Any], index: int) -> str:
    for key in ("sample_id", "id", "uid", "name", "key"):
        value = sample.get(key)
        if value is not None and str(value):
            return str(value)
    for key in ("output_path", "video", "video_path", "generated_video_path", "generated_path", "path"):
        value = sample.get(key)
        if isinstance(value, str) and value:
            return f"{index:06d}_{Path(value).stem}"
    return f"{index:06d}"


def failed_metric_result(metric_name: str, error: Exception) -> MetricResult:
    if metric_name == "sync_global":
        scores = {field: None for field in SYNC_SCORE_FIELDS}
        coverage = {
            "valid": 0,
            "total": 1,
            "rate": 0.0,
            "min_required": 1.0,
            "min_valid_windows": 1,
        }
    elif metric_name == "facesim_sampled":
        scores = {field: None for field in FACESIM_SCORE_FIELDS}
        coverage = {
            "valid": 0,
            "total": 0,
            "rate": 0.0,
            "min_required": None,
            "min_valid_frames": None,
            "sampled_frame_count": 0,
            "valid_frame_count": 0,
        }
    elif metric_name == "iqa_aes_sampled":
        scores = {field: None for field in IQA_AES_SCORE_FIELDS}
        coverage = {
            "valid": 0,
            "total": 0,
            "rate": 0.0,
            "min_required": None,
            "sampled_frame_count": 0,
            "valid_frame_count": 0,
        }
    elif metric_name == "sanity":
        scores = {
            "video_readable": None,
            "fps": None,
            "frame_count": None,
            "duration": None,
            "width": None,
            "height": None,
            "audio_readable": None,
            "has_audio_stream": None,
            "black_frame_rate": None,
            "frozen_frame_rate": None,
        }
        coverage = {
            "video_metadata": 0.0,
            "audio_metadata": None,
            "frozen_frame_check": None,
            "black_frame_check": None,
            "sampled_frame_count": 0,
            "valid_frame_count": 0,
        }
    elif metric_name == "dwpose_sampled":
        scores = {field: None for field in DWPOSE_SCORE_FIELDS}
        coverage = {
            "valid": 0,
            "total": 0,
            "rate": 0.0,
            "min_required": None,
            "sampled_frame_count": 0,
            "valid_pose_frame_count": 0,
        }
    elif metric_name == "akd_pck_body18":
        scores = {field: None for field in AKD_PCK_SCORE_FIELDS}
        coverage = {
            "valid": 0,
            "total": 0,
            "rate": 0.0,
            "min_required": None,
            "valid_frame_count": 0,
            "total_aligned_frame_count": 0,
            "valid_keypoint_count": 0,
            "total_possible_keypoint_count": 0,
        }
    else:
        scores = {"value": None}
        coverage = {metric_name: 0.0}
    error_payload = {
        "message": str(error),
        "taxonomy": "metric_runtime",
        "phase": "metric_exception",
    }
    return MetricResult(
        status="failed",
        scores=scores,
        coverage=coverage,
        details={"failure_taxonomy": "metric_runtime"},
        provenance={"metric_name": metric_name, "metric_version": "0.1", "created_at": utc_now_iso()},
        warnings=[],
        error=error_payload,
        artifacts={},
    )


def build_resolved_config(
    *,
    args: argparse.Namespace,
    run_id: str,
    manifest_path: Path,
    output_dir: Path,
    metric_names: list[str],
    eval_profile: dict[str, Any],
    benchmark_profile: dict[str, Any],
    sample_selection: dict[str, Any],
    resource_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested_preset = args.preset
    resolved_preset = args.preset
    full_pose_gating_requirements = [
        "manifest contains benchmark_key",
        "strict benchmark_key binding can be validated",
        "materialized_target_clip exists",
        "generated_video exists",
        "body18 mapping is validated by final validation doc",
        "task profile supports paired target-video / pose comparison",
        "no numeric row fallback",
        "no basename fallback",
        "no row-index fallback",
    ]
    cost_levels = {
        "sanity": "low",
        "facesim_sampled": "medium",
        "sync_global": "medium",
        "iqa_aes_sampled": "high",
        "dwpose_sampled": "high",
        "akd_pck_body18": "high",
    }
    enabled_metric_costs = {name: cost_levels.get(name, "unknown") for name in metric_names}
    resource_resolution = (resource_payload or {}).get("resource_resolution") or {}
    resources_config = (resource_payload or {}).get("config") or None
    return {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "run_mode": "manifest_eval",
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "metrics_requested": metric_names,
        "metric_resolution": {
            "requested_preset": requested_preset,
            "resolved_preset": resolved_preset,
            "resolved_metrics": metric_names,
            "enabled_metric_costs": enabled_metric_costs,
            "high_cost_metrics": [name for name, level in enabled_metric_costs.items() if level == "high"],
            "preset_warnings": (
                [
                    "full_pose enables high-cost paired pose accuracy and requires strict benchmark_key target binding"
                ]
                if requested_preset == "full_pose"
                else []
            ),
        },
        "metric_plan_path": str(output_dir / "metric_plan.json"),
        "requested_preset": requested_preset,
        "resolved_preset": resolved_preset,
        "preset": args.preset,
        "eval_profile": eval_profile,
        "benchmark_profile": benchmark_profile,
        "resources": resources_config,
        "sample_selection": sample_selection,
        "paper_readiness_gate": paper_readiness_gate(eval_profile, sample_selection),
        "max_samples": args.max_samples,
        "fail_fast": bool(args.fail_fast),
        "sanity": {
            "metric_variant": SANITY_VARIANT,
            "video_probe_backend": "ffprobe",
            "frame_extract_backend": "ffmpeg",
            "opencv_video_capture_used": False,
            "frame_sample_count": args.sanity_frame_sample_count,
            "frame_diff_threshold": args.sanity_frame_diff_threshold,
            "black_luma_threshold": args.sanity_black_luma_threshold,
            "max_frozen_frame_rate": args.sanity_max_frozen_frame_rate,
            "max_black_frame_rate": args.sanity_max_black_frame_rate,
            "eval_profile": eval_profile.get("name"),
        },
        "sync": {
            "model_path": _maybe_path(args.sync_model_path),
            "resource_resolution": resource_resolution.get("syncnet_model"),
            "allow_generated_audio_track": bool(args.allow_generated_audio_track),
            "audio_crop_policy": args.sync_audio_crop_policy,
            "worker_timeout_sec": args.sync_worker_timeout_sec,
            "cuda_visible_devices": args.sync_cuda_visible_devices,
            "batch_size": args.sync_batch_size,
            "vshift": args.sync_vshift,
        },
        "facesim": {
            "backend": args.facesim_backend,
            "model_path": _maybe_path(args.facesim_model_path),
            "resource_resolution": resource_resolution.get("facesim_model_root"),
            "detector_model_path": (
                str(Path(args.facesim_detector_model_path).resolve())
                if args.facesim_detector_model_path
                else None
            ),
            "num_frames": args.facesim_num_frames,
            "min_detection_rate": args.facesim_min_detection_rate,
            "min_valid_frames": args.facesim_min_valid_frames,
            "reference_policy": args.facesim_reference_policy,
            "backend_unavailable_status": args.facesim_backend_unavailable_status,
            "metric_variant": FACESIM_VARIANT,
        },
        "iqa_aes": {
            "backend": args.iqa_aes_backend,
            "model_root": _maybe_path(args.vbench_model_root),
            "resource_resolution": resource_resolution.get("vbench_model_root"),
            "iqa_model_path": str(Path(args.iqa_model_path).resolve()) if args.iqa_model_path else None,
            "aes_model_path": str(Path(args.aes_model_path).resolve()) if args.aes_model_path else None,
            "clip_model_path": str(Path(args.clip_model_path).resolve()) if args.clip_model_path else None,
            "num_frames": args.iqa_aes_num_frames,
            "min_frame_coverage": args.iqa_aes_min_frame_coverage,
            "backend_unavailable_status": args.iqa_aes_backend_unavailable_status,
            "device": args.iqa_aes_device,
            "offline": bool(args.iqa_aes_offline),
            "max_frozen_frame_rate": args.iqa_aes_max_frozen_frame_rate,
            "max_black_frame_rate": args.iqa_aes_max_black_frame_rate,
        },
        "dwpose": {
            "metric_variant": DWPOSE_VARIANT,
            "detector_model_path": _maybe_path(args.dwpose_detector_model_path),
            "pose_model_path": _maybe_path(args.dwpose_pose_model_path),
            "detector_resource_resolution": resource_resolution.get("dwpose_detector"),
            "pose_resource_resolution": resource_resolution.get("dwpose_pose"),
            "num_frames": args.dwpose_num_frames,
            "min_detection_rate": args.dwpose_min_detection_rate,
            "keypoint_confidence_threshold": args.dwpose_keypoint_confidence_threshold,
            "provider": args.dwpose_provider,
            "backend_unavailable_status": args.dwpose_backend_unavailable_status,
        },
        "pose_accuracy": {
            "metric_variant": AKD_PCK_VARIANT,
            "enabled_reason": "preset=full_pose" if args.preset == "full_pose" else "explicit_metric_request",
            "dwpose_resource_resolution": {
                "detector": resource_resolution.get("dwpose_detector"),
                "pose": resource_resolution.get("dwpose_pose"),
            },
            "target_source": args.pose_accuracy_target_source,
            "generated_source": args.pose_accuracy_generated_source,
            "frame_alignment": args.pose_accuracy_frame_alignment,
            "coordinate_space": "normalized_xy",
            "normalization_mode": "normalized_image",
            "visibility_score_min": args.pose_accuracy_visibility_score_min,
            "pck_threshold": args.pose_accuracy_pck_threshold,
            "pck_threshold_units": "normalized_image",
            "min_valid_keypoints_per_frame": args.pose_accuracy_min_valid_keypoints_per_frame,
            "min_valid_frame_rate": args.pose_accuracy_min_valid_frame_rate,
            "min_matched_keypoint_rate": args.pose_accuracy_min_matched_keypoint_rate,
            "require_binding_validated": bool(args.pose_accuracy_require_binding_validated),
            "allow_experimental": bool(args.pose_accuracy_allow_experimental),
            "missing_keypoint_policy": "exclude_unmatched",
            "out_of_frame_policy": "exclude_from_primary_metric",
            "body18_mapping_status": "validated",
            "not_in_paper_table": True,
            "paper_ready": False,
            "cost_level": "high",
            "task_applicability": "paired_key_preserving_target_video_or_pose_comparable_eval",
            "gating_requirements": full_pose_gating_requirements,
            "forbidden_fallbacks": ["numeric_row", "basename", "row_index"],
            "gating_failure_policy": "metric_level_failed_or_skipped_with_null_scores",
        },
    }


def write_resolved_config(resolved_config: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resolved_config.json"
    path.write_text(json.dumps(resolved_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_metric_plan(
    *,
    metric_definitions: list[Any],
    resolved_config: dict[str, Any],
    output_dir: Path,
) -> Path:
    path = output_dir / "metric_plan.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for definition in metric_definitions:
        eval_profile = resolved_config.get("eval_profile") or {}
        entry: dict[str, Any] = {
            "name": definition.name,
            "metric_cost_level": definition.metric_cost_level,
            "description": definition.description,
            "requires_generated_video": definition.requires_generated_video,
            "requires_reference": definition.requires_reference,
            "requires_target_video_or_pose": getattr(definition, "requires_target_video_or_pose", False),
            "requires_binding_validated": getattr(definition, "requires_binding_validated", False),
            "requires_body18_mapping_validated": getattr(definition, "requires_body18_mapping_validated", False),
            "eval_profile_name": eval_profile.get("name"),
            "eval_profile_version": eval_profile.get("version"),
            "profile_compatibility": metric_profile_compatibility(definition.name, eval_profile),
            "benchmark_profile_name": (resolved_config.get("benchmark_profile") or {}).get("name"),
            "benchmark_profile_version": (resolved_config.get("benchmark_profile") or {}).get("version"),
            "benchmark_profile_sha256": (resolved_config.get("benchmark_profile") or {}).get("sha256"),
        }
        if definition.name == "facesim_sampled":
            entry["config"] = resolved_config.get("facesim")
        elif definition.name == "sanity":
            entry["config"] = resolved_config.get("sanity")
        elif definition.name == "sync_global":
            entry["config"] = resolved_config.get("sync")
        elif definition.name == "iqa_aes_sampled":
            entry["config"] = resolved_config.get("iqa_aes")
        elif definition.name == "dwpose_sampled":
            entry["config"] = resolved_config.get("dwpose")
        elif definition.name == "akd_pck_body18":
            entry["config"] = resolved_config.get("pose_accuracy")
            entry["enabled_reason"] = resolved_config.get("pose_accuracy", {}).get("enabled_reason")
            entry["gating_requirements"] = resolved_config.get("pose_accuracy", {}).get("gating_requirements")
            entry["forbidden_fallbacks"] = resolved_config.get("pose_accuracy", {}).get("forbidden_fallbacks")
            entry["paper_table_status"] = "excluded"
        entries.append(entry)
    payload = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "run_id": resolved_config.get("run_id"),
        "requested_preset": resolved_config.get("requested_preset"),
        "resolved_preset": resolved_config.get("resolved_preset"),
        "preset": resolved_config.get("preset"),
        "sample_selection": resolved_config.get("sample_selection"),
        "benchmark_profile": resolved_config.get("benchmark_profile"),
        "paper_readiness_gate": resolved_config.get("paper_readiness_gate"),
        "metrics_requested": resolved_config.get("metrics_requested"),
        "metric_resolution": resolved_config.get("metric_resolution"),
        "metrics": entries,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_manifest_eval(args: argparse.Namespace) -> int:
    eval_profile = resolve_eval_profile(args.eval_profile)
    if not bool(eval_profile.get("implemented")):
        print(
            json.dumps(
                {
                    "error": "eval_profile_not_implemented",
                    "message": (
                        f"eval_profile={eval_profile.get('name')} is {eval_profile.get('status')}; "
                        "this profile is planned only and cannot run metrics yet"
                    ),
                    "eval_profile": eval_profile,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        benchmark_profile = resolve_benchmark_profile(
            args.benchmark_profile,
            benchmark_root=args.benchmark_root,
        ).as_resolved_config()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": "benchmark_profile_failed",
                    "message": str(exc),
                    "benchmark_profile": args.benchmark_profile,
                    "benchmark_root": args.benchmark_root,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        sample_selection = resolve_sample_selection(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": "sample_selection_failed",
                    "message": str(exc),
                    "split": args.split,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    if args.metrics:
        metric_names = parse_metric_names(args.metrics)
    elif args.preset:
        metric_names = resolve_preset(args.preset)
    else:
        metric_names = ["sanity"]
    metric_definitions = resolve_metrics(metric_names)
    try:
        resource_payload = apply_resource_defaults(args, metric_names)
    except ResourceConfigError as exc:
        print(
            json.dumps(
                {
                    "error": "resource_config_failed",
                    "message": str(exc),
                    "resource_config": args.resource_config,
                    "pretrained_root": args.pretrained_root,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    run_id = args.run_id or output_dir.name
    resolved_config = build_resolved_config(
        args=args,
        run_id=run_id,
        manifest_path=manifest_path,
        output_dir=output_dir,
        metric_names=metric_names,
        eval_profile=eval_profile,
        benchmark_profile=benchmark_profile,
        sample_selection=sample_selection,
        resource_payload=resource_payload,
    )
    resolved_config_path = write_resolved_config(resolved_config, output_dir)
    resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    metric_plan_path = write_metric_plan(
        metric_definitions=metric_definitions,
        resolved_config=resolved_config,
        output_dir=output_dir,
    )
    if args.dry_run_config:
        report = {
            "run_id": run_id,
            "dry_run_config": True,
            "resolved_config_json": str(resolved_config_path),
            "metric_plan_json": str(metric_plan_path),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    samples = load_jsonl(manifest_path, max_samples=args.max_samples)
    try:
        validate_samples_against_split(samples, resolved_config["sample_selection"])
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": "official_split_manifest_mismatch",
                    "message": str(exc),
                    "split": args.split,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    eval_samples: list[SampleEvalResult] = []

    for index, sample in enumerate(samples):
        sample_id = infer_sample_id(sample, index)
        sample_result = SampleEvalResult(
            sample_id=sample_id,
            sample=sample,
            metrics={},
        )
        context = {
            "sample_id": sample_id,
            "output_dir": str(output_dir),
            "run_id": run_id,
            "resolved_config": resolved_config,
            "resolved_config_path": str(resolved_config_path),
            "metric_plan_path": str(metric_plan_path),
            "sync": resolved_config["sync"],
            "sanity": resolved_config["sanity"],
            "facesim": resolved_config["facesim"],
            "iqa_aes": resolved_config["iqa_aes"],
            "dwpose": resolved_config["dwpose"],
            "pose_accuracy": resolved_config["pose_accuracy"],
        }
        for metric_definition in metric_definitions:
            try:
                metric_result = metric_definition.fn(sample, context=context)
            except Exception as exc:
                if args.fail_fast:
                    raise
                metric_result = failed_metric_result(metric_definition.name, exc)
            sample_result.metrics[metric_definition.name] = apply_eval_profile_to_metric_result(
                metric_result,
                metric_name=metric_definition.name,
                eval_profile=resolved_config["eval_profile"],
            )
        eval_samples.append(sample_result)

    run_metrics = RunMetrics(
        run_id=run_id,
        run_mode="manifest_eval",
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        created_at=utc_now_iso(),
        metrics_requested=metric_names,
        samples=eval_samples,
    )
    report_paths = write_reports(run_metrics, output_dir)
    report_paths["resolved_config_json"] = str(resolved_config_path)
    report_paths["metric_plan_json"] = str(metric_plan_path)

    print(json.dumps({"run_id": run_id, "reports": report_paths}, ensure_ascii=False, indent=2))
    return 0


def run_report_paper_candidate(args: argparse.Namespace) -> int:
    source_eval_dir = Path(args.source_eval_dir).resolve()
    metrics_json = source_eval_dir / "metrics.json"
    if not metrics_json.exists():
        print(
            json.dumps(
                {
                    "error": "metrics_json_missing",
                    "message": f"source eval dir does not contain metrics.json: {metrics_json}",
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    output_dir = Path(args.output_dir).resolve()
    if output_dir == source_eval_dir:
        print(
            json.dumps(
                {
                    "error": "paper_candidate_output_dir_must_differ",
                    "message": "--output_dir must differ from --source_eval_dir for report_paper_candidate",
                    "source_eval_dir": str(source_eval_dir),
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    result = write_paper_table_candidate(
        metrics_json,
        output_dir,
        config={
            "method": args.method,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("generated_candidate") else 2


def run_doctor_resources(args: argparse.Namespace) -> int:
    try:
        payload = doctor_resources_payload(
            resource_config_path=args.resource_config,
            pretrained_root=args.pretrained_root,
        )
    except ResourceConfigError as exc:
        payload = {
            "status": "failed",
            "error": "resource_config_failed",
            "message": str(exc),
            "resource_config": args.resource_config,
            "pretrained_root": args.pretrained_root,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("status") == "ok" else 2


def run_benchmark(args: argparse.Namespace) -> int:
    try:
        if args.benchmark_action == "profile_validate":
            result = run_profile_validate(
                benchmark_jsonl=args.benchmark,
                benchmark_profile=args.benchmark_profile,
                benchmark_root=args.benchmark_root,
                output_dir=args.output_dir,
            )
        elif args.benchmark_action == "index":
            result = run_index_build(
                benchmark_jsonl=args.benchmark,
                benchmark_profile=args.benchmark_profile,
                benchmark_root=args.benchmark_root,
                output_dir=args.output_dir,
                derive_missing_keys=bool(args.derive_missing_keys),
            )
        elif args.benchmark_action == "materialize":
            if not args.dry_run:
                raise ValueError("benchmark materialize currently supports --dry_run only through this public CLI")
            result = run_materialize_dry_run(
                benchmark_jsonl=args.benchmark,
                benchmark_profile=args.benchmark_profile,
                benchmark_root=args.benchmark_root,
                output_dir=args.output_dir,
                count=args.count,
                width=args.generation_width,
                height=args.generation_height,
                fps=args.generation_fps,
                frame_num=args.generation_frame_num,
                crop_policy=args.gt_clip_policy,
            )
        else:
            raise ValueError(f"unsupported benchmark action: {args.benchmark_action}")
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": "benchmark_command_failed",
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"ok", "validated"} else 2


def run_splits(args: argparse.Namespace) -> int:
    try:
        if args.splits_action == "generate":
            benchmark_path = Path(args.benchmark).resolve()
            split_dir = Path(args.split_dir).resolve()
            split_dir.mkdir(parents=True, exist_ok=True)
            profile = resolve_benchmark_profile(args.benchmark_profile, benchmark_root=args.benchmark_root)
            requested_v2 = any(str(split_id).endswith(".v2") for split_id in args.splits)
            if requested_v2:
                benchmark_index_path = split_dir / "profile_aware_index_v2" / "benchmark_index.json"
                if not benchmark_index_path.exists():
                    raise FileNotFoundError(
                        f"profile-aware v2 split generation requires {benchmark_index_path}. "
                        "Build a profile-aware benchmark index that preserves the v1 benchmark_key set first."
                    )
                benchmark_index = json.loads(benchmark_index_path.read_text(encoding="utf-8"))
                index_profile = benchmark_index.get("benchmark_profile") or {}
                if profile.sha256 and index_profile.get("sha256") != profile.sha256:
                    raise ValueError(
                        "benchmark profile mismatch for v2 split generation: "
                        f"index={index_profile.get('sha256')} requested={profile.sha256}"
                    )
                benchmark_index_sha256 = sha256_file(benchmark_index_path)
            else:
                benchmark_index = build_index_from_benchmark(benchmark_path, benchmark_profile=profile)
                benchmark_index_path = split_dir / "benchmark_index.derived_row_hash.v1.json"
                write_json(benchmark_index_path, benchmark_index)
                benchmark_index_sha256 = sha256_file(benchmark_index_path)
            collision_count = int(benchmark_index.get("collision_check", {}).get("collision_count", -1))
            if collision_count != 0:
                raise ValueError(f"benchmark key collision_count must be 0, got {collision_count}")
            generated: dict[str, dict[str, Any]] = {}
            for split_id in args.splits:
                split_path = split_dir / f"{split_id}.json"
                created_at = None
                if split_path.exists():
                    try:
                        created_at = json.loads(split_path.read_text(encoding="utf-8")).get("created_at")
                    except Exception:
                        created_at = None
                split_payload = build_official_split(
                    split_id=split_id,
                    benchmark_jsonl=str(benchmark_path),
                    benchmark_index_path=str(benchmark_index_path),
                    benchmark_index_sha256=benchmark_index_sha256,
                    created_at=created_at,
                )
                write_json(split_path, split_payload)
                validation = validate_official_split(split_path, split_dir=split_dir)
                if validation["status"] != "validated":
                    raise ValueError(f"generated split failed validation: {split_path}: {validation}")
                generated[split_id] = {
                    "path": str(split_path),
                    "split_sha256": split_payload["split_sha256"],
                    "sample_count": split_payload["sample_count"],
                    "benchmark_profile_sha256": split_payload.get("benchmark_profile_sha256"),
                }
            result = {
                "status": "ok",
                "benchmark_index_path": str(benchmark_index_path),
                "benchmark_index_sha256": benchmark_index_sha256,
                "generated_splits": generated,
            }
        elif args.splits_action == "validate":
            output_dir = Path(args.output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            results = []
            failed = False
            for split in args.splits:
                validation = validate_official_split(split, split_dir=args.split_dir)
                split_name = validation.get("split_name") or Path(str(split)).stem
                output_path = output_dir / f"{split_name}.validation.json"
                write_json(output_path, validation)
                validation["validation_output"] = str(output_path)
                results.append(validation)
                failed = failed or validation["status"] != "validated"
            result = {"status": "failed" if failed else "validated", "results": results}
            write_json(output_dir / "summary.json", result)
        else:
            raise ValueError(f"unsupported splits action: {args.splits_action}")
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": "splits_command_failed",
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"ok", "validated"} else 2


def run_compare(args: argparse.Namespace) -> int:
    try:
        if args.compare_action != "dashboard":
            raise ValueError(f"unsupported compare action: {args.compare_action}")
        eval_dirs = [Path(path) for path in args.eval_dir or []]
        run_specs = [parse_run_spec(spec) for spec in args.run_spec or []]
        result = build_compare_dashboard(
            eval_dirs=eval_dirs,
            run_specs=run_specs,
            output_dir=Path(args.output_dir),
            title=args.title,
            link_mode=args.link_mode,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": "compare_command_failed",
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "ok" else 2


def run_planned_command(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "status": "planned",
                "command": args.command,
                "message": "This public CLI surface is reserved; implementation remains in existing compatibility scripts/workflows.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline digital human video eval agent")
    parser.add_argument("--list_metrics", action="store_true", help="List available metrics and exit")
    parser.add_argument("--list_presets", action="store_true", help="List available metric presets and exit")

    subparsers = parser.add_subparsers(dest="command")
    manifest_parser = subparsers.add_parser("manifest_eval", help="Evaluate existing outputs from a manifest")
    manifest_parser.add_argument("--manifest", required=True, help="Path to inference_manifest.jsonl")
    manifest_parser.add_argument("--output_dir", required=True, help="Directory for metrics.json and reports")
    manifest_parser.add_argument("--metrics", nargs="+", default=None, help="Metric names, e.g. sanity")
    manifest_parser.add_argument(
        "--preset",
        default=None,
        help=(
            "Metric preset. smoke=sanity; quick=sanity+facesim_sampled; "
            "quick_lipsync=sanity+sync_global; full excludes pose; "
            "full_pose adds high-cost akd_pck_body18 with strict target binding."
        ),
    )
    manifest_parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for smoke tests")
    manifest_parser.add_argument("--split", default=None, help="Official split id, e.g. official_smoke_10.v1")
    manifest_parser.add_argument(
        "--allow_split_subset",
        type=parse_bool,
        default=False,
        help="Allow --max_samples to truncate an official split. Marks the run not paper comparable.",
    )
    manifest_parser.add_argument(
        "--dry_run_config",
        action="store_true",
        help="Write resolved_config.json and metric_plan.json, then exit before loading samples or running metrics.",
    )
    manifest_parser.add_argument("--run_id", default=None, help="Optional run id; defaults to output_dir basename")
    manifest_parser.add_argument("--fail_fast", action="store_true", help="Abort the whole run on first metric exception")
    manifest_parser.add_argument(
        "--eval_profile",
        default="dev_native_v1",
        choices=list(PROFILE_NAMES),
        help=(
            "Evaluation profile. dev_native_v1 is default; paper_model_native_v1 is model-native "
            "paper-candidate profile; paper_hoivg_720p25fps5s_v1 is planned and currently fails."
        ),
    )
    manifest_parser.add_argument(
        "--resource_config",
        default=None,
        help="ResourceConfig YAML/JSON. CLI explicit model paths override resources.",
    )
    manifest_parser.add_argument(
        "--pretrained_root",
        default=None,
        help="Override pretrained_root for ResourceConfig resolution.",
    )
    manifest_parser.add_argument(
        "--benchmark_profile",
        default=None,
        help=(
            "Benchmark profile YAML/JSON. Defaults to configs/benchmark_profiles/"
            "audio_ref_text2video.current.yaml and records default_used=true."
        ),
    )
    manifest_parser.add_argument(
        "--benchmark_root",
        default=None,
        help="Optional root for resolving relative media paths in the benchmark profile.",
    )
    manifest_parser.add_argument(
        "--sanity_frame_sample_count",
        type=int,
        default=DEFAULT_FRAME_SAMPLE_COUNT,
        help="Number of uniformly sampled frames for sanity black/frozen checks.",
    )
    manifest_parser.add_argument(
        "--sanity_frame_diff_threshold",
        type=float,
        default=DEFAULT_FRAME_DIFF_THRESHOLD,
        help="Mean grayscale frame-difference threshold for frozen transition detection.",
    )
    manifest_parser.add_argument(
        "--sanity_black_luma_threshold",
        type=float,
        default=DEFAULT_BLACK_LUMA_THRESHOLD,
        help="Mean grayscale luma threshold for black-frame detection.",
    )
    manifest_parser.add_argument(
        "--sanity_max_frozen_frame_rate",
        type=float,
        default=DEFAULT_MAX_FROZEN_FRAME_RATE,
        help="Sanity degraded threshold for frozen frame rate.",
    )
    manifest_parser.add_argument(
        "--sanity_max_black_frame_rate",
        type=float,
        default=DEFAULT_MAX_BLACK_FRAME_RATE,
        help="Sanity degraded threshold for black frame rate.",
    )
    manifest_parser.add_argument(
        "--sync_model_path",
        default=None,
        help="Path to SyncNet v2 checkpoint. Defaults to resources.syncnet.model_relpath.",
    )
    manifest_parser.add_argument(
        "--allow_generated_audio_track",
        action="store_true",
        help="Allow sync_global to use generated video embedded audio when no external audio is available",
    )
    manifest_parser.add_argument(
        "--sync_audio_crop_policy",
        default="align_to_generated_video_duration",
        choices=["none", "align_to_generated_video_duration", "manifest_segment_bounds", "fail_on_mismatch"],
        help="Audio crop policy recorded and used when preparing sync_global inputs",
    )
    manifest_parser.add_argument(
        "--sync_worker_timeout_sec",
        type=int,
        default=900,
        help="Timeout for one SyncNet v2 subprocess worker",
    )
    manifest_parser.add_argument(
        "--sync_cuda_visible_devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for the isolated SyncNet worker subprocess",
    )
    manifest_parser.add_argument("--sync_batch_size", type=int, default=20, help="SyncNet v2 batch size")
    manifest_parser.add_argument("--sync_vshift", type=int, default=15, help="SyncNet v2 vshift")
    manifest_parser.add_argument(
        "--facesim_backend",
        default=FACESIM_VARIANT,
        choices=[FACESIM_VARIANT],
        help="FaceSim backend variant",
    )
    manifest_parser.add_argument(
        "--facesim_model_path",
        default=None,
        help="InsightFace model root; defaults to resources.facesim.model_root_relpath.",
    )
    manifest_parser.add_argument(
        "--facesim_detector_model_path",
        default=None,
        help="Optional explicit scrfd_10g_bnkps.onnx path for preflight validation",
    )
    manifest_parser.add_argument("--facesim_num_frames", type=int, default=16, help="Uniform sampled frame count")
    manifest_parser.add_argument(
        "--facesim_min_detection_rate",
        type=float,
        default=0.8,
        help="Minimum generated-frame face detection rate before FaceSim becomes degraded",
    )
    manifest_parser.add_argument(
        "--facesim_min_valid_frames",
        type=int,
        default=1,
        help="Minimum sampled generated frames with detected face required for numeric FaceSim",
    )
    manifest_parser.add_argument(
        "--facesim_reference_policy",
        default="auto_generation_ref",
        choices=["auto_generation_ref", "benchmark_img", "target_first_frame"],
        help="Reference resolution policy. auto_generation_ref never silently falls back to benchmark imgpath.",
    )
    manifest_parser.add_argument(
        "--facesim_backend_unavailable_status",
        default="failed",
        choices=["failed", "skipped"],
        help="Status when FaceSim backend/model/dependency is unavailable.",
    )
    manifest_parser.add_argument(
        "--iqa_aes_backend",
        default="vbench_local",
        choices=["auto", "vbench_local", "iqa_only", "aes_only"],
        help="IQA/AES backend mode. vbench_local requests both IQA and AES from local VBench models.",
    )
    manifest_parser.add_argument(
        "--vbench_model_root",
        default=None,
        help="Root containing local VBench models, e.g. aesthetic_model/clip_model/pyiqa_model.",
    )
    manifest_parser.add_argument("--iqa_model_path", default=None, help="Optional explicit MUSIQ IQA checkpoint path")
    manifest_parser.add_argument("--aes_model_path", default=None, help="Optional explicit LAION aesthetic head path")
    manifest_parser.add_argument("--clip_model_path", default=None, help="Optional explicit CLIP ViT-L/14 checkpoint path")
    manifest_parser.add_argument(
        "--iqa_aes_num_frames",
        type=int,
        default=8,
        help="Uniform sampled frame count for iqa_aes_sampled",
    )
    manifest_parser.add_argument(
        "--iqa_aes_min_frame_coverage",
        type=float,
        default=0.8,
        help="Minimum sampled frame coverage before iqa_aes_sampled becomes degraded",
    )
    manifest_parser.add_argument(
        "--iqa_aes_backend_unavailable_status",
        default="failed",
        choices=["failed", "skipped"],
        help="Status when all requested IQA/AES components are unavailable.",
    )
    manifest_parser.add_argument(
        "--iqa_aes_device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for VBench local IQA/AES models.",
    )
    manifest_parser.add_argument(
        "--iqa_aes_offline",
        type=parse_bool,
        default=True,
        help="Force offline environment flags for VBench IQA/AES. Accepts true/false.",
    )
    manifest_parser.add_argument(
        "--iqa_aes_max_frozen_frame_rate",
        type=float,
        default=0.3,
        help="Sampled frozen-frame warning threshold for iqa_aes_sampled.",
    )
    manifest_parser.add_argument(
        "--iqa_aes_max_black_frame_rate",
        type=float,
        default=0.3,
        help="Sampled black-frame warning threshold for iqa_aes_sampled.",
    )
    manifest_parser.add_argument(
        "--dwpose_detector_model_path",
        default=None,
        help="Path to DWPose YOLOX-L detector ONNX checkpoint. Defaults to resources.dwpose.detector_model_relpath.",
    )
    manifest_parser.add_argument(
        "--dwpose_pose_model_path",
        default=None,
        help="Path to DWPose dw-ll_ucoco_384 pose ONNX checkpoint. Defaults to resources.dwpose.pose_model_relpath.",
    )
    manifest_parser.add_argument(
        "--dwpose_num_frames",
        type=int,
        default=DEFAULT_DWPOSE_NUM_FRAMES,
        help="Uniform sampled frame count for dwpose_sampled.",
    )
    manifest_parser.add_argument(
        "--dwpose_min_detection_rate",
        type=float,
        default=DEFAULT_DWPOSE_MIN_DETECTION_RATE,
        help="Minimum sampled frame pose detection rate before dwpose_sampled becomes degraded.",
    )
    manifest_parser.add_argument(
        "--dwpose_keypoint_confidence_threshold",
        type=float,
        default=DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD,
        help="Body keypoint confidence threshold for DWPose visibility diagnostics.",
    )
    manifest_parser.add_argument(
        "--dwpose_provider",
        default=DEFAULT_DWPOSE_PROVIDER,
        choices=["cpu", "cuda"],
        help="ONNXRuntime provider request for DWPose. CUDA fails if CUDAExecutionProvider is unavailable.",
    )
    manifest_parser.add_argument(
        "--dwpose_backend_unavailable_status",
        default=DEFAULT_DWPOSE_BACKEND_UNAVAILABLE_STATUS,
        choices=["failed", "skipped"],
        help="Status when DWPose detector/pose model checkpoint is unavailable.",
    )
    manifest_parser.add_argument(
        "--pose_accuracy_target_source",
        default="materialized_target_clip_dwpose",
        choices=["materialized_target_clip_dwpose", "manifest_target_pose"],
        help="Target pose source for explicit AKD/PCK. MVP uses materialized target clip DWPose.",
    )
    manifest_parser.add_argument(
        "--pose_accuracy_generated_source",
        default="generated_video_dwpose",
        choices=["generated_video_dwpose", "existing_dwpose_artifact"],
        help="Generated pose source for explicit AKD/PCK. MVP extracts DWPose from generated video.",
    )
    manifest_parser.add_argument(
        "--pose_accuracy_frame_alignment",
        default="sampled_index_match",
        choices=["sampled_index_match"],
        help="Frame alignment policy for explicit AKD/PCK. Only sampled_index_match is implemented.",
    )
    manifest_parser.add_argument(
        "--pose_accuracy_visibility_score_min",
        type=float,
        default=0.3,
        help="Minimum generated and target body18 keypoint confidence for AKD/PCK matching.",
    )
    manifest_parser.add_argument(
        "--pose_accuracy_pck_threshold",
        type=float,
        default=0.05,
        help="PCK threshold in normalized image coordinates.",
    )
    manifest_parser.add_argument(
        "--pose_accuracy_min_valid_keypoints_per_frame",
        type=int,
        default=6,
        help="Minimum matched in-bounds body18 keypoints required for a valid frame.",
    )
    manifest_parser.add_argument(
        "--pose_accuracy_min_valid_frame_rate",
        type=float,
        default=0.5,
        help="Minimum valid aligned frame rate before AKD/PCK becomes degraded.",
    )
    manifest_parser.add_argument(
        "--pose_accuracy_min_matched_keypoint_rate",
        type=float,
        default=0.3,
        help="Minimum matched body18 keypoint rate before AKD/PCK becomes degraded.",
    )
    manifest_parser.add_argument(
        "--pose_accuracy_require_binding_validated",
        type=parse_bool,
        default=True,
        help="Require strict benchmark_key/materialized target binding for formal AKD/PCK.",
    )
    manifest_parser.add_argument(
        "--pose_accuracy_allow_experimental",
        type=parse_bool,
        default=False,
        help="Allow AKD/PCK to run without the validated body18 mapping gate. Keep false for formal metric.",
    )
    manifest_parser.set_defaults(func=run_manifest_eval)

    paper_candidate_parser = subparsers.add_parser(
        "report_paper_candidate",
        help="Report-only paper_table_candidate writer from an existing eval output directory",
    )
    paper_candidate_parser.add_argument(
        "--source_eval_dir",
        required=True,
        help="Existing eval output directory containing metrics.json/resolved_config.json/metric_plan.json",
    )
    paper_candidate_parser.add_argument(
        "--output_dir",
        required=True,
        help="New directory for paper_table_candidate.csv and readiness artifacts",
    )
    paper_candidate_parser.add_argument(
        "--method",
        default=None,
        help="Optional method name for the candidate row; defaults to empty/null.",
    )
    paper_candidate_parser.set_defaults(func=run_report_paper_candidate)

    doctor_parser = subparsers.add_parser("doctor_resources", help="Check ResourceConfig paths and lightweight providers")
    doctor_parser.add_argument("--resource_config", default=None, help="ResourceConfig YAML/JSON path")
    doctor_parser.add_argument("--pretrained_root", default=None, help="Override pretrained_root")
    doctor_parser.set_defaults(func=run_doctor_resources)

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark profile/index/materialization utilities")
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_action", required=True)

    benchmark_validate = benchmark_subparsers.add_parser("profile_validate", help="Validate a benchmark profile against JSONL rows")
    benchmark_validate.add_argument("--benchmark", required=True, help="Benchmark JSONL path")
    benchmark_validate.add_argument("--benchmark_profile", default=None, help="Benchmark profile YAML/JSON")
    benchmark_validate.add_argument("--benchmark_root", default=None, help="Root for relative benchmark media paths")
    benchmark_validate.add_argument("--output_dir", required=True, help="Directory for validation artifacts")
    benchmark_validate.set_defaults(func=run_benchmark)

    benchmark_index = benchmark_subparsers.add_parser("index", help="Build profile-aware benchmark_index.json/csv")
    benchmark_index.add_argument("--benchmark", required=True, help="Benchmark JSONL path")
    benchmark_index.add_argument("--benchmark_profile", default=None, help="Benchmark profile YAML/JSON")
    benchmark_index.add_argument("--benchmark_root", default=None, help="Root for relative benchmark media paths")
    benchmark_index.add_argument("--output_dir", required=True, help="Directory for benchmark index artifacts")
    benchmark_index.add_argument("--derive_missing_keys", type=parse_bool, default=True, help="Derive benchmark_key when row has no explicit key")
    benchmark_index.set_defaults(func=run_benchmark)

    benchmark_materialize = benchmark_subparsers.add_parser("materialize", help="Create a planned materialized benchmark manifest")
    benchmark_materialize.add_argument("--benchmark", required=True, help="Benchmark JSONL path")
    benchmark_materialize.add_argument("--benchmark_profile", default=None, help="Benchmark profile YAML/JSON")
    benchmark_materialize.add_argument("--benchmark_root", default=None, help="Root for relative benchmark media paths")
    benchmark_materialize.add_argument("--output_dir", required=True, help="Directory for planned materialization artifacts")
    benchmark_materialize.add_argument("--count", type=int, default=3, help="Number of rows to include in dry-run manifest")
    benchmark_materialize.add_argument("--dry_run", action="store_true", help="Required; do not invoke ffmpeg or copy media")
    benchmark_materialize.add_argument("--generation_width", type=int, default=480)
    benchmark_materialize.add_argument("--generation_height", type=int, default=832)
    benchmark_materialize.add_argument("--generation_fps", type=int, default=20)
    benchmark_materialize.add_argument("--generation_frame_num", type=int, default=81)
    benchmark_materialize.add_argument("--gt_clip_policy", default="cover_center_crop")
    benchmark_materialize.set_defaults(func=run_benchmark)

    splits_parser = subparsers.add_parser("splits", help="Generate or validate official benchmark splits")
    splits_subparsers = splits_parser.add_subparsers(dest="splits_action", required=True)
    splits_generate = splits_subparsers.add_parser("generate", help="Generate official split files")
    splits_generate.add_argument("--benchmark", default="benchmark/benchmark.jsonl", help="Benchmark JSONL path")
    splits_generate.add_argument("--benchmark_profile", default=None, help="Benchmark profile YAML/JSON")
    splits_generate.add_argument("--benchmark_root", default=None, help="Root for relative benchmark media paths")
    splits_generate.add_argument("--split_dir", default="benchmark/splits", help="Directory for split JSON files")
    splits_generate.add_argument("--splits", nargs="+", default=list(OFFICIAL_SPLITS), choices=list(OFFICIAL_SPLITS))
    splits_generate.set_defaults(func=run_splits)

    splits_validate = splits_subparsers.add_parser("validate", help="Validate official split files")
    splits_validate.add_argument("--splits", nargs="+", default=["official_smoke_10.v1", "official_dev_50.v1"], help="Split ids or paths")
    splits_validate.add_argument("--split_dir", default="benchmark/splits")
    splits_validate.add_argument("--output_dir", default="eval_outputs/official_split_validation_v1")
    splits_validate.set_defaults(func=run_splits)

    prepare_parser = subparsers.add_parser("prepare_resources", help="Compatibility entry; use scripts/prepare_eval_agent_pretrained.py for execution")
    prepare_parser.set_defaults(func=run_planned_command)
    for planned in ["infer_eval", "video_dir_eval", "promote_paper_table"]:
        planned_parser = subparsers.add_parser(planned, help=f"Planned public CLI surface for {planned}")
        planned_parser.set_defaults(func=run_planned_command)

    compare_parser = subparsers.add_parser("compare", help="Cross-run comparison utilities")
    compare_subparsers = compare_parser.add_subparsers(dest="compare_action", required=True)
    compare_dashboard = compare_subparsers.add_parser("dashboard", help="Build a static experiment/checkpoint/case comparison dashboard")
    compare_dashboard.add_argument("--eval_dir", action="append", default=[], help="Eval output directory containing metrics.json. Can be repeated.")
    compare_dashboard.add_argument(
        "--run_spec",
        action="append",
        default=[],
        help="Explicit run metadata in EXPERIMENT|CHECKPOINT|EVAL_DIR form. Can be repeated.",
    )
    compare_dashboard.add_argument("--output_dir", required=True)
    compare_dashboard.add_argument("--title", default="Digital Human Eval Compare Dashboard")
    compare_dashboard.add_argument("--link_mode", choices=["relative", "file_uri"], default="relative")
    compare_dashboard.set_defaults(func=run_compare)

    workflows_parser = subparsers.add_parser("workflows", help="Higher-level workflow entrypoints")
    workflows_subparsers = workflows_parser.add_subparsers(dest="workflow_action", required=True)
    for workflow_name in ["akd_pck_preflight", "body18_validation", "paper_review_package"]:
        workflow_parser = workflows_subparsers.add_parser(workflow_name, help=f"Planned workflow: {workflow_name}")
        workflow_parser.set_defaults(func=run_planned_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_metrics:
        print(json.dumps(registry_summary(), ensure_ascii=False, indent=2))
        return 0
    if args.list_presets:
        print(json.dumps(preset_summary(), ensure_ascii=False, indent=2))
        return 0
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
