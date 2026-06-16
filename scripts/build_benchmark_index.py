#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_agent.binding import build_benchmark_index, load_jsonl, resolve_strict_binding


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str], *, stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess[str]:
    write_text(stdout_path.with_suffix(".cmd.txt"), shlex.join(command))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    write_text(stdout_path, completed.stdout)
    write_text(stderr_path, completed.stderr)
    return completed


def probe_exists(path: str | Path | None) -> bool:
    return bool(path) and Path(str(path)).exists()


def index_csv_rows(index: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in index.get("entries", []):
        metadata = entry.get("metadata") or {}
        collision = entry.get("collision_check") or {}
        rows.append(
            {
                "benchmark_key": entry.get("benchmark_key") or entry.get("key"),
                "key_source": entry.get("key_source"),
                "row_index": entry.get("row_index"),
                "row_hash": entry.get("row_hash"),
                "imgpath": entry.get("imgpath"),
                "videopath": entry.get("videopath"),
                "wav_path": entry.get("wav_path"),
                "posepath": entry.get("posepath"),
                "language": metadata.get("language"),
                "gender": metadata.get("gender"),
                "type": metadata.get("type"),
                "category": metadata.get("category"),
                "body": metadata.get("body"),
                "collision_status": "collision" if collision.get("explicit_key_collision") else "ok",
                "media_warnings": ";".join(entry.get("media_warnings") or []),
            }
        )
    return rows


def selected_sample_rows(index: dict[str, Any], *, count: int, selection_mode: str, selection_seed: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in index.get("entries", [])[:count]:
        row = {
            "benchmark_key": entry.get("benchmark_key") or entry.get("key"),
            "benchmark_row_index": entry.get("row_index"),
            "benchmark_row_hash": entry.get("row_hash"),
            "selection_mode": selection_mode,
            "selection_seed": selection_seed,
            "imgpath": entry.get("imgpath"),
            "videopath": entry.get("videopath"),
            "wav_path": entry.get("wav_path"),
            "posepath": entry.get("posepath"),
            "metadata": entry.get("metadata"),
            "media_warnings": entry.get("media_warnings") or [],
        }
        rows.append(row)
    return rows


def materialize_one(
    selected: dict[str, Any],
    *,
    output_dir: Path,
    width: int,
    height: int,
    fps: int,
    frame_num: int,
    crop_policy: str,
    materialize: bool,
) -> dict[str, Any]:
    benchmark_key = str(selected["benchmark_key"])
    mat_root = output_dir / "benchmark_materialized"
    target_path = mat_root / "target_clips" / f"{benchmark_key}.mp4"
    ref_path = mat_root / "refs" / f"{benchmark_key}.png"
    audio_path = mat_root / "audio_segments" / f"{benchmark_key}.wav"
    logs_dir = mat_root / "logs" / benchmark_key
    duration_sec = frame_num / float(fps)
    status = "planned"
    warnings: list[str] = []
    commands: dict[str, str] = {}

    target_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(selected["videopath"]),
        "-vf",
        (
            f"fps={fps},"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"trim=start_frame=0:end_frame={frame_num},"
            "setpts=PTS-STARTPTS"
        ),
        "-frames:v",
        str(frame_num),
        "-an",
        "-pix_fmt",
        "yuv420p",
        str(target_path),
    ]
    ref_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(target_path),
        "-frames:v",
        "1",
        str(ref_path),
    ]
    audio_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        "0",
        "-i",
        str(selected["wav_path"]),
        "-t",
        f"{duration_sec:.6f}",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    commands["target"] = shlex.join(target_cmd)
    commands["ref"] = shlex.join(ref_cmd)
    commands["audio"] = shlex.join(audio_cmd)

    if materialize:
        for path in (target_path.parent, ref_path.parent, audio_path.parent, logs_dir):
            path.mkdir(parents=True, exist_ok=True)
        steps = [
            ("target", target_cmd),
            ("ref", ref_cmd),
            ("audio", audio_cmd),
        ]
        failed = False
        for name, command in steps:
            completed = run_command(
                command,
                stdout_path=logs_dir / f"{name}_stdout.txt",
                stderr_path=logs_dir / f"{name}_stderr.txt",
            )
            if completed.returncode != 0:
                failed = True
                message = completed.stderr.strip() or completed.stdout.strip() or f"{name} materialization failed"
                warnings.append(f"{name}_materialization_failed: {message}")
                break
        if failed:
            status = "failed"
        elif probe_exists(target_path) and probe_exists(ref_path) and probe_exists(audio_path):
            status = "actual_materialized"
        else:
            status = "failed"
            if not probe_exists(target_path):
                warnings.append("materialized_target_clip_missing")
            if not probe_exists(ref_path):
                warnings.append("materialized_ref_missing")
            if not probe_exists(audio_path):
                warnings.append("materialized_audio_missing")

    return {
        "key": benchmark_key,
        "benchmark_key": benchmark_key,
        "benchmark_row_index": selected["benchmark_row_index"],
        "benchmark_row_hash": selected["benchmark_row_hash"],
        "materialized_ref": str(ref_path),
        "materialized_target_clip": str(target_path),
        "materialized_audio": str(audio_path),
        "original_imgpath": selected.get("imgpath"),
        "original_videopath": selected.get("videopath"),
        "original_wav_path": selected.get("wav_path"),
        "original_posepath": selected.get("posepath"),
        "gt_clip_policy": crop_policy,
        "gt_clip_start_sec": 0.0,
        "gt_clip_end_sec": duration_sec,
        "generation_width": width,
        "generation_height": height,
        "generation_fps": fps,
        "generation_frame_num": frame_num,
        "materialization_status": status,
        "materialization_commands": commands,
        "materialization_warnings": warnings,
        "conditioning": {
            "reference_mode": "materialized_ref_from_target_first_frame",
            "audio_mode": "materialized_audio",
        },
    }


def future_inference_rows(materialized_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(materialized_rows):
        benchmark_key = row["benchmark_key"]
        rows.append(
            {
                "key": f"generated_{index + 1:04d}",
                "generated_sample_id": f"generated_{index + 1:04d}",
                "benchmark_key": benchmark_key,
                "benchmark_row_hash": row.get("benchmark_row_hash"),
                "benchmark_row_index": row.get("benchmark_row_index"),
                "prepared_ref": row.get("materialized_ref"),
                "audio_segment": row.get("materialized_audio"),
                "generated_video": None,
                "video": None,
                "materialized_target_clip": row.get("materialized_target_clip"),
                "materialized_audio": row.get("materialized_audio"),
                "conditioning": row.get("conditioning"),
                "generation_width": row.get("generation_width"),
                "generation_height": row.get("generation_height"),
                "generation_fps": row.get("generation_fps"),
                "generation_frame_num": row.get("generation_frame_num"),
            }
        )
    return rows


def strict_binding_table(
    inference_rows: list[dict[str, Any]],
    benchmark_index: dict[str, Any],
    materialized_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in inference_rows:
        result = resolve_strict_binding(row, benchmark_index, materialized_rows=materialized_rows)
        benchmark_entry = result.benchmark_entry or {}
        materialized_entry = result.materialized_entry or {}
        materialization_status = materialized_entry.get("materialization_status")
        status = result.binding_status
        if status == "validated" and materialization_status == "planned":
            status = "planned_linkage_ok"
        rows.append(
            {
                "generated_sample_id": row.get("generated_sample_id") or row.get("key"),
                "generated_video_path": row.get("generated_video") or row.get("video"),
                "inference_key": row.get("key"),
                "benchmark_key": row.get("benchmark_key"),
                "resolved_key": result.key,
                "binding_source": result.source,
                "benchmark_row_index": benchmark_entry.get("row_index"),
                "benchmark_row_hash": benchmark_entry.get("row_hash"),
                "materialized_key": materialized_entry.get("benchmark_key") or materialized_entry.get("key"),
                "materialized_target_clip": materialized_entry.get("materialized_target_clip"),
                "materialized_audio": materialized_entry.get("materialized_audio"),
                "materialization_status": materialization_status,
                "binding_policy": result.binding_policy,
                "binding_status": status,
                "binding_warning": result.warning,
                "error_code": (result.error or {}).get("code"),
                "is_debug_binding": False,
            }
        )
    return rows


def write_binding_report(path: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["binding_status"]] = counts.get(row["binding_status"], 0) + 1
    text = f"""# Strict Binding Report After Benchmark Key Materialization

## Summary

- Samples: `{len(rows)}`
- Validated: `{counts.get('validated', 0)}`
- Planned linkage OK: `{counts.get('planned_linkage_ok', 0)}`
- Blocked: `{counts.get('blocked', 0)}`

## Result

Strict binding uses `benchmark_key` propagated through selected samples, materialized benchmark manifest, and future inference manifest rows. It does not use numeric row fallback.

## Per-sample Results

{chr(10).join(f"- `{row['benchmark_key']}`: `{row['binding_status']}` via `{row['binding_source']}`; warning={row['binding_warning']}" for row in rows)}
"""
    write_text(path, text)


def write_inference_patch_plan(path: Path) -> None:
    text = """# Inference Manifest Key Patch Plan

## Required Fields

Future inference manifests must write:

- `generated_sample_id`
- `benchmark_key`
- `benchmark_row_hash`
- `benchmark_row_index`
- `prepared_ref`
- `audio_segment`
- `generated_video`
- `materialized_target_clip`
- `materialized_audio`
- `conditioning.reference_mode`
- `conditioning.audio_mode`
- generation `width`, `height`, `fps`, and `frame_num`

## Where To Write Keys

`infer_eval` should first build `selected_samples.jsonl` and `benchmark_materialized/manifest.jsonl`. The inference job loader should consume the materialized manifest and propagate `benchmark_key`, `benchmark_row_hash`, and materialized paths into every generated manifest row.

The generated output `key` may remain a run-local output id. It must not replace `benchmark_key`.

## Manifest Eval Verification

`manifest_eval` should fail strict paired metrics when:

- `benchmark_key` is missing
- `benchmark_key` does not resolve uniquely in `benchmark_index.json`
- `benchmark_row_hash` differs from the benchmark index
- `materialized_target_clip` or `materialized_audio` is missing for paired metrics that require them

## Old Runs

Existing generated manifests must not be edited in place. Old runs without `benchmark_key` remain `debug_only` for paired diagnostics and `not_for_paper`.
"""
    write_text(path, text)


def write_gate(path: Path, *, index: dict[str, Any], strict_rows: list[dict[str, Any]], materialize: bool) -> None:
    collision = index["collision_check"]
    validated = sum(1 for row in strict_rows if row["binding_status"] == "validated")
    planned = sum(1 for row in strict_rows if row["binding_status"] == "planned_linkage_ok")
    blocked = sum(1 for row in strict_rows if row["binding_status"] == "blocked")
    key_established = collision["collision_count"] == 0 and collision["key_count"] == collision["row_count"]
    strict_pass = blocked == 0 and (validated + planned == len(strict_rows))
    text = f"""# AKD/PCK Gate After Benchmark Key Materialization

## Key Chain

- Benchmark key established for all rows: `{key_established}`
- Benchmark rows: `{collision['row_count']}`
- Key count: `{collision['key_count']}`
- Derived key count: `{collision['derived_key_count']}`
- Collision count: `{collision['collision_count']}`

## Selected Samples / Materialized Manifest

- Selected samples carry `benchmark_key`: `true`
- Materialized manifest carries `benchmark_key`: `true`
- Actual materialization requested: `{materialize}`
- Strict binding pass: `{strict_pass}`
- Strict binding validated: `{validated}/{len(strict_rows)}`
- Planned linkage OK: `{planned}/{len(strict_rows)}`
- Blocked: `{blocked}/{len(strict_rows)}`

## Remaining Gate

- Need to rerun inference with the key-preserving materialized manifest: `true`
- Formal AKD/PCK allowed now: `false`

Formal AKD/PCK remains blocked because the current generated inference manifests do not yet carry `benchmark_key`, and body18 mapping is still `provisional`. The next step is key-preserving `infer_eval` smoke, then body18 final validation.
"""
    write_text(path, text)


def missing_key_simulation(output_dir: Path, index: dict[str, Any], materialized_rows: list[dict[str, Any]]) -> None:
    payload = {
        "missing_key": resolve_strict_binding({"video": "dummy.mp4"}, index, materialized_rows=materialized_rows).to_dict(),
        "wrong_key": resolve_strict_binding(
            {"key": "generated_without_benchmark_key"}, index, materialized_rows=materialized_rows
        ).to_dict(),
    }
    write_json(output_dir / "missing_key_simulation.json", payload)


def main() -> int:
    warnings.warn(
        "scripts/build_benchmark_index.py is a legacy compatibility workflow. "
        "Prefer scripts/eval_digital_human_agent.py benchmark index/materialize for new Skill usage.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(description="Build benchmark keys and materialized benchmark manifest.")
    parser.add_argument("--benchmark", default="benchmark/benchmark.jsonl")
    parser.add_argument("--output_dir", default="eval_outputs/benchmark_key_materialization_v1")
    parser.add_argument("--select_count", type=int, default=3)
    parser.add_argument("--selection_mode", default="first_n")
    parser.add_argument("--selection_seed", type=int, default=None)
    parser.add_argument("--generation_width", type=int, default=480)
    parser.add_argument("--generation_height", type=int, default=832)
    parser.add_argument("--generation_fps", type=int, default=20)
    parser.add_argument("--generation_frame_num", type=int, default=81)
    parser.add_argument("--gt_clip_policy", default="cover_center_crop")
    parser.add_argument("--planned_only", action="store_true")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    output_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_rows = load_jsonl((repo_root / args.benchmark).resolve() if not Path(args.benchmark).is_absolute() else args.benchmark)
    benchmark_index = build_benchmark_index(benchmark_rows, derive_missing_keys=True)
    write_json(output_dir / "benchmark_index.json", benchmark_index)
    index_rows = index_csv_rows(benchmark_index)
    write_csv(
        output_dir / "benchmark_index.csv",
        index_rows,
        fieldnames=[
            "benchmark_key",
            "key_source",
            "row_index",
            "row_hash",
            "imgpath",
            "videopath",
            "wav_path",
            "posepath",
            "language",
            "gender",
            "type",
            "category",
            "body",
            "collision_status",
            "media_warnings",
        ],
    )

    selected = selected_sample_rows(
        benchmark_index,
        count=args.select_count,
        selection_mode=args.selection_mode,
        selection_seed=args.selection_seed,
    )
    write_jsonl(output_dir / "selected_samples.jsonl", selected)

    materialized_rows = [
        materialize_one(
            row,
            output_dir=output_dir,
            width=args.generation_width,
            height=args.generation_height,
            fps=args.generation_fps,
            frame_num=args.generation_frame_num,
            crop_policy=args.gt_clip_policy,
            materialize=not args.planned_only,
        )
        for row in selected
    ]
    write_jsonl(output_dir / "benchmark_materialized" / "manifest.jsonl", materialized_rows)

    future_rows = future_inference_rows(materialized_rows)
    write_jsonl(output_dir / "future_inference_manifest_example.jsonl", future_rows)
    strict_rows = strict_binding_table(future_rows, benchmark_index, materialized_rows)
    write_csv(
        output_dir / "target_binding_table_strict.csv",
        strict_rows,
        fieldnames=[
            "generated_sample_id",
            "generated_video_path",
            "inference_key",
            "benchmark_key",
            "resolved_key",
            "binding_source",
            "benchmark_row_index",
            "benchmark_row_hash",
            "materialized_key",
            "materialized_target_clip",
            "materialized_audio",
            "materialization_status",
            "binding_policy",
            "binding_status",
            "binding_warning",
            "error_code",
            "is_debug_binding",
        ],
    )
    write_binding_report(output_dir / "target_binding_report_strict.md", strict_rows)
    write_inference_patch_plan(output_dir / "inference_manifest_key_patch_plan.md")
    write_gate(
        output_dir / "akd_pck_gate_after_key_materialization.md",
        index=benchmark_index,
        strict_rows=strict_rows,
        materialize=not args.planned_only,
    )
    missing_key_simulation(output_dir, benchmark_index, materialized_rows)
    summary = {
        "benchmark_index": str(output_dir / "benchmark_index.json"),
        "benchmark_index_csv": str(output_dir / "benchmark_index.csv"),
        "selected_samples": str(output_dir / "selected_samples.jsonl"),
        "benchmark_materialized_manifest": str(output_dir / "benchmark_materialized" / "manifest.jsonl"),
        "future_inference_manifest_example": str(output_dir / "future_inference_manifest_example.jsonl"),
        "target_binding_table_strict": str(output_dir / "target_binding_table_strict.csv"),
        "target_binding_report_strict": str(output_dir / "target_binding_report_strict.md"),
        "gate": str(output_dir / "akd_pck_gate_after_key_materialization.md"),
        "collision_check": benchmark_index["collision_check"],
        "strict_binding_status_counts": {
            status: sum(1 for row in strict_rows if row["binding_status"] == status)
            for status in sorted({row["binding_status"] for row in strict_rows})
        },
        "materialization_status_counts": {
            status: sum(1 for row in materialized_rows if row["materialization_status"] == status)
            for status in sorted({row["materialization_status"] for row in materialized_rows})
        },
    }
    write_json(output_dir / "benchmark_key_materialization_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
