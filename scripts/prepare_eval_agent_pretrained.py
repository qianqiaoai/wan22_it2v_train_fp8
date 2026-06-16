#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_agent.resources import sha256_file


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_source_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise ValueError(f"source manifest must contain an assets list: {path}")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(assets):
        if not isinstance(item, dict):
            raise ValueError(f"asset entry {index} is not an object")
        asset_key = item.get("asset_key")
        source_path = item.get("source_path")
        target_relpath = item.get("target_relpath") or item.get("relpath")
        if not asset_key or not source_path or not target_relpath:
            raise ValueError(f"asset entry {index} must include asset_key/source_path/target_relpath")
        normalized.append(
            {
                "asset_key": str(asset_key),
                "source_path": str(source_path),
                "target_relpath": str(target_relpath),
            }
        )
    return normalized


def _plan_asset(asset: dict[str, Any], pretrained_root: Path) -> dict[str, Any]:
    source = Path(asset["source_path"]).expanduser().resolve()
    target = (pretrained_root / asset["target_relpath"]).resolve()
    try:
        target.relative_to(pretrained_root)
    except ValueError as exc:
        raise ValueError(f"target relpath escapes pretrained_root: {asset['target_relpath']}") from exc
    source_exists = source.exists()
    target_exists = target.exists()
    source_sha = sha256_file(source) if source_exists and source.is_file() else None
    target_sha = sha256_file(target) if target_exists and target.is_file() else None
    if target_exists and source_sha and target_sha == source_sha:
        action = "already_exists"
        note = "target exists and sha256 matches"
    elif target_exists and source_sha and target_sha and source_sha != target_sha:
        action = "conflict"
        note = "target exists with different sha256; refusing to overwrite"
    elif source_exists:
        action = "copy_or_symlink"
        note = "ready"
    else:
        action = "missing_source"
        note = "source path does not exist"
    return {
        "asset_key": asset["asset_key"],
        "source_path": str(source),
        "target_relpath": asset["target_relpath"],
        "target_path": str(target),
        "source_exists": source_exists,
        "target_exists": target_exists,
        "source_sha256": source_sha,
        "target_sha256": target_sha,
        "size": source.stat().st_size if source_exists and source.is_file() else None,
        "action": action,
        "note": note,
    }


def _execute_asset(plan: dict[str, Any], *, mode: str) -> dict[str, Any]:
    if plan["action"] == "already_exists":
        migration_mode = "already_exists"
    elif plan["action"] == "copy_or_symlink":
        source = Path(plan["source_path"])
        target = Path(plan["target_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copy2(source, target)
        elif mode == "symlink":
            os.symlink(source, target)
        else:
            raise ValueError(f"unsupported mode: {mode}")
        migration_mode = mode
        plan["target_exists"] = target.exists()
        plan["target_sha256"] = sha256_file(target) if target.is_file() else None
        if plan["source_sha256"] != plan["target_sha256"]:
            raise RuntimeError(f"post-{mode} sha256 mismatch for {plan['asset_key']}")
    else:
        raise RuntimeError(f"cannot execute asset {plan['asset_key']}: {plan['action']}")
    return {
        "asset_key": plan["asset_key"],
        "relpath": plan["target_relpath"],
        "sha256": plan["target_sha256"] or plan["source_sha256"],
        "size": plan["size"],
        "original_source_path": plan["source_path"],
        "migration_mode": migration_mode,
        "created_at": _now(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare eval_agent_pretrained assets from an explicit source manifest")
    parser.add_argument("--pretrained_root", required=True, help="Target pretrained root")
    parser.add_argument("--source_manifest", required=True, help="Local JSON manifest with source_path and target_relpath")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--dry_run", action="store_true", default=True, help="Default behavior; do not modify target")
    parser.add_argument("--execute", action="store_true", help="Actually copy or symlink assets")
    parser.add_argument("--write_manifest", action="store_true", help="Write MANIFEST.json after successful execute")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pretrained_root = Path(args.pretrained_root).expanduser().resolve()
    assets = _load_source_manifest(Path(args.source_manifest).expanduser().resolve())
    plans = [_plan_asset(asset, pretrained_root) for asset in assets]
    dry_run = not args.execute
    result: dict[str, Any] = {
        "created_at": _now(),
        "pretrained_root": str(pretrained_root),
        "source_manifest": str(Path(args.source_manifest).expanduser().resolve()),
        "mode": args.mode,
        "dry_run": dry_run,
        "assets": plans,
    }
    conflicts = [plan for plan in plans if plan["action"] == "conflict"]
    missing = [plan for plan in plans if plan["action"] == "missing_source"]
    if conflicts or missing:
        result["status"] = "blocked"
        result["conflicts"] = conflicts
        result["missing_sources"] = missing
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2
    if dry_run:
        result["status"] = "dry_run"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    manifest_entries = [_execute_asset(plan, mode=args.mode) for plan in plans]
    result["status"] = "executed"
    result["manifest_entries"] = manifest_entries
    if args.write_manifest:
        manifest = {
            "schema_version": 1,
            "created_at": _now(),
            "pretrained_root": str(pretrained_root),
            "assets": manifest_entries,
        }
        manifest_path = pretrained_root / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        result["manifest_path"] = str(manifest_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
