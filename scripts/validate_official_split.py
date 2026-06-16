#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate versioned official benchmark split files.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["official_smoke_10.v1", "official_dev_50.v1"],
        help="Split ids or split JSON paths.",
    )
    parser.add_argument("--split_dir", default="benchmark/splits")
    parser.add_argument("--output_dir", default="eval_outputs/official_split_validation_v1")
    args = parser.parse_args()
    warnings.warn(
        "scripts/validate_official_split.py is deprecated; use "
        "auto_eval/eval_digital_human_agent.py splits validate",
        DeprecationWarning,
        stacklevel=2,
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "auto_eval/eval_digital_human_agent.py"),
        "splits",
        "validate",
        "--split_dir",
        args.split_dir,
        "--output_dir",
        args.output_dir,
        "--splits",
        *args.splits,
    ]
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
