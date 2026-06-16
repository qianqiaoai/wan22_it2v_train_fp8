#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_agent.splits import OFFICIAL_SPLITS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate versioned official benchmark split files.")
    parser.add_argument("--benchmark", default="benchmark/benchmark.jsonl", help="Benchmark JSONL path.")
    parser.add_argument("--split_dir", default="benchmark/splits", help="Directory for official split files.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(OFFICIAL_SPLITS),
        choices=list(OFFICIAL_SPLITS),
        help="Official split ids to generate.",
    )
    args = parser.parse_args()
    warnings.warn(
        "scripts/generate_official_splits.py is deprecated; use "
        "auto_eval/eval_digital_human_agent.py splits generate",
        DeprecationWarning,
        stacklevel=2,
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "auto_eval/eval_digital_human_agent.py"),
        "splits",
        "generate",
        "--benchmark",
        args.benchmark,
        "--split_dir",
        args.split_dir,
        "--splits",
        *args.splits,
    ]
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
