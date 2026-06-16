#!/usr/bin/env python
from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "auto_eval/eval_agent/workflows/compare_dashboard_workflow.py"

if __name__ == "__main__":
    warnings.warn(
        f"{Path(__file__).name} is deprecated at scripts/; use "
        "python auto_eval/eval_digital_human_agent.py compare dashboard ... instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")
