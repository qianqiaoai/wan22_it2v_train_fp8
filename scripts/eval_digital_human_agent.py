#!/usr/bin/env python
from __future__ import annotations

import sys
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auto_eval.eval_digital_human_agent import main  # noqa: E402


if __name__ == "__main__":
    warnings.warn(
        "scripts/eval_digital_human_agent.py is deprecated for Skill workflows; "
        "use auto_eval/eval_digital_human_agent.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise SystemExit(main())
