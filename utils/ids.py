# grpo_single_policy_prm/utils/ids.py
from __future__ import annotations

import time
from typing import Optional


def run_id(prefix: str = "run") -> str:
    """
    Make a run id like: run_2025-09-04_15-12-33
    """
    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    return f"{prefix}_{ts}"


def short_answer_id(example_id: str, i: int) -> str:
    return f"{example_id}#{i}"
