# grpo_single_policy_prm/utils/logging.py
from __future__ import annotations

import io
import json
import os
import time
from typing import Any, Dict, Optional


class JSONLWriter:
    """
    Append-only JSONL writer with atomic-like write semantics (best-effort).
    Creates parent directories if needed.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, obj: Dict[str, Any]) -> None:
        # Ensure simple, compact encoding
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        # Best-effort atomic append
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
