# grpo_single_policy_prm/data/loaders/jsonl_generic.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional

from ..schema import Example


@dataclass
class JSONLGenericLoader:
    """
    Minimal JSONL loader for smoke tests or custom dumps.

    Expected JSONL schema per line:
      {
        "example_id": str,
        "question": str,
        "answer": str,
        "meta": {...}  # optional
      }

    You can point this to one or more files; .prepare(split) will select files whose
    basename contains the split (e.g., "train", "valid", "test"). If none match, it
    will use all files.
    """
    paths: List[str]

    def _select_paths_for_split(self, split: str) -> List[str]:
        # If any file name mentions the split, use those; else use all.
        selected = [p for p in self.paths if split.lower() in os.path.basename(p).lower()]
        return selected if selected else list(self.paths)

    def prepare(self, split: str) -> Iterable[Example]:
        use_paths = self._select_paths_for_split(split)
        for p in use_paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    yield Example(
                        example_id=str(obj["example_id"]),
                        question=str(obj["question"]),
                        answer=str(obj["answer"]),
                        meta=dict(obj.get("meta", {})),
                    )
