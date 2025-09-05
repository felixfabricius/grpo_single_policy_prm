# grpo_single_policy_prm/data/loaders/_common.py
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, List

from ..schema import Example


@dataclass
class _PathLoaderBase:
    """
    Base class for local-path dataset loaders.

    Behavior contract (per your spec):
    - If path/env var is missing:
        * emit a clear WARNING: "Path not set — use jsonl_fallback"
        * if 'strict' is True (user explicitly selected this dataset) -> RAISE
        * else -> mark unavailable and skip in the mixer
    - We assume a local JSONL file layout like:
        <path>/{train,valid,test}.jsonl   OR a single file per split.
    """
    dataset_name: str
    env_var: str
    path: Optional[str] = None
    strict: bool = False  # if True and path missing -> raise

    def _resolve_path(self) -> Optional[str]:
        # Precedence: explicit path > env var
        if self.path is not None:
            return self.path
        env = os.getenv(self.env_var)
        return env

    def is_available(self) -> bool:
        p = self._resolve_path()
        if not p:
            warnings.warn(f"[{self.dataset_name}] Path not set — use jsonl_fallback (env {self.env_var})")
            return False
        return True

    def _split_candidates(self, root: str, split: str) -> List[str]:
        # Try common names first
        cand = [
            os.path.join(root, f"{split}.jsonl"),
            os.path.join(root, f"{self.dataset_name}_{split}.jsonl"),
        ]
        # If a direct file path was given instead of a dir, use it as-is
        if os.path.isfile(root):
            cand = [root]
        return [c for c in cand if os.path.exists(c)]

    def prepare(self, split: str) -> Iterable[Example]:
        root = self._resolve_path()
        if not root:
            msg = f"[{self.dataset_name}] Path not set — use jsonl_fallback (env {self.env_var})"
            if self.strict:
                raise FileNotFoundError(msg)
            else:
                warnings.warn(msg)
                return iter([])  # empty iterator

        files = self._split_candidates(root, split)
        if not files:
            msg = f"[{self.dataset_name}] No files found for split '{split}' under '{root}'"
            if self.strict:
                raise FileNotFoundError(msg)
            else:
                warnings.warn(msg)
                return iter([])

        def _iter_files():
            for p in files:
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
        return _iter_files()
