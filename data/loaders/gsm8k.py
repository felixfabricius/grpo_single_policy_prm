# grpo_single_policy_prm/data/loaders/gsm8k.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ._common import _PathLoaderBase
from ..schema import Example


@dataclass
class GSM8KLoader(_PathLoaderBase):
    """
    GSM8K local JSONL loader.

    Set path via:
      - env var DATA_GSM8K
      - or explicit path=...
    """
    dataset_name: str = "gsm8k"
    env_var: str = "DATA_GSM8K"
