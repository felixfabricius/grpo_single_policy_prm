# grpo_single_policy_prm/data/loaders/math.py
from __future__ import annotations

from dataclasses import dataclass

from ._common import _PathLoaderBase


@dataclass
class MATHLoader(_PathLoaderBase):
    dataset_name: str = "math"
    env_var: str = "DATA_MATH"
