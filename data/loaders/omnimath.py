# grpo_single_policy_prm/data/loaders/omnimath.py
from __future__ import annotations

from dataclasses import dataclass

from ._common import _PathLoaderBase


@dataclass
class OmniMATHLoader(_PathLoaderBase):
    dataset_name: str = "omnimath"
    env_var: str = "DATA_OMNIMATH"
