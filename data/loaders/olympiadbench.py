# grpo_single_policy_prm/data/loaders/olympiadbench.py
from __future__ import annotations

from dataclasses import dataclass

from ._common import _PathLoaderBase


@dataclass
class OlympiadBenchLoader(_PathLoaderBase):
    dataset_name: str = "olympiad"
    env_var: str = "DATA_OLYMPIADB"
