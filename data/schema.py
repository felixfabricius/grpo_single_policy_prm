# grpo_single_policy_prm/data/schema.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Example:
    """
    Unified training example schema.
    """
    example_id: str
    question: str
    answer: str  # gold string for outcome grading
    meta: Dict[str, Any] = field(default_factory=dict)
