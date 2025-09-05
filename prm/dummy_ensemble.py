# grpo_single_policy_prm/prm/dummy_ensemble.py
from __future__ import annotations

import math
import random
from typing import Sequence, Dict, Any, List

from .interface import PRMEnsemble, PRMOutputs


class DummyPRMEnsemble(PRMEnsemble):
    """
    A stub PRM ensemble that simulates M members by producing
    per-step probabilities with controlled disagreement.

    Use for smoke tests when a real PRM isn't wired yet.
    """

    def __init__(self, M: int = 2, seed: int = 123):
        assert M >= 1, "Ensemble size M must be >= 1"
        self.M = M
        self._rng = random.Random(seed)

    def _member_prob(self, base: float, jitter: float) -> float:
        # clamp to [0,1]
        return max(0.0, min(1.0, base + self._rng.uniform(-jitter, jitter)))

    def score_steps_in_context(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        step_texts_per_answer: Sequence[Sequence[str]],
        meta: Sequence[Dict[str, Any]] | None = None,
    ) -> PRMOutputs:
        probs_per_answer: List[List[float]] = []
        # For each answer, create M synthetic member predictions per step,
        # then average across members to yield the ensemble mean.
        for steps in step_texts_per_answer:
            T = len(steps)
            if T == 0:
                probs_per_answer.append([])
                continue
            # Base rises slightly with step index; jitter controls disagreement.
            base_line = [0.45 + 0.1 * (t / max(1, T - 1)) for t in range(T)]
            jitter = 0.15
            # Simulate M members
            member_probs = [
                [self._member_prob(b, jitter) for b in base_line] for _ in range(self.M)
            ]
            # Ensemble mean per step
            mean_per_step = [sum(col) / self.M for col in zip(*member_probs)]
            probs_per_answer.append(mean_per_step)

        return PRMOutputs(probs_per_answer=probs_per_answer, info={"M": self.M})
