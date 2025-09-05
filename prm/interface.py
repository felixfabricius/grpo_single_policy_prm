# grpo_single_policy_prm/prm/interface.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Dict, Any


@dataclass
class PRMOutputs:
    """
    Holds per-step probabilities for each (response) in the batch.

    shapes:
      - probs_per_answer: List[List[float]] of length = num_answers_in_batch.
        Each inner list has length = number of steps in that answer.
      - Optional debug/info dict for logging / inspection.
    """
    probs_per_answer: List[List[float]]
    info: Dict[str, Any] | None = None


class PRMEnsemble(ABC):
    """
    Abstract ensemble interface.
    Implementations should be *stateless w.r.t. parameters* (frozen inference),
    accept batched inputs, and return per-step probabilities in [0,1] for each answer.
    """

    @abstractmethod
    def score_steps_in_context(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        step_texts_per_answer: Sequence[Sequence[str]],
        # Optionally: any metadata the PRM needs to condition on
        meta: Sequence[Dict[str, Any]] | None = None,
    ) -> PRMOutputs:
        """
        Score steps in context for each (question, answer).

        Args:
            questions: list of question strings (len = B).
            answers: list of full answer strings (len = B).
            step_texts_per_answer: for each answer, a list of step strings in order.
            meta: optional list of dicts (len = B) to pass dataset/meta-info.

        Returns:
            PRMOutputs where probs_per_answer[b][t] is the probability in [0,1]
            for step t of answer b. If an answer has T steps, the inner list has length T.
        """
        raise NotImplementedError
