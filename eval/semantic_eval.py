# grpo_single_policy_prm/eval/semantic_eval.py
from __future__ import annotations

from typing import Dict, Any, List

from ..metrics.semantic_entropy import log_semantic_diversity_rows


def semantic_eval_on_current_batch(
    writer,
    run_id: str,
    step: int,
    prompt_id: str,
    answers: List[str],
    decoding_params: Dict[str, Any],
) -> None:
    """
    Thin wrapper to log semantic diversity on current-batch generations.
    """
    log_semantic_diversity_rows(
        writer=writer,
        run_id=run_id,
        step=step,
        prompt_id=prompt_id,
        answers=answers,
        decoding_params=decoding_params,
    )
