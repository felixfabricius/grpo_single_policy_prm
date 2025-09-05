# grpo_single_policy_prm/grpo/sampler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import torch
from transformers import GenerationConfig

from ..models.policy_lora import PolicyWithLoRA
from ..rewards.steps import split_steps_by_newline, StepSplitResult
from ..utils.token_map import decode_tokens_with_spans, step_char_to_token_spans


@dataclass
class SamplerAnswer:
    answer_id: str                   # e.g., f"{example_id}#{i}"
    token_ids: List[int]             # answer-only token ids
    text: str                        # reconstructed from token pieces (deterministic)
    step_split: StepSplitResult      # steps + overflow flag
    step_token_spans: List[Tuple[int, int]]  # [(tok_start, tok_end)] for each step
    logp_old: torch.Tensor           # (T,) per-token log-probs under *current* policy at sampling time
    total_tokens: int                # len(token_ids)


@dataclass
class SamplerBatch:
    prompt_id: str
    prompt_text: str
    prompt_token_ids: List[int]
    answers: List[SamplerAnswer]
    decoding_params: Dict[str, Any]
    full_lengths: List[int]          # prompt+answer lengths for sanity
    N: int


class Sampler:
    """
    Online sampler: generate N answers for a single prompt and produce all
    spans and cached logp_old needed downstream.
    """

    def __init__(self, policy: PolicyWithLoRA, max_steps_per_response: int = 64):
        self.policy = policy
        self.max_steps = max_steps_per_response

    @torch.no_grad()
    def sample(
        self,
        example_id: str,
        prompt: str,
        N: int,
        gen_cfg: GenerationConfig,
        generator: Optional[torch.Generator] = None,
    ) -> SamplerBatch:
        assert N >= 2, "N must be >= 2 for mean-centering outcome rewards."
        answers_token_ids, prompt_input_ids, full_lengths = self.policy.generate_n(
            prompt=prompt, N=N, gen_cfg=gen_cfg, generator=generator
        )

        # Per-answer logp_old under *current* policy
        logp_old_list = self.policy.compute_logprobs_for_answers(
            prompt=prompt, answers_token_ids=answers_token_ids
        )

        answers: List[SamplerAnswer] = []
        for i, (ans_ids, lp_old) in enumerate(zip(answers_token_ids, logp_old_list)):
            # Reconstruct answer text & per-token char spans directly from token IDs
            ans_text, token_char_spans = decode_tokens_with_spans(self.policy.tokenizer, ans_ids)
            # Split into steps on this reconstructed text
            split = split_steps_by_newline(ans_text, max_steps_per_response=self.max_steps, collapse_blank=True)
            # Map step char spans -> token spans
            step_char_spans = [s.char_span for s in split.steps]
            step_tok_spans = step_char_to_token_spans(step_char_spans, token_char_spans)

            answers.append(
                SamplerAnswer(
                    answer_id=f"{example_id}#{i}",
                    token_ids=ans_ids,
                    text=ans_text,
                    step_split=split,
                    step_token_spans=step_tok_spans,
                    logp_old=lp_old.detach(),   # (T,)
                    total_tokens=len(ans_ids),
                )
            )

        decoding_params = dict(
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=gen_cfg.do_sample,
        )
        return SamplerBatch(
            prompt_id=example_id,
            prompt_text=prompt,
            prompt_token_ids=prompt_input_ids,
            answers=answers,
            decoding_params=decoding_params,
            full_lengths=full_lengths,
            N=N,
        )
