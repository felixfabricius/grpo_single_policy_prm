# grpo_single_policy_prm/utils/token_map.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional

from transformers import PreTrainedTokenizerBase


@dataclass
class TokenCharSpan:
    """Character span for a *single decoded token* within the full decoded answer string."""
    start: int
    end: int  # exclusive


def decode_tokens_with_spans(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Sequence[int],
) -> Tuple[str, List[TokenCharSpan]]:
    """
    Robustly reconstruct the *answer string* and per-token character spans directly from token IDs.

    We decode each token ID individually (skip special tokens, no cleanup),
    concatenate in order, and track cumulative character positions. This avoids
    drift that can arise from re-encoding text with the fast tokenizer offsets.

    NOTE: This method assumes the same decode parameters used for the full answer:
      - skip_special_tokens=True
      - clean_up_tokenization_spaces=False

    Returns:
        (answer_text, spans), where 'spans[i]' is the [start,end) char span of token_ids[i].
    """
    pieces: List[str] = []
    spans: List[TokenCharSpan] = []
    cursor = 0
    for tid in token_ids:
        piece = tokenizer.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pieces.append(piece)
        start = cursor
        end = start + len(piece)
        spans.append(TokenCharSpan(start=start, end=end))
        cursor = end
    answer_text = "".join(pieces)
    return answer_text, spans


def step_char_to_token_spans(
    step_char_spans: List[Tuple[int, int]],
    token_char_spans: List[TokenCharSpan],
) -> List[Tuple[int, int]]:
    """
    Map step character spans to token index ranges [tok_start, tok_end) over the *answer tokens*.

    Coverage rule:
      - A token is included in a step if it overlaps the step's char span by at least 1 character.
      - Step token spans are contiguous and non-overlapping if step_char_spans are non-overlapping.

    Args:
        step_char_spans: list of [start,end) over the answer string.
        token_char_spans: per-token [start,end) over the same string.

    Returns:
        List of [tok_start, tok_end) for each step span.
    """
    result: List[Tuple[int, int]] = []
    n_tokens = len(token_char_spans)
    t_idx = 0
    for (s_start, s_end) in step_char_spans:
        # Move t_idx to the first token that could overlap s_start
        while t_idx < n_tokens and token_char_spans[t_idx].end <= s_start:
            t_idx += 1
        tok_start = t_idx
        tok_end = tok_start
        # Extend while overlapping the step span
        while tok_end < n_tokens and token_char_spans[tok_end].start < s_end:
            tok_end += 1
        result.append((tok_start, tok_end))
        # next step continues from tok_end (since step spans are ordered / disjoint)
        t_idx = tok_end
    return result
