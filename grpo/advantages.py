# grpo_single_policy_prm/grpo/advantages.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Sequence, Literal, Dict, Any

import math
import numpy as np


Version = Literal["a", "b", "c"]


@dataclass
class OutcomeProcessStats:
    """
    Diagnostics for a single prompt-group of N responses.
    """
    N: int
    T_list: List[int]         # number of steps per response
    r_out: List[float]        # outcome rewards (0/1)
    A_out: List[float]        # mean-centered outcome advantages
    mu: float                 # mean over response-average step scores
    sigma: float              # std over response-average step scores
    overflow_mask: List[bool] # which responses exceeded max_steps cap (if tracked)


def mean_center_outcome(rewards: Sequence[float]) -> List[float]:
    N = len(rewards)
    if N == 0:
        return []
    mu = float(sum(rewards)) / N
    return [float(r) - mu for r in rewards]


def compute_Z_standardized(
    mean_step_scores_per_response: Sequence[Sequence[float]],
    eps: float = 1e-6,
) -> Tuple[List[List[float]], float, float]:
    """
    Given, for each response i, the per-step *mean* PRM score \bar{s}_{i,t},
    compute μ, σ across response-level averages \bar{r}_i and then Z_{i,t}.

    Returns:
        Z_per_response: same shape as input, with Z-scores per step.
        mu: float
        sigma: float
    """
    if len(mean_step_scores_per_response) == 0:
        return [], 0.0, 0.0

    avg_per_resp = []
    for s_list in mean_step_scores_per_response:
        if len(s_list) == 0:
            avg_per_resp.append(0.0)
        else:
            avg_per_resp.append(float(sum(s_list)) / len(s_list))
    mu = float(sum(avg_per_resp)) / len(avg_per_resp)
    # population std across responses
    var = float(sum((x - mu) ** 2 for x in avg_per_resp)) / max(1, len(avg_per_resp))
    sigma = math.sqrt(max(0.0, var))
    Z_per_response: List[List[float]] = []
    denom = (sigma + eps)
    for s_list in mean_step_scores_per_response:
        Z_per_response.append([(s - mu) / denom for s in s_list])
    return Z_per_response, mu, sigma


def broadcast_step_advantages_to_tokens(
    step_token_spans: Sequence[Tuple[int, int]],
    step_advantages: Sequence[float],
    total_tokens: int,
    apply_to_unspanned: Literal["omit", "reuse_last"] = "omit",
) -> np.ndarray:
    """
    Expand per-step advantages A_{i,t} to a vector over answer tokens.

    Args:
        step_token_spans: [(tok_start, tok_end)] for each step t
        step_advantages: A_{i,t} values (same length as step_token_spans)
        total_tokens: number of answer tokens
        apply_to_unspanned: how to treat tokens outside any step span.
            - "omit": zeros (i.e., only outcome term may remain)
            - "reuse_last": fill with last non-empty step's advantage

    Returns:
        np.ndarray of shape (total_tokens,) with per-token advantages.
    """
    adv = np.zeros((total_tokens,), dtype=np.float32)
    last_val: float | None = None
    for (span, val) in zip(step_token_spans, step_advantages):
        s, e = span
        if e > s:
            adv[s:e] = val
            last_val = val
    # handle gaps (should be rare if spans are contiguous)
    if apply_to_unspanned == "reuse_last" and last_val is not None:
        # fill any zeros with last_val
        mask = (adv == 0.0)
        adv[mask] = last_val
    return adv


def build_per_token_advantages(
    version: Version,
    A_out: Sequence[float],                              # length N
    Z_per_response: Sequence[Sequence[float]] | None,    # list of lists
    w_per_response: Sequence[Sequence[float]] | None,    # list of lists (version c only)
    step_token_spans_per_response: Sequence[Sequence[Tuple[int, int]]],
    total_tokens_per_response: Sequence[int],
    alpha: float = 1.0,
    apply_to_unspanned: Literal["omit", "reuse_last"] = "omit",
) -> List[np.ndarray]:
    """
    Combine outcome + (optional) process (+ uncertainty weights) and broadcast to tokens.

    Returns:
        List[np.ndarray] with shape (Toks_i,) for each response i.
    """
    N = len(A_out)
    assert len(step_token_spans_per_response) == N
    assert len(total_tokens_per_response) == N

    adv_per_token: List[np.ndarray] = []
    for i in range(N):
        if version == "a":
            step_vals: List[float] = []
        elif version == "b":
            assert Z_per_response is not None
            step_vals = [A_out[i] + alpha * z for z in Z_per_response[i]]
        elif version == "c":
            assert Z_per_response is not None and w_per_response is not None
            step_vals = [A_out[i] + alpha * w * z for (w, z) in zip(w_per_response[i], Z_per_response[i])]
        else:
            raise ValueError(f"Unknown version: {version}")

        if version == "a":
            # Outcome-only: constant per-token value equal to A_out[i]
            arr = np.full((total_tokens_per_response[i],), float(A_out[i]), dtype=np.float32)
        else:
            arr = broadcast_step_advantages_to_tokens(
                step_token_spans=step_token_spans_per_response[i],
                step_advantages=step_vals,
                total_tokens=total_tokens_per_response[i],
                apply_to_unspanned=apply_to_unspanned,
            )
        adv_per_token.append(arr)
    return adv_per_token
