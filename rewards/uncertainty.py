# grpo_single_policy_prm/rewards/uncertainty.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Literal

import math
import statistics


WeightScheme = Literal["one_minus_pow", "exp_quad", "rational"]
Estimator = Literal["population", "unbiased"]


@dataclass
class UncertaintyConfig:
    """
    Config for uncertainty normalization and weight mapping.
    """
    M: int  # ensemble size
    estimator: Estimator = "population"
    scheme: WeightScheme = "one_minus_pow"
    gamma: float = 1.0  # for one_minus_pow
    beta: float = 4.0   # for exp_quad/rational
    w_min: float = 0.2  # final weights in [w_min, 1]
    clip_u_to_one: bool = True

    def sd_max(self) -> float:
        """
        Theoretical maximum standard deviation for variables in [0,1].
        For population: 0.5
        For unbiased: 0.5 * sqrt(M/(M-1))
        """
        if self.estimator == "population":
            return 0.5
        else:
            return 0.5 * math.sqrt(self.M / max(1, self.M - 1))


def std_across_members(
    member_probs_per_step: Sequence[Sequence[float]],
    estimator: Estimator = "population",
    eps: float = 1e-12,
) -> List[float]:
    """
    Compute per-step std across members.

    Args:
        member_probs_per_step: shape (M, T) list-of-lists
        estimator: "population" uses population std (denominator M),
                   "unbiased" uses sample std (denominator M-1)
    Returns:
        List[float] of length T with std per step.
    """
    if len(member_probs_per_step) == 0:
        return []
    M = len(member_probs_per_step)
    T = len(member_probs_per_step[0])
    for row in member_probs_per_step:
        assert len(row) == T, "All member rows must have same length T"

    sds: List[float] = []
    for t in range(T):
        col = [member_probs_per_step[m][t] for m in range(M)]
        if estimator == "population":
            mu = sum(col) / M
            var = sum((x - mu) ** 2 for x in col) / M
            sds.append(math.sqrt(max(0.0, var)))
        else:
            if M <= 1:
                sds.append(0.0)
            else:
                mu = sum(col) / M
                var = sum((x - mu) ** 2 for x in col) / (M - 1)
                sds.append(math.sqrt(max(0.0, var)))
    return sds


def normalize_u(sd: Sequence[float], sd_max: float, clip_to_one: bool = True) -> List[float]:
    """u = sd / sd_max, optionally clipped to [0,1]."""
    if sd_max <= 0:
        return [0.0 for _ in sd]
    u = [x / sd_max for x in sd]
    if clip_to_one:
        u = [max(0.0, min(1.0, x)) for x in u]
    return u


def map_u_to_weight(u: Sequence[float], scheme: WeightScheme, gamma: float, beta: float, w_min: float) -> List[float]:
    """
    Map normalized uncertainties u in [0,1] to weights in [w_min, 1].

    Schemes:
      - one_minus_pow: w_raw = (1 - u)^gamma
      - exp_quad:     w_raw = exp(-beta * u)
      - rational:     w_raw = 1 / (1 + beta * u)
    """
    w: List[float] = []
    for val in u:
        if scheme == "one_minus_pow":
            w_raw = (1.0 - val) ** gamma
        elif scheme == "exp_quad":
            w_raw = math.exp(-beta * val)
        elif scheme == "rational":
            w_raw = 1.0 / (1.0 + beta * val)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
        w_final = w_min + (1.0 - w_min) * w_raw
        # clamp to [w_min, 1] for numerical safety
        w.append(min(1.0, max(w_min, w_final)))
    return w
