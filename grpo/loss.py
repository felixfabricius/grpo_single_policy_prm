# grpo_single_policy_prm/grpo/loss.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

import torch


LossType = Literal["plain", "dapo"]


@dataclass
class GRPOLossConfig:
    beta: float = 0.0  # KL strength; 0.0 disables KL term
    loss_type: LossType = "plain"
    # DAPO-style ratio clipping (used if loss_type == "dapo")
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28


@dataclass
class GRPOLossOutputs:
    loss_total: torch.Tensor
    loss_policy: torch.Tensor
    loss_kl: torch.Tensor
    diagnostics: Dict[str, Any]


def _masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum()


def compute_grpo_loss(
    logp_new: torch.Tensor,         # (T,) token log-probs under current policy
    logp_old: Optional[torch.Tensor],  # (T,) token log-probs under "old" policy (for ratio); required if loss_type=="dapo"
    advantages: torch.Tensor,       # (T,) per-token advantages (already combined outcome/process)
    mask: torch.Tensor,             # (T,) 1.0 for valid tokens, 0.0 for padding/ignored
    logp_ref: Optional[torch.Tensor] = None,  # (T,) token log-probs under frozen reference (for KL)
    cfg: GRPOLossConfig = GRPOLossConfig(),
) -> GRPOLossOutputs:
    """
    Compute GRPO-style objective with optional KL and DAPO ratio clipping.

    Objective (sum over tokens):
      - plain:  L = - A * logp_new  + beta * (logp_new - logp_ref)
      - dapo:   L = - min( r * A, clip(r, 1-eps_low, 1+eps_high) * A )  + beta * KL
                where r = exp(logp_new - logp_old)

    Notes:
      - Sums (not means), as requested.
      - If beta==0.0, KL term is identically zero and `logp_ref` may be None.
      - All inputs are expected on the same device and dtype (usually float32/bf16).
    """
    assert logp_new.shape == advantages.shape == mask.shape, "Shape mismatch"
    if cfg.loss_type == "dapo":
        assert logp_old is not None, "logp_old is required when loss_type='dapo'"

    # Policy loss
    if cfg.loss_type == "plain":
        policy_term = -advantages * logp_new
    elif cfg.loss_type == "dapo":
        with torch.no_grad():
            ratio = torch.exp(logp_new - logp_old)  # (T,)
            clipped = torch.clamp(ratio, 1.0 - cfg.epsilon_low, 1.0 + cfg.epsilon_high)
            # min(r*A, clip(r)*A) elementwise
            chosen = torch.minimum(ratio * advantages, clipped * advantages)
        # We *do not* detach logp_new; only the comparator is computed under no_grad.
        policy_term = -chosen
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type}")

    loss_policy = _masked_sum(policy_term, mask)

    # KL loss (token-wise forward KL to frozen ref)
    if cfg.beta > 0.0:
        assert logp_ref is not None, "logp_ref is required when beta>0"
        kl_term = (logp_new - logp_ref)  # token-level KL contribution in log-space
        loss_kl = cfg.beta * _masked_sum(kl_term, mask)
    else:
        # keep dtype/device consistent
        loss_kl = torch.zeros((), dtype=logp_new.dtype, device=logp_new.device)

    loss_total = loss_policy + loss_kl

    # Diagnostics
    diag: Dict[str, Any] = {}
    if cfg.loss_type == "dapo":
        with torch.no_grad():
            ratio = torch.exp(logp_new - logp_old)
            clipped = torch.clamp(ratio, 1.0 - cfg.epsilon_low, 1.0 + cfg.epsilon_high)
            frac_clipped = ((ratio != clipped) & (mask > 0)).float().sum() / (mask.sum() + 1e-9)
            diag.update(
                ratio_mean=float((ratio * mask).sum() / (mask.sum() + 1e-9)),
            )
            diag["frac_clipped"] = float(frac_clipped)
    if cfg.beta > 0.0:
        with torch.no_grad():
            # mean per-token KL (before multiplying by beta)
            kl_term = (logp_new - logp_ref) * mask
            mean_kl = float(kl_term.sum() / (mask.sum() + 1e-9))
            diag["mean_kl"] = mean_kl

    return GRPOLossOutputs(
        loss_total=loss_total,
        loss_policy=loss_policy,
        loss_kl=loss_kl,
        diagnostics=diag,
    )
