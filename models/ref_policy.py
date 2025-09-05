# grpo_single_policy_prm/models/ref_policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RefInitConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    bf16: bool = True


class FrozenReference:
    """
    Frozen base model (no LoRA) for KL computation.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, cfg: RefInitConfig) -> "FrozenReference":
        dtype = torch.bfloat16 if cfg.bf16 else torch.float16
        tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return cls(model=model, tokenizer=tok)

    @torch.no_grad()
    def compute_logprobs_for_answers(
        self,
        prompt: str,
        answers_token_ids: List[List[int]],
    ) -> List[torch.Tensor]:
        """
        Per-token log-probs for each answer under the *frozen* reference model.
        """
        device = next(self.model.parameters()).device
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
        prompt_len = prompt_ids.numel()

        out_list: List[torch.Tensor] = []
        for ans_ids in answers_token_ids:
            ans = torch.tensor(ans_ids, dtype=prompt_ids.dtype, device=device)
            full = torch.cat([prompt_ids, ans], dim=0).unsqueeze(0)  # (1,S)
            out = self.model(input_ids=full, use_cache=False)
            logits = out.logits  # (1,S,V)
            logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target = full[:, 1:]
            start = prompt_len
            end = target.shape[1]
            idx = torch.arange(start, end, device=device)
            gather = logp[0, idx, target[0, idx]].contiguous()
            out_list.append(gather)
        return out_list
