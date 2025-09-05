# grpo_single_policy_prm/models/policy_lora.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model


@dataclass
class PolicyInitConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    bf16: bool = True
    grad_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Sequence[str] = ("q_proj", "v_proj", "o_proj")


class PolicyWithLoRA(nn.Module):
    """
    Single policy: base LM + one LoRA adapter (trainable).
    Provides generation and per-token log-prob utilities on the *answer* segment.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerBase, adapter_name: str = "policy_lora"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_name = adapter_name

    @classmethod
    def from_pretrained(cls, cfg: PolicyInitConfig) -> "PolicyWithLoRA":
        dtype = torch.bfloat16 if cfg.bf16 else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            # For causal LM, set pad to eos if missing
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        if cfg.grad_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False  # must disable when using grad ckpt

        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=list(cfg.target_modules),
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
        return cls(model=model, tokenizer=tokenizer)

    def set_train_mode(self, flag: bool = True) -> None:
        self.train(flag)

    def generation_config(
        self,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 768,
        do_sample: bool = True,
    ) -> GenerationConfig:
        gc = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return gc

    @torch.no_grad()
    def generate_n(
        self,
        prompt: str,
        N: int,
        gen_cfg: GenerationConfig,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[List[List[int]], List[int], List[int]]:
        """
        Generate N responses for a single prompt.

        Returns:
            answer_token_ids_list: list length N of token-id lists for the *answer only*
            prompt_input_ids: token ids for the prompt (single copy; used for re-scoring)
            full_input_ids_lengths: length of prompt+answer for each sample (for sanity)
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Tokenize prompt once (no special tokens for causal LM prompt)
        prompt_enc = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        prompt_input_ids = prompt_enc["input_ids"][0].tolist()

        # Replicate batch
        batch_input = self.tokenizer(
            [prompt] * N,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        ).to(device)

        outputs = self.model.generate(
            **batch_input,
            generation_config=gen_cfg,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            synced_gpus=False,
            generator=generator,
        )

        # outputs shape: (N, prompt_len + new_len) because we passed inputs
        answer_token_ids_list: List[List[int]] = []
        full_lengths: List[int] = []
        prompt_len = batch_input["input_ids"].shape[1]
        for i in range(outputs.shape[0]):
            full_seq = outputs[i].tolist()
            answer_ids = full_seq[prompt_len:]
            answer_token_ids_list.append(answer_ids)
            full_lengths.append(len(full_seq))
        return answer_token_ids_list, prompt_input_ids, full_lengths

    def _gather_answer_logprobs(
        self,
        input_ids: torch.Tensor,      # (1, S)
        prompt_len: int,
    ) -> torch.Tensor:
        """
        Compute per-token log-probs for the *answer* tokens only.

        Returns:
            logp_answer: (T,) tensor with log-prob for each answer token.
        """
        device = input_ids.device
        # Forward once
        out = self.model(input_ids=input_ids, use_cache=False)
        logits = out.logits  # (1, S, V)
        # Next-token prediction: token at t is predicted from logits[t-1]
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)  # (1, S-1, V)

        # Targets are input_ids[:, 1:]
        target = input_ids[:, 1:]  # (1, S-1)

        # Answer region in target space starts at index 'prompt_len'
        # Explanation: targets index t corresponds to token at original position t+1
        # Prompt occupies [0 .. prompt_len-1], so answer targets are [prompt_len .. end-1]
        start = prompt_len
        end = target.shape[1]

        idx = torch.arange(start, end, device=device)
        # Gather logp for the correct token at each position
        gather = logp[0, idx, target[0, idx]]  # (T,)
        return gather.contiguous()

    @torch.no_grad()
    def compute_logprobs_for_answers(
        self,
        prompt: str,
        answers_token_ids: List[List[int]],
    ) -> List[torch.Tensor]:
        """
        Compute per-token log-probs for each answer under the *current* policy.

        Returns:
            List of 1D tensors (length = len(answer_i)) for each answer.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Tokenize prompt once
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
        prompt_len = prompt_ids.numel()

        out_list: List[torch.Tensor] = []
        for ans_ids in answers_token_ids:
            ans = torch.tensor(ans_ids, dtype=prompt_ids.dtype, device=device)
            full = torch.cat([prompt_ids, ans], dim=0).unsqueeze(0)  # (1,S)
            lp = self._gather_answer_logprobs(full, prompt_len=prompt_len)  # (T,)
            out_list.append(lp)
        return out_list
