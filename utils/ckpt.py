# grpo_single_policy_prm/utils/ckpt.py
from __future__ import annotations

import json
import os
import re
import shutil
import time
from typing import Any, Dict, Optional

import numpy as np
import torch


def _save_rng_state(root: str) -> None:
    state = {
        "python_time": time.time(),
        "numpy": np.random.get_state()[1].tolist(),
        "torch": torch.get_rng_state().cpu().numpy().tolist(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state().cpu().numpy().tolist()
    with open(os.path.join(root, "rng_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f)


def _load_rng_state(root: str) -> None:
    p = os.path.join(root, "rng_state.json")
    if not os.path.exists(p):
        return
    with open(p, "r", encoding="utf-8") as f:
        state = json.load(f)
    if "numpy" in state:
        np.random.set_state(("MT19937", np.array(state["numpy"], dtype=np.uint32), 0, 0.0, 0.0))
    if "torch" in state:
        torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state(torch.tensor(state["torch_cuda"], dtype=torch.uint8))


def save_checkpoint(
    save_root: str,
    step: int,
    policy_peft_model,        # PeftModel (policy with LoRA)
    tokenizer,
    optimizer,
    scheduler,
    trainer_state: Dict[str, Any],
    keep_last: int = 3,
) -> str:
    """
    Save LoRA adapter, optimizer, scheduler, tokenizer, RNG and trainer state.

    Layout:
      save_root/step-000500/
        adapter/...
        tokenizer/...
        optimizer.pt
        scheduler.pt
        rng_state.json
        trainer_state.json
    """
    step_dir = os.path.join(save_root, f"step-{step:06d}")
    os.makedirs(step_dir, exist_ok=True)

    # 1) LoRA adapter weights (PEFT)
    adapter_dir = os.path.join(step_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    policy_peft_model.save_pretrained(adapter_dir)

    # 2) Tokenizer
    tok_dir = os.path.join(step_dir, "tokenizer")
    tokenizer.save_pretrained(tok_dir)

    # 3) Optimizer & scheduler
    torch.save(optimizer.state_dict(), os.path.join(step_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(step_dir, "scheduler.pt"))

    # 4) RNG
    _save_rng_state(step_dir)

    # 5) Trainer state snapshot
    with open(os.path.join(step_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, ensure_ascii=False, indent=2)

    # 6) Prune old checkpoints
    _prune_keep_last(save_root, keep_last=keep_last)

    return step_dir


def load_checkpoint(
    step_dir: str,
    policy_peft_model,
    tokenizer,
    optimizer,
    scheduler,
) -> Dict[str, Any]:
    """
    Load LoRA adapter, tokenizer, optimizer, scheduler, RNG and trainer state.

    Returns:
        trainer_state (dict)
    """
    # Adapter
    policy_peft_model.load_adapter(step_dir + "/adapter", adapter_name=policy_peft_model.active_adapter, is_trainable=True)

    # Tokenizer
    tokenizer.from_pretrained(step_dir + "/tokenizer")  # type: ignore[attr-defined]

    # Optim & sched
    opt_state = torch.load(step_dir + "/optimizer.pt", map_location="cpu")
    sch_state = torch.load(step_dir + "/scheduler.pt", map_location="cpu")
    optimizer.load_state_dict(opt_state)
    scheduler.load_state_dict(sch_state)

    # RNG
    _load_rng_state(step_dir)

    # Trainer state
    with open(step_dir + "/trainer_state.json", "r", encoding="utf-8") as f:
        trainer_state = json.load(f)
    return trainer_state


def _prune_keep_last(root: str, keep_last: int) -> None:
    """
    Keep only the most recent 'keep_last' step-* directories.
    """
    if keep_last <= 0:
        return
    if not os.path.exists(root):
        return
    dirs = [d for d in os.listdir(root) if re.match(r"^step-\d+$", d)]
    if len(dirs) <= keep_last:
        return
    # Sort by step number
    dirs_sorted = sorted(dirs, key=lambda s: int(s.split("-")[1]))
    to_delete = dirs_sorted[:-keep_last]
    for d in to_delete:
        shutil.rmtree(os.path.join(root, d), ignore_errors=True)
