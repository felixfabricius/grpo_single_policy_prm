# grpo_single_policy_prm/README.md
# Minimal Single-Policy GRPO (with Process Rewards & Uncertainty)

A minimal-but-complete GRPO trainer for a **single policy** (Qwen2.5-Math-1.5B) with one LoRA adapter. It supports:

- N samples per prompt from the same policy
- Outcome-only (a), outcome+process (b), outcome+process+uncertainty (c)
- Step splitting by `\n`, robust token-span mapping
- Optional PRM ensemble uncertainty (member SD per step → weight)
- JSONL logging, periodic semantic diversity (exact & math)
- Checkpoints with keep-last pruning
- Slurm launcher template

> **Note:** This is intentionally minimal and single-GPU (48 GB). KL is fixed-β; no adaptive β. Loss uses **sums** (not means), per spec.

---

## Quickstart (local, smoke test)

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121    # choose your CUDA
pip install transformers peft pyyaml sympy pynvml

# Optional (if you want strict numeric equivalence via Math-Verify)
# pip install math-verify   # or your local fork providing math_verify.is_equiv()

python grpo_single_policy_prm/scripts/train.py \
  --config grpo_single_policy_prm/configs/train.yaml \
  --run-id run_local_smoke \
  --output-dir runs \
  --max-steps 5
