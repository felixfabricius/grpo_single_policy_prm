# grpo_single_policy_prm/scripts/train.py
import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml

from grpo_single_policy_prm.data.loaders.gsm8k import GSM8KLoader
from grpo_single_policy_prm.data.loaders.math import MATHLoader
from grpo_single_policy_prm.data.loaders.omnimath import OmniMATHLoader
from grpo_single_policy_prm.data.loaders.olympiadbench import OlympiadBenchLoader
from grpo_single_policy_prm.data.loaders.jsonl_generic import JSONLGenericLoader
from grpo_single_policy_prm.data.mix import build_mixed_stream
from grpo_single_policy_prm.data.schema import Example

from grpo_single_policy_prm.prm.dummy_ensemble import DummyPRMEnsemble
from grpo_single_policy_prm.rewards.uncertainty import UncertaintyConfig
from grpo_single_policy_prm.grpo.trainer import GRPOTrainer, GRPOTrainerConfig
from grpo_single_policy_prm.models.policy_lora import PolicyInitConfig
from grpo_single_policy_prm.models.ref_policy import RefInitConfig
from grpo_single_policy_prm.utils.ckpt import save_checkpoint
from grpo_single_policy_prm.utils.ids import run_id as make_run_id


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_loaders_from_config(cfg: Dict[str, Any]):
    ds_cfg = cfg.get("datasets", {})
    weights = ds_cfg.get("mix", {"gsm8k": 0.4, "math": 0.3, "omnimath": 0.2, "olympiad": 0.1})
    split = ds_cfg.get("split", "train")
    fallback_paths = ds_cfg.get("jsonl_fallback", ["data/local_eval/gsm8k_small.jsonl"])

    # Optional explicit paths section:
    # datasets:
    #   paths: {gsm8k: /path/to/gsm8k, math: ...}
    paths = (ds_cfg.get("paths") or {})

    gsm = GSM8KLoader(path=paths.get("gsm8k"), strict=bool("gsm8k" in paths))
    math = MATHLoader(path=paths.get("math"), strict=bool("math" in paths))
    omni = OmniMATHLoader(path=paths.get("omnimath"), strict=bool("omnimath" in paths))
    olymp = OlympiadBenchLoader(path=paths.get("olympiad"), strict=bool("olympiad" in paths))

    streams = {}
    if weights.get("gsm8k", 0) > 0 and gsm.is_available():
        streams["gsm8k"] = (gsm.prepare(split), float(weights["gsm8k"]))
    if weights.get("math", 0) > 0 and math.is_available():
        streams["math"] = (math.prepare(split), float(weights["math"]))
    if weights.get("omnimath", 0) > 0 and omni.is_available():
        streams["omnimath"] = (omni.prepare(split), float(weights["omnimath"]))
    if weights.get("olympiad", 0) > 0 and olymp.is_available():
        streams["olympiad"] = (olymp.prepare(split), float(weights["olympiad"]))

    # Fallback JSONL(s) if nothing else is available
    if not streams:
        jl = JSONLGenericLoader(paths=fallback_paths)
        streams["jsonl"] = (jl.prepare(split), 1.0)

    return build_mixed_stream(streams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default="runs")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = run forever (or until data exhausts)")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)

    # Resolve run id and dirs
    run_id = args.run_id or cfg.get("run_id") or make_run_id("run")
    out_root = args.output_dir
    run_dir = os.path.join(out_root, run_id)
    _ensure_dir(run_dir)
    _ensure_dir(os.path.join(run_dir, "checkpoints"))

    # Seeds
    seed = int(cfg.get("seed", 123))
    set_global_seed(seed)
    # Optional: slightly relax determinism for speed (as agreed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Policy/ref init
    model_name = cfg.get("model_name", "Qwen/Qwen2.5-Math-1.5B")
    precision = cfg.get("precision", "bf16")
    bf16 = (precision.lower() == "bf16")
    policy_init = PolicyInitConfig(
        model_name=model_name,
        bf16=bf16,
        grad_checkpointing=bool(cfg.get("grad_checkpointing", True)),
        lora_r=cfg.get("lora", {}).get("r", 16),
        lora_alpha=cfg.get("lora", {}).get("alpha", 32),
        lora_dropout=cfg.get("lora", {}).get("dropout", 0.05),
        target_modules=tuple(cfg.get("lora", {}).get("target_modules", ["q_proj", "v_proj", "o_proj"])),
    )
    ref_init = RefInitConfig(model_name=model_name, bf16=bf16)

    # Build trainer config
    tr_cfg = GRPOTrainerConfig(
        run_id=run_id,
        output_dir=out_root,
        seed=seed,
        N=int(cfg.get("gen", {}).get("N", 8)),
        temperature=float(cfg.get("gen", {}).get("temperature", 0.7)),
        top_p=float(cfg.get("gen", {}).get("top_p", 0.95)),
        max_new_tokens=int(cfg.get("gen", {}).get("max_new_tokens", 768)),
        training_version=str(cfg.get("training_version", "a")),
        alpha=float(cfg.get("adv", {}).get("alpha", 1.0)),
        eps=float(cfg.get("adv", {}).get("eps", 1e-6)),
        apply_to_unspanned=str(cfg.get("process", {}).get("apply_to_unspanned", "omit")),
        overflow_penalty=float(cfg.get("process", {}).get("overflow_penalty", 0.05)),
        loss_type=str(cfg.get("grpo", {}).get("loss_type", "plain")),
        epsilon_low=float(cfg.get("grpo", {}).get("epsilon_low", 0.2)),
        epsilon_high=float(cfg.get("grpo", {}).get("epsilon_high", 0.28)),
        beta=float(cfg.get("grpo", {}).get("beta", 0.10)),
        lr=float(cfg.get("optim", {}).get("lr", 2.0e-5)),
        betas=tuple(cfg.get("optim", {}).get("betas", [0.9, 0.95])),
        weight_decay=float(cfg.get("optim", {}).get("weight_decay", 0.0)),
        warmup_steps=int(cfg.get("optim", {}).get("warmup_steps", 200)),
        grad_clip=float(cfg.get("optim", {}).get("grad_clip", 1.0)),
        log_every=int(cfg.get("logging", {}).get("log_every", 50)),
        semantic_every_steps=int(cfg.get("logging", {}).get("semantic_every_steps", 500)),
        unc_log_every_steps=int(cfg.get("logging", {}).get("unc_log_every_steps", 200)),
        nvml_log=True,
        policy_init=policy_init,
        ref_init=ref_init,
    )

    # Uncertainty config (version c only)
    if tr_cfg.training_version == "c":
        prm_cfg = cfg.get("prm", {})
        use_unbiased = bool(prm_cfg.get("use_unbiased_std", False))
        unc_cfg = cfg.get("uncertainty", {})
        tr_cfg.uncertainty = UncertaintyConfig(
            M=int(prm_cfg.get("M", 4)),
            estimator=("unbiased" if use_unbiased else "population"),
            scheme=str(unc_cfg.get("scheme", "one_minus_pow")),
            gamma=float(unc_cfg.get("gamma", 1.0)),
            beta=float(unc_cfg.get("beta", 4.0)),
            w_min=float(unc_cfg.get("w_min", 0.2)),
            clip_u_to_one=True,
        )

    # Optimizer
    # Note: We train *only* LoRA params (PEFT set them requires_grad=True).
    trainer_dummy = GRPOTrainer(tr_cfg, prm=DummyPRMEnsemble(M=int(cfg.get("prm", {}).get("M", 2))), optimizer=None)  # type: ignore[arg-type]
    # Build optimizer on the policy model now available
    params = [p for p in trainer_dummy.policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tr_cfg.lr, betas=tr_cfg.betas, weight_decay=tr_cfg.weight_decay)
    # Recreate trainer with optimizer wired
    trainer = GRPOTrainer(tr_cfg, prm=DummyPRMEnsemble(M=int(cfg.get("prm", {}).get("M", 2))), optimizer=optimizer)

    # Data stream
    mixed = build_loaders_from_config(cfg)

    # Generator for deterministic sampling
    g = torch.Generator(device=trainer.device if torch.cuda.is_available() else "cpu")
    g.manual_seed(seed)

    # Training loop (prompts_per_step=1 enforced)
    prompts_per_step = int(cfg.get("batching", {}).get("prompts_per_step", 1))
    if prompts_per_step != 1:
        print("[WARN] prompts_per_step must be 1 for correct group-relative stats; overriding to 1.", file=sys.stderr)

    save_every = int(cfg.get("ckpt", {}).get("save_every", 500))
    keep_last = int(cfg.get("ckpt", {}).get("keep_last", 3))
    ckpt_root = os.path.join(run_dir, "checkpoints")
    _ensure_dir(ckpt_root)

    max_steps = int(args.max_steps or 0)
    step_count = 0

    for ex in mixed:
        # Single-example step
        out = trainer.train_on_example(ex, generator=g)

        step_count += 1
        if (trainer.global_step % save_every) == 0:
            # Save checkpoint
            save_checkpoint(
                save_root=ckpt_root,
                step=trainer.global_step,
                policy_peft_model=trainer.policy.model,  # PeftModel
                tokenizer=trainer.policy.tokenizer,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                trainer_state=dict(
                    global_step=trainer.global_step,
                    run_id=run_id,
                    config_snapshot=cfg,
                ),
                keep_last=keep_last,
            )

        if max_steps > 0 and step_count >= max_steps:
            break

    # Final checkpoint
    save_checkpoint(
        save_root=ckpt_root,
        step=trainer.global_step,
        policy_peft_model=trainer.policy.model,
        tokenizer=trainer.policy.tokenizer,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        trainer_state=dict(
            global_step=trainer.global_step,
            run_id=run_id,
            config_snapshot=cfg,
        ),
        keep_last=keep_last,
    )
    print(f"[DONE] Run {run_id} finished at step {trainer.global_step}. Output in {run_dir}")


if __name__ == "__main__":
    main()
