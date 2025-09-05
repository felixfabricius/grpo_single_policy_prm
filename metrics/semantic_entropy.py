# grpo_single_policy_prm/metrics/semantic_entropy.py
from __future__ import annotations

import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Literal

from ..rewards.outcome import extract_numeric_answer

try:
    from sympy import nsimplify  # type: ignore
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False


ClusterMode = Literal["exact", "math"]


def _cluster_labels_exact(answers: List[str]) -> List[str]:
    """
    Cluster by *canonical numeric string* if extractable; else special bucket.
    """
    labels: List[str] = []
    for a in answers:
        can = extract_numeric_answer(a)
        labels.append(can if can is not None else "__NA__")
    return labels


def _cluster_labels_math(answers: List[str]) -> List[str]:
    """
    Cluster by *math equivalence* (numbers/fractions).
    If SymPy is available, we map to nsimplify() string; else fallback to exact.
    """
    if not _HAS_SYMPY:
        return _cluster_labels_exact(answers)

    labels: List[str] = []
    for a in answers:
        can = extract_numeric_answer(a)
        if can is None:
            labels.append("__NA__")
            continue
        try:
            r = nsimplify(can, rational=True)
            labels.append(str(r))
        except Exception:
            # If nsimplify fails, fallback to exact canonical string
            labels.append(can)
    return labels


def _cluster_labels(answers: List[str], mode: ClusterMode) -> List[str]:
    return _cluster_labels_exact(answers) if mode == "exact" else _cluster_labels_math(answers)


def _entropy_norm_and_stats(labels: List[str]) -> Dict[str, float]:
    """
    Compute normalized Shannon entropy (by log K), HHI, top1_share, uniques_ratio.
    """
    N = len(labels)
    counts = Counter(labels)
    K = len(counts)
    if N == 0:
        return dict(entropy_norm=0.0, hhi=0.0, top1_share=0.0, uniques_ratio=0.0)

    ps = [c / N for c in counts.values()]
    # Shannon entropy
    H = -sum(p * math.log(p + 1e-12) for p in ps)
    H_norm = (H / math.log(K)) if K > 1 else 0.0
    hhi = sum(p * p for p in ps)
    top1 = max(ps)
    uniques_ratio = K / N
    return dict(entropy_norm=float(H_norm), hhi=float(hhi), top1_share=float(top1), uniques_ratio=float(uniques_ratio))


def compute_semantic_diversity(
    answers: List[str],
    mode: ClusterMode,
) -> Dict[str, float]:
    labels = _cluster_labels(answers, mode=mode)
    return _entropy_norm_and_stats(labels)


def log_semantic_diversity_rows(
    writer,
    run_id: str,
    step: int,
    prompt_id: str,
    answers: List[str],
    decoding_params: Dict[str, Any],
) -> None:
    """
    Append two JSONL rows (exact + math) to semantic_diversity.jsonl using the same N answers.
    """
    n_samples = len(answers)
    for mode in ("exact", "math"):
        metrics = compute_semantic_diversity(answers, mode=mode)  # type: ignore[arg-type]
        row = {
            "run": run_id,
            "step": step,
            "prompt_id": prompt_id,
            "n_samples": n_samples,
            "cluster": mode,
            **metrics,
            "decoding": dict(temperature=decoding_params.get("temperature"),
                             top_p=decoding_params.get("top_p"),
                             max_new_tokens=decoding_params.get("max_new_tokens")),
        }
        writer.write(row)
