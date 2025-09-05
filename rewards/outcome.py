# grpo_single_policy_prm/rewards/outcome.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Optional deps: Math-Verify and SymPy
try:
    # Minimal, lazy import pattern. We'll only call these when present.
    import math_verify  # type: ignore
    _HAS_MATH_VERIFY = True
except Exception:
    _HAS_MATH_VERIFY = False

try:
    from sympy import Rational, nsimplify  # type: ignore
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False


_NUM_RE = re.compile(
    r"""
    (?P<sign>[-+])?
    (?:
        (?P<frac>\d+\s*/\s*\d+)      # a/b
        |
        (?P<float>\d+\.\d+)          # 1.23
        |
        (?P<int>\d+)                 # 123
    )
    """,
    re.VERBOSE,
)

_GSM8K_TRAILER_RE = re.compile(r"####\s*(.+)$")


@dataclass
class OutcomeGrade:
    correct: int           # 1 or 0
    pred_canonical: str    # canonical string we compared with
    gold_canonical: str
    info: Dict[str, Any]


def _strip_commas(s: str) -> str:
    return s.replace(",", "")


def _parse_fraction(s: str) -> Optional[Tuple[int, int]]:
    s = s.strip()
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            return int(a.strip()), int(b.strip())
        except Exception:
            return None
    return None


def _canonical_numeric(s: str) -> Optional[str]:
    """
    Turn a numeric-like string into a canonical representation:
      - Prefer exact rationals "p/q" (in lowest terms) when possible.
      - Otherwise decimal normalized with no leading '+' and no commas.

    Returns None if parsing fails completely.
    """
    s = _strip_commas(s).strip()
    # Fraction?
    frac = _parse_fraction(s)
    if frac is not None:
        p, q = frac
        if q == 0:
            return None
        # Reduce fraction via SymPy if available; else naive sign handling
        if _HAS_SYMPY:
            try:
                r = Rational(p, q)
                return f"{int(r.p)}" if r.q == 1 else f"{int(r.p)}/{int(r.q)}"
            except Exception:
                pass
        # Fallback: no reduction
        # Normalize sign to numerator only
        sign = "-" if (p < 0) ^ (q < 0) else ""
        return f"{sign}{abs(p)}/{abs(q)}"

    # Float / int
    try:
        if _HAS_SYMPY:
            # nsimplify will convert simple decimals to rationals exactly
            r = nsimplify(s, rational=True)
            if r.is_Rational:
                return f"{int(r.p)}" if r.q == 1 else f"{int(r.p)}/{int(r.q)}"
            # Otherwise fallback to decimal string with normalized sign
            val = float(str(r))
            return str(val)
        else:
            val = float(s)
            # Keep integer look if it's integral
            if abs(val - round(val)) < 1e-12:
                return str(int(round(val)))
            return str(val)
    except Exception:
        return None


def _extract_gsm8k_trailer(s: str) -> Optional[str]:
    """
    GSM8K-style: pick the text after '####'.
    """
    m = _GSM8K_TRAILER_RE.search(s)
    if m:
        return m.group(1).strip()
    return None


def extract_numeric_answer(answer_text: str) -> Optional[str]:
    """
    Heuristic extraction for a numeric answer from a free-form solution.

    Preference order:
      1) GSM8K '#### <ans>' trailer
      2) The *last* numeric-like token in the text (fraction > float > int by regex)

    Returns:
        Canonical numeric string (e.g. "7/3" or "42") or None if not found/parsable.
    """
    # 1) GSM8K trailer
    trailer = _extract_gsm8k_trailer(answer_text)
    if trailer:
        can = _canonical_numeric(trailer)
        if can is not None:
            return can

    # 2) Last numeric match
    found = None
    for m in _NUM_RE.finditer(answer_text):
        sign = m.group("sign") or ""
        if m.group("frac"):
            tok = sign + m.group("frac")
        elif m.group("float"):
            tok = sign + m.group("float")
        else:
            tok = sign + m.group("int")
        found = tok  # keep last
    if found is not None:
        return _canonical_numeric(found)
    return None


def _equiv_math_verify(pred: str, gold: str) -> Optional[bool]:
    """
    Try Math-Verify if installed. Returns None if not available or on error.
    """
    if not _HAS_MATH_VERIFY:
        return None
    try:
        # API surface of math_verify may vary; we attempt the common "is_equiv" pattern.
        # If your local variant differs, adapt here.
        return bool(math_verify.is_equiv(pred, gold))  # type: ignore[attr-defined]
    except Exception:
        return None


def _equiv_sympy(pred: str, gold: str) -> Optional[bool]:
    if not _HAS_SYMPY:
        return None
    try:
        rp = nsimplify(pred, rational=True)
        rg = nsimplify(gold, rational=True)
        return bool(rp.equals(rg))
    except Exception:
        return None


def grade_outcome_numeric(
    pred_answer_text: str,
    gold_answer_text: str,
) -> OutcomeGrade:
    """
    Strictly numeric canonicalization & grading.

    - Extract numeric from pred using GSM8K trailer or last-number heuristic.
    - Canonicalize gold (assumed already numeric string but we canonicalize anyway).
    - Equivalence check priority:
        1) Math-Verify (if present)
        2) SymPy equivalence (if present)
        3) Exact string equality of canonical forms
    """
    pred_can = extract_numeric_answer(pred_answer_text)
    gold_can = _canonical_numeric(gold_answer_text)

    info: Dict[str, Any] = {
        "method": None,
        "pred_extracted": pred_can,
        "gold_canonical": gold_can,
        "used_math_verify": _HAS_MATH_VERIFY,
        "used_sympy": _HAS_SYMPY,
    }

    if pred_can is None or gold_can is None:
        info["method"] = "missing_numeric"
        return OutcomeGrade(correct=0, pred_canonical=str(pred_can), gold_canonical=str(gold_can), info=info)

    # 1) Math-Verify
    eq = _equiv_math_verify(pred_can, gold_can)
    if eq is not None:
        info["method"] = "math_verify"
        return OutcomeGrade(correct=int(eq), pred_canonical=pred_can, gold_canonical=gold_can, info=info)

    # 2) SymPy
    eq = _equiv_sympy(pred_can, gold_can)
    if eq is not None:
        info["method"] = "sympy"
        return OutcomeGrade(correct=int(eq), pred_canonical=pred_can, gold_canonical=gold_can, info=info)

    # 3) Exact canonical string match
    info["method"] = "string_equal"
    return OutcomeGrade(correct=int(pred_can == gold_can), pred_canonical=pred_can, gold_canonical=gold_can, info=info)
