"""Math answer verification utilities.

VENDORED from: sdk/demo/zo-training/rewards.py
Vendored on: 2026-04-11
Reason: The sdk/demo code may change independently; evals needs a stable copy.

Uses ``math_verify`` + ``latex2sympy2_extended`` for robust LaTeX math
comparison (strict boxed parse → lenient fallback → symbolic verify).
"""

import re
from typing import Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (e.g. from Qwen3 thinking mode).

    If the output is truncated mid-think (opened <think> but never closed),
    returns empty string — the model never produced an answer.
    """
    # Complete thinking blocks: remove them
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # If a <think> tag remains, the block was never closed (truncated)
    if "<think>" in stripped:
        _, _, after = stripped.partition("</think>")
        if after.strip():
            return after.strip()
        # Truncated mid-think with no answer — return empty
        return ""
    return stripped if stripped else text


def normalize_latex(text: str) -> str:
    """Normalize LaTeX formatting for string comparison.

    Strips \\left, \\right, \\text{...} → content, and collapses whitespace.
    """
    t = text.strip()
    # \text{Evelyn} → Evelyn
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    # \left( → (  and \right) → )  etc.
    t = re.sub(r"\\left\s*([(\[{|])", r"\1", t)
    t = re.sub(r"\\right\s*([)\]}|])", r"\1", t)
    # \dfrac → \frac
    t = t.replace("\\dfrac", "\\frac")
    # Collapse whitespace
    t = " ".join(t.split())
    return t


def _extract_boxed_content(text: str) -> str | None:
    """Extract the content inside \\boxed{...}, handling nested braces."""
    start = text.find("\\boxed{")
    if start < 0:
        return None
    depth = 0
    inner_start = start + len("\\boxed{")
    for i in range(inner_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[inner_start:i]
            depth -= 1
    return None


def accuracy_reward(completion: str, solution: str) -> Optional[float]:
    """Check if *completion* is mathematically equivalent to *solution*.

    Returns 1.0 (correct), 0.0 (incorrect), or None (unparseable gold).
    Tries:
      1. Normalized string comparison (handles \\text{}, \\left/\\right, etc.)
      2. Strict math_verify parse (requires \\boxed{})
      3. Lenient math_verify parse
    Strips ``<think>`` blocks from thinking models (e.g. Qwen3).
    """
    completion = strip_thinking(completion)

    if completion.lower().strip() == solution.lower().strip():
        return 1.0

    # --- Normalized string comparison fallback ---
    # Handles cases like \text{Evelyn} vs Evelyn, or
    # \left( 3, \frac{\pi}{2} \right) vs (3, \frac{\pi}{2})
    boxed_content = _extract_boxed_content(completion)
    if boxed_content is not None:
        norm_boxed = normalize_latex(boxed_content)
        norm_gold = normalize_latex(solution)
        if norm_boxed.lower() == norm_gold.lower():
            return 1.0

    # --- math_verify symbolic comparison ---
    gold_parsed = parse(solution)
    if not gold_parsed:
        # Gold unparseable by math_verify — if we have boxed content,
        # the normalized comparison above was the best we could do
        return 0.0 if boxed_content is not None else None

    # 1. Strict: requires \boxed{} or similar wrapper
    strict_config = LatexExtractionConfig(
        normalization_config=NormalizationConfig(
            nits=False,
            malformed_operators=False,
            basic_latex=True,
            boxed="all",
            units=True,
        ),
        boxed_match_priority=0,
        try_extract_without_anchor=False,
    )
    answer_parsed = parse(
        completion, extraction_config=[strict_config], extraction_mode="first_match"
    )

    # 2. Lenient fallback: find answers without \boxed{}
    if not answer_parsed:
        lenient_config = LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=True,
                basic_latex=True,
                boxed="all",
                units=True,
            ),
            try_extract_without_anchor=True,
        )
        answer_parsed = parse(
            completion,
            extraction_config=[lenient_config],
            extraction_mode="first_match",
        )

    try:
        reward = float(verify(gold_parsed, answer_parsed))
    except Exception:
        reward = None

    return reward


def extract_answer(completion: str) -> Optional[str]:
    """Extract text between <answer> and </answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
