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
    """Remove <think>...</think> blocks (e.g. from Qwen3 thinking mode)."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # If the model is still mid-think (truncated), take content after the tag
    if "<think>" in stripped:
        _, _, after = stripped.partition("</think>")
        if after.strip():
            return after.strip()
        # Truncated mid-think with no answer after — return as-is
    return stripped if stripped else text


def accuracy_reward(completion: str, solution: str) -> Optional[float]:
    """Check if *completion* is mathematically equivalent to *solution*.

    Returns 1.0 (correct), 0.0 (incorrect), or None (unparseable gold).
    Tries strict parsing first (requires ``\\boxed{}``) then lenient.
    Strips ``<think>`` blocks from thinking models (e.g. Qwen3).
    """
    completion = strip_thinking(completion)

    if completion.lower().strip() == solution.lower().strip():
        return 1.0

    gold_parsed = parse(solution)
    if not gold_parsed:
        return None

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
