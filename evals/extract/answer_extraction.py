"""Answer extraction utilities for parsing free-text LLM output."""

import re
import string


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (e.g. from Qwen3 thinking mode)."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in stripped:
        _, _, after = stripped.partition("</think>")
        if after.strip():
            return after.strip()
    return stripped if stripped else text


def extract_multiple_choice(
    text: str, valid_choices: str = "ABCDE"
) -> str | None:
    """Extract a multiple-choice letter from free-text model output.

    Tries, in order:
      1. Entire output (stripped) is a single valid letter.
      2. Pattern like "the answer is (C)" / "Answer: C".
      3. Last standalone valid letter in the text.
    """
    text = strip_thinking(text)
    stripped = text.strip().rstrip(".")
    upper_valid = valid_choices.upper()

    # 1. Single letter
    if len(stripped) == 1 and stripped.upper() in upper_valid:
        return stripped.upper()

    # 2. "the answer is" / "answer:" patterns
    pattern = (
        r"(?:the\s+answer\s+is|answer\s*[:=]|my\s+answer\s+is)\s*"
        r"\(?([" + upper_valid + r"])\)?"
    )
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3. Last standalone letter (word-boundary bounded)
    all_matches = re.findall(
        r"\b([" + upper_valid + r"])\b", text.upper()
    )
    if all_matches:
        return all_matches[-1]

    return None


def extract_integer(
    text: str, min_val: int = 0, max_val: int = 999
) -> int | None:
    """Extract an integer answer from free-text model output.

    Tries:
      1. Entire output is a number in [min_val, max_val].
      2. Number after "the answer is" / "answer:" / boxed.
      3. Last number in the text within range.
    """
    stripped = text.strip()

    # 1. Whole output is a number
    try:
        n = int(stripped)
        if min_val <= n <= max_val:
            return n
    except ValueError:
        pass

    # 2. After "answer is" style
    m = re.search(
        r"(?:the\s+answer\s+is|answer\s*[:=])\s*(\d+)", text, re.IGNORECASE
    )
    if m:
        n = int(m.group(1))
        if min_val <= n <= max_val:
            return n

    # 2b. Boxed
    m = re.search(r"\\boxed\{(\d+)\}", text)
    if m:
        n = int(m.group(1))
        if min_val <= n <= max_val:
            return n

    # 3. Last number in range
    for m in reversed(list(re.finditer(r"\b(\d+)\b", text))):
        n = int(m.group(1))
        if min_val <= n <= max_val:
            return n

    return None


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip articles and punctuation."""
    t = text.lower().strip()
    # Remove articles
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    # Remove punctuation
    t = t.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    t = " ".join(t.split())
    return t


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth (SQuAD-style)."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
