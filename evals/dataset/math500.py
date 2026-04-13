"""MATH-500 dataset (math problems with LaTeX solutions)."""

from datasets import load_dataset

from .base import AnswerType, Dataset, Question
from ..extract.math_verify_utils import accuracy_reward


class MATH500(Dataset):
    """HuggingFace MATH-500 benchmark.

    500 representative math problems spanning algebra, geometry, number theory,
    etc.  Solutions are LaTeX expressions; evaluation uses symbolic comparison
    via ``math_verify``.
    """

    def __init__(self):
        super().__init__("math500", AnswerType.MATH_EXPRESSION)

    def load(self, limit: int | None = None) -> list[Question]:
        ds = load_dataset("HuggingFaceTB/MATH-500", split="test")
        questions: list[Question] = []
        for row in ds:
            questions.append(
                Question(
                    id=f"math500_{len(questions)}",
                    question_text=row["problem"],
                    correct_answer=row["answer"],
                    answer_type=AnswerType.MATH_EXPRESSION,
                    metadata={
                        "level": row.get("level", ""),
                        "subject": row.get("subject", ""),
                    },
                )
            )
            if limit and len(questions) >= limit:
                break
        return questions

    def format_prompt(self, question: Question) -> str:
        return (
            "Solve the following math problem. "
            "Put your final answer in \\boxed{}.\n\n"
            f"{question.question_text}"
        )

    def evaluate(self, question: Question, raw_output: str) -> tuple[bool, str]:
        reward = accuracy_reward(raw_output, question.correct_answer)
        if reward is None:
            # Gold solution unparseable — skip (count as incorrect)
            return False, ""
        correct = reward >= 1.0
        # Extract what we can for logging
        extracted = raw_output.strip()[:200]
        return correct, extracted
