"""ARC-Easy and ARC-Challenge datasets (multiple-choice science questions)."""

from datasets import load_dataset

from .base import AnswerType, Dataset, Question
from ..extract.answer_extraction import extract_multiple_choice


class ARCDataset(Dataset):
    """AI2 Reasoning Challenge dataset.

    Loads from ``allenai/ai2_arc`` on HuggingFace.  Use *config* to pick
    ``"ARC-Easy"`` or ``"ARC-Challenge"``.
    """

    def __init__(self, name: str, config: str = "ARC-Easy", split: str = "test"):
        super().__init__(name, AnswerType.MULTIPLE_CHOICE)
        self.config = config
        self.split = split

    def load(self, limit: int | None = None) -> list[Question]:
        ds = load_dataset("allenai/ai2_arc", self.config, split=self.split)
        questions: list[Question] = []
        for row in ds:
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            choices = dict(zip(labels, texts))
            questions.append(
                Question(
                    id=row["id"],
                    question_text=row["question"],
                    correct_answer=row["answerKey"],
                    answer_type=AnswerType.MULTIPLE_CHOICE,
                    choices=choices,
                    metadata={"dataset": self.config},
                )
            )
            if limit and len(questions) >= limit:
                break
        return questions

    def format_prompt(self, question: Question) -> str:
        assert question.choices is not None
        lines = [f"Question: {question.question_text}"]
        for label in sorted(question.choices):
            lines.append(f"{label}) {question.choices[label]}")
        lines.append("")
        lines.append("Answer with just the letter.")
        return "\n".join(lines)

    def evaluate(self, question: Question, raw_output: str) -> tuple[bool, str]:
        valid = "".join(sorted(question.choices)) if question.choices else "ABCD"
        extracted = extract_multiple_choice(raw_output, valid_choices=valid)
        if extracted is None:
            return False, ""
        return extracted == question.correct_answer, extracted


class ARCEasy(ARCDataset):
    def __init__(self):
        super().__init__("arc_easy", config="ARC-Easy")


class ARCChallenge(ARCDataset):
    def __init__(self):
        super().__init__("arc_challenge", config="ARC-Challenge")
