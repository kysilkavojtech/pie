"""Abstract base classes for evaluation datasets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class AnswerType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    INTEGER = "integer"
    MATH_EXPRESSION = "math_expression"
    SHORT_TEXT = "short_text"


@dataclass
class Question:
    """A single evaluation question."""
    id: str
    question_text: str
    correct_answer: str
    answer_type: AnswerType
    choices: dict[str, str] | None = None  # {"A": "...", "B": "..."} for MC
    metadata: dict = field(default_factory=dict)


class Dataset(ABC):
    """Abstract evaluation dataset.

    Each dataset knows how to load its questions from a source (typically
    HuggingFace ``datasets``), format them into prompts, and evaluate
    whether a model's free-text output is correct.
    """

    def __init__(self, name: str, answer_type: AnswerType):
        self.name = name
        self.answer_type = answer_type

    @abstractmethod
    def load(self, limit: int | None = None) -> list[Question]:
        """Load questions. *limit* caps the count (useful for quick tests)."""

    @abstractmethod
    def format_prompt(self, question: Question) -> str:
        """Turn a Question into the user-message string sent to the LLM."""

    @abstractmethod
    def evaluate(self, question: Question, raw_output: str) -> tuple[bool, str]:
        """Judge a model response.

        Returns ``(correct, extracted_answer)`` where *extracted_answer*
        is the parsed answer string (for logging/debugging).
        """
