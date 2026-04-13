"""Abstract base classes for inference engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationParams:
    """Parameters controlling text generation."""
    max_tokens: int = 2048
    temperature: float = 0.0  # greedy by default for reproducibility
    top_p: float = 1.0
    system: str = "You are a helpful assistant."


class Engine(ABC):
    """Abstract inference engine.

    Each engine knows how to connect to a serving backend, send a prompt,
    and return the generated text.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name

    @abstractmethod
    async def setup(self) -> None:
        """One-time setup: connect, install inferlet, etc."""

    @abstractmethod
    async def generate(self, prompt: str, params: GenerationParams) -> str:
        """Send a prompt and return the generated text."""

    @abstractmethod
    async def teardown(self) -> None:
        """Release resources (close connections, etc.)."""

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, *exc):
        await self.teardown()
