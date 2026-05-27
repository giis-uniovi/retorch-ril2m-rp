from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for LLM chat clients."""

    @abstractmethod
    def chat(self, prompt: str, seed: int = None) -> str:
        """Send a prompt and return the response text."""
        ...
