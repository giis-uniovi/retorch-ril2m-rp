import logging
import os
import random

from openai import OpenAI

from ril2m.helpers.llmClient import LLMClient
from ril2m.helpers.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenAICompatibleClient(LLMClient):
    """Chat client for OpenAI and OpenRouter APIs.

    Reads credentials from environment variables:
      - OpenAI:      OPENAI_API_KEY
      - OpenRouter:  OPENROUTER_API_KEY

    Note: this client only supports chat(). Embeddings always use OllamaClient
    so that ChromaDB vector stores remain consistent across providers.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.5,
        api_key: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.provider = provider

        if provider == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Export it before running with provider='openai'."
                )
            self.client = OpenAI(api_key=key)
        elif provider == "openrouter":
            key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable is not set. "
                    "Export it before running with provider='openrouter'."
                )
            self.client = OpenAI(api_key=key, base_url=_OPENROUTER_BASE_URL)
        else:
            raise ValueError(
                f"Unknown provider {provider!r}. Valid values: 'openai', 'openrouter'."
            )

        logger.debug(
            "OpenAICompatibleClient initialised: provider=%s  model=%s  temperature=%s",
            provider, model, temperature,
        )

    def chat(self, prompt: str, seed: int = None) -> str:
        if seed is None:
            seed = random.randint(0, 2 ** 31 - 1)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            seed=seed,
        )
        try:
            return response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            logger.exception(
                "chat: unexpected response structure from %s (model=%s): %s",
                self.provider, self.model, response,
            )
            raise ValueError(
                f"chat: unexpected {self.provider} response — {exc}"
            ) from exc
