import ollama
import requests
import logging
import random

from ril2m.helpers import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = "http://ollama-gpu:11434", temperature: float = 0.5,
                 embed_model: str = "nomic-embed-text:v1.5", model: str = "codellama"):
        self.base_url = base_url
        self.temperature = temperature
        self.embed_model = embed_model
        self.model = model
        self.client = ollama.Client(host=self.base_url)
        # List available models for debugging purposes
        logger.debug("Available LLM models:")
        for model in self.client.list()['models']:
            logger.debug(model)

    def embed(self, text: str) -> list[float]:
        """Genera embeddings con Ollama."""
        # ollama 0.20+ reduced the default num_ctx for the embed endpoint and ignores options overrides.
        # 6000 chars ≈ 2000 tokens at 3 chars/token (dense Java) — safely under the 2048-token default.
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.embed_model, "input": text[:6000]},
            timeout=60,
        )
        response.raise_for_status()
        try:
            data = response.json()
            return data["embeddings"][0]
        except (KeyError, IndexError, ValueError) as exc:
            logger.exception(
                "embed: unexpected response structure from %s (model=%s): %s",
                self.base_url, self.embed_model, response.text[:500],
            )
            raise ValueError(
                f"embed: unexpected Ollama response — {exc}"
            ) from exc

    def chat(self, prompt: str, seed: int = None) -> str:
        if seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        response = self.client.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=self.model,
            options={"temperature": self.temperature, "seed": seed},
        )
        try:
            return response["message"]["content"]
        except (KeyError, TypeError) as exc:
            logger.exception(
                "chat: unexpected response structure from %s (model=%s): %s",
                self.base_url, self.model, response,
            )
            raise ValueError(
                f"chat: unexpected Ollama response — {exc}"
            ) from exc
