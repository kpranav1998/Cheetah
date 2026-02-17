from __future__ import annotations

from openai import OpenAI

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Wrapper for LLM API calls."""

    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.llm_model

        if self.provider == "openai":
            self.client = OpenAI(api_key=settings.llm_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def query(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return f"LLM analysis unavailable: {e}"
