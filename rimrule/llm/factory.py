from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rimrule.config import LLMConfig

__all__ = ["get_llm"]


def get_llm(config: LLMConfig) -> ChatOpenAI:
    """Build the LangChain chat model for one LLM role.

    The API key is read from the environment variable named by
    ``config.api_key_env`` (``OPENAI_API_KEY`` by default). Setting
    ``config.base_url`` points the client at any OpenAI-compatible endpoint.
    Retries and timeouts come from ``config``.

    Args:
        config: The role configuration (provider, model, sampling, retries).

    Raises:
        ValueError: If the provider is not ``openai``.
    """
    provider = config.provider.lower()
    if provider == "openai":
        api_key = os.getenv(config.api_key_env)
        return ChatOpenAI(
            model=config.model,
            api_key=SecretStr(api_key) if api_key else None,
            base_url=config.base_url,
            temperature=config.temperature,
            max_retries=config.max_retries,
            timeout=config.timeout_seconds,
            reasoning_effort=config.reasoning_effort,
        )
    raise ValueError(f"Unsupported LLM provider: {config.provider}")
