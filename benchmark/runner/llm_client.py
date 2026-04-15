"""Direct OpenAI SDK wrapper — replaces llm.py (LangChain removed)."""
from __future__ import annotations

import os
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Export it before running:\n"
                "  export OPENAI_API_KEY=sk-..."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def chat(messages: list[dict], model: str = "gpt-4o") -> str:
    """Send a chat request and return the text content of the reply."""
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


def chat_structured(
    messages: list[dict],
    response_format: Type[T],
    model: str = "gpt-4o",
) -> T:
    """Send a chat request and parse the reply into a Pydantic model."""
    response = _get_client().beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=0,
    )
    return response.choices[0].message.parsed
