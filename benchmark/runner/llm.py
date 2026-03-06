"""Shared LLM client for pipeline nodes."""

from __future__ import annotations

import os

_llm = None


def get_llm():
    """Return a shared ChatOpenAI instance, lazily initialized."""
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI

        _llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.environ["OPENAI_API_KEY"],
        )
    return _llm
