"""Shared LLM client for pipeline nodes."""

from __future__ import annotations

import os

_llm = None


def get_llm():
    """Return a shared ChatOpenAI instance, lazily initialized."""
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Export it on the host before running the harness:\n"
                "  export OPENAI_API_KEY=sk-..."
            )
        _llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=api_key,
        )
    return _llm
