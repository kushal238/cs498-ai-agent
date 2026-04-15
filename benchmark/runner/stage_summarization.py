"""Stage 2: Produce a clinician-facing clinical summary."""
from __future__ import annotations

import json
from pydantic import BaseModel, Field
import llm_client


class SummarizationResult(BaseModel):
    reasoning: str = Field(description="Key findings identified and decisions made")
    confidence: str = Field(description="high, medium, or low")
    clinical_summary: str = Field(description="2-4 sentence clinician-facing summary")


def run(context: dict) -> dict:
    """Summarize the cleaned transcript and chart notes.

    Args:
        context: Must contain 'transcription_cleaned', 'chart_notes', 'patient_history'.
                 May contain '_validation_error'.

    Returns:
        {"reasoning": str, "confidence": str, "output": {"clinical_summary": str}}
    """
    validation_hint = ""
    if context.get("_validation_error"):
        validation_hint = (
            f"\n\nYour previous output was invalid: {context['_validation_error']}. "
            "Please fix it."
        )

    content = (
        f"Cleaned Transcript:\n{context.get('transcription_cleaned', '')}\n\n"
        f"Chart Notes:\n{context.get('chart_notes', '')}\n\n"
        f"Patient History:\n{json.dumps(context.get('patient_history', {}), indent=2)}"
        + validation_hint
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical summarization assistant. Produce a concise "
                "2-4 sentence clinician-facing summary covering the chief complaint, "
                "relevant history, current medications, and key findings."
            ),
        },
        {"role": "user", "content": content},
    ]
    result = llm_client.chat_structured(messages, SummarizationResult)
    return {
        "reasoning": result.reasoning,
        "confidence": result.confidence,
        "output": {"clinical_summary": result.clinical_summary},
    }
