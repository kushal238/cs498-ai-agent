"""Stage 1: Clean raw patient transcript."""
from __future__ import annotations

from pydantic import BaseModel, Field
import llm_client


class TranscriptionResult(BaseModel):
    reasoning: str = Field(description="Brief explanation of edits made and confidence level")
    confidence: str = Field(description="high, medium, or low")
    transcription_cleaned: str = Field(description="The cleaned transcript text with Doctor:/Patient: labels")


def run(context: dict) -> dict:
    """Clean the raw patient-doctor transcript.

    Args:
        context: Must contain 'patient_transcript'. May contain '_validation_error'.

    Returns:
        {"reasoning": str, "confidence": str, "output": {"transcription_cleaned": str}}
    """
    validation_hint = ""
    if context.get("_validation_error"):
        validation_hint = (
            f"\n\nYour previous output was invalid: {context['_validation_error']}. "
            "Please fix it."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a medical transcription editor. Clean up the raw "
                "patient-doctor transcript: fix transcription errors, remove "
                "filler words (um, uh, like, you know), add proper punctuation, "
                "and label each speaker turn as 'Doctor:' or 'Patient:'."
            ),
        },
        {
            "role": "user",
            "content": context["patient_transcript"] + validation_hint,
        },
    ]
    result = llm_client.chat_structured(messages, TranscriptionResult)
    return {
        "reasoning": result.reasoning,
        "confidence": result.confidence,
        "output": {"transcription_cleaned": result.transcription_cleaned},
    }
