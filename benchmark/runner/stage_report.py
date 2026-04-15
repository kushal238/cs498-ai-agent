# benchmark/runner/stage_report.py
"""Stage 6: Generate a SOAP-format clinical report, memory-informed."""
from __future__ import annotations

import json

from pydantic import BaseModel, Field
import llm_client


class SOAPReportResult(BaseModel):
    reasoning: str = Field(description="Brief explanation of how prior stage outputs shaped this report")
    confidence: str = Field(description="high, medium, or low")
    subjective: str = Field(description="Patient's reported symptoms and history in their own words")
    objective: str = Field(description="Measurable clinical findings (vitals, labs, exam)")
    assessment: str = Field(description="Clinician's interpretation and working diagnosis")
    plan: str = Field(description="Proposed treatment, follow-up, and patient instructions")


def run(context: dict) -> dict:
    """Generate a comprehensive SOAP clinical note.

    This stage receives the accumulated scratchpad from all prior stages
    (via context['scratchpad_summary']) so the LLM knows what was uncertain
    or degraded earlier.

    Args:
        context: Must contain all prior stage outputs plus 'patient_history',
                 'chart_notes'. May contain 'scratchpad_summary' and '_validation_error'.

    Returns:
        {"reasoning": str, "confidence": str,
         "output": {"final_report": {"subjective", "objective", "assessment", "plan"}}}
    """
    scratchpad_section = ""
    if context.get("scratchpad_summary"):
        scratchpad_section = (
            f"\n\nAgent reasoning from prior stages:\n{context['scratchpad_summary']}\n"
        )

    validation_hint = ""
    if context.get("_validation_error"):
        validation_hint = (
            f"\n\nYour previous output was invalid: {context['_validation_error']}. "
            "Please fix it."
        )

    content = (
        f"Cleaned Transcript:\n{context.get('transcription_cleaned', '')}\n\n"
        f"Clinical Summary:\n{context.get('clinical_summary', '')}\n\n"
        f"Patient History:\n{json.dumps(context.get('patient_history', {}), indent=2)}\n\n"
        f"Chart Notes:\n{context.get('chart_notes', '')}\n\n"
        f"Differential Diagnosis:\n"
        f"{json.dumps(context.get('differential_diagnosis', []), indent=2)}\n\n"
        f"Normalized Medications:\n"
        f"{json.dumps(context.get('normalized_medications', []), indent=2)}\n\n"
        f"Drug Interactions:\n"
        f"{json.dumps(context.get('drug_interactions', []), indent=2)}"
        + scratchpad_section
        + validation_hint
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a physician writing a SOAP note. Using all the clinical "
                "data provided, generate a comprehensive SOAP note with four sections: "
                "Subjective, Objective, Assessment, and Plan. Each section should be "
                "a detailed paragraph."
            ),
        },
        {"role": "user", "content": content},
    ]
    result = llm_client.chat_structured(messages, SOAPReportResult)
    return {
        "reasoning": result.reasoning,
        "confidence": result.confidence,
        "output": {
            "final_report": {
                "subjective": result.subjective,
                "objective": result.objective,
                "assessment": result.assessment,
                "plan": result.plan,
            }
        },
    }
