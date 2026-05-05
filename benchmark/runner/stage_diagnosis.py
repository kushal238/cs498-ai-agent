# benchmark/runner/stage_diagnosis.py
"""Stage 3: Generate a PubMed-backed differential diagnosis."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field
import llm_client

# Make shared/ importable from both container (/app/) and host (benchmark/)
_benchmark_root = Path(os.environ.get("BENCHMARK_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(_benchmark_root))
from shared.tools.pubmed import search_pubmed  # noqa: E402


class _Diagnosis(BaseModel):
    condition: str = Field(description="Name of the clinical condition")
    rationale: str = Field(description="One-sentence explanation of why this condition fits")


class DiagnosisResult(BaseModel):
    reasoning: str = Field(description="Explanation of the diagnostic reasoning process")
    confidence: str = Field(description="high, medium, or low")
    diagnoses: list[_Diagnosis] = Field(description="Top 3 conditions ranked by likelihood")


def run(context: dict) -> dict:
    """Generate differential diagnoses and fetch supporting PubMed PMIDs.

    Args:
        context: Must contain 'clinical_summary' and 'patient_history'.
                 May contain 'scratchpad_summary' and '_validation_error'.

    Returns:
        {"reasoning": str, "confidence": str,
         "output": {"differential_diagnosis": list[dict]}}
    """
    validation_hint = ""
    if context.get("_validation_error"):
        validation_hint = (
            f"\n\nYour previous output was invalid: {context['_validation_error']}. "
            "Please fix it."
        )

    scratchpad_section = ""
    if context.get("scratchpad_summary"):
        scratchpad_section = (
            f"\n\nPrior stage reasoning:\n{context['scratchpad_summary']}"
        )

    content = (
        f"Clinical Summary:\n{context.get('clinical_summary', '')}\n\n"
        f"Patient History:\n{json.dumps(context.get('patient_history', {}), indent=2)}"
        + scratchpad_section
        + validation_hint
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a diagnostic reasoning assistant. Given the clinical summary "
                "and patient history, produce the top 3 most likely differential "
                "diagnoses ranked by probability. For each, give the condition name "
                "and a one-sentence rationale."
            ),
        },
        {"role": "user", "content": content},
    ]
    result = llm_client.chat_structured(messages, DiagnosisResult)

    diagnoses = []
    for dx in result.diagnoses:
        pmids = search_pubmed(f"{dx.condition} diagnosis", max_results=1)
        diagnoses.append({
            "condition": dx.condition,
            "pmid": pmids[0] if pmids else None,
            "rationale": dx.rationale,
        })

    return {
        "reasoning": result.reasoning,
        "confidence": result.confidence,
        "output": {"differential_diagnosis": diagnoses},
    }
