"""
zero_shot_main.py — Zero-shot single-prompt baseline.

Reads input JSON from stdin. Sends ONE prompt to GPT-4o asking for all six
pipeline outputs simultaneously. No pipeline decomposition. No external tools.
Prints result JSON to stdout.

Baseline type: Single-Prompt Agent (Zero-Shot)
Research question: Can a strong LLM complete the full clinical workflow without
multi-stage orchestration or external medical tools?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

# Allow running from repo root or inside container (/app/llm_client.py copied flat)
_here = Path(__file__).resolve().parent
_runner_dir = _here.parent.parent / "runner"
sys.path.insert(0, str(_runner_dir))
sys.path.insert(0, str(_here.parent.parent))

from llm_client import chat_structured  # noqa: E402


# ---------------------------------------------------------------------------
# Output schema (mirrors ground_truth_schema.json)
# ---------------------------------------------------------------------------

class DifferentialDiagnosisItem(BaseModel):
    condition: str
    pmid: str | None = None
    rationale: str


class NormalizedMedication(BaseModel):
    original: str
    rxnorm_id: str | None = None
    ingredient: str | None = None


class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: str = "unknown"
    recommendation: str


class SOAPReport(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str


class ClinicalWorkflowOutput(BaseModel):
    """Full output for all six pipeline stages — produced in a single LLM call."""
    transcription_cleaned: str = Field(
        description="Cleaned, punctuated transcript with speaker labels (Doctor:/Patient:)"
    )
    clinical_summary: str = Field(
        description="2-4 sentence clinician-facing summary of the case"
    )
    differential_diagnosis: list[DifferentialDiagnosisItem] = Field(
        description="Top 3 differential diagnoses ranked by likelihood. Set pmid to null."
    )
    normalized_medications: list[NormalizedMedication] = Field(
        description="Medication list with generic ingredient names. Set rxnorm_id to null."
    )
    drug_interactions: list[DrugInteraction] = Field(
        description="Clinically significant drug-drug interactions from the medication list."
    )
    final_report: SOAPReport = Field(
        description="SOAP-format clinical report synthesizing all findings."
    )


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def run_zero_shot(input_data: dict) -> dict:
    """Run the zero-shot baseline: one prompt → all six outputs.

    Args:
        input_data: Validated input dict matching input_schema.json.

    Returns:
        Dict with all six ground_truth_schema.json output keys.
    """
    prompt = (
        f"Patient Transcript:\n{input_data.get('patient_transcript', '')}\n\n"
        f"Chart Notes:\n{input_data.get('chart_notes', '')}\n\n"
        f"Patient History:\n{json.dumps(input_data.get('patient_history', {}), indent=2)}\n\n"
        f"Medication List:\n{json.dumps(input_data.get('medication_list', []), indent=2)}\n\n"
        "Produce all required clinical documentation:\n"
        "1. A cleaned transcript with Doctor:/Patient: speaker labels\n"
        "2. A 2-4 sentence clinical summary\n"
        "3. Top 3 differential diagnoses ranked by likelihood (pmid=null)\n"
        "4. Normalized medications with generic ingredient names (rxnorm_id=null)\n"
        "5. Clinically significant drug-drug interactions\n"
        "6. A complete SOAP report"
    )

    messages = [
        {"role": "system", "content": (
            "You are an expert clinical documentation AI. Produce all six required "
            "outputs accurately and concisely based solely on the provided patient data."
        )},
        {"role": "user", "content": prompt},
    ]
    result = chat_structured(messages, ClinicalWorkflowOutput)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    raw = sys.stdin.read()
    if not raw:
        print(json.dumps({"error": "No input on stdin"}), file=sys.stderr)
        sys.exit(1)
    try:
        input_data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"JSON parse failed: {exc}"}), file=sys.stderr)
        sys.exit(1)

    result = run_zero_shot(input_data)
    output = {"case_id": input_data.get("case_id"), **result}
    print(json.dumps(output))


if __name__ == "__main__":
    main()
