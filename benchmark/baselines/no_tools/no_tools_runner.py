"""
no_tools_runner.py — Six-stage pipeline baseline with NO external API calls.

All six stages use only the LLM (GPT-4o). Stages 3, 4, and 5 that normally
call PubMed, RxNorm, and OpenFDA are replaced with LLM structured output using
GPT-4o's parametric medical knowledge.

Baseline type: No-Tools Pipeline Ablation
Research question: Does external medical grounding (RxNorm/PubMed/OpenFDA)
improve performance over a pipeline that uses only LLM parametric knowledge?
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jsonschema
from pydantic import BaseModel, Field

BENCHMARK_ROOT = Path(os.environ.get("BENCHMARK_ROOT", Path(__file__).resolve().parent.parent.parent))
SCHEMAS_DIR    = BENCHMARK_ROOT / "shared" / "schemas"
INPUT_SCHEMA_PATH = SCHEMAS_DIR / "input_schema.json"

# llm.py lives at benchmark/runner/llm.py on host, or /app/llm.py in container (copied flat)
sys.path.insert(0, str(BENCHMARK_ROOT / "runner"))
sys.path.insert(0, str(BENCHMARK_ROOT))

from llm import get_llm  # noqa: E402


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DiagnosisList(BaseModel):
    class Diagnosis(BaseModel):
        condition: str
        rationale: str
    diagnoses: list[Diagnosis]


class NormalizedMedNoTools(BaseModel):
    original: str
    rxnorm_id: str | None = None       # Always None — no RxNorm API available
    ingredient: str | None = Field(None, description="Generic ingredient name from LLM knowledge")


class NormalizedMedList(BaseModel):
    medications: list[NormalizedMedNoTools]


class DrugInteractionNoTools(BaseModel):
    drug_a: str
    drug_b: str
    severity: str = Field(default="unknown")
    recommendation: str


class DrugInteractionList(BaseModel):
    interactions: list[DrugInteractionNoTools]


class SOAPReport(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def node_transcription_cleanup(state: dict) -> dict:
    """Stage 1: identical to main agent — LLM only.

    Input state keys:  patient_transcript
    Output key:        transcription_cleaned (str)
    """
    print("[Stage 1] transcription_cleanup (no-tools)", file=sys.stderr)
    llm = get_llm()
    response = llm.invoke([
        {"role": "system", "content": (
            "You are a medical transcription editor. Clean the following raw patient-doctor "
            "transcript. Fix transcription errors, remove filler words (um, uh, like), "
            "add proper punctuation, and label each speaker turn as 'Doctor:' or 'Patient:'. "
            "Return ONLY the cleaned transcript text."
        )},
        {"role": "user", "content": state["patient_transcript"]},
    ])
    return {"transcription_cleaned": response.content}


def node_clinical_summarization(state: dict) -> dict:
    """Stage 2: identical to main agent — LLM only.

    Input state keys:  transcription_cleaned, chart_notes, patient_history
    Output key:        clinical_summary (str)
    """
    print("[Stage 2] clinical_summarization (no-tools)", file=sys.stderr)
    llm = get_llm()
    context = (
        f"Cleaned Transcript:\n{state.get('transcription_cleaned', '')}\n\n"
        f"Chart Notes:\n{state.get('chart_notes', '')}\n\n"
        f"Patient History:\n{state.get('patient_history', '')}"
    )
    response = llm.invoke([
        {"role": "system", "content": (
            "You are a clinical summarization assistant. Produce a concise 2-4 sentence "
            "clinician-facing summary covering the chief complaint, relevant history, "
            "current medications, and key findings. Return ONLY the summary text."
        )},
        {"role": "user", "content": context},
    ])
    return {"clinical_summary": response.content}


def node_differential_diagnosis(state: dict) -> dict:
    """Stage 3: LLM reasoning only — no PubMed lookup. pmid is always None.

    Input state keys:  clinical_summary, patient_history
    Output key:        differential_diagnosis (list[dict])
    """
    print("[Stage 3] differential_diagnosis (no-tools — pmid=null)", file=sys.stderr)
    llm = get_llm()
    context = (
        f"Clinical Summary:\n{state.get('clinical_summary', '')}\n\n"
        f"Patient History:\n{state.get('patient_history', '')}"
    )
    result = llm.with_structured_output(DiagnosisList).invoke([
        {"role": "system", "content": (
            "You are a diagnostic reasoning assistant. Given the clinical summary and patient "
            "history, produce the top 3 most likely differential diagnoses ranked by probability. "
            "For each, provide the condition name and a one-sentence rationale."
        )},
        {"role": "user", "content": context},
    ])
    return {
        "differential_diagnosis": [
            {"condition": dx.condition, "pmid": None, "rationale": dx.rationale}
            for dx in result.diagnoses
        ]
    }


def node_medication_normalization(state: dict) -> dict:
    """Stage 4: LLM normalizes medications from parametric knowledge. rxnorm_id is always None.

    Input state keys:  medication_list
    Output key:        normalized_medications (list[dict])
    """
    print("[Stage 4] medication_normalization (no-tools — rxnorm_id=null)", file=sys.stderr)
    llm = get_llm()
    med_list = state.get("medication_list", [])
    result = llm.with_structured_output(NormalizedMedList).invoke([
        {"role": "system", "content": (
            "For each medication in the list, identify the generic ingredient name using your "
            "medical knowledge. Set rxnorm_id to null for all entries — no database is available."
        )},
        {"role": "user", "content": f"Medication list: {json.dumps(med_list)}"},
    ])
    return {
        "normalized_medications": [
            {"original": m.original, "rxnorm_id": None, "ingredient": m.ingredient}
            for m in result.medications
        ]
    }


def node_drug_interaction_check(state: dict) -> dict:
    """Stage 5: LLM identifies interactions from medical knowledge. No OpenFDA call.

    Input state keys:  normalized_medications
    Output key:        drug_interactions (list[dict])
    """
    print("[Stage 5] drug_interaction_check (no-tools — LLM knowledge)", file=sys.stderr)
    llm = get_llm()
    normalized = state.get("normalized_medications", [])
    ingredients = [m.get("ingredient") for m in normalized if m.get("ingredient")]
    if len(ingredients) < 2:
        return {"drug_interactions": []}
    result = llm.with_structured_output(DrugInteractionList).invoke([
        {"role": "system", "content": (
            "Identify clinically significant drug-drug interactions among the listed medications "
            "using your medical knowledge. Only include interactions that are well-documented. "
            "Set severity to 'unknown' if the exact grade is uncertain."
        )},
        {"role": "user", "content": f"Medications: {json.dumps(ingredients)}"},
    ])
    return {
        "drug_interactions": [
            {
                "drug_a": i.drug_a,
                "drug_b": i.drug_b,
                "severity": i.severity,
                "recommendation": i.recommendation,
            }
            for i in result.interactions
        ]
    }


def node_final_report_generation(state: dict) -> dict:
    """Stage 6: identical to main agent — LLM structured output.

    Input state keys:  all prior stage outputs + patient_history + chart_notes
    Output key:        final_report (dict with keys: subjective, objective, assessment, plan)
    """
    print("[Stage 6] final_report_generation (no-tools)", file=sys.stderr)
    llm = get_llm()
    context = (
        f"Cleaned Transcript:\n{state.get('transcription_cleaned', '')}\n\n"
        f"Clinical Summary:\n{state.get('clinical_summary', '')}\n\n"
        f"Patient History:\n{state.get('patient_history', '')}\n\n"
        f"Chart Notes:\n{state.get('chart_notes', '')}\n\n"
        f"Differential Diagnosis:\n{json.dumps(state.get('differential_diagnosis', []), indent=2)}\n\n"
        f"Normalized Medications:\n{json.dumps(state.get('normalized_medications', []), indent=2)}\n\n"
        f"Drug Interactions:\n{json.dumps(state.get('drug_interactions', []), indent=2)}"
    )
    result = llm.with_structured_output(SOAPReport).invoke([
        {"role": "system", "content": (
            "You are a physician writing a SOAP note. Using all the clinical data provided, "
            "generate a comprehensive SOAP note with four sections: Subjective, Objective, "
            "Assessment, and Plan. Each section should be a detailed paragraph."
        )},
        {"role": "user", "content": context},
    ])
    return {"final_report": result.model_dump()}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

STAGE_NODES = [
    node_transcription_cleanup,
    node_clinical_summarization,
    node_differential_diagnosis,
    node_medication_normalization,
    node_drug_interaction_check,
    node_final_report_generation,
]


def run_pipeline(input_data: dict) -> dict:
    """Run the no-tools pipeline on a pre-loaded input dict.

    Args:
        input_data: Validated input dict matching input_schema.json.

    Returns:
        Full state dict including all six pipeline output keys.
    """
    with INPUT_SCHEMA_PATH.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.validate(instance=input_data, schema=schema)
    state: dict = {**input_data}
    for node_fn in STAGE_NODES:
        state.update(node_fn(state))
    print(f"[NoTools] Pipeline complete for: {input_data['case_id']}", file=sys.stderr)
    return state
