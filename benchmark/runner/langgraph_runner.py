"""
LangGraph runner stub for the clinical workflow benchmark.

This file is part of the BENCHMARK, not an agent implementation.
It defines the interface that a submitted agent must fulfill — each node
function must accept the state dict and return a partial update.

Intended LangGraph node structure (pseudocode):
----------------------------------------------------------------------
graph = StateGraph(ClinicalWorkflowState)

graph.add_node("transcription_cleanup",   node_transcription_cleanup)
graph.add_node("clinical_summarization",  node_clinical_summarization)
graph.add_node("differential_diagnosis",  node_differential_diagnosis)
graph.add_node("medication_normalization",node_medication_normalization)
graph.add_node("drug_interaction_check",  node_drug_interaction_check)
graph.add_node("final_report_generation", node_final_report_generation)

graph.set_entry_point("transcription_cleanup")
graph.add_edge("transcription_cleanup",    "clinical_summarization")
graph.add_edge("clinical_summarization",   "differential_diagnosis")
graph.add_edge("differential_diagnosis",   "medication_normalization")
graph.add_edge("medication_normalization",  "drug_interaction_check")
graph.add_edge("drug_interaction_check",   "final_report_generation")
graph.add_edge("final_report_generation",  END)

app = graph.compile()
----------------------------------------------------------------------

Each node receives a ClinicalWorkflowState TypedDict and returns a
partial update to that state.  State fields map 1-to-1 with the
ground_truth.json schema fields plus the raw input.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jsonschema
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# In the agent container, BENCHMARK_ROOT is set to /app via ENV.
# On the host, it defaults to two levels up from this file (benchmark/).
BENCHMARK_ROOT = Path(os.environ.get("BENCHMARK_ROOT", Path(__file__).resolve().parent.parent))
SCHEMAS_DIR = BENCHMARK_ROOT / "shared" / "schemas"
INPUT_SCHEMA_PATH = SCHEMAS_DIR / "input_schema.json"

# Ensure shared/ is importable (works both in container /app/ and on host)
sys.path.insert(0, str(BENCHMARK_ROOT))

from llm import get_llm  # noqa: E402
from shared.tools.rxnorm import get_rxcui, normalize_medication_list, check_interactions  # noqa: E402
from shared.tools.pubmed import search_pubmed  # noqa: E402

WORKFLOW_STAGES = [
    "transcription_cleanup",
    "clinical_summarization",
    "differential_diagnosis",
    "medication_normalization",
    "drug_interaction_check",
    "final_report_generation",
]


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

def load_schema(schema_path: Path) -> dict:
    """Load a JSON Schema file from disk."""
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_input(input_data: dict, schema: dict) -> None:
    """Validate input_data against schema; raises jsonschema.ValidationError on failure."""
    jsonschema.validate(instance=input_data, schema=schema)


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def load_case(case_dir: str | Path) -> dict:
    """Load and validate a benchmark case from a folder.

    Args:
        case_dir: Path to the case folder containing input.json.

    Returns:
        The validated input data as a dict.

    Raises:
        FileNotFoundError:           If input.json or the schema file is missing.
        jsonschema.ValidationError:  If input.json does not conform to the schema.
    """
    case_path = Path(case_dir)
    input_path = case_path / "input.json"

    if not input_path.exists():
        raise FileNotFoundError(f"input.json not found in {case_path}")

    with input_path.open("r", encoding="utf-8") as f:
        input_data = json.load(f)

    schema = load_schema(INPUT_SCHEMA_PATH)
    validate_input(input_data, schema)

    print(f"[Runner] Loaded and validated case: {input_data['case_id']}", file=sys.stderr)
    return input_data


# ---------------------------------------------------------------------------
# Workflow node stubs
# ---------------------------------------------------------------------------

def node_transcription_cleanup(state: dict) -> dict:
    """Stage 1: Clean raw patient transcript.

    Input state keys:  patient_transcript
    Output key:        transcription_cleaned (str)
    """
    print("[Stage 1] transcription_cleanup", file=sys.stderr)
    llm = get_llm()
    response = llm.invoke([
        {"role": "system", "content": (
            "You are a medical transcription editor. Clean up the following "
            "raw patient-doctor transcript. Fix transcription errors, remove "
            "filler words (um, uh, like, you know), add proper punctuation, "
            "and label each speaker turn as 'Doctor:' or 'Patient:'. "
            "Return ONLY the cleaned transcript text, nothing else."
        )},
        {"role": "user", "content": state["patient_transcript"]},
    ])
    return {"transcription_cleaned": response.content}


def node_clinical_summarization(state: dict) -> dict:
    """Stage 2: Summarize the cleaned transcript + chart notes.

    Input state keys:  transcription_cleaned, chart_notes, patient_history
    Output key:        clinical_summary (str)
    """
    print("[Stage 2] clinical_summarization", file=sys.stderr)
    llm = get_llm()
    context = (
        f"Cleaned Transcript:\n{state.get('transcription_cleaned', '')}\n\n"
        f"Chart Notes:\n{state.get('chart_notes', '')}\n\n"
        f"Patient History:\n{state.get('patient_history', '')}"
    )
    response = llm.invoke([
        {"role": "system", "content": (
            "You are a clinical summarization assistant. Produce a concise "
            "2-4 sentence clinician-facing summary covering the chief "
            "complaint, relevant history, current medications, and key "
            "findings. Return ONLY the summary text."
        )},
        {"role": "user", "content": context},
    ])
    return {"clinical_summary": response.content}


class DiagnosisList(BaseModel):
    """Structured output for differential diagnosis."""

    class Diagnosis(BaseModel):
        condition: str = Field(description="Name of the clinical condition")
        rationale: str = Field(description="One-sentence explanation of why this condition is on the differential")

    diagnoses: list[Diagnosis] = Field(description="Top 3 candidate conditions ranked by likelihood")


def node_differential_diagnosis(state: dict) -> dict:
    """Stage 3: Generate a PubMed-backed differential diagnosis list.

    Input state keys:  clinical_summary, patient_history
    Output key:        differential_diagnosis (list[dict])
    """
    print("[Stage 3] differential_diagnosis", file=sys.stderr)
    llm = get_llm()
    structured_llm = llm.with_structured_output(DiagnosisList)
    context = (
        f"Clinical Summary:\n{state.get('clinical_summary', '')}\n\n"
        f"Patient History:\n{state.get('patient_history', '')}"
    )
    result = structured_llm.invoke([
        {"role": "system", "content": (
            "You are a diagnostic reasoning assistant. Given the clinical "
            "summary and patient history, produce the top 3 most likely "
            "differential diagnoses ranked by probability. For each, give "
            "the condition name and a one-sentence rationale."
        )},
        {"role": "user", "content": context},
    ])

    diagnoses = []
    for dx in result.diagnoses:
        pmids = search_pubmed(f"{dx.condition} diagnosis", max_results=1)
        diagnoses.append({
            "condition": dx.condition,
            "pmid": pmids[0] if pmids else None,
            "rationale": dx.rationale,
        })
    return {"differential_diagnosis": diagnoses}


def _extract_drug_names(medication_string: str, llm) -> list[str]:
    """Use the LLM to extract generic drug name(s) from a medication string.

    Handles both single drugs and combination products (e.g.
    'oxycodone/acetaminophen (Percocet 10/325 mg)' → ['oxycodone', 'acetaminophen']).
    Also strips dose, route, frequency, weight-based rates (e.g. '10 U/kg/h'),
    and concentration qualifiers (e.g. '160mg/5mL').

    Returns a list of one or more lower-stripped drug name strings.
    Falls back to [medication_string] if the LLM call fails.
    """
    try:
        response = llm.invoke([
            {"role": "system", "content": (
                "You are a pharmacist. Extract the generic drug name(s) from the "
                "medication description. If it is a combination product (e.g. "
                "'oxycodone/acetaminophen'), return each ingredient on its own line. "
                "Strip all doses, routes, frequencies, weight-based rates, "
                "concentrations, and brand names. Return only the drug name(s), "
                "one per line, nothing else."
            )},
            {"role": "user", "content": medication_string},
        ])
        names = [n.strip().lower() for n in response.content.strip().splitlines() if n.strip()]
        return names if names else [medication_string]
    except Exception as exc:
        print(f"[Stage 4] _extract_drug_names failed for {medication_string!r}: {exc}", file=sys.stderr)
        return [medication_string]


def node_medication_normalization(state: dict) -> dict:
    """Stage 4: Normalize medication list via RxNorm.

    An LLM preprocessing step first extracts the bare generic drug name(s) from
    each free-text medication string. Combination products (e.g.
    'oxycodone/acetaminophen (Percocet 10/325 mg)') are split into separate
    entries so each ingredient gets its own RxNorm lookup and appears
    individually in the output — matching the ground truth schema which stores
    one ingredient per entry.

    Input state keys:  medication_list
    Output key:        normalized_medications (list[dict])
    """
    print("[Stage 4] medication_normalization", file=sys.stderr)
    llm = get_llm()
    raw_meds = state.get("medication_list", [])
    normalized = []
    for med in raw_meds:
        drug_names = _extract_drug_names(med, llm)
        for drug_name in drug_names:
            lookup = get_rxcui(drug_name)
            normalized.append({
                "original": med,
                "rxnorm_id": lookup["rxnorm_id"],
                "ingredient": lookup["ingredient"],
            })
    return {"normalized_medications": normalized}


def node_drug_interaction_check(state: dict) -> dict:
    """Stage 5: Check for drug-drug interactions via NIH RxNav.

    Input state keys:  normalized_medications
    Output key:        drug_interactions (list[dict])
    """
    print("[Stage 5] drug_interaction_check", file=sys.stderr)
    normalized = state.get("normalized_medications", [])
    rxcui_list = [m["rxnorm_id"] for m in normalized if m.get("rxnorm_id")]
    interactions = check_interactions(rxcui_list)
    # Strip "description" key -- ground_truth_schema has additionalProperties: false
    for interaction in interactions:
        interaction.pop("description", None)
    return {"drug_interactions": interactions}


class SOAPReport(BaseModel):
    """Structured output for the final SOAP report."""
    subjective: str = Field(description="Patient's reported symptoms and history in their own words")
    objective: str = Field(description="Measurable clinical findings (vitals, labs, exam)")
    assessment: str = Field(description="Clinician's interpretation and working diagnosis")
    plan: str = Field(description="Proposed treatment, follow-up, and patient instructions")


def node_final_report_generation(state: dict) -> dict:
    """Stage 6: Generate a SOAP-format clinical report.

    Input state keys:  all prior stage outputs + patient_history + chart_notes
    Output key:        final_report (dict with keys: subjective, objective, assessment, plan)
    """
    print("[Stage 6] final_report_generation", file=sys.stderr)
    llm = get_llm()
    structured_llm = llm.with_structured_output(SOAPReport)
    context = (
        f"Cleaned Transcript:\n{state.get('transcription_cleaned', '')}\n\n"
        f"Clinical Summary:\n{state.get('clinical_summary', '')}\n\n"
        f"Patient History:\n{state.get('patient_history', '')}\n\n"
        f"Chart Notes:\n{state.get('chart_notes', '')}\n\n"
        f"Differential Diagnosis:\n{json.dumps(state.get('differential_diagnosis', []), indent=2)}\n\n"
        f"Normalized Medications:\n{json.dumps(state.get('normalized_medications', []), indent=2)}\n\n"
        f"Drug Interactions:\n{json.dumps(state.get('drug_interactions', []), indent=2)}"
    )
    result = structured_llm.invoke([
        {"role": "system", "content": (
            "You are a physician writing a SOAP note. Using all the clinical "
            "data provided, generate a comprehensive SOAP note with four "
            "sections: Subjective, Objective, Assessment, and Plan. Each "
            "section should be a detailed paragraph."
        )},
        {"role": "user", "content": context},
    ])
    return {"final_report": result.model_dump()}


# ---------------------------------------------------------------------------
# Runner entrypoint
# ---------------------------------------------------------------------------

STAGE_NODES = [
    node_transcription_cleanup,
    node_clinical_summarization,
    node_differential_diagnosis,
    node_medication_normalization,
    node_drug_interaction_check,
    node_final_report_generation,
]


def run_case(case_dir: str | Path) -> dict:
    """Run the full pipeline for a single benchmark case loaded from disk."""
    input_data = load_case(case_dir)
    state: dict = {**input_data}
    for node_fn in STAGE_NODES:
        state.update(node_fn(state))
    print(f"\n[Runner] Pipeline complete for case: {input_data['case_id']}", file=sys.stderr)
    return state


def run_pipeline(input_data: dict) -> dict:
    """Run the full pipeline given a pre-loaded input dict (container entrypoint)."""
    schema = load_schema(INPUT_SCHEMA_PATH)
    validate_input(input_data, schema)
    state: dict = {**input_data}
    for node_fn in STAGE_NODES:
        state.update(node_fn(state))
    print(f"[Runner] Pipeline complete for case: {input_data['case_id']}", file=sys.stderr)
    return state


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python langgraph_runner.py <path/to/case_folder>")
        sys.exit(1)
    result = run_case(sys.argv[1])
    print("\n[Runner] Final state keys:", list(result.keys()))
