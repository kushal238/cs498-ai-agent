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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# In the agent container, BENCHMARK_ROOT is set to /app via ENV.
# On the host, it defaults to two levels up from this file (benchmark/).
BENCHMARK_ROOT = Path(os.environ.get("BENCHMARK_ROOT", Path(__file__).resolve().parent.parent))
SCHEMAS_DIR = BENCHMARK_ROOT / "shared" / "schemas"
INPUT_SCHEMA_PATH = SCHEMAS_DIR / "input_schema.json"

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

    Expected behaviour:
        - Fix transcription errors and remove filler words (um, uh, like).
        - Add punctuation and label speaker turns as 'Doctor:' / 'Patient:'.
    """
    print("[Stage 1] transcription_cleanup — NOT IMPLEMENTED", file=sys.stderr)
    return {"transcription_cleaned": None}


def node_clinical_summarization(state: dict) -> dict:
    """Stage 2: Summarize the cleaned transcript + chart notes.

    Input state keys:  transcription_cleaned, chart_notes, patient_history
    Output key:        clinical_summary (str)

    Expected behaviour:
        - Produce a concise 2-4 sentence clinician-facing summary covering
          chief complaint, relevant history, current medications, and key findings.
    """
    print("[Stage 2] clinical_summarization — NOT IMPLEMENTED", file=sys.stderr)
    return {"clinical_summary": None}


def node_differential_diagnosis(state: dict) -> dict:
    """Stage 3: Generate a PubMed-backed differential diagnosis list.

    Input state keys:  clinical_summary, patient_history
    Output key:        differential_diagnosis (list[dict])

    Each item: {"condition": str, "pmid": str | None, "rationale": str}

    Expected behaviour:
        - Rank the top 3 candidate conditions by likelihood.
        - Attach a real PubMed PMID to each via shared/tools/pubmed.py.
    """
    print("[Stage 3] differential_diagnosis — NOT IMPLEMENTED", file=sys.stderr)
    return {"differential_diagnosis": []}


def node_medication_normalization(state: dict) -> dict:
    """Stage 4: Normalize medication list via RxNorm.

    Input state keys:  medication_list
    Output key:        normalized_medications (list[dict])

    Each item: {"original": str, "rxnorm_id": str | None, "ingredient": str | None}

    Expected behaviour:
        - Call normalize_medication_list() from shared/tools/rxnorm.py.
    """
    print("[Stage 4] medication_normalization — NOT IMPLEMENTED", file=sys.stderr)
    return {"normalized_medications": []}


def node_drug_interaction_check(state: dict) -> dict:
    """Stage 5: Check for drug-drug interactions via NIH RxNav.

    Input state keys:  normalized_medications
    Output key:        drug_interactions (list[dict])

    Each item: {"drug_a": str, "drug_b": str, "severity": str, "recommendation": str}

    Expected behaviour:
        - Extract RxCUIs from normalized_medications.
        - Call check_interactions() from shared/tools/rxnorm.py.
    """
    print("[Stage 5] drug_interaction_check — NOT IMPLEMENTED", file=sys.stderr)
    return {"drug_interactions": []}


def node_final_report_generation(state: dict) -> dict:
    """Stage 6: Generate a SOAP-format clinical report.

    Input state keys:  all prior stage outputs + patient_history + chart_notes
    Output key:        final_report (dict)

    Keys: {"subjective": str, "objective": str, "assessment": str, "plan": str}

    Expected behaviour:
        - Aggregate all prior stage outputs into a physician-ready SOAP note.
    """
    print("[Stage 6] final_report_generation — NOT IMPLEMENTED", file=sys.stderr)
    return {"final_report": {"subjective": None, "objective": None, "assessment": None, "plan": None}}


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
