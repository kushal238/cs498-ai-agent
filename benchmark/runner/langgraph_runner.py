"""
LangGraph runner stub for the clinical workflow benchmark.

Loads a case folder, validates input.json against its schema, then
iterates through the six workflow stages, printing placeholder messages.

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
import sys
from pathlib import Path

import jsonschema

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
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

    print(f"[Runner] Loaded and validated case: {input_data['case_id']}")
    return input_data


# ---------------------------------------------------------------------------
# Stub workflow nodes
# ---------------------------------------------------------------------------

def node_transcription_cleanup(state: dict) -> dict:
    """Stage 1: Clean raw patient transcript.

    TODO: Replace with a real LangGraph node that:
        - Calls an LLM to fix transcription errors, remove filler words,
          add punctuation, and label speaker turns.
        - Returns {"transcription_cleaned": <cleaned_string>}.
    """
    print("[Stage 1] transcription_cleanup — NOT IMPLEMENTED")
    return {"transcription_cleaned": None}


def node_clinical_summarization(state: dict) -> dict:
    """Stage 2: Summarize the cleaned transcript + chart notes.

    TODO: Replace with a real LangGraph node that:
        - Combines transcription_cleaned and chart_notes.
        - Calls an LLM to produce a concise clinical summary.
        - Returns {"clinical_summary": <summary_string>}.
    """
    print("[Stage 2] clinical_summarization — NOT IMPLEMENTED")
    return {"clinical_summary": None}


def node_differential_diagnosis(state: dict) -> dict:
    """Stage 3: Generate a PubMed-backed differential diagnosis list.

    TODO: Replace with a real LangGraph node that:
        - Uses the clinical_summary to query PubMed via shared/tools/pubmed.py.
        - Calls an LLM to rank candidate conditions and assign PMIDs.
        - Returns {"differential_diagnosis": [...]}.
    """
    print("[Stage 3] differential_diagnosis — NOT IMPLEMENTED")
    return {"differential_diagnosis": []}


def node_medication_normalization(state: dict) -> dict:
    """Stage 4: Normalize medication list via RxNorm.

    TODO: Replace with a real LangGraph node that:
        - Calls normalize_medication_list() from shared/tools/rxnorm.py.
        - Returns {"normalized_medications": [...]}.
    """
    print("[Stage 4] medication_normalization — NOT IMPLEMENTED")
    return {"normalized_medications": []}


def node_drug_interaction_check(state: dict) -> dict:
    """Stage 5: Check for drug-drug interactions via NIH RxNav.

    TODO: Replace with a real LangGraph node that:
        - Extracts RxCUIs from normalized_medications.
        - Calls check_interactions() from shared/tools/rxnorm.py.
        - Returns {"drug_interactions": [...]}.
    """
    print("[Stage 5] drug_interaction_check — NOT IMPLEMENTED")
    return {"drug_interactions": []}


def node_final_report_generation(state: dict) -> dict:
    """Stage 6: Generate a SOAP-format clinical report.

    TODO: Replace with a real LangGraph node that:
        - Aggregates all prior stage outputs.
        - Calls an LLM with a structured prompt to produce the SOAP report.
        - Returns {"final_report": {subjective, objective, assessment, plan}}.
    """
    print("[Stage 6] final_report_generation — NOT IMPLEMENTED")
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
    """Run the full workflow pipeline for a single benchmark case.

    Args:
        case_dir: Path to a case folder (must contain input.json).

    Returns:
        Final state dict with all stage outputs populated (or None for stubs).

    TODO: Replace the sequential node loop with a compiled LangGraph app:
        app = build_graph()
        final_state = app.invoke(initial_state)
    """
    input_data = load_case(case_dir)

    # Initial state carries all input fields; stage outputs are added incrementally.
    state: dict = {**input_data}

    for node_fn in STAGE_NODES:
        partial = node_fn(state)
        state.update(partial)

    print(f"\n[Runner] Pipeline complete for case: {input_data['case_id']}")
    return state


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python langgraph_runner.py <path/to/case_folder>")
        sys.exit(1)

    result = run_case(sys.argv[1])
    print("\n[Runner] Final state keys:", list(result.keys()))
