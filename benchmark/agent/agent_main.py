"""
agent_main.py — Container entrypoint for the clinical AI benchmark agent.

Reads the INPUT_JSON environment variable (a JSON-serialized benchmark case
input), runs it through the 6-stage clinical workflow pipeline, and prints
the result as a single JSON line to stdout.

The harness captures stdout to extract the agent's prediction.
ALL diagnostic/logging output goes to stderr to avoid corrupting the stdout
JSON stream.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# In the container, langgraph_runner.py is copied to /app/ (same dir as this file).
# On the host, it lives at benchmark/runner/langgraph_runner.py.
_runner_dir = Path(__file__).resolve().parent.parent / "runner"
if _runner_dir.exists():
    sys.path.insert(0, str(_runner_dir))

from langgraph_runner import run_pipeline  # noqa: E402


def main() -> None:
    raw = os.environ.get("INPUT_JSON")
    if not raw:
        print(json.dumps({"error": "INPUT_JSON environment variable not set"}))
        sys.exit(1)

    try:
        input_data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Failed to parse INPUT_JSON: {exc}"}))
        sys.exit(1)

    result = run_pipeline(input_data)

    # Emit only the fields that match ground_truth_schema.json
    output = {
        "case_id": input_data.get("case_id"),
        "transcription_cleaned": result.get("transcription_cleaned"),
        "clinical_summary": result.get("clinical_summary"),
        "differential_diagnosis": result.get("differential_diagnosis", []),
        "normalized_medications": result.get("normalized_medications", []),
        "drug_interactions": result.get("drug_interactions", []),
        "final_report": result.get("final_report", {}),
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
