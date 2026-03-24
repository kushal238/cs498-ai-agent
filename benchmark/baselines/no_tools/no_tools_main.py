"""
no_tools_main.py — Container entrypoint for the no-tools pipeline baseline.

Reads input JSON from stdin, runs through the six-stage no-tools pipeline,
and prints the result as a single JSON line to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

from no_tools_runner import run_pipeline  # noqa: E402


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

    result = run_pipeline(input_data)
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
