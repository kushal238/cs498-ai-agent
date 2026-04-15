"""
agent_main.py — Container entrypoint for the clinical AI benchmark agent.

Reads a JSON-serialized benchmark case input from stdin, runs it through
the 6-stage clinical workflow pipeline, and prints the result as a single
JSON line to stdout.

The harness pipes input via `docker run -i` and captures stdout to extract
the agent's prediction. ALL diagnostic/logging output goes to stderr to
avoid corrupting the stdout JSON stream.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_runner_dir = Path(__file__).resolve().parent.parent / "runner"
if _runner_dir.exists():
    sys.path.insert(0, str(_runner_dir))

from agent import ClinicalAgent  # noqa: E402


def main() -> None:
    raw = sys.stdin.read()
    if not raw:
        print(json.dumps({"error": "No input received on stdin"}), file=sys.stderr)
        sys.exit(1)

    try:
        input_data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Failed to parse stdin JSON: {exc}"}), file=sys.stderr)
        sys.exit(1)

    agent = ClinicalAgent()
    result = agent.run(input_data)

    # Include execution log for debugging — harness ignores unknown keys
    execution_log = [
        {"stage": e.stage, "event": e.event, "detail": e.detail}
        for e in agent._last_state.memory.execution_log
    ] if agent._last_state else []

    output = {
        "case_id": input_data.get("case_id"),
        "transcription_cleaned": result.get("transcription_cleaned"),
        "clinical_summary": result.get("clinical_summary"),
        "differential_diagnosis": result.get("differential_diagnosis", []),
        "normalized_medications": result.get("normalized_medications", []),
        "drug_interactions": result.get("drug_interactions", []),
        "final_report": result.get("final_report", {}),
        "_debug_execution_log": execution_log,
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
