"""
harness.py — Host-side benchmark orchestrator (SWE-bench style).

Runs on the HOST, never inside a container. For each case:
  1. Loads input.json from benchmark/cases/{case_id}/
  2. Runs: docker run --rm --network=none -i <image>  (input piped via stdin)
  3. Captures stdout, parses as JSON (the agent's prediction)
  4. Loads ground_truth from benchmark/ground_truths/{case_id}.json
  5. Scores prediction vs ground truth using shared/scoring/
  6. Prints a results table

Ground truth never enters the container — isolation is enforced at the
filesystem level (agent/Dockerfile does not COPY ground_truths/) and at the
network level (--network=none prevents outbound calls).

Usage:
    python benchmark/harness/harness.py [--cases-dir PATH] [--build] [--timeout N] [--image NAME]

Options:
    --cases-dir PATH   Path to cases/ directory (default: benchmark/cases/)
    --build            Rebuild the agent Docker image before running
    --timeout N        Seconds to wait per case container (default: 120)
    --image NAME       Docker image name/tag (default: clinical-agent:latest)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR      = BENCHMARK_ROOT / "cases"
GT_DIR         = BENCHMARK_ROOT / "ground_truths"
AGENT_DIR      = BENCHMARK_ROOT / "agent"

DEFAULT_IMAGE   = "clinical-agent:latest"
DEFAULT_TIMEOUT = 120

# Add benchmark root to path so shared/ is importable
sys.path.insert(0, str(BENCHMARK_ROOT))

from shared.scoring.rouge_score import score_stage_text          # noqa: E402
from shared.scoring.concept_f1 import (                          # noqa: E402
    score_differential_diagnosis,
    score_normalized_medications,
    score_drug_interactions,
)
from shared.scoring.ndcg import score_differential_ndcg          # noqa: E402


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

def discover_cases(cases_dir: Path) -> list[Path]:
    """Return sorted list of case dirs that contain input.json."""
    return sorted(
        p for p in cases_dir.iterdir()
        if p.is_dir() and (p / "input.json").exists()
    )


def load_input(case_dir: Path) -> dict:
    with (case_dir / "input.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth(case_id: str) -> dict | None:
    gt_path = GT_DIR / f"{case_id}.json"
    if not gt_path.exists():
        print(f"  WARNING: no ground truth at {gt_path}", file=sys.stderr)
        return None
    with gt_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Docker operations
# ---------------------------------------------------------------------------

def build_image(image_name: str) -> None:
    """Build the agent Docker image. Build context is benchmark/."""
    print(f"[Harness] Building Docker image: {image_name}")
    subprocess.run(
        [
            "docker", "build",
            "-t", image_name,
            "-f", str(AGENT_DIR / "Dockerfile"),
            str(BENCHMARK_ROOT),  # build context = benchmark/
        ],
        check=True,
    )


def run_agent(input_data: dict, image_name: str, timeout: int, network: str = "bridge") -> dict | None:
    """Run the agent container for one case and return its parsed JSON output.

    The container receives input as JSON on stdin and must print a single
    JSON object to stdout. All other output should go to stderr inside the container.

    Returns None on timeout, non-zero exit, or JSON parse failure.
    """
    input_json_str = json.dumps(input_data)
    cmd = [
        "docker", "run", "--rm",
        f"--network={network}",
        "-i",
    ]
    # Pass OPENAI_API_KEY into the container if set in host env
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        cmd.extend(["-e", f"OPENAI_API_KEY={api_key}"])
    cmd.append(image_name)
    try:
        result = subprocess.run(
            cmd,
            input=input_json_str,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  ERROR: container timed out after {timeout}s", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"  ERROR: container exited {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(f"  STDERR: {result.stderr[:500]}", file=sys.stderr)
        return None

    stdout = result.stdout.strip()
    if not stdout:
        print("  ERROR: container produced no stdout", file=sys.stderr)
        return None

    # Take only the last non-empty line — earlier lines may be debug leakage
    last_line = next(
        (line for line in reversed(stdout.splitlines()) if line.strip()),
        ""
    )
    try:
        return json.loads(last_line)
    except json.JSONDecodeError as exc:
        print(f"  ERROR: failed to parse agent output as JSON: {exc}", file=sys.stderr)
        print(f"  Raw stdout (last 300 chars): ...{stdout[-300:]}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Scoring (host-side only)
# ---------------------------------------------------------------------------

def score_case(predicted: dict, ground_truth: dict) -> dict[str, dict]:
    """Score all pipeline stages for one case against ground truth."""
    scores: dict[str, dict] = {}

    # Stage 1 — ROUGE on cleaned transcript
    scores["transcription_cleanup"] = score_stage_text(
        "transcription_cleaned",
        hypothesis=predicted.get("transcription_cleaned") or "",
        reference=ground_truth.get("transcription_cleaned") or "",
    )

    # Stage 2 — ROUGE on clinical summary
    scores["clinical_summarization"] = score_stage_text(
        "clinical_summary",
        hypothesis=predicted.get("clinical_summary") or "",
        reference=ground_truth.get("clinical_summary") or "",
    )

    # Stage 3 — Concept F1 + nDCG on differential diagnosis
    pred_dx = predicted.get("differential_diagnosis", [])
    gt_dx   = ground_truth.get("differential_diagnosis", [])
    scores["differential_diagnosis"] = {
        **score_differential_diagnosis(pred_dx, gt_dx),
        "ndcg": score_differential_ndcg(pred_dx, gt_dx),
    }

    # Stage 4 — Concept F1 on normalized medications
    scores["medication_normalization"] = score_normalized_medications(
        predicted.get("normalized_medications", []),
        ground_truth.get("normalized_medications", []),
    )

    # Stage 5 — Concept F1 on drug interactions
    scores["drug_interaction_check"] = score_drug_interactions(
        predicted.get("drug_interactions", []),
        ground_truth.get("drug_interactions", []),
    )

    # Stage 6 — ROUGE-L per SOAP section
    pred_report = predicted.get("final_report") or {}
    gt_report   = ground_truth.get("final_report") or {}
    soap_scores = {}
    for section in ("subjective", "objective", "assessment", "plan"):
        s = score_stage_text(
            section,
            hypothesis=pred_report.get(section) or "",
            reference=gt_report.get(section) or "",
        )
        soap_scores[f"{section}_rougeL"] = s.get("rougeL", 0.0)
    scores["final_report_generation"] = soap_scores

    return scores


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(all_scores: dict[str, dict]) -> None:
    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS")
    print("=" * 72)

    for case_id, stage_scores in all_scores.items():
        print(f"\nCase: {case_id}")
        print("-" * 52)
        if "error" in stage_scores:
            print(f"  ERROR: {stage_scores['error']}")
            continue
        for stage, metrics in stage_scores.items():
            if isinstance(metrics, dict):
                metric_str = "  ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in metrics.items()
                    if k != "stage"
                )
            else:
                metric_str = str(metrics)
            print(f"  {stage:<34} {metric_str}")

    print("\n" + "=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical AI benchmark harness.")
    parser.add_argument("--cases-dir", type=Path, default=CASES_DIR,
                        help="Path to cases/ directory")
    parser.add_argument("--build", action="store_true",
                        help="Rebuild the agent Docker image before running")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Seconds to wait per case container")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE,
                        help="Docker image name/tag")
    parser.add_argument("--network", type=str, default="bridge",
                        help="Docker network mode (default: bridge, use 'none' for locked-down eval)")
    args = parser.parse_args()

    if args.build:
        build_image(args.image)

    cases = discover_cases(args.cases_dir)
    if not cases:
        print(f"No cases found in {args.cases_dir}")
        sys.exit(0)

    print(f"[Harness] Found {len(cases)} case(s): {[c.name for c in cases]}\n")

    all_scores: dict[str, dict] = {}

    for case_dir in cases:
        case_id = case_dir.name
        print(f"[Harness] Running case: {case_id}")

        input_data   = load_input(case_dir)
        ground_truth = load_ground_truth(case_id)

        if ground_truth is None:
            all_scores[case_id] = {"error": "missing ground truth"}
            continue

        predicted = run_agent(input_data, args.image, args.timeout, args.network)
        if predicted is None:
            all_scores[case_id] = {"error": "agent run failed"}
            continue

        try:
            scores = score_case(predicted, ground_truth)
            all_scores[case_id] = scores
        except Exception as exc:
            print(f"  ERROR scoring {case_id}: {exc}", file=sys.stderr)
            all_scores[case_id] = {"error": str(exc)}

    print_results_table(all_scores)


if __name__ == "__main__":
    main()
