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
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
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
# Multi-trial helpers
# ---------------------------------------------------------------------------

def flatten_scores(stage_scores: dict) -> list[tuple[str, str, float]]:
    """Convert {stage: {metric: value}} to [(stage, metric, value), ...].

    Skips stages with an 'error' key and skips the 'stage' label key that
    score_stage_text() adds to its return dict.
    """
    rows = []
    for stage, metrics in stage_scores.items():
        if isinstance(metrics, dict) and "error" not in metrics:
            for metric, value in metrics.items():
                if metric != "stage" and isinstance(value, (int, float)):
                    rows.append((stage, metric, float(value)))
    return rows


def average_trial_scores(trial_scores: list[dict]) -> dict:
    """Average a list of per-trial score dicts across trials.

    Args:
        trial_scores: List of {stage: {metric: value}} dicts, one per trial.
                      Dicts containing an 'error' key are skipped.

    Returns:
        {stage: {metric: {"mean": float, "stddev": float}}}
    """
    combined: dict[str, dict[str, list[float]]] = {}
    for scores in trial_scores:
        if "error" in scores:
            continue
        for stage, metrics in scores.items():
            if not isinstance(metrics, dict) or "error" in metrics:
                continue
            combined.setdefault(stage, {})
            for metric, value in metrics.items():
                if metric != "stage" and isinstance(value, (int, float)):
                    combined[stage].setdefault(metric, []).append(float(value))
    return {
        stage: {
            metric: {
                "mean": statistics.mean(vals),
                "stddev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
            for metric, vals in metrics.items()
        }
        for stage, metrics in combined.items()
    }


def write_results_csv(
    all_trial_scores: dict[str, list[dict]],
    output_dir: Path,
    run_id: str,
) -> tuple[Path, Path]:
    """Write raw and summary CSVs to output_dir.

    Args:
        all_trial_scores: {case_id: [trial_score_dict, ...]}
        output_dir:       Directory to write CSV files into (created if needed).
        run_id:           Identifier string for this run, used in filenames.

    Returns:
        (raw_path, summary_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path     = output_dir / f"{run_id}_raw.csv"
    summary_path = output_dir / f"{run_id}_summary.csv"

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "case_id", "trial", "stage", "metric", "value"])
        for case_id, trial_scores in all_trial_scores.items():
            for trial_num, scores in enumerate(trial_scores, start=1):
                if "error" in scores:
                    continue
                for stage, metric, value in flatten_scores(scores):
                    writer.writerow([run_id, case_id, trial_num, stage, metric, value])

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "case_id", "stage", "metric", "mean", "stddev"])
        for case_id, trial_scores in all_trial_scores.items():
            summary = average_trial_scores(trial_scores)
            for stage, metrics in summary.items():
                for metric, stats in metrics.items():
                    writer.writerow([
                        run_id, case_id, stage, metric,
                        round(stats["mean"], 4), round(stats["stddev"], 4),
                    ])

    return raw_path, summary_path


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
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of trials per case (scores are averaged across trials)")
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_ROOT / "results",
                        help="Directory to write CSV results (default: benchmark/results/)")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save each agent's raw JSON output to output-dir/predictions/")
    args = parser.parse_args()

    if args.build:
        build_image(args.image)

    cases = discover_cases(args.cases_dir)
    if not cases:
        print(f"No cases found in {args.cases_dir}")
        sys.exit(0)

    print(f"[Harness] Found {len(cases)} case(s): {[c.name for c in cases]}")
    print(f"[Harness] Trials per case: {args.trials}\n")

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    all_trial_scores: dict[str, list[dict]] = {}

    for case_dir in cases:
        case_id = case_dir.name
        print(f"[Harness] Running case: {case_id}")

        input_data   = load_input(case_dir)
        ground_truth = load_ground_truth(case_id)

        if ground_truth is None:
            all_trial_scores[case_id] = [{"error": "missing ground truth"}]
            continue

        trial_scores = []
        for trial in range(1, args.trials + 1):
            if args.trials > 1:
                print(f"  Trial {trial}/{args.trials}")

            predicted = run_agent(input_data, args.image, args.timeout, args.network)
            if predicted is None:
                trial_scores.append({"error": "agent run failed"})
                continue

            if args.save_predictions:
                pred_dir = args.output_dir / "predictions"
                pred_dir.mkdir(parents=True, exist_ok=True)
                pred_path = pred_dir / f"{run_id}_{case_id}_trial{trial}.json"
                with pred_path.open("w", encoding="utf-8") as f:
                    json.dump(predicted, f, indent=2)

            try:
                scores = score_case(predicted, ground_truth)
                trial_scores.append(scores)
            except Exception as exc:
                print(f"  ERROR scoring {case_id} trial {trial}: {exc}", file=sys.stderr)
                trial_scores.append({"error": str(exc)})

        all_trial_scores[case_id] = trial_scores

    # Print results table using first trial scores for display
    display_scores = {
        cid: ts[0] if ts else {"error": "no trials completed"}
        for cid, ts in all_trial_scores.items()
    }
    print_results_table(display_scores)

    # Write CSVs
    raw_path, summary_path = write_results_csv(all_trial_scores, args.output_dir, run_id)
    print(f"[Harness] Results written to:\n  {raw_path}\n  {summary_path}")


if __name__ == "__main__":
    main()
