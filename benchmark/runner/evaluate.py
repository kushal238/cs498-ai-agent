"""
Batch evaluator for the clinical workflow benchmark.

Discovers all case folders under benchmark/cases/, runs each case through
the pipeline (langgraph_runner), scores each stage output against its
ground truth, and prints a summary results table.

Usage:
    python evaluate.py [--cases-dir PATH]

TODO: Once the workflow nodes and scoring functions are implemented,
replace the stub score calls below with real metric computations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add the benchmark root to sys.path so shared/ and runner/ are importable.
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCHMARK_ROOT))

from runner.langgraph_runner import run_case  # noqa: E402


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

def discover_cases(cases_dir: Path) -> list[Path]:
    """Return a sorted list of case folder paths under cases_dir.

    A valid case folder must contain input.json.
    """
    return sorted(
        p for p in cases_dir.iterdir()
        if p.is_dir() and (p / "input.json").exists()
    )


def load_ground_truth(case_dir: Path) -> dict | None:
    """Load ground_truth.json for a case, or return None if missing."""
    gt_path = case_dir / "ground_truth.json"
    if not gt_path.exists():
        return None
    with gt_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stub scoring
# ---------------------------------------------------------------------------

def score_case(predicted: dict, ground_truth: dict) -> dict[str, dict]:
    """Compute scores for all stages of a single case.

    Args:
        predicted:    Final state dict returned by run_case().
        ground_truth: Loaded ground_truth.json dict.

    Returns:
        Dict mapping stage name to a score dict, e.g.:
        {
            "transcription_cleanup": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            "clinical_summarization": {...},
            ...
        }

    TODO: Replace each stub call below with the real scoring functions from
          shared/scoring/ once those are implemented.
    """
    scores: dict[str, dict] = {}

    # Stage 1 — ROUGE on cleaned transcript
    # TODO: from shared.scoring.rouge_score import score_stage_text
    #       scores["transcription_cleanup"] = score_stage_text(
    #           "transcription_cleaned",
    #           predicted.get("transcription_cleaned", ""),
    #           ground_truth.get("transcription_cleaned", ""),
    #       )
    scores["transcription_cleanup"] = {"rouge1": None, "rouge2": None, "rougeL": None}

    # Stage 2 — ROUGE on clinical summary
    # TODO: same as above for "clinical_summarization" / "clinical_summary"
    scores["clinical_summarization"] = {"rouge1": None, "rouge2": None, "rougeL": None}

    # Stage 3 — Concept F1 + nDCG on differential diagnosis
    # TODO: from shared.scoring.concept_f1 import score_differential_diagnosis
    #       from shared.scoring.ndcg import score_differential_ndcg
    scores["differential_diagnosis"] = {"precision": None, "recall": None, "f1": None, "ndcg": None}

    # Stage 4 — Concept F1 on normalized medications
    # TODO: from shared.scoring.concept_f1 import score_normalized_medications
    scores["medication_normalization"] = {"precision": None, "recall": None, "f1": None}

    # Stage 5 — Concept F1 on drug interactions
    # TODO: from shared.scoring.concept_f1 import score_drug_interactions
    scores["drug_interaction_check"] = {"precision": None, "recall": None, "f1": None}

    # Stage 6 — ROUGE on each SOAP section
    # TODO: score each of subjective/objective/assessment/plan separately
    scores["final_report_generation"] = {
        "subjective_rougeL": None,
        "objective_rougeL": None,
        "assessment_rougeL": None,
        "plan_rougeL": None,
    }

    return scores


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(all_scores: dict[str, dict[str, dict]]) -> None:
    """Print a simple ASCII results table to stdout.

    Args:
        all_scores: Dict mapping case_id -> {stage_name -> score_dict}.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    for case_id, stage_scores in all_scores.items():
        print(f"\nCase: {case_id}")
        print("-" * 50)
        for stage, metrics in stage_scores.items():
            metric_str = ", ".join(
                f"{k}={v if v is not None else 'N/A'}"
                for k, v in metrics.items()
            )
            print(f"  {stage:<32} {metric_str}")

    print("\n" + "=" * 70)
    print("NOTE: All scores are None — workflow nodes not yet implemented.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cases_dir: Path) -> None:
    cases = discover_cases(cases_dir)
    if not cases:
        print(f"No cases found in {cases_dir}")
        sys.exit(0)

    print(f"[Evaluator] Found {len(cases)} case(s): {[c.name for c in cases]}\n")

    all_scores: dict[str, dict] = {}

    for case_path in cases:
        print(f"[Evaluator] Running case: {case_path.name}")
        try:
            predicted = run_case(case_path)
            ground_truth = load_ground_truth(case_path)

            if ground_truth is None:
                print(f"  WARNING: no ground_truth.json found for {case_path.name}, skipping scoring.")
                all_scores[case_path.name] = {}
                continue

            scores = score_case(predicted, ground_truth)
            all_scores[case_path.name] = scores

        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR running {case_path.name}: {exc}")
            all_scores[case_path.name] = {"error": str(exc)}

    print_results_table(all_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all benchmark cases and print a results table.")
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=BENCHMARK_ROOT / "cases",
        help="Path to the cases/ directory (default: benchmark/cases/)",
    )
    args = parser.parse_args()
    main(args.cases_dir)
