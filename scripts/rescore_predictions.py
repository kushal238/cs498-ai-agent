"""
Re-score saved predictions using the current scoring code.

Predictions land in benchmark/results/final_<ts>/<system>/predictions/ as
run_*_case_*_trial*.json (one file per case+trial). This script reads each,
matches it to ground_truths/<case_id>.json, and writes fresh
*_raw.csv / *_summary.csv that include BERTScore alongside ROUGE-L.

Usage:
    source .venv/bin/activate
    python scripts/rescore_predictions.py benchmark/results/final_<ts>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
GT_DIR = BENCHMARK_ROOT / "ground_truths"
sys.path.insert(0, str(BENCHMARK_ROOT))

from harness.harness import score_case, flatten_scores, average_trial_scores  # noqa: E402

SYSTEMS = ["agent", "zs", "nt"]

# Match files like:
#   run_agent_20260505_141201_case_03_trial2.json
#   run_zs_20260505_152340_case_01_template_trial1.json
PRED_RE = re.compile(
    r"^(?P<run_id>run_[a-zA-Z0-9_]+?_\d{8}_\d{6})_"
    r"(?P<case_id>case_\d+(?:_template)?)_"
    r"trial(?P<trial>\d+)\.json$"
)


def load_ground_truth(case_id: str) -> dict | None:
    gt_path = GT_DIR / f"{case_id}.json"
    if not gt_path.exists():
        print(f"  WARN: missing ground truth for {case_id}", file=sys.stderr)
        return None
    with gt_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rescore_system(system_dir: Path) -> None:
    pred_dir = system_dir / "predictions"
    if not pred_dir.exists():
        print(f"  SKIP: no predictions dir at {pred_dir}", file=sys.stderr)
        return

    files = sorted(pred_dir.glob("*.json"))
    print(f"[rescore] {system_dir.name}: {len(files)} prediction files")

    # Group by case, then trial
    grouped: dict[str, dict[int, dict]] = {}
    run_id_seen: str | None = None
    for f in files:
        m = PRED_RE.match(f.name)
        if not m:
            print(f"  WARN: unmatched filename {f.name}", file=sys.stderr)
            continue
        case_id = m.group("case_id")
        trial = int(m.group("trial"))
        run_id_seen = m.group("run_id")
        with f.open("r", encoding="utf-8") as fh:
            grouped.setdefault(case_id, {})[trial] = json.load(fh)

    if run_id_seen is None:
        print(f"  WARN: no usable predictions in {pred_dir}", file=sys.stderr)
        return

    rescored_run_id = f"{run_id_seen}_rescored"

    # Score every prediction
    all_trial_scores: dict[str, list[dict]] = {}
    for case_id in sorted(grouped):
        gt = load_ground_truth(case_id)
        if gt is None:
            all_trial_scores[case_id] = [{"error": "missing ground truth"}]
            continue
        trial_scores = []
        for trial in sorted(grouped[case_id]):
            pred = grouped[case_id][trial]
            try:
                trial_scores.append(score_case(pred, gt))
            except Exception as exc:
                print(f"  ERROR scoring {case_id} trial {trial}: {exc}", file=sys.stderr)
                trial_scores.append({"error": str(exc)})
        all_trial_scores[case_id] = trial_scores

    # Write CSVs (mirrors harness.write_results_csv)
    raw_path = system_dir / f"{rescored_run_id}_raw.csv"
    summary_path = system_dir / f"{rescored_run_id}_summary.csv"

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "case_id", "trial", "stage", "metric", "value"])
        for case_id, trials in all_trial_scores.items():
            for trial_num, scores in enumerate(trials, start=1):
                if "error" in scores:
                    continue
                for stage, metric, value in flatten_scores(scores):
                    w.writerow([rescored_run_id, case_id, trial_num,
                                stage, metric, value])

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "case_id", "stage", "metric", "mean", "stddev"])
        for case_id, trials in all_trial_scores.items():
            summary = average_trial_scores(trials)
            for stage, metrics in summary.items():
                for metric, stats in metrics.items():
                    w.writerow([
                        rescored_run_id, case_id, stage, metric,
                        round(stats["mean"], 4), round(stats["stddev"], 4),
                    ])

    print(f"[rescore] Wrote {summary_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=Path,
                        help="Path to benchmark/results/final_<ts>/")
    args = parser.parse_args()

    if not args.base_dir.exists():
        print(f"ERROR: {args.base_dir} does not exist", file=sys.stderr)
        sys.exit(2)

    for system in SYSTEMS:
        sys_dir = args.base_dir / system
        if not sys_dir.exists():
            print(f"[rescore] {system}: skip (dir missing)")
            continue
        rescore_system(sys_dir)


if __name__ == "__main__":
    main()
