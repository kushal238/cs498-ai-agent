"""
Aggregate per-system summary CSVs into the comparison table + bar chart used
in the paper.

Reads each system's `*_summary.csv` (case x stage x metric x mean x stddev)
under benchmark/results/final_<timestamp>/<system>/, then:

  1. Computes overall (across-cases) mean ± stddev per (stage, metric, system).
  2. Writes a wide CSV with one row per (stage, metric) and one set of mean/std
     columns per system: zs, nt, agent.
  3. Writes a markdown summary table (for quick review).
  4. Renders the stage-breakdown bar chart used in the appendix figure.

Usage:
    source .venv/bin/activate
    python scripts/aggregate_results.py benchmark/results/final_<timestamp>
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SYSTEMS = ["zs", "nt", "agent"]
SYSTEM_LABEL = {
    "zs": "Zero-Shot",
    "nt": "No-Tools",
    "agent": "Agent (Ours)",
}

# Stages and the headline metrics we care about for the comparison table.
HEADLINE_METRICS: list[tuple[str, str]] = [
    ("transcription_cleanup", "rougeL"),
    ("transcription_cleanup", "bertscore_f1"),
    ("clinical_summarization", "rougeL"),
    ("clinical_summarization", "bertscore_f1"),
    ("differential_diagnosis", "f1"),
    ("differential_diagnosis", "ndcg"),
    ("medication_normalization", "f1"),
    ("drug_interaction_check", "f1"),
    ("final_report_generation", "subjective_rougeL"),
    ("final_report_generation", "objective_rougeL"),
    ("final_report_generation", "assessment_rougeL"),
    ("final_report_generation", "plan_rougeL"),
    ("final_report_generation", "subjective_bertscore_f1"),
    ("final_report_generation", "objective_bertscore_f1"),
    ("final_report_generation", "assessment_bertscore_f1"),
    ("final_report_generation", "plan_bertscore_f1"),
]

# Stages and metrics for the bar chart (collapsed view).
BAR_STAGES = [
    ("Transcription\n(ROUGE-L)", "transcription_cleanup", "rougeL"),
    ("Transcription\n(BERTScore F1)", "transcription_cleanup", "bertscore_f1"),
    ("Summarization\n(ROUGE-L)", "clinical_summarization", "rougeL"),
    ("Summarization\n(BERTScore F1)", "clinical_summarization", "bertscore_f1"),
    ("Diagnosis\n(F1)", "differential_diagnosis", "f1"),
    ("Medication\n(F1)", "medication_normalization", "f1"),
    ("Interactions\n(F1)", "drug_interaction_check", "f1"),
    ("Report\n(ROUGE-L)", "final_report_generation", "_report_rougeL"),
    ("Report\n(BERTScore F1)", "final_report_generation", "_report_bertscore_f1"),
]


def load_summary(path: Path) -> list[dict]:
    """Load a *_summary.csv file written by harness.write_results_csv."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "case_id": row["case_id"],
                "stage": row["stage"],
                "metric": row["metric"],
                "mean": float(row["mean"]),
                "stddev": float(row["stddev"]),
            })
    return rows


def find_summary_csv(system_dir: Path) -> Path:
    # Prefer rescored summaries (post-fix) over the original in-process ones,
    # which may be partial if scoring crashed mid-run.
    rescored = sorted(system_dir.glob("*_rescored_summary.csv"))
    if rescored:
        return rescored[-1]
    candidates = sorted(system_dir.glob("*_summary.csv"))
    if not candidates:
        raise FileNotFoundError(f"No *_summary.csv in {system_dir}")
    return candidates[-1]


def aggregate_across_cases(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Per (stage, metric): mean across cases of the case-level means.

    Each case already had its own (mean, stddev) across trials. To get an
    across-cases summary we average the per-case means and report the
    pooled stddev across cases.
    """
    grouped: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        key = (r["stage"], r["metric"])
        grouped.setdefault(key, []).append(r["mean"])

    out: dict[tuple[str, str], dict] = {}
    for key, vals in grouped.items():
        out[key] = {
            "mean": statistics.mean(vals),
            "stddev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out


def synth_report_avg(agg: dict[tuple[str, str], dict], suffix: str) -> dict | None:
    """Average the four SOAP section metrics ending in `suffix` (e.g. _rougeL)."""
    sections = ["subjective", "objective", "assessment", "plan"]
    means = []
    for sec in sections:
        m = agg.get(("final_report_generation", f"{sec}{suffix}"))
        if m is not None:
            means.append(m["mean"])
    if not means:
        return None
    return {"mean": statistics.mean(means), "stddev": 0.0, "n": len(means)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=Path,
                        help="Path to benchmark/results/final_<ts>/")
    args = parser.parse_args()

    base = args.base_dir
    if not base.exists():
        print(f"ERROR: {base} does not exist", file=sys.stderr)
        sys.exit(2)

    per_system: dict[str, dict[tuple[str, str], dict]] = {}
    for sys_tag in SYSTEMS:
        sys_dir = base / sys_tag
        if not sys_dir.exists():
            print(f"WARN: missing system dir {sys_dir}", file=sys.stderr)
            continue
        summary_csv = find_summary_csv(sys_dir)
        rows = load_summary(summary_csv)
        per_system[sys_tag] = aggregate_across_cases(rows)
        print(f"[aggregate] Loaded {sys_tag}: {summary_csv.name} ({len(rows)} rows)")

    # ---- write wide comparison CSV ----
    out_csv = base / "comparison_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["stage", "metric"]
        for s in SYSTEMS:
            header.extend([f"{s}_mean", f"{s}_std", f"{s}_n"])
        w.writerow(header)
        for stage, metric in HEADLINE_METRICS:
            row = [stage, metric]
            for s in SYSTEMS:
                cell = per_system.get(s, {}).get((stage, metric))
                if cell is None:
                    row.extend(["", "", ""])
                else:
                    row.extend([f"{cell['mean']:.4f}",
                                f"{cell['stddev']:.4f}",
                                cell["n"]])
            w.writerow(row)
    print(f"[aggregate] Wrote {out_csv}")

    # ---- markdown summary ----
    md_path = base / "comparison_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Final benchmark comparison\n\n")
        f.write("Aggregated across all cases (mean of per-case means).\n\n")
        f.write("| Stage | Metric | "
                + " | ".join(SYSTEM_LABEL[s] for s in SYSTEMS) + " |\n")
        f.write("|---|---|" + "|".join(["---"] * len(SYSTEMS)) + "|\n")
        for stage, metric in HEADLINE_METRICS:
            cells = []
            for s in SYSTEMS:
                cell = per_system.get(s, {}).get((stage, metric))
                if cell is None:
                    cells.append("—")
                else:
                    cells.append(f"{cell['mean']:.3f} ± {cell['stddev']:.3f}")
            f.write(f"| {stage} | {metric} | " + " | ".join(cells) + " |\n")
    print(f"[aggregate] Wrote {md_path}")

    # ---- bar chart ----
    # Synthesize report-average rows on the fly.
    for s in per_system:
        for suffix, key in (("_rougeL", "_report_rougeL"),
                             ("_bertscore_f1", "_report_bertscore_f1")):
            avg = synth_report_avg(per_system[s], suffix)
            if avg is not None:
                per_system[s][("final_report_generation", key)] = avg

    fig, ax = plt.subplots(figsize=(11, 4.2))
    n_groups = len(BAR_STAGES)
    n_systems = len(SYSTEMS)
    width = 0.26
    x_centers = list(range(n_groups))

    colors = {"zs": "#4C78A8", "nt": "#F58518", "agent": "#54A24B"}
    for i, s in enumerate(SYSTEMS):
        means = []
        for _, stage, metric in BAR_STAGES:
            cell = per_system.get(s, {}).get((stage, metric))
            means.append(cell["mean"] if cell is not None else 0.0)
        offsets = [c + (i - 1) * width for c in x_centers]
        ax.bar(offsets, means, width=width, label=SYSTEM_LABEL[s], color=colors[s])

    ax.set_xticks(x_centers)
    ax.set_xticklabels([label for label, _, _ in BAR_STAGES], fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Performance by Stage Across Systems (with semantic-similarity supplement)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    chart_path = base / "stage_breakdown.png"
    fig.savefig(chart_path, dpi=200)
    print(f"[aggregate] Wrote {chart_path}")


if __name__ == "__main__":
    main()
