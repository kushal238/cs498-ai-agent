"""
Sanity check for the BERTScore module before running the full benchmark.

Run from repo root:
    source .venv/bin/activate
    python scripts/sanity_bertscore.py

Prints scores for one paraphrase pair and one unrelated pair so we can
confirm that:
  - the Bio_ClinicalBERT encoder downloads and loads
  - clinical paraphrases get a high F1
  - unrelated text gets a noticeably lower F1
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
sys.path.insert(0, str(BENCHMARK_ROOT))

from shared.scoring.bertscore import score_bertscore  # noqa: E402
from shared.scoring.rouge_score import score_rouge    # noqa: E402


PAIRS = [
    (
        "paraphrase",
        "Patient has type 2 diabetes mellitus poorly controlled with metformin.",
        "T2DM patient is poorly managed on metformin therapy.",
    ),
    (
        "near-identical",
        "The patient was prescribed lisinopril 10mg daily for hypertension.",
        "The patient was prescribed lisinopril 10 mg once daily for hypertension.",
    ),
    (
        "unrelated",
        "The patient was prescribed lisinopril 10mg daily for hypertension.",
        "Patient reports a productive cough and fever for the past three days.",
    ),
]


def main() -> None:
    print("[sanity] Loading BERTScore (first call downloads Bio_ClinicalBERT ~440MB)...")
    t0 = time.time()
    # Warm-up call so the model load cost is not attributed to pair 1
    score_bertscore("warm up", "warm up the encoder")
    print(f"[sanity] Loaded encoder in {time.time() - t0:.1f}s\n")

    for label, hyp, ref in PAIRS:
        rouge = score_rouge(hyp, ref)
        bert = score_bertscore(hyp, ref)
        print(f"[{label}]")
        print(f"  hyp: {hyp}")
        print(f"  ref: {ref}")
        print(f"  ROUGE-L:      {rouge['rougeL']:.3f}")
        print(f"  BERTScore F1: {bert['bertscore_f1']:.3f}  "
              f"(P={bert['bertscore_p']:.3f}, R={bert['bertscore_r']:.3f})")
        print()


if __name__ == "__main__":
    main()
