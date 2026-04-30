# benchmark/runner/stage_interactions.py
"""Stage 5: Check for drug-drug interactions via RxList."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_benchmark_root = Path(os.environ.get("BENCHMARK_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(_benchmark_root))
from shared.tools.rxlist import check_rxlist_interactions  # noqa: E402


def run(context: dict) -> dict:
    """Check drug-drug interactions for normalized medications.

    Args:
        context: Must contain 'normalized_medications' (list of dicts with 'ingredient').

    Returns:
        {"reasoning": str, "confidence": str,
         "output": {"drug_interactions": list[dict]}}
    """
    normalized: list[dict] = context.get("normalized_medications", [])
    interactions = check_rxlist_interactions(normalized)

    n = len(interactions)
    return {
        "reasoning": (
            f"Checked {len(normalized)} medication(s); found {n} interaction(s)."
        ),
        "confidence": "high" if normalized else "low",
        "output": {"drug_interactions": interactions},
    }
