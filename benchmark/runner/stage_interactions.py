# benchmark/runner/stage_interactions.py
"""Stage 5: Check for drug-drug interactions via NIH RxNav / OpenFDA."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_benchmark_root = Path(os.environ.get("BENCHMARK_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(_benchmark_root))
from shared.tools.rxnorm import check_interactions  # noqa: E402


def run(context: dict) -> dict:
    """Check drug-drug interactions for normalized medications.

    Args:
        context: Must contain 'normalized_medications' (list of dicts with 'rxnorm_id').

    Returns:
        {"reasoning": str, "confidence": str,
         "output": {"drug_interactions": list[dict]}}
    """
    normalized: list[dict] = context.get("normalized_medications", [])
    rxcui_list = [m["rxnorm_id"] for m in normalized if m.get("rxnorm_id")]
    interactions = check_interactions(rxcui_list)
    for interaction in interactions:
        interaction.pop("description", None)

    n = len(interactions)
    return {
        "reasoning": (
            f"Checked {len(rxcui_list)} RxCUI(s); found {n} interaction(s)."
        ),
        "confidence": "high" if rxcui_list else "low",
        "output": {"drug_interactions": interactions},
    }
