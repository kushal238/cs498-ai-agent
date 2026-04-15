# benchmark/runner/stage_medications.py
"""Stage 4: Normalize medication list via RxNorm."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import llm_client

_benchmark_root = Path(os.environ.get("BENCHMARK_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(_benchmark_root))
from shared.tools.rxnorm import get_rxcui  # noqa: E402


def _extract_generic_names(medication_list: list[str]) -> dict[str, list[str]]:
    """Batch-extract generic drug name(s) for the entire medication list.

    Returns a dict mapping each original medication string to a list of one or
    more generic names. Combination products are split into separate entries.
    Falls back to {med: [med]} for the whole batch if the LLM call fails.
    """
    if not medication_list:
        return {}

    numbered = "\n".join(f"{i+1}. {med}" for i, med in enumerate(medication_list))
    messages = [
        {
            "role": "system",
            "content": (
                "You are a pharmacist. For each numbered medication below, extract "
                "the generic drug name(s). If a medication is a combination product "
                "(e.g. 'oxycodone/acetaminophen'), list each ingredient separately "
                "separated by commas. Strip all doses, routes, frequencies, "
                "weight-based rates, concentrations, and brand names. "
                "Respond with one line per input medication, keeping the same "
                "numbering. Format: '1. drug1, drug2' or '1. drug1'. Nothing else."
            ),
        },
        {"role": "user", "content": numbered},
    ]
    try:
        raw = llm_client.chat(messages)
        result: dict[str, list[str]] = {}
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            dot = line.find(".")
            if dot == -1:
                continue
            try:
                idx = int(line[:dot].strip()) - 1
            except ValueError:
                continue
            if idx < 0 or idx >= len(medication_list):
                continue
            names = [n.strip().lower() for n in line[dot + 1:].split(",") if n.strip()]
            if names:
                result[medication_list[idx]] = names
        for med in medication_list:
            if med not in result:
                result[med] = [med]
        return result
    except Exception:
        return {med: [med] for med in medication_list}


def run(context: dict) -> dict:
    """Normalize the medication list using an LLM + RxNorm API.

    Args:
        context: Must contain 'medication_list'.

    Returns:
        {"reasoning": str, "confidence": str,
         "output": {"normalized_medications": list[dict]}}
    """
    raw_meds: list[str] = context.get("medication_list", [])
    extracted = _extract_generic_names(raw_meds)
    normalized = []
    for med in raw_meds:
        for drug_name in extracted.get(med, [med]):
            lookup = get_rxcui(drug_name)
            normalized.append({
                "original": med,
                "rxnorm_id": lookup["rxnorm_id"],
                "ingredient": lookup["ingredient"],
            })

    n = len(normalized)
    found = sum(1 for m in normalized if m["rxnorm_id"] is not None)
    return {
        "reasoning": f"Extracted and normalized {n} medication entries; {found} resolved via RxNorm.",
        "confidence": "high" if found == n else "medium" if found > 0 else "low",
        "output": {"normalized_medications": normalized},
    }
