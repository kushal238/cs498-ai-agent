"""
RxNorm / NIH RxNav API wrapper.

Drug normalization:  https://rxnav.nlm.nih.gov/REST/
    GET /rxcui.json?name={drug_name}
        Finds the RxNorm Concept Unique Identifier (RxCUI) for a drug name.
    GET /rxcui/{rxcui}/properties.json
        Returns the canonical ingredient name for a given RxCUI.

Drug interaction checking:  https://api.fda.gov/drug/label.json
    The RxNav /interaction/ endpoint was discontinued by NLM. Interactions
    are now sourced from FDA drug labels via the OpenFDA API. For each drug,
    the FDA label's drug_interactions field is searched for mentions of the
    other drugs in the patient's medication list.

No API key required for either service.
"""

from __future__ import annotations

import itertools
import sys

import requests

RXNAV_BASE  = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_URL = "https://api.fda.gov/drug/label.json"


def get_rxcui(drug_name: str) -> dict:
    """Look up the RxNorm CUI for a human-entered drug name string.

    Args:
        drug_name: Free-text medication name as entered by a clinician,
                   e.g. "warfarin 5mg daily" or "Coumadin".

    Returns:
        A dict with keys:
            - "rxnorm_id"   (str | None): The primary RxCUI, or None if not found.
            - "ingredient"  (str | None): The normalized generic ingredient name.
            - "raw_response" (dict):      The full JSON response from RxNav.
    """
    result = {"rxnorm_id": None, "ingredient": None, "raw_response": {}}
    try:
        # Try exact name first; if empty, fall back to just the first token
        # (strips dosage info like "5mg daily" that the API doesn't understand)
        for query in [drug_name, drug_name.split()[0]]:
            resp = requests.get(
                f"{RXNAV_BASE}/rxcui.json",
                params={"name": query},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            ids = data.get("idGroup", {}).get("rxnormId") or []
            if ids:
                result["raw_response"] = data
                break
        if not ids:
            return result

        rxcui = ids[0]
        result["rxnorm_id"] = rxcui

        # Fetch the canonical ingredient name
        prop_resp = requests.get(
            f"{RXNAV_BASE}/rxcui/{rxcui}/properties.json",
            timeout=10,
        )
        prop_resp.raise_for_status()
        prop_data = prop_resp.json()
        name = prop_data.get("properties", {}).get("name")
        result["ingredient"] = name.lower() if name else None

    except Exception as exc:
        print(f"[RxNorm] get_rxcui({drug_name!r}) failed: {exc}", file=sys.stderr)

    return result


def _get_ingredient_name(rxcui: str) -> str | None:
    """Return the canonical ingredient name for an RxCUI, or None on failure."""
    try:
        resp = requests.get(f"{RXNAV_BASE}/rxcui/{rxcui}/properties.json", timeout=10)
        resp.raise_for_status()
        name = resp.json().get("properties", {}).get("name")
        return name.lower() if name else None
    except Exception:
        return None


def check_interactions(rxcui_list: list[str]) -> list[dict]:
    """Check for drug-drug interactions among a set of RxCUIs.

    Uses OpenFDA drug label API (the RxNav /interaction/ endpoint was
    discontinued by NLM). For each drug pair, fetches the first drug's
    FDA label and checks whether the second drug is mentioned in its
    drug_interactions section.

    Args:
        rxcui_list: List of RxNorm CUI strings for the patient's medications.

    Returns:
        A list of interaction dicts, each with keys:
            - "drug_a"         (str): Ingredient name of the first drug.
            - "drug_b"         (str): Ingredient name of the second drug.
            - "severity"       (str): "unknown" (FDA labels don't provide severity grades).
            - "recommendation" (str): Relevant excerpt from the FDA label.
            - "description"    (str): Same as recommendation.
    """
    if len(rxcui_list) < 2:
        return []

    # Resolve RxCUIs to ingredient names first
    ingredient_map: dict[str, str] = {}
    for rxcui in rxcui_list:
        name = _get_ingredient_name(rxcui)
        if name:
            ingredient_map[rxcui] = name

    interactions = []
    for rxcui_a, rxcui_b in itertools.combinations(rxcui_list, 2):
        drug_a = ingredient_map.get(rxcui_a)
        drug_b = ingredient_map.get(rxcui_b)
        if not drug_a or not drug_b:
            continue
        try:
            resp = requests.get(
                OPENFDA_URL,
                params={"search": f"openfda.generic_name:{drug_a}", "limit": 1},
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                continue

            interactions_text = " ".join(results[0].get("drug_interactions", []))
            if drug_b.lower() not in interactions_text.lower():
                continue

            # Extract the sentence mentioning drug_b
            sentences = interactions_text.split(".")
            excerpt = next(
                (s.strip() for s in sentences if drug_b.lower() in s.lower()),
                interactions_text[:300],
            )

            interactions.append({
                "drug_a": drug_a,
                "drug_b": drug_b,
                "severity": "unknown",
                "recommendation": excerpt,
                "description": excerpt,
            })

        except Exception as exc:
            print(f"[RxNorm] check_interactions({drug_a}, {drug_b}) failed: {exc}", file=sys.stderr)

    return interactions


def normalize_medication_list(medication_list: list[str]) -> list[dict]:
    """Normalize an entire medication list by calling get_rxcui() for each entry.

    Args:
        medication_list: List of human-entered medication strings from input.json.

    Returns:
        List of dicts matching the normalized_medications schema:
            [{"original": str, "rxnorm_id": str | None, "ingredient": str | None}, ...]
    """
    results = []
    for med in medication_list:
        try:
            lookup = get_rxcui(med)
            results.append({
                "original": med,
                "rxnorm_id": lookup["rxnorm_id"],
                "ingredient": lookup["ingredient"],
            })
        except Exception as exc:
            print(f"[RxNorm] normalize_medication_list: failed for {med!r}: {exc}", file=sys.stderr)
            results.append({"original": med, "rxnorm_id": None, "ingredient": None})
    return results
