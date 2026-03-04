"""
RxNorm / NIH RxNav API wrapper stubs.

Real API base URL: https://rxnav.nlm.nih.gov/REST/

Key endpoints used:
    GET /rxcui.json?name={drug_name}
        Finds the RxNorm Concept Unique Identifier (RxCUI) for a drug name.
        Returns: {"idGroup": {"rxnormId": ["1234567"]}}

    GET /interaction/list.json?rxcuis={rxcui1}+{rxcui2}
        Checks for drug-drug interactions between two or more RxCUIs.
        Returns a list of interaction pairs with severity and description.

No API key is required for RxNav. Rate limit: ~20 requests/second.
"""

from __future__ import annotations


def get_rxcui(drug_name: str) -> dict:
    """Look up the RxNorm CUI for a human-entered drug name string.

    Args:
        drug_name: Free-text medication name as entered by a clinician,
                   e.g. "warfarin 5mg daily" or "Coumadin".

    Returns:
        A dict with keys:
            - "rxnorm_id" (str | None): The primary RxCUI, or None if not found.
            - "ingredient"  (str | None): The normalized generic ingredient name.
            - "raw_response" (dict): The full JSON response from RxNav.

    Raises:
        NotImplementedError: Always — real API call not yet implemented.

    TODO: Implement using requests.get() to
          https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}
          then parse idGroup.rxnormId[0] for the CUI and call
          /REST/rxcui/{rxcui}/properties.json to get the ingredient name.
    """
    raise NotImplementedError("TODO: implement RxNorm CUI lookup via NIH RxNav REST API")


def check_interactions(rxcui_list: list[str]) -> list[dict]:
    """Check for drug-drug interactions among a set of RxCUIs.

    Args:
        rxcui_list: List of RxNorm CUI strings for the patient's medications,
                    e.g. ["11289", "1191"] for warfarin and aspirin.

    Returns:
        A list of interaction dicts, each with keys:
            - "drug_a"          (str): Ingredient name of the first drug.
            - "drug_b"          (str): Ingredient name of the second drug.
            - "severity"        (str): One of minor/moderate/major/contraindicated/unknown.
            - "recommendation"  (str): Clinical recommendation from NDF-RT / DrugBank.
            - "description"     (str): Full interaction description text.

    Raises:
        NotImplementedError: Always — real API call not yet implemented.

    TODO: Implement using requests.get() to
          https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={joined}
          where joined = "+".join(rxcui_list).
          Parse interactionTypeGroup[*].interactionType[*].interactionPair.
    """
    raise NotImplementedError("TODO: implement drug-interaction check via NIH RxNav REST API")


def normalize_medication_list(medication_list: list[str]) -> list[dict]:
    """Normalize an entire medication list by calling get_rxcui() for each entry.

    Args:
        medication_list: List of human-entered medication strings from input.json.

    Returns:
        List of dicts matching the normalized_medications schema:
            [{"original": str, "rxnorm_id": str | None, "ingredient": str | None}, ...]

    Raises:
        NotImplementedError: Always — real API call not yet implemented.

    TODO: Call get_rxcui() for each medication, collect results, handle failures
          gracefully (log warning, set rxnorm_id/ingredient to None).
    """
    raise NotImplementedError("TODO: implement batch medication normalization")
