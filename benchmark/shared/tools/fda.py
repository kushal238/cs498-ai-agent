"""
OpenFDA drug label API wrapper.

Provides:
    get_fda_id(ingredient) -> application_number (e.g. NDA021081)
    check_fda_interactions(normalized_medications) -> list of interaction dicts

Uses https://api.fda.gov/drug/label.json. No API key required.
"""

from __future__ import annotations

import sys

import requests

OPENFDA_URL = "https://api.fda.gov/drug/label.json"


def get_fda_id(ingredient: str) -> str | None:
    """Look up OpenFDA drug label application_number by generic ingredient name.

    Queries the OpenFDA drug label API and returns the first matching
    application_number (e.g. "NDA021081", "ANDA207418") for the given ingredient.

    Args:
        ingredient: Normalized generic ingredient name (e.g. "metformin", "oxycodone").

    Returns:
        Application number string, or None if not found or on error.
    """
    if not ingredient or not ingredient.strip():
        return None
    search_name = ingredient.split(";")[0].split("/")[0].strip()
    try:
        resp = requests.get(
            OPENFDA_URL,
            params={"search": f"openfda.generic_name:{search_name}", "limit": 1},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        app_numbers = results[0].get("openfda", {}).get("application_number") or []
        return app_numbers[0] if app_numbers else None
    except Exception as exc:
        print(f"[FDA] get_fda_id({ingredient!r}) failed: {exc}", file=sys.stderr)
        return None


def _ingredient_tokens(ingredient: str) -> list[str]:
    s = (ingredient or "").lower().replace(";", " ").replace("/", " ")
    return [t.strip() for t in s.split() if len(t.strip()) > 2]


def _fetch_label_excerpt(app_number: str, search_tokens: list[str]) -> str | None:
    try:
        resp = requests.get(
            OPENFDA_URL,
            params={"search": f'openfda.application_number:"{app_number}"', "limit": 1},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        if not results:
            return None
        interactions_text = " ".join(results[0].get("drug_interactions", []))
        text_lower = interactions_text.lower()
        if not any(tok in text_lower for tok in search_tokens):
            return None
        # Split on ". " so "e.g." is not broken into " (e" + "g."
        sentences = interactions_text.split(". ")
        excerpt = next(
            (
                s.strip()
                for s in sentences
                if any(tok in s.lower() for tok in search_tokens)
            ),
            interactions_text[:1500],
        )
        return excerpt
    except Exception as exc:
        print(f"[FDA] _fetch_label_excerpt failed: {exc}", file=sys.stderr)
        return None


def check_fda_interactions(
    normalized_medications: list[dict],
) -> list[dict]:
    """Check drug-drug interactions using OpenFDA labels.

    For each pair of medications, uses application_number (fda_id or
    openfda_application_number) when available to fetch the FDA label and
    checks whether the other drug's ingredient is mentioned in the
    drug_interactions section.

    Args:
        normalized_medications: List of dicts with keys:
            - "ingredient" (str)
            - "openfda_application_number" or "fda_id" (str | None)

    Returns:
        List of interaction dicts with keys: drug_a, drug_b, severity, recommendation.
        Severity is "unknown" (FDA labels do not provide severity grades).
    """
    n = len(normalized_medications)
    if n < 2:
        return []

    interactions = []
    for i in range(n):
        for j in range(i + 1, n):
            med_a = normalized_medications[i]
            med_b = normalized_medications[j]
            ingredient_a = (med_a.get("ingredient") or "").strip()
            ingredient_b = (med_b.get("ingredient") or "").strip()
            if not ingredient_a or not ingredient_b:
                continue
            app_a = med_a.get("openfda_application_number") or med_a.get("fda_id")
            app_b = med_b.get("openfda_application_number") or med_b.get("fda_id")
            tokens_a = _ingredient_tokens(ingredient_a)
            tokens_b = _ingredient_tokens(ingredient_b)
            excerpt = None
            if app_a:
                excerpt = _fetch_label_excerpt(app_a, tokens_b)
            if excerpt is None and app_b:
                excerpt = _fetch_label_excerpt(app_b, tokens_a)
            if excerpt:
                interactions.append({
                    "drug_a": ingredient_a,
                    "drug_b": ingredient_b,
                    "severity": "unknown",
                    "recommendation": excerpt,
                })

    return interactions
