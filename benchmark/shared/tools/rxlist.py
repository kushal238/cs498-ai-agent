"""
RxList drug interaction checker tool.

Provides:
    check_rxlist_interactions(normalized_medications) -> list of interaction dicts

Uses https://www.rxlist.com/api/drugchecker/drugchecker.svc/
No API key required.
"""

from __future__ import annotations

import functools
import re
import sys

import requests

RXLIST_BASE = "https://www.rxlist.com/api/drugchecker/drugchecker.svc"

_SEVERITY_MAP = {
    "contraindicated": "contraindicated",
    "serious": "major",
    "significant": "moderate",
    "minor": "minor",
}


@functools.lru_cache(maxsize=256)
def _get_rxlist_id(ingredient: str) -> str | None:
    """Return the RxList numeric drug ID for an ingredient. Prefers oral route."""
    name = ingredient.strip().lower()
    if not name:
        return None
    try:
        resp = requests.get(f"{RXLIST_BASE}/druglist/{name}", timeout=10)
        if resp.status_code != 200:
            return None
        results = resp.json()
        if not results:
            return None
        for entry in results:
            entry_name = entry.get("Name", "").lower()
            if entry_name == f"{name} oral":
                return entry["ID"]
        for entry in results:
            if "oral" in entry.get("Name", "").lower() and "(" not in entry.get("Name", ""):
                return entry["ID"]
        return results[0]["ID"]
    except Exception as exc:
        print(f"[RxList] _get_rxlist_id({ingredient!r}) failed: {exc}", file=sys.stderr)
        return None


def _parse_severity(html: str) -> str:
    match = re.search(r'class="pillIcon (\w+)"', html)
    if match:
        return _SEVERITY_MAP.get(match.group(1), "unknown")
    return "unknown"


def _parse_drug_names(html: str) -> tuple[str, str] | None:
    """Extract (drug_a, drug_b) from the <h3> tag, stripping route suffixes."""
    h3 = re.search(r"<h3>(.*?)</h3>", html)
    if not h3:
        return None
    parts = re.split(r"\s+and\s+", h3.group(1).strip(), maxsplit=1)
    if len(parts) != 2:
        return None
    route_suffix = r"\s+(oral|iv|opht|topical|injectable|inhalation|nasal|rectal|transdermal).*$"
    drug_a = re.sub(route_suffix, "", parts[0]).strip()
    drug_b = re.sub(route_suffix, "", parts[1]).strip()
    return drug_a, drug_b


def _parse_clinician_text(html: str) -> str | None:
    """Extract the clinician explanation paragraph from tab-*-2."""
    tab2 = re.search(r'id="tab-\d+-2".*?</div>\s*</div>\s*</div>', html, re.DOTALL)
    block = tab2.group() if tab2 else html
    p = re.search(r"<p>(.*?)</p>", block, re.DOTALL)
    if not p:
        return None
    return re.sub(r"\s+", " ", p.group(1)).strip()


def check_rxlist_interactions(normalized_medications: list[dict]) -> list[dict]:
    """Check drug-drug interactions using the RxList interaction checker API.

    Looks up each drug's RxList ID then calls the interactionlist endpoint
    with all IDs at once, returning structured interaction data.

    Args:
        normalized_medications: List of dicts with at least an "ingredient" key.

    Returns:
        List of interaction dicts with keys: drug_a, drug_b, severity, recommendation.
    """
    if len(normalized_medications) < 2:
        return []

    id_map: dict[str, str] = {}
    for med in normalized_medications:
        ingredient = (med.get("ingredient") or "").strip()
        if not ingredient:
            continue
        rxlist_id = _get_rxlist_id(ingredient)
        if rxlist_id:
            id_map[ingredient] = rxlist_id

    if len(id_map) < 2:
        return []

    ids_str = "_".join(id_map.values())
    try:
        resp = requests.get(f"{RXLIST_BASE}/interactionlist/{ids_str}", timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception as exc:
        print(f"[RxList] check_rxlist_interactions failed: {exc}", file=sys.stderr)
        return []

    interactions = []
    for html in data.get("DetailList", []):
        names = _parse_drug_names(html)
        text = _parse_clinician_text(html)
        if not names or not text:
            continue
        drug_a, drug_b = names
        interactions.append({
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity": _parse_severity(html),
            "recommendation": text,
        })

    return interactions
