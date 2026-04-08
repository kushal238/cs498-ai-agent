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

import functools
import itertools
import re
import sys

import requests

RXNAV_BASE  = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_URL = "https://api.fda.gov/drug/label.json"

ROUTE_WORDS = {
    "oral", "intravenous", "iv", "subcutaneous", "sq", "sc", "intramuscular",
    "im", "topical", "ophthalmic", "otic", "nasal", "intranasal", "inhaled",
    "inhalation", "transdermal", "rectal",
}

FREQUENCY_WORDS = {
    "daily", "weekly", "monthly", "bid", "tid", "qid", "prn",
    "needed", "hour", "hours", "day", "days",
}


def _clean_drug_query(drug_name: str) -> str:
    """Strip route, dose, and schedule noise while keeping the drug name."""
    text = (drug_name or "").strip().lower()
    if not text:
        return ""

    # Keep separators used in combination products, but normalize most punctuation.
    text = re.sub(r"[(),]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize slash-separated combos (e.g. "oxycodone/acetaminophen") into
    # space-separated form so individual components become separate tokens.
    text = re.sub(r"/", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for token in text.split():
        bare = token.strip()
        if not bare:
            continue
        if not tokens and bare in ROUTE_WORDS:
            continue
        if any(ch.isdigit() for ch in bare):
            break
        if bare in FREQUENCY_WORDS:
            break
        tokens.append(bare)

    cleaned = " ".join(tokens).strip()
    return cleaned


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
        cleaned_query = _clean_drug_query(drug_name)
        candidates = []
        for query in [drug_name, cleaned_query, cleaned_query.split()[0] if cleaned_query else "", drug_name.split()[0]]:
            query = (query or "").strip()
            if query and query not in candidates:
                candidates.append(query)
        # For combo drugs (e.g. "oxycodone acetaminophen"), also try each
        # individual component so at least one resolves to an RxCUI.
        cleaned_parts = cleaned_query.split()
        if len(cleaned_parts) > 1:
            for part in cleaned_parts:
                if part and part not in candidates:
                    candidates.append(part)

        # Try the original string first, then progressively cleaner fallbacks.
        for query in candidates:
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
        result["ingredient"] = _canonicalize_ingredient_name(name)

    except Exception as exc:
        print(f"[RxNorm] get_rxcui({drug_name!r}) failed: {exc}", file=sys.stderr)

    return result


def _get_ingredient_name(rxcui: str) -> str | None:
    """Return the canonical ingredient name for an RxCUI, or None on failure."""
    try:
        resp = requests.get(f"{RXNAV_BASE}/rxcui/{rxcui}/properties.json", timeout=10)
        resp.raise_for_status()
        name = resp.json().get("properties", {}).get("name")
        return _canonicalize_ingredient_name(name)
    except Exception:
        return None


def _normalize_name(value: str) -> str:
    """Normalize a drug name for exact-ish matching against FDA labels."""
    normalized = re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _canonicalize_ingredient_name(name: str | None) -> str | None:
    """Strip common RxNorm qualifiers so outputs stay close to base ingredients."""
    if not name:
        return None
    base = name.split(",", 1)[0].strip().lower()
    return base or None


def _phrase_pattern(value: str) -> re.Pattern[str]:
    """Return a regex that matches a drug phrase on token boundaries."""
    tokens = [re.escape(t) for t in _normalize_name(value).split() if t]
    if not tokens:
        return re.compile(r"$^")
    phrase = r"(?:\s+|[-/])".join(tokens)
    return re.compile(rf"\b{phrase}\b", flags=re.IGNORECASE)


def _score_generic_candidate(candidate: str, target: str) -> int:
    """Score how well an OpenFDA generic_name candidate matches a target ingredient."""
    cand = _normalize_name(candidate)
    tgt = _normalize_name(target)
    if not cand or not tgt:
        return -1

    # De-prioritize combination products when looking for single-ingredient labels.
    if " and " in cand or ";" in candidate or "/" in candidate:
        combo_penalty = 40
    else:
        combo_penalty = 0

    if cand == tgt:
        return 100 - combo_penalty
    if cand.startswith(f"{tgt} ") or cand.startswith(f"{tgt}-"):
        return 90 - combo_penalty - max(0, len(cand.split()) - len(tgt.split()))
    if cand.endswith(f" {tgt}"):
        return 85 - combo_penalty - max(0, len(cand.split()) - len(tgt.split()))
    if _phrase_pattern(target).search(cand):
        return 70 - combo_penalty - max(0, len(cand.split()) - len(tgt.split()))
    return -1


@functools.lru_cache(maxsize=128)
def _fetch_best_label_result(ingredient: str) -> dict | None:
    """Fetch the best matching OpenFDA label record for an ingredient."""
    target = _normalize_name(ingredient)
    if not target:
        return None

    # Quote the ingredient to improve precision, but still filter locally because
    # OpenFDA search remains fuzzy for some terms (e.g. metformin).
    queries = [
        f'openfda.generic_name:"{ingredient}"',
        f"openfda.generic_name:{ingredient}",
    ]
    best_result: dict | None = None
    best_score = -1

    for query in queries:
        try:
            resp = requests.get(
                OPENFDA_URL,
                params={"search": query, "limit": 10},
                timeout=10,
            )
            resp.raise_for_status()
        except Exception:
            continue

        for result in resp.json().get("results", []):
            generic_names = result.get("openfda", {}).get("generic_name") or []
            if not generic_names:
                continue
            score = max(_score_generic_candidate(name, target) for name in generic_names)
            if score > best_score:
                best_result = result
                best_score = score

        if best_score >= 90:
            break

    return best_result if best_score >= 0 else None


def _best_matching_sentence(interactions_text: str, target_drug: str) -> str | None:
    """Return the best interaction sentence that explicitly mentions the target drug."""
    if not interactions_text:
        return None
    pattern = _phrase_pattern(target_drug)
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", interactions_text)
        if s.strip()
    ]
    exact_matches = [s for s in sentences if pattern.search(s)]
    if exact_matches:
        # Prefer the shortest sentence that still contains the exact mention.
        best = min(exact_matches, key=len).strip(" ;:")
        if len(best) <= 400:
            return best

        # Some FDA labels flatten whole tables into one "sentence". In that case,
        # return a compact local excerpt around the exact drug mention.
        match = pattern.search(best)
        if match:
            left_window = best[max(0, match.start() - 180):match.start()]
            right_window = best[match.end():min(len(best), match.end() + 220)]

            left_breaks = [m.end() for m in re.finditer(r"[.;:]\s+|\s{2,}", left_window)]
            right_break = re.search(r"[.;:]\s+|\s{2,}", right_window)
            keyword_match = None
            for keyword in [r"increase", r"decrease", r"contraindicat", r"monitor", r"avoid"]:
                for m in re.finditer(keyword, left_window, flags=re.IGNORECASE):
                    keyword_match = m
                if keyword_match:
                    break

            if keyword_match:
                start = max(0, match.start() - 180) + keyword_match.start()
            elif left_breaks:
                start = max(0, match.start() - 180) + left_breaks[-1]
            else:
                start = best.rfind(" ", 0, match.start() - 40)
                start = 0 if start == -1 else start + 1

            if right_break:
                end = match.end() + right_break.start()
            else:
                end = best.find(" ", match.end() + 180)
                end = len(best) if end == -1 else end

            compact = best[start:end].strip(" ;:")
            return compact
        return best
    return None


def _infer_severity(excerpt: str) -> str:
    """Infer a conservative severity level from FDA label wording."""
    text = (excerpt or "").lower()
    if not text:
        return "unknown"

    contraindicated_patterns = [
        r"\bcontraindicat",
        r"\bdo not (?:use|coadminister|administer)\b",
        r"\bnever (?:use|coadminister|administer)\b",
    ]
    major_patterns = [
        r"\bavoid concomitant use\b",
        r"\bavoid coadministration\b",
        r"\blife[- ]threatening\b",
        r"\bfatal\b",
        r"\bserious\b",
        r"\bmajor\b",
        r"\bhemorrhag",
        r"\bbleeding risk\b",
        r"\brisk of bleeding\b",
        r"\bqt prolong",
    ]
    moderate_patterns = [
        r"\bmonitor\b",
        r"\bmonitoring\b",
        r"\bdose adjustment\b",
        r"\badjust dose\b",
        r"\breduce dose\b",
        r"\bincrease(?:d|s)? [a-z0-9\s-]{0,30}(?:level|levels|concentration|concentrations|exposure|bioavailability)\b",
        r"\bincrease(?:d)? .*exposure\b",
        r"\bincreased .*bioavailability\b",
        r"\bauc\b",
        r"\bc max\b",
        r"\bconcentration\b",
        r"\bincreased activity\b",
        r"\blactic acidosis\b",
    ]
    minor_patterns = [
        r"\binterfere with absorption\b",
        r"\bdecrease absorption\b",
        r"\bseparate administration\b",
        r"\badminister.*separately\b",
    ]

    if any(re.search(p, text) for p in contraindicated_patterns):
        return "contraindicated"
    if any(re.search(p, text) for p in major_patterns):
        return "major"
    if any(re.search(p, text) for p in moderate_patterns):
        return "moderate"
    if any(re.search(p, text) for p in minor_patterns):
        return "minor"
    return "unknown"


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
            label_a = _fetch_best_label_result(drug_a)
            label_b = _fetch_best_label_result(drug_b)

            excerpt = None
            if label_a:
                text_a = " ".join(label_a.get("drug_interactions", []))
                excerpt = _best_matching_sentence(text_a, drug_b)
            if excerpt is None and label_b:
                text_b = " ".join(label_b.get("drug_interactions", []))
                excerpt = _best_matching_sentence(text_b, drug_a)
            if excerpt is None:
                continue

            interactions.append({
                "drug_a": drug_a,
                "drug_b": drug_b,
                "severity": _infer_severity(excerpt),
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
