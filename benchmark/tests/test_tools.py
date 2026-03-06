"""
Integration tests for benchmark/shared/tools/ (PubMed + RxNorm).

These tests make REAL network calls to NCBI and NIH APIs.
They are skipped automatically when the network is unavailable.

Run with:
    pytest benchmark/tests/test_tools.py -v -m integration

Mark: @pytest.mark.integration
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ---------------------------------------------------------------------------
# Network availability check
# ---------------------------------------------------------------------------

def _can_reach_ncbi() -> bool:
    try:
        import requests
        r = requests.get("https://eutils.ncbi.nlm.nih.gov", timeout=5)
        return r.status_code < 500
    except Exception:
        return False


def _can_reach_rxnav() -> bool:
    try:
        import requests
        r = requests.get("https://rxnav.nlm.nih.gov", timeout=5)
        return r.status_code < 500
    except Exception:
        return False


network_ncbi  = pytest.mark.skipif(not _can_reach_ncbi(),  reason="NCBI unreachable")
network_rxnav = pytest.mark.skipif(not _can_reach_rxnav(), reason="RxNav unreachable")

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------

class TestSearchPubmed:
    @network_ncbi
    def test_returns_list_of_pmids(self):
        from benchmark.shared.tools.pubmed import search_pubmed
        pmids = search_pubmed("unstable angina", max_results=3)
        assert isinstance(pmids, list)
        assert len(pmids) <= 3
        assert all(isinstance(p, str) for p in pmids)

    @network_ncbi
    def test_known_query_returns_results(self):
        from benchmark.shared.tools.pubmed import search_pubmed
        pmids = search_pubmed("aspirin myocardial infarction", max_results=5)
        assert len(pmids) > 0

    @network_ncbi
    def test_empty_query_returns_list(self):
        from benchmark.shared.tools.pubmed import search_pubmed
        # Even a garbage query should return an empty list, not raise
        pmids = search_pubmed("zzzzzzzzz_not_a_real_medical_term_xyz", max_results=3)
        assert isinstance(pmids, list)

    @network_ncbi
    def test_max_results_respected(self):
        from benchmark.shared.tools.pubmed import search_pubmed
        pmids = search_pubmed("diabetes", max_results=2)
        assert len(pmids) <= 2


class TestFetchAbstract:
    @network_ncbi
    def test_returns_expected_keys(self):
        from benchmark.shared.tools.pubmed import search_pubmed, fetch_abstract
        pmids = search_pubmed("heart failure treatment", max_results=1)
        if not pmids:
            pytest.skip("No PMIDs returned by search")
        result = fetch_abstract(pmids[0])
        assert set(result.keys()) == {"pmid", "title", "abstract", "authors", "year"}

    @network_ncbi
    def test_pmid_preserved(self):
        from benchmark.shared.tools.pubmed import search_pubmed, fetch_abstract
        pmids = search_pubmed("hypertension", max_results=1)
        if not pmids:
            pytest.skip("No PMIDs returned by search")
        result = fetch_abstract(pmids[0])
        assert result["pmid"] == pmids[0]

    @network_ncbi
    def test_invalid_pmid_returns_empty(self):
        from benchmark.shared.tools.pubmed import fetch_abstract
        result = fetch_abstract("00000000000")  # extremely unlikely to be a real PMID
        assert result["pmid"] == "00000000000"
        assert isinstance(result["title"], str)
        assert isinstance(result["authors"], list)

    @network_ncbi
    def test_title_nonempty_for_real_pmid(self):
        from benchmark.shared.tools.pubmed import fetch_abstract
        # PMID 7628028 is a well-known aspirin/MI paper (NEJM 1994)
        result = fetch_abstract("7628028")
        assert len(result["title"]) > 0


class TestFindSupportingCitations:
    @network_ncbi
    def test_returns_list_of_dicts(self):
        from benchmark.shared.tools.pubmed import find_supporting_citations
        citations = find_supporting_citations(
            condition="unstable angina",
            context="45-year-old male with chest pain",
            max_results=2,
        )
        assert isinstance(citations, list)
        for c in citations:
            assert "pmid" in c
            assert "title" in c

    @network_ncbi
    def test_returns_at_most_max_results(self):
        from benchmark.shared.tools.pubmed import find_supporting_citations
        citations = find_supporting_citations("GERD", "chest pain", max_results=2)
        assert len(citations) <= 2


# ---------------------------------------------------------------------------
# RxNorm
# ---------------------------------------------------------------------------

class TestGetRxcui:
    @network_rxnav
    def test_known_drug_returns_rxcui(self):
        from benchmark.shared.tools.rxnorm import get_rxcui
        result = get_rxcui("warfarin")
        assert result["rxnorm_id"] is not None
        assert result["ingredient"] is not None

    @network_rxnav
    def test_brand_name_resolves(self):
        from benchmark.shared.tools.rxnorm import get_rxcui
        result = get_rxcui("Coumadin")
        assert result["rxnorm_id"] is not None

    @network_rxnav
    def test_drug_with_dosage_string(self):
        from benchmark.shared.tools.rxnorm import get_rxcui
        # Dosage strings like "5mg daily" should be stripped and still resolve
        result = get_rxcui("warfarin 5mg daily")
        assert result["rxnorm_id"] is not None
        assert result["ingredient"] == "warfarin"

    @network_rxnav
    def test_unknown_drug_returns_none(self):
        from benchmark.shared.tools.rxnorm import get_rxcui
        result = get_rxcui("xyzzzz_not_a_drug_99999")
        assert result["rxnorm_id"] is None
        assert result["ingredient"] is None

    @network_rxnav
    def test_returns_required_keys(self):
        from benchmark.shared.tools.rxnorm import get_rxcui
        result = get_rxcui("aspirin")
        assert set(result.keys()) >= {"rxnorm_id", "ingredient", "raw_response"}


class TestNormalizeMedicationList:
    @network_rxnav
    def test_normalizes_multiple_drugs(self):
        from benchmark.shared.tools.rxnorm import normalize_medication_list
        meds = ["warfarin 5mg daily", "aspirin 81mg", "metoprolol 25mg"]
        results = normalize_medication_list(meds)
        assert len(results) == 3
        originals = [r["original"] for r in results]
        assert originals == meds

    @network_rxnav
    def test_preserves_original_string(self):
        from benchmark.shared.tools.rxnorm import normalize_medication_list
        meds = ["aspirin 81mg QD"]
        results = normalize_medication_list(meds)
        assert results[0]["original"] == "aspirin 81mg QD"

    @network_rxnav
    def test_empty_list_returns_empty(self):
        from benchmark.shared.tools.rxnorm import normalize_medication_list
        assert normalize_medication_list([]) == []


class TestCheckInteractions:
    @network_rxnav
    def test_warfarin_aspirin_interaction_detected(self):
        from benchmark.shared.tools.rxnorm import get_rxcui, check_interactions
        warfarin_rxcui = get_rxcui("warfarin")["rxnorm_id"]
        aspirin_rxcui  = get_rxcui("aspirin")["rxnorm_id"]
        if not warfarin_rxcui or not aspirin_rxcui:
            pytest.skip("Could not resolve RxCUIs")
        interactions = check_interactions([warfarin_rxcui, aspirin_rxcui])
        assert isinstance(interactions, list)
        # Known major interaction — should be detected
        assert len(interactions) > 0

    @network_rxnav
    def test_single_drug_returns_empty(self):
        from benchmark.shared.tools.rxnorm import get_rxcui, check_interactions
        rxcui = get_rxcui("aspirin")["rxnorm_id"]
        if not rxcui:
            pytest.skip("Could not resolve RxCUI for aspirin")
        interactions = check_interactions([rxcui])
        assert interactions == []

    @network_rxnav
    def test_empty_list_returns_empty(self):
        from benchmark.shared.tools.rxnorm import check_interactions
        assert check_interactions([]) == []

    @network_rxnav
    def test_interaction_has_required_keys(self):
        from benchmark.shared.tools.rxnorm import get_rxcui, check_interactions
        warfarin_rxcui = get_rxcui("warfarin")["rxnorm_id"]
        aspirin_rxcui  = get_rxcui("aspirin")["rxnorm_id"]
        if not warfarin_rxcui or not aspirin_rxcui:
            pytest.skip("Could not resolve RxCUIs")
        interactions = check_interactions([warfarin_rxcui, aspirin_rxcui])
        if not interactions:
            pytest.skip("No interactions detected (OpenFDA label may have changed)")
        for ix in interactions:
            assert "drug_a" in ix
            assert "drug_b" in ix
            assert "severity" in ix
            assert "recommendation" in ix
