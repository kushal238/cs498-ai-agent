"""
NCBI E-utilities (PubMed) API wrapper.

Real API base URL: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/

Key endpoints used:
    esearch.fcgi?db=pubmed&term={query}&retmax={n}&retmode=json
        Searches PubMed and returns a list of PMIDs matching the query.

    efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract&retmode=xml
        Fetches the abstract text for a given PMID.

NCBI requests an API key for >3 requests/second; set as env var NCBI_API_KEY.
NCBI usage policy: https://www.ncbi.nlm.nih.gov/home/about/policies/
"""

from __future__ import annotations

import os
import sys
import time
import xml.etree.ElementTree as ET

import requests

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Polite delay between requests when no API key is set (NCBI limit: 3 req/s)
_RATE_DELAY = 0.4


def _base_params() -> dict:
    params: dict = {"db": "pubmed"}
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def search_pubmed(query: str, max_results: int = 5) -> list[str]:
    """Search PubMed and return a list of PMIDs for the given clinical query.

    Args:
        query:       Free-text clinical query, e.g.
                     "chest pain atrial fibrillation differential diagnosis".
        max_results: Maximum number of PMIDs to return (default 5).

    Returns:
        List of PMID strings, e.g. ["38012345", "37654321"].
        Returns an empty list if no results are found.
    """
    params = {
        **_base_params(),
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    try:
        resp = requests.get(ESEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as exc:
        print(f"[PubMed] search_pubmed failed: {exc}", file=sys.stderr)
        return []
    finally:
        time.sleep(_RATE_DELAY)


def fetch_abstract(pmid: str) -> dict:
    """Fetch the abstract and metadata for a single PubMed article.

    Args:
        pmid: PubMed ID string, e.g. "38012345".

    Returns:
        A dict with keys:
            - "pmid"     (str):        The PMID.
            - "title"    (str):        Article title.
            - "abstract" (str):        Abstract text (may be empty for some articles).
            - "authors"  (list[str]):  Author names.
            - "year"     (str | None): Publication year.
    """
    params = {
        **_base_params(),
        "id": pmid,
        "rettype": "abstract",
        "retmode": "xml",
    }
    empty = {"pmid": pmid, "title": "", "abstract": "", "authors": [], "year": None}
    try:
        resp = requests.get(EFETCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        article = root.find(".//PubmedArticle/MedlineCitation/Article")
        if article is None:
            return empty

        title_el = article.find("ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""

        abstract_el = article.find("Abstract/AbstractText")
        abstract = "".join(abstract_el.itertext()) if abstract_el is not None else ""

        authors = []
        for author in article.findall("AuthorList/Author"):
            last  = author.findtext("LastName", "")
            first = author.findtext("ForeName", "")
            if last:
                authors.append(f"{last} {first}".strip())

        year_el = root.find(".//PubmedArticle/MedlineCitation/Article/Journal/JournalIssue/PubDate/Year")
        year = year_el.text if year_el is not None else None

        return {"pmid": pmid, "title": title, "abstract": abstract, "authors": authors, "year": year}
    except Exception as exc:
        print(f"[PubMed] fetch_abstract({pmid}) failed: {exc}", file=sys.stderr)
        return empty
    finally:
        time.sleep(_RATE_DELAY)


def find_supporting_citations(condition: str, context: str, max_results: int = 3) -> list[dict]:
    """Find PubMed citations that support a differential diagnosis entry.

    Args:
        condition:   The clinical condition being considered, e.g. "unstable angina".
        context:     Brief patient context to refine the search, e.g.
                     "45-year-old male with atrial fibrillation and chest pain".
        max_results: Maximum number of citations to return.

    Returns:
        List of citation dicts (see fetch_abstract return format), sorted by
        relevance (as ranked by PubMed's default relevance sort).
    """
    # Use just the condition name for the primary search to avoid over-filtering.
    # The context is used as a fallback if the condition alone returns nothing.
    pmids = search_pubmed(condition, max_results=max_results)
    if not pmids:
        pmids = search_pubmed(f"{condition} {context.split()[0]}", max_results=max_results)
    return [fetch_abstract(pmid) for pmid in pmids]
