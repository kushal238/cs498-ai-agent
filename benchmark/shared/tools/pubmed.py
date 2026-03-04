"""
NCBI E-utilities (PubMed) API wrapper stubs.

Real API base URL: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/

Key endpoints used:
    esearch.fcgi?db=pubmed&term={query}&retmax={n}&retmode=json
        Searches PubMed and returns a list of PMIDs matching the query.

    efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract&retmode=text
        Fetches the abstract text for a given PMID.

NCBI requests an API key for >3 requests/second; set as env var NCBI_API_KEY.
NCBI usage policy: https://www.ncbi.nlm.nih.gov/home/about/policies/
"""

from __future__ import annotations


def search_pubmed(query: str, max_results: int = 5) -> list[str]:
    """Search PubMed and return a list of PMIDs for the given clinical query.

    Args:
        query:       Free-text clinical query, e.g.
                     "chest pain atrial fibrillation differential diagnosis".
        max_results: Maximum number of PMIDs to return (default 5).

    Returns:
        List of PMID strings, e.g. ["38012345", "37654321"].
        Returns an empty list if no results are found.

    Raises:
        NotImplementedError: Always — real API call not yet implemented.

    TODO: Implement using requests.get() to
          https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
          with params db="pubmed", term=query, retmax=max_results, retmode="json".
          Parse esearchresult.idlist from the JSON response.
          Include NCBI_API_KEY from os.environ if set.
    """
    raise NotImplementedError("TODO: implement PubMed search via NCBI E-utilities")


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

    Raises:
        NotImplementedError: Always — real API call not yet implemented.

    TODO: Implement using requests.get() to
          https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
          with params db="pubmed", id=pmid, rettype="abstract", retmode="xml".
          Parse the XML response with xml.etree.ElementTree.
    """
    raise NotImplementedError("TODO: implement PubMed abstract fetch via NCBI E-utilities")


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

    Raises:
        NotImplementedError: Always — real API call not yet implemented.

    TODO: Construct a query string combining condition + context keywords,
          call search_pubmed(), then call fetch_abstract() for each PMID.
    """
    raise NotImplementedError("TODO: implement citation lookup combining search + fetch")
