"""
INSPIRE-HEP source module.

INSPIRE (https://inspirehep.net) is the main literature database
for high-energy physics. Used to find paper metadata and links.

API Documentation: https://inspirehep.net/api
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

INSPIRE_API_BASE = "https://inspirehep.net/api"
INSPIRE_LITERATURE = f"{INSPIRE_API_BASE}/literature"


def plan_queries(candidate: "Candidate") -> List[str]:
    """
    Plan INSPIRE search queries for a candidate.

    Returns list of query URLs that would be executed.
    """
    queries = []

    # Query by search terms
    for term in candidate.search_terms:
        encoded = quote_plus(term)
        queries.append(f"{INSPIRE_LITERATURE}?q={encoded}&size=10")

    # Query by collaboration + state
    if candidate.collaboration:
        for state in candidate.states[:3]:
            term = f"collaboration:{candidate.collaboration} {state}"
            encoded = quote_plus(term)
            queries.append(f"{INSPIRE_LITERATURE}?q={encoded}&size=5")

    return queries


def execute_search(candidate: "Candidate") -> List[Dict[str, Any]]:
    """
    Execute INSPIRE searches and return discovered URLs.

    NOTE: This actually makes network requests. Only call when --execute is set.

    Returns list of dicts with keys: url, type, inspire_id, arxiv_id, doi, title
    """
    import requests

    discovered = []
    queries = plan_queries(candidate)

    for query_url in queries:
        try:
            logger.info(f"INSPIRE query: {query_url}")
            response = requests.get(query_url, timeout=30)
            response.raise_for_status()

            data = response.json()
            hits = data.get("hits", {}).get("hits", [])

            for hit in hits:
                metadata = hit.get("metadata", {})

                # Extract arXiv ID if available
                arxiv_id = None
                arxiv_eprints = metadata.get("arxiv_eprints", [])
                if arxiv_eprints:
                    arxiv_id = arxiv_eprints[0].get("value")

                # Extract DOI if available
                doi = None
                dois = metadata.get("dois", [])
                if dois:
                    doi = dois[0].get("value")

                discovered.append({
                    "url": f"https://inspirehep.net/literature/{hit.get('id')}",
                    "type": "inspire_record",
                    "inspire_id": hit.get("id"),
                    "arxiv_id": arxiv_id,
                    "doi": doi,
                    "title": metadata.get("titles", [{}])[0].get("title", ""),
                    "collaboration": metadata.get("collaborations", [{}])[0].get("value", ""),
                })

        except Exception as e:
            logger.warning(f"INSPIRE query failed: {e}")

    # Deduplicate by INSPIRE ID
    seen = set()
    unique = []
    for item in discovered:
        inspire_id = item.get("inspire_id")
        if inspire_id and inspire_id not in seen:
            seen.add(inspire_id)
            unique.append(item)

    return unique


def get_paper_metadata(inspire_id: str) -> Optional[Dict]:
    """
    Get detailed metadata for an INSPIRE record.

    NOTE: Makes network request.
    """
    import requests

    try:
        url = f"{INSPIRE_LITERATURE}/{inspire_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get INSPIRE metadata: {e}")
        return None


def find_hepdata_link(inspire_id: str) -> Optional[str]:
    """
    Check if an INSPIRE record has associated HEPData.

    NOTE: Makes network request.
    """
    import requests

    try:
        url = f"{INSPIRE_LITERATURE}/{inspire_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        metadata = data.get("metadata", {})

        # Check for HEPData link in external references
        external_refs = metadata.get("external_system_identifiers", [])
        for ref in external_refs:
            if ref.get("schema") == "HEPData":
                return f"https://www.hepdata.net/record/ins{ref.get('value')}"

        # Check documents for HEPData
        documents = metadata.get("documents", [])
        for doc in documents:
            if "hepdata" in doc.get("url", "").lower():
                return doc.get("url")

    except Exception as e:
        logger.error(f"Failed to check HEPData link: {e}")

    return None
