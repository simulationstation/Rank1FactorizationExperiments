"""
arXiv source module.

arXiv (https://arxiv.org) hosts preprints of physics papers.
We use it to download PDFs for table extraction.

API Documentation: https://info.arxiv.org/help/api/
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"
ARXIV_ABS_BASE = "https://arxiv.org/abs"


def plan_queries(candidate: "Candidate") -> List[str]:
    """
    Plan arXiv search queries for a candidate.

    Returns list of query URLs that would be executed.
    """
    queries = []

    # Direct arXiv IDs
    for arxiv_id in candidate.arxiv_ids:
        # Clean up ID format
        clean_id = arxiv_id.replace("arXiv:", "").replace("arxiv:", "")
        queries.append(f"{ARXIV_API_BASE}?id_list={clean_id}")

    # Search by terms
    for term in candidate.search_terms:
        encoded = quote_plus(term)
        # Search in title and abstract
        queries.append(
            f"{ARXIV_API_BASE}?search_query=all:{encoded}&start=0&max_results=10"
        )

    # Search by collaboration + category
    if candidate.collaboration:
        collab = candidate.collaboration.lower()
        # Map to arXiv categories
        category = "hep-ex"  # Default to experimental HEP
        if "theory" in candidate.notes.lower():
            category = "hep-ph"

        for state in candidate.states[:2]:  # Limit to avoid too many queries
            term = f"{collab} {state}"
            encoded = quote_plus(term)
            queries.append(
                f"{ARXIV_API_BASE}?search_query=all:{encoded}+AND+cat:{category}&max_results=5"
            )

    return queries


def execute_search(candidate: "Candidate") -> List[Dict[str, Any]]:
    """
    Execute arXiv searches and return discovered URLs.

    NOTE: This actually makes network requests. Only call when --execute is set.

    Returns list of dicts with keys: url, type, arxiv_id, title, pdf_url
    """
    import requests
    import xml.etree.ElementTree as ET

    discovered = []
    queries = plan_queries(candidate)

    for query_url in queries:
        try:
            logger.info(f"arXiv query: {query_url}")
            response = requests.get(query_url, timeout=30)
            response.raise_for_status()

            # Parse Atom XML response
            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", ns):
                title_elem = entry.find("atom:title", ns)
                id_elem = entry.find("atom:id", ns)

                if id_elem is not None:
                    arxiv_url = id_elem.text
                    # Extract arXiv ID from URL
                    match = re.search(r"arxiv.org/abs/(.+)$", arxiv_url)
                    if match:
                        arxiv_id = match.group(1)
                        discovered.append({
                            "url": arxiv_url,
                            "type": "arxiv_paper",
                            "arxiv_id": arxiv_id,
                            "title": title_elem.text.strip() if title_elem is not None else "",
                            "pdf_url": f"{ARXIV_PDF_BASE}/{arxiv_id}.pdf",
                        })

        except Exception as e:
            logger.warning(f"arXiv query failed: {e}")

    # Deduplicate by arXiv ID
    seen = set()
    unique = []
    for item in discovered:
        arxiv_id = item.get("arxiv_id", "")
        if arxiv_id and arxiv_id not in seen:
            seen.add(arxiv_id)
            unique.append(item)

    return unique


def download(url: str, dest_dir: Path, timeout: int = 120) -> List[str]:
    """
    Download arXiv PDF to destination directory.

    Returns list of downloaded file paths.
    """
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    try:
        # Convert abs URL to PDF URL if needed
        pdf_url = url
        if "/abs/" in url:
            arxiv_id = url.split("/abs/")[-1]
            pdf_url = f"{ARXIV_PDF_BASE}/{arxiv_id}.pdf"
        elif not url.endswith(".pdf"):
            pdf_url = url + ".pdf"

        logger.info(f"Downloading: {pdf_url}")
        response = requests.get(pdf_url, timeout=timeout)
        response.raise_for_status()

        # Extract filename
        arxiv_id = pdf_url.split("/")[-1].replace(".pdf", "")
        filename = f"arxiv_{arxiv_id.replace('/', '_')}.pdf"
        filepath = dest_dir / filename

        with open(filepath, 'wb') as f:
            f.write(response.content)
        downloaded.append(str(filepath))

    except Exception as e:
        logger.error(f"arXiv download failed: {e}")

    return downloaded


def get_paper_info(arxiv_id: str) -> Dict[str, Any]:
    """
    Get metadata for an arXiv paper.

    NOTE: Makes network request.
    """
    import requests
    import xml.etree.ElementTree as ET

    try:
        url = f"{ARXIV_API_BASE}?id_list={arxiv_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entry = root.find("atom:entry", ns)
        if entry is not None:
            return {
                "arxiv_id": arxiv_id,
                "title": entry.findtext("atom:title", "", ns).strip(),
                "summary": entry.findtext("atom:summary", "", ns).strip(),
                "published": entry.findtext("atom:published", "", ns),
                "updated": entry.findtext("atom:updated", "", ns),
                "pdf_url": f"{ARXIV_PDF_BASE}/{arxiv_id}.pdf",
            }

    except Exception as e:
        logger.error(f"Failed to get arXiv info: {e}")

    return {}
