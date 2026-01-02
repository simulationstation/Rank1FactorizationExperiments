"""
CERN Open Data Portal source module.

CERN Open Data (https://opendata.cern.ch) provides datasets from
LHC experiments including CMS, ATLAS, LHCb, and ALICE.

API Documentation: https://opendata.cern.ch/docs/api
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

OPENDATA_API_BASE = "https://opendata.cern.ch/api/records"
OPENDATA_SEARCH_URL = "https://opendata.cern.ch/search"


# Map collaboration names to CERN Open Data experiment filters
EXPERIMENT_FILTERS = {
    "CMS": "CMS",
    "ATLAS": "ATLAS",
    "LHCb": "LHCb",
    "ALICE": "ALICE",
}


def plan_queries(candidate: "Candidate") -> List[str]:
    """
    Plan CERN Open Data search queries for a candidate.

    Returns list of query URLs that would be executed.
    """
    queries = []

    # Determine experiment filter
    experiment = None
    if candidate.collaboration:
        experiment = EXPERIMENT_FILTERS.get(candidate.collaboration.upper())

    # Query by search terms
    for term in candidate.search_terms:
        encoded = quote_plus(term)
        query = f"{OPENDATA_API_BASE}?q={encoded}&size=10"
        if experiment:
            query += f"&experiment={experiment}"
        queries.append(query)

    # Query by states
    for state in candidate.states:
        encoded = quote_plus(state)
        query = f"{OPENDATA_API_BASE}?q={encoded}&size=5"
        if experiment:
            query += f"&experiment={experiment}"
        queries.append(query)

    return queries


def execute_search(candidate: "Candidate") -> List[Dict[str, Any]]:
    """
    Execute CERN Open Data searches and return discovered URLs.

    NOTE: This actually makes network requests. Only call when --execute is set.

    Returns list of dicts with keys: url, type, record_id, title, files
    """
    import requests

    discovered = []
    queries = plan_queries(candidate)

    for query_url in queries:
        try:
            logger.info(f"CERN Open Data query: {query_url}")
            response = requests.get(query_url, timeout=30)
            response.raise_for_status()

            data = response.json()
            hits = data.get("hits", {}).get("hits", [])

            for hit in hits:
                metadata = hit.get("metadata", {})
                record_id = hit.get("id")

                # Get file URLs
                files = []
                for f in metadata.get("files", []):
                    files.append({
                        "key": f.get("key", ""),
                        "size": f.get("size", 0),
                        "url": f.get("uri", ""),
                    })

                discovered.append({
                    "url": f"https://opendata.cern.ch/record/{record_id}",
                    "type": "cern_opendata_record",
                    "record_id": record_id,
                    "title": metadata.get("title", ""),
                    "experiment": metadata.get("experiment", ""),
                    "files": files,
                })

        except Exception as e:
            logger.warning(f"CERN Open Data query failed: {e}")

    # Deduplicate by record ID
    seen = set()
    unique = []
    for item in discovered:
        record_id = item.get("record_id")
        if record_id and record_id not in seen:
            seen.add(record_id)
            unique.append(item)

    return unique


def download(url: str, dest_dir: Path, timeout: int = 300) -> List[str]:
    """
    Download CERN Open Data files to destination directory.

    NOTE: CERN Open Data files can be large. Use appropriate timeouts.

    Returns list of downloaded file paths.
    """
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    try:
        # Get record metadata first
        if "/record/" in url:
            record_id = url.split("/record/")[-1].rstrip("/")
            api_url = f"{OPENDATA_API_BASE}/{record_id}"

            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            data = response.json()

            metadata = data.get("metadata", {})
            files = metadata.get("files", [])

            # Download each file (limit to reasonable size)
            MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB limit

            for f in files:
                file_url = f.get("uri", "")
                file_size = f.get("size", 0)
                file_key = f.get("key", "unknown")

                if file_size > MAX_FILE_SIZE:
                    logger.warning(f"Skipping large file: {file_key} ({file_size} bytes)")
                    continue

                if file_url:
                    logger.info(f"Downloading: {file_key}")
                    file_response = requests.get(file_url, timeout=timeout)
                    file_response.raise_for_status()

                    filepath = dest_dir / f"opendata_{record_id}_{file_key}"
                    with open(filepath, 'wb') as out:
                        out.write(file_response.content)
                    downloaded.append(str(filepath))

    except Exception as e:
        logger.error(f"CERN Open Data download failed: {e}")

    return downloaded
