"""
HEPData source module.

HEPData (https://www.hepdata.net) is the primary source for published
high-energy physics data tables.

API Documentation: https://www.hepdata.net/api
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

HEPDATA_API_BASE = "https://www.hepdata.net/api"
HEPDATA_SEARCH_URL = "https://www.hepdata.net/search"


def plan_queries(candidate: "Candidate") -> List[str]:
    """
    Plan HEPData search queries for a candidate.

    Returns list of query URLs that would be executed.
    """
    queries = []

    # Query by search terms
    for term in candidate.search_terms:
        encoded = quote_plus(term)
        queries.append(f"{HEPDATA_SEARCH_URL}?q={encoded}&format=json")

    # Query by known HEPData IDs
    for hep_id in candidate.hepdata_ids:
        queries.append(f"{HEPDATA_API_BASE}/record/ins{hep_id}")

    # Query by collaboration + states
    if candidate.collaboration:
        for state in candidate.states:
            term = f"{candidate.collaboration} {state}"
            encoded = quote_plus(term)
            queries.append(f"{HEPDATA_SEARCH_URL}?q={encoded}&format=json")

    return queries


def execute_search(candidate: "Candidate") -> List[Dict[str, Any]]:
    """
    Execute HEPData searches and return discovered URLs.

    NOTE: This actually makes network requests. Only call when --execute is set.

    Returns list of dicts with keys: url, type, record_id, table_id, description
    """
    import requests

    discovered = []
    queries = plan_queries(candidate)

    for query_url in queries:
        try:
            logger.info(f"HEPData query: {query_url}")
            response = requests.get(query_url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Handle search results
            if "results" in data:
                for result in data["results"]:
                    record_id = result.get("recid")
                    if record_id:
                        discovered.append({
                            "url": f"{HEPDATA_API_BASE}/record/ins{record_id}",
                            "type": "hepdata_record",
                            "record_id": record_id,
                            "title": result.get("title", ""),
                            "collaboration": result.get("collaboration", ""),
                        })

            # Handle direct record lookup
            elif "record" in data:
                record = data["record"]
                record_id = record.get("recid")
                tables = record.get("tables", [])
                for table in tables:
                    table_id = table.get("id")
                    discovered.append({
                        "url": f"{HEPDATA_API_BASE}/table/ins{record_id}/{table_id}",
                        "type": "hepdata_table",
                        "record_id": record_id,
                        "table_id": table_id,
                        "description": table.get("description", ""),
                    })

        except Exception as e:
            logger.warning(f"HEPData query failed: {e}")

    # Deduplicate by URL
    seen = set()
    unique = []
    for item in discovered:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)

    return unique


def download(url: str, dest_dir: Path, timeout: int = 60) -> List[str]:
    """
    Download HEPData table(s) to destination directory.

    Returns list of downloaded file paths.
    """
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    try:
        # Determine if this is a record or table URL
        if "/table/" in url:
            # Single table - download CSV
            csv_url = url.replace("/api/", "/download/table/") + "?format=csv"
            response = requests.get(csv_url, timeout=timeout)
            response.raise_for_status()

            # Extract filename from URL
            parts = url.split("/")
            filename = f"hepdata_{parts[-2]}_{parts[-1]}.csv"
            filepath = dest_dir / filename

            with open(filepath, 'wb') as f:
                f.write(response.content)
            downloaded.append(str(filepath))

        elif "/record/" in url:
            # Full record - download JSON
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            parts = url.split("/")
            filename = f"hepdata_{parts[-1]}.json"
            filepath = dest_dir / filename

            with open(filepath, 'wb') as f:
                f.write(response.content)
            downloaded.append(str(filepath))

            # Also download individual tables as CSV
            data = response.json()
            if "record" in data:
                tables = data["record"].get("tables", [])
                record_id = parts[-1]
                for table in tables:
                    table_id = table.get("id")
                    if table_id:
                        table_files = download(
                            f"{HEPDATA_API_BASE}/table/{record_id}/{table_id}",
                            dest_dir,
                            timeout
                        )
                        downloaded.extend(table_files)

    except Exception as e:
        logger.error(f"HEPData download failed: {e}")

    return downloaded


def get_record_metadata(record_id: str) -> Optional[Dict]:
    """
    Get metadata for a HEPData record.

    NOTE: Makes network request.
    """
    import requests

    try:
        url = f"{HEPDATA_API_BASE}/record/ins{record_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get HEPData metadata: {e}")
        return None
