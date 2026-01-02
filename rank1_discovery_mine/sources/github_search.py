"""
GitHub source module.

Searches GitHub for analysis code, data tables, and workspaces
that may contain useful numeric data for rank-1 tests.

API Documentation: https://docs.github.com/en/rest/search
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
GITHUB_SEARCH_CODE = f"{GITHUB_API_BASE}/search/code"
GITHUB_SEARCH_REPOS = f"{GITHUB_API_BASE}/search/repositories"


# Common HEP analysis repositories to search
KNOWN_HEP_REPOS = [
    "cms-sw/cmssw",
    "lhcb/lhcb-analysis",
    "atlas/athena",
]

# File patterns to search for
DATA_FILE_PATTERNS = [
    "*.csv",
    "*.json",
    "*.yaml",
    "*.root",
    "*.txt",
]


def plan_queries(candidate: "Candidate") -> List[str]:
    """
    Plan GitHub search queries for a candidate.

    Returns list of query URLs that would be executed.
    """
    queries = []

    # Search for code containing state names
    for state in candidate.states:
        encoded = quote_plus(f"{state} extension:csv extension:json")
        queries.append(f"{GITHUB_SEARCH_CODE}?q={encoded}&per_page=10")

    # Search repositories by topic
    for term in candidate.search_terms[:3]:  # Limit queries
        encoded = quote_plus(term)
        queries.append(f"{GITHUB_SEARCH_REPOS}?q={encoded}+topic:physics&per_page=5")

    # Search specific known repos if collaboration matches
    if candidate.collaboration:
        collab_lower = candidate.collaboration.lower()
        for repo in KNOWN_HEP_REPOS:
            if collab_lower in repo.lower():
                for state in candidate.states[:2]:
                    encoded = quote_plus(f"{state} repo:{repo}")
                    queries.append(f"{GITHUB_SEARCH_CODE}?q={encoded}&per_page=5")

    return queries


def execute_search(candidate: "Candidate") -> List[Dict[str, Any]]:
    """
    Execute GitHub searches and return discovered URLs.

    NOTE: This actually makes network requests. Only call when --execute is set.
    NOTE: GitHub API has rate limits. Handle 403 responses gracefully.

    Returns list of dicts with keys: url, type, repo, path, description
    """
    import requests

    discovered = []
    queries = plan_queries(candidate)

    headers = {
        "Accept": "application/vnd.github.v3+json",
        # Add token here if rate-limited: "Authorization": "token YOUR_TOKEN"
    }

    for query_url in queries:
        try:
            logger.info(f"GitHub query: {query_url}")
            response = requests.get(query_url, headers=headers, timeout=30)

            # Handle rate limiting
            if response.status_code == 403:
                logger.warning("GitHub rate limit reached")
                break

            response.raise_for_status()
            data = response.json()

            # Handle code search results
            if "items" in data:
                for item in data["items"]:
                    discovered.append({
                        "url": item.get("html_url", ""),
                        "type": "github_code",
                        "repo": item.get("repository", {}).get("full_name", ""),
                        "path": item.get("path", ""),
                        "name": item.get("name", ""),
                        "raw_url": item.get("html_url", "").replace(
                            "github.com", "raw.githubusercontent.com"
                        ).replace("/blob/", "/"),
                    })

        except Exception as e:
            logger.warning(f"GitHub query failed: {e}")

    # Deduplicate by URL
    seen = set()
    unique = []
    for item in discovered:
        url = item.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(item)

    return unique


def download(url: str, dest_dir: Path, timeout: int = 60) -> List[str]:
    """
    Download file from GitHub to destination directory.

    Returns list of downloaded file paths.
    """
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    try:
        # Convert to raw URL if needed
        raw_url = url
        if "github.com" in url and "/blob/" in url:
            raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

        logger.info(f"Downloading: {raw_url}")
        response = requests.get(raw_url, timeout=timeout)
        response.raise_for_status()

        # Extract filename
        filename = raw_url.split("/")[-1]
        if not filename:
            filename = "github_download"

        # Sanitize filename
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        filepath = dest_dir / f"github_{filename}"

        with open(filepath, 'wb') as f:
            f.write(response.content)
        downloaded.append(str(filepath))

    except Exception as e:
        logger.error(f"GitHub download failed: {e}")

    return downloaded
