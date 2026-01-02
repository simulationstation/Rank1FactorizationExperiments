"""
Source modules for data discovery.

Each module provides:
- plan_queries(candidate): Returns list of query strings to execute
- execute_search(candidate): Actually runs queries, returns list of URLs
- download(url, dest_dir): Downloads data to destination directory
"""

from . import hepdata
from . import arxiv
from . import cern_opendata
from . import github_search
from . import inspire

__all__ = ["hepdata", "arxiv", "cern_opendata", "github_search", "inspire"]
