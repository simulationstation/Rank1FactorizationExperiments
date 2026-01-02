"""
Rank1 Discovery Mine - Automated exotic hadron data discovery and rank-1 testing.

This package provides tools for:
- Searching public data repositories (HEPData, CERN Open Data, arXiv, GitHub)
- Extracting numeric tables from PDFs and structured data
- Converting extracted data to rank-1 test harness format
- Running rank-1 factorization tests
- Generating reports and tracking progress

Usage:
    python -m rank1_discovery_mine plan          # Build plan + scaffold directories
    python -m rank1_discovery_mine validate      # Fast schema checks
    python -m rank1_discovery_mine status        # Show current progress
    python -m rank1_discovery_mine run --resume  # Resume processing (requires --execute)
    python -m rank1_discovery_mine run --one <slug>  # Process single candidate
"""

__version__ = "0.1.0"
__author__ = "Rank1 Factorization Experiments"

from .registry import Registry
from .candidates import CandidateLoader, validate_candidates
from .pipeline import Pipeline, PipelineStep, CandidateStatus

__all__ = [
    "Registry",
    "CandidateLoader",
    "validate_candidates",
    "Pipeline",
    "PipelineStep",
    "CandidateStatus",
]
