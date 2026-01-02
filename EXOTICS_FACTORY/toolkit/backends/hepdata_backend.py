"""HEPData backend interface (no network calls on import)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HEPDataSource:
    """Descriptor for a HEPData table reference."""

    record_id: str
    table_name: str
    doi: str | None = None


def build_hepdata_url(source: HEPDataSource) -> str:
    """Build a placeholder URL for a HEPData table."""
    base = "https://www.hepdata.net/record"
    return f"{base}/{source.record_id}?table={source.table_name}"


def fetch_table(source: HEPDataSource) -> dict[str, Any]:
    """Placeholder fetch that defines expected return type.

    This function intentionally does not perform any network calls. It should be
    implemented by the user in a run environment with network access enabled.
    """
    return {
        "source": source,
        "status": "NOT_FETCHED",
        "data": None,
        "note": "Network calls disabled in factory scaffolding.",
    }
