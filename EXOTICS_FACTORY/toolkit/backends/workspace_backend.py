"""Workspace/datacard ingestion interface (placeholder)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkspaceSpec:
    """Descriptor for a RooWorkspace or Combine workspace ingestion."""

    workspace_path: str
    snapshot: str
    parameter_map: dict[str, str]
    datacard_path: str | None = None


def describe_workspace(spec: WorkspaceSpec) -> dict[str, str]:
    """Describe required workspace inputs."""
    return {
        "workspace_path": spec.workspace_path,
        "snapshot": spec.snapshot,
        "datacard_path": spec.datacard_path or "",
    }
