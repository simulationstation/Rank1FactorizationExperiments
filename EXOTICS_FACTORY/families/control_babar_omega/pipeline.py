"""Pipeline definition for this family (no execution on import)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from EXOTICS_FACTORY.toolkit.common.logging import configure_logger


LOGGER = configure_logger(__name__)


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load a spec YAML file."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_pipeline(spec: dict[str, Any]) -> list[str]:
    """Return the ordered list of steps for the pipeline."""
    runtime = spec.get("runtime", {})
    steps = runtime.get("steps", [])
    return steps


def run_pipeline(spec: dict[str, Any], mode: str = "DRY_RUN") -> None:
    """Run pipeline in DRY_RUN mode by default."""
    steps = build_pipeline(spec)
    if mode != "RUN":
        LOGGER.info("DRY_RUN: pipeline will not execute.")
        LOGGER.info("Planned steps: %s", json.dumps(steps, indent=2))
        return
    raise RuntimeError("Execution disabled in factory scaffolding. Use DRY_RUN only.")
