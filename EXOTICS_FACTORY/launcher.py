"""Master launcher for EXOTICS_FACTORY pipelines (dry-run by default)."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import yaml

from EXOTICS_FACTORY.toolkit.common.logging import configure_logger


LOGGER = configure_logger(__name__)


def load_registry(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def find_family(registry: dict[str, Any], family_id: str) -> dict[str, Any]:
    for entry in registry.get("families", []):
        if entry.get("id") == family_id:
            return entry
    raise KeyError(f"Family '{family_id}' not found in registry")


def load_spec(family_id: str) -> dict[str, Any]:
    spec_path = Path("EXOTICS_FACTORY/families") / family_id / "spec.yaml"
    return yaml.safe_load(spec_path.read_text(encoding="utf-8"))


def run_pipeline(family_id: str, mode: str) -> None:
    module_path = f"EXOTICS_FACTORY.families.{family_id}.pipeline"
    pipeline = importlib.import_module(module_path)
    spec = load_spec(family_id)
    pipeline.run_pipeline(spec, mode=mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXOTICS_FACTORY pipeline launcher")
    parser.add_argument("--family", required=True, help="Family id from registry")
    parser.add_argument("--run", action="store_true", help="Execute pipeline steps")
    args = parser.parse_args()

    registry = load_registry("EXOTICS_FACTORY/registry/families.yaml")
    entry = find_family(registry, args.family)
    LOGGER.info("Selected family: %s", json.dumps(entry, indent=2))

    mode = "RUN" if args.run else "DRY_RUN"
    run_pipeline(args.family, mode)


if __name__ == "__main__":
    main()
