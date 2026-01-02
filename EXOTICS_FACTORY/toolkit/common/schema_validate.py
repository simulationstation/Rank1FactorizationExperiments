"""Spec schema validation for EXOTICS_FACTORY."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SchemaError(Exception):
    """Raised when a spec fails validation."""

    message: str

    def __str__(self) -> str:
        return self.message


def _require(mapping: dict[str, Any], key: str, expected_type: type) -> Any:
    if key not in mapping:
        raise SchemaError(f"Missing required key: {key}")
    value = mapping[key]
    if not isinstance(value, expected_type):
        raise SchemaError(f"Key '{key}' expected {expected_type.__name__}, got {type(value).__name__}")
    return value


def validate_spec_data(data: dict[str, Any]) -> None:
    _require(data, "schema_version", str)
    family = _require(data, "family", dict)
    sources = _require(data, "sources", list)
    channels = _require(data, "channels", list)
    model = _require(data, "model", dict)
    outputs = _require(data, "outputs", dict)
    runtime = _require(data, "runtime", dict)

    for key in (
        "id",
        "name",
        "category",
        "states",
        "channels",
        "preferred_backends",
        "model_class",
        "amplitude_level_requires",
        "proxy_only",
        "source_pointers",
    ):
        if key not in family:
            raise SchemaError(f"family.{key} is required")

    if not isinstance(family["states"], list) or not isinstance(family["channels"], list):
        raise SchemaError("family.states and family.channels must be lists")

    for entry in sources:
        if not isinstance(entry, dict):
            raise SchemaError("sources entries must be objects")
        for key in ("id", "backend", "description", "placeholders"):
            if key not in entry:
                raise SchemaError(f"sources entry missing {key}")

    for entry in channels:
        if not isinstance(entry, dict):
            raise SchemaError("channels entries must be objects")
        for key in ("id", "label", "source_ref"):
            if key not in entry:
                raise SchemaError(f"channels entry missing {key}")

    for key in ("type", "parameters", "shared_structure"):
        if key not in model:
            raise SchemaError(f"model.{key} is required")

    for key in ("output_dir", "report_template", "artifacts"):
        if key not in outputs:
            raise SchemaError(f"outputs.{key} is required")

    for key in ("dry_run_only", "steps", "backend_settings"):
        if key not in runtime:
            raise SchemaError(f"runtime.{key} is required")


def validate_spec_file(path: str | Path) -> dict[str, Any]:
    """Load and validate a spec.yaml file, returning parsed data."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SchemaError("Spec file must parse to a mapping")
    validate_spec_data(data)
    return data
