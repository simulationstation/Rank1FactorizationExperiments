"""Report writer utilities."""
from __future__ import annotations

from pathlib import Path

from ..common.io import write_text


def build_report_skeleton(output_path: str | Path, family_name: str) -> Path:
    """Write a report skeleton with standard sections."""
    content = f"""# Report: {family_name}

## Provenance
- Sources:
- Dataset versions:

## Extraction Proofs
- Overlay images:
- Vector/raster QA notes:

## Model
- Model class:
- Parameter sharing:

## Inference
- Rank-1 test summary:
- Proxy limitations:

## Outputs
- Artifacts:
- Fit outputs:
"""
    return write_text(output_path, content)
