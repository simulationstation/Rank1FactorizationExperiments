"""PDF extraction backend interface (vector-first)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PdfFigureSpec:
    """Specification for a figure to extract."""

    pdf_path: str
    page: int
    figure_id: str
    vector_first: bool = True


def extract_vector_paths(spec: PdfFigureSpec) -> list[dict[str, Any]]:
    """Placeholder for vector extraction using PyMuPDF (fitz).

    Expected output: list of path dictionaries containing coordinates and style.
    """
    return []


def extract_raster_segments(spec: PdfFigureSpec) -> list[dict[str, Any]]:
    """Placeholder for raster fallback extraction.

    Expected output: list of segmented curve points.
    """
    return []


def build_overlay_proof(
    spec: PdfFigureSpec,
    extracted_paths: list[dict[str, Any]],
    output_path: str,
) -> str:
    """Define overlay proof output path for QA."""
    return output_path
