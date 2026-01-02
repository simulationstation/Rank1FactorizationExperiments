"""
PDF table extraction module.

Extracts numeric tables from PDF files using multiple backends:
1. pdfplumber (primary)
2. tabula-py (fallback)
3. camelot (fallback)

Tables are validated for numeric content relevant to HEP analyses.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def extract_tables(
    pdf_path: Path,
    pages: Optional[List[int]] = None,
    min_rows: int = 3,
    min_cols: int = 2,
) -> List[pd.DataFrame]:
    """
    Extract all tables from a PDF file.

    Args:
        pdf_path: Path to PDF file
        pages: List of page numbers to process (1-indexed), or None for all
        min_rows: Minimum rows for a valid table
        min_cols: Minimum columns for a valid table

    Returns:
        List of DataFrames, each representing an extracted table
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return []

    tables = []

    # Try pdfplumber first (best for simple tables)
    try:
        tables = _extract_with_pdfplumber(pdf_path, pages, min_rows, min_cols)
        if tables:
            logger.info(f"Extracted {len(tables)} tables with pdfplumber")
            return tables
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")

    # Try tabula as fallback
    try:
        tables = _extract_with_tabula(pdf_path, pages, min_rows, min_cols)
        if tables:
            logger.info(f"Extracted {len(tables)} tables with tabula")
            return tables
    except Exception as e:
        logger.warning(f"tabula extraction failed: {e}")

    logger.warning(f"No tables extracted from {pdf_path}")
    return []


def _extract_with_pdfplumber(
    pdf_path: Path,
    pages: Optional[List[int]],
    min_rows: int,
    min_cols: int,
) -> List[pd.DataFrame]:
    """Extract tables using pdfplumber."""
    import pdfplumber

    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        page_nums = pages if pages else range(len(pdf.pages))

        for page_num in page_nums:
            if isinstance(page_num, int) and page_num > 0:
                page_idx = page_num - 1  # Convert to 0-indexed
            else:
                page_idx = page_num

            if page_idx >= len(pdf.pages):
                continue

            page = pdf.pages[page_idx]
            page_tables = page.extract_tables()

            for table_data in page_tables:
                if not table_data:
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(table_data[1:], columns=table_data[0])

                # Validate
                if len(df) >= min_rows and len(df.columns) >= min_cols:
                    # Clean up
                    df = _clean_table(df)
                    if _is_numeric_table(df):
                        df.attrs["source_page"] = page_idx + 1
                        df.attrs["extraction_method"] = "pdfplumber"
                        tables.append(df)

    return tables


def _extract_with_tabula(
    pdf_path: Path,
    pages: Optional[List[int]],
    min_rows: int,
    min_cols: int,
) -> List[pd.DataFrame]:
    """Extract tables using tabula-py."""
    try:
        import tabula
    except ImportError:
        logger.warning("tabula-py not installed")
        return []

    tables = []

    # Convert pages to tabula format
    if pages:
        pages_str = ",".join(str(p) for p in pages)
    else:
        pages_str = "all"

    try:
        raw_tables = tabula.read_pdf(
            str(pdf_path),
            pages=pages_str,
            multiple_tables=True,
            silent=True,
        )

        for df in raw_tables:
            if len(df) >= min_rows and len(df.columns) >= min_cols:
                df = _clean_table(df)
                if _is_numeric_table(df):
                    df.attrs["extraction_method"] = "tabula"
                    tables.append(df)

    except Exception as e:
        logger.warning(f"tabula extraction error: {e}")

    return tables


def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up extracted table."""
    # Remove empty rows/columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')

    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Try to convert numeric columns
    for col in df.columns:
        try:
            # Remove common non-numeric characters
            cleaned = df[col].astype(str).str.replace(r'[^\d.\-+eE]', '', regex=True)
            df[col] = pd.to_numeric(cleaned, errors='coerce')
        except Exception:
            pass

    return df


def _is_numeric_table(df: pd.DataFrame, min_numeric_frac: float = 0.5) -> bool:
    """
    Check if table has sufficient numeric content.

    Returns True if at least min_numeric_frac of cells are numeric.
    """
    total_cells = df.size
    if total_cells == 0:
        return False

    numeric_cells = 0
    for col in df.columns:
        numeric_cells += df[col].apply(lambda x: isinstance(x, (int, float)) and pd.notna(x)).sum()

    return (numeric_cells / total_cells) >= min_numeric_frac


def identify_table_type(df: pd.DataFrame) -> str:
    """
    Identify the type of HEP table based on column names.

    Returns one of:
    - "mass_spectrum": Mass bins with counts/events
    - "cross_section": Energy vs cross section
    - "branching_ratio": Decay branching ratios
    - "fit_parameters": Fitted parameter values
    - "unknown": Cannot determine type
    """
    cols_lower = [str(c).lower() for c in df.columns]
    col_text = " ".join(cols_lower)

    # Check for mass spectrum
    mass_keywords = ["mass", "m(", "mev", "gev", "bin", "invariant"]
    count_keywords = ["count", "event", "yield", "n(", "dn/dm", "entries"]

    if any(k in col_text for k in mass_keywords) and any(k in col_text for k in count_keywords):
        return "mass_spectrum"

    # Check for cross section
    xsec_keywords = ["sigma", "cross", "section", "pb", "fb", "nb"]
    energy_keywords = ["sqrt", "energy", "ecm", "s^", "gev"]

    if any(k in col_text for k in xsec_keywords) or (
        any(k in col_text for k in energy_keywords) and "sigma" in col_text
    ):
        return "cross_section"

    # Check for branching ratio
    br_keywords = ["branch", "ratio", "bf(", "br(", "fraction"]
    if any(k in col_text for k in br_keywords):
        return "branching_ratio"

    # Check for fit parameters
    fit_keywords = ["param", "fit", "mass", "width", "gamma", "amplitude"]
    if any(k in col_text for k in fit_keywords) and len(df) < 20:
        return "fit_parameters"

    return "unknown"


def extract_mass_spectrum(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Extract standardized mass spectrum from table.

    Returns DataFrame with columns: mass_GeV, counts, stat_err
    Or None if extraction fails.
    """
    cols = df.columns.tolist()
    cols_lower = [str(c).lower() for c in cols]

    # Find mass column
    mass_col = None
    for i, c in enumerate(cols_lower):
        if any(k in c for k in ["mass", "m(", "mev", "gev"]):
            mass_col = cols[i]
            break

    # Find count column
    count_col = None
    for i, c in enumerate(cols_lower):
        if any(k in c for k in ["count", "event", "yield", "n(", "dn/dm", "entries"]):
            count_col = cols[i]
            break

    # Find error column
    err_col = None
    for i, c in enumerate(cols_lower):
        if any(k in c for k in ["err", "unc", "sigma", "stat"]):
            err_col = cols[i]
            break

    if mass_col is None or count_col is None:
        return None

    result = pd.DataFrame()
    result["mass_GeV"] = pd.to_numeric(df[mass_col], errors='coerce')
    result["counts"] = pd.to_numeric(df[count_col], errors='coerce')

    if err_col:
        result["stat_err"] = pd.to_numeric(df[err_col], errors='coerce')
    else:
        # Estimate Poisson error
        result["stat_err"] = result["counts"].apply(lambda x: max(1, x**0.5) if pd.notna(x) else 1)

    # Check if mass is in MeV and convert
    if result["mass_GeV"].median() > 100:  # Likely MeV
        result["mass_GeV"] = result["mass_GeV"] / 1000

    return result.dropna()
