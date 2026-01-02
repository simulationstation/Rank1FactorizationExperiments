"""
Adapter module for converting extracted data to rank-1 harness CSV format.

The rank-1 harness expects CSV files with columns:
    mass_GeV, counts, stat_err

Or for cross-section data:
    sqrt_s_GeV, sigma_pb, stat_err, sys_err (optional)

This module provides converters for various input formats.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


# Standard output column names
MASS_SPECTRUM_COLS = ["mass_GeV", "counts", "stat_err"]
XSEC_COLS = ["sqrt_s_GeV", "sigma_pb", "stat_err", "sys_err"]


def convert_hepdata_json(
    json_path: Path,
    output_dir: Path,
    table_filter: Optional[str] = None,
) -> List[str]:
    """
    Convert HEPData JSON to rank-1 input CSVs.

    Args:
        json_path: Path to HEPData JSON file
        output_dir: Directory to write output CSVs
        table_filter: Optional regex to filter table names

    Returns:
        List of paths to created CSV files
    """
    import re

    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    created_files = []

    # Handle HEPData record format
    if "record" in data:
        tables = data["record"].get("tables", [])
    elif "tables" in data:
        tables = data["tables"]
    else:
        # Single table format
        tables = [data]

    for i, table in enumerate(tables):
        table_name = table.get("name", f"table_{i}")

        # Apply filter if specified
        if table_filter and not re.search(table_filter, table_name, re.IGNORECASE):
            continue

        # Extract data
        df = _parse_hepdata_table(table)
        if df is None or df.empty:
            continue

        # Determine output filename
        safe_name = re.sub(r'[^\w\-_]', '_', table_name)
        output_path = output_dir / f"{safe_name}.csv"

        df.to_csv(output_path, index=False)
        created_files.append(str(output_path))
        logger.info(f"Created: {output_path}")

    return created_files


def _parse_hepdata_table(table: Dict) -> Optional[pd.DataFrame]:
    """
    Parse a HEPData table structure to DataFrame.

    HEPData tables have:
    - independent_variables: list of {header, values}
    - dependent_variables: list of {header, values, qualifiers}
    """
    ind_vars = table.get("independent_variables", [])
    dep_vars = table.get("dependent_variables", [])

    if not ind_vars or not dep_vars:
        # Try alternative format
        if "data" in table:
            return pd.DataFrame(table["data"])
        return None

    # Build DataFrame
    data = {}

    # Process independent variables (usually x-axis)
    for var in ind_vars:
        header = var.get("header", {}).get("name", "x")
        values = var.get("values", [])

        # Extract values (may have low/high for bins)
        extracted = []
        for v in values:
            if isinstance(v, dict):
                if "value" in v:
                    extracted.append(v["value"])
                elif "low" in v and "high" in v:
                    # Use bin center
                    extracted.append((v["low"] + v["high"]) / 2)
            else:
                extracted.append(v)

        data[_normalize_column_name(header)] = extracted

    # Process dependent variables (usually y-axis)
    for var in dep_vars:
        header = var.get("header", {}).get("name", "y")
        values = var.get("values", [])

        # Extract values and errors
        y_values = []
        y_err_up = []
        y_err_down = []

        for v in values:
            if isinstance(v, dict):
                y_values.append(v.get("value", 0))

                # Handle errors
                errors = v.get("errors", [])
                err_up = 0
                err_down = 0
                for err in errors:
                    if isinstance(err, dict):
                        if "asymerror" in err:
                            err_up = abs(err["asymerror"].get("plus", 0))
                            err_down = abs(err["asymerror"].get("minus", 0))
                        elif "symerror" in err:
                            err_up = err_down = abs(err["symerror"])
                y_err_up.append(err_up)
                y_err_down.append(err_down)
            else:
                y_values.append(v)
                y_err_up.append(0)
                y_err_down.append(0)

        col_name = _normalize_column_name(header)
        data[col_name] = y_values
        data[f"{col_name}_err_up"] = y_err_up
        data[f"{col_name}_err_down"] = y_err_down

    return pd.DataFrame(data)


def _normalize_column_name(name: str) -> str:
    """Normalize column name to standard format."""
    name = name.lower().strip()

    # Common normalizations
    replacements = {
        "m(": "mass_",
        "sqrt(s)": "sqrt_s",
        "sigma": "sigma",
        "cross section": "sigma",
        "events": "counts",
        "entries": "counts",
        "yield": "counts",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    # Remove units in parentheses
    import re
    name = re.sub(r'\s*\([^)]*\)\s*', '', name)

    # Replace spaces and special chars with underscore
    name = re.sub(r'[^\w]', '_', name)
    name = re.sub(r'_+', '_', name)

    return name.strip('_')


def convert_to_mass_spectrum(
    df: pd.DataFrame,
    mass_col: str,
    count_col: str,
    err_col: Optional[str] = None,
    mass_unit: str = "GeV",
) -> pd.DataFrame:
    """
    Convert DataFrame to standard mass spectrum format.

    Args:
        df: Input DataFrame
        mass_col: Name of mass column
        count_col: Name of count/yield column
        err_col: Name of error column (optional, will use sqrt(N) if None)
        mass_unit: Unit of mass column ("GeV" or "MeV")

    Returns:
        DataFrame with columns: mass_GeV, counts, stat_err
    """
    result = pd.DataFrame()

    # Extract mass
    mass = pd.to_numeric(df[mass_col], errors='coerce')
    if mass_unit.lower() == "mev":
        mass = mass / 1000
    result["mass_GeV"] = mass

    # Extract counts
    result["counts"] = pd.to_numeric(df[count_col], errors='coerce')

    # Extract or compute error
    if err_col and err_col in df.columns:
        result["stat_err"] = pd.to_numeric(df[err_col], errors='coerce')
    else:
        # Poisson error
        result["stat_err"] = result["counts"].apply(
            lambda x: max(1.0, x**0.5) if pd.notna(x) and x >= 0 else 1.0
        )

    return result.dropna()


def convert_to_cross_section(
    df: pd.DataFrame,
    energy_col: str,
    sigma_col: str,
    stat_err_col: Optional[str] = None,
    sys_err_col: Optional[str] = None,
    energy_unit: str = "GeV",
    sigma_unit: str = "pb",
) -> pd.DataFrame:
    """
    Convert DataFrame to standard cross-section format.

    Args:
        df: Input DataFrame
        energy_col: Name of energy column
        sigma_col: Name of cross-section column
        stat_err_col: Name of statistical error column
        sys_err_col: Name of systematic error column
        energy_unit: Unit of energy ("GeV" or "MeV")
        sigma_unit: Unit of cross-section ("pb", "fb", "nb")

    Returns:
        DataFrame with columns: sqrt_s_GeV, sigma_pb, stat_err, sys_err
    """
    result = pd.DataFrame()

    # Extract energy
    energy = pd.to_numeric(df[energy_col], errors='coerce')
    if energy_unit.lower() == "mev":
        energy = energy / 1000
    result["sqrt_s_GeV"] = energy

    # Extract cross-section
    sigma = pd.to_numeric(df[sigma_col], errors='coerce')
    # Convert to pb if needed
    if sigma_unit.lower() == "fb":
        sigma = sigma / 1000
    elif sigma_unit.lower() == "nb":
        sigma = sigma * 1000
    result["sigma_pb"] = sigma

    # Extract errors
    if stat_err_col and stat_err_col in df.columns:
        result["stat_err"] = pd.to_numeric(df[stat_err_col], errors='coerce')
    else:
        result["stat_err"] = result["sigma_pb"] * 0.1  # Default 10%

    if sys_err_col and sys_err_col in df.columns:
        result["sys_err"] = pd.to_numeric(df[sys_err_col], errors='coerce')
    else:
        result["sys_err"] = 0.0

    return result.dropna(subset=["sqrt_s_GeV", "sigma_pb"])


def validate_rank1_input(csv_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate that a CSV file is suitable for rank-1 testing.

    Returns (is_valid, list_of_issues).
    """
    issues = []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, [f"Cannot read CSV: {e}"]

    # Check for required columns
    has_mass = any(c.lower().startswith("mass") for c in df.columns)
    has_energy = any("sqrt_s" in c.lower() or "energy" in c.lower() for c in df.columns)
    has_counts = any(c.lower() in ["counts", "events", "yield", "entries"] for c in df.columns)
    has_sigma = any("sigma" in c.lower() for c in df.columns)

    if not (has_mass or has_energy):
        issues.append("Missing mass/energy column")

    if not (has_counts or has_sigma):
        issues.append("Missing counts/sigma column")

    # Check for minimum rows
    if len(df) < 5:
        issues.append(f"Too few data points: {len(df)} (need >= 5)")

    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        issues.append(f"Contains {nan_count} NaN values")

    # Check for negative counts
    for col in df.columns:
        if "count" in col.lower() or "event" in col.lower():
            if (df[col] < 0).any():
                issues.append(f"Negative values in {col}")

    return len(issues) == 0, issues


def auto_detect_and_convert(
    input_path: Path,
    output_dir: Path,
) -> List[str]:
    """
    Automatically detect input format and convert to rank-1 input.

    Handles:
    - HEPData JSON
    - Generic CSV
    - PDF (via extraction)

    Returns list of created output CSV paths.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()

    if suffix == ".json":
        return convert_hepdata_json(input_path, output_dir)

    elif suffix == ".csv":
        # Try to auto-convert CSV
        df = pd.read_csv(input_path)

        # Detect type
        cols_lower = [c.lower() for c in df.columns]

        if any("mass" in c for c in cols_lower):
            # Mass spectrum
            mass_col = next(c for c in df.columns if "mass" in c.lower())
            count_col = next(
                (c for c in df.columns if any(k in c.lower() for k in ["count", "event", "yield"])),
                df.columns[1]  # Fallback to second column
            )
            err_col = next(
                (c for c in df.columns if "err" in c.lower()),
                None
            )
            result = convert_to_mass_spectrum(df, mass_col, count_col, err_col)

        elif any("sqrt_s" in c or "energy" in c for c in cols_lower):
            # Cross-section
            energy_col = next(c for c in df.columns if "sqrt_s" in c.lower() or "energy" in c.lower())
            sigma_col = next(
                (c for c in df.columns if "sigma" in c.lower()),
                df.columns[1]
            )
            result = convert_to_cross_section(df, energy_col, sigma_col)

        else:
            # Unknown format, just copy
            result = df

        output_path = output_dir / f"converted_{input_path.stem}.csv"
        result.to_csv(output_path, index=False)
        return [str(output_path)]

    else:
        logger.warning(f"Unknown input format: {suffix}")
        return []
