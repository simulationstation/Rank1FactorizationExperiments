"""
Run rank-1 test on discovered HEPData tables.

Identifies suitable table pairs and runs the rank-1 factorization test.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re

from .rank1_core import load_hepdata_csv, select_window, run_pair_test
from .state_configs import get_state_config, StateConfig

logger = logging.getLogger(__name__)


@dataclass
class TableInfo:
    """Information about a HEPData table."""
    path: Path
    name: str
    description: str
    inspire_id: str
    n_rows: int
    has_mass_column: bool
    has_counts: bool


def parse_hepdata_header(filepath: Path) -> Dict[str, str]:
    """Parse HEPData CSV header comments."""
    info = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#:'):
                parts = line[2:].strip().split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    info[key] = val
            elif not line.startswith('#') and not line.startswith('$'):
                break
    return info


def analyze_table(filepath: Path) -> Optional[TableInfo]:
    """Analyze a HEPData CSV table."""
    try:
        header = parse_hepdata_header(filepath)
        name = header.get('name', filepath.stem)
        desc = header.get('description', '')

        # Check column structure
        data_lines = []
        columns = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                if line.startswith('$'):
                    # Column header line
                    columns = [c.strip() for c in line.strip().split(',')]
                else:
                    data_lines.append(line)

        # Check for mass column
        cols_lower = [c.lower() for c in columns]
        has_mass = any('mass' in c or 'm(' in c or 'mev' in c.lower() or 'gev' in c.lower()
                       for c in cols_lower)

        # Check for counts/events
        has_counts = any('dn' in c or 'event' in c or 'count' in c or 'yield' in c
                         for c in cols_lower)

        # Extract INSPIRE ID from filename
        inspire_match = re.search(r'ins(\d+)', filepath.name)
        inspire_id = inspire_match.group(1) if inspire_match else ''

        return TableInfo(
            path=filepath,
            name=name,
            description=desc,
            inspire_id=inspire_id,
            n_rows=len(data_lines),
            has_mass_column=has_mass,
            has_counts=has_counts,
        )
    except Exception as e:
        logger.warning(f"Failed to analyze {filepath}: {e}")
        return None


def get_mass_range(filepath: Path) -> Optional[Tuple[float, float]]:
    """Extract the mass range from first column of CSV data."""
    try:
        with open(filepath, 'r') as f:
            data_started = False
            first_val = None
            last_val = None
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                if line.startswith('$'):
                    data_started = True
                    continue
                if data_started:
                    parts = line.strip().split(',')
                    if parts:
                        try:
                            val = float(parts[0])
                            if first_val is None:
                                first_val = val
                            last_val = val
                        except ValueError:
                            continue
            if first_val is not None and last_val is not None:
                return (min(first_val, last_val), max(first_val, last_val))
    except Exception:
        pass
    return None


def ranges_overlap(r1: Tuple[float, float], r2: Tuple[float, float],
                   fit_window: Tuple[float, float]) -> bool:
    """Check if two mass ranges both overlap with fit window."""
    # Convert to MeV if needed (detect GeV by checking if range is < 100)
    def to_mev(r):
        if r[1] < 100:  # Likely in GeV
            return (r[0] * 1000, r[1] * 1000)
        return r

    r1_mev = to_mev(r1)
    r2_mev = to_mev(r2)

    # Check if both ranges overlap with fit window
    def overlaps_window(r):
        return r[0] <= fit_window[1] and r[1] >= fit_window[0]

    return overlaps_window(r1_mev) and overlaps_window(r2_mev)


def find_suitable_pairs(raw_dir: Path, config: StateConfig) -> List[Tuple[TableInfo, TableInfo]]:
    """
    Find suitable table pairs for rank-1 test.

    Prioritizes tables from records that match the state configuration
    (by checking for state names in description) AND have compatible mass ranges.

    Returns list of (table_A, table_B) pairs.
    """
    # Find all CSV files
    csv_files = list(raw_dir.glob("hepdata_*.csv"))
    if not csv_files:
        logger.warning(f"No HEPData CSV files found in {raw_dir}")
        return []

    # State names to look for (lowercase)
    state_keywords = [
        config.state1_name.lower(),
        config.state2_name.lower(),
    ]
    # Also check for common patterns
    state_keywords.extend([
        config.state1_name.lower().replace('(', '').replace(')', ''),
        config.state2_name.lower().replace('(', '').replace(')', ''),
    ])

    # Analyze each table
    tables = []
    relevant_tables = []
    mass_ranges = {}  # path -> (min, max) in original units

    for f in csv_files:
        info = analyze_table(f)
        if info and info.n_rows >= 5:
            tables.append(info)

            # Get mass range for this table
            mass_range = get_mass_range(f)
            if mass_range:
                mass_ranges[f] = mass_range

            # Check if this table is relevant to our exotic states
            desc_lower = info.description.lower()
            name_lower = info.name.lower()

            is_relevant = any(kw in desc_lower or kw in name_lower for kw in state_keywords)

            # Also check for generic mass spectrum keywords
            mass_keywords = ['invariant mass', 'm(', 'mass spectrum', 'mass distribution']
            has_mass_content = any(kw in desc_lower for kw in mass_keywords)

            if is_relevant or (info.has_mass_column and has_mass_content):
                relevant_tables.append(info)

    logger.info(f"Found {len(tables)} tables, {len(relevant_tables)} potentially relevant")

    # Prefer relevant tables, fallback to all tables with mass columns
    target_tables = relevant_tables if len(relevant_tables) >= 2 else [t for t in tables if t.has_mass_column]

    if len(target_tables) < 2:
        logger.warning(f"Need at least 2 suitable tables, found {len(target_tables)}")
        return []

    # Filter to tables that overlap with fit window
    fit_window = config.fit_window
    compatible_tables = []
    for t in target_tables:
        if t.path in mass_ranges:
            r = mass_ranges[t.path]
            # Convert to MeV if in GeV
            r_mev = (r[0] * 1000, r[1] * 1000) if r[1] < 100 else r
            # Check if range overlaps with fit window
            if r_mev[0] <= fit_window[1] and r_mev[1] >= fit_window[0]:
                compatible_tables.append(t)
                logger.info(f"  Compatible: {t.name} (range {r_mev[0]:.0f}-{r_mev[1]:.0f} MeV)")

    if len(compatible_tables) < 2:
        logger.warning(f"Need at least 2 tables overlapping fit window {fit_window}, found {len(compatible_tables)}")
        # Fall back to original target_tables if filtering was too aggressive
        compatible_tables = target_tables

    # Group by INSPIRE ID
    by_inspire = {}
    for t in compatible_tables:
        if t.inspire_id:
            if t.inspire_id not in by_inspire:
                by_inspire[t.inspire_id] = []
            by_inspire[t.inspire_id].append(t)

    # Generate pairs - prioritize tables with compatible mass ranges
    pairs = []

    # Within same record: pair tables with overlapping mass ranges
    for inspire_id, record_tables in by_inspire.items():
        if len(record_tables) >= 2:
            # Sort by name/number
            record_tables.sort(key=lambda t: t.name)
            # Pair tables that have compatible mass ranges
            for i in range(len(record_tables)):
                for j in range(i + 1, len(record_tables)):
                    t1, t2 = record_tables[i], record_tables[j]
                    r1 = mass_ranges.get(t1.path)
                    r2 = mass_ranges.get(t2.path)
                    if r1 and r2 and ranges_overlap(r1, r2, fit_window):
                        pairs.append((t1, t2))
                        logger.info(f"  Paired: {t1.name} + {t2.name} (ins{inspire_id})")
                        if len(pairs) >= 5:  # Limit pairs per record
                            break
                if len(pairs) >= 5:
                    break

    # Only do cross-record if we have no within-record pairs
    if not pairs and len(by_inspire) >= 2:
        all_inspire_ids = list(by_inspire.keys())
        for i in range(min(len(all_inspire_ids) - 1, 3)):
            t1 = by_inspire[all_inspire_ids[i]][0]
            t2 = by_inspire[all_inspire_ids[i+1]][0]
            pairs.append((t1, t2))

    logger.info(f"Generated {len(pairs)} table pairs")
    return pairs


def run_rank1_test(candidate_dir: Path, slug: str, n_boot: int = 100) -> Dict[str, Any]:
    """
    Run rank-1 test for a candidate.

    Args:
        candidate_dir: Path to candidate directory
        slug: Candidate slug
        n_boot: Number of bootstrap replicates

    Returns:
        Dict with test results
    """
    # Prefer extracted/ directory (processed CSVs), fallback to raw/
    extracted_dir = candidate_dir / "extracted"
    raw_dir = candidate_dir / "raw"

    # Check both directories for CSV files
    extracted_csvs = list(extracted_dir.glob("*.csv")) if extracted_dir.exists() else []
    raw_csvs = list(raw_dir.glob("hepdata_*.csv")) if raw_dir.exists() else []

    # Use extracted if it has files, otherwise raw
    if extracted_csvs:
        data_dir = extracted_dir
        logger.info(f"Using extracted/ directory ({len(extracted_csvs)} CSVs)")
    elif raw_csvs:
        data_dir = raw_dir
        logger.info(f"Using raw/ directory ({len(raw_csvs)} CSVs)")
    else:
        data_dir = raw_dir
        logger.warning(f"No CSV files found in extracted/ or raw/")

    out_dir = candidate_dir / "out"
    out_dir.mkdir(exist_ok=True)

    # Get state configuration
    config = get_state_config(slug)
    if config is None:
        logger.error(f"No state configuration for {slug}")
        return {
            "status": "ERROR",
            "error": f"No state configuration for {slug}",
            "slug": slug,
        }

    logger.info(f"Running rank-1 test for {slug}")
    logger.info(f"States: {config.state1_name} vs {config.state2_name}")
    logger.info(f"Fit window: {config.fit_window[0]}-{config.fit_window[1]} MeV")

    # Find suitable table pairs
    pairs = find_suitable_pairs(data_dir, config)

    if not pairs:
        return {
            "status": "NO_SUITABLE_DATA",
            "error": "No suitable table pairs found",
            "slug": slug,
            "states": [config.state1_name, config.state2_name],
        }

    # Run tests on each pair
    results = {
        "status": "COMPLETED",
        "slug": slug,
        "states": [config.state1_name, config.state2_name],
        "fit_window": list(config.fit_window),
        "n_boot": n_boot,
        "pairs": [],
    }

    for i, (table_A, table_B) in enumerate(pairs[:3]):  # Limit to 3 pairs
        pair_name = f"Pair {i+1}: {table_A.name} vs {table_B.name}"
        logger.info(f"Testing {pair_name}")

        # Load data
        data_A = load_hepdata_csv(str(table_A.path))
        data_B = load_hepdata_csv(str(table_B.path))

        if len(data_A) == 0 or len(data_B) == 0:
            logger.warning(f"Empty data for {pair_name}")
            continue

        # Apply fit window
        data_A = select_window(data_A, config.fit_window[0], config.fit_window[1])
        data_B = select_window(data_B, config.fit_window[0], config.fit_window[1])

        if len(data_A) < 5 or len(data_B) < 5:
            logger.warning(f"Too few points in window for {pair_name}")
            continue

        # Run test
        pair_result = run_pair_test(
            data_A, data_B, pair_name,
            config.state1_mass, config.state1_width,
            config.state2_mass, config.state2_width,
            config.fit_window,
            n_boot=n_boot,
        )

        pair_result['table_A'] = str(table_A.path.name)
        pair_result['table_B'] = str(table_B.path.name)
        results['pairs'].append(pair_result)

        logger.info(f"  Verdict: {pair_result['verdict']}")

    # Determine overall verdict
    if not results['pairs']:
        results['status'] = "NO_VALID_PAIRS"
        results['overall_verdict'] = "INCONCLUSIVE"
    else:
        verdicts = [p['verdict'] for p in results['pairs']]
        if all(v == "NOT_REJECTED" for v in verdicts):
            results['overall_verdict'] = "NOT_REJECTED"
        elif any(v == "DISFAVORED" for v in verdicts):
            results['overall_verdict'] = "DISFAVORED"
        else:
            results['overall_verdict'] = "INCONCLUSIVE"

    # Save results
    result_file = out_dir / "rank1_result.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {result_file}")

    # Generate report
    _generate_report(results, out_dir, config)

    return results


def _format_r_value(r_dict: Optional[Dict], label: str = "") -> str:
    """Format R value from dict with r and phi_deg."""
    if r_dict is None:
        return "N/A"
    r = r_dict.get('r', 0)
    phi = r_dict.get('phi_deg', 0)
    if label:
        return f"|R|={r:.3f}, φ={phi:.1f}°"
    return f"{r:.3f}"


def _generate_report(results: Dict, out_dir: Path, config: StateConfig):
    """Generate detailed markdown report from JSON results."""
    report = f"""# Rank-1 Test Report: {results['slug']}

## Summary

| Metric | Value |
|--------|-------|
| Status | {results['status']} |
| Overall Verdict | **{results.get('overall_verdict', 'N/A')}** |
| States Tested | {config.state1_name} vs {config.state2_name} |
| Fit Window | {config.fit_window[0]}-{config.fit_window[1]} MeV |
| Bootstrap Replicates | {results.get('n_boot', 'N/A')} |

## State Parameters

| State | Mass (MeV) | Width (MeV) |
|-------|------------|-------------|
| {config.state1_name} | {config.state1_mass} | {config.state1_width} |
| {config.state2_name} | {config.state2_mass} | {config.state2_width} |

## Pair Results

"""
    for i, pair in enumerate(results.get('pairs', [])):
        # Format values safely
        lambda_val = pair.get('Lambda')
        lambda_raw = pair.get('Lambda_raw')
        lambda_str = f"{lambda_val:.4f}" if lambda_val is not None else "N/A"
        lambda_raw_str = f"{lambda_raw:.4f}" if lambda_raw is not None else "N/A"

        p_boot = pair.get('p_boot')
        p_boot_str = f"{p_boot:.4f}" if p_boot is not None else "N/A"
        p_wilks = pair.get('p_wilks')
        p_wilks_str = f"{p_wilks:.4f}" if p_wilks is not None else "N/A"

        chi2_A = pair.get('chi2_A')
        chi2_A_str = f"{chi2_A:.1f}" if chi2_A is not None else "?"
        chi2_B = pair.get('chi2_B')
        chi2_B_str = f"{chi2_B:.1f}" if chi2_B is not None else "?"

        nll_con = pair.get('nll_con')
        nll_unc = pair.get('nll_unc')
        nll_con_str = f"{nll_con:.4f}" if nll_con is not None else "N/A"
        nll_unc_str = f"{nll_unc:.4f}" if nll_unc is not None else "N/A"

        invariant = pair.get('invariant_holds', 'N/A')
        invariant_str = "✓ PASS" if invariant is True else ("✗ FAIL" if invariant is False else str(invariant))

        n_boot_valid = pair.get('n_boot_valid', pair.get('n_boot', '?'))
        n_boot_failed = pair.get('n_boot_failed', 0)
        k = pair.get('k', '?')

        report += f"""### {pair.get('pair', f'Pair {i+1}')}

**Tables**: `{pair.get('table_A', 'N/A')}` vs `{pair.get('table_B', 'N/A')}`

**Verdict**: **{pair.get('verdict', 'N/A')}**

**Reason**: {pair.get('reason', 'N/A')}

#### Test Statistics

| Metric | Value |
|--------|-------|
| Λ (clamped) | {lambda_str} |
| Λ_raw | {lambda_raw_str} |
| p_boot | {p_boot_str} ({k}/{n_boot_valid} exceedances) |
| p_wilks (ref) | {p_wilks_str} |
| NLL constrained | {nll_con_str} |
| NLL unconstrained | {nll_unc_str} |

#### Fit Health

| Channel | Health | χ²/dof |
|---------|--------|--------|
| A | {pair.get('health_A', 'N/A')} | {chi2_A_str}/{pair.get('dof_A', '?')} |
| B | {pair.get('health_B', 'N/A')} | {chi2_B_str}/{pair.get('dof_B', '?')} |

#### Coupling Ratios

| Fit Type | Channel | |R| | φ (deg) |
|----------|---------|-----|---------|
"""
        # Individual fits
        r_a_ind = pair.get('R_A_ind')
        r_b_ind = pair.get('R_B_ind')
        if r_a_ind:
            report += f"| Individual | A | {r_a_ind.get('r', 0):.3f} | {r_a_ind.get('phi_deg', 0):.1f}° |\n"
        if r_b_ind:
            report += f"| Individual | B | {r_b_ind.get('r', 0):.3f} | {r_b_ind.get('phi_deg', 0):.1f}° |\n"

        # Unconstrained joint fits
        r_a_unc = pair.get('R_A_unc')
        r_b_unc = pair.get('R_B_unc')
        if r_a_unc:
            report += f"| Unconstrained | A | {r_a_unc.get('r', 0):.3f} | {r_a_unc.get('phi_deg', 0):.1f}° |\n"
        if r_b_unc:
            report += f"| Unconstrained | B | {r_b_unc.get('r', 0):.3f} | {r_b_unc.get('phi_deg', 0):.1f}° |\n"

        # Constrained (shared)
        r_shared = pair.get('R_shared')
        if r_shared:
            report += f"| **Constrained** | **Shared** | **{r_shared.get('r', 0):.3f}** | **{r_shared.get('phi_deg', 0):.1f}°** |\n"

        # Legacy R_A/R_B (if present but no R_A_ind)
        if not r_a_ind:
            r_a = pair.get('R_A')
            r_b = pair.get('R_B')
            if r_a:
                report += f"| (legacy) | A | {r_a.get('r', 0):.3f} | {r_a.get('phi_deg', 0):.1f}° |\n"
            if r_b:
                report += f"| (legacy) | B | {r_b.get('r', 0):.3f} | {r_b.get('phi_deg', 0):.1f}° |\n"

        # Sanity checks section
        report += f"""
#### Sanity Checks

| Check | Status |
|-------|--------|
| Nested invariant (nll_unc ≤ nll_con) | {invariant_str} |
| Invariant violation | {pair.get('invariant_violation', 'N/A')} |
| Bootstrap valid/failed | {n_boot_valid}/{n_boot_failed} |

"""
        # Bootstrap stats if available
        if pair.get('lambda_boot_mean') is not None:
            report += f"""#### Bootstrap Distribution

| Statistic | Value |
|-----------|-------|
| Mean Λ_boot | {pair.get('lambda_boot_mean', 'N/A'):.4f} |
| Std Λ_boot | {pair.get('lambda_boot_std', 'N/A'):.4f} |
| Median Λ_boot | {pair.get('lambda_boot_median', 'N/A'):.4f} |

"""

    report += f"""
## Interpretation

The rank-1 test examines whether the relative mixture ratio R = a₂/a₁
between {config.state1_name} and {config.state2_name} is consistent across different
analyses or cuts of the same underlying data.

**Verdicts**:
- **NOT_REJECTED**: p ≥ 0.05, consistent with rank-1 factorization
- **DISFAVORED**: p < 0.05, evidence against rank-1
- **INCONCLUSIVE**: Fit issues prevent reliable conclusion
- **OPTIMIZER_FAILURE**: Nested model invariant violated (nll_unc > nll_con)

**Nested Model Invariant**: For a valid likelihood ratio test, the unconstrained
fit (more parameters) must achieve NLL ≤ constrained fit NLL. Violation indicates
optimizer failure, not a physics result.

{config.notes}

---
*Generated by rank1_discovery_mine v2.0 (with nested invariant checks)*
"""

    report_file = out_dir / "RANK1_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Saved: {report_file}")
