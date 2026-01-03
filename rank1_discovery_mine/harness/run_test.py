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


def find_suitable_pairs(raw_dir: Path, config: StateConfig) -> List[Tuple[TableInfo, TableInfo]]:
    """
    Find suitable table pairs for rank-1 test.

    Prioritizes tables from records that match the state configuration
    (by checking for state names in description).

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

    for f in csv_files:
        info = analyze_table(f)
        if info and info.n_rows >= 5:
            tables.append(info)

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

    # Group by INSPIRE ID
    by_inspire = {}
    for t in target_tables:
        if t.inspire_id:
            if t.inspire_id not in by_inspire:
                by_inspire[t.inspire_id] = []
            by_inspire[t.inspire_id].append(t)

    # Generate pairs - prioritize within same record
    pairs = []

    # Within same record: pair different tables (e.g., different cuts)
    for inspire_id, record_tables in by_inspire.items():
        if len(record_tables) >= 2:
            # Sort by name/number
            record_tables.sort(key=lambda t: t.name)
            # Pair consecutive tables
            for i in range(len(record_tables) - 1):
                pairs.append((record_tables[i], record_tables[i+1]))
                logger.info(f"  Paired: {record_tables[i].name} + {record_tables[i+1].name} (ins{inspire_id})")

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
    raw_dir = candidate_dir / "raw"
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
    pairs = find_suitable_pairs(raw_dir, config)

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


def _generate_report(results: Dict, out_dir: Path, config: StateConfig):
    """Generate markdown report."""
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
        report += f"""### {pair.get('pair', f'Pair {i+1}')}

- **Tables**: {pair.get('table_A', 'N/A')} vs {pair.get('table_B', 'N/A')}
- **Verdict**: {pair.get('verdict', 'N/A')}
- **Λ**: {pair.get('Lambda', 'N/A'):.2f if pair.get('Lambda') else 'N/A'}
- **p_boot**: {pair.get('p_boot', 'N/A'):.4f if pair.get('p_boot') else 'N/A'} ({pair.get('k', '?')}/{pair.get('n_boot', '?')})
- **Health A**: {pair.get('health_A', 'N/A')} (χ²/dof = {pair.get('chi2_A', '?'):.1f if pair.get('chi2_A') else '?'}/{pair.get('dof_A', '?')})
- **Health B**: {pair.get('health_B', 'N/A')} (χ²/dof = {pair.get('chi2_B', '?'):.1f if pair.get('chi2_B') else '?'}/{pair.get('dof_B', '?')})

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

{config.notes}

---
*Generated by rank1_discovery_mine*
"""

    report_file = out_dir / "RANK1_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Saved: {report_file}")
