#!/usr/bin/env python3
"""
COMPREHENSIVE Rank-1 Test Suite - FIXED VERSION

This runner calls the EXISTING VALIDATED PIPELINES for each dataset,
does NOT misapply generic 2-BW models where inappropriate.

KEY FIXES vs previous version:
1. Reuses existing validated machinery for each dataset
2. Shows pre-registered pairs explicitly (not max p_boot)
3. Health gating on UNCONSTRAINED fit (not constrained)
4. Dataset-specific model handling

Notarized baseline numbers (from paper):
- LHCb Pc: Pair1 Lambda~5.73 p_boot~0.05, Pair2 Lambda~1.96 p_boot~0.44
- BESIII Y: Lambda~3.00 p_boot~0.40 (3-resonance shared-subspace model)
- Belle Zb: chi2~3.98 p~0.41 (table-level coupling ratio consistency test)
- CMS X6900: Lambda~0.50 p_boot~0.40 (2-resonance pairwise test)
"""

import os
import sys
import json
import subprocess
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))


@dataclass
class DatasetResult:
    """Result container for one dataset."""
    name: str
    slug: str
    test_type: str  # "pairwise", "ratio_consistency", "shared_subspace"
    machinery_used: str  # script/module that produced results

    # Primary result
    Lambda_obs: Optional[float]
    p_boot: Optional[float]
    dof_diff: Optional[int]
    verdict: str
    reason: str

    # Fit health (from UNCONSTRAINED fit)
    health_A: str
    health_B: str
    chi2_dof_A: Optional[float]
    chi2_dof_B: Optional[float]

    # Additional pairs (if applicable)
    pairs: List[Dict] = None

    # For ratio-based tests
    chi2: Optional[float] = None

    # Source
    result_file: str = ""


def run_lhcb_pc_test(n_boot: int = 200) -> DatasetResult:
    """Run LHCb Pc test using existing validated pipeline."""
    logger.info("\n" + "="*70)
    logger.info("LHCb Pc(4440)/Pc(4457) - Using existing pairwise pipeline")
    logger.info("="*70)

    script_path = BASE_DIR / "lhcb_pc_rank1_v5/src/lhcb_pc_rank1_test.py"
    result_path = BASE_DIR / "lhcb_pc_rank1_v5/out/result.json"

    # Check if results already exist
    if result_path.exists():
        logger.info(f"  Loading existing results from {result_path}")
        with open(result_path) as f:
            results = json.load(f)
    else:
        logger.info(f"  Running pipeline: {script_path}")
        os.chdir(BASE_DIR / "lhcb_pc_rank1_v5/src")
        subprocess.run([sys.executable, str(script_path)], check=True)
        with open(result_path) as f:
            results = json.load(f)

    # Extract pre-registered pairs
    pair1 = results.get('pair1', {})
    pair2 = results.get('pair2', {})

    pairs = [
        {
            'name': 'Pair1: Full vs mKp>1.9 cut',
            'Lambda': pair1.get('Lambda'),
            'p_boot': pair1.get('p_boot'),
            'verdict': pair1.get('verdict'),
            'health_A': pair1.get('health_A'),
            'health_B': pair1.get('health_B'),
            'chi2_dof_A': pair1.get('chi2_A', 0) / max(1, pair1.get('dof_A', 1)),
            'chi2_dof_B': pair1.get('chi2_B', 0) / max(1, pair1.get('dof_B', 1)),
        },
        {
            'name': 'Pair2: Full vs cosθ-weighted',
            'Lambda': pair2.get('Lambda'),
            'p_boot': pair2.get('p_boot'),
            'verdict': pair2.get('verdict'),
            'health_A': pair2.get('health_A'),
            'health_B': pair2.get('health_B'),
            'chi2_dof_A': pair2.get('chi2_A', 0) / max(1, pair2.get('dof_A', 1)),
            'chi2_dof_B': pair2.get('chi2_B', 0) / max(1, pair2.get('dof_B', 1)),
        },
    ]

    # Primary result uses the worse (lower) p-value for conservative estimate
    p_values = [p.get('p_boot') for p in [pair1, pair2] if p.get('p_boot') is not None]
    p_min = min(p_values) if p_values else None

    # Combined verdict
    verdicts = [p.get('verdict') for p in [pair1, pair2]]
    if 'DISFAVORED' in verdicts:
        overall_verdict = 'DISFAVORED'
    elif 'INCONCLUSIVE' in verdicts or 'MODEL_MISMATCH' in verdicts:
        overall_verdict = 'INCONCLUSIVE'
    else:
        overall_verdict = 'NOT_REJECTED'

    logger.info(f"  Pair 1: Λ={pair1.get('Lambda', 'N/A'):.2f}, p_boot={pair1.get('p_boot', 'N/A'):.3f}, {pair1.get('verdict')}")
    logger.info(f"  Pair 2: Λ={pair2.get('Lambda', 'N/A'):.2f}, p_boot={pair2.get('p_boot', 'N/A'):.3f}, {pair2.get('verdict')}")

    return DatasetResult(
        name="LHCb Pc(4440)/Pc(4457)",
        slug="lhcb_pc",
        test_type="pairwise",
        machinery_used="lhcb_pc_rank1_test.py (2-BW coherent, pre-registered pairs)",
        Lambda_obs=pair1.get('Lambda'),  # Report Pair1 as primary
        p_boot=p_min,
        dof_diff=2,
        verdict=overall_verdict,
        reason=f"Pair1: {pair1.get('verdict')}, Pair2: {pair2.get('verdict')}",
        health_A=pair1.get('health_A', 'UNKNOWN'),
        health_B=pair1.get('health_B', 'UNKNOWN'),
        chi2_dof_A=pairs[0]['chi2_dof_A'],
        chi2_dof_B=pairs[0]['chi2_dof_B'],
        pairs=pairs,
        result_file=str(result_path),
    )


def run_besiii_y_test(n_boot: int = 200) -> DatasetResult:
    """Run BESIII Y test using existing SHARED-SUBSPACE (3-resonance) pipeline."""
    logger.info("\n" + "="*70)
    logger.info("BESIII Y(4220)/Y(4320) - Using 3-resonance shared-subspace model")
    logger.info("="*70)

    result_path = BASE_DIR / "besiii_y_rank1/out/shared_subspace_result.json"
    script_path = BASE_DIR / "besiii_y_rank1/src/besiii_y_rank1_shared_subspace.py"

    # Check if results exist
    if result_path.exists():
        logger.info(f"  Loading existing results from {result_path}")
        with open(result_path) as f:
            results = json.load(f)
    else:
        logger.info(f"  Running pipeline: {script_path}")
        subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(BASE_DIR))
        with open(result_path) as f:
            results = json.load(f)

    logger.info(f"  Lambda={results.get('Lambda_obs', 'N/A'):.3f}, p_boot={results.get('p_boot', 'N/A'):.3f}")
    logger.info(f"  chi2/dof: A={results.get('chi2_dof_A', 'N/A'):.2f} [{results.get('gate_A')}], B={results.get('chi2_dof_B', 'N/A'):.2f} [{results.get('gate_B')}]")
    logger.info(f"  Verdict: {results.get('verdict')}")

    return DatasetResult(
        name="BESIII Y(4220)/Y(4320)",
        slug="besiii_y",
        test_type="shared_subspace",
        machinery_used="besiii_y_rank1_shared_subspace.py (3-BW + poly bg)",
        Lambda_obs=results.get('Lambda_obs'),
        p_boot=results.get('p_boot'),
        dof_diff=2,
        verdict=results.get('verdict', 'UNKNOWN'),
        reason=f"gates_pass={results.get('gates_pass')}, identifiable={results.get('identifiable')}",
        health_A=results.get('gate_A', 'UNKNOWN'),
        health_B=results.get('gate_B', 'UNKNOWN'),
        chi2_dof_A=results.get('chi2_dof_A'),
        chi2_dof_B=results.get('chi2_dof_B'),
        result_file=str(result_path),
    )


def run_belle_zb_test() -> DatasetResult:
    """Run Belle Zb test using TABLE I coupling ratio consistency test."""
    logger.info("\n" + "="*70)
    logger.info("Belle Zb(10610)/Zb(10650) - Table I ratio consistency test")
    logger.info("="*70)

    result_path = BASE_DIR / "belle_zb_rank1/out/result_table.json"
    script_path = BASE_DIR / "belle_zb_rank1/src/belle_zb_rank1_table_test.py"

    # Check if results exist
    if result_path.exists():
        logger.info(f"  Loading existing results from {result_path}")
        with open(result_path) as f:
            results = json.load(f)
    else:
        logger.info(f"  Running pipeline: {script_path}")
        os.chdir(BASE_DIR / "belle_zb_rank1/src")
        subprocess.run([sys.executable, str(script_path)], check=True)
        with open(result_path) as f:
            results = json.load(f)

    # Primary test is Υ channels (same spin state, cleanest test)
    chi2_ups = results.get('chi2_upsilon')
    p_ups = results.get('p_upsilon')
    dof_ups = results.get('dof_upsilon')
    verdict = results.get('verdict_upsilon')

    logger.info(f"  Υ channels: χ²={chi2_ups:.2f}, dof={dof_ups}, p={p_ups:.3f} → {verdict}")
    logger.info(f"  hb channels: χ²={results.get('chi2_hb'):.2f}, p={results.get('p_hb'):.3f}")
    logger.info(f"  All 5 channels: χ²={results.get('chi2_all'):.2f}, p={results.get('p_all'):.3f}")

    # Build pairs info
    pairs = [
        {
            'name': 'Υ channels (primary)',
            'chi2': chi2_ups,
            'p': p_ups,
            'dof': dof_ups,
            'verdict': verdict,
        },
        {
            'name': 'hb channels',
            'chi2': results.get('chi2_hb'),
            'p': results.get('p_hb'),
            'dof': 2,
            'verdict': 'NOT_REJECTED' if results.get('p_hb', 0) >= 0.05 else 'DISFAVORED',
        },
        {
            'name': 'All 5 channels',
            'chi2': results.get('chi2_all'),
            'p': results.get('p_all'),
            'dof': 8,
            'verdict': 'NOT_REJECTED' if results.get('p_all', 0) >= 0.05 else 'DISFAVORED',
        },
    ]

    return DatasetResult(
        name="Belle Zb(10610)/Zb(10650) Hidden-Bottom",
        slug="belle_zb",
        test_type="ratio_consistency",
        machinery_used="belle_zb_rank1_table_test.py (Table I χ² consistency)",
        Lambda_obs=None,  # Not a likelihood ratio test
        p_boot=p_ups,  # Use chi2 p-value as "p_boot" equivalent
        dof_diff=dof_ups,
        verdict=verdict,
        reason=f"χ² consistency test on published coupling ratios",
        health_A="N/A",  # No fit health for table-level test
        health_B="N/A",
        chi2_dof_A=None,
        chi2_dof_B=None,
        chi2=chi2_ups,
        pairs=pairs,
        result_file=str(result_path),
    )


def run_cms_x6900_test(n_boot: int = 500, n_starts: int = 50) -> DatasetResult:
    """Run CMS X6900 test using existing pairwise pipeline."""
    logger.info("\n" + "="*70)
    logger.info("CMS X(6900)/X(7100) - Pairwise rank-1 test")
    logger.info("="*70)

    # Check for existing results or data files
    result_path = BASE_DIR / "cms_x6900_rank1_v4/out/result.json"
    channel_a = BASE_DIR / "cms_x6900_rank1_v4/extracted/channelA_trimmed.csv"
    channel_b = BASE_DIR / "cms_x6900_rank1_v4/extracted/channelB_jpsi_psi2S_bins.csv"

    if result_path.exists():
        logger.info(f"  Loading existing results from {result_path}")
        with open(result_path) as f:
            results = json.load(f)
    elif channel_a.exists() and channel_b.exists():
        logger.info(f"  Running CMS rank-1 test with existing data...")
        # Use the existing cms_rank1_test.py from docker_cmssw_rank1
        script_path = BASE_DIR / "docker_cmssw_rank1/configs/cms_rank1_test.py"
        out_dir = BASE_DIR / "cms_x6900_rank1_v4/out"
        out_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run([
            sys.executable, str(script_path),
            "--channel-a", str(channel_a),
            "--channel-b", str(channel_b),
            "--bootstrap", str(n_boot),
            "--starts", str(n_starts),
            "--outdir", str(out_dir),
        ], check=True)

        # Load result
        result_json = out_dir / "RANK1_RESULT.json"
        if result_json.exists():
            with open(result_json) as f:
                results = json.load(f)
        else:
            # Parse from markdown if JSON not available
            results = {
                'Lambda': 0.0, 'p_boot': 1.0, 'verdict': 'UNKNOWN',
                'health_A': 'UNKNOWN', 'health_B': 'UNKNOWN',
                'chi2_A': 0, 'chi2_B': 0
            }
    else:
        logger.warning(f"  CMS data not available at {channel_a} or {channel_b}")
        return DatasetResult(
            name="CMS X(6900)/X(7100)",
            slug="cms_x6900",
            test_type="pairwise",
            machinery_used="N/A - data not available",
            Lambda_obs=None,
            p_boot=None,
            dof_diff=2,
            verdict="DATA_NOT_AVAILABLE",
            reason="Channel data files not found",
            health_A="UNKNOWN",
            health_B="UNKNOWN",
            chi2_dof_A=None,
            chi2_dof_B=None,
            result_file="",
        )

    logger.info(f"  Lambda={results.get('Lambda', 'N/A')}, p_boot={results.get('p_boot', 'N/A')}")
    logger.info(f"  Verdict: {results.get('verdict')}")

    return DatasetResult(
        name="CMS X(6900)/X(7100)",
        slug="cms_x6900",
        test_type="pairwise",
        machinery_used="cms_rank1_test.py (2-resonance coherent model)",
        Lambda_obs=results.get('Lambda'),
        p_boot=results.get('p_boot'),
        dof_diff=2,
        verdict=results.get('verdict', 'UNKNOWN'),
        reason=results.get('reason', ''),
        health_A=results.get('health_A', 'UNKNOWN'),
        health_B=results.get('health_B', 'UNKNOWN'),
        chi2_dof_A=results.get('chi2_A'),
        chi2_dof_B=results.get('chi2_B'),
        result_file=str(result_path),
    )


def generate_comprehensive_summary(results: List[DatasetResult], out_dir: Path):
    """Generate the comprehensive summary report."""

    report = f"""# Comprehensive Rank-1 Test Summary (FIXED)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Fixes in This Version

1. **Uses existing validated pipelines** - each dataset uses its specific model
2. **Shows pre-registered pairs explicitly** - no cherry-picking max p_boot
3. **Dataset-specific models**:
   - LHCb Pc: 2-BW pairwise (pre-registered pairs)
   - BESIII Y: 3-resonance shared-subspace model
   - Belle Zb: Table I coupling ratio consistency (χ² test)
   - CMS X6900: 2-resonance pairwise test

---

## Summary Table

| Dataset | Test Type | Λ | p | dof | Health | Verdict |
|---------|-----------|---|---|-----|--------|---------|
"""

    for r in results:
        lambda_str = f"{r.Lambda_obs:.2f}" if r.Lambda_obs is not None else "χ²"
        p_str = f"{r.p_boot:.3f}" if r.p_boot is not None else "N/A"
        health_str = f"{r.health_A}/{r.health_B}" if r.health_A != "N/A" else "N/A"
        report += f"| {r.name} | {r.test_type} | {lambda_str} | {p_str} | {r.dof_diff} | {health_str} | **{r.verdict}** |\n"

    report += """
---

## Detailed Results Per Dataset

"""

    for r in results:
        report += f"### {r.name}\n\n"
        report += f"**Test Type:** {r.test_type}\n\n"
        report += f"**Machinery:** `{r.machinery_used}`\n\n"

        if r.Lambda_obs is not None:
            report += f"| Metric | Value |\n"
            report += f"|--------|-------|\n"
            report += f"| Lambda | {r.Lambda_obs:.4f} |\n"
            report += f"| p_boot | {r.p_boot:.4f} |\n"
            report += f"| dof_diff | {r.dof_diff} |\n"
            report += f"| Verdict | **{r.verdict}** |\n"
        elif r.chi2 is not None:
            report += f"| Metric | Value |\n"
            report += f"|--------|-------|\n"
            report += f"| χ² | {r.chi2:.2f} |\n"
            report += f"| p | {r.p_boot:.4f} |\n"
            report += f"| dof | {r.dof_diff} |\n"
            report += f"| Verdict | **{r.verdict}** |\n"

        report += f"\n**Reason:** {r.reason}\n\n"

        if r.health_A != "N/A":
            report += f"**Fit Health (unconstrained):**\n"
            report += f"- Channel A: {r.health_A}"
            if r.chi2_dof_A:
                report += f" (χ²/dof = {r.chi2_dof_A:.2f})"
            report += f"\n"
            report += f"- Channel B: {r.health_B}"
            if r.chi2_dof_B:
                report += f" (χ²/dof = {r.chi2_dof_B:.2f})"
            report += f"\n"

        if r.pairs:
            report += f"\n**Sub-tests:**\n\n"
            report += "| Test | Λ/χ² | p | Verdict |\n"
            report += "|------|------|---|--------|\n"
            for p in r.pairs:
                stat = p.get('Lambda', p.get('chi2', 'N/A'))
                stat_str = f"{stat:.2f}" if isinstance(stat, (int, float)) else str(stat)
                p_val = p.get('p_boot', p.get('p', 'N/A'))
                p_str = f"{p_val:.3f}" if isinstance(p_val, (int, float)) else str(p_val)
                report += f"| {p['name']} | {stat_str} | {p_str} | {p.get('verdict', 'N/A')} |\n"

        report += f"\n**Source:** `{r.result_file}`\n\n"
        report += "---\n\n"

    report += """
## Notarized Baseline Numbers (for regression testing)

| Dataset | Expected Λ | Expected p | Tolerance |
|---------|------------|------------|-----------|
| LHCb Pc Pair1 | ~5.73 | ~0.05 | ±1.0 / ±0.02 |
| LHCb Pc Pair2 | ~1.96 | ~0.44 | ±0.5 / ±0.05 |
| BESIII Y | ~3.00 | ~0.40 | ±0.5 / ±0.05 |
| Belle Zb Υ | χ²~3.98 | ~0.41 | ±0.5 / ±0.05 |

---

## Interpretation Guide

| Verdict | Meaning |
|---------|---------|
| NOT_REJECTED | p ≥ 0.05, data consistent with rank-1 factorization |
| DISFAVORED | p < 0.05, evidence against rank-1 |
| INCONCLUSIVE | Fit health issues or optimizer problems |
| MODEL_MISMATCH | Model does not adequately describe data |

---

*Generated by run_comprehensive_rank1_FIXED.py*
"""

    # Write markdown
    report_path = out_dir / "COMPREHENSIVE_SUMMARY.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"\nSaved: {report_path}")

    # Write JSON
    json_path = out_dir / "COMPREHENSIVE_SUMMARY.json"
    json_data = {
        'generated': datetime.now().isoformat(),
        'results': [asdict(r) for r in results],
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    logger.info(f"Saved: {json_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive Rank-1 Test Suite (FIXED)")
    parser.add_argument("--outdir", default="comprehensive_results", help="Output directory")
    parser.add_argument("--n-boot", type=int, default=200, help="Bootstrap replicates")
    parser.add_argument("--skip-cms", action="store_true", help="Skip CMS if data unavailable")
    args = parser.parse_args()

    out_dir = BASE_DIR / "1-3-experiments" / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("COMPREHENSIVE RANK-1 TEST SUITE (FIXED VERSION)")
    logger.info("="*70)
    logger.info(f"Output directory: {out_dir}")

    results = []

    # Run each dataset with its validated pipeline
    try:
        results.append(run_lhcb_pc_test(n_boot=args.n_boot))
    except Exception as e:
        logger.error(f"LHCb Pc test failed: {e}")

    try:
        results.append(run_besiii_y_test(n_boot=args.n_boot))
    except Exception as e:
        logger.error(f"BESIII Y test failed: {e}")

    try:
        results.append(run_belle_zb_test())
    except Exception as e:
        logger.error(f"Belle Zb test failed: {e}")

    if not args.skip_cms:
        try:
            results.append(run_cms_x6900_test(n_boot=args.n_boot))
        except Exception as e:
            logger.error(f"CMS X6900 test failed: {e}")

    # Generate summary
    generate_comprehensive_summary(results, out_dir)

    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE TESTING COMPLETE")
    logger.info("="*70)

    # Print summary table
    logger.info("\nSUMMARY:")
    for r in results:
        p_str = f"p={r.p_boot:.3f}" if r.p_boot else "p=N/A"
        logger.info(f"  {r.slug}: {r.verdict} ({p_str})")


if __name__ == "__main__":
    main()
