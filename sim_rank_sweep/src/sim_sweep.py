#!/usr/bin/env python3
"""
sim_sweep.py - Power study sweep across tests and statistics levels

Runs the full simulation + power analysis:
- Type I error under M0
- Power under M1 and M4
- CI widths and identifiability
- Grid search for 80% power threshold

Uses multiprocessing for parallelization.
"""

import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from sim_generate import generate_dataset
from sim_fit import run_full_fit, check_identifiability


@dataclass
class PowerResult:
    """Results for a single test at a single stats level"""
    test_name: str
    family: str
    data_type: str
    nbins: int
    stats_level: float
    n_trials: int
    n_bootstrap: int

    # Type I error (M0)
    typeI_M0: float
    typeI_M0_se: float

    # Power (M1)
    power_M1: float
    power_M1_se: float

    # Power (M4)
    power_M4: float
    power_M4_se: float

    # CI widths
    median_CI_r: float
    median_CI_phi: float

    # Identifiability
    identifiability_rate: float

    # Fit quality
    pass_rate: float
    avg_chi2_dof: float

    # Additional
    median_lambda_M0: float
    median_lambda_M1: float
    median_lambda_M4: float


def run_single_trial(args: Tuple) -> Dict:
    """
    Run a single trial for power estimation.
    Used by multiprocessing pool.
    """
    test_config, mechanism, scale_factor, trial_idx, n_bootstrap = args

    seed = trial_idx * 1000 + int(scale_factor * 100)

    try:
        # Generate data
        dataset = generate_dataset(test_config, mechanism, scale_factor, seed=seed)

        # Run fit
        result = run_full_fit(dataset, n_bootstrap=n_bootstrap, n_restarts=5)

        # Check identifiability
        ident = check_identifiability(dataset, n_fits=5)

        return {
            'success': True,
            'mechanism': mechanism,
            'Lambda': result['Lambda'],
            'p_value': result['p_value'],
            'gates': result['gates'],
            'chi2_dof_a': result.get('chi2_dof_a', np.nan),
            'chi2_dof_b': result.get('chi2_dof_b', np.nan),
            'r_ci': result.get('r_ci_width', np.nan),
            'phi_ci': result.get('phi_ci_width', np.nan),
            'identifiable': ident['identifiable']
        }
    except Exception as e:
        return {
            'success': False,
            'mechanism': mechanism,
            'error': str(e)
        }


def run_power_analysis(test_config: Dict, stats_level: float,
                       n_trials: int = 100, n_bootstrap: int = 200,
                       n_workers: Optional[int] = None) -> PowerResult:
    """
    Run power analysis for a single test at a given statistics level.

    Args:
        test_config: Test configuration dict
        stats_level: Statistics multiplier
        n_trials: Number of MC trials
        n_bootstrap: Bootstrap samples per trial
        n_workers: Number of parallel workers

    Returns:
        PowerResult with all metrics
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    test_name = test_config['name']
    family = test_config['family']

    # Determine data type and nbins
    ch_a_type = test_config['channelA']['type']
    ch_b_type = test_config['channelB']['type']
    data_type = 'mixed' if ch_a_type != ch_b_type else ch_a_type
    nbins = (test_config['channelA']['nbins'] + test_config['channelB']['nbins']) // 2

    # Prepare trial arguments for each mechanism
    args_M0 = [(test_config, 'M0', stats_level, i, n_bootstrap) for i in range(n_trials)]
    args_M1 = [(test_config, 'M1', stats_level, i, n_bootstrap) for i in range(n_trials)]
    args_M4 = [(test_config, 'M4', stats_level, i, n_bootstrap) for i in range(n_trials)]

    all_args = args_M0 + args_M1 + args_M4

    # Run in parallel
    with Pool(n_workers) as pool:
        all_results = pool.map(run_single_trial, all_args)

    # Split results by mechanism
    results_M0 = all_results[:n_trials]
    results_M1 = all_results[n_trials:2*n_trials]
    results_M4 = all_results[2*n_trials:]

    # Analyze M0 (Type I error)
    valid_M0 = [r for r in results_M0 if r['success'] and r['gates'] == 'PASS']
    if len(valid_M0) > 0:
        rejections_M0 = sum(1 for r in valid_M0 if r['p_value'] < 0.05)
        typeI_M0 = rejections_M0 / len(valid_M0)
        typeI_M0_se = np.sqrt(typeI_M0 * (1 - typeI_M0) / len(valid_M0))
        lambdas_M0 = [r['Lambda'] for r in valid_M0 if not np.isnan(r['Lambda'])]
        median_lambda_M0 = np.median(lambdas_M0) if lambdas_M0 else np.nan
    else:
        typeI_M0 = np.nan
        typeI_M0_se = np.nan
        median_lambda_M0 = np.nan

    # Analyze M1 (Power)
    valid_M1 = [r for r in results_M1 if r['success'] and r['gates'] == 'PASS']
    if len(valid_M1) > 0:
        rejections_M1 = sum(1 for r in valid_M1 if r['p_value'] < 0.05)
        power_M1 = rejections_M1 / len(valid_M1)
        power_M1_se = np.sqrt(power_M1 * (1 - power_M1) / len(valid_M1))
        lambdas_M1 = [r['Lambda'] for r in valid_M1 if not np.isnan(r['Lambda'])]
        median_lambda_M1 = np.median(lambdas_M1) if lambdas_M1 else np.nan
    else:
        power_M1 = np.nan
        power_M1_se = np.nan
        median_lambda_M1 = np.nan

    # Analyze M4 (Power)
    valid_M4 = [r for r in results_M4 if r['success'] and r['gates'] == 'PASS']
    if len(valid_M4) > 0:
        rejections_M4 = sum(1 for r in valid_M4 if r['p_value'] < 0.05)
        power_M4 = rejections_M4 / len(valid_M4)
        power_M4_se = np.sqrt(power_M4 * (1 - power_M4) / len(valid_M4))
        lambdas_M4 = [r['Lambda'] for r in valid_M4 if not np.isnan(r['Lambda'])]
        median_lambda_M4 = np.median(lambdas_M4) if lambdas_M4 else np.nan
    else:
        power_M4 = np.nan
        power_M4_se = np.nan
        median_lambda_M4 = np.nan

    # CI widths (from M0 fits)
    r_cis = [r['r_ci'] for r in valid_M0 if not np.isnan(r.get('r_ci', np.nan))]
    phi_cis = [r['phi_ci'] for r in valid_M0 if not np.isnan(r.get('phi_ci', np.nan))]
    median_CI_r = np.median(r_cis) if r_cis else np.nan
    median_CI_phi = np.median(phi_cis) if phi_cis else np.nan

    # Identifiability rate
    ident_checks = [r['identifiable'] for r in valid_M0 if 'identifiable' in r]
    identifiability_rate = np.mean(ident_checks) if ident_checks else np.nan

    # Pass rate and chi2
    all_valid = [r for r in all_results if r['success']]
    pass_rate = len([r for r in all_valid if r['gates'] == 'PASS']) / len(all_valid) if all_valid else 0.0

    chi2s = []
    for r in all_valid:
        if not np.isnan(r.get('chi2_dof_a', np.nan)):
            chi2s.append(r['chi2_dof_a'])
        if not np.isnan(r.get('chi2_dof_b', np.nan)):
            chi2s.append(r['chi2_dof_b'])
    avg_chi2_dof = np.mean(chi2s) if chi2s else np.nan

    return PowerResult(
        test_name=test_name,
        family=family,
        data_type=data_type,
        nbins=nbins,
        stats_level=stats_level,
        n_trials=n_trials,
        n_bootstrap=n_bootstrap,
        typeI_M0=typeI_M0,
        typeI_M0_se=typeI_M0_se,
        power_M1=power_M1,
        power_M1_se=power_M1_se,
        power_M4=power_M4,
        power_M4_se=power_M4_se,
        median_CI_r=median_CI_r,
        median_CI_phi=median_CI_phi,
        identifiability_rate=identifiability_rate,
        pass_rate=pass_rate,
        avg_chi2_dof=avg_chi2_dof,
        median_lambda_M0=median_lambda_M0,
        median_lambda_M1=median_lambda_M1,
        median_lambda_M4=median_lambda_M4
    )


def find_80_power_level(test_config: Dict, stats_levels: List[float],
                        n_trials: int = 50, n_bootstrap: int = 100) -> Tuple[float, float]:
    """
    Find the statistics level needed for 80% power against M1.
    Returns (level_for_M1, level_for_M4) or (inf, inf) if not reachable.
    """
    power_M1_vals = []
    power_M4_vals = []

    for level in stats_levels:
        result = run_power_analysis(test_config, level, n_trials=n_trials,
                                   n_bootstrap=n_bootstrap)
        power_M1_vals.append((level, result.power_M1))
        power_M4_vals.append((level, result.power_M4))

    # Find first level with power >= 0.8
    level_M1 = float('inf')
    for level, power in power_M1_vals:
        if not np.isnan(power) and power >= 0.8:
            level_M1 = level
            break

    level_M4 = float('inf')
    for level, power in power_M4_vals:
        if not np.isnan(power) and power >= 0.8:
            level_M4 = level
            break

    return level_M1, level_M4


def run_full_sweep(config_path: str = 'configs/tests.json',
                   output_dir: str = 'out',
                   n_trials: int = 100,
                   n_bootstrap: int = 200,
                   stats_levels: Optional[List[float]] = None) -> List[PowerResult]:
    """
    Run full power sweep across all tests and statistics levels.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    tests = config['tests']
    settings = config.get('global_settings', {})

    if stats_levels is None:
        stats_levels = settings.get('stats_levels', [0.5, 1.0, 2.0, 4.0])

    n_trials = settings.get('n_mc_trials', n_trials)
    n_bootstrap = settings.get('n_bootstrap', n_bootstrap)

    all_results = []

    total_jobs = len(tests) * len(stats_levels)
    job_idx = 0

    for test in tests:
        test_name = test['name']
        print(f"\n{'='*60}")
        print(f"Processing: {test_name}")
        print(f"{'='*60}")

        for level in stats_levels:
            job_idx += 1
            print(f"\n[{job_idx}/{total_jobs}] Stats level = {level}x")

            result = run_power_analysis(test, level, n_trials=n_trials,
                                       n_bootstrap=n_bootstrap)
            all_results.append(result)

            print(f"  Type I (M0): {result.typeI_M0:.3f} ± {result.typeI_M0_se:.3f}")
            print(f"  Power (M1): {result.power_M1:.3f} ± {result.power_M1_se:.3f}")
            print(f"  Power (M4): {result.power_M4:.3f} ± {result.power_M4_se:.3f}")
            print(f"  Identifiability: {result.identifiability_rate:.2f}")
            print(f"  Pass rate: {result.pass_rate:.2f}")

    return all_results


def save_results(results: List[PowerResult], output_dir: str = 'out'):
    """Save results to CSV and Markdown files."""
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, 'POWER_TABLE.csv')
    with open(csv_path, 'w') as f:
        # Header
        fields = ['test_name', 'data_type', 'nbins', 'stats_level',
                  'power_M1', 'power_M4', 'typeI_M0',
                  'median_CI_r', 'median_CI_phi',
                  'identifiability_rate', 'pass_rate', 'recommended_priority']

        f.write(','.join(fields) + '\n')

        for r in results:
            # Calculate priority score
            priority = calculate_priority(r)

            row = [
                r.test_name,
                r.data_type,
                str(r.nbins),
                f"{r.stats_level:.1f}",
                f"{r.power_M1:.3f}" if not np.isnan(r.power_M1) else "nan",
                f"{r.power_M4:.3f}" if not np.isnan(r.power_M4) else "nan",
                f"{r.typeI_M0:.3f}" if not np.isnan(r.typeI_M0) else "nan",
                f"{r.median_CI_r:.3f}" if not np.isnan(r.median_CI_r) else "nan",
                f"{r.median_CI_phi:.1f}" if not np.isnan(r.median_CI_phi) else "nan",
                f"{r.identifiability_rate:.2f}" if not np.isnan(r.identifiability_rate) else "nan",
                f"{r.pass_rate:.2f}",
                priority
            ]
            f.write(','.join(row) + '\n')

    print(f"\nSaved: {csv_path}")

    # Markdown
    md_path = os.path.join(output_dir, 'POWER_TABLE.md')
    with open(md_path, 'w') as f:
        f.write("# POWER STUDY RESULTS\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary by test
        f.write("## Summary by Test (at highest stats level)\n\n")
        f.write("| Test | Family | Power M1 | Power M4 | Type I | Identifiable | Priority |\n")
        f.write("|------|--------|----------|----------|--------|--------------|----------|\n")

        # Get max stats level result for each test
        test_names = list(set(r.test_name for r in results))
        for test_name in test_names:
            test_results = [r for r in results if r.test_name == test_name]
            if test_results:
                best = max(test_results, key=lambda x: x.stats_level)
                priority = calculate_priority(best)
                f.write(f"| {test_name} | {best.family} | "
                       f"{best.power_M1:.2f} | {best.power_M4:.2f} | "
                       f"{best.typeI_M0:.3f} | {best.identifiability_rate:.2f} | "
                       f"{priority} |\n")

        f.write("\n## Full Results Table\n\n")
        f.write("| Test | Stats | Power M1 | Power M4 | Type I | CI(r) | CI(φ) | Pass Rate |\n")
        f.write("|------|-------|----------|----------|--------|-------|-------|----------|\n")

        for r in results:
            f.write(f"| {r.test_name[:30]} | {r.stats_level:.1f}x | "
                   f"{r.power_M1:.3f} | {r.power_M4:.3f} | {r.typeI_M0:.3f} | "
                   f"{r.median_CI_r:.3f} | {r.median_CI_phi:.1f} | {r.pass_rate:.2f} |\n")

        f.write("\n## Legend\n\n")
        f.write("- **Power M1**: Probability of rejecting rank-1 when M1 (unconstrained) is true\n")
        f.write("- **Power M4**: Probability of rejecting rank-1 when M4 (rank-2) is true\n")
        f.write("- **Type I**: Probability of falsely rejecting rank-1 when it's true\n")
        f.write("- **CI(r), CI(φ)**: Median confidence interval widths\n")
        f.write("- **Priority**: HIGH = decisive test, MEDIUM = usable, LOW = underpowered\n")

    print(f"Saved: {md_path}")


def calculate_priority(result: PowerResult) -> str:
    """
    Calculate recommended priority based on power and identifiability.
    """
    if np.isnan(result.power_M1) or np.isnan(result.identifiability_rate):
        return "UNKNOWN"

    # Check for pathologies
    if result.identifiability_rate < 0.5:
        return "LOW_IDENT"

    if result.pass_rate < 0.5:
        return "LOW_PASS"

    if not np.isnan(result.typeI_M0) and result.typeI_M0 > 0.15:
        return "HIGH_TYPEI"

    # Power-based priority
    if result.power_M1 >= 0.8:
        if result.power_M4 >= 0.5:
            return "HIGH"
        else:
            return "MEDIUM_M4LOW"
    elif result.power_M1 >= 0.5:
        return "MEDIUM"
    else:
        return "LOW_POWER"


def generate_summary(results: List[PowerResult], output_dir: str = 'out'):
    """Generate summary analysis."""
    summary_path = os.path.join(output_dir, 'POWER_SUMMARY.md')

    # Get best result per test (highest stats level with good identifiability)
    test_names = list(set(r.test_name for r in results))
    best_per_test = {}

    for test_name in test_names:
        test_results = [r for r in results if r.test_name == test_name]
        # Filter for identifiable
        good_results = [r for r in test_results
                       if not np.isnan(r.identifiability_rate) and r.identifiability_rate > 0.5]
        if good_results:
            best_per_test[test_name] = max(good_results, key=lambda x: x.power_M1 if not np.isnan(x.power_M1) else -1)
        elif test_results:
            best_per_test[test_name] = max(test_results, key=lambda x: x.stats_level)

    with open(summary_path, 'w') as f:
        f.write("# POWER STUDY SUMMARY\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Top 5 decisive tests
        f.write("## Top 5 Decisive Tests (Best Chance to Confirm/Falsify Rank-1)\n\n")
        ranked = sorted(best_per_test.values(),
                       key=lambda x: (x.power_M1 if not np.isnan(x.power_M1) else 0) *
                                    (x.identifiability_rate if not np.isnan(x.identifiability_rate) else 0),
                       reverse=True)

        for i, r in enumerate(ranked[:5], 1):
            priority = calculate_priority(r)
            f.write(f"### {i}. {r.test_name}\n\n")
            f.write(f"- **Family**: {r.family}\n")
            f.write(f"- **Power (M1)**: {r.power_M1:.2f} at {r.stats_level}x stats\n")
            f.write(f"- **Power (M4)**: {r.power_M4:.2f}\n")
            f.write(f"- **Type I error**: {r.typeI_M0:.3f}\n")
            f.write(f"- **Identifiability**: {r.identifiability_rate:.2f}\n")
            f.write(f"- **Priority**: {priority}\n")
            f.write(f"- **Key requirement**: {get_requirement(r)}\n\n")

        # Top 5 underpowered tests
        f.write("## Top 5 Underpowered Tests (Phase Not Identifiable)\n\n")
        underpowered = sorted(best_per_test.values(),
                             key=lambda x: (x.identifiability_rate if not np.isnan(x.identifiability_rate) else 1,
                                           x.power_M1 if not np.isnan(x.power_M1) else 1))

        for i, r in enumerate(underpowered[:5], 1):
            if r.identifiability_rate > 0.8 and r.power_M1 > 0.5:
                continue  # Skip good tests
            f.write(f"### {i}. {r.test_name}\n\n")
            f.write(f"- **Family**: {r.family}\n")
            f.write(f"- **Power (M1)**: {r.power_M1:.2f}\n")
            f.write(f"- **Identifiability**: {r.identifiability_rate:.2f}\n")
            f.write(f"- **Issue**: {get_issue(r)}\n\n")

        # Requirements table
        f.write("## Key Requirements by Test\n\n")
        f.write("| Test | Data Type | Requirement |\n")
        f.write("|------|-----------|-------------|\n")
        for name, r in best_per_test.items():
            f.write(f"| {name} | {r.data_type} | {get_requirement(r)} |\n")

    print(f"\nSaved: {summary_path}")


def get_requirement(result: PowerResult) -> str:
    """Get key data requirement for a test."""
    if result.data_type == 'gaussian':
        return "Needs binned cross-section tables with correlated systematics"
    elif result.data_type == 'poisson':
        if result.nbins > 40:
            return "Needs fine-binned invariant mass spectrum (>40 bins)"
        else:
            return "Needs binned event counts in 2 channels with coherent interference"
    else:
        return "Needs multi-channel amplitude analysis workspace"


def get_issue(result: PowerResult) -> str:
    """Get main issue with an underpowered test."""
    if result.identifiability_rate < 0.5:
        return "Phase not identifiable - multiple degenerate solutions"
    if result.pass_rate < 0.5:
        return "Poor fit quality - model mismatch or underconstrained"
    if result.power_M1 < 0.3:
        return "Insufficient statistics - need higher luminosity/more events"
    if not np.isnan(result.typeI_M0) and result.typeI_M0 > 0.15:
        return "High false positive rate - test not calibrated"
    return "Unknown issue"


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run power study sweep')
    parser.add_argument('--config', default='configs/tests.json',
                       help='Path to test config file')
    parser.add_argument('--output', default='out',
                       help='Output directory')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of MC trials per condition')
    parser.add_argument('--bootstrap', type=int, default=200,
                       help='Number of bootstrap samples')
    parser.add_argument('--levels', type=float, nargs='+',
                       default=[0.5, 1.0, 2.0, 4.0],
                       help='Statistics levels to test')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (fewer trials/bootstrap)')

    args = parser.parse_args()

    if args.quick:
        args.trials = 30
        args.bootstrap = 50
        args.levels = [1.0, 2.0]

    print("="*60)
    print("RANK-1 BOTTLENECK POWER STUDY")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Trials: {args.trials}")
    print(f"Bootstrap: {args.bootstrap}")
    print(f"Stats levels: {args.levels}")
    print(f"Workers: {max(1, cpu_count() - 1)}")
    print("="*60)

    # Run sweep
    results = run_full_sweep(
        config_path=args.config,
        output_dir=args.output,
        n_trials=args.trials,
        n_bootstrap=args.bootstrap,
        stats_levels=args.levels
    )

    # Save results
    save_results(results, args.output)
    generate_summary(results, args.output)

    print("\n" + "="*60)
    print("SWEEP COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
