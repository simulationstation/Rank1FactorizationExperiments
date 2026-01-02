#!/usr/bin/env python3
"""
smoke_diagnostics.py - Targeted diagnostics for calibration failures

Runs specific diagnostic tests to identify root cause of calibration failures:
A) Optimization sufficiency test
B) Bootstrap refit symmetry test
C) Nestedness / Lambda sanity check
D) Noise model match (Y-states specific)
"""

import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(__file__))

from sim_generate import generate_dataset
from sim_fit_v3 import (
    fit_joint_unconstrained, fit_joint_constrained,
    generate_bootstrap_data, run_calibration_trial,
    DEFAULT_STARTS
)


def log(msg: str, logfile: Optional[str] = None):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, 'a') as f:
            f.write(line + '\n')


def select_representative_trials(results: List[Dict]) -> List[Dict]:
    """Select 3 representative trials for diagnostics."""
    passed = [r for r in results if r.get('gates') == 'PASS' and 'dataset' in r]
    if len(passed) < 3:
        return passed

    # Sort by Lambda_obs
    passed.sort(key=lambda r: r.get('lambda_obs', 0))

    # Select: median, max outlier, random
    mid_idx = len(passed) // 2
    representatives = [
        ('median', passed[mid_idx]),
        ('outlier', passed[-1]),  # Max Lambda_obs
        ('random', passed[np.random.randint(0, len(passed))])
    ]

    return representatives


def run_optimization_sufficiency_test(dataset: Dict, base_starts: int = 120) -> Dict:
    """
    Test A: Check if optimizer is finding global minimum.
    Refit with increasing starts and check if NLL keeps improving.
    """
    starts_levels = [30, 80, 150, 300]
    results = {'unconstrained': {}, 'constrained': {}}

    for n_starts in starts_levels:
        # Unconstrained fit
        fit_unc = fit_joint_unconstrained(dataset, n_starts=n_starts)
        results['unconstrained'][n_starts] = {
            'nll': fit_unc.get('nll', np.inf),
            'converged': fit_unc.get('converged', False)
        }

        # Constrained fit
        fit_con = fit_joint_constrained(dataset, n_starts=n_starts)
        results['constrained'][n_starts] = {
            'nll': fit_con.get('nll', np.inf),
            'converged': fit_con.get('converged', False)
        }

    # Check if NLL improves past base_starts
    base_nll_unc = results['unconstrained'].get(base_starts, {}).get('nll', np.inf)
    high_nll_unc = results['unconstrained'].get(300, {}).get('nll', np.inf)

    base_nll_con = results['constrained'].get(base_starts, {}).get('nll', np.inf)
    high_nll_con = results['constrained'].get(300, {}).get('nll', np.inf)

    improvement_unc = base_nll_unc - high_nll_unc
    improvement_con = base_nll_con - high_nll_con

    results['diagnosis'] = {
        'unc_improvement': improvement_unc,
        'con_improvement': improvement_con,
        'under_optimized': improvement_unc > 0.1 or improvement_con > 0.1,
        'recommendation': 'Increase starts' if improvement_unc > 0.1 or improvement_con > 0.1 else 'Starts adequate'
    }

    return results


def run_bootstrap_symmetry_test(dataset: Dict, n_bootstrap: int = 40,
                                 low_starts: int = 50, high_starts: int = 300) -> Dict:
    """
    Test B: Check if bootstrap refit is under-optimized.
    Compare Lambda_boot distributions with low vs high optimization budget.
    """
    # Get fitted predictions for bootstrap generation
    fit_unc = fit_joint_unconstrained(dataset, n_starts=high_starts)
    fit_con = fit_joint_constrained(dataset, n_starts=high_starts)

    if not fit_unc.get('converged') or not fit_con.get('converged'):
        return {'error': 'Fit did not converge'}

    ch_a = dataset['channelA']
    ch_b = dataset['channelB']

    lambda_low = []
    lambda_high = []

    rng = np.random.default_rng(42)

    for i in range(n_bootstrap):
        # Generate bootstrap data
        boot_a = generate_bootstrap_data(ch_a, fit_con.get('y_pred_a'), rng)
        boot_b = generate_bootstrap_data(ch_b, fit_con.get('y_pred_b'), rng)

        boot_dataset = {
            'channelA': boot_a,
            'channelB': boot_b,
            'bw_params': dataset['bw_params']
        }

        # Low-budget refit
        fit_unc_low = fit_joint_unconstrained(boot_dataset, n_starts=low_starts)
        fit_con_low = fit_joint_constrained(boot_dataset, n_starts=low_starts)
        if fit_unc_low.get('converged') and fit_con_low.get('converged'):
            lam_low = 2 * (fit_con_low['nll'] - fit_unc_low['nll'])
            lambda_low.append(max(0, lam_low))

        # High-budget refit
        fit_unc_high = fit_joint_unconstrained(boot_dataset, n_starts=high_starts)
        fit_con_high = fit_joint_constrained(boot_dataset, n_starts=high_starts)
        if fit_unc_high.get('converged') and fit_con_high.get('converged'):
            lam_high = 2 * (fit_con_high['nll'] - fit_unc_high['nll'])
            lambda_high.append(max(0, lam_high))

    results = {
        'lambda_low_mean': np.mean(lambda_low) if lambda_low else np.nan,
        'lambda_low_median': np.median(lambda_low) if lambda_low else np.nan,
        'lambda_high_mean': np.mean(lambda_high) if lambda_high else np.nan,
        'lambda_high_median': np.median(lambda_high) if lambda_high else np.nan,
        'n_low': len(lambda_low),
        'n_high': len(lambda_high),
    }

    # If high-budget Lambda is materially larger, under-optimization is the cause
    if lambda_low and lambda_high:
        shift = results['lambda_high_mean'] - results['lambda_low_mean']
        results['shift'] = shift
        results['under_optimized'] = shift > 0.1
        results['recommendation'] = 'Increase bootstrap starts' if shift > 0.1 else 'Bootstrap starts adequate'
    else:
        results['shift'] = np.nan
        results['under_optimized'] = False
        results['recommendation'] = 'Could not compare'

    return results


def run_nestedness_test(dataset: Dict, n_bootstrap: int = 20, n_starts: int = 150) -> Dict:
    """
    Test C: Check nestedness constraint NLL_unc <= NLL_con.
    Violations indicate optimizer not finding global minima.
    """
    violations = []
    lambda_values = []

    # Test on observed data
    fit_unc = fit_joint_unconstrained(dataset, n_starts=n_starts)
    fit_con = fit_joint_constrained(dataset, n_starts=n_starts)

    if fit_unc.get('converged') and fit_con.get('converged'):
        nll_unc = fit_unc['nll']
        nll_con = fit_con['nll']
        if nll_unc > nll_con + 1e-6:
            violations.append(('observed', nll_unc - nll_con))
        lambda_values.append(2 * max(0, nll_con - nll_unc))

    # Test on bootstrap replicates
    ch_a = dataset['channelA']
    ch_b = dataset['channelB']
    rng = np.random.default_rng(123)

    for i in range(n_bootstrap):
        boot_a = generate_bootstrap_data(ch_a, fit_con.get('y_pred_a'), rng)
        boot_b = generate_bootstrap_data(ch_b, fit_con.get('y_pred_b'), rng)

        boot_dataset = {
            'channelA': boot_a,
            'channelB': boot_b,
            'bw_params': dataset['bw_params']
        }

        fit_unc_b = fit_joint_unconstrained(boot_dataset, n_starts=n_starts)
        fit_con_b = fit_joint_constrained(boot_dataset, n_starts=n_starts)

        if fit_unc_b.get('converged') and fit_con_b.get('converged'):
            nll_unc_b = fit_unc_b['nll']
            nll_con_b = fit_con_b['nll']
            if nll_unc_b > nll_con_b + 1e-6:
                violations.append((f'bootstrap_{i}', nll_unc_b - nll_con_b))
            lambda_values.append(2 * max(0, nll_con_b - nll_unc_b))

    results = {
        'n_tests': 1 + n_bootstrap,
        'n_violations': len(violations),
        'violation_rate': len(violations) / (1 + n_bootstrap),
        'violations': violations[:5],  # First 5
        'lambda_mean': np.mean(lambda_values) if lambda_values else np.nan,
        'lambda_median': np.median(lambda_values) if lambda_values else np.nan,
    }

    results['has_issue'] = results['violation_rate'] > 0.05
    results['recommendation'] = 'Optimizer not reliable' if results['has_issue'] else 'Nestedness OK'

    return results


def run_noise_model_test(dataset: Dict, test_config: Dict) -> Dict:
    """
    Test D: Check noise model match (Y-states specific).
    Verify generation uses stat-only noise when likelihood has nuisance priors.
    """
    ch_a = dataset['channelA']
    ch_b = dataset['channelB']

    results = {
        'channel_a': {},
        'channel_b': {},
    }

    for name, ch in [('channel_a', ch_a), ('channel_b', ch_b)]:
        results[name] = {
            'type': ch.get('type', 'unknown'),
            'has_stat_error': 'stat_error' in ch,
            'has_syst_error': 'syst_error' in ch,
            'sigma_mean': np.mean(ch['sigma']) if 'sigma' in ch else np.nan,
            'stat_error_mean': np.mean(ch['stat_error']) if 'stat_error' in ch else np.nan,
            'syst_error_mean': np.mean(ch['syst_error']) if 'syst_error' in ch else np.nan,
        }

        if 'stat_error' in ch and 'sigma' in ch:
            ratio = np.mean(ch['sigma'] / ch['stat_error'])
            results[name]['sigma_to_stat_ratio'] = ratio
            results[name]['has_systematics'] = ratio > 1.1

    # Check config
    test_type = test_config.get('type', 'unknown')
    results['config_type'] = test_type
    results['has_correlated_syst'] = test_config.get('channelA', {}).get('correlated_syst', False)

    # Diagnosis
    if test_type == 'gaussian' and results['has_correlated_syst']:
        # For Gaussian with correlated systematics, bootstrap should use stat_error only
        if results['channel_a'].get('has_stat_error') and results['channel_b'].get('has_stat_error'):
            results['recommendation'] = 'Noise model correctly separates stat/syst'
            results['has_issue'] = False
        else:
            results['recommendation'] = 'Missing stat_error separation - potential mismatch'
            results['has_issue'] = True
    else:
        results['recommendation'] = 'Not applicable (Poisson or no systematics)'
        results['has_issue'] = False

    return results


def write_diagnostic_report(test_name: str, trial_label: str, trial: Dict,
                            opt_test: Dict, boot_test: Dict, nest_test: Dict,
                            noise_test: Dict, diagdir: str) -> str:
    """Write diagnostic report to markdown file."""
    filename = f"DIAG_{test_name.replace('-', '_')}_{trial_label}.md"
    filepath = os.path.join(diagdir, filename)

    with open(filepath, 'w') as f:
        f.write(f"# Diagnostic Report: {test_name} (Trial: {trial_label})\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Trial Info\n\n")
        f.write(f"- Lambda_obs: {trial.get('lambda_obs', np.nan):.4f}\n")
        f.write(f"- p_boot: {trial.get('p_boot', np.nan):.4f}\n")
        f.write(f"- chi2_dof_a: {trial.get('chi2_dof_a', np.nan):.4f}\n")
        f.write(f"- chi2_dof_b: {trial.get('chi2_dof_b', np.nan):.4f}\n")
        f.write(f"- gates: {trial.get('gates', 'unknown')}\n\n")

        f.write("## A) Optimization Sufficiency Test\n\n")
        f.write("NLL by starts level:\n\n")
        f.write("| Starts | NLL (Unconstrained) | NLL (Constrained) |\n")
        f.write("|--------|---------------------|-------------------|\n")
        for starts in [30, 80, 150, 300]:
            nll_unc = opt_test['unconstrained'].get(starts, {}).get('nll', np.nan)
            nll_con = opt_test['constrained'].get(starts, {}).get('nll', np.nan)
            f.write(f"| {starts} | {nll_unc:.4f} | {nll_con:.4f} |\n")
        f.write(f"\n**Diagnosis**: {opt_test['diagnosis']['recommendation']}\n")
        f.write(f"- Improvement (unc): {opt_test['diagnosis']['unc_improvement']:.4f}\n")
        f.write(f"- Improvement (con): {opt_test['diagnosis']['con_improvement']:.4f}\n")
        f.write(f"- Under-optimized: {opt_test['diagnosis']['under_optimized']}\n\n")

        f.write("## B) Bootstrap Refit Symmetry Test\n\n")
        f.write(f"| Budget | Lambda Mean | Lambda Median | N |\n")
        f.write(f"|--------|-------------|---------------|---|\n")
        f.write(f"| Low (50) | {boot_test['lambda_low_mean']:.4f} | {boot_test['lambda_low_median']:.4f} | {boot_test['n_low']} |\n")
        f.write(f"| High (300) | {boot_test['lambda_high_mean']:.4f} | {boot_test['lambda_high_median']:.4f} | {boot_test['n_high']} |\n")
        f.write(f"\n**Shift**: {boot_test.get('shift', np.nan):.4f}\n")
        f.write(f"**Diagnosis**: {boot_test['recommendation']}\n")
        f.write(f"- Under-optimized: {boot_test.get('under_optimized', False)}\n\n")

        f.write("## C) Nestedness / Lambda Sanity Test\n\n")
        f.write(f"- Tests run: {nest_test['n_tests']}\n")
        f.write(f"- Violations (NLL_unc > NLL_con): {nest_test['n_violations']} ({nest_test['violation_rate']:.1%})\n")
        f.write(f"- Lambda mean: {nest_test['lambda_mean']:.4f}\n")
        f.write(f"- Lambda median: {nest_test['lambda_median']:.4f}\n")
        f.write(f"\n**Diagnosis**: {nest_test['recommendation']}\n\n")

        f.write("## D) Noise Model Match Test\n\n")
        f.write(f"- Config type: {noise_test['config_type']}\n")
        f.write(f"- Has correlated systematics: {noise_test['has_correlated_syst']}\n\n")
        for ch_name in ['channel_a', 'channel_b']:
            ch = noise_test[ch_name]
            f.write(f"**{ch_name}**:\n")
            f.write(f"- Type: {ch['type']}\n")
            f.write(f"- sigma mean: {ch['sigma_mean']:.4f}\n")
            f.write(f"- stat_error mean: {ch['stat_error_mean']:.4f}\n")
            if 'sigma_to_stat_ratio' in ch:
                f.write(f"- sigma/stat ratio: {ch['sigma_to_stat_ratio']:.2f}\n")
            f.write("\n")
        f.write(f"**Diagnosis**: {noise_test['recommendation']}\n")
        f.write(f"- Has issue: {noise_test.get('has_issue', False)}\n\n")

        # Summary
        f.write("## Summary: Root Cause Analysis\n\n")
        issues = []
        if opt_test['diagnosis']['under_optimized']:
            issues.append("PATCH 1: Main fit under-optimized")
        if boot_test.get('under_optimized', False):
            issues.append("PATCH 1: Bootstrap refit under-optimized")
        if nest_test['has_issue']:
            issues.append("PATCH 1: Nestedness violations (optimizer unreliable)")
        if noise_test.get('has_issue', False):
            issues.append("PATCH 3: Noise model mismatch")

        if issues:
            f.write("**Issues detected:**\n")
            for issue in issues:
                f.write(f"- {issue}\n")
        else:
            f.write("**No obvious issues detected** - may need deeper investigation.\n")

    return filepath


def run_targeted_diagnostics(test_config: Dict, results: List[Dict],
                             diagdir: str, base_starts: int = 120,
                             logfile: Optional[str] = None) -> Dict:
    """Run all targeted diagnostics on representative trials."""
    test_name = test_config['name']
    log(f"Running targeted diagnostics for {test_name}...", logfile)

    # Select representative trials
    representatives = select_representative_trials(results)
    if not representatives:
        log(f"  No valid trials for diagnostics", logfile)
        return {'error': 'No valid trials'}

    all_diag_results = {}

    for label, trial in representatives:
        log(f"  Diagnosing {label} trial (Lambda_obs={trial.get('lambda_obs', np.nan):.4f})", logfile)

        dataset = trial.get('dataset')
        if dataset is None:
            log(f"    No dataset available for trial", logfile)
            continue

        # Run all diagnostic tests
        log(f"    A) Optimization sufficiency test...", logfile)
        opt_test = run_optimization_sufficiency_test(dataset, base_starts)

        log(f"    B) Bootstrap refit symmetry test...", logfile)
        boot_test = run_bootstrap_symmetry_test(dataset, n_bootstrap=20, low_starts=50, high_starts=300)

        log(f"    C) Nestedness test...", logfile)
        nest_test = run_nestedness_test(dataset, n_bootstrap=15, n_starts=150)

        log(f"    D) Noise model test...", logfile)
        noise_test = run_noise_model_test(dataset, test_config)

        # Write report
        report_path = write_diagnostic_report(
            test_name, label, trial,
            opt_test, boot_test, nest_test, noise_test,
            diagdir
        )
        log(f"    Wrote: {report_path}", logfile)

        all_diag_results[label] = {
            'optimization': opt_test,
            'bootstrap': boot_test,
            'nestedness': nest_test,
            'noise_model': noise_test,
        }

    return all_diag_results


if __name__ == '__main__':
    print("This module is meant to be imported by smoke_calibrate.py")
