#!/usr/bin/env python3
"""
run_diagnostics.py - Diagnostic batch for calibration analysis

Runs 50 trials per test at 1.0x stats with full diagnostic logging.
Outputs: diagnostics/calibration_trace_<test>.csv
"""

import os
import sys
import json
import csv
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from sim_generate import generate_dataset
from sim_fit import (
    run_full_fit, set_diagnostic_mode, get_diagnostic_log,
    DOF_DIFF, DEFAULT_BOOTSTRAP, DEFAULT_STARTS
)

# Configuration
N_TRIALS = 50
N_BOOTSTRAP = 200  # As specified
N_STARTS = 80      # As specified
STATS_LEVEL = 1.0
N_WORKERS = min(31, cpu_count() - 1)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
DIAG_DIR = os.path.join(PROJECT_ROOT, 'diagnostics')
CONFIG_PATH = os.path.join(SCRIPT_DIR, '..', 'configs', 'tests_top3.json')  # code/configs/
LOG_PATH = os.path.join(PROJECT_ROOT, 'logs', 'diagnostics.log')

os.makedirs(DIAG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg):
    """Log to both stdout and file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def run_single_trial(args):
    """Run a single diagnostic trial."""
    test, trial_idx, stats = args

    try:
        # Generate M0 dataset (rank-1 true - for Type I calibration check)
        seed = trial_idx * 1000 + hash(test['name']) % 10000
        dataset = generate_dataset(test, mechanism='M0', scale_factor=stats, seed=seed)

        # Run full fit with diagnostics
        set_diagnostic_mode(True, DIAG_DIR, test['name'], trial_idx)
        result = run_full_fit(dataset, n_bootstrap=N_BOOTSTRAP, n_restarts=N_STARTS)
        diag_log = get_diagnostic_log()
        set_diagnostic_mode(False)

        # Build row
        row = {
            'test': test['name'],
            'trial': trial_idx,
            'stats_level': stats,
            'converged': result.converged,
            'Lambda_obs': result.Lambda,
            'p_boot': result.p_boot,
            'p_wilks': result.p_wilks,
            'verdict': result.verdict,
            'gates': result.gates,
            'chi2_dof_a': result.chi2_dof_a,
            'chi2_dof_b': result.chi2_dof_b,
            'identifiable': result.identifiable,
            'r_spread': result.r_spread,
            'phi_spread': result.phi_spread,
            'n_bootstrap': result.n_bootstrap,
            'dof_diff': DOF_DIFF,
            'n_restarts': N_STARTS
        }

        # Add bootstrap diagnostics from log
        for key, val in diag_log:
            if key not in row:
                row[key] = val

        # Extract rejection (p < 0.05)
        row['rejected'] = 1 if (result.p_boot is not None and not np.isnan(result.p_boot) and result.p_boot < 0.05) else 0

        return row

    except Exception as e:
        return {
            'test': test['name'],
            'trial': trial_idx,
            'stats_level': stats,
            'error': str(e),
            'converged': False
        }


def run_test_diagnostics(test: dict):
    """Run diagnostic trials for one test."""
    test_name = test['name']
    log(f"Starting diagnostics for {test_name}: {N_TRIALS} trials")

    # Prepare trial args
    trial_args = [(test, i, STATS_LEVEL) for i in range(N_TRIALS)]

    # Run in parallel
    results = []
    with Pool(N_WORKERS) as pool:
        for i, row in enumerate(pool.imap_unordered(run_single_trial, trial_args)):
            results.append(row)
            if (i + 1) % 10 == 0 or (i + 1) == N_TRIALS:
                log(f"  [{test_name}] {i+1}/{N_TRIALS} completed")

    # Save to CSV
    csv_path = os.path.join(DIAG_DIR, f'calibration_trace_{test_name.replace(" ", "_").replace("-", "_")}.csv')

    # Get all columns
    all_keys = set()
    for row in results:
        all_keys.update(row.keys())
    all_keys = sorted(all_keys)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    log(f"Saved: {csv_path}")

    # Compute summary stats
    valid = [r for r in results if r.get('converged', False)]
    rejected = [r for r in valid if r.get('rejected', 0) == 1]

    type_i = len(rejected) / len(valid) if valid else float('nan')
    lambda_vals = [r['Lambda_obs'] for r in valid if 'Lambda_obs' in r and not np.isnan(r.get('Lambda_obs', float('nan')))]

    log(f"  [{test_name}] Summary:")
    log(f"    Valid trials: {len(valid)}/{N_TRIALS}")
    log(f"    Type I error: {type_i:.3f} (target: 0.05)")
    if lambda_vals:
        log(f"    Lambda mean: {np.mean(lambda_vals):.3f}, std: {np.std(lambda_vals):.3f}")
        log(f"    Lambda median: {np.median(lambda_vals):.3f}")

    return {
        'test': test_name,
        'n_valid': len(valid),
        'n_trials': N_TRIALS,
        'type_i': type_i,
        'lambda_mean': np.mean(lambda_vals) if lambda_vals else float('nan'),
        'lambda_std': np.std(lambda_vals) if lambda_vals else float('nan'),
        'lambda_median': np.median(lambda_vals) if lambda_vals else float('nan')
    }


def main():
    log("=" * 70)
    log("CALIBRATION DIAGNOSTICS - 50 trials per test @ 1.0x stats")
    log("=" * 70)
    log(f"n_bootstrap = {N_BOOTSTRAP}, n_restarts = {N_STARTS}, dof_diff = {DOF_DIFF}")
    log(f"Workers: {N_WORKERS}")
    log("")

    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    tests = config['tests']
    log(f"Tests: {[t['name'] for t in tests]}")
    log("")

    summaries = []
    for test in tests:
        summary = run_test_diagnostics(test)
        summaries.append(summary)
        log("")

    # Generate summary report
    log("=" * 70)
    log("DIAGNOSTIC SUMMARY")
    log("=" * 70)

    report_path = os.path.join(DIAG_DIR, 'DIAGNOSTIC_SUMMARY.md')
    with open(report_path, 'w') as f:
        f.write("# Calibration Diagnostic Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"- Trials per test: {N_TRIALS}\n")
        f.write(f"- Stats level: {STATS_LEVEL}x\n")
        f.write(f"- Bootstrap: {N_BOOTSTRAP}\n")
        f.write(f"- Optimizer restarts: {N_STARTS}\n")
        f.write(f"- dof_diff: {DOF_DIFF}\n\n")

        f.write("## Results\n\n")
        f.write("| Test | Valid/Total | Type I | Target | Lambda Mean | Lambda Std |\n")
        f.write("|------|-------------|--------|--------|-------------|------------|\n")

        for s in summaries:
            status = "PASS" if abs(s['type_i'] - 0.05) < 0.03 else "CHECK"
            f.write(f"| {s['test']} | {s['n_valid']}/{s['n_trials']} | {s['type_i']:.3f} | 0.05 | {s['lambda_mean']:.3f} | {s['lambda_std']:.3f} |\n")
            log(f"{s['test']}: Type I = {s['type_i']:.3f}, Lambda = {s['lambda_mean']:.3f} +/- {s['lambda_std']:.3f}")

        f.write("\n## Interpretation\n\n")
        f.write("- Type I error should be ~0.05 (within 2-8% for 50 trials)\n")
        f.write("- Lambda distribution under H0 should follow chi2(2)\n")
        f.write("- Expected chi2(2) mean = 2.0, median = 1.386\n\n")

        # Check for issues
        f.write("## Calibration Status\n\n")
        for s in summaries:
            if s['type_i'] < 0.02:
                f.write(f"- **{s['test']}**: TOO CONSERVATIVE (Type I = {s['type_i']:.3f} < 0.02)\n")
            elif s['type_i'] > 0.10:
                f.write(f"- **{s['test']}**: INFLATED (Type I = {s['type_i']:.3f} > 0.10)\n")
            else:
                f.write(f"- **{s['test']}**: OK (Type I = {s['type_i']:.3f})\n")

    log(f"\nSaved: {report_path}")
    log("=" * 70)
    log("DIAGNOSTICS COMPLETE")
    log("=" * 70)


if __name__ == '__main__':
    main()
