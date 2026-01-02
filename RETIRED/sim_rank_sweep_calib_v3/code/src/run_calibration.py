#!/usr/bin/env python3
"""
run_calibration.py - Calibration diagnostic runner for Top-3 tests

Runs M0 calibration trials with full diagnostic tracing.
Outputs:
- Per-test trace CSV
- Calibration summary table
- p_boot and Lambda histograms
- KS test results
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from sim_generate import generate_dataset
from sim_fit_v3 import (
    run_calibration_trial, sanity_injection_check,
    write_trace_csv, compute_calibration_summary,
    DEFAULT_BOOTSTRAP, DEFAULT_STARTS
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run calibration diagnostics')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to tests_top3.json')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--trials', type=int, default=200,
                        help='Number of M0 trials per test')
    parser.add_argument('--bootstrap', type=int, default=200,
                        help='Bootstrap replicates per trial')
    parser.add_argument('--starts', type=int, default=120,
                        help='Optimizer starts')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    parser.add_argument('--stats', type=float, default=1.0,
                        help='Statistics level multiplier')
    return parser.parse_args()


def log(msg, logfile=None):
    """Log to stdout and optionally to file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, 'a') as f:
            f.write(line + '\n')


def run_single_trial(args):
    """Run a single calibration trial."""
    test_config, trial_idx, stats_level, n_bootstrap, n_starts = args

    try:
        # Generate M0 dataset
        seed = trial_idx * 1000 + hash(test_config['name']) % 10000
        dataset = generate_dataset(test_config, 'M0', scale_factor=stats_level, seed=seed)

        # Run calibration trial
        result = run_calibration_trial(dataset, n_bootstrap=n_bootstrap, n_starts=n_starts)
        result['test'] = test_config['name']
        result['trial'] = trial_idx

        return result

    except Exception as e:
        return {
            'test': test_config['name'],
            'trial': trial_idx,
            'converged': False,
            'error': str(e)
        }


def plot_pboot_histogram(p_boots, test_name, outdir):
    """Plot p_boot histogram with uniform reference."""
    if not p_boots or len(p_boots) < 5:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(p_boots, bins=20, range=(0, 1), density=True, alpha=0.7,
            color='steelblue', edgecolor='black', label='p_boot')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Uniform(0,1)')

    ax.set_xlabel('p_boot', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{test_name} - p_boot Distribution under M0', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 1)

    filepath = os.path.join(outdir, f'pboot_hist_{test_name.replace(" ", "_").replace("-", "_")}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


def plot_lambda_histogram(lambda_obs_list, test_name, outdir):
    """Plot Lambda_obs histogram with chi2(2) reference."""
    if not lambda_obs_list or len(lambda_obs_list) < 5:
        return

    from scipy.stats import chi2

    fig, ax = plt.subplots(figsize=(8, 6))

    # Histogram
    max_val = min(np.percentile(lambda_obs_list, 99), 15)
    bins = np.linspace(0, max_val, 30)
    ax.hist(lambda_obs_list, bins=bins, density=True, alpha=0.7,
            color='steelblue', edgecolor='black', label=r'$\Lambda_{obs}$')

    # Chi2(2) reference
    x = np.linspace(0, max_val, 200)
    ax.plot(x, chi2.pdf(x, 2), 'r-', linewidth=2, label=r'$\chi^2(2)$')

    ax.set_xlabel(r'$\Lambda$', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{test_name} - Lambda Distribution under M0', fontsize=14)
    ax.legend()

    # Add mean annotation
    ax.axvline(np.mean(lambda_obs_list), color='green', linestyle=':', linewidth=2)
    ax.annotate(f'Mean={np.mean(lambda_obs_list):.2f}',
                xy=(np.mean(lambda_obs_list), ax.get_ylim()[1]*0.9),
                fontsize=10, color='green')

    filepath = os.path.join(outdir, f'lambda_hist_{test_name.replace(" ", "_").replace("-", "_")}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


def run_test_calibration(test_config, n_trials, n_bootstrap, n_starts,
                         stats_level, n_workers, outdir, logfile):
    """Run calibration for one test."""
    test_name = test_config['name']
    log(f"Starting calibration for {test_name}: {n_trials} trials", logfile)

    # Sanity injection check first
    sanity_ok, sanity_msg = sanity_injection_check(test_config)
    log(f"  Sanity check: {sanity_msg}", logfile)
    if not sanity_ok:
        log(f"  FATAL: Sanity check failed for {test_name}", logfile)
        return None

    # Prepare trial arguments
    trial_args = [(test_config, i, stats_level, n_bootstrap, n_starts)
                  for i in range(n_trials)]

    # Run in parallel
    traces = []
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_trial, trial_args)):
            traces.append(result)
            if (i + 1) % 20 == 0 or (i + 1) == n_trials:
                log(f"  [{test_name}] {i+1}/{n_trials} completed", logfile)

    # Write trace CSV
    diag_dir = os.path.join(os.path.dirname(outdir), 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    trace_path = os.path.join(diag_dir, f'trace_{test_name.replace(" ", "_").replace("-", "_")}_m0_1x.csv')
    write_trace_csv(traces, trace_path)
    log(f"  Saved trace: {trace_path}", logfile)

    # Compute summary
    summary = compute_calibration_summary(traces)
    summary['test'] = test_name

    log(f"  [{test_name}] Summary:", logfile)
    log(f"    Pass rate: {summary['pass_rate']:.3f} ({summary['n_pass']}/{summary['n_total']})", logfile)
    log(f"    Type I: {summary['type_i']:.3f} +/- {summary['type_i_se']:.3f} (target: 0.05)", logfile)
    log(f"    Lambda_obs mean: {summary['lambda_obs_mean']:.3f}, median: {summary.get('lambda_obs_median', np.nan):.3f}", logfile)
    log(f"    Lambda_boot mean: {summary['lambda_boot_mean']:.3f}", logfile)
    log(f"    KS test: D={summary['ks_stat']:.3f}, p={summary['ks_pval']:.4f}", logfile)

    # Generate plots
    pboot_plot = plot_pboot_histogram(summary['p_boots'], test_name, outdir)
    lambda_plot = plot_lambda_histogram(summary['lambda_obs_list'], test_name, outdir)

    if pboot_plot:
        log(f"  Saved plot: {pboot_plot}", logfile)
    if lambda_plot:
        log(f"  Saved plot: {lambda_plot}", logfile)

    return summary


def check_calibration_pass(summaries):
    """Check if calibration passes for all tests."""
    all_pass = True
    failures = []

    for s in summaries:
        test = s['test']
        type_i = s['type_i']
        pass_rate = s['pass_rate']

        # Criteria
        type_i_ok = 0.02 <= type_i <= 0.08 if not np.isnan(type_i) else False
        pass_rate_ok = pass_rate >= 0.8

        if not type_i_ok:
            failures.append(f"{test}: Type I = {type_i:.3f} (expected [0.02, 0.08])")
            all_pass = False
        if not pass_rate_ok:
            failures.append(f"{test}: Pass rate = {pass_rate:.3f} (expected >= 0.8)")
            all_pass = False

    return all_pass, failures


def write_calibration_table(summaries, outdir):
    """Write calibration summary table."""
    md_path = os.path.join(outdir, 'CALIBRATION_DIAG_TABLE.md')
    csv_path = os.path.join(outdir, 'CALIBRATION_DIAG_TABLE.csv')

    # Markdown
    with open(md_path, 'w') as f:
        f.write("# Calibration Diagnostic Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary Table\n\n")
        f.write("| Test | Pass Rate | Type I | Type I SE | Lambda Mean | Lambda Median | Lambda Boot Mean | KS Stat | KS p-val | Status |\n")
        f.write("|------|-----------|--------|-----------|-------------|---------------|------------------|---------|----------|--------|\n")

        for s in summaries:
            type_i = s['type_i']
            status = "PASS" if (0.02 <= type_i <= 0.08 and s['pass_rate'] >= 0.8) else "FAIL"
            f.write(f"| {s['test']} | {s['pass_rate']:.3f} | {type_i:.3f} | {s['type_i_se']:.3f} | ")
            f.write(f"{s['lambda_obs_mean']:.3f} | {s.get('lambda_obs_median', np.nan):.3f} | ")
            f.write(f"{s['lambda_boot_mean']:.3f} | {s['ks_stat']:.3f} | {s['ks_pval']:.4f} | {status} |\n")

        f.write("\n## Criteria\n\n")
        f.write("- Type I error: [0.02, 0.08]\n")
        f.write("- Pass rate: >= 0.8\n")
        f.write("- KS test: Approximate check against Uniform(0,1)\n")

    # CSV
    with open(csv_path, 'w') as f:
        f.write("test,pass_rate,type_i,type_i_se,lambda_obs_mean,lambda_obs_median,lambda_boot_mean,ks_stat,ks_pval,n_pass,n_total\n")
        for s in summaries:
            f.write(f"{s['test']},{s['pass_rate']:.4f},{s['type_i']:.4f},{s['type_i_se']:.4f},")
            f.write(f"{s['lambda_obs_mean']:.4f},{s.get('lambda_obs_median', np.nan):.4f},")
            f.write(f"{s['lambda_boot_mean']:.4f},{s['ks_stat']:.4f},{s['ks_pval']:.4f},")
            f.write(f"{s['n_pass']},{s['n_total']}\n")

    return md_path, csv_path


def write_calibration_fail(failures, outdir):
    """Write calibration failure report."""
    path = os.path.join(outdir, 'CALIBRATION_FAIL.md')
    with open(path, 'w') as f:
        f.write("# Calibration FAILED\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Failures\n\n")
        for fail in failures:
            f.write(f"- {fail}\n")
        f.write("\n## Recommendations\n\n")
        f.write("1. Check bootstrap generation matches likelihood assumptions\n")
        f.write("2. Verify optimizer convergence with more restarts\n")
        f.write("3. Check for model mismatch between generator and fitter\n")
    return path


def main():
    args = parse_args()

    # Setup
    os.makedirs(args.outdir, exist_ok=True)
    logdir = os.path.join(os.path.dirname(args.outdir), 'logs')
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, 'calibration.log')

    n_workers = args.workers or max(1, cpu_count() - 1)

    log("=" * 70, logfile)
    log("CALIBRATION DIAGNOSTICS - v3 with all fixes", logfile)
    log("=" * 70, logfile)
    log(f"Config: {args.config}", logfile)
    log(f"Trials: {args.trials}, Bootstrap: {args.bootstrap}, Starts: {args.starts}", logfile)
    log(f"Stats level: {args.stats}x", logfile)
    log(f"Workers: {n_workers}", logfile)
    log("", logfile)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    tests = config['tests']
    log(f"Tests: {[t['name'] for t in tests]}", logfile)
    log("", logfile)

    # Run calibration for each test
    summaries = []
    for test in tests:
        summary = run_test_calibration(
            test, args.trials, args.bootstrap, args.starts,
            args.stats, n_workers, args.outdir, logfile
        )
        if summary:
            summaries.append(summary)
        log("", logfile)

    # Write summary table
    md_path, csv_path = write_calibration_table(summaries, args.outdir)
    log(f"Saved: {md_path}", logfile)
    log(f"Saved: {csv_path}", logfile)

    # Check if calibration passed
    all_pass, failures = check_calibration_pass(summaries)

    log("=" * 70, logfile)
    if all_pass:
        log("CALIBRATION PASSED for all tests!", logfile)
    else:
        log("CALIBRATION FAILED", logfile)
        fail_path = write_calibration_fail(failures, args.outdir)
        log(f"See: {fail_path}", logfile)
        for fail in failures:
            log(f"  - {fail}", logfile)
    log("=" * 70, logfile)

    # Print final summary
    log("\nFINAL SUMMARY:", logfile)
    for s in summaries:
        log(f"  {s['test']}: Type I = {s['type_i']:.3f}, Pass Rate = {s['pass_rate']:.3f}", logfile)


if __name__ == '__main__':
    main()
