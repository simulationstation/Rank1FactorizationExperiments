#!/usr/bin/env python3
"""
smoke_calibrate.py - Fast early-stopping calibration smoke test

Runs small number of trials with early stopping to quickly identify calibration failures.
If failure detected, runs targeted diagnostics to identify root cause.
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
from scipy import stats
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))

from sim_generate import generate_dataset
from sim_fit_v3 import run_calibration_trial, DEFAULT_BOOTSTRAP, DEFAULT_STARTS


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for proportion."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = successes / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0, center - margin), min(1, center + margin))


def log(msg: str, logfile: Optional[str] = None):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, 'a') as f:
            f.write(line + '\n')


def run_single_trial(args) -> Dict:
    """Run a single calibration trial."""
    test_config, trial_idx, n_bootstrap, n_starts = args
    try:
        seed = trial_idx * 1000 + hash(test_config['name']) % 10000
        dataset = generate_dataset(test_config, 'M0', scale_factor=1.0, seed=seed)
        result = run_calibration_trial(dataset, n_bootstrap=n_bootstrap, n_starts=n_starts)
        result['test'] = test_config['name']
        result['trial'] = trial_idx
        result['dataset'] = dataset  # Keep for diagnostics
        return result
    except Exception as e:
        return {
            'test': test_config['name'],
            'trial': trial_idx,
            'converged': False,
            'gates': 'ERROR',
            'error': str(e)
        }


class SmokeCalibrator:
    """Early-stopping smoke calibration with diagnostics."""

    def __init__(self, config_path: str, outdir: str, n_trials: int = 25,
                 n_bootstrap: int = 80, n_starts: int = 120, n_workers: int = None):
        self.config_path = config_path
        self.outdir = outdir
        self.n_trials = n_trials
        self.n_bootstrap = n_bootstrap
        self.n_starts = n_starts
        self.n_workers = n_workers or max(1, cpu_count() - 1)

        # Load config
        with open(config_path) as f:
            self.config = json.load(f)

        # Logging
        self.logdir = os.path.join(os.path.dirname(outdir), 'logs')
        os.makedirs(self.logdir, exist_ok=True)
        self.logfile = os.path.join(self.logdir, 'smoke.log')

        # Diagnostics dir
        self.diagdir = os.path.join(os.path.dirname(outdir), 'diagnostics')
        os.makedirs(self.diagdir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)

    def get_test_config(self, test_name: str) -> Dict:
        """Get test config by name."""
        name_map = {
            'y_states': 'Y-states',
            'di_charmonium': 'Di-charmonium',
            'zc_like': 'Zc-like'
        }
        target = name_map.get(test_name.lower(), test_name)
        for test in self.config['tests']:
            if test['name'] == target:
                return test
        raise ValueError(f"Test not found: {test_name}")

    def check_early_stop(self, results: List[Dict], trial_num: int) -> Tuple[bool, str]:
        """Check early stopping conditions."""
        passed = [r for r in results if r.get('gates') == 'PASS']
        n_pass = len(passed)
        n_total = len(results)

        # Condition 1: Structural model mismatch (pass rate too low)
        if trial_num >= 15 and n_pass < 0.8 * n_total:
            return True, f"FAIL_EARLY: Pass rate {n_pass}/{n_total} < 80% by trial {trial_num}"

        if n_pass < 15:
            return False, "Collecting more trials..."

        # Compute Type I among passed trials
        p_boots = [r.get('p_boot') for r in passed if r.get('p_boot') is not None]
        if not p_boots:
            return False, "No valid p_boot values yet"

        rejections = sum(1 for p in p_boots if p < 0.05)
        type_i = rejections / len(p_boots)
        ci_low, ci_high = wilson_ci(rejections, len(p_boots))

        # Condition 2: Wilson CI excludes [0.02, 0.08]
        if ci_low > 0.08:
            return True, f"FAIL_EARLY: Type I CI [{ci_low:.3f}, {ci_high:.3f}] above 0.08"
        if ci_high < 0.02:
            return True, f"FAIL_EARLY: Type I CI [{ci_low:.3f}, {ci_high:.3f}] below 0.02"

        # Condition 3: KS test (only if enough samples)
        if n_pass >= 20:
            ks_stat, ks_pval = stats.kstest(p_boots, 'uniform')
            if ks_pval < 1e-3:
                return True, f"FAIL_EARLY: KS p-value {ks_pval:.6f} < 1e-3"

        return False, "Continuing..."

    def compute_summary(self, results: List[Dict]) -> Dict:
        """Compute calibration summary statistics."""
        passed = [r for r in results if r.get('gates') == 'PASS']
        n_pass = len(passed)
        n_total = len(results)

        summary = {
            'n_total': n_total,
            'n_pass': n_pass,
            'pass_rate': n_pass / n_total if n_total > 0 else 0,
        }

        if n_pass == 0:
            summary.update({
                'type_i': np.nan,
                'type_i_ci': (np.nan, np.nan),
                'ks_stat': np.nan,
                'ks_pval': np.nan,
                'lambda_obs_mean': np.nan,
                'lambda_obs_median': np.nan,
                'lambda_boot_mean': np.nan,
            })
            return summary

        p_boots = [r.get('p_boot') for r in passed if r.get('p_boot') is not None]
        lambda_obs = [r.get('lambda_obs', np.nan) for r in passed]
        lambda_boot_means = [r.get('lambda_boot_mean', np.nan) for r in passed]

        rejections = sum(1 for p in p_boots if p is not None and p < 0.05)
        n_valid = len([p for p in p_boots if p is not None])

        type_i = rejections / n_valid if n_valid > 0 else np.nan
        ci = wilson_ci(rejections, n_valid) if n_valid > 0 else (np.nan, np.nan)

        ks_stat, ks_pval = (np.nan, np.nan)
        if n_valid >= 10:
            valid_pboots = [p for p in p_boots if p is not None]
            ks_stat, ks_pval = stats.kstest(valid_pboots, 'uniform')

        summary.update({
            'type_i': type_i,
            'type_i_ci': ci,
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'lambda_obs_mean': np.nanmean(lambda_obs),
            'lambda_obs_median': np.nanmedian(lambda_obs),
            'lambda_boot_mean': np.nanmean(lambda_boot_means),
            'p_boots': p_boots,
            'lambda_obs_list': lambda_obs,
        })

        # Find outliers (worst Lambda_obs)
        passed_with_lambda = [(r, r.get('lambda_obs', 0)) for r in passed]
        passed_with_lambda.sort(key=lambda x: -x[1])
        summary['outliers'] = passed_with_lambda[:5]

        return summary

    def write_smoke_status(self, test_name: str, summary: Dict, verdict: str, reason: str):
        """Write SMOKE_STATUS.md file."""
        filepath = os.path.join(self.outdir, f'SMOKE_STATUS_{test_name.replace("-", "_")}.md')

        with open(filepath, 'w') as f:
            f.write(f"# Smoke Calibration Status: {test_name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Verdict: **{verdict}**\n\n")
            f.write(f"Reason: {reason}\n\n")

            f.write("## Summary\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Trials | {summary['n_total']} |\n")
            f.write(f"| Pass Rate | {summary['n_pass']}/{summary['n_total']} ({summary['pass_rate']:.1%}) |\n")
            f.write(f"| Type I | {summary['type_i']:.3f} |\n")
            ci = summary['type_i_ci']
            f.write(f"| Type I 95% CI | [{ci[0]:.3f}, {ci[1]:.3f}] |\n")
            f.write(f"| KS Statistic | {summary['ks_stat']:.4f} |\n")
            f.write(f"| KS p-value | {summary['ks_pval']:.6f} |\n")
            f.write(f"| Lambda_obs mean | {summary['lambda_obs_mean']:.4f} |\n")
            f.write(f"| Lambda_obs median | {summary['lambda_obs_median']:.4f} |\n")
            f.write(f"| Lambda_boot mean | {summary['lambda_boot_mean']:.4f} |\n")

            f.write("\n## Top 5 Lambda Outliers\n\n")
            f.write("| Trial | Lambda_obs | p_boot | NLL_con | NLL_unc |\n")
            f.write("|-------|------------|--------|---------|----------|\n")
            for r, lam in summary.get('outliers', []):
                f.write(f"| {r.get('trial', '?')} | {lam:.4f} | {r.get('p_boot', np.nan):.4f} | ")
                f.write(f"{r.get('nll_con', np.nan):.4f} | {r.get('nll_unc', np.nan):.4f} |\n")

        return filepath

    def run_test(self, test_name: str) -> Tuple[str, Dict, List[Dict]]:
        """Run smoke calibration for a single test with early stopping."""
        test_config = self.get_test_config(test_name)
        log(f"Starting smoke calibration for {test_config['name']}", self.logfile)
        log(f"  Trials: {self.n_trials}, Bootstrap: {self.n_bootstrap}, Starts: {self.n_starts}", self.logfile)

        results = []
        verdict = "UNKNOWN"
        reason = ""

        # Run trials in batches for early stopping checks
        batch_size = 5
        for batch_start in range(0, self.n_trials, batch_size):
            batch_end = min(batch_start + batch_size, self.n_trials)
            trial_args = [
                (test_config, i, self.n_bootstrap, self.n_starts)
                for i in range(batch_start, batch_end)
            ]

            with Pool(min(self.n_workers, len(trial_args))) as pool:
                batch_results = pool.map(run_single_trial, trial_args)

            results.extend(batch_results)

            # Check early stop
            should_stop, stop_reason = self.check_early_stop(results, len(results))

            # Log progress
            passed = [r for r in results if r.get('gates') == 'PASS']
            p_boots = [r.get('p_boot') for r in passed if r.get('p_boot') is not None]
            rejections = sum(1 for p in p_boots if p is not None and p < 0.05)
            type_i = rejections / len(p_boots) if p_boots else np.nan

            log(f"  [{test_config['name']}] {len(results)}/{self.n_trials}: "
                f"PASS={len(passed)}, Type I={type_i:.3f}", self.logfile)

            if should_stop:
                verdict = "FAIL_EARLY"
                reason = stop_reason
                log(f"  EARLY STOP: {stop_reason}", self.logfile)
                break
        else:
            # Completed all trials
            summary = self.compute_summary(results)
            ci = summary['type_i_ci']

            # Final verdict
            if summary['pass_rate'] < 0.8:
                verdict = "FAIL"
                reason = f"Pass rate {summary['pass_rate']:.1%} < 80%"
            elif ci[0] > 0.08:
                verdict = "FAIL"
                reason = f"Type I CI [{ci[0]:.3f}, {ci[1]:.3f}] above target"
            elif ci[1] < 0.02:
                verdict = "FAIL"
                reason = f"Type I CI [{ci[0]:.3f}, {ci[1]:.3f}] below target"
            elif summary['ks_pval'] < 0.001:
                verdict = "FAIL"
                reason = f"KS p-value {summary['ks_pval']:.6f} indicates non-uniform p_boot"
            elif ci[0] <= 0.08 and ci[1] >= 0.02:
                verdict = "PASS"
                reason = f"Type I CI [{ci[0]:.3f}, {ci[1]:.3f}] overlaps [0.02, 0.08]"
            else:
                verdict = "MARGINAL"
                reason = "Inconclusive - need more trials"

        summary = self.compute_summary(results)
        status_file = self.write_smoke_status(test_config['name'], summary, verdict, reason)
        log(f"  Wrote: {status_file}", self.logfile)
        log(f"  Verdict: {verdict} - {reason}", self.logfile)

        return verdict, summary, results

    def run_diagnostics(self, test_name: str, results: List[Dict]):
        """Run targeted diagnostics on failed test."""
        log(f"Running diagnostics for {test_name}...", self.logfile)

        # Import diagnostics module
        from smoke_diagnostics import run_targeted_diagnostics

        test_config = self.get_test_config(test_name)
        diag_results = run_targeted_diagnostics(
            test_config, results, self.diagdir, self.n_starts, self.logfile
        )

        return diag_results


def parse_args():
    parser = argparse.ArgumentParser(description='Smoke calibration with early stopping')
    parser.add_argument('--config', type=str, required=True, help='Path to tests_top3.json')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    parser.add_argument('--tests', nargs='+', default=['y_states', 'di_charmonium'],
                        help='Tests to run')
    parser.add_argument('--trials', type=int, default=25, help='Number of trials')
    parser.add_argument('--bootstrap', type=int, default=80, help='Bootstrap replicates')
    parser.add_argument('--starts', type=int, default=120, help='Optimizer starts')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--run-diagnostics', action='store_true',
                        help='Run diagnostics on failure')
    return parser.parse_args()


def main():
    args = parse_args()

    calibrator = SmokeCalibrator(
        config_path=args.config,
        outdir=args.outdir,
        n_trials=args.trials,
        n_bootstrap=args.bootstrap,
        n_starts=args.starts,
        n_workers=args.workers
    )

    log("=" * 70, calibrator.logfile)
    log("SMOKE CALIBRATION - Early Stop Mode", calibrator.logfile)
    log("=" * 70, calibrator.logfile)

    all_results = {}
    all_verdicts = {}

    for test_name in args.tests:
        verdict, summary, results = calibrator.run_test(test_name)
        all_results[test_name] = results
        all_verdicts[test_name] = (verdict, summary)

        # Run diagnostics if failed and requested
        if verdict in ['FAIL', 'FAIL_EARLY'] and args.run_diagnostics:
            calibrator.run_diagnostics(test_name, results)

    # Write combined status
    log("\n" + "=" * 70, calibrator.logfile)
    log("SMOKE CALIBRATION SUMMARY", calibrator.logfile)
    log("=" * 70, calibrator.logfile)

    all_pass = True
    for test_name, (verdict, summary) in all_verdicts.items():
        ci = summary['type_i_ci']
        log(f"  {test_name}: {verdict} (Type I: {summary['type_i']:.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}])", calibrator.logfile)
        if verdict not in ['PASS', 'MARGINAL']:
            all_pass = False

    if all_pass:
        log("\nAll tests PASSED smoke calibration!", calibrator.logfile)
    else:
        log("\nSome tests FAILED - run with --run-diagnostics for analysis", calibrator.logfile)


if __name__ == '__main__':
    main()
