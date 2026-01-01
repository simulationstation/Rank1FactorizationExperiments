#!/usr/bin/env python3
"""
Medium calibration for Di-charmonium and Zc-like.
Early-stop if Type-I CI excludes [0.02, 0.08] or KS p < 1e-3 after 60 PASS trials.
"""
import sys
import os
import json
import argparse
import numpy as np
from scipy.stats import kstest
from multiprocessing import Pool, cpu_count
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_generate import generate_dataset
from sim_fit_v3 import run_calibration_trial

def wilson_ci(p, n, z=1.96):
    """Wilson score interval for proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0, center - spread), min(1, center + spread))

def run_single_trial(args):
    """Run a single trial."""
    test_config, trial_idx, n_bootstrap, n_starts = args
    dataset = generate_dataset(test_config, 'M0', scale_factor=1.0, seed=trial_idx * 1000)
    result = run_calibration_trial(dataset, n_bootstrap=n_bootstrap, n_starts=n_starts)
    result['trial'] = trial_idx
    result['test_name'] = test_config['name']
    return result

def run_medium_calibration(test_config, test_name, n_trials, n_bootstrap, n_starts, outdir):
    """Run medium calibration with early stopping."""
    print(f"\n{'='*60}")
    print(f"MEDIUM CALIBRATION: {test_name}")
    print(f"Trials={n_trials}, Bootstrap={n_bootstrap}, Starts={n_starts}")
    print(f"{'='*60}\n")

    results = []
    pass_trials = []
    p_boots = []

    n_workers = max(1, cpu_count() - 1)
    batch_size = min(10, n_workers)

    for batch_start in range(0, n_trials, batch_size):
        batch_end = min(batch_start + batch_size, n_trials)
        actual_batch = batch_end - batch_start

        args_list = [(test_config, batch_start + i, n_bootstrap, n_starts)
                     for i in range(actual_batch)]

        with Pool(n_workers) as pool:
            batch_results = pool.map(run_single_trial, args_list)

        for result in batch_results:
            results.append(result)
            if result.get('pass_trial', False):
                pass_trials.append(result)
                if not np.isnan(result.get('p_boot', np.nan)):
                    p_boots.append(result['p_boot'])

            trial_idx = result['trial']
            lam = result.get('lambda_obs', np.nan)
            p = result.get('p_boot', np.nan)
            gates = result.get('gates', 'UNKNOWN')
            print(f"  Trial {trial_idx:3d}: Lambda={lam:.4f}, p_boot={p:.3f}, gates={gates}")

        n_pass = len(pass_trials)
        n_total = len(results)

        # Checkpoint every 10 trials
        if n_total % 10 == 0:
            print(f"\n--- Checkpoint: {n_total} trials, {n_pass} PASS ({100*n_pass/n_total:.1f}%) ---")

            if n_pass >= 60:
                n_rejected = sum(1 for t in pass_trials if t.get('rejected', False))
                type_i = n_rejected / n_pass
                ci_low, ci_high = wilson_ci(type_i, n_pass)
                print(f"  Type-I: {n_rejected}/{n_pass} = {type_i:.1%} (Wilson CI: [{ci_low:.1%}, {ci_high:.1%}])")

                # Early stop if CI excludes [0.02, 0.08]
                if ci_low > 0.08 or ci_high < 0.02:
                    print(f"\n*** FAIL_EARLY: Type-I CI [{ci_low:.1%}, {ci_high:.1%}] excludes [2%, 8%] ***")
                    return results, 'FAIL_EARLY_TYPE_I', None

                # KS test
                if len(p_boots) >= 60:
                    ks_stat, ks_pval = kstest(p_boots, 'uniform')
                    print(f"  KS test: stat={ks_stat:.4f}, p={ks_pval:.4f}")
                    if ks_pval < 1e-3:
                        print(f"\n*** FAIL_EARLY: KS p={ks_pval:.2e} < 1e-3 ***")
                        return results, 'FAIL_EARLY_KS', None

            print()

    # Final summary
    n_pass = len(pass_trials)
    n_total = len(results)
    pass_rate = n_pass / n_total if n_total > 0 else 0

    summary = {
        'test_name': test_name,
        'n_total': n_total,
        'n_pass': n_pass,
        'pass_rate': pass_rate
    }

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY: {test_name}")
    print(f"{'='*60}")
    print(f"Total trials: {n_total}")
    print(f"Pass rate: {n_pass}/{n_total} = {pass_rate:.1%}")

    if n_pass > 0:
        n_rejected = sum(1 for t in pass_trials if t.get('rejected', False))
        type_i = n_rejected / n_pass
        ci_low, ci_high = wilson_ci(type_i, n_pass)
        summary['n_rejected'] = n_rejected
        summary['type_i'] = type_i
        summary['type_i_ci'] = (ci_low, ci_high)

        print(f"Type-I: {n_rejected}/{n_pass} = {type_i:.1%} (CI: [{ci_low:.1%}, {ci_high:.1%}])")

        if len(p_boots) >= 10:
            ks_stat, ks_pval = kstest(p_boots, 'uniform')
            summary['ks_stat'] = ks_stat
            summary['ks_pval'] = ks_pval
            print(f"KS test: stat={ks_stat:.4f}, p={ks_pval:.4f}")

        lambda_obs_list = [t['lambda_obs'] for t in pass_trials if not np.isnan(t.get('lambda_obs', np.nan))]
        summary['lambda_mean'] = np.mean(lambda_obs_list)
        summary['lambda_max'] = np.max(lambda_obs_list)
        print(f"Lambda_obs mean: {np.mean(lambda_obs_list):.4f}")
        print(f"Lambda_obs max: {np.max(lambda_obs_list):.4f}")

        # Check success
        success = (pass_rate >= 0.8 and
                   ci_low <= 0.08 and ci_high >= 0.02 and
                   (summary.get('ks_pval', 1.0) > 1e-3))

        if success:
            print(f"\n*** MEDIUM CALIBRATION PASS ***")
            return results, 'PASS', summary
        else:
            print(f"\n*** MEDIUM CALIBRATION FAIL ***")
            return results, 'FAIL', summary
    else:
        print("No PASS trials - calibration FAIL")
        return results, 'FAIL', summary

def write_results_md(summary, results, outpath):
    """Write results to markdown file."""
    with open(outpath, 'w') as f:
        f.write(f"# Medium Calibration: {summary['test_name']}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total trials | {summary['n_total']} |\n")
        f.write(f"| Pass rate | {summary['n_pass']}/{summary['n_total']} = {summary['pass_rate']:.1%} |\n")
        if 'type_i' in summary:
            ci_low, ci_high = summary['type_i_ci']
            f.write(f"| Type-I | {summary['n_rejected']}/{summary['n_pass']} = {summary['type_i']:.1%} |\n")
            f.write(f"| Type-I 95% CI | [{ci_low:.1%}, {ci_high:.1%}] |\n")
        if 'ks_pval' in summary:
            f.write(f"| KS p-value | {summary['ks_pval']:.4f} |\n")
        if 'lambda_mean' in summary:
            f.write(f"| Lambda mean | {summary['lambda_mean']:.4f} |\n")
            f.write(f"| Lambda max | {summary['lambda_max']:.4f} |\n")

def write_trace_csv(results, outpath):
    """Write trace CSV."""
    fieldnames = ['test_name', 'trial', 'lambda_obs', 'p_boot', 'p_wilks', 'gates',
                  'chi2_dof_a', 'chi2_dof_b', 'pass_trial', 'rejected']
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)

def main():
    parser = argparse.ArgumentParser(description='Medium calibration')
    parser.add_argument('--tests', nargs='+', default=['dicharm', 'zclike'])
    parser.add_argument('--trials', type=int, default=120)
    parser.add_argument('--bootstrap', type=int, default=120)
    parser.add_argument('--starts', type=int, default=150)
    parser.add_argument('--outdir', type=str, default='out')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests_top3.json')
    with open(config_path) as f:
        config = json.load(f)

    tests = {
        'dicharm': (config['tests'][2], 'Di-charmonium'),
        'zclike': (config['tests'][1], 'Zc-like')
    }

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'medium_traces'), exist_ok=True)

    all_passed = True
    summaries = {}

    for test_key in args.tests:
        if test_key not in tests:
            print(f"Unknown test: {test_key}")
            continue

        test_config, test_name = tests[test_key]
        results, status, summary = run_medium_calibration(
            test_config, test_name, args.trials, args.bootstrap, args.starts, args.outdir
        )

        # Write outputs
        md_path = os.path.join(args.outdir, f'medium_calib_{test_key}.md')
        csv_path = os.path.join(args.outdir, 'medium_traces', f'{test_key}_trace.csv')

        if summary:
            write_results_md(summary, results, md_path)
            summaries[test_key] = summary
        write_trace_csv(results, csv_path)

        if status != 'PASS':
            all_passed = False
            print(f"\n*** {test_name} FAILED: {status} ***")

    print(f"\n{'='*60}")
    print("OVERALL MEDIUM CALIBRATION RESULTS")
    print(f"{'='*60}")
    for test_key in args.tests:
        if test_key in summaries:
            s = summaries[test_key]
            print(f"{tests[test_key][1]}: Type-I={s.get('type_i', np.nan):.1%}, KS p={s.get('ks_pval', np.nan):.4f}")

    if all_passed:
        print("\n*** ALL MEDIUM CALIBRATIONS PASSED - READY FOR POWER RUNS ***")
        # Write success marker
        with open(os.path.join(args.outdir, 'MEDIUM_PASS'), 'w') as f:
            f.write("PASS\n")
    else:
        print("\n*** SOME CALIBRATIONS FAILED - DO NOT PROCEED TO POWER RUNS ***")

if __name__ == '__main__':
    main()
