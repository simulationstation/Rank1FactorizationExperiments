#!/usr/bin/env python3
"""
Smoke calibration for Di-charmonium and Zc-like (control).
Early-stop if calibration clearly fails.
"""
import sys
import os
import json
import argparse
import numpy as np
from scipy.stats import kstest
from multiprocessing import Pool, cpu_count

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_generate import generate_dataset
from sim_fit_v3 import run_calibration_trial

# Parameters
DEFAULT_TRIALS = 25
DEFAULT_BOOTSTRAP = 50
DEFAULT_STARTS = 80

def wilson_ci(p, n, z=1.96):
    """Wilson score interval for proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0, center - spread), min(1, center + spread))

def run_single_trial(args):
    """Run a single trial (for parallel execution)."""
    test_config, trial_idx, n_bootstrap, n_starts = args
    dataset = generate_dataset(test_config, 'M0', scale_factor=1.0, seed=trial_idx * 1000)
    result = run_calibration_trial(dataset, n_bootstrap=n_bootstrap, n_starts=n_starts)
    result['trial'] = trial_idx
    result['test_name'] = test_config['name']
    return result

def run_smoke_calibration(test_config, n_trials, n_bootstrap, n_starts, test_name):
    """Run smoke calibration with early stopping."""
    print(f"\n{'='*60}")
    print(f"SMOKE CALIBRATION: {test_name}")
    print(f"Trials={n_trials}, Bootstrap={n_bootstrap}, Starts={n_starts}")
    print(f"{'='*60}\n")

    results = []
    pass_trials = []
    p_boots = []

    # Use parallel processing
    n_workers = max(1, cpu_count() - 1)

    for batch_start in range(0, n_trials, 5):
        batch_end = min(batch_start + 5, n_trials)
        batch_size = batch_end - batch_start

        # Prepare arguments for parallel execution
        args_list = [(test_config, batch_start + i, n_bootstrap, n_starts)
                     for i in range(batch_size)]

        # Run batch in parallel
        with Pool(n_workers) as pool:
            batch_results = pool.map(run_single_trial, args_list)

        for result in batch_results:
            results.append(result)
            if result.get('pass_trial', False):
                pass_trials.append(result)
                if not np.isnan(result.get('p_boot', np.nan)):
                    p_boots.append(result['p_boot'])

            # Print individual result
            trial_idx = result['trial']
            lam = result.get('lambda_obs', np.nan)
            p = result.get('p_boot', np.nan)
            gates = result.get('gates', 'UNKNOWN')
            print(f"  Trial {trial_idx:2d}: Lambda={lam:.4f}, p_boot={p:.3f}, gates={gates}")

        # Checkpoint after every 5 trials
        n_pass = len(pass_trials)
        n_total = len(results)
        pass_rate = n_pass / n_total if n_total > 0 else 0

        print(f"\n--- Checkpoint: {n_total} trials, {n_pass} PASS ({pass_rate:.1%}) ---")

        # Early stopping checks
        if n_pass >= 15:
            # Check Type-I
            n_rejected = sum(1 for t in pass_trials if t.get('rejected', False))
            type_i = n_rejected / n_pass
            ci_low, ci_high = wilson_ci(type_i, n_pass)

            print(f"  Type-I: {n_rejected}/{n_pass} = {type_i:.1%} (Wilson CI: [{ci_low:.1%}, {ci_high:.1%}])")

            # Fail if CI excludes [0.02, 0.08]
            if ci_low > 0.08 or ci_high < 0.02:
                print(f"\n*** FAIL_EARLY: Type-I CI [{ci_low:.1%}, {ci_high:.1%}] excludes [2%, 8%] ***")
                return results, 'FAIL_EARLY_TYPE_I'

        if n_pass >= 20 and len(p_boots) >= 20:
            # KS test
            ks_stat, ks_pval = kstest(p_boots, 'uniform')
            print(f"  KS test: stat={ks_stat:.4f}, p={ks_pval:.4f}")

            if ks_pval < 1e-3:
                print(f"\n*** FAIL_EARLY: KS p={ks_pval:.2e} < 1e-3 ***")
                return results, 'FAIL_EARLY_KS'

        print()

    # Final summary
    n_pass = len(pass_trials)
    n_total = len(results)
    pass_rate = n_pass / n_total if n_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY: {test_name}")
    print(f"{'='*60}")
    print(f"Total trials: {n_total}")
    print(f"Pass rate: {n_pass}/{n_total} = {pass_rate:.1%}")

    if n_pass > 0:
        n_rejected = sum(1 for t in pass_trials if t.get('rejected', False))
        type_i = n_rejected / n_pass
        ci_low, ci_high = wilson_ci(type_i, n_pass)
        print(f"Type-I: {n_rejected}/{n_pass} = {type_i:.1%} (CI: [{ci_low:.1%}, {ci_high:.1%}])")

        if len(p_boots) >= 10:
            ks_stat, ks_pval = kstest(p_boots, 'uniform')
            print(f"KS test: stat={ks_stat:.4f}, p={ks_pval:.4f}")
        else:
            ks_stat, ks_pval = np.nan, np.nan

        lambda_obs_list = [t['lambda_obs'] for t in pass_trials if not np.isnan(t.get('lambda_obs', np.nan))]
        print(f"Lambda_obs mean: {np.mean(lambda_obs_list):.4f}")
        print(f"Lambda_obs max: {np.max(lambda_obs_list):.4f}")

        # Top 3 outliers
        sorted_lambdas = sorted(lambda_obs_list, reverse=True)[:3]
        print(f"Top 3 Lambda outliers: {[f'{l:.2f}' for l in sorted_lambdas]}")

        # Check success criteria
        success = (pass_rate >= 0.8 and
                   ci_low <= 0.08 and ci_high >= 0.02 and
                   (np.isnan(ks_pval) or ks_pval > 1e-3))

        if success:
            print(f"\n*** SMOKE PASS ***")
            return results, 'PASS'
        else:
            print(f"\n*** SMOKE FAIL ***")
            return results, 'FAIL'
    else:
        print("No PASS trials - calibration FAIL")
        return results, 'FAIL'

def write_status(results_dict, filepath):
    """Write status file."""
    with open(filepath, 'w') as f:
        f.write("# Smoke Calibration Status\n\n")
        for test_name, (results, status) in results_dict.items():
            f.write(f"## {test_name}\n")
            f.write(f"Status: **{status}**\n\n")

            n_pass = sum(1 for r in results if r.get('pass_trial', False))
            n_total = len(results)
            f.write(f"- Trials: {n_total}\n")
            f.write(f"- Pass rate: {n_pass}/{n_total}\n")

            pass_trials = [r for r in results if r.get('pass_trial', False)]
            if pass_trials:
                n_rejected = sum(1 for t in pass_trials if t.get('rejected', False))
                type_i = n_rejected / n_pass
                f.write(f"- Type-I: {n_rejected}/{n_pass} = {type_i:.1%}\n")

                p_boots = [t['p_boot'] for t in pass_trials if not np.isnan(t.get('p_boot', np.nan))]
                if len(p_boots) >= 10:
                    ks_stat, ks_pval = kstest(p_boots, 'uniform')
                    f.write(f"- KS p-value: {ks_pval:.4f}\n")

            f.write("\n")

def write_trace(results, filepath):
    """Write trace CSV."""
    import csv
    fieldnames = ['test_name', 'trial', 'lambda_obs', 'p_boot', 'p_wilks', 'gates',
                  'chi2_dof_a', 'chi2_dof_b', 'pass_trial', 'rejected']

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)

def main():
    parser = argparse.ArgumentParser(description='Smoke calibration for Di-charmonium')
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS, help='Number of trials')
    parser.add_argument('--bootstrap', type=int, default=DEFAULT_BOOTSTRAP, help='Bootstrap replicates')
    parser.add_argument('--starts', type=int, default=DEFAULT_STARTS, help='Optimizer starts')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests_top3.json')
    with open(config_path) as f:
        config = json.load(f)

    di_charm = config['tests'][2]  # Di-charmonium
    zc_like = config['tests'][1]   # Zc-like (control)

    results_dict = {}

    # Run Di-charmonium
    results_di, status_di = run_smoke_calibration(
        di_charm, args.trials, args.bootstrap, args.starts, 'Di-charmonium'
    )
    results_dict['Di-charmonium'] = (results_di, status_di)

    # Run Zc-like (control)
    results_zc, status_zc = run_smoke_calibration(
        zc_like, args.trials, args.bootstrap, args.starts, 'Zc-like (control)'
    )
    results_dict['Zc-like'] = (results_zc, status_zc)

    # Write outputs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    write_status(results_dict, os.path.join(base_dir, 'smoke_status.md'))
    write_trace(results_di + results_zc, os.path.join(base_dir, 'smoke_trace.csv'))

    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Di-charmonium: {status_di}")
    print(f"Zc-like: {status_zc}")

if __name__ == '__main__':
    main()
