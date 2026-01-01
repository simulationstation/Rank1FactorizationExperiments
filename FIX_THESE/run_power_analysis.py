#!/usr/bin/env python3
"""
Power analysis for Di-charmonium and Zc-like.
Stats levels: 0.5x, 1.0x, 2.0x
Trials: M0=300, M1=300
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
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0, center - spread), min(1, center + spread))

def run_single_trial(args):
    test_config, mechanism, scale_factor, trial_idx, n_bootstrap, n_starts = args
    dataset = generate_dataset(test_config, mechanism, scale_factor=scale_factor, seed=trial_idx * 1000 + int(scale_factor * 100))
    result = run_calibration_trial(dataset, n_bootstrap=n_bootstrap, n_starts=n_starts)
    result['trial'] = trial_idx
    result['mechanism'] = mechanism
    result['scale_factor'] = scale_factor
    return result

def run_power_for_level(test_config, test_name, scale_factor, n_trials_m0, n_trials_m1, n_bootstrap, n_starts):
    """Run power analysis for a single stats level."""
    print(f"\n  Stats level: {scale_factor}x")
    print(f"  M0 trials: {n_trials_m0}, M1 trials: {n_trials_m1}")

    n_workers = max(1, cpu_count() - 1)
    results = {'M0': [], 'M1': []}

    for mechanism, n_trials in [('M0', n_trials_m0), ('M1', n_trials_m1)]:
        print(f"    Running {mechanism}...")
        args_list = [(test_config, mechanism, scale_factor, i, n_bootstrap, n_starts)
                     for i in range(n_trials)]

        with Pool(n_workers) as pool:
            batch_results = pool.map(run_single_trial, args_list)

        results[mechanism] = batch_results

        # Quick summary
        pass_trials = [r for r in batch_results if r.get('pass_trial', False)]
        n_pass = len(pass_trials)
        n_rejected = sum(1 for t in pass_trials if t.get('rejected', False))
        rate = n_rejected / n_pass if n_pass > 0 else np.nan
        print(f"      {mechanism}: {n_rejected}/{n_pass} rejected ({rate:.1%})")

    # Compute metrics
    m0_pass = [r for r in results['M0'] if r.get('pass_trial', False)]
    m1_pass = [r for r in results['M1'] if r.get('pass_trial', False)]

    m0_rejected = sum(1 for t in m0_pass if t.get('rejected', False))
    m1_rejected = sum(1 for t in m1_pass if t.get('rejected', False))

    type_i = m0_rejected / len(m0_pass) if m0_pass else np.nan
    power = m1_rejected / len(m1_pass) if m1_pass else np.nan

    m0_pass_rate = len(m0_pass) / len(results['M0']) if results['M0'] else 0
    m1_pass_rate = len(m1_pass) / len(results['M1']) if results['M1'] else 0

    return {
        'scale_factor': scale_factor,
        'type_i': type_i,
        'type_i_n': (m0_rejected, len(m0_pass)),
        'power': power,
        'power_n': (m1_rejected, len(m1_pass)),
        'm0_pass_rate': m0_pass_rate,
        'm1_pass_rate': m1_pass_rate,
        'results': results
    }

def run_power_analysis(test_config, test_name, stats_levels, n_trials_m0, n_trials_m1, n_bootstrap, n_starts, outdir):
    """Run full power analysis."""
    print(f"\n{'='*60}")
    print(f"POWER ANALYSIS: {test_name}")
    print(f"{'='*60}")

    all_results = []

    for scale in stats_levels:
        level_results = run_power_for_level(
            test_config, test_name, scale, n_trials_m0, n_trials_m1, n_bootstrap, n_starts
        )
        all_results.append(level_results)

    return all_results

def write_power_md(test_name, results, outpath):
    """Write power results to markdown."""
    with open(outpath, 'w') as f:
        f.write(f"# Power Analysis: {test_name}\n\n")
        f.write("## Results by Stats Level\n\n")
        f.write("| Stats | Type-I | Power (vs M1) | M0 Pass Rate | M1 Pass Rate |\n")
        f.write("|-------|--------|---------------|--------------|---------------|\n")

        for r in results:
            scale = r['scale_factor']
            type_i = r['type_i']
            power = r['power']
            m0_pr = r['m0_pass_rate']
            m1_pr = r['m1_pass_rate']
            t_n = r['type_i_n']
            p_n = r['power_n']
            f.write(f"| {scale}x | {type_i:.1%} ({t_n[0]}/{t_n[1]}) | {power:.1%} ({p_n[0]}/{p_n[1]}) | {m0_pr:.1%} | {m1_pr:.1%} |\n")

def write_power_csv(test_name, results, outpath):
    """Write power results to CSV."""
    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['test_name', 'scale_factor', 'type_i', 'type_i_rejected', 'type_i_total',
                        'power', 'power_rejected', 'power_total', 'm0_pass_rate', 'm1_pass_rate'])
        for r in results:
            writer.writerow([
                test_name, r['scale_factor'], r['type_i'], r['type_i_n'][0], r['type_i_n'][1],
                r['power'], r['power_n'][0], r['power_n'][1], r['m0_pass_rate'], r['m1_pass_rate']
            ])

def main():
    parser = argparse.ArgumentParser(description='Power analysis')
    parser.add_argument('--tests', nargs='+', default=['dicharm', 'zclike'])
    parser.add_argument('--trials-m0', type=int, default=300)
    parser.add_argument('--trials-m1', type=int, default=300)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--starts', type=int, default=80)
    parser.add_argument('--outdir', type=str, default='out')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests_top3.json')
    with open(config_path) as f:
        config = json.load(f)

    tests = {
        'dicharm': (config['tests'][2], 'Di-charmonium'),
        'zclike': (config['tests'][1], 'Zc-like')
    }

    stats_levels = [0.5, 1.0, 2.0]
    os.makedirs(args.outdir, exist_ok=True)

    for test_key in args.tests:
        if test_key not in tests:
            continue

        test_config, test_name = tests[test_key]
        results = run_power_analysis(
            test_config, test_name, stats_levels,
            args.trials_m0, args.trials_m1, args.bootstrap, args.starts, args.outdir
        )

        md_path = os.path.join(args.outdir, f'POWER_{test_key.upper()}.md')
        csv_path = os.path.join(args.outdir, f'POWER_{test_key.upper()}.csv')
        write_power_md(test_name, results, md_path)
        write_power_csv(test_name, results, csv_path)

        print(f"\n  Written: {md_path}")

    print("\n*** POWER ANALYSIS COMPLETE ***")

if __name__ == '__main__':
    main()
