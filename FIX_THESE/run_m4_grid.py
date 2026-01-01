#!/usr/bin/env python3
"""
M4 detectability grid for Di-charmonium and Zc-like.
Stats levels: 1.0x, 2.0x
dr ∈ {0.03, 0.05, 0.08, 0.10}
dphi ∈ {5, 10, 15, 20} degrees
"""
import sys
import os
import json
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_generate import generate_dataset
from sim_fit_v3 import run_calibration_trial

def run_single_m4_trial(args):
    test_config, scale_factor, dr, dphi, trial_idx, n_bootstrap, n_starts = args
    dataset = generate_dataset(test_config, 'M4', scale_factor=scale_factor,
                               seed=trial_idx * 1000 + int(dr*1000) + int(dphi*10),
                               dr=dr, dphi_deg=dphi)
    result = run_calibration_trial(dataset, n_bootstrap=n_bootstrap, n_starts=n_starts)
    result['trial'] = trial_idx
    result['dr'] = dr
    result['dphi'] = dphi
    result['scale_factor'] = scale_factor
    return result

def run_m4_grid(test_config, test_name, stats_levels, dr_values, dphi_values, n_trials, n_bootstrap, n_starts, outdir):
    """Run M4 detectability grid."""
    print(f"\n{'='*60}")
    print(f"M4 DETECTABILITY GRID: {test_name}")
    print(f"{'='*60}")

    n_workers = max(1, cpu_count() - 1)
    all_results = []

    for scale in stats_levels:
        print(f"\n  Stats level: {scale}x")

        for dr in dr_values:
            for dphi in dphi_values:
                print(f"    dr={dr:.2f}, dphi={dphi}°...", end=' ', flush=True)

                args_list = [(test_config, scale, dr, dphi, i, n_bootstrap, n_starts)
                             for i in range(n_trials)]

                with Pool(n_workers) as pool:
                    batch_results = pool.map(run_single_m4_trial, args_list)

                pass_trials = [r for r in batch_results if r.get('pass_trial', False)]
                n_pass = len(pass_trials)
                n_rejected = sum(1 for t in pass_trials if t.get('rejected', False))
                power = n_rejected / n_pass if n_pass > 0 else np.nan
                pass_rate = n_pass / len(batch_results)

                print(f"Power={power:.1%} ({n_rejected}/{n_pass})")

                all_results.append({
                    'scale_factor': scale,
                    'dr': dr,
                    'dphi': dphi,
                    'power': power,
                    'n_rejected': n_rejected,
                    'n_pass': n_pass,
                    'pass_rate': pass_rate
                })

    return all_results

def write_m4_md(test_name, results, outpath):
    """Write M4 results to markdown."""
    with open(outpath, 'w') as f:
        f.write(f"# M4 Detectability Grid: {test_name}\n\n")

        scales = sorted(set(r['scale_factor'] for r in results))
        dr_vals = sorted(set(r['dr'] for r in results))
        dphi_vals = sorted(set(r['dphi'] for r in results))

        for scale in scales:
            f.write(f"\n## Stats Level: {scale}x\n\n")
            f.write("| dr \\ dphi |")
            for dphi in dphi_vals:
                f.write(f" {dphi}° |")
            f.write("\n|" + "---|" * (len(dphi_vals) + 1) + "\n")

            for dr in dr_vals:
                f.write(f"| {dr:.2f} |")
                for dphi in dphi_vals:
                    r = next((x for x in results if x['scale_factor'] == scale and x['dr'] == dr and x['dphi'] == dphi), None)
                    if r:
                        power = r['power']
                        if power >= 0.8:
                            f.write(f" **{power:.0%}** |")
                        else:
                            f.write(f" {power:.0%} |")
                    else:
                        f.write(" - |")
                f.write("\n")

        # Find smallest detectable at 80%
        f.write("\n## Smallest Detectable (≥80% power)\n\n")
        for scale in scales:
            scale_results = [r for r in results if r['scale_factor'] == scale and r['power'] >= 0.8]
            if scale_results:
                # Sort by dr then dphi to find smallest
                scale_results.sort(key=lambda x: (x['dr'], x['dphi']))
                smallest = scale_results[0]
                f.write(f"- {scale}x: dr={smallest['dr']:.2f}, dphi={smallest['dphi']}° (power={smallest['power']:.1%})\n")
            else:
                f.write(f"- {scale}x: No configuration reaches 80% power\n")

def write_m4_csv(test_name, results, outpath):
    """Write M4 results to CSV."""
    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['test_name', 'scale_factor', 'dr', 'dphi', 'power', 'n_rejected', 'n_pass', 'pass_rate'])
        for r in results:
            writer.writerow([test_name, r['scale_factor'], r['dr'], r['dphi'],
                           r['power'], r['n_rejected'], r['n_pass'], r['pass_rate']])

def main():
    parser = argparse.ArgumentParser(description='M4 detectability grid')
    parser.add_argument('--tests', nargs='+', default=['dicharm', 'zclike'])
    parser.add_argument('--trials', type=int, default=150)
    parser.add_argument('--bootstrap', type=int, default=150)
    parser.add_argument('--starts', type=int, default=60)
    parser.add_argument('--outdir', type=str, default='out')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(config_path, 'tests_top3.json')
    with open(config_path) as f:
        config = json.load(f)

    tests = {
        'dicharm': (config['tests'][2], 'Di-charmonium'),
        'zclike': (config['tests'][1], 'Zc-like')
    }

    stats_levels = [1.0, 2.0]
    dr_values = [0.03, 0.05, 0.08, 0.10]
    dphi_values = [5, 10, 15, 20]

    os.makedirs(args.outdir, exist_ok=True)

    for test_key in args.tests:
        if test_key not in tests:
            continue

        test_config, test_name = tests[test_key]
        results = run_m4_grid(
            test_config, test_name, stats_levels, dr_values, dphi_values,
            args.trials, args.bootstrap, args.starts, args.outdir
        )

        md_path = os.path.join(args.outdir, f'M4_{test_key.upper()}.md')
        csv_path = os.path.join(args.outdir, f'M4_{test_key.upper()}.csv')
        write_m4_md(test_name, results, md_path)
        write_m4_csv(test_name, results, csv_path)

        print(f"\n  Written: {md_path}")

    print("\n*** M4 GRID ANALYSIS COMPLETE ***")

if __name__ == '__main__':
    main()
