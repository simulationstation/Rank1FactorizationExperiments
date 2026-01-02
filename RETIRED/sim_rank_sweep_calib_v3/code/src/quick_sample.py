#!/usr/bin/env python3
"""Quick sample of 10 trials to estimate calibration quality."""
import sys
import os
import json
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))

from sim_generate import generate_dataset
from sim_fit_v3 import run_calibration_trial

config_path = '/home/primary/DarkBItParticleColiderPredictions/sim_rank_sweep_calib_v3/tests_top3.json'
with open(config_path) as f:
    config = json.load(f)

def run_trial(args):
    test_config, seed = args
    dataset = generate_dataset(test_config, 'M0', scale_factor=1.0, seed=seed)
    result = run_calibration_trial(dataset, n_bootstrap=50, n_starts=50)
    return result

print("=" * 60)
print("PRELIMINARY CALIBRATION SAMPLE (10 trials per test)")
print("=" * 60)

for test_config in config['tests']:
    test_name = test_config['name']
    print(f"\n{test_name}:")

    # Run 10 trials in parallel
    with Pool(10) as pool:
        args = [(test_config, 1000 + i) for i in range(10)]
        results = pool.map(run_trial, args)

    # Analyze
    passed = [r for r in results if r.get('gates') == 'PASS']
    n_pass = len(passed)
    print(f"  Pass rate: {n_pass}/10")

    if passed:
        p_boots = [r.get('p_boot', np.nan) for r in passed if r.get('p_boot') is not None]
        lambdas = [r.get('lambda_obs', np.nan) for r in passed]
        chi2s_a = [r.get('chi2_dof_a', np.nan) for r in passed]
        chi2s_b = [r.get('chi2_dof_b', np.nan) for r in passed]

        rejections = sum(1 for p in p_boots if p < 0.05)
        type_i = rejections / len(p_boots) if p_boots else np.nan

        print(f"  Type I (p<0.05): {rejections}/{len(p_boots)} = {type_i:.2f}")
        print(f"  Lambda_obs: mean={np.nanmean(lambdas):.3f}, median={np.nanmedian(lambdas):.3f}")
        print(f"  chi2/dof_a: mean={np.nanmean(chi2s_a):.3f}")
        print(f"  chi2/dof_b: mean={np.nanmean(chi2s_b):.3f}")
        print(f"  p_boot values: {[f'{p:.3f}' for p in p_boots[:5]]}...")
