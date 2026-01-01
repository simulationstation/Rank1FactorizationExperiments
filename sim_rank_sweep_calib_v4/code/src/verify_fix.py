#!/usr/bin/env python3
"""Quick verification that NLL fix improves calibration."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from sim_generate import generate_dataset
from sim_fit_v3 import run_calibration_trial

config_path = '/home/primary/DarkBItParticleColiderPredictions/sim_rank_sweep_calib_v4/tests_top3.json'
with open(config_path) as f:
    config = json.load(f)

y_test = config['tests'][0]  # Y-states
print(f"=== VERIFY FIX: {y_test['name']} ===\n")
print("Running 10 quick trials with n_bootstrap=30, n_starts=50...\n")

results = []
for i in range(10):
    dataset = generate_dataset(y_test, 'M0', scale_factor=1.0, seed=i*100)
    result = run_calibration_trial(dataset, n_bootstrap=30, n_starts=50)
    results.append(result)
    print(f"Trial {i}: Lambda={result.get('lambda_obs', np.nan):.4f}, p_boot={result.get('p_boot', np.nan):.3f}, gates={result.get('gates')}")

# Compute Type I
passed = [r for r in results if r.get('gates') == 'PASS']
p_boots = [r['p_boot'] for r in passed if r.get('p_boot') is not None]
rejections = sum(1 for p in p_boots if p < 0.05)
type_i = rejections / len(p_boots) if p_boots else np.nan

lambda_obs = [r.get('lambda_obs', np.nan) for r in passed]
lambda_boot = [r.get('lambda_boot_mean', np.nan) for r in passed]

print(f"\n=== RESULTS ===")
print(f"Pass rate: {len(passed)}/10")
print(f"Type I: {rejections}/{len(p_boots)} = {type_i:.2f}")
print(f"Lambda_obs mean: {np.nanmean(lambda_obs):.4f}")
print(f"Lambda_boot mean: {np.nanmean(lambda_boot):.4f}")
print(f"Ratio (should be ~1): {np.nanmean(lambda_obs)/np.nanmean(lambda_boot):.2f}")

if type_i <= 0.20 and np.nanmean(lambda_obs)/np.nanmean(lambda_boot) < 2:
    print("\nFIX LOOKS PROMISING - Lambda ratio improved!")
else:
    print("\nFIX MAY NOT BE SUFFICIENT - need more investigation")
