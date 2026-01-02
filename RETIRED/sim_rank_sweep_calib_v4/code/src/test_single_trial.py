#!/usr/bin/env python3
"""Quick test of a single calibration trial."""
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

from sim_generate import generate_dataset
from sim_fit_v3 import run_calibration_trial

# Load config
config_path = '/home/primary/DarkBItParticleColiderPredictions/sim_rank_sweep_calib_v3/tests_top3.json'
with open(config_path) as f:
    config = json.load(f)

y_test = config['tests'][0]  # Y-states
print(f"Test: {y_test['name']}")

# Generate dataset
dataset = generate_dataset(y_test, 'M0', scale_factor=1.0, seed=12345)

# Run trial with fewer bootstrap/starts for speed test
print("Running trial with n_bootstrap=10, n_starts=20...")
start = time.time()
result = run_calibration_trial(dataset, n_bootstrap=10, n_starts=20)
elapsed = time.time() - start

print(f"\nResult:")
print(f"  converged: {result.get('converged')}")
print(f"  lambda_obs: {result.get('lambda_obs'):.4f}")
print(f"  p_boot: {result.get('p_boot')}")
print(f"  chi2_dof_a: {result.get('chi2_dof_a'):.4f}")
print(f"  chi2_dof_b: {result.get('chi2_dof_b'):.4f}")
print(f"  gates: {result.get('gates')}")
print(f"  Elapsed: {elapsed:.1f}s")

# Now estimate full trial time
print(f"\nEstimate for full trial (200 bootstrap, 120 starts):")
boot_factor = 200 / 10
starts_factor = 120 / 20
estimated = elapsed * boot_factor * starts_factor
print(f"  ~{estimated:.0f}s per trial ({estimated/60:.1f} min)")
print(f"  ~{estimated * 200 / 3600:.1f} hours for 200 trials (single worker)")
print(f"  ~{estimated * 200 / 3600 / 31:.1f} hours with 31 workers")
