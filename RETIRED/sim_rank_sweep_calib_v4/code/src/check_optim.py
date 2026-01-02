#!/usr/bin/env python3
"""Check if outlier Lambda is due to under-optimization."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from sim_generate import generate_dataset
from sim_fit_v3 import fit_joint_unconstrained, fit_joint_constrained

config_path = '/home/primary/DarkBItParticleColiderPredictions/sim_rank_sweep_calib_v4/tests_top3.json'
with open(config_path) as f:
    config = json.load(f)

y_test = config['tests'][0]

# Use seed from Trial 5 (Lambda=1.05, outlier)
seed = 5 * 1000 + hash('Y-states') % 10000
dataset = generate_dataset(y_test, 'M0', scale_factor=1.0, seed=seed)

print("=== OPTIMIZATION CHECK FOR OUTLIER TRIAL ===\n")
print("Testing if Lambda decreases with more starts...\n")

for n_starts in [50, 120, 200, 400]:
    fit_unc = fit_joint_unconstrained(dataset, n_starts=n_starts)
    fit_con = fit_joint_constrained(dataset, n_starts=n_starts)
    lam = 2 * max(0, fit_con['nll'] - fit_unc['nll'])
    print(f"Starts={n_starts:3d}: NLL_unc={fit_unc['nll']:.4f}, NLL_con={fit_con['nll']:.4f}, Lambda={lam:.4f}")

print("\nIf Lambda stays high with more starts => real signal, not optimization issue")
print("If Lambda drops significantly => CONSTRAINED FIT IS UNDER-OPTIMIZED")
