#!/usr/bin/env python3
"""Diagnose Di-charmonium calibration issues."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from sim_generate import generate_dataset
from sim_fit_v3 import fit_joint_unconstrained, fit_joint_constrained, run_calibration_trial

config_path = '/home/primary/DarkBItParticleColiderPredictions/sim_rank_sweep_calib_v4/tests_top3.json'
with open(config_path) as f:
    config = json.load(f)

test = config['tests'][2]  # Di-charmonium (index 2)
print(f"=== DIAGNOSTIC: {test['name']} ===\n")

# 1) Data structure check
print("1) DATA STRUCTURE:")
dataset = generate_dataset(test, 'M0', scale_factor=1.0, seed=42)
ch_a = dataset['channelA']
ch_b = dataset['channelB']
print(f"   Channel A type: {ch_a['type']}")
print(f"   Channel B type: {ch_b['type']}")
print(f"   Channel A nbins: {len(ch_a['y'])}")
print(f"   Channel B nbins: {len(ch_b['y'])}")
print(f"   Channel A mean counts: {np.mean(ch_a['y']):.1f}")
print(f"   Channel B mean counts: {np.mean(ch_b['y']):.1f}")

# 2) Single fit check
print("\n2) FIT OBSERVED DATA:")
fit_unc = fit_joint_unconstrained(dataset, n_starts=80)
fit_con = fit_joint_constrained(dataset, n_starts=80)
lambda_obs = 2 * max(0, fit_con['nll'] - fit_unc['nll'])
print(f"   NLL_unc: {fit_unc['nll']:.4f}")
print(f"   NLL_con: {fit_con['nll']:.4f}")
print(f"   Lambda_obs: {lambda_obs:.4f}")
print(f"   Fit_unc gates: {fit_unc.get('gates')}")
print(f"   Fit_con gates: {fit_con.get('gates')}")

# 3) Run several trials to check distribution
print("\n3) QUICK CALIBRATION CHECK (15 trials):")
lambdas_obs = []
p_boots = []
for i in range(15):
    dataset = generate_dataset(test, 'M0', scale_factor=1.0, seed=i*100)
    result = run_calibration_trial(dataset, n_bootstrap=50, n_starts=80)
    lam = result.get('lambda_obs', np.nan)
    p = result.get('p_boot', np.nan)
    lambdas_obs.append(lam)
    p_boots.append(p)
    print(f"   Trial {i:2d}: Lambda={lam:.4f}, p_boot={p:.3f}, gates={result.get('gates')}")

# 4) Summary
print("\n4) SUMMARY:")
rejections = sum(1 for p in p_boots if p < 0.05)
type_i = rejections / len(p_boots)
print(f"   Rejections at α=0.05: {rejections}/{len(p_boots)} = {type_i:.1%}")
print(f"   Lambda_obs mean: {np.nanmean(lambdas_obs):.4f}")
print(f"   Lambda_obs median: {np.nanmedian(lambdas_obs):.4f}")
print(f"   p_boot mean: {np.nanmean(p_boots):.3f}")
print(f"   p_boot median: {np.nanmedian(p_boots):.3f}")

# Check for p_boot=0 cases
p_zero = sum(1 for p in p_boots if p == 0)
print(f"   Trials with p_boot=0: {p_zero}/{len(p_boots)}")

if type_i > 0.15:
    print("\n   DIAGNOSIS: Type I inflated - need investigation")
elif type_i < 0.02:
    print("\n   DIAGNOSIS: Type I too low - test may be conservative")
else:
    print("\n   DIAGNOSIS: Type I looks reasonable")
