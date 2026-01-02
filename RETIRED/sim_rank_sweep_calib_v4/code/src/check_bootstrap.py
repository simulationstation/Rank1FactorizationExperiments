#!/usr/bin/env python3
"""Check bootstrap Lambda distribution for outlier trial."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from sim_generate import generate_dataset
from sim_fit_v3 import fit_joint_unconstrained, fit_joint_constrained, bootstrap_pvalue_calibrated

config_path = '/home/primary/DarkBItParticleColiderPredictions/sim_rank_sweep_calib_v4/tests_top3.json'
with open(config_path) as f:
    config = json.load(f)

test = config['tests'][2]  # Di-charmonium
dataset = generate_dataset(test, 'M0', scale_factor=1.0, seed=0)  # Trial 0

print("=== BOOTSTRAP ANALYSIS: Di-charmonium Trial 0 ===\n")

# Fit observed
fit_unc = fit_joint_unconstrained(dataset, n_starts=80)
fit_con = fit_joint_constrained(dataset, n_starts=80)
lambda_obs = 2 * max(0, fit_con['nll'] - fit_unc['nll'])
print(f"Lambda_obs: {lambda_obs:.4f}")
print(f"NLL_unc: {fit_unc['nll']:.4f}")
print(f"NLL_con: {fit_con['nll']:.4f}")

# Get bootstrap distribution with more samples
print(f"\nRunning 100 bootstrap samples...")
p_boot, lambdas_boot, n_exceed, n_fail = bootstrap_pvalue_calibrated(
    dataset, lambda_obs, fit_con, fit_unc, n_bootstrap=100, n_starts=80
)

print(f"\nBootstrap results:")
print(f"  p_boot: {p_boot:.4f}")
print(f"  n_exceed: {n_exceed}/100")
print(f"  n_fail: {n_fail}")
print(f"  Lambda_boot mean: {np.mean(lambdas_boot):.4f}")
print(f"  Lambda_boot max: {np.max(lambdas_boot):.4f}")
print(f"  Lambda_boot 95th percentile: {np.percentile(lambdas_boot, 95):.4f}")
print(f"  Lambda_boot 99th percentile: {np.percentile(lambdas_boot, 99):.4f}")

print(f"\nExpected under chi2(2):")
from scipy.stats import chi2
print(f"  Mean: {chi2.mean(df=2):.4f}")
print(f"  95th percentile: {chi2.ppf(0.95, df=2):.4f}")
print(f"  99th percentile: {chi2.ppf(0.99, df=2):.4f}")

# Compare
print(f"\nDIAGNOSIS:")
if np.mean(lambdas_boot) < 1.0:
    print(f"  Bootstrap Lambda mean ({np.mean(lambdas_boot):.2f}) << chi2(2) mean (2.0)")
    print(f"  => Bootstrap distribution too narrow - not capturing true null variance")
else:
    print(f"  Bootstrap Lambda mean looks reasonable")

if lambda_obs > chi2.ppf(0.999, df=2):
    print(f"  Lambda_obs ({lambda_obs:.2f}) > chi2(2) 99.9th percentile ({chi2.ppf(0.999, df=2):.2f})")
    print(f"  => This is a rare event even under chi2(2) - may just be unlucky")
