#!/usr/bin/env python3
"""Quick diagnostic to identify Y-states calibration issue."""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from sim_generate import generate_dataset
from sim_fit_v3 import (
    fit_joint_unconstrained, fit_joint_constrained,
    generate_bootstrap_data, run_calibration_trial
)

# Load config
config_path = '/home/primary/DarkBItParticleColiderPredictions/sim_rank_sweep_calib_v4/tests_top3.json'
with open(config_path) as f:
    config = json.load(f)

y_test = config['tests'][0]  # Y-states
print(f"=== QUICK DIAGNOSTIC: {y_test['name']} ===\n")

# Generate one dataset
dataset = generate_dataset(y_test, 'M0', scale_factor=1.0, seed=42)

ch_a = dataset['channelA']
ch_b = dataset['channelB']

print("1) DATA STRUCTURE:")
print(f"   Channel A type: {ch_a['type']}")
print(f"   Has stat_error: {'stat_error' in ch_a}")
print(f"   Has syst_error: {'syst_error' in ch_a}")
print(f"   sigma mean: {np.mean(ch_a['sigma']):.4f}")
print(f"   stat_error mean: {np.mean(ch_a.get('stat_error', ch_a['sigma'])):.4f}")
if 'syst_error' in ch_a:
    print(f"   syst_error mean: {np.mean(ch_a['syst_error']):.4f}")
    print(f"   sigma/stat ratio: {np.mean(ch_a['sigma']/ch_a['stat_error']):.2f}")
if 'nuisance_s0' in ch_a:
    print(f"   nuisance_s0: {ch_a['nuisance_s0']:.4f}")
    print(f"   nuisance_s1: {ch_a['nuisance_s1']:.4f}")

print("\n2) FIT OBSERVED DATA:")
fit_unc = fit_joint_unconstrained(dataset, n_starts=100)
fit_con = fit_joint_constrained(dataset, n_starts=100)
lambda_obs = 2 * max(0, fit_con['nll'] - fit_unc['nll'])
print(f"   NLL_unc: {fit_unc['nll']:.4f}")
print(f"   NLL_con: {fit_con['nll']:.4f}")
print(f"   Lambda_obs: {lambda_obs:.4f}")

print("\n3) BOOTSTRAP COMPARISON (20 reps):")
print("   Comparing bootstrap WITH vs WITHOUT new systematic shifts...")

rng = np.random.default_rng(123)
lambda_boot_current = []  # Current: stat-only noise
lambda_boot_with_syst = []  # With new systematic shifts

for i in range(20):
    # Current bootstrap: stat-only noise
    boot_a = generate_bootstrap_data(ch_a, fit_con['y_pred_a'], rng)
    boot_b = generate_bootstrap_data(ch_b, fit_con['y_pred_b'], rng)
    boot_dataset = {'channelA': boot_a, 'channelB': boot_b, 'bw_params': dataset['bw_params']}

    fit_unc_b = fit_joint_unconstrained(boot_dataset, n_starts=50)
    fit_con_b = fit_joint_constrained(boot_dataset, n_starts=50)
    if fit_unc_b['converged'] and fit_con_b['converged']:
        lam = 2 * max(0, fit_con_b['nll'] - fit_unc_b['nll'])
        lambda_boot_current.append(lam)

    # Bootstrap WITH systematic shifts (proposed fix)
    y_pred_a = fit_con['y_pred_a']
    y_pred_b = fit_con['y_pred_b']
    stat_a = ch_a.get('stat_error', ch_a['sigma'])
    stat_b = ch_b.get('stat_error', ch_b['sigma'])

    # Add NEW systematic shifts
    s0_new = rng.normal(0, 1)
    s1_new = rng.normal(0, 1)
    syst_scale = y_test['channelA'].get('syst_scale', 0.05)
    syst_tilt = y_test['channelA'].get('syst_tilt', 0.02)
    x_a = ch_a['x']
    x_b = ch_b['x']
    x_norm_a = (x_a - np.mean(x_a)) / (np.max(x_a) - np.min(x_a))
    x_norm_b = (x_b - np.mean(x_b)) / (np.max(x_b) - np.min(x_b))

    syst_shift_a = syst_scale * y_pred_a * s0_new + syst_tilt * y_pred_a * x_norm_a * s1_new
    syst_shift_b = syst_scale * y_pred_b * s0_new + syst_tilt * y_pred_b * x_norm_b * s1_new

    y_boot_a_syst = rng.normal(y_pred_a + syst_shift_a, stat_a)
    y_boot_b_syst = rng.normal(y_pred_b + syst_shift_b, stat_b)

    boot_a_syst = {'x': ch_a['x'], 'y': y_boot_a_syst, 'sigma': ch_a['sigma'],
                   'stat_error': stat_a, 'type': 'gaussian'}
    boot_b_syst = {'x': ch_b['x'], 'y': y_boot_b_syst, 'sigma': ch_b['sigma'],
                   'stat_error': stat_b, 'type': 'gaussian'}
    boot_dataset_syst = {'channelA': boot_a_syst, 'channelB': boot_b_syst, 'bw_params': dataset['bw_params']}

    fit_unc_s = fit_joint_unconstrained(boot_dataset_syst, n_starts=50)
    fit_con_s = fit_joint_constrained(boot_dataset_syst, n_starts=50)
    if fit_unc_s['converged'] and fit_con_s['converged']:
        lam_s = 2 * max(0, fit_con_s['nll'] - fit_unc_s['nll'])
        lambda_boot_with_syst.append(lam_s)

print(f"\n   CURRENT (stat-only bootstrap):")
print(f"     Lambda_boot mean: {np.mean(lambda_boot_current):.4f}")
print(f"     Lambda_boot median: {np.median(lambda_boot_current):.4f}")
print(f"     p_boot estimate: {np.mean([l >= lambda_obs for l in lambda_boot_current]):.3f}")

print(f"\n   WITH SYSTEMATIC SHIFTS (proposed fix):")
print(f"     Lambda_boot mean: {np.mean(lambda_boot_with_syst):.4f}")
print(f"     Lambda_boot median: {np.median(lambda_boot_with_syst):.4f}")
print(f"     p_boot estimate: {np.mean([l >= lambda_obs for l in lambda_boot_with_syst]):.3f}")

print(f"\n4) DIAGNOSIS:")
if np.mean(lambda_boot_with_syst) > np.mean(lambda_boot_current) * 1.3:
    print("   ROOT CAUSE: Bootstrap missing systematic shifts")
    print("   FIX: Add new s0/s1 nuisance draws to bootstrap generation")
else:
    print("   Systematic shifts don't explain the gap - check optimization")
