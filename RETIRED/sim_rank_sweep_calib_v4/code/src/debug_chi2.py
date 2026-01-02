#!/usr/bin/env python3
"""Debug script to check chi2 calculation for Y-states."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from sim_generate import generate_dataset
from sim_fit_v3 import fit_joint_unconstrained, compute_chi2
import json

# Load config
config_path = '/home/primary/DarkBItParticleColiderPredictions/sim_rank_sweep_calib_v3/tests_top3.json'
with open(config_path) as f:
    config = json.load(f)

y_test = config['tests'][0]  # Y-states
print(f"Test: {y_test['name']}")
print(f"Type: {y_test['type']}")

# Generate dataset
dataset = generate_dataset(y_test, 'M0', scale_factor=1.0, seed=12345)

ch_a = dataset['channelA']
ch_b = dataset['channelB']

print("\n=== Channel A ===")
print(f"Type: {ch_a['type']}")
print(f"y range: [{ch_a['y'].min():.2f}, {ch_a['y'].max():.2f}]")
print(f"y mean: {ch_a['y'].mean():.2f}")
print(f"sigma range: [{ch_a['sigma'].min():.4f}, {ch_a['sigma'].max():.4f}]")
print(f"sigma mean: {ch_a['sigma'].mean():.4f}")
if 'stat_error' in ch_a:
    print(f"stat_error range: [{ch_a['stat_error'].min():.4f}, {ch_a['stat_error'].max():.4f}]")
    print(f"stat_error mean: {ch_a['stat_error'].mean():.4f}")
    print(f"sigma / stat_error ratio: {(ch_a['sigma'] / ch_a['stat_error']).mean():.2f}")
else:
    print("NO stat_error in channel A!")

print("\n=== Channel B ===")
print(f"Type: {ch_b['type']}")
print(f"y range: [{ch_b['y'].min():.2f}, {ch_b['y'].max():.2f}]")
print(f"y mean: {ch_b['y'].mean():.2f}")
print(f"sigma range: [{ch_b['sigma'].min():.4f}, {ch_b['sigma'].max():.4f}]")
print(f"sigma mean: {ch_b['sigma'].mean():.4f}")
if 'stat_error' in ch_b:
    print(f"stat_error range: [{ch_b['stat_error'].min():.4f}, {ch_b['stat_error'].max():.4f}]")
    print(f"stat_error mean: {ch_b['stat_error'].mean():.4f}")
    print(f"sigma / stat_error ratio: {(ch_b['sigma'] / ch_b['stat_error']).mean():.2f}")
else:
    print("NO stat_error in channel B!")

# Do fit
print("\n=== Fitting ===")
result = fit_joint_unconstrained(dataset, n_starts=50)

print(f"Converged: {result['converged']}")
if result['converged']:
    print(f"chi2_dof_a: {result['chi2_dof_a']:.4f}")
    print(f"chi2_dof_b: {result['chi2_dof_b']:.4f}")
    print(f"dof_a: {result['dof_a']}")
    print(f"dof_b: {result['dof_b']}")

    # Check residuals manually
    y_pred_a = result.get('y_pred_a')
    y_pred_b = result.get('y_pred_b')

    if y_pred_a is not None:
        residuals_a = ch_a['y'] - y_pred_a
        print(f"\n=== Residuals A ===")
        print(f"Residuals std: {residuals_a.std():.4f}")
        print(f"stat_error mean: {ch_a['stat_error'].mean():.4f}")
        print(f"Ratio residuals.std/stat_error.mean: {residuals_a.std() / ch_a['stat_error'].mean():.4f}")

        # Compute chi2 manually
        chi2_with_stat = np.sum((residuals_a / ch_a['stat_error'])**2)
        chi2_with_sigma = np.sum((residuals_a / ch_a['sigma'])**2)
        dof = len(ch_a['y']) - 5
        print(f"Chi2 with stat_error: {chi2_with_stat:.2f}, dof: {dof}, chi2/dof: {chi2_with_stat/dof:.4f}")
        print(f"Chi2 with sigma: {chi2_with_sigma:.2f}, dof: {dof}, chi2/dof: {chi2_with_sigma/dof:.4f}")
