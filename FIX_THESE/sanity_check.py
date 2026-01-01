#!/usr/bin/env python3
"""Noiseless sanity check - verify generator/fitter match."""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_generate import generate_dataset
from sim_fit_v3 import fit_joint_unconstrained, fit_joint_constrained, compute_lambda

def run_sanity_check(test_config, test_name):
    """Run noiseless sanity check for a test."""
    print(f"\n{'='*50}")
    print(f"SANITY CHECK: {test_name}")
    print(f"{'='*50}")

    # Generate M0 dataset
    dataset = generate_dataset(test_config, 'M0', scale_factor=10.0, seed=99999)

    # Replace observed with true (noiseless)
    dataset['channelA']['y'] = dataset['channelA']['y_true'].copy()
    dataset['channelB']['y'] = dataset['channelB']['y_true'].copy()

    print(f"Channel A: {len(dataset['channelA']['y'])} bins, mean={np.mean(dataset['channelA']['y']):.1f}")
    print(f"Channel B: {len(dataset['channelB']['y'])} bins, mean={np.mean(dataset['channelB']['y']):.1f}")

    # Fit with moderate starts
    print("\nFitting (noiseless data)...")
    fit_unc = fit_joint_unconstrained(dataset, n_starts=80, adaptive=False)
    fit_con = fit_joint_constrained(dataset, n_starts=80, adaptive=False)

    if not fit_unc['converged'] or not fit_con['converged']:
        print("FAIL: Fit did not converge")
        return False

    nll_unc = fit_unc['nll']
    nll_con = fit_con['nll']
    lambda_val = compute_lambda(nll_con, nll_unc)

    print(f"NLL_unconstrained: {nll_unc:.4f}")
    print(f"NLL_constrained:   {nll_con:.4f}")
    print(f"Lambda:            {lambda_val:.4f}")

    # Lambda should be < 0.5 for noiseless M0 data
    if lambda_val < 0.5:
        print(f"\n*** PASS: Lambda={lambda_val:.4f} < 0.5 ***")
        return True
    else:
        print(f"\n*** FAIL: Lambda={lambda_val:.4f} >= 0.5 (model mismatch) ***")
        return False

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests_top3.json')
    with open(config_path) as f:
        config = json.load(f)

    di_charm = config['tests'][2]  # Di-charmonium
    zc_like = config['tests'][1]   # Zc-like

    pass_di = run_sanity_check(di_charm, 'Di-charmonium')
    pass_zc = run_sanity_check(zc_like, 'Zc-like')

    print(f"\n{'='*50}")
    print("SANITY CHECK RESULTS")
    print(f"{'='*50}")
    print(f"Di-charmonium: {'PASS' if pass_di else 'FAIL'}")
    print(f"Zc-like:       {'PASS' if pass_zc else 'FAIL'}")

    if pass_di and pass_zc:
        print("\n*** All sanity checks PASS - ready for smoke calibration ***")
        sys.exit(0)
    else:
        print("\n*** Sanity check FAIL - model mismatch still present ***")
        sys.exit(1)
