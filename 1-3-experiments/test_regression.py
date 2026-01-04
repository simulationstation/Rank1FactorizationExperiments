#!/usr/bin/env python3
"""
Regression tests for comprehensive rank-1 results against notarized paper numbers.

Run with: python3 test_regression.py
"""

import json
import sys
from pathlib import Path

# Notarized paper baselines
BASELINES = {
    'lhcb_pc': {
        'pair1_Lambda': 5.73,
        'pair1_Lambda_tol': 1.0,
        'pair1_p': 0.05,
        'pair1_p_tol': 0.02,
        'pair2_Lambda': 1.96,
        'pair2_Lambda_tol': 0.5,
        'pair2_p': 0.44,
        'pair2_p_tol': 0.05,
        'verdict': 'NOT_REJECTED',
    },
    'besiii_y': {
        'Lambda': 3.00,
        'Lambda_tol': 0.5,
        'p': 0.40,
        'p_tol': 0.05,
        'verdict': 'NOT_REJECTED',
    },
    'belle_zb': {
        'chi2': 3.98,
        'chi2_tol': 0.5,
        'p': 0.41,
        'p_tol': 0.05,
        'verdict': 'NOT_REJECTED',
    },
}


def check_within_tolerance(actual, expected, tol, name):
    """Check if value is within tolerance."""
    if actual is None:
        return False, f"{name}: actual=None"
    diff = abs(actual - expected)
    ok = diff <= tol
    status = "PASS" if ok else "FAIL"
    return ok, f"{name}: expected={expected:.3f}±{tol}, actual={actual:.3f}, diff={diff:.3f} [{status}]"


def run_regression_tests():
    """Run all regression tests."""
    results_path = Path(__file__).parent / "comprehensive_results/COMPREHENSIVE_SUMMARY.json"

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("Run run_comprehensive_rank1_FIXED.py first")
        return 1

    with open(results_path) as f:
        data = json.load(f)

    results = {r['slug']: r for r in data['results']}
    failures = []
    passes = []

    print("="*70)
    print("REGRESSION TESTS against notarized paper numbers")
    print("="*70)

    # LHCb Pc
    print("\n--- LHCb Pc ---")
    lhcb = results.get('lhcb_pc', {})
    pairs = lhcb.get('pairs', [])

    if len(pairs) >= 2:
        pair1 = pairs[0]
        pair2 = pairs[1]

        ok, msg = check_within_tolerance(
            pair1.get('Lambda'), BASELINES['lhcb_pc']['pair1_Lambda'],
            BASELINES['lhcb_pc']['pair1_Lambda_tol'], "Pair1 Lambda"
        )
        (passes if ok else failures).append(msg)
        print(f"  {msg}")

        ok, msg = check_within_tolerance(
            pair1.get('p_boot'), BASELINES['lhcb_pc']['pair1_p'],
            BASELINES['lhcb_pc']['pair1_p_tol'], "Pair1 p_boot"
        )
        (passes if ok else failures).append(msg)
        print(f"  {msg}")

        ok, msg = check_within_tolerance(
            pair2.get('Lambda'), BASELINES['lhcb_pc']['pair2_Lambda'],
            BASELINES['lhcb_pc']['pair2_Lambda_tol'], "Pair2 Lambda"
        )
        (passes if ok else failures).append(msg)
        print(f"  {msg}")

        ok, msg = check_within_tolerance(
            pair2.get('p_boot'), BASELINES['lhcb_pc']['pair2_p'],
            BASELINES['lhcb_pc']['pair2_p_tol'], "Pair2 p_boot"
        )
        (passes if ok else failures).append(msg)
        print(f"  {msg}")

    if lhcb.get('verdict') == BASELINES['lhcb_pc']['verdict']:
        msg = f"Verdict: {lhcb.get('verdict')} [PASS]"
        passes.append(msg)
    else:
        msg = f"Verdict: expected={BASELINES['lhcb_pc']['verdict']}, actual={lhcb.get('verdict')} [FAIL]"
        failures.append(msg)
    print(f"  {msg}")

    # BESIII Y
    print("\n--- BESIII Y ---")
    besiii = results.get('besiii_y', {})

    ok, msg = check_within_tolerance(
        besiii.get('Lambda_obs'), BASELINES['besiii_y']['Lambda'],
        BASELINES['besiii_y']['Lambda_tol'], "Lambda"
    )
    (passes if ok else failures).append(msg)
    print(f"  {msg}")

    ok, msg = check_within_tolerance(
        besiii.get('p_boot'), BASELINES['besiii_y']['p'],
        BASELINES['besiii_y']['p_tol'], "p_boot"
    )
    (passes if ok else failures).append(msg)
    print(f"  {msg}")

    if besiii.get('verdict') == BASELINES['besiii_y']['verdict']:
        msg = f"Verdict: {besiii.get('verdict')} [PASS]"
        passes.append(msg)
    else:
        msg = f"Verdict: expected={BASELINES['besiii_y']['verdict']}, actual={besiii.get('verdict')} [FAIL]"
        failures.append(msg)
    print(f"  {msg}")

    # Belle Zb
    print("\n--- Belle Zb ---")
    belle = results.get('belle_zb', {})

    ok, msg = check_within_tolerance(
        belle.get('chi2'), BASELINES['belle_zb']['chi2'],
        BASELINES['belle_zb']['chi2_tol'], "chi2"
    )
    (passes if ok else failures).append(msg)
    print(f"  {msg}")

    ok, msg = check_within_tolerance(
        belle.get('p_boot'), BASELINES['belle_zb']['p'],
        BASELINES['belle_zb']['p_tol'], "p"
    )
    (passes if ok else failures).append(msg)
    print(f"  {msg}")

    if belle.get('verdict') == BASELINES['belle_zb']['verdict']:
        msg = f"Verdict: {belle.get('verdict')} [PASS]"
        passes.append(msg)
    else:
        msg = f"Verdict: expected={BASELINES['belle_zb']['verdict']}, actual={belle.get('verdict')} [FAIL]"
        failures.append(msg)
    print(f"  {msg}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Passed: {len(passes)}")
    print(f"Failed: {len(failures)}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("\nAll regression tests PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(run_regression_tests())
