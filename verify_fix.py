#!/usr/bin/env python3
"""
Quick verification script for the rank1 harness fix.

Runs lhcb_pc_4312_extensions with small bootstrap to verify:
1. Nested invariant is enforced (nll_unc <= nll_con)
2. Lambda is computed correctly
3. Report generation works
"""

import sys
import json
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from rank1_discovery_mine.harness.run_test import run_rank1_test

def main():
    print("=" * 60)
    print("VERIFICATION: Rank1 Discovery Mine Fix")
    print("=" * 60)

    candidate_dir = Path("discoveries/lhcb_pc_4312_extensions")
    slug = "lhcb_pc_4312_extensions"

    print(f"\nCandidate: {slug}")
    print(f"Data dir: {candidate_dir / 'raw'}")

    # Run with small bootstrap for speed
    n_boot = 50
    print(f"\nRunning with n_boot={n_boot}...")

    results = run_rank1_test(candidate_dir, slug, n_boot=n_boot)

    # Check results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nStatus: {results.get('status')}")
    print(f"Overall verdict: {results.get('overall_verdict')}")

    all_passed = True

    for pair in results.get('pairs', []):
        print(f"\n--- {pair.get('pair')} ---")

        nll_con = pair.get('nll_con')
        nll_unc = pair.get('nll_unc')
        lambda_val = pair.get('Lambda')
        lambda_raw = pair.get('Lambda_raw')
        invariant = pair.get('invariant_holds')
        p_boot = pair.get('p_boot')

        print(f"  NLL constrained:   {nll_con:.4f}" if nll_con else "  NLL constrained: N/A")
        print(f"  NLL unconstrained: {nll_unc:.4f}" if nll_unc else "  NLL unconstrained: N/A")
        print(f"  Lambda_raw:        {lambda_raw:.4f}" if lambda_raw is not None else "  Lambda_raw: N/A")
        print(f"  Lambda (clamped):  {lambda_val:.4f}" if lambda_val is not None else "  Lambda: N/A")
        print(f"  Invariant holds:   {invariant}")
        print(f"  p_boot:            {p_boot:.4f}" if p_boot is not None else "  p_boot: N/A")
        print(f"  Verdict:           {pair.get('verdict')}")
        print(f"  Reason:            {pair.get('reason')}")

        # Check nested invariant
        if invariant is False:
            print("  ⚠️  INVARIANT VIOLATED - this is EXPECTED if optimizer failed")
            # This is now correctly detected
        elif nll_unc is not None and nll_con is not None:
            if nll_unc > nll_con + 1e-4:
                print("  ❌ FAIL: nll_unc > nll_con but invariant_holds is True!")
                all_passed = False
            else:
                print("  ✓ PASS: Nested invariant correctly enforced")

        # Check Lambda computation
        if lambda_raw is not None and nll_con is not None and nll_unc is not None:
            expected_raw = 2 * (nll_con - nll_unc)
            if abs(lambda_raw - expected_raw) > 1e-6:
                print(f"  ❌ FAIL: Lambda_raw != 2*(nll_con - nll_unc)")
                all_passed = False
            else:
                print("  ✓ PASS: Lambda_raw correctly computed")

        # Check bootstrap stats
        if pair.get('n_boot_valid') is not None:
            print(f"  Bootstrap: {pair.get('n_boot_valid')} valid, {pair.get('n_boot_failed', 0)} failed")

    # Check output files
    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)

    out_dir = candidate_dir / "out"
    json_file = out_dir / "rank1_result.json"
    report_file = out_dir / "RANK1_REPORT.md"

    print(f"\n  JSON: {json_file} - {'EXISTS' if json_file.exists() else 'MISSING'}")
    print(f"  Report: {report_file} - {'EXISTS' if report_file.exists() else 'MISSING'}")

    if report_file.exists():
        content = report_file.read_text()
        has_sanity = "Sanity Checks" in content
        has_coupling = "Coupling Ratios" in content
        has_lambda_raw = "Λ_raw" in content
        print(f"  Report has Sanity Checks: {has_sanity}")
        print(f"  Report has Coupling Ratios: {has_coupling}")
        print(f"  Report has Lambda_raw: {has_lambda_raw}")

        if not (has_sanity and has_coupling and has_lambda_raw):
            print("  ❌ FAIL: Report missing expected sections")
            all_passed = False
        else:
            print("  ✓ PASS: Report has all expected sections")

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ VERIFICATION PASSED")
    else:
        print("❌ VERIFICATION FAILED")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
