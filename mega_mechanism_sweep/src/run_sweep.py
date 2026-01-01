#!/usr/bin/env python3
"""
MEGA MECHANISM SWEEP - Orchestrator
====================================

Runs mechanism comparison across ALL registered tests and generates
MEGA_TABLE summary with AIC/BIC comparisons.

Usage:
    python run_sweep.py
    python run_sweep.py --fresh  # Recompute everything
"""

import yaml
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# Import the test runner
from run_test import run_test, CONFIGS_DIR, RUNS_DIR, OUT_DIR, LOGS_DIR

# ==============================================================================
# SWEEP ORCHESTRATOR
# ==============================================================================

def run_sweep(use_prior=True):
    """
    Run mechanism comparison for all tests in registry.

    Args:
        use_prior: Use existing results when available

    Returns:
        list: All test results
    """
    # Load test registry
    with open(CONFIGS_DIR / 'tests.yaml', 'r') as f:
        config_data = yaml.safe_load(f)

    tests = config_data['tests']
    mechanisms = config_data['mechanisms']

    print("=" * 70)
    print("MEGA MECHANISM SWEEP")
    print("=" * 70)
    print(f"Running {len(tests)} tests...")
    print(f"Mechanisms: {', '.join(mechanisms.keys())}")
    print()

    # Run all tests
    all_results = []
    for test_config in tests:
        try:
            result = run_test(test_config, use_prior=use_prior)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR running {test_config['name']}: {e}")
            all_results.append({
                'test_name': test_config['name'],
                'display_name': test_config.get('display_name', test_config['name']),
                'experiment': test_config.get('experiment', 'Unknown'),
                'mechanism_verdict': 'ERROR',
                'mechanism_reason': str(e),
                'best_mechanism': 'UNDETERMINED',
                'proxy_test': test_config.get('proxy_test', False),
            })

    # Generate summaries
    generate_mega_table(all_results)
    generate_summary(all_results, mechanisms)

    return all_results


def generate_mega_table(results):
    """Generate MEGA_TABLE.csv and MEGA_TABLE.md"""

    # Build table data
    rows = []
    for r in results:
        row = {
            'Test': r.get('display_name', r.get('test_name', 'Unknown')),
            'Experiment': r.get('experiment', 'Unknown'),
            'Proxy': 'Yes' if r.get('proxy_test') else '',
            'M0_Verdict': r.get('M0', {}).get('verdict', 'N/A'),
            'Best_Mechanism': r.get('best_mechanism', 'N/A'),
            'Lambda': r.get('M0', {}).get('Lambda'),
            'p_boot': r.get('M0', {}).get('p_value'),
            'chi2_A': r.get('M0', {}).get('chi2_dof_A'),
            'chi2_B': r.get('M0', {}).get('chi2_dof_B'),
            'Health_A': r.get('M0', {}).get('health_A', 'N/A'),
            'Health_B': r.get('M0', {}).get('health_B', 'N/A'),
            'delta_AIC': r.get('delta_AIC'),
            'delta_BIC': r.get('delta_BIC'),
            'Mechanism_Verdict': r.get('mechanism_verdict', 'N/A'),
            'Notes': r.get('mechanism_reason', '')[:50] + '...' if len(r.get('mechanism_reason', '')) > 50 else r.get('mechanism_reason', ''),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    df.to_csv(OUT_DIR / 'MEGA_TABLE.csv', index=False)

    # Generate Markdown
    md = "# MEGA MECHANISM SWEEP - Results Table\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    md += "## Overview\n\n"

    # Count verdicts
    verdict_counts = {}
    for r in results:
        v = r.get('mechanism_verdict', 'UNKNOWN')
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    md += "| Verdict | Count |\n"
    md += "|---------|-------|\n"
    for v, c in sorted(verdict_counts.items()):
        md += f"| {v} | {c} |\n"
    md += f"| **Total** | **{len(results)}** |\n"
    md += "\n"

    # Main table
    md += "## Detailed Results\n\n"
    md += "| Test | Experiment | Proxy | M0 Verdict | Best Mech | Λ | p_boot | χ²/dof_A | χ²/dof_B | ΔAIC | Notes |\n"
    md += "|------|------------|-------|------------|-----------|---|--------|----------|----------|------|-------|\n"

    for r in results:
        M0 = r.get('M0', {})
        Lambda = M0.get('Lambda')
        p_boot = M0.get('p_value')
        chi2_A = M0.get('chi2_dof_A')
        chi2_B = M0.get('chi2_dof_B')
        delta_AIC = r.get('delta_AIC')

        Lambda_str = f"{Lambda:.2f}" if Lambda is not None else "-"
        p_str = f"{p_boot:.3f}" if p_boot is not None else "-"
        chi2_A_str = f"{chi2_A:.2f}" if chi2_A is not None else "-"
        chi2_B_str = f"{chi2_B:.2f}" if chi2_B is not None else "-"
        delta_AIC_str = f"{delta_AIC:.1f}" if delta_AIC is not None else "-"

        proxy = "✓" if r.get('proxy_test') else ""

        notes = r.get('mechanism_reason', '')
        if len(notes) > 40:
            notes = notes[:40] + "..."

        md += f"| {r.get('display_name', 'N/A')[:30]} | {r.get('experiment', 'N/A')} | {proxy} | "
        md += f"**{M0.get('verdict', 'N/A')}** | {r.get('best_mechanism', 'N/A')} | "
        md += f"{Lambda_str} | {p_str} | {chi2_A_str} | {chi2_B_str} | {delta_AIC_str} | {notes} |\n"

    md += "\n"

    # Save MD
    with open(OUT_DIR / 'MEGA_TABLE.md', 'w') as f:
        f.write(md)

    print(f"Saved: {OUT_DIR / 'MEGA_TABLE.csv'}")
    print(f"Saved: {OUT_DIR / 'MEGA_TABLE.md'}")


def generate_summary(results, mechanisms):
    """Generate SUMMARY.md with narrative analysis."""

    # Categorize results
    M0_supported = [r for r in results if r.get('M0', {}).get('verdict') == 'SUPPORTED']
    M0_disfavored = [r for r in results if r.get('M0', {}).get('verdict') == 'DISFAVORED']
    M0_inconclusive = [r for r in results if r.get('M0', {}).get('verdict') == 'INCONCLUSIVE']
    M0_mismatch = [r for r in results if r.get('M0', {}).get('verdict') == 'MODEL MISMATCH']
    M0_other = [r for r in results if r.get('M0', {}).get('verdict') not in
                ['SUPPORTED', 'DISFAVORED', 'INCONCLUSIVE', 'MODEL MISMATCH']]

    # Separate proxy tests
    amplitude_supported = [r for r in M0_supported if not r.get('proxy_test')]
    proxy_supported = [r for r in M0_supported if r.get('proxy_test')]

    amplitude_disfavored = [r for r in M0_disfavored if not r.get('proxy_test')]
    proxy_disfavored = [r for r in M0_disfavored if r.get('proxy_test')]

    summary = f"""# MEGA MECHANISM SWEEP - Summary Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report compares the rank-1 bottleneck mechanism (M0) against standard mechanisms
(unconstrained coherent, incoherent sum) across {len(results)} experimental tests from
BaBar, BESIII, Belle, CMS, ATLAS, and LHCb.

### Key Findings

| Category | Amplitude-Level | Proxy (Yield) | Total |
|----------|-----------------|---------------|-------|
| M0 SUPPORTED | {len(amplitude_supported)} | {len(proxy_supported)} | {len(M0_supported)} |
| M0 DISFAVORED | {len(amplitude_disfavored)} | {len(proxy_disfavored)} | {len(M0_disfavored)} |
| INCONCLUSIVE | {len(M0_inconclusive)} | - | {len(M0_inconclusive)} |
| MODEL MISMATCH | {len(M0_mismatch)} | - | {len(M0_mismatch)} |
| Other | {len(M0_other)} | - | {len(M0_other)} |

---

## Mechanism Definitions

| Mechanism | Description |
|-----------|-------------|
"""

    for mech_id, mech_info in mechanisms.items():
        name = mech_info.get('name', mech_id) if isinstance(mech_info, dict) else mech_info
        desc = mech_info.get('description', '') if isinstance(mech_info, dict) else ''
        summary += f"| {mech_id} | {name}: {desc[:60]}{'...' if len(desc) > 60 else ''} |\n"

    summary += """
---

## Tests Where M0 (Rank-1) is SUPPORTED

"""

    if M0_supported:
        for r in M0_supported:
            M0 = r.get('M0', {})
            proxy_tag = " **(PROXY)**" if r.get('proxy_test') else ""
            summary += f"### {r.get('display_name', 'N/A')}{proxy_tag}\n\n"
            summary += f"- **Experiment**: {r.get('experiment', 'N/A')}\n"
            summary += f"- **Paper**: {r.get('paper_ref', 'N/A')}\n"
            summary += f"- **Lambda**: {M0.get('Lambda', 'N/A')}\n"
            summary += f"- **p-value**: {M0.get('p_value', 'N/A')}\n"
            summary += f"- **chi2/dof**: A={M0.get('chi2_dof_A', 'N/A')}, B={M0.get('chi2_dof_B', 'N/A')}\n"
            if M0.get('shared_in_both_95') is not None:
                summary += f"- **Shared in both 95% CL**: {'Yes' if M0.get('shared_in_both_95') else 'No'}\n"
            summary += f"\n**Interpretation**: {r.get('mechanism_reason', 'N/A')}\n\n"
    else:
        summary += "_No tests with M0 SUPPORTED._\n\n"

    summary += """---

## Tests Where M0 (Rank-1) is DISFAVORED

"""

    if M0_disfavored:
        for r in M0_disfavored:
            M0 = r.get('M0', {})
            summary += f"### {r.get('display_name', 'N/A')}\n\n"
            summary += f"- **Experiment**: {r.get('experiment', 'N/A')}\n"
            summary += f"- **Paper**: {r.get('paper_ref', 'N/A')}\n"
            summary += f"- **Lambda**: {M0.get('Lambda', 'N/A')}\n"
            summary += f"- **p-value**: {M0.get('p_value', 'N/A')}\n"
            summary += f"- **chi2/dof**: A={M0.get('chi2_dof_A', 'N/A')}, B={M0.get('chi2_dof_B', 'N/A')}\n"
            delta_aic = r.get('delta_AIC')
            if delta_aic is not None:
                summary += f"- **ΔAIC (M0-M1)**: {delta_aic:.1f}\n"
            summary += f"\n**Interpretation**: {r.get('mechanism_reason', 'N/A')}\n\n"
    else:
        summary += "_No tests with M0 DISFAVORED._\n\n"

    summary += """---

## Tests with INCONCLUSIVE Results

"""

    if M0_inconclusive:
        for r in M0_inconclusive:
            M0 = r.get('M0', {})
            summary += f"### {r.get('display_name', 'N/A')}\n\n"
            summary += f"- **Reason**: {r.get('mechanism_reason', 'N/A')}\n"
            summary += f"- **chi2/dof**: A={M0.get('chi2_dof_A', 'N/A')}, B={M0.get('chi2_dof_B', 'N/A')}\n\n"
    else:
        summary += "_No inconclusive tests._\n\n"

    summary += """---

## Tests with MODEL MISMATCH

These tests have fit quality issues that prevent reliable mechanism comparison.

"""

    if M0_mismatch:
        for r in M0_mismatch:
            M0 = r.get('M0', {})
            summary += f"### {r.get('display_name', 'N/A')}\n\n"
            summary += f"- **Health A**: {M0.get('health_A', 'N/A')} (chi2/dof = {M0.get('chi2_dof_A', 'N/A')})\n"
            summary += f"- **Health B**: {M0.get('health_B', 'N/A')} (chi2/dof = {M0.get('chi2_dof_B', 'N/A')})\n"
            summary += f"- **Issue**: {r.get('mechanism_reason', 'N/A')}\n\n"
    else:
        summary += "_No MODEL MISMATCH tests._\n\n"

    summary += """---

## Top 3 Strongest M0 Supports

Tests with highest statistical confidence for rank-1 mechanism:

"""

    # Sort by p-value (higher is better for SUPPORTED)
    sorted_supported = sorted(
        [r for r in M0_supported if r.get('M0', {}).get('p_value') is not None],
        key=lambda x: x['M0']['p_value'],
        reverse=True
    )[:3]

    if sorted_supported:
        for i, r in enumerate(sorted_supported, 1):
            M0 = r.get('M0', {})
            proxy_tag = " (PROXY)" if r.get('proxy_test') else ""
            summary += f"{i}. **{r.get('display_name', 'N/A')}{proxy_tag}**: p={M0.get('p_value', 'N/A'):.3f}, Λ={M0.get('Lambda', 'N/A')}\n"
    else:
        summary += "_No M0 SUPPORTED tests with p-values._\n"

    summary += """
---

## Top 3 Strongest M0 Rejections

Tests with highest statistical confidence against rank-1 mechanism:

"""

    # Sort by Lambda (higher is stronger rejection)
    sorted_disfavored = sorted(
        [r for r in M0_disfavored if r.get('M0', {}).get('Lambda') is not None],
        key=lambda x: x['M0']['Lambda'],
        reverse=True
    )[:3]

    if sorted_disfavored:
        for i, r in enumerate(sorted_disfavored, 1):
            M0 = r.get('M0', {})
            summary += f"{i}. **{r.get('display_name', 'N/A')}**: Λ={M0.get('Lambda', 'N/A'):.1f}, p={M0.get('p_value', 0):.4f}\n"
    else:
        summary += "_No M0 DISFAVORED tests with Lambda values._\n"

    summary += """
---

## Physics Interpretation

### Where Rank-1 Works

"""

    if amplitude_supported:
        summary += "The rank-1 bottleneck mechanism is **supported** in amplitude-level tests:\n\n"
        for r in amplitude_supported:
            summary += f"- {r.get('display_name', 'N/A')} ({r.get('experiment', 'N/A')})\n"
        summary += "\nThese results suggest that the coupling structure g_{iα} = a_i * c_α may describe "
        summary += "the underlying production mechanism for these exotic states.\n\n"
    else:
        summary += "_No amplitude-level tests show M0 SUPPORTED._\n\n"

    if proxy_supported:
        summary += "Proxy tests (yield-ratio based) also support rank-1:\n\n"
        for r in proxy_supported:
            summary += f"- {r.get('display_name', 'N/A')} ({r.get('experiment', 'N/A')})\n"
        summary += "\n**Note**: Proxy tests provide indirect evidence as they don't test the full "
        summary += "amplitude structure.\n\n"

    summary += """### Where Rank-1 Fails

"""

    if M0_disfavored:
        summary += "The rank-1 constraint is **rejected** in:\n\n"
        for r in M0_disfavored:
            summary += f"- {r.get('display_name', 'N/A')} ({r.get('experiment', 'N/A')}): Λ={r.get('M0', {}).get('Lambda', 'N/A')}\n"
        summary += "\nThese systems likely have more complex production mechanisms that cannot be "
        summary += "described by a rank-1 coupling matrix.\n\n"
    else:
        summary += "_No clear rank-1 rejections._\n\n"

    summary += """### Inconclusive Systems

"""

    if M0_inconclusive or M0_mismatch:
        summary += "Several systems require further investigation:\n\n"
        for r in M0_inconclusive + M0_mismatch:
            summary += f"- {r.get('display_name', 'N/A')}: {r.get('mechanism_reason', 'N/A')[:60]}\n"
        summary += "\nThese may benefit from higher-statistics data or improved amplitude models.\n\n"
    else:
        summary += "_All tests have definitive verdicts._\n\n"

    summary += """---

## Methodology Notes

1. **M0 (Rank-1)**: Tests whether both channels share a common complex ratio R = c₂·exp(iφ)/c₁
2. **Fit Health Gates**: Require 0.5 < χ²/dof < 3.0 for valid interpretation
3. **Bootstrap p-value**: Parametric bootstrap under H₀ (rank-1 constraint holds)
4. **SUPPORTED**: p ≥ 0.05 AND shared R within both 95% CL contours
5. **DISFAVORED**: p < 0.05 AND shared R outside 95% CL contours
6. **Proxy Tests**: Use yield ratios instead of full amplitude analysis

---

*Generated by mega_mechanism_sweep framework*
"""

    # Save summary
    with open(OUT_DIR / 'SUMMARY.md', 'w') as f:
        f.write(summary)

    print(f"Saved: {OUT_DIR / 'SUMMARY.md'}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run mega mechanism sweep')
    parser.add_argument('--fresh', action='store_true',
                       help='Force fresh computation (ignore prior results)')

    args = parser.parse_args()

    use_prior = not args.fresh
    results = run_sweep(use_prior=use_prior)

    print()
    print("=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print()
    print(f"Total tests: {len(results)}")

    # Verdict counts
    verdicts = {}
    for r in results:
        v = r.get('M0', {}).get('verdict', 'UNKNOWN')
        verdicts[v] = verdicts.get(v, 0) + 1

    print("\nM0 (Rank-1) Verdicts:")
    for v, c in sorted(verdicts.items()):
        print(f"  {v}: {c}")

    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
