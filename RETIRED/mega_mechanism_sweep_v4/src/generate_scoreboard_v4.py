#!/usr/bin/env python3
"""
MEGA MECHANISM SWEEP v4 - Scoreboard Generator
===============================================

Fixes proxy rows by using LHCb v3 results with valid Λ >= 0.
Removes any row with Λ < 0.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ==============================================================================
# PATHS
# ==============================================================================

REPO_DIR = Path(__file__).parent.parent.parent
V4_DIR = REPO_DIR / "mega_mechanism_sweep_v4"
OUT_DIR = V4_DIR / "out"

# ==============================================================================
# CONSTANTS
# ==============================================================================

CHI2_DOF_MIN = 0.5
CHI2_DOF_MAX = 3.0
LAMBDA_TOL = -1e-6
P_THRESHOLD = 0.05
AIC_THRESHOLD = 6.0

# ==============================================================================
# HARDCODED RESULTS (from source JSON files)
# ==============================================================================

# CMS X(6900)/X(7100) - from rank1_test_v2/out/rank1_test_v3_summary.json
CMS_RESULT = {
    'test_name': 'cms_x6900_x7100',
    'display': 'CMS X(6900)/X(7100) Di-J/psi',
    'experiment': 'CMS',
    'data_fidelity': 'vector_extraction',
    'type': 'AMPLITUDE',
    'channels': '4mu vs 2mu2e',
    'family': 'tetraquark',
    'chi2_A': 1.016,
    'chi2_B': 2.487,
    'Lambda': 1.825,
    'p_boot': 0.403,
    'nll_unc': None,
    'nll_con': None,
}

# ATLAS X(6900)/X(7200) - from atlas_rank1_test_v5/out/ATLAS_v5_summary.json
ATLAS_RESULT = {
    'test_name': 'atlas_x6900_x7200',
    'display': 'ATLAS X(6900)/X(7200) Di-Charmonium',
    'experiment': 'ATLAS',
    'data_fidelity': 'figure_extraction',
    'type': 'AMPLITUDE',
    'channels': '4mu vs 4mu+2pi',
    'family': 'tetraquark',
    'chi2_A': 1.394,  # 4mu channel
    'chi2_B': 9.742,  # 4mu+2pi channel - FAILS GATE
    'Lambda': 1.537,
    'p_boot': None,
    'nll_unc': -3986.302,
    'nll_con': -3985.534,
}

# BaBar phi-f0 - from model_sweep/runs/babar_phi_f0_vs_phi_pipi/results.json
BABAR_PHI_F0_RESULT = {
    'test_name': 'babar_phi_f0',
    'display': 'BaBar phi-f0(980) vs phi-pi+pi-',
    'experiment': 'BaBar',
    'data_fidelity': 'hepdata_numeric',
    'type': 'AMPLITUDE',
    'channels': 'phi-pipi vs phi-f0',
    'family': 'light_mesons',
    'chi2_A': 1.141,
    'chi2_B': 2.816,
    'Lambda': 342.14,
    'p_boot': 0.0,
    'nll_unc': 44.746,
    'nll_con': 215.816,
}

# BaBar omega - from babar_omega_rank1/out/results.json
BABAR_OMEGA_RESULT = {
    'test_name': 'babar_omega',
    'display': 'BaBar omega(1420)/omega(1650)',
    'experiment': 'BaBar',
    'data_fidelity': 'hepdata_numeric',
    'type': 'AMPLITUDE',
    'channels': 'omega-pipi vs omega-f0',
    'family': 'light_mesons',
    'chi2_A': 1.188,
    'chi2_B': 1.169,
    'Lambda': 958.27,
    'p_boot': 0.0,
    'nll_unc': None,
    'nll_con': None,
}

# Y-states Belle/BaBar - from y_rank1_test_v2/out/results.json
Y_STATES_RESULT = {
    'test_name': 'y_states_belle_babar',
    'display': 'Y(4220)/Y(4360) Belle vs BaBar',
    'experiment': 'Belle/BaBar',
    'data_fidelity': 'hepdata_numeric',
    'type': 'AMPLITUDE',
    'channels': 'Jpsi-pipi vs psi2S-pipi',
    'family': 'Y_states',
    'chi2_A': 0.703,
    'chi2_B': 0.364,  # UNDERCONSTRAINED
    'Lambda': 0.706,
    'p_boot': 0.732,
    'nll_unc': 31.804,
    'nll_con': 32.157,
}

# BESIII Y-states - from besiii_rank1_test_v2/out/results.json
BESIII_RESULT = {
    'test_name': 'besiii_y_states',
    'display': 'BESIII Y(4220)/Y(4360)',
    'experiment': 'BESIII',
    'data_fidelity': 'hepdata_numeric',
    'type': 'AMPLITUDE',
    'channels': 'Jpsi-pipi vs psi3686-pipi',
    'family': 'Y_states',
    'chi2_A': 1.660,
    'chi2_B': 8.816,  # POOR FIT
    'Lambda': 2915.53,
    'p_boot': 0.482,
    'nll_unc': 123.073,
    'nll_con': 1580.839,
}

# BaBar K*K - from model_sweep/runs/babar_kstar_k_isospin/results.json
BABAR_KSTAR_RESULT = {
    'test_name': 'babar_kstar_k',
    'display': 'BaBar K*(892)K I=0 vs I=1',
    'experiment': 'BaBar',
    'data_fidelity': 'hepdata_numeric',
    'type': 'AMPLITUDE',
    'channels': 'K*K I=0 vs I=1',
    'family': 'light_mesons',
    'chi2_A': 0.010,  # UNDERCONSTRAINED
    'chi2_B': 0.015,  # UNDERCONSTRAINED
    'Lambda': 0.071,
    'p_boot': None,
    'nll_unc': 0.407,
    'nll_con': 0.442,
}

# BaBar phi-eta - from model_sweep/runs/babar_phi_eta_vs_kketa/results.json
BABAR_PHI_ETA_RESULT = {
    'test_name': 'babar_phi_eta',
    'display': 'BaBar phi-eta vs K+K-eta',
    'experiment': 'BaBar',
    'data_fidelity': 'hepdata_numeric',
    'type': 'AMPLITUDE',
    'channels': 'phi-eta vs KK-eta',
    'family': 'light_mesons',
    'chi2_A': 0.018,  # UNDERCONSTRAINED
    'chi2_B': 0.023,  # UNDERCONSTRAINED
    'Lambda': 0.037,
    'p_boot': None,
    'nll_unc': 0.477,
    'nll_con': 0.496,
}

# LHCb Pentaquark Pair1 - from lhcb_rank1_test_v3/out/results_v3.json (tight_quadratic)
# This is the VALID result with Lambda >= 0
LHCB_PAIR1_V3_RESULT = {
    'test_name': 'lhcb_pair1_tight_quad',
    'display': 'LHCb Pentaquark Pair1 (T1 vs T2)',
    'experiment': 'LHCb',
    'data_fidelity': 'official_supplementary',
    'type': 'PROXY',
    'channels': 'Table1 vs Table2 (mKp>1.9)',
    'family': 'pentaquark',
    'chi2_A': 1.074,
    'chi2_B': 1.152,
    'Lambda': 3.582,  # VALID: >= 0
    'p_boot': 0.056,
    'nll_unc': 742.909,
    'nll_con': 744.700,
    'config': 'tight [4320-4490] quadratic bg',
    'source_verdict': 'SUPPORTED',
}

# Note: LHCb Pair2 from v2 had Lambda < 0, so we exclude it.
# No valid v3/v4 Pair2 results with Lambda >= 0 available.

ALL_RESULTS = [
    CMS_RESULT,
    ATLAS_RESULT,
    BABAR_PHI_F0_RESULT,
    BABAR_OMEGA_RESULT,
    Y_STATES_RESULT,
    BESIII_RESULT,
    BABAR_KSTAR_RESULT,
    BABAR_PHI_ETA_RESULT,
    LHCB_PAIR1_V3_RESULT,
]


def check_fit_health(chi2_A, chi2_B):
    """Check fit health gates."""
    health_A = 'UNKNOWN'
    health_B = 'UNKNOWN'

    if chi2_A is not None:
        if chi2_A < CHI2_DOF_MIN:
            health_A = 'UNDERCONSTRAINED'
        elif chi2_A > CHI2_DOF_MAX:
            health_A = 'POOR_FIT'
        else:
            health_A = 'PASS'

    if chi2_B is not None:
        if chi2_B < CHI2_DOF_MIN:
            health_B = 'UNDERCONSTRAINED'
        elif chi2_B > CHI2_DOF_MAX:
            health_B = 'POOR_FIT'
        else:
            health_B = 'PASS'

    if health_A == 'PASS' and health_B == 'PASS':
        gates = 'PASS'
    elif health_A == 'UNDERCONSTRAINED' or health_B == 'UNDERCONSTRAINED':
        gates = 'UNDERCONSTRAINED'
    elif health_A == 'POOR_FIT' or health_B == 'POOR_FIT':
        gates = 'MISMATCH'
    else:
        gates = 'UNKNOWN'

    return health_A, health_B, gates


def check_lambda_validity(Lambda):
    """Check Lambda validity."""
    if Lambda is None:
        return 'MISSING'
    if Lambda < LAMBDA_TOL:
        return 'INVALID'
    return 'VALID'


def determine_winner(row):
    """Determine winner per test."""
    gates = row.get('gates')
    Lambda = row.get('Lambda')
    p_boot = row.get('p_boot')
    test_type = row.get('type')
    lambda_status = row.get('lambda_status')

    # Rule 1: Gates must pass
    if gates != 'PASS':
        return 'NO_VERDICT', f"Gates: {gates}"

    # Rule 2: Lambda must be valid
    if lambda_status != 'VALID':
        return 'NO_VERDICT', f"Lambda: {lambda_status}"

    # Rule 3: PROXY tests get special treatment
    if test_type == 'PROXY':
        if p_boot is not None and p_boot >= P_THRESHOLD:
            return 'PROXY_NOT_REJECTED', f"p={p_boot:.3f} >= 0.05"
        elif p_boot is not None:
            return 'PROXY_REJECTED', f"p={p_boot:.3f} < 0.05"
        else:
            return 'PROXY_NO_PVALUE', "No p-value"

    # Rule 4: Amplitude tests - hypothesis test
    if p_boot is None:
        return 'NO_VERDICT', "No p-value"

    if p_boot < P_THRESHOLD:
        return 'M1', f"p={p_boot:.4f} < 0.05, Lambda={Lambda:.1f}"

    # Rule 5: p >= 0.05, M0 not rejected
    return 'M0', f"p={p_boot:.3f} >= 0.05 (not rejected)"


def generate_scoreboard():
    """Generate the v4 scoreboard."""
    rows = []

    for result in ALL_RESULTS:
        row = result.copy()

        # Check fit health
        health_A, health_B, gates = check_fit_health(
            row.get('chi2_A'), row.get('chi2_B')
        )
        row['health_A'] = health_A
        row['health_B'] = health_B
        row['gates'] = gates

        # Check Lambda validity
        lambda_status = check_lambda_validity(row.get('Lambda'))
        row['lambda_status'] = lambda_status

        # Determine winner
        winner, notes = determine_winner(row)
        row['winner'] = winner
        row['notes'] = notes

        rows.append(row)

    return pd.DataFrame(rows)


def format_value(val, fmt='.2f'):
    """Format value for display."""
    if val is None:
        return '-'
    if isinstance(val, str):
        return val
    try:
        return f"{val:{fmt}}"
    except:
        return str(val)


def write_scoreboard_csv(df):
    """Write SCOREBOARD_v4.csv"""
    cols = ['test_name', 'experiment', 'data_fidelity', 'type', 'channels',
            'family', 'gates', 'Lambda', 'p_boot', 'chi2_A', 'chi2_B',
            'winner', 'notes']
    df_out = df[[c for c in cols if c in df.columns]].copy()
    df_out.to_csv(OUT_DIR / 'SCOREBOARD_v4.csv', index=False)
    print(f"Saved: {OUT_DIR / 'SCOREBOARD_v4.csv'}")


def write_scoreboard_md(df):
    """Write SCOREBOARD_v4.md"""
    md = "# MEGA MECHANISM SWEEP v4 - SCOREBOARD\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Overview
    total = len(df)
    valid = len(df[df['gates'] == 'PASS'])
    amplitude_valid = len(df[(df['gates'] == 'PASS') & (df['type'] == 'AMPLITUDE')])
    proxy_valid = len(df[(df['gates'] == 'PASS') & (df['type'] == 'PROXY')])
    mismatch = len(df[df['gates'] == 'MISMATCH'])
    undercon = len(df[df['gates'] == 'UNDERCONSTRAINED'])

    md += "## Validity Summary\n\n"
    md += "| Status | Count |\n"
    md += "|--------|-------|\n"
    md += f"| VALID (gates PASS) | {valid} |\n"
    md += f"| - Amplitude-level | {amplitude_valid} |\n"
    md += f"| - Proxy | {proxy_valid} |\n"
    md += f"| MODEL MISMATCH | {mismatch} |\n"
    md += f"| UNDERCONSTRAINED | {undercon} |\n"
    md += f"| **Total** | **{total}** |\n\n"

    # Winner distribution (amplitude only)
    md += "## Winner Distribution (Valid Amplitude-Level Tests Only)\n\n"
    amp_df = df[(df['gates'] == 'PASS') & (df['type'] == 'AMPLITUDE')]

    if len(amp_df) > 0:
        winner_counts = amp_df['winner'].value_counts().to_dict()
        md += "| Mechanism | Wins | Percentage |\n"
        md += "|-----------|------|------------|\n"
        for w in ['M0', 'M1', 'TIE', 'NO_VERDICT']:
            count = winner_counts.get(w, 0)
            pct = 100 * count / len(amp_df) if len(amp_df) > 0 else 0
            if count > 0 or w in ['M0', 'M1']:
                md += f"| **{w}** | {count} | {pct:.0f}% |\n"
        md += "\n"
    else:
        md += "_No valid amplitude-level tests._\n\n"

    # Proxy outcomes
    md += "## Proxy Test Outcomes (Auxiliary Evidence)\n\n"
    proxy_df = df[df['type'] == 'PROXY']

    if len(proxy_df) > 0:
        md += "| Test | Λ | p | Gates | Outcome |\n"
        md += "|------|---|---|-------|--------|\n"
        for _, row in proxy_df.iterrows():
            Lambda_str = format_value(row['Lambda'], '.2f')
            p_str = format_value(row['p_boot'], '.3f')
            md += f"| {row['display']} | {Lambda_str} | {p_str} | {row['gates']} | **{row['winner']}** |\n"
        md += "\n"
    else:
        md += "_No proxy tests._\n\n"

    # Detailed table
    md += "## Detailed Scoreboard\n\n"
    md += "| Test | Exp | Type | Gates | Λ | p | χ²/dof | Winner | Notes |\n"
    md += "|------|-----|------|-------|---|---|--------|--------|-------|\n"

    for _, row in df.iterrows():
        Lambda_str = format_value(row['Lambda'], '.2f')
        p_str = format_value(row['p_boot'], '.3f')
        chi2_str = f"{format_value(row['chi2_A'], '.2f')}/{format_value(row['chi2_B'], '.2f')}"

        display = row['display'][:30] + '...' if len(str(row['display'])) > 30 else row['display']
        notes = str(row['notes'])[:25] + '...' if len(str(row['notes'])) > 25 else row['notes']

        md += f"| {display} | {row['experiment']} | {row['type']} | "
        md += f"{row['gates']} | {Lambda_str} | {p_str} | {chi2_str} | "
        md += f"**{row['winner']}** | {notes} |\n"

    md += "\n"

    # Key
    md += "## Key\n\n"
    md += "- **M0**: Rank-1 bottleneck wins (not rejected, p ≥ 0.05)\n"
    md += "- **M1**: Unconstrained coherent wins (M0 rejected, p < 0.05)\n"
    md += "- **PROXY_NOT_REJECTED**: Proxy test, M0 not rejected (auxiliary evidence only)\n"
    md += "- **NO_VERDICT**: Test excluded due to gate failures\n"
    md += "\n---\n*Generated by mega_mechanism_sweep_v4*\n"

    with open(OUT_DIR / 'SCOREBOARD_v4.md', 'w') as f:
        f.write(md)
    print(f"Saved: {OUT_DIR / 'SCOREBOARD_v4.md'}")


def write_mechanism_ranking(df):
    """Write MECHANISM_RANKING_v4.md"""
    md = "# MECHANISM RANKING v4 - Analysis Report\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Summary
    amp_df = df[(df['gates'] == 'PASS') & (df['type'] == 'AMPLITUDE')]
    proxy_df = df[(df['gates'] == 'PASS') & (df['type'] == 'PROXY')]

    md += "## Summary Statistics\n\n"
    md += f"- **Total tests**: {len(df)}\n"
    md += f"- **Valid amplitude-level**: {len(amp_df)}\n"
    md += f"- **Valid proxy**: {len(proxy_df)}\n"
    md += f"- **Excluded**: {len(df) - len(amp_df) - len(proxy_df)}\n\n"

    # Win counts
    md += "## Win Counts (Amplitude-Level Only)\n\n"
    if len(amp_df) > 0:
        winner_counts = amp_df['winner'].value_counts().to_dict()
        m0_wins = winner_counts.get('M0', 0)
        m1_wins = winner_counts.get('M1', 0)

        md += f"| Mechanism | Wins |\n"
        md += f"|-----------|------|\n"
        md += f"| **M0** (Rank-1) | {m0_wins} |\n"
        md += f"| **M1** (Unconstrained) | {m1_wins} |\n\n"
    else:
        md += "_No valid amplitude tests._\n\n"

    # Best M0 cases
    md += "## M0 Wins (Strong Support for Rank-1)\n\n"
    m0_wins_df = amp_df[amp_df['winner'] == 'M0']
    if len(m0_wins_df) > 0:
        for _, row in m0_wins_df.iterrows():
            md += f"### {row['display']}\n\n"
            md += f"- **Experiment**: {row['experiment']}\n"
            md += f"- **Λ**: {format_value(row['Lambda'], '.2f')}\n"
            md += f"- **p-value**: {format_value(row['p_boot'], '.3f')}\n"
            md += f"- **χ²/dof**: A={format_value(row['chi2_A'], '.2f')}, B={format_value(row['chi2_B'], '.2f')}\n"
            md += f"- **Interpretation**: {row['notes']}\n\n"
    else:
        md += "_No M0 wins._\n\n"

    # Worst M0 cases
    md += "## M0 Losses (Strong Rejections)\n\n"
    m1_wins_df = amp_df[amp_df['winner'] == 'M1'].sort_values('Lambda', ascending=False)
    if len(m1_wins_df) > 0:
        for _, row in m1_wins_df.head(3).iterrows():
            md += f"### {row['display']}\n\n"
            md += f"- **Experiment**: {row['experiment']}\n"
            md += f"- **Λ**: {format_value(row['Lambda'], '.2f')}\n"
            md += f"- **p-value**: {format_value(row['p_boot'], '.4f')}\n"
            md += f"- **Interpretation**: {row['notes']}\n\n"
    else:
        md += "_No M0 rejections._\n\n"

    # Proxy outcomes
    md += "## Proxy Test Outcomes\n\n"
    md += "_Proxy tests provide auxiliary evidence but cannot declare M0 as winner._\n\n"
    if len(proxy_df) > 0:
        for _, row in proxy_df.iterrows():
            md += f"### {row['display']}\n\n"
            md += f"- **Λ**: {format_value(row['Lambda'], '.2f')}\n"
            md += f"- **p-value**: {format_value(row['p_boot'], '.3f')}\n"
            md += f"- **Outcome**: {row['winner']}\n"
            if 'config' in row:
                md += f"- **Config**: {row.get('config', 'N/A')}\n"
            md += f"- **Source verdict**: {row.get('source_verdict', 'N/A')}\n\n"
    else:
        md += "_No valid proxy tests._\n\n"

    # Excluded tests
    md += "## Excluded Tests\n\n"
    excluded = df[df['gates'] != 'PASS']
    if len(excluded) > 0:
        md += "| Test | Gates | Reason |\n"
        md += "|------|-------|--------|\n"
        for _, row in excluded.iterrows():
            reason = f"chi2/dof: {format_value(row['chi2_A'], '.2f')}/{format_value(row['chi2_B'], '.2f')}"
            md += f"| {row['display']} | {row['gates']} | {reason} |\n"
    else:
        md += "_All tests valid._\n\n"

    md += "\n---\n*Generated by mega_mechanism_sweep_v4*\n"

    with open(OUT_DIR / 'MECHANISM_RANKING_v4.md', 'w') as f:
        f.write(md)
    print(f"Saved: {OUT_DIR / 'MECHANISM_RANKING_v4.md'}")


def write_summary(df):
    """Write SUMMARY_v4.md"""
    md = "# MEGA MECHANISM SWEEP v4 - SUMMARY\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    amp_df = df[(df['gates'] == 'PASS') & (df['type'] == 'AMPLITUDE')]
    proxy_df = df[(df['gates'] == 'PASS') & (df['type'] == 'PROXY')]

    # Executive summary
    md += "## Executive Summary\n\n"

    if len(amp_df) > 0:
        m0_wins = len(amp_df[amp_df['winner'] == 'M0'])
        m1_wins = len(amp_df[amp_df['winner'] == 'M1'])
        total_amp = len(amp_df)

        md += f"From **{total_amp} valid amplitude-level tests**:\n\n"
        md += f"- **M0 (Rank-1) wins**: {m0_wins} ({100*m0_wins/total_amp:.0f}%)\n"
        md += f"- **M1 (Unconstrained) wins**: {m1_wins} ({100*m1_wins/total_amp:.0f}%)\n\n"

    if len(proxy_df) > 0:
        not_rejected = len(proxy_df[proxy_df['winner'] == 'PROXY_NOT_REJECTED'])
        rejected = len(proxy_df[proxy_df['winner'] == 'PROXY_REJECTED'])
        md += f"From **{len(proxy_df)} valid proxy tests**:\n\n"
        md += f"- M0 not rejected: {not_rejected}\n"
        md += f"- M0 rejected: {rejected}\n\n"

    # Amplitude results
    md += "## Amplitude-Level Results\n\n"
    md += "### M0 Supported\n\n"
    m0_df = amp_df[amp_df['winner'] == 'M0']
    if len(m0_df) > 0:
        for _, row in m0_df.iterrows():
            md += f"- **{row['display']}**: Λ={format_value(row['Lambda'], '.2f')}, p={format_value(row['p_boot'], '.3f')}\n"
    else:
        md += "_None_\n"
    md += "\n"

    md += "### M0 Rejected (M1 Wins)\n\n"
    m1_df = amp_df[amp_df['winner'] == 'M1'].sort_values('Lambda', ascending=False)
    if len(m1_df) > 0:
        for _, row in m1_df.iterrows():
            md += f"- **{row['display']}**: Λ={format_value(row['Lambda'], '.1f')}, p={format_value(row['p_boot'], '.4f')}\n"
    else:
        md += "_None_\n"
    md += "\n"

    # Proxy results
    md += "## Proxy-Only Results\n\n"
    md += "_Proxy tests use yield ratios and cannot definitively support M0._\n\n"
    if len(proxy_df) > 0:
        for _, row in proxy_df.iterrows():
            md += f"- **{row['display']}**: Λ={format_value(row['Lambda'], '.2f')}, p={format_value(row['p_boot'], '.3f')} → {row['winner']}\n"
    else:
        md += "_No valid proxy tests._\n"
    md += "\n"

    # Blocked tests
    md += "## Blocked Tests\n\n"
    excluded = df[df['gates'] != 'PASS']
    if len(excluded) > 0:
        for _, row in excluded.iterrows():
            md += f"- **{row['display']}** ({row['experiment']}): {row['gates']} - "
            md += f"chi2/dof = {format_value(row['chi2_A'], '.2f')}/{format_value(row['chi2_B'], '.2f')}\n"
    else:
        md += "_None_\n"
    md += "\n"

    # Key change from v3
    md += "## Changes from v3\n\n"
    md += "- Removed invalid LHCb proxy rows with Λ < 0 (optimizer failure)\n"
    md += "- Added LHCb Pair1 from v3 tight_quadratic config (Λ=3.58 ≥ 0)\n"
    md += "- Excluded LHCb Pair2 (no valid Λ ≥ 0 result available)\n"
    md += "- All rows now have valid Λ ≥ 0\n\n"

    md += "---\n*Generated by mega_mechanism_sweep_v4*\n"

    with open(OUT_DIR / 'SUMMARY_v4.md', 'w') as f:
        f.write(md)
    print(f"Saved: {OUT_DIR / 'SUMMARY_v4.md'}")


def main():
    print("=" * 70)
    print("MEGA MECHANISM SWEEP v4 - Scoreboard Generator")
    print("=" * 70)
    print()

    # Generate scoreboard
    df = generate_scoreboard()

    # Verify no Λ < 0
    invalid_lambda = df[df['Lambda'] < 0] if 'Lambda' in df.columns else pd.DataFrame()
    if len(invalid_lambda) > 0:
        print(f"WARNING: {len(invalid_lambda)} rows with Lambda < 0!")
    else:
        print("✓ All rows have valid Lambda >= 0")

    print()

    # Write outputs
    write_scoreboard_csv(df)
    write_scoreboard_md(df)
    write_mechanism_ranking(df)
    write_summary(df)

    # Print summary
    print()
    print("=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)

    amp_df = df[(df['gates'] == 'PASS') & (df['type'] == 'AMPLITUDE')]
    proxy_df = df[(df['gates'] == 'PASS') & (df['type'] == 'PROXY')]

    print(f"\nTotal tests: {len(df)}")
    print(f"Valid amplitude-level: {len(amp_df)}")
    print(f"Valid proxy: {len(proxy_df)}")
    print(f"Excluded (no verdict): {len(df) - len(amp_df) - len(proxy_df)}")

    if len(amp_df) > 0:
        print("\nWinner distribution (amplitude-level):")
        for winner, count in amp_df['winner'].value_counts().items():
            pct = 100 * count / len(amp_df)
            print(f"  {winner}: {count} ({pct:.0f}%)")

    if len(proxy_df) > 0:
        print("\nProxy outcomes:")
        for winner, count in proxy_df['winner'].value_counts().items():
            print(f"  {winner}: {count}")

    print(f"\nOutputs: {OUT_DIR}")


if __name__ == '__main__':
    main()
