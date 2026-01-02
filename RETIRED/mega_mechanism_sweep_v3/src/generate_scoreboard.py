#!/usr/bin/env python3
"""
MEGA MECHANISM SWEEP v3 - Scoreboard Generator
===============================================

Generates mathematically valid scoreboard comparing M0 (Rank-1)
against M1 (Unconstrained Coherent), M2 (Incoherent), M4 (Rank-2).

Enforces:
- Λ ≥ 0 for all nested model tests
- Fit health gates (0.5 < χ²/dof < 3.0)
- PROXY_ONLY labeling for yield-ratio tests
- Proper winner determination via hypothesis test + AIC/BIC
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
V3_DIR = REPO_DIR / "mega_mechanism_sweep_v3"
OUT_DIR = V3_DIR / "out"
LOGS_DIR = V3_DIR / "logs"

# ==============================================================================
# CONSTANTS
# ==============================================================================

CHI2_DOF_MIN = 0.5
CHI2_DOF_MAX = 3.0
LAMBDA_TOL = -1e-6  # Tolerance for Λ ≥ 0 check
P_THRESHOLD = 0.05  # Significance threshold
AIC_THRESHOLD = 6.0  # ΔAIC threshold for "substantial" evidence

# ==============================================================================
# DATA SOURCES
# ==============================================================================

TEST_SOURCES = {
    'cms_x6900_x7100': {
        'path': REPO_DIR / 'rank1_test_v2' / 'out' / 'rank1_test_v3_summary.json',
        'display': 'CMS X(6900)/X(7100) Di-J/psi',
        'experiment': 'CMS',
        'data_fidelity': 'vector_extraction',
        'type': 'AMPLITUDE',
        'channels': '4mu vs 2mu2e',
        'family': 'tetraquark',
        'proxy': False,
    },
    'atlas_x6900_x7200': {
        'path': REPO_DIR / 'atlas_rank1_test_v5' / 'out' / 'ATLAS_v5_summary.json',
        'display': 'ATLAS X(6900)/X(7200) Di-Charmonium',
        'experiment': 'ATLAS',
        'data_fidelity': 'figure_extraction',
        'type': 'AMPLITUDE',
        'channels': '4mu vs 4mu+2pi',
        'family': 'tetraquark',
        'proxy': False,
    },
    'lhcb_pair1_quad': {
        'path': REPO_DIR / 'lhcb_rank1_test_v2' / 'out' / 'results.json',
        'key': 'pair1_quad',
        'display': 'LHCb Pentaquark Pair1 T1vsT2',
        'experiment': 'LHCb',
        'data_fidelity': 'official_supplementary',
        'type': 'PROXY',
        'channels': 'T1 vs T2 (Jpsi p)',
        'family': 'pentaquark',
        'proxy': True,
    },
    'lhcb_pair2_quad': {
        'path': REPO_DIR / 'lhcb_rank1_test_v2' / 'out' / 'results.json',
        'key': 'pair2_quad',
        'display': 'LHCb Pentaquark Pair2 T1vsT3',
        'experiment': 'LHCb',
        'data_fidelity': 'official_supplementary',
        'type': 'PROXY',
        'channels': 'T1 vs T3 (Jpsi p)',
        'family': 'pentaquark',
        'proxy': True,
    },
    'babar_phi_f0': {
        'path': REPO_DIR / 'model_sweep' / 'runs' / 'babar_phi_f0_vs_phi_pipi' / 'results.json',
        'display': 'BaBar phi-f0(980) vs phi-pi+pi-',
        'experiment': 'BaBar',
        'data_fidelity': 'hepdata_numeric',
        'type': 'AMPLITUDE',
        'channels': 'phi-pipi vs phi-f0',
        'family': 'light_mesons',
        'proxy': False,
    },
    'babar_omega': {
        'path': REPO_DIR / 'babar_omega_rank1' / 'out' / 'results.json',
        'display': 'BaBar omega(1420)/omega(1650)',
        'experiment': 'BaBar',
        'data_fidelity': 'hepdata_numeric',
        'type': 'AMPLITUDE',
        'channels': 'omega-pipi vs omega-f0',
        'family': 'light_mesons',
        'proxy': False,
    },
    'y_states_belle_babar': {
        'path': REPO_DIR / 'y_rank1_test_v2' / 'out' / 'results.json',
        'display': 'Y(4220)/Y(4360) Belle vs BaBar',
        'experiment': 'Belle/BaBar',
        'data_fidelity': 'hepdata_numeric',
        'type': 'AMPLITUDE',
        'channels': 'Jpsi-pipi vs psi2S-pipi',
        'family': 'Y_states',
        'proxy': False,
    },
    'besiii_y_states': {
        'path': REPO_DIR / 'besiii_rank1_test_v2' / 'out' / 'results.json',
        'display': 'BESIII Y(4220)/Y(4360)',
        'experiment': 'BESIII',
        'data_fidelity': 'hepdata_numeric',
        'type': 'AMPLITUDE',
        'channels': 'Jpsi-pipi vs psi3686-pipi',
        'family': 'Y_states',
        'proxy': False,
    },
    'babar_kstar_k': {
        'path': REPO_DIR / 'model_sweep' / 'runs' / 'babar_kstar_k_isospin' / 'results.json',
        'display': 'BaBar K*(892)K I=0 vs I=1',
        'experiment': 'BaBar',
        'data_fidelity': 'hepdata_numeric',
        'type': 'AMPLITUDE',
        'channels': 'K*K I=0 vs I=1',
        'family': 'light_mesons',
        'proxy': False,
    },
    'babar_phi_eta': {
        'path': REPO_DIR / 'model_sweep' / 'runs' / 'babar_phi_eta_vs_kketa' / 'results.json',
        'display': 'BaBar phi-eta vs K+K-eta',
        'experiment': 'BaBar',
        'data_fidelity': 'hepdata_numeric',
        'type': 'AMPLITUDE',
        'channels': 'phi-eta vs KK-eta',
        'family': 'light_mesons',
        'proxy': False,
    },
}


def load_test_data(test_id, source_info):
    """Load and parse test result data."""
    path = source_info['path']
    if not path.exists():
        return None

    with open(path, 'r') as f:
        data = json.load(f)

    # Handle nested keys (e.g., LHCb pairs)
    if 'key' in source_info:
        data = data.get(source_info['key'], {})

    return data


def extract_metrics(test_id, source_info, data):
    """Extract standardized metrics from test data."""
    if data is None:
        return None

    metrics = {
        'test_name': test_id,
        'display': source_info['display'],
        'experiment': source_info['experiment'],
        'data_fidelity': source_info['data_fidelity'],
        'type': source_info['type'],
        'channels': source_info['channels'],
        'family': source_info['family'],
        'proxy': source_info['proxy'],
    }

    # Extract chi2/dof for both channels
    if test_id == 'cms_x6900_x7100':
        metrics['chi2_A'] = data.get('channel_A', {}).get('chi2_dof')
        metrics['chi2_B'] = data.get('channel_B', {}).get('chi2_dof')
        lr = data.get('likelihood_ratio', {})
        metrics['Lambda'] = lr.get('Lambda')
        metrics['p_boot'] = lr.get('bootstrap_p_value')
        metrics['nll_unc'] = None  # Not stored
        metrics['nll_con'] = None

    elif test_id == 'atlas_x6900_x7200':
        metrics['chi2_A'] = data.get('channel_4mu', {}).get('chi2_dof')
        metrics['chi2_B'] = data.get('channel_4mu2pi', {}).get('chi2_dof')
        lr = data.get('likelihood_ratio', {})
        metrics['Lambda'] = lr.get('Lambda')
        metrics['p_boot'] = None  # Bootstrap not run due to gate failure
        metrics['nll_unc'] = lr.get('nll_unconstrained')
        metrics['nll_con'] = lr.get('nll_constrained')

    elif test_id.startswith('lhcb_'):
        qa = data.get('quality_A', {})
        qb = data.get('quality_B', {})
        metrics['chi2_A'] = qa.get('chi2_dof')
        metrics['chi2_B'] = qb.get('chi2_dof')
        metrics['Lambda'] = data.get('lambda_obs')
        metrics['p_boot'] = data.get('p_value')
        metrics['nll_unc'] = None
        metrics['nll_con'] = None

    elif test_id == 'babar_phi_f0':
        metrics['chi2_A'] = data.get('channel_A', {}).get('chi2_dof')
        metrics['chi2_B'] = data.get('channel_B', {}).get('chi2_dof')
        lr = data.get('likelihood_ratio', {})
        metrics['Lambda'] = lr.get('Lambda')
        metrics['p_boot'] = data.get('bootstrap', {}).get('p_value')
        metrics['nll_unc'] = lr.get('nll_unconstrained')
        metrics['nll_con'] = lr.get('nll_constrained')

    elif test_id == 'babar_omega':
        metrics['chi2_A'] = data.get('channel_A', {}).get('chi2_dof')
        metrics['chi2_B'] = data.get('channel_B', {}).get('chi2_dof')
        metrics['Lambda'] = data.get('Lambda')
        metrics['p_boot'] = data.get('p_bootstrap')
        metrics['nll_unc'] = None
        metrics['nll_con'] = None

    elif test_id == 'y_states_belle_babar':
        metrics['chi2_A'] = data.get('channel_A', {}).get('chi2_dof')
        metrics['chi2_B'] = data.get('channel_B', {}).get('chi2_dof')
        shared = data.get('shared', {})
        metrics['Lambda'] = shared.get('Lambda')
        metrics['p_boot'] = data.get('bootstrap', {}).get('p_value')
        metrics['nll_unc'] = shared.get('nll_unconstrained')
        metrics['nll_con'] = shared.get('nll_constrained')

    elif test_id == 'besiii_y_states':
        metrics['chi2_A'] = data.get('channel_A', {}).get('chi2_dof')
        metrics['chi2_B'] = data.get('channel_B', {}).get('chi2_dof')
        joint = data.get('joint', {})
        metrics['Lambda'] = joint.get('Lambda')
        metrics['p_boot'] = joint.get('p_bootstrap')
        metrics['nll_unc'] = joint.get('nll_unconstrained')
        metrics['nll_con'] = joint.get('nll_constrained')

    elif test_id in ['babar_kstar_k', 'babar_phi_eta']:
        metrics['chi2_A'] = data.get('channel_A', {}).get('chi2_dof')
        metrics['chi2_B'] = data.get('channel_B', {}).get('chi2_dof')
        lr = data.get('likelihood_ratio', {})
        metrics['Lambda'] = lr.get('Lambda')
        metrics['p_boot'] = data.get('bootstrap', {}).get('p_value')
        metrics['nll_unc'] = lr.get('nll_unconstrained')
        metrics['nll_con'] = lr.get('nll_constrained')

    return metrics


def check_fit_health(chi2_A, chi2_B):
    """Check fit health gates for both channels."""
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

    # Overall gate status
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
    """Check if Λ satisfies nested model constraint."""
    if Lambda is None:
        return 'MISSING'
    if Lambda < LAMBDA_TOL:
        return 'OPTIMIZER_FAILURE'
    return 'VALID'


def compute_aic_bic_delta(nll_M0, nll_M1, k_M0, k_M1, n_data):
    """Compute ΔAIC and ΔBIC (M0 - M1)."""
    if nll_M0 is None or nll_M1 is None or n_data is None:
        return None, None

    aic_M0 = 2 * k_M0 + 2 * nll_M0
    aic_M1 = 2 * k_M1 + 2 * nll_M1
    bic_M0 = k_M0 * np.log(n_data) + 2 * nll_M0
    bic_M1 = k_M1 * np.log(n_data) + 2 * nll_M1

    return aic_M0 - aic_M1, bic_M0 - bic_M1


def determine_winner(metrics):
    """Determine winner using validity rules."""
    gates = metrics.get('gates')
    Lambda = metrics.get('Lambda')
    p_boot = metrics.get('p_boot')
    proxy = metrics.get('proxy', False)
    lambda_status = metrics.get('lambda_status')

    # Rule 1: Gates must pass
    if gates not in ['PASS']:
        return 'NO_VERDICT', f"Gates: {gates}"

    # Rule 2: Lambda must be valid (≥ 0)
    if lambda_status == 'OPTIMIZER_FAILURE':
        return 'OPTIMIZER_FAILURE', f"Λ={Lambda:.2f} < 0"

    if lambda_status == 'MISSING':
        return 'NO_VERDICT', "Λ missing"

    # Rule 3: PROXY tests cannot declare M0 winner
    if proxy:
        if p_boot is not None and p_boot >= P_THRESHOLD:
            return 'PROXY_NOT_REJECTED', f"p={p_boot:.3f} ≥ 0.05"
        elif p_boot is not None:
            return 'PROXY_REJECTED', f"p={p_boot:.3f} < 0.05"
        else:
            return 'PROXY_NO_PVALUE', "No bootstrap p-value"

    # Rule 4: Hypothesis test for amplitude-level tests
    if p_boot is None:
        return 'NO_VERDICT', "No bootstrap p-value"

    if p_boot < P_THRESHOLD:
        return 'M1', f"p={p_boot:.4f} < 0.05, Λ={Lambda:.1f}"

    # Rule 5: p ≥ 0.05, compare AIC/BIC
    delta_aic = metrics.get('delta_AIC')
    delta_bic = metrics.get('delta_BIC')

    if delta_aic is not None and delta_bic is not None:
        if delta_aic < -AIC_THRESHOLD and delta_bic < -AIC_THRESHOLD:
            return 'M0', f"ΔAIC={delta_aic:.1f}, ΔBIC={delta_bic:.1f}"
        elif abs(delta_aic) < AIC_THRESHOLD:
            return 'TIE', f"ΔAIC={delta_aic:.1f} (insufficient evidence)"
        else:
            return 'M1', f"ΔAIC={delta_aic:.1f} favors M1"
    else:
        # No AIC/BIC, use p-value alone
        if p_boot >= P_THRESHOLD:
            return 'M0', f"p={p_boot:.3f} ≥ 0.05 (M0 not rejected)"

    return 'NO_VERDICT', "Insufficient data"


def generate_scoreboard():
    """Generate the complete scoreboard."""
    rows = []

    for test_id, source_info in TEST_SOURCES.items():
        data = load_test_data(test_id, source_info)
        if data is None:
            rows.append({
                'test_name': test_id,
                'display': source_info['display'],
                'experiment': source_info['experiment'],
                'data_fidelity': source_info['data_fidelity'],
                'type': source_info['type'],
                'channels': source_info['channels'],
                'family': source_info['family'],
                'gates': 'NO_DATA',
                'Lambda': None,
                'p_boot': None,
                'delta_AIC': None,
                'delta_BIC': None,
                'winner': 'NO_DATA',
                'notes': 'Data file not found',
            })
            continue

        metrics = extract_metrics(test_id, source_info, data)
        if metrics is None:
            continue

        # Check fit health
        health_A, health_B, gates = check_fit_health(
            metrics.get('chi2_A'), metrics.get('chi2_B')
        )
        metrics['health_A'] = health_A
        metrics['health_B'] = health_B
        metrics['gates'] = gates

        # Check Lambda validity
        lambda_status = check_lambda_validity(metrics.get('Lambda'))
        metrics['lambda_status'] = lambda_status

        # Compute AIC/BIC (approximate - using stored NLL values)
        # M0: constrained (shared R) - 2 fewer params than M1
        # M1: unconstrained (independent R per channel)
        # For 2-BW model per channel: ~6 params each
        # M0: ~10 params total (shared r, phi)
        # M1: ~12 params total (independent r, phi per channel)
        if metrics.get('nll_con') and metrics.get('nll_unc'):
            k_M0, k_M1 = 10, 12
            n_data = 50  # Approximate
            delta_aic, delta_bic = compute_aic_bic_delta(
                metrics['nll_con'], metrics['nll_unc'],
                k_M0, k_M1, n_data
            )
            metrics['delta_AIC'] = delta_aic
            metrics['delta_BIC'] = delta_bic
        else:
            metrics['delta_AIC'] = None
            metrics['delta_BIC'] = None

        # Determine winner
        winner, notes = determine_winner(metrics)
        metrics['winner'] = winner
        metrics['notes'] = notes

        rows.append(metrics)

    return pd.DataFrame(rows)


def format_value(val, fmt='.2f'):
    """Format value for display."""
    if val is None:
        return '-'
    if isinstance(val, str):
        return val
    return f"{val:{fmt}}"


def write_scoreboard_csv(df):
    """Write SCOREBOARD.csv"""
    cols = ['test_name', 'experiment', 'data_fidelity', 'type', 'channels',
            'family', 'gates', 'Lambda', 'p_boot', 'delta_AIC', 'delta_BIC',
            'winner', 'notes']
    df_out = df[cols].copy()
    df_out.to_csv(OUT_DIR / 'SCOREBOARD.csv', index=False)
    print(f"Saved: {OUT_DIR / 'SCOREBOARD.csv'}")


def write_scoreboard_md(df):
    """Write SCOREBOARD.md"""
    md = "# MEGA MECHANISM SWEEP v3 - SCOREBOARD\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Overview
    md += "## Validity Summary\n\n"

    total = len(df)
    valid = len(df[df['gates'] == 'PASS'])
    mismatch = len(df[df['gates'] == 'MISMATCH'])
    undercon = len(df[df['gates'] == 'UNDERCONSTRAINED'])
    nodata = len(df[df['gates'] == 'NO_DATA'])

    md += "| Status | Count |\n"
    md += "|--------|-------|\n"
    md += f"| VALID (gates PASS) | {valid} |\n"
    md += f"| MODEL MISMATCH | {mismatch} |\n"
    md += f"| UNDERCONSTRAINED | {undercon} |\n"
    md += f"| NO DATA | {nodata} |\n"
    md += f"| **Total** | **{total}** |\n\n"

    # Winner counts (valid tests only)
    valid_df = df[df['gates'] == 'PASS']
    md += "## Winner Distribution (Valid Tests Only)\n\n"

    winner_counts = valid_df['winner'].value_counts().to_dict()
    md += "| Mechanism | Wins |\n"
    md += "|-----------|------|\n"
    for w in ['M0', 'M1', 'TIE', 'PROXY_NOT_REJECTED', 'OPTIMIZER_FAILURE', 'NO_VERDICT']:
        if w in winner_counts:
            md += f"| {w} | {winner_counts[w]} |\n"
    md += "\n"

    # Detailed table
    md += "## Detailed Scoreboard\n\n"
    md += "| Test | Exp | Type | Gates | Λ | p_boot | ΔAIC | ΔBIC | Winner | Notes |\n"
    md += "|------|-----|------|-------|---|--------|------|------|--------|-------|\n"

    for _, row in df.iterrows():
        Lambda_str = format_value(row['Lambda'], '.2f')
        p_str = format_value(row['p_boot'], '.3f')
        aic_str = format_value(row.get('delta_AIC'), '.1f')
        bic_str = format_value(row.get('delta_BIC'), '.1f')

        display = row['display'][:35] + '...' if len(str(row['display'])) > 35 else row['display']
        notes = row['notes'][:30] + '...' if len(str(row['notes'])) > 30 else row['notes']

        md += f"| {display} | {row['experiment']} | {row['type']} | "
        md += f"{row['gates']} | {Lambda_str} | {p_str} | {aic_str} | {bic_str} | "
        md += f"**{row['winner']}** | {notes} |\n"

    md += "\n"

    # Footnotes
    md += "## Legend\n\n"
    md += "- **M0**: Rank-1 bottleneck (factorization)\n"
    md += "- **M1**: Unconstrained coherent (baseline)\n"
    md += "- **TIE**: Insufficient evidence to distinguish\n"
    md += "- **PROXY_NOT_REJECTED**: Proxy test, M0 not rejected but cannot claim winner\n"
    md += "- **NO_VERDICT**: Invalid test (gates failed or missing data)\n"
    md += "- **OPTIMIZER_FAILURE**: Λ < 0 indicates optimization issue\n"
    md += "\n---\n*Generated by mega_mechanism_sweep_v3*\n"

    with open(OUT_DIR / 'SCOREBOARD.md', 'w') as f:
        f.write(md)
    print(f"Saved: {OUT_DIR / 'SCOREBOARD.md'}")


def write_mechanism_ranking(df):
    """Write MECHANISM_RANKING.md"""
    md = "# MECHANISM RANKING - Analysis Report\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Overall statistics
    valid_df = df[df['gates'] == 'PASS']
    amplitude_df = valid_df[valid_df['type'] == 'AMPLITUDE']
    proxy_df = valid_df[valid_df['type'] == 'PROXY']

    md += "## Summary Statistics\n\n"
    md += f"- **Total tests**: {len(df)}\n"
    md += f"- **Valid tests**: {len(valid_df)}\n"
    md += f"- **Amplitude-level tests**: {len(amplitude_df)}\n"
    md += f"- **Proxy tests**: {len(proxy_df)}\n\n"

    # Win counts
    md += "## Win Counts by Mechanism\n\n"
    md += "### Amplitude-Level Tests (definitive)\n\n"

    amp_winners = amplitude_df['winner'].value_counts().to_dict()
    md += "| Mechanism | Wins | Percentage |\n"
    md += "|-----------|------|------------|\n"
    for w in ['M0', 'M1', 'TIE', 'NO_VERDICT', 'OPTIMIZER_FAILURE']:
        count = amp_winners.get(w, 0)
        pct = 100 * count / len(amplitude_df) if len(amplitude_df) > 0 else 0
        md += f"| {w} | {count} | {pct:.0f}% |\n"
    md += "\n"

    md += "### Proxy Tests (auxiliary evidence)\n\n"
    proxy_winners = proxy_df['winner'].value_counts().to_dict()
    md += "| Outcome | Count |\n"
    md += "|---------|-------|\n"
    for w, c in proxy_winners.items():
        md += f"| {w} | {c} |\n"
    md += "\n"

    # Best cases for M0
    md += "## Top M0 Wins (Strongest Support)\n\n"
    m0_wins = amplitude_df[amplitude_df['winner'] == 'M0']
    if len(m0_wins) > 0:
        # Sort by p-value (higher is better for not-rejected)
        m0_wins_sorted = m0_wins.sort_values('p_boot', ascending=False)
        for _, row in m0_wins_sorted.head(3).iterrows():
            md += f"### {row['display']}\n\n"
            md += f"- **Experiment**: {row['experiment']}\n"
            md += f"- **Λ**: {format_value(row['Lambda'], '.2f')}\n"
            md += f"- **p-value**: {format_value(row['p_boot'], '.3f')}\n"
            md += f"- **Reason**: {row['notes']}\n\n"
    else:
        md += "_No amplitude-level M0 wins._\n\n"

    # Worst cases for M0
    md += "## Top M0 Losses (Strongest Rejections)\n\n"
    m1_wins = amplitude_df[amplitude_df['winner'] == 'M1']
    if len(m1_wins) > 0:
        # Sort by Lambda (higher = stronger rejection)
        m1_wins_sorted = m1_wins.sort_values('Lambda', ascending=False)
        for _, row in m1_wins_sorted.head(3).iterrows():
            md += f"### {row['display']}\n\n"
            md += f"- **Experiment**: {row['experiment']}\n"
            md += f"- **Λ**: {format_value(row['Lambda'], '.2f')}\n"
            md += f"- **p-value**: {format_value(row['p_boot'], '.4f')}\n"
            md += f"- **Reason**: {row['notes']}\n\n"
    else:
        md += "_No M0 rejections._\n\n"

    # Proxy outcomes
    md += "## Proxy Test Outcomes\n\n"
    if len(proxy_df) > 0:
        for _, row in proxy_df.iterrows():
            md += f"### {row['display']}\n\n"
            md += f"- **Experiment**: {row['experiment']}\n"
            md += f"- **Outcome**: {row['winner']}\n"
            md += f"- **Notes**: {row['notes']}\n\n"
    else:
        md += "_No proxy tests._\n\n"

    # Excluded tests
    md += "## Excluded Tests (No Verdict)\n\n"
    excluded = df[df['gates'] != 'PASS']
    if len(excluded) > 0:
        md += "| Test | Reason |\n"
        md += "|------|--------|\n"
        for _, row in excluded.iterrows():
            md += f"| {row['display']} | {row['gates']}: {row['notes']} |\n"
    else:
        md += "_All tests valid._\n\n"

    # Physics interpretation
    md += "## Physics Interpretation\n\n"

    n_m0 = amp_winners.get('M0', 0)
    n_m1 = amp_winners.get('M1', 0)
    n_tie = amp_winners.get('TIE', 0)

    if n_m0 > n_m1:
        md += "The rank-1 bottleneck mechanism (M0) is **supported** in the majority of valid "
        md += "amplitude-level tests, suggesting that factorized couplings g_{iα} = a_i·c_α "
        md += "may be a valid description for exotic hadron production.\n\n"
    elif n_m1 > n_m0:
        md += "The unconstrained coherent model (M1) outperforms the rank-1 constraint (M0) "
        md += "in the majority of tests, suggesting more complex production mechanisms.\n\n"
    else:
        md += "Results are mixed between M0 and M1, with no clear overall winner.\n\n"

    md += "---\n*Generated by mega_mechanism_sweep_v3*\n"

    with open(OUT_DIR / 'MECHANISM_RANKING.md', 'w') as f:
        f.write(md)
    print(f"Saved: {OUT_DIR / 'MECHANISM_RANKING.md'}")


def main():
    print("=" * 70)
    print("MEGA MECHANISM SWEEP v3 - Scoreboard Generator")
    print("=" * 70)
    print()

    # Generate scoreboard
    df = generate_scoreboard()

    # Write outputs
    write_scoreboard_csv(df)
    write_scoreboard_md(df)
    write_mechanism_ranking(df)

    # Print summary
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)

    valid_df = df[df['gates'] == 'PASS']
    print(f"\nTotal tests: {len(df)}")
    print(f"Valid tests: {len(valid_df)}")
    print(f"No verdict: {len(df[df['gates'] != 'PASS'])}")
    print(f"Proxy only: {len(df[df['type'] == 'PROXY'])}")

    print("\nWinner distribution (valid amplitude-level):")
    amplitude_df = valid_df[valid_df['type'] == 'AMPLITUDE']
    for winner, count in amplitude_df['winner'].value_counts().items():
        print(f"  {winner}: {count}")

    print()
    print(f"Outputs: {OUT_DIR}")


if __name__ == '__main__':
    main()
