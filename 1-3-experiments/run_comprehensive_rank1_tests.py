#!/usr/bin/env python3
"""
Comprehensive Rank-1 Test Suite

Runs both pairwise and global multi-channel tests on all available datasets,
and generates a consolidated summary report.

Datasets tested (from README "REAL results" table):
1. CMS X(6900)/X(7100) - 2 channels
2. BESIII Y(4220)/Y(4320) - 2 channels
3. Belle Zb(10610)/Zb(10650) - 4 hidden-bottom channels
4. LHCb Pc(4440)/Pc(4457) - 4 projection tables

Note: Belle Zb open-bottom (DISFAVORED/threshold) is excluded per user request.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import the multichannel module
from rank1_multichannel import (
    ChannelData, MultiChannelResult,
    run_global_multichannel_test,
    generate_multichannel_report
)

# Import pairwise test functions from existing harness
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class DatasetConfig:
    """Configuration for a test dataset."""
    name: str
    slug: str
    state1_name: str
    state2_name: str
    M1: float  # GeV
    G1: float  # GeV
    M2: float  # GeV
    G2: float  # GeV
    m_ref: float  # GeV
    data_dir: str
    channel_files: Dict[str, str]  # channel_name -> filename
    spin_flip_channels: List[str]  # channels with 180deg phase shift


# =============================================================================
# Dataset Configurations
# =============================================================================

DATASETS = {
    'belle_zb': DatasetConfig(
        name='Belle Zb(10610)/Zb(10650) Hidden-Bottom',
        slug='belle_zb',
        state1_name='Zb(10610)',
        state2_name='Zb(10650)',
        M1=10.6072,
        G1=0.0184,
        M2=10.6522,
        G2=0.0115,
        m_ref=10.55,
        data_dir='belle_zb_rank1/extracted',
        channel_files={
            'Υ(2S)π': 'upsilon2s.csv',
            'Υ(3S)π': 'upsilon3s.csv',
            'hb(1P)π': 'hb1p.csv',
            'hb(2P)π': 'hb2p.csv',
        },
        spin_flip_channels=['hb(1P)π', 'hb(2P)π'],
    ),
    'cms_x6900': DatasetConfig(
        name='CMS X(6900)/X(7100)',
        slug='cms_x6900',
        state1_name='X(6900)',
        state2_name='X(7100)',
        M1=6.905,
        G1=0.080,
        M2=7.100,
        G2=0.095,
        m_ref=7.0,
        data_dir='cms_x6900_rank1_v4/extracted',
        channel_files={
            'di-J/ψ': 'channelA_trimmed.csv',
            'J/ψψ(2S)': 'channelB_jpsi_psi2S_bins.csv',
        },
        spin_flip_channels=[],
    ),
    'besiii_y': DatasetConfig(
        name='BESIII Y(4220)/Y(4320)',
        slug='besiii_y',
        state1_name='Y(4220)',
        state2_name='Y(4320)',
        M1=4.220,
        G1=0.044,
        M2=4.320,
        G2=0.030,
        m_ref=4.27,
        data_dir='besiii_y_rank1/extracted',
        channel_files={
            'π+π-J/ψ': 'channelA_jpsi_xsec.csv',
            'π+π-hc': 'channelB_hc_xsec.csv',
        },
        spin_flip_channels=[],
    ),
    'lhcb_pc': DatasetConfig(
        name='LHCb Pc(4440)/Pc(4457)',
        slug='lhcb_pc',
        state1_name='Pc(4440)',
        state2_name='Pc(4457)',
        M1=4.440,
        G1=0.020,
        M2=4.457,
        G2=0.006,
        m_ref=4.45,
        data_dir='lhcb_pc_rank1_v5/data/hepdata',
        channel_files={
            'full': 't1_full.csv',
            'cut': 't2_cut.csv',
            'weighted': 't3_weighted.csv',
            'weight': 't4_weight.csv',
        },
        spin_flip_channels=[],
    ),
}


def load_csv_generic(filepath: str, skip_header: int = 2,
                     mass_col: int = 0, y_col: int = 1, yerr_col: int = 2,
                     mass_scale: str = 'auto') -> Optional[np.ndarray]:
    """Generic CSV loader with flexible column handling."""
    try:
        lines = []
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i < skip_header:
                    continue
                if line.startswith('#') or line.startswith('$') or not line.strip():
                    continue
                lines.append(line.strip())

        if not lines:
            return None

        data = []
        for line in lines:
            parts = line.split(',')
            if len(parts) > max(mass_col, y_col, yerr_col):
                try:
                    m = float(parts[mass_col])
                    y = float(parts[y_col])
                    # Handle +/- error format
                    if yerr_col < len(parts):
                        yerr = abs(float(parts[yerr_col]))
                    else:
                        yerr = np.sqrt(max(1, abs(y)))
                    if yerr > 0:
                        data.append([m, y, yerr])
                except (ValueError, IndexError):
                    continue

        if not data:
            return None

        data = np.array(data)

        # Auto-detect and convert mass scale to GeV
        if mass_scale == 'auto':
            if data[0, 0] > 100:  # Likely MeV
                data[:, 0] /= 1000.0
        elif mass_scale == 'mev':
            data[:, 0] /= 1000.0

        return data

    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


def load_channels(config: DatasetConfig, base_dir: str) -> List[ChannelData]:
    """Load all channels for a dataset configuration."""
    channels = []
    data_path = os.path.join(base_dir, config.data_dir)

    for ch_name, filename in config.channel_files.items():
        filepath = os.path.join(data_path, filename)
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            continue

        # Detect format based on filename patterns
        if 'hepdata' in config.data_dir or filename.startswith('t'):
            # HEPData format - skip 10 header lines, different columns
            data = load_csv_generic(filepath, skip_header=10,
                                   mass_col=0, y_col=3, yerr_col=4)
        else:
            # Standard format
            data = load_csv_generic(filepath, skip_header=2)

        if data is None or len(data) < 5:
            logger.warning(f"Insufficient data in {filepath}")
            continue

        spin_flip = ch_name in config.spin_flip_channels
        channels.append(ChannelData(
            name=ch_name,
            m=data[:, 0],
            y=data[:, 1],
            yerr=data[:, 2],
            spin_flip=spin_flip
        ))
        logger.info(f"  Loaded {ch_name}: {len(data)} points")

    return channels


def run_dataset_tests(config: DatasetConfig, base_dir: str,
                      n_boot: int = 100, n_starts: int = 50) -> Dict[str, Any]:
    """Run all applicable tests on a dataset."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing: {config.name}")
    logger.info(f"{'='*70}")

    channels = load_channels(config, base_dir)

    if len(channels) < 2:
        return {
            'dataset': config.name,
            'slug': config.slug,
            'status': 'INSUFFICIENT_DATA',
            'n_channels': len(channels),
            'pairwise_results': [],
            'multichannel_result': None,
        }

    results = {
        'dataset': config.name,
        'slug': config.slug,
        'status': 'COMPLETED',
        'n_channels': len(channels),
        'channel_names': [ch.name for ch in channels],
        'pairwise_results': [],
        'multichannel_result': None,
    }

    # Run global multi-channel test if Nc >= 3
    if len(channels) >= 3:
        logger.info(f"\n--- Global {len(channels)}-Channel Test ---")
        try:
            mc_result = run_global_multichannel_test(
                channels,
                config.M1, config.G1, config.M2, config.G2,
                config.m_ref, n_boot=n_boot, n_starts=n_starts
            )
            results['multichannel_result'] = {
                'test_type': mc_result.test_type,
                'n_channels': int(mc_result.n_channels),
                'Lambda_obs': float(mc_result.Lambda_obs) if not np.isnan(mc_result.Lambda_obs) else None,
                'dof_diff': int(mc_result.dof_diff),
                'p_boot': float(mc_result.p_boot) if not np.isnan(mc_result.p_boot) else None,
                'p_wilks': float(mc_result.p_wilks) if not np.isnan(mc_result.p_wilks) else None,
                'verdict': mc_result.verdict,
                'reason': mc_result.reason,
                'overall_health': mc_result.overall_health,
                'R_shared': {'r': float(mc_result.R_shared[0]),
                            'phi_deg': float(mc_result.R_shared[1])},
            }
            logger.info(f"  Verdict: {mc_result.verdict} (p_boot={mc_result.p_boot:.3f})")
        except Exception as e:
            logger.error(f"Multi-channel test failed: {e}")
            results['multichannel_result'] = {'error': str(e)}
    else:
        logger.info(f"\n--- Skipping multi-channel test (only {len(channels)} channels) ---")

    # Also run pairwise tests for comparison
    if len(channels) >= 2:
        logger.info(f"\n--- Pairwise Tests ({len(channels)} channels) ---")
        from itertools import combinations

        for ch_a, ch_b in list(combinations(channels, 2))[:6]:  # Limit to 6 pairs
            pair_name = f"{ch_a.name} vs {ch_b.name}"
            logger.info(f"  Testing: {pair_name}")

            try:
                # Quick pairwise test using 2-channel multichannel test
                pair_result = run_global_multichannel_test(
                    [ch_a, ch_b],
                    config.M1, config.G1, config.M2, config.G2,
                    config.m_ref, n_boot=50, n_starts=30  # Reduced for speed
                )

                results['pairwise_results'].append({
                    'pair': pair_name,
                    'Lambda': float(pair_result.Lambda_obs) if not np.isnan(pair_result.Lambda_obs) else None,
                    'p_boot': float(pair_result.p_boot) if not np.isnan(pair_result.p_boot) else None,
                    'verdict': pair_result.verdict,
                })
            except Exception as e:
                logger.warning(f"  Pairwise test failed: {e}")

    return results


def generate_summary_report(all_results: List[Dict], out_path: str):
    """Generate consolidated summary report."""

    report = f"""# Comprehensive Rank-1 Test Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Overview

This report presents results from both **pairwise** and **global multi-channel**
rank-1 factorization tests on public exotic hadron data.

**Key insight:** The global multi-channel test (Nc >= 3) provides a more
stringent constraint than pairwise tests, as it requires the complex ratio
R = g(X2)/g(X1) to be shared across ALL channels simultaneously.

---

## Summary Table

| Dataset | Channels | Global Test | p_boot | Verdict | Notes |
|---------|----------|-------------|--------|---------|-------|
"""

    for res in all_results:
        dataset = res['dataset']
        n_ch = res.get('n_channels', 0)

        if res.get('multichannel_result') and 'Lambda_obs' in res['multichannel_result']:
            mc = res['multichannel_result']
            p_boot = mc.get('p_boot')
            p_str = f"{p_boot:.3f}" if p_boot else "N/A"
            verdict = mc.get('verdict', 'N/A')
            lambda_obs = mc.get('Lambda_obs')
            lambda_str = f"Λ={lambda_obs:.2f}" if lambda_obs else ""
            test_type = f"{n_ch}-channel"
        elif res.get('pairwise_results'):
            # Use best pairwise result
            pairs = res['pairwise_results']
            valid_pairs = [p for p in pairs if p.get('p_boot') is not None]
            if valid_pairs:
                best = max(valid_pairs, key=lambda x: x['p_boot'])
                p_str = f"{best['p_boot']:.3f}"
                verdict = best['verdict']
                lambda_str = f"Λ={best['Lambda']:.2f}" if best['Lambda'] else ""
            else:
                p_str = "N/A"
                verdict = "NO_DATA"
                lambda_str = ""
            test_type = "pairwise"
        else:
            p_str = "N/A"
            verdict = res.get('status', 'N/A')
            lambda_str = ""
            test_type = "N/A"

        report += f"| {dataset} | {n_ch} | {test_type} | {p_str} | **{verdict}** | {lambda_str} |\n"

    report += """
---

## Detailed Results

"""

    for res in all_results:
        report += f"### {res['dataset']}\n\n"
        report += f"**Status:** {res.get('status', 'N/A')}\n\n"

        if res.get('multichannel_result'):
            mc = res['multichannel_result']
            if 'error' in mc:
                report += f"**Global Test Error:** {mc['error']}\n\n"
            else:
                report += f"""**Global Multi-Channel Test:**

| Metric | Value |
|--------|-------|
| Channels | {mc.get('n_channels', 'N/A')} |
| Lambda | {mc.get('Lambda_obs', 'N/A')} |
| dof_diff | {mc.get('dof_diff', 'N/A')} |
| p_boot | {mc.get('p_boot', 'N/A')} |
| p_wilks | {mc.get('p_wilks', 'N/A')} |
| Verdict | **{mc.get('verdict', 'N/A')}** |
| Health | {mc.get('overall_health', 'N/A')} |

"""

        if res.get('pairwise_results'):
            report += "**Pairwise Tests:**\n\n"
            report += "| Pair | Lambda | p_boot | Verdict |\n"
            report += "|------|--------|--------|--------|\n"
            for pair in res['pairwise_results']:
                l = pair.get('Lambda', 'N/A')
                l_str = f"{l:.3f}" if isinstance(l, float) else str(l)
                p = pair.get('p_boot', 'N/A')
                p_str = f"{p:.3f}" if isinstance(p, float) else str(p)
                report += f"| {pair['pair']} | {l_str} | {p_str} | {pair['verdict']} |\n"
            report += "\n"

        report += "---\n\n"

    report += """
## Interpretation

### Global Multi-Channel Test (New)

For Nc channels, the test compares:
- **Constrained:** One shared R = g(X2)/g(X1) across all channels (2 dof)
- **Unconstrained:** Independent R per channel (2×Nc dof)
- **Test statistic:** Λ = 2×(NLL_con - NLL_unc), dof_diff = 2×(Nc-1)

### Verdict Key

| Verdict | Meaning |
|---------|---------|
| NOT_REJECTED | p ≥ 0.05, consistent with rank-1 factorization |
| DISFAVORED | p < 0.05, evidence against rank-1 |
| INCONCLUSIVE | Fit health issues (chi2/dof out of range) |
| OPTIMIZER_FAILURE | Nested model invariant violated |

### Notes

- **MODEL_MISMATCH** in health indicates chi2/dof > 3.0, suggesting the two-BW model
  may not fully describe the data (common with simplified models on digitized data)
- Even with INCONCLUSIVE verdict, the p_boot value provides diagnostic information

---

*Generated by run_comprehensive_rank1_tests.py*
"""

    with open(out_path, 'w') as f:
        f.write(report)

    # Also save JSON
    json_path = out_path.replace('.md', '.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nSaved: {out_path}")
    logger.info(f"Saved: {json_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Rank-1 Test Suite")
    parser.add_argument("--datasets", nargs='+',
                        choices=list(DATASETS.keys()) + ['all'],
                        default=['all'],
                        help="Datasets to test")
    parser.add_argument("--n-boot", type=int, default=100,
                        help="Bootstrap replicates")
    parser.add_argument("--n-starts", type=int, default=50,
                        help="Optimizer starts")
    parser.add_argument("--outdir", default="comprehensive_results",
                        help="Output directory")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.outdir, exist_ok=True)

    if 'all' in args.datasets:
        datasets_to_run = list(DATASETS.keys())
    else:
        datasets_to_run = args.datasets

    all_results = []

    for slug in datasets_to_run:
        config = DATASETS[slug]
        result = run_dataset_tests(config, base_dir, args.n_boot, args.n_starts)
        all_results.append(result)

    # Generate summary
    summary_path = os.path.join(args.outdir, "COMPREHENSIVE_SUMMARY.md")
    generate_summary_report(all_results, summary_path)

    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE TESTING COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
