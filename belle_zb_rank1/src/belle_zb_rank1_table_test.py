#!/usr/bin/env python3
"""
Belle Zb(10610)/Zb(10650) Rank-1 Test Using Table I Parameters

Uses the published fit parameters from arXiv:1110.2251 Table I directly.
This is more reliable than manually digitized spectra since Belle already
performed unbinned maximum likelihood fits.

Table I provides per-channel:
- Relative normalization aZ2/aZ1 = |R|
- Relative phase δZ2 - δZ1 = arg(R)
- With statistical and systematic uncertainties

The rank-1 hypothesis: R is the same across all channels.
"""

import numpy as np
from scipy import stats
import json
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class ChannelParams:
    """Parameters for one channel from Table I."""
    name: str
    r: float        # |R| = aZ2/aZ1
    r_err: float    # uncertainty on |R|
    phi: float      # arg(R) = δZ2 - δZ1 in degrees
    phi_err: float  # uncertainty on φ
    spin_flip: bool # True for hb channels

# Table I data from arXiv:1110.2251
# Combining stat and sys errors in quadrature
TABLE_I = {
    'upsilon1s': ChannelParams(
        name='Υ(1S)π⁺π⁻',
        r=0.57, r_err=np.sqrt(0.21**2 + 0.19**2),  # 0.28
        phi=58, phi_err=np.sqrt(43**2 + 4**2),  # 43
        spin_flip=False
    ),
    'upsilon2s': ChannelParams(
        name='Υ(2S)π⁺π⁻',
        r=0.86, r_err=np.sqrt(0.11**2 + 0.04**2),  # 0.12
        phi=-13, phi_err=np.sqrt(13**2 + 17**2),  # 21
        spin_flip=False
    ),
    'upsilon3s': ChannelParams(
        name='Υ(3S)π⁺π⁻',
        r=0.96, r_err=np.sqrt(0.14**2 + 0.08**2),  # 0.16
        phi=-9, phi_err=np.sqrt(19**2 + 11**2),  # 22
        spin_flip=False
    ),
    'hb1p': ChannelParams(
        name='hb(1P)π⁺π⁻',
        r=1.39, r_err=np.sqrt(0.37**2 + 0.05**2),  # 0.37
        phi=187, phi_err=np.sqrt(44**2 + 3**2),  # 44
        spin_flip=True
    ),
    'hb2p': ChannelParams(
        name='hb(2P)π⁺π⁻',
        r=1.6, r_err=np.sqrt(0.6**2 + 0.4**2),  # 0.72
        phi=181, phi_err=np.sqrt(65**2 + 74**2),  # 98
        spin_flip=True
    )
}


def complex_R(r, phi_deg):
    """Convert magnitude and phase (degrees) to complex R."""
    return r * np.exp(1j * np.radians(phi_deg))


def weighted_mean_complex(R_values, R_errors):
    """
    Compute weighted mean of complex numbers.
    Weights are inverse variance (1/σ²).
    """
    # Separate into real and imaginary parts
    reals = np.array([R.real for R in R_values])
    imags = np.array([R.imag for R in R_values])

    # Approximate errors on real/imag parts
    # For small errors, error on Re(R) ≈ error on |R| * cos(φ) - |R| * sin(φ) * σ_φ
    # Simplified: use |R| errors as proxy
    errors = np.array(R_errors)
    weights = 1.0 / errors**2

    # Weighted mean
    mean_real = np.sum(weights * reals) / np.sum(weights)
    mean_imag = np.sum(weights * imags) / np.sum(weights)

    R_mean = mean_real + 1j * mean_imag

    # Error on weighted mean
    err_mean = 1.0 / np.sqrt(np.sum(weights))

    return R_mean, err_mean


def chi2_test_consistency(channels: Dict[str, ChannelParams]) -> Tuple[float, float, int]:
    """
    Test if coupling ratios R are consistent across channels using χ² test.

    H0: All R values come from the same underlying R
    Test statistic: χ² = Σ (R_i - R_mean)² / σ_i²

    Returns: (chi2, p_value, dof)
    """
    # Get R values (accounting for 180° phase shift in hb channels)
    R_values = []
    R_mag_errors = []

    for ch in channels.values():
        phi = ch.phi
        # Account for spin-flip: hb channels have ~180° phase difference
        # Normalize all to same convention
        if ch.spin_flip:
            phi = phi - 180  # Bring to same convention as Υ channels

        R = complex_R(ch.r, phi)
        R_values.append(R)
        R_mag_errors.append(ch.r_err)

    R_values = np.array(R_values)
    R_mag_errors = np.array(R_mag_errors)

    # Compute weighted mean
    R_mean, _ = weighted_mean_complex(R_values, R_mag_errors)

    # Chi-squared: compare magnitudes (more robust)
    r_values = np.array([np.abs(R) for R in R_values])
    r_mean = np.abs(R_mean)

    chi2_r = np.sum(((r_values - r_mean) / R_mag_errors)**2)

    # Compare phases (wrap to [-180, 180])
    phi_values = np.array([np.angle(R, deg=True) for R in R_values])
    phi_mean = np.angle(R_mean, deg=True)
    phi_errors = np.array([ch.phi_err for ch in channels.values()])

    # Wrap phase differences
    phi_diff = phi_values - phi_mean
    phi_diff = (phi_diff + 180) % 360 - 180  # Wrap to [-180, 180]

    chi2_phi = np.sum((phi_diff / phi_errors)**2)

    # Total chi2 (treating r and phi as independent)
    chi2_total = chi2_r + chi2_phi

    # Degrees of freedom: 2 × (n_channels - 1) for complex R constraint
    # (each channel loses 2 dof: |R| and arg(R))
    dof = 2 * (len(channels) - 1)

    p_value = 1 - stats.chi2.cdf(chi2_total, dof)

    return chi2_total, p_value, dof


def bootstrap_test(channels: Dict[str, ChannelParams], n_boot: int = 1000) -> Tuple[float, float]:
    """
    Bootstrap test for rank-1 consistency.

    Resample from Gaussian distributions around measured R values.
    """
    np.random.seed(42)

    n_channels = len(channels)
    chi2_null = []

    # Get nominal values and errors
    channel_list = list(channels.values())

    for _ in range(n_boot):
        # Resample each channel's R
        R_boot = []
        R_err_boot = []

        for ch in channel_list:
            r_sample = np.random.normal(ch.r, ch.r_err)
            r_sample = max(0.01, r_sample)  # Keep positive

            phi_sample = np.random.normal(ch.phi, ch.phi_err)

            # Account for spin-flip
            if ch.spin_flip:
                phi_sample = phi_sample - 180

            R = complex_R(r_sample, phi_sample)
            R_boot.append(R)
            R_err_boot.append(ch.r_err)

        # Compute chi2 for this bootstrap sample
        R_boot = np.array(R_boot)
        R_mean, _ = weighted_mean_complex(R_boot, R_err_boot)

        # Chi2 based on magnitude only (more robust)
        r_boot = np.array([np.abs(R) for R in R_boot])
        r_mean = np.abs(R_mean)
        chi2_r = np.sum(((r_boot - r_mean) / np.array(R_err_boot))**2)

        chi2_null.append(chi2_r)

    chi2_null = np.array(chi2_null)

    # Observed chi2
    chi2_obs, _, _ = chi2_test_consistency(channels)

    # p-value from bootstrap
    k_exceed = np.sum(chi2_null >= chi2_obs)
    p_boot = k_exceed / n_boot

    return chi2_obs, p_boot


def run_analysis():
    """Run the rank-1 test using Table I parameters."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, 'out')

    print("=" * 70)
    print("Belle Zb Rank-1 Test Using Table I Parameters")
    print("=" * 70)

    # Test 1: All 5 channels
    print("\n" + "=" * 70)
    print("TEST 1: All 5 channels (accounting for 180° spin-flip)")
    print("=" * 70)

    chi2_all, p_all, dof_all = chi2_test_consistency(TABLE_I)
    print(f"\nχ² = {chi2_all:.2f}, dof = {dof_all}, p = {p_all:.4f}")

    # Test 2: Υ channels only (same spin state)
    upsilon_channels = {k: v for k, v in TABLE_I.items() if not v.spin_flip}
    print("\n" + "=" * 70)
    print("TEST 2: Υ channels only (Υ(1S), Υ(2S), Υ(3S))")
    print("=" * 70)

    chi2_ups, p_ups, dof_ups = chi2_test_consistency(upsilon_channels)
    print(f"\nχ² = {chi2_ups:.2f}, dof = {dof_ups}, p = {p_ups:.4f}")

    # Test 3: hb channels only (same spin state)
    hb_channels = {k: v for k, v in TABLE_I.items() if v.spin_flip}
    print("\n" + "=" * 70)
    print("TEST 3: hb channels only (hb(1P), hb(2P))")
    print("=" * 70)

    chi2_hb, p_hb, dof_hb = chi2_test_consistency(hb_channels)
    print(f"\nχ² = {chi2_hb:.2f}, dof = {dof_hb}, p = {p_hb:.4f}")

    # Bootstrap test for Υ channels
    print("\n" + "=" * 70)
    print("Bootstrap test for Υ channels")
    print("=" * 70)
    chi2_boot, p_boot = bootstrap_test(upsilon_channels, n_boot=1000)
    print(f"χ²_obs = {chi2_boot:.2f}, p_boot = {p_boot:.4f}")

    # Determine verdicts
    def get_verdict(p):
        if p < 0.05:
            return "DISFAVORED"
        else:
            return "NOT_REJECTED"

    verdict_ups = get_verdict(p_ups)
    verdict_hb = get_verdict(p_hb)
    verdict_all = get_verdict(p_all)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Print per-channel R values
    print("\nPer-channel coupling ratios R = g(Zb10650)/g(Zb10610):")
    print("-" * 50)
    for name, ch in TABLE_I.items():
        phi_adj = ch.phi - 180 if ch.spin_flip else ch.phi
        print(f"  {ch.name}: |R| = {ch.r:.2f} ± {ch.r_err:.2f}, "
              f"φ = {phi_adj:.0f}° ± {ch.phi_err:.0f}°"
              f" {'(spin-flip adjusted)' if ch.spin_flip else ''}")

    print("\n" + "-" * 50)
    print(f"Υ channels only: χ² = {chi2_ups:.2f}, p = {p_ups:.4f} → {verdict_ups}")
    print(f"hb channels only: χ² = {chi2_hb:.2f}, p = {p_hb:.4f} → {verdict_hb}")
    print(f"All 5 channels:   χ² = {chi2_all:.2f}, p = {p_all:.4f} → {verdict_all}")
    print("-" * 50)

    # Primary result: Υ channels (cleanest, same spin state)
    print(f"\n*** PRIMARY RESULT (Υ channels): {verdict_ups} (p = {p_ups:.4f}) ***")

    # Generate report
    generate_report(TABLE_I, chi2_ups, p_ups, dof_ups, verdict_ups,
                    chi2_hb, p_hb, chi2_all, p_all, out_dir)

    # Generate plots
    generate_plots(TABLE_I, out_dir)

    return verdict_ups, p_ups


def generate_plots(table_data, out_dir):
    """Generate visualization plots."""

    # Plot 1: R magnitude and phase for all channels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    channels = list(table_data.keys())
    names = [table_data[c].name for c in channels]
    r_vals = [table_data[c].r for c in channels]
    r_errs = [table_data[c].r_err for c in channels]
    phi_vals = [table_data[c].phi for c in channels]
    phi_errs = [table_data[c].phi_err for c in channels]
    spin_flip = [table_data[c].spin_flip for c in channels]

    colors = ['blue' if not sf else 'green' for sf in spin_flip]

    # Magnitude
    ax = axes[0]
    x = np.arange(len(channels))
    ax.bar(x, r_vals, yerr=r_errs, color=colors, alpha=0.7, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split('⁺')[0] for n in names], rotation=45, ha='right')
    ax.set_ylabel('|R| = aZ₂/aZ₁', fontsize=12)
    ax.set_title('Coupling Ratio Magnitude', fontsize=13)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Phase (adjusted for spin-flip)
    ax = axes[1]
    phi_adj = [p - 180 if sf else p for p, sf in zip(phi_vals, spin_flip)]
    ax.errorbar(x, phi_adj, yerr=phi_errs, fmt='o', color='red', markersize=10, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split('⁺')[0] for n in names], rotation=45, ha='right')
    ax.set_ylabel('arg(R) (degrees)', fontsize=12)
    ax.set_title('Coupling Ratio Phase (spin-flip adjusted)', fontsize=13)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Υ channels'),
                       Patch(facecolor='green', alpha=0.7, label='hb channels (spin-flip)')]
    axes[0].legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'coupling_ratios.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Complex plane visualization
    fig, ax = plt.subplots(figsize=(8, 8))

    for ch_name, ch in table_data.items():
        phi = ch.phi - 180 if ch.spin_flip else ch.phi
        R = complex_R(ch.r, phi)

        color = 'blue' if not ch.spin_flip else 'green'

        # Plot point
        ax.plot(R.real, R.imag, 'o', markersize=12, color=color)

        # Error ellipse (approximate)
        circle = plt.Circle((R.real, R.imag), ch.r_err, fill=False,
                             color=color, linestyle='--', alpha=0.5)
        ax.add_patch(circle)

        # Label
        ax.annotate(ch.name.split('⁺')[0], (R.real, R.imag),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Re(R)', fontsize=12)
    ax.set_ylabel('Im(R)', fontsize=12)
    ax.set_title('Complex Coupling Ratio R in Complex Plane\n(hb phases shifted by 180°)', fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'complex_plane.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {out_dir}/")


def generate_report(table_data, chi2_ups, p_ups, dof_ups, verdict_ups,
                    chi2_hb, p_hb, chi2_all, p_all, out_dir):
    """Generate markdown report."""

    report = f"""# Belle Zb(10610)/Zb(10650) Rank-1 Factorization Test

## Executive Summary

**Primary Result (Υ channels): {verdict_ups}**

| Test | χ² | dof | p-value | Verdict |
|------|-----|-----|---------|---------|
| Υ(1S,2S,3S)π | {chi2_ups:.2f} | {dof_ups} | {p_ups:.4f} | **{verdict_ups}** |
| hb(1P,2P)π | {chi2_hb:.2f} | 2 | {p_hb:.4f} | {'NOT_REJECTED' if p_hb >= 0.05 else 'DISFAVORED'} |
| All 5 channels | {chi2_all:.2f} | 8 | {p_all:.4f} | {'NOT_REJECTED' if p_all >= 0.05 else 'DISFAVORED'} |

---

## Method

This analysis uses the published fit parameters from **Table I of arXiv:1110.2251** directly,
rather than re-fitting the spectra. Belle performed unbinned maximum likelihood fits to
extract the coupling ratio R = g(Zb10650)/g(Zb10610) per channel.

The rank-1 hypothesis states that R should be identical across all decay channels.

### Spin-Flip Correction

The hb(mP)π channels involve heavy-quark spin-flip and show a ~180° phase shift
relative to the Υ(nS)π channels. This is expected from theory and observed in the data.
We account for this by subtracting 180° from the hb phases before comparison.

---

## Per-Channel Coupling Ratios

| Channel | |R| | σ(|R|) | φ (deg) | σ(φ) | Spin-flip |
|---------|-----|-------|---------|-------|-----------|
"""

    for name, ch in table_data.items():
        phi_adj = ch.phi - 180 if ch.spin_flip else ch.phi
        report += f"| {ch.name} | {ch.r:.2f} | {ch.r_err:.2f} | {phi_adj:.0f} | {ch.phi_err:.0f} | {'Yes' if ch.spin_flip else 'No'} |\n"

    report += f"""
---

## Interpretation

### Υ Channels (Primary Test)

The three Υ(nS)π channels show:
- |R| values: 0.57, 0.86, 0.96 (increasing trend)
- Phases: 58°, -13°, -9° (roughly consistent near 0°)

The χ² test gives p = {p_ups:.4f}, meaning we **{'cannot reject' if p_ups >= 0.05 else 'reject'}**
the hypothesis that R is the same across Υ channels at the 5% level.

### hb Channels

The two hb(mP)π channels show:
- |R| values: 1.39, 1.6 (larger than Υ channels)
- Phases: 7°, 1° after spin-flip adjustment (consistent)

### Physical Implications

The coupling ratios are approximately consistent within the Υ family and within
the hb family, but differ between families. This pattern is physically expected:
the Zb → Υπ transitions conserve heavy-quark spin, while Zb → hbπ require spin-flip.

---

## Data Source

Belle Collaboration, arXiv:1110.2251
"Observation of two charged bottomonium-like resonances in Υ(5S) decays"

---

## Visualizations

### Coupling Ratio Magnitudes and Phases

![Coupling Ratios](coupling_ratios.png)

### Complex Plane Representation

![Complex Plane](complex_plane.png)

---

## Files Generated

- `REPORT.md` - This report
- `coupling_ratios.png` - Per-channel R values
- `complex_plane.png` - R in complex plane
- `result_table.json` - Machine-readable results

---

*Generated by belle_zb_rank1_table_test.py*
"""

    with open(os.path.join(out_dir, 'REPORT.md'), 'w') as f:
        f.write(report)

    # Save JSON
    result = {
        'verdict_upsilon': verdict_ups,
        'chi2_upsilon': chi2_ups,
        'p_upsilon': p_ups,
        'dof_upsilon': dof_ups,
        'chi2_hb': chi2_hb,
        'p_hb': p_hb,
        'chi2_all': chi2_all,
        'p_all': p_all,
        'table_params': {
            name: {
                'r': ch.r, 'r_err': ch.r_err,
                'phi': ch.phi, 'phi_err': ch.phi_err,
                'spin_flip': ch.spin_flip
            }
            for name, ch in table_data.items()
        }
    }

    with open(os.path.join(out_dir, 'result_table.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nReport saved to: {out_dir}/REPORT.md")


if __name__ == '__main__':
    run_analysis()
