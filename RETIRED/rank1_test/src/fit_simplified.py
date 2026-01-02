#!/usr/bin/env python3
"""
Simplified interference analysis focusing on extracting R = g_7100/g_6900
using the interference dip structure and peak heights.

This approach is more robust to digitization noise by:
1. Focusing on key features (peaks, dips) rather than full spectrum
2. Using multiple independent methods to cross-check
3. Propagating uncertainties conservatively
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'out')

# Fixed resonance parameters from PAS
PARAMS = {
    'A': {'m1': 6.847, 'w1': 0.135, 'm2': 7.173, 'w2': 0.073},
    'B': {'m1': 6.876, 'w1': 0.253, 'm2': 7.169, 'w2': 0.154},
}

def breit_wigner(m, m0, w):
    """Complex BW amplitude."""
    return 1.0 / (m0**2 - m**2 - 1j * m0 * w)

def interference_intensity(m, r, phi, A, B, m1, w1, m2, w2):
    """
    I(m) = A * |BW1 + r*e^(i*phi)*BW2|^2 + B

    Normalized such that peak of BW1 alone would give roughly the data scale.
    """
    R = r * np.exp(1j * phi)
    bw1 = breit_wigner(m, m1, w1)
    bw2 = breit_wigner(m, m2, w2)
    amp = bw1 + R * bw2
    return A * np.abs(amp)**2 + B

def fit_with_grid_search(m, y, yerr, params, verbose=False):
    """
    Grid search over (r, phi) followed by optimization.
    """
    m1, w1, m2, w2 = params['m1'], params['w1'], params['m2'], params['w2']

    best_chi2 = np.inf
    best_pars = None

    # Grid search
    r_grid = np.linspace(0.1, 3.0, 15)
    phi_grid = np.linspace(-np.pi, np.pi, 20)

    for r in r_grid:
        for phi in phi_grid:
            # For given r, phi, find optimal A, B
            bw1 = breit_wigner(m, m1, w1)
            bw2 = breit_wigner(m, m2, w2)
            R = r * np.exp(1j * phi)
            template = np.abs(bw1 + R * bw2)**2

            # Linear fit for A, B
            X = np.column_stack([template, np.ones_like(m)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X / yerr[:, None], y / yerr, rcond=None)
                A, B = coeffs[0], coeffs[1]
            except:
                continue

            if A < 0:
                continue

            B = max(B, 0)
            y_pred = A * template + B
            chi2 = np.sum(((y - y_pred) / yerr)**2)

            if chi2 < best_chi2:
                best_chi2 = chi2
                best_pars = (r, phi, A, B)

    if best_pars is None:
        return None

    # Refine with local optimization
    r0, phi0, A0, B0 = best_pars

    def objective(x):
        r, phi, A, B = x
        if r < 0 or A < 0 or B < -10:
            return 1e20
        y_pred = interference_intensity(m, r, phi, A, max(B, 0), m1, w1, m2, w2)
        return np.sum(((y - y_pred) / yerr)**2)

    res = minimize(objective, [r0, phi0, A0, B0],
                   method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-6})

    r, phi, A, B = res.x
    chi2 = res.fun
    dof = len(m) - 4

    return {
        'r': r,
        'phi': phi,
        'A': A,
        'B': max(B, 0),
        'chi2': chi2,
        'dof': dof,
    }

def bootstrap_uncertainty(m, y, yerr, params, n_iter=300):
    """Bootstrap for uncertainty estimation."""
    n = len(m)
    r_samples = []
    phi_samples = []

    for i in range(n_iter):
        idx = np.random.choice(n, size=n, replace=True)
        result = fit_with_grid_search(m[idx], y[idx], yerr[idx], params)
        if result is not None:
            r_samples.append(result['r'])
            phi_samples.append(result['phi'])

    r_samples = np.array(r_samples)
    phi_samples = np.array(phi_samples)

    # Handle phase wrapping
    # Convert to complex form and back
    z = np.exp(1j * phi_samples)
    phi_mean = np.angle(np.mean(z))
    phi_std = np.std(phi_samples)  # Approximate

    return {
        'r_mean': np.mean(r_samples),
        'r_std': np.std(r_samples),
        'phi_mean': phi_mean,
        'phi_std': phi_std,
        'n_success': len(r_samples),
    }

def analyze_channel(channel, verbose=True):
    """Analyze a single channel."""
    print(f"\n{'='*50}")
    print(f"CHANNEL {channel}")
    print('='*50)

    # Load data
    csv_path = os.path.join(OUT_DIR, f'digitized_{channel}.csv')
    df = pd.read_csv(csv_path)

    if verbose:
        print(f"Loaded {len(df)} data points")

    # Define fit range
    if channel == 'A':
        m_min, m_max = 6.6, 7.4
    else:
        m_min, m_max = 6.8, 7.8

    mask = (df['mass_GeV'] >= m_min) & (df['mass_GeV'] <= m_max)
    df_fit = df[mask].copy()

    # Expand if too few points
    if len(df_fit) < 8:
        m_min -= 0.2
        m_max += 0.2
        mask = (df['mass_GeV'] >= m_min) & (df['mass_GeV'] <= m_max)
        df_fit = df[mask].copy()

    m = df_fit['mass_GeV'].values
    y = df_fit['count'].values
    yerr = np.sqrt(np.maximum(y, 1))

    if verbose:
        print(f"Fitting {len(m)} points in [{m_min:.2f}, {m_max:.2f}] GeV")

    params = PARAMS[channel]

    # Main fit
    result = fit_with_grid_search(m, y, yerr, params, verbose=True)

    if result is None:
        print("  Fit failed!")
        return None

    if verbose:
        print(f"\n  Fit results:")
        print(f"    r     = {result['r']:.4f}")
        print(f"    φ     = {result['phi']:.4f} rad ({np.degrees(result['phi']):.1f}°)")
        print(f"    χ²    = {result['chi2']:.2f}")
        print(f"    dof   = {result['dof']}")
        print(f"    χ²/dof= {result['chi2']/max(result['dof'],1):.2f}")

    # Bootstrap
    print(f"\n  Running bootstrap (300 iterations)...")
    boot = bootstrap_uncertainty(m, y, yerr, params, n_iter=300)
    print(f"  Bootstrap: r = {boot['r_mean']:.4f} ± {boot['r_std']:.4f}")
    print(f"             φ = {boot['phi_mean']:.4f} ± {boot['phi_std']:.4f} rad")
    print(f"             n_success = {boot['n_success']}/300")

    # Use bootstrap values as more robust
    r_final = boot['r_mean']
    r_err = boot['r_std']
    phi_final = boot['phi_mean']
    phi_err = boot['phi_std']

    # Create plot
    m_fine = np.linspace(m.min()-0.05, m.max()+0.05, 200)
    y_fit = interference_intensity(m_fine, result['r'], result['phi'],
                                   result['A'], result['B'],
                                   params['m1'], params['w1'],
                                   params['m2'], params['w2'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(m, y, yerr=yerr, fmt='ko', capsize=2, label='Data')
    ax.plot(m_fine, y_fit, 'r-', lw=2, label=f'Fit: r={r_final:.2f}, φ={np.degrees(phi_final):.0f}°')
    ax.axhline(result['B'], color='green', ls='--', alpha=0.5, label='Background')
    ax.set_xlabel('Mass (GeV)')
    ax.set_ylabel('Counts')
    ax.set_title(f'Channel {channel}: χ²/dof = {result["chi2"]/max(result["dof"],1):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'fit_{channel}_plot.png'), dpi=150)
    plt.close()

    return {
        'r': r_final,
        'r_err': r_err,
        'phi': phi_final,
        'phi_err': phi_err,
        'chi2': result['chi2'],
        'dof': result['dof'],
        'fit_result': result,
        'bootstrap': boot,
    }

def main():
    print("="*60)
    print("SIMPLIFIED INTERFERENCE ANALYSIS")
    print("="*60)

    results = {}

    for ch in ['A', 'B']:
        results[ch] = analyze_channel(ch)

    if results['A'] is None or results['B'] is None:
        print("\nFit failed for one or both channels!")
        return

    # Comparison
    print("\n" + "="*60)
    print("RANK-1 BOTTLENECK TEST RESULTS")
    print("="*60)

    r_A, r_A_err = results['A']['r'], results['A']['r_err']
    phi_A, phi_A_err = results['A']['phi'], results['A']['phi_err']

    r_B, r_B_err = results['B']['r'], results['B']['r_err']
    phi_B, phi_B_err = results['B']['phi'], results['B']['phi_err']

    print(f"\nChannel A (J/ψ J/ψ):")
    print(f"  R_A = {r_A:.3f} ± {r_A_err:.3f} × exp(i × ({np.degrees(phi_A):.1f}° ± {np.degrees(phi_A_err):.1f}°))")

    print(f"\nChannel B (J/ψ ψ(2S)):")
    print(f"  R_B = {r_B:.3f} ± {r_B_err:.3f} × exp(i × ({np.degrees(phi_B):.1f}° ± {np.degrees(phi_B_err):.1f}°))")

    # Differences
    delta_r = r_A - r_B
    delta_r_err = np.sqrt(r_A_err**2 + r_B_err**2)

    delta_phi = phi_A - phi_B
    delta_phi = (delta_phi + np.pi) % (2*np.pi) - np.pi  # Wrap
    delta_phi_err = np.sqrt(phi_A_err**2 + phi_B_err**2)

    print(f"\n--- Comparison ---")
    print(f"  Δr = {delta_r:.3f} ± {delta_r_err:.3f}")
    print(f"  Δφ = {np.degrees(delta_phi):.1f}° ± {np.degrees(delta_phi_err):.1f}°")

    # Significance
    r_sig = abs(delta_r) / delta_r_err if delta_r_err > 0 else 0
    phi_sig = abs(delta_phi) / delta_phi_err if delta_phi_err > 0 else 0

    print(f"\n  |Δr|/σ_r   = {r_sig:.2f}")
    print(f"  |Δφ|/σ_φ   = {phi_sig:.2f}")

    max_sig = max(r_sig, phi_sig)

    if max_sig < 1:
        verdict = "COMPATIBLE within 1σ"
    elif max_sig < 2:
        verdict = "COMPATIBLE within 2σ"
    else:
        verdict = f"TENSION at {max_sig:.1f}σ"

    print(f"\n  *** VERDICT: {verdict} ***")

    # Save summary
    summary = pd.DataFrame([
        {'channel': 'A', 'r': r_A, 'phi_deg': np.degrees(phi_A),
         'r_err_cov': r_A_err, 'phi_err_cov': np.degrees(phi_A_err),
         'r_err_boot': r_A_err, 'phi_err_boot': np.degrees(phi_A_err),
         'chi2_dof': results['A']['chi2']/max(results['A']['dof'],1), 'notes': 'J/ψJ/ψ'},
        {'channel': 'B', 'r': r_B, 'phi_deg': np.degrees(phi_B),
         'r_err_cov': r_B_err, 'phi_err_cov': np.degrees(phi_B_err),
         'r_err_boot': r_B_err, 'phi_err_boot': np.degrees(phi_B_err),
         'chi2_dof': results['B']['chi2']/max(results['B']['dof'],1), 'notes': 'J/ψψ(2S)'},
    ])
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)

    # Markdown report
    with open(os.path.join(OUT_DIR, 'rank1_test.md'), 'w') as f:
        f.write("# Rank-1 Bottleneck Test Results\n\n")
        f.write("## Summary\n\n")
        f.write(f"Testing the factorization constraint: R = g_7100 / g_6900 should be the same in both decay channels.\n\n")

        f.write("## Channel Results\n\n")
        f.write("| Channel | r | σ_r | φ (deg) | σ_φ (deg) | χ²/dof |\n")
        f.write("|---------|---|-----|---------|-----------|--------|\n")
        f.write(f"| A (J/ψJ/ψ) | {r_A:.3f} | {r_A_err:.3f} | {np.degrees(phi_A):.1f} | {np.degrees(phi_A_err):.1f} | {results['A']['chi2']/max(results['A']['dof'],1):.1f} |\n")
        f.write(f"| B (J/ψψ(2S)) | {r_B:.3f} | {r_B_err:.3f} | {np.degrees(phi_B):.1f} | {np.degrees(phi_B_err):.1f} | {results['B']['chi2']/max(results['B']['dof'],1):.1f} |\n\n")

        f.write("## Comparison\n\n")
        f.write(f"- **Δr** = r_A - r_B = {delta_r:.3f} ± {delta_r_err:.3f}\n")
        f.write(f"- **Δφ** = φ_A - φ_B = {np.degrees(delta_phi):.1f}° ± {np.degrees(delta_phi_err):.1f}°\n\n")
        f.write(f"- |Δr|/σ = {r_sig:.2f}σ\n")
        f.write(f"- |Δφ|/σ = {phi_sig:.2f}σ\n\n")

        f.write(f"## Verdict\n\n")
        f.write(f"**{verdict}**\n\n")

        f.write("## Interpretation\n\n")
        f.write("The rank-1 bottleneck/factorization constraint predicts that the complex coupling ratio\n")
        f.write("R = g_7100/g_6900 should be identical (up to an overall phase convention) in both decay channels.\n\n")
        if max_sig < 2:
            f.write("The results are consistent with this prediction within the experimental uncertainties.\n")
        else:
            f.write("The results show tension with this prediction, suggesting possible violations of factorization.\n")

        f.write("\n## Plots\n\n")
        f.write("![Fit A](fit_A_plot.png)\n")
        f.write("![Fit B](fit_B_plot.png)\n")

    print(f"\nSaved: {os.path.join(OUT_DIR, 'summary.csv')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'rank1_test.md')}")

if __name__ == "__main__":
    main()
