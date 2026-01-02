#!/usr/bin/env python3
"""
Improved coherent interference fit with better normalization.
Uses a scale-independent approach to extract R = g_7100/g_6900.
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'out')

# Fixed resonance parameters from PAS documents (in GeV)
RESONANCE_PARAMS = {
    'A': {  # BPH-24-003 J/ψJ/ψ
        'm6900': 6.847, 'w6900': 0.135,
        'm7100': 7.173, 'w7100': 0.073,
    },
    'B': {  # BPH-22-004 J/ψψ(2S)
        'm6900': 6.876, 'w6900': 0.253,
        'm7100': 7.169, 'w7100': 0.154,
    }
}

# Fit windows
FIT_WINDOWS = {
    'A': (6.6, 7.5),
    'B': (6.9, 7.8),  # Wider window for channel B
}

def breit_wigner(m, m0, w):
    """Breit-Wigner amplitude."""
    return 1.0 / (m0**2 - m**2 - 1j * m0 * w)

def model_intensity(m, r, phi, A, bg_c, bg_s, params):
    """
    Coherent interference intensity model.

    I(m) = A * |BW_6900(m) + r*exp(i*phi)*BW_7100(m)|^2 + background

    Parameters:
    - r: magnitude ratio
    - phi: relative phase
    - A: amplitude scale
    - bg_c, bg_s: background (constant + slope)
    """
    R = r * np.exp(1j * phi)

    bw1 = breit_wigner(m, params['m6900'], params['w6900'])
    bw2 = breit_wigner(m, params['m7100'], params['w7100'])

    coherent = bw1 + R * bw2
    signal = A * np.abs(coherent)**2

    m_ref = 7.0
    background = np.maximum(bg_c + bg_s * (m - m_ref), 0)

    return signal + background

def chi2_func(params, m, y, yerr, res_params):
    """Chi-squared function."""
    r, phi, A, bg_c, bg_s = params

    if r < 0 or A < 0:
        return 1e20

    y_model = model_intensity(m, r, phi, A, bg_c, bg_s, res_params)

    # Weighted residuals
    chi2 = np.sum(((y - y_model) / yerr) ** 2)

    return chi2

def fit_channel(df, channel, verbose=True):
    """Fit a channel with the interference model."""
    res_params = RESONANCE_PARAMS[channel]
    fit_min, fit_max = FIT_WINDOWS[channel]

    # Filter data
    mask = (df['mass_GeV'] >= fit_min) & (df['mass_GeV'] <= fit_max)
    df_fit = df[mask].copy()

    if len(df_fit) < 4:
        if verbose:
            print(f"  Warning: Only {len(df_fit)} points, expanding window...")
        fit_min -= 0.3
        fit_max += 0.3
        mask = (df['mass_GeV'] >= fit_min) & (df['mass_GeV'] <= fit_max)
        df_fit = df[mask].copy()

    m = df_fit['mass_GeV'].values
    y = df_fit['count'].values
    yerr = np.sqrt(np.maximum(y, 1))

    if verbose:
        print(f"  Fitting {len(m)} points in [{fit_min:.2f}, {fit_max:.2f}] GeV")
        print(f"  Data: y_min={y.min():.1f}, y_max={y.max():.1f}")

    # Estimate initial scale
    # At the BW peak, |BW|^2 ~ 1/w^2
    # So A ~ y_max * w6900^2
    A_init = y.max() * res_params['w6900']**2 * 1e-2

    # Bounds
    bounds = [
        (0.01, 10.0),        # r
        (-np.pi, np.pi),     # phi
        (A_init * 0.001, A_init * 1000),  # A
        (0, y.max() * 0.5),  # bg_c
        (-50, 50),           # bg_s
    ]

    # Global optimization
    result = differential_evolution(
        chi2_func,
        bounds,
        args=(m, y, yerr, res_params),
        seed=42,
        maxiter=2000,
        tol=1e-8,
        polish=True,
        workers=1
    )

    r_fit, phi_fit, A_fit, bg_c_fit, bg_s_fit = result.x
    chi2_min = result.fun
    dof = len(m) - 5

    if verbose:
        print(f"\n  Results:")
        print(f"    r     = {r_fit:.4f}")
        print(f"    phi   = {phi_fit:.4f} rad ({np.degrees(phi_fit):.1f}°)")
        print(f"    A     = {A_fit:.4e}")
        print(f"    bg_c  = {bg_c_fit:.2f}")
        print(f"    bg_s  = {bg_s_fit:.4f}")
        print(f"    chi2  = {chi2_min:.2f}, dof={dof}, chi2/dof={chi2_min/max(dof,1):.2f}")

    # Compute Hessian-based uncertainties
    eps = 1e-5
    x0 = result.x
    n = len(x0)
    hess = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            xpp, xpm, xmp, xmm = [x0.copy() for _ in range(4)]
            hi = max(abs(x0[i]) * eps, eps)
            hj = max(abs(x0[j]) * eps, eps)

            xpp[i] += hi; xpp[j] += hj
            xpm[i] += hi; xpm[j] -= hj
            xmp[i] -= hi; xmp[j] += hj
            xmm[i] -= hi; xmm[j] -= hj

            hess[i,j] = (chi2_func(xpp, m, y, yerr, res_params)
                        - chi2_func(xpm, m, y, yerr, res_params)
                        - chi2_func(xmp, m, y, yerr, res_params)
                        + chi2_func(xmm, m, y, yerr, res_params)) / (4*hi*hj)
            hess[j,i] = hess[i,j]

    try:
        cov = 2.0 * np.linalg.inv(hess)
        errs = np.sqrt(np.abs(np.diag(cov)))
        r_err_cov = errs[0]
        phi_err_cov = errs[1]
    except:
        r_err_cov = 1.0
        phi_err_cov = 1.0

    return {
        'r': r_fit,
        'phi': phi_fit,
        'A': A_fit,
        'bg_c': bg_c_fit,
        'bg_s': bg_s_fit,
        'chi2': chi2_min,
        'dof': dof,
        'r_err_cov': r_err_cov,
        'phi_err_cov': phi_err_cov,
        'm': m,
        'y': y,
        'yerr': yerr,
        'res_params': res_params,
        'fit_window': [fit_min, fit_max]
    }

def bootstrap(df, channel, n_iter=300, verbose=False):
    """Bootstrap uncertainty estimation."""
    res_params = RESONANCE_PARAMS[channel]
    fit_min, fit_max = FIT_WINDOWS[channel]

    mask = (df['mass_GeV'] >= fit_min) & (df['mass_GeV'] <= fit_max)
    df_fit = df[mask].copy()

    if len(df_fit) < 4:
        fit_min -= 0.3
        fit_max += 0.3
        mask = (df['mass_GeV'] >= fit_min) & (df['mass_GeV'] <= fit_max)
        df_fit = df[mask].copy()

    m = df_fit['mass_GeV'].values
    y = df_fit['count'].values
    yerr = np.sqrt(np.maximum(y, 1))
    n_pts = len(m)

    A_init = y.max() * res_params['w6900']**2 * 1e-2
    bounds = [
        (0.01, 10.0),
        (-np.pi, np.pi),
        (A_init * 0.001, A_init * 1000),
        (0, y.max() * 0.5),
        (-50, 50),
    ]

    r_samples = []
    phi_samples = []

    for i in range(n_iter):
        idx = np.random.choice(n_pts, size=n_pts, replace=True)

        try:
            res = differential_evolution(
                chi2_func, bounds,
                args=(m[idx], y[idx], yerr[idx], res_params),
                seed=i,
                maxiter=500,
                tol=1e-6,
                polish=False,
                workers=1
            )
            r_samples.append(res.x[0])
            phi_samples.append(res.x[1])
        except:
            pass

        if verbose and (i+1) % 100 == 0:
            print(f"    Bootstrap: {i+1}/{n_iter}")

    r_err = np.std(r_samples)
    phi_err = np.std(phi_samples)

    return r_err, phi_err, np.array(r_samples), np.array(phi_samples)

def make_plot(result, channel, path):
    """Create fit plot."""
    m = result['m']
    y = result['y']
    yerr = result['yerr']
    rp = result['res_params']

    m_fine = np.linspace(m.min()-0.1, m.max()+0.1, 300)
    y_fit = model_intensity(m_fine, result['r'], result['phi'],
                            result['A'], result['bg_c'], result['bg_s'], rp)

    bg_only = np.maximum(result['bg_c'] + result['bg_s']*(m_fine-7.0), 0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True)

    ax = axes[0]
    ax.errorbar(m, y, yerr=yerr, fmt='ko', capsize=2, label='Data')
    ax.plot(m_fine, y_fit, 'r-', lw=2, label='Interference fit')
    ax.plot(m_fine, bg_only, 'g--', lw=1.5, alpha=0.7, label='Background')
    ax.set_ylabel('Counts')
    ax.set_title(f'Channel {channel}: r={result["r"]:.3f}, φ={np.degrees(result["phi"]):.1f}°, χ²/dof={result["chi2"]/max(result["dof"],1):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pulls
    y_at_data = model_intensity(m, result['r'], result['phi'],
                                result['A'], result['bg_c'], result['bg_s'], rp)
    pulls = (y - y_at_data) / yerr

    ax = axes[1]
    ax.bar(m, pulls, width=(m.max()-m.min())/len(m)*0.8, color='steelblue', alpha=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(2, color='r', ls='--', alpha=0.5)
    ax.axhline(-2, color='r', ls='--', alpha=0.5)
    ax.set_ylabel('Pull')
    ax.set_xlabel('Mass (GeV)')
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    print("="*60)
    print("COHERENT INTERFERENCE FIT (v2)")
    print("="*60)

    all_results = {}

    for ch in ['A', 'B']:
        print(f"\n{'='*60}")
        print(f"CHANNEL {ch}")
        print('='*60)

        csv_path = os.path.join(OUT_DIR, f'digitized_{ch}.csv')
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} points")

        result = fit_channel(df, ch, verbose=True)

        print(f"\n  Running bootstrap (300 iterations)...")
        r_err_boot, phi_err_boot, r_samp, phi_samp = bootstrap(df, ch, n_iter=300)
        print(f"  Bootstrap: r_err={r_err_boot:.4f}, phi_err={phi_err_boot:.4f}")

        result['r_err_boot'] = r_err_boot
        result['phi_err_boot'] = phi_err_boot

        # Use the larger of cov and boot errors
        result['r_err'] = max(result['r_err_cov'], r_err_boot)
        result['phi_err'] = max(result['phi_err_cov'], phi_err_boot)

        # Plot
        plot_path = os.path.join(OUT_DIR, f'fit_{ch}_plot.png')
        make_plot(result, ch, plot_path)
        print(f"  Saved: {plot_path}")

        # JSON
        out = {
            'r': float(result['r']),
            'phi_rad': float(result['phi']),
            'phi_deg': float(np.degrees(result['phi'])),
            'r_err': float(result['r_err']),
            'phi_err_rad': float(result['phi_err']),
            'phi_err_deg': float(np.degrees(result['phi_err'])),
            'r_err_cov': float(result['r_err_cov']),
            'phi_err_cov': float(result['phi_err_cov']),
            'r_err_boot': float(r_err_boot),
            'phi_err_boot': float(phi_err_boot),
            'chi2': float(result['chi2']),
            'dof': int(result['dof']),
            'chi2_per_dof': float(result['chi2']/max(result['dof'],1)),
            'fit_window': result['fit_window'],
            'A': float(result['A']),
            'bg_c': float(result['bg_c']),
            'bg_s': float(result['bg_s']),
        }

        json_path = os.path.join(OUT_DIR, f'fit_{ch}.json')
        with open(json_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  Saved: {json_path}")

        all_results[ch] = result

    # Summary comparison
    print("\n" + "="*60)
    print("RANK-1 BOTTLENECK TEST")
    print("="*60)

    r_A = all_results['A']['r']
    phi_A = all_results['A']['phi']
    r_A_err = all_results['A']['r_err']
    phi_A_err = all_results['A']['phi_err']

    r_B = all_results['B']['r']
    phi_B = all_results['B']['phi']
    r_B_err = all_results['B']['r_err']
    phi_B_err = all_results['B']['phi_err']

    print(f"\nChannel A (J/ψ J/ψ):")
    print(f"  r_A   = {r_A:.4f} ± {r_A_err:.4f}")
    print(f"  φ_A   = {phi_A:.4f} ± {phi_A_err:.4f} rad")
    print(f"        = {np.degrees(phi_A):.2f}° ± {np.degrees(phi_A_err):.2f}°")

    print(f"\nChannel B (J/ψ ψ(2S)):")
    print(f"  r_B   = {r_B:.4f} ± {r_B_err:.4f}")
    print(f"  φ_B   = {phi_B:.4f} ± {phi_B_err:.4f} rad")
    print(f"        = {np.degrees(phi_B):.2f}° ± {np.degrees(phi_B_err):.2f}°")

    # Differences
    delta_r = r_A - r_B
    delta_r_err = np.sqrt(r_A_err**2 + r_B_err**2)

    delta_phi = phi_A - phi_B
    # Wrap to [-π, π]
    delta_phi = (delta_phi + np.pi) % (2*np.pi) - np.pi
    delta_phi_err = np.sqrt(phi_A_err**2 + phi_B_err**2)

    print(f"\nComparison:")
    print(f"  Δr    = r_A - r_B = {delta_r:.4f} ± {delta_r_err:.4f}")
    print(f"  Δφ    = φ_A - φ_B = {delta_phi:.4f} ± {delta_phi_err:.4f} rad")
    print(f"                    = {np.degrees(delta_phi):.2f}° ± {np.degrees(delta_phi_err):.2f}°")

    # Significance
    if delta_r_err > 0:
        r_sig = abs(delta_r) / delta_r_err
    else:
        r_sig = 0

    if delta_phi_err > 0:
        phi_sig = abs(delta_phi) / delta_phi_err
    else:
        phi_sig = 0

    print(f"\n  |Δr|/σ   = {r_sig:.2f}σ")
    print(f"  |Δφ|/σ   = {phi_sig:.2f}σ")

    max_sig = max(r_sig, phi_sig)

    if max_sig < 1:
        verdict = "COMPATIBLE within 1σ"
    elif max_sig < 2:
        verdict = "COMPATIBLE within 2σ"
    else:
        verdict = f"INCONSISTENT ({max_sig:.1f}σ deviation)"

    print(f"\n  VERDICT: {verdict}")

    # Save summary
    summary = {
        'channel_A': {
            'r': float(r_A),
            'r_err': float(r_A_err),
            'phi_rad': float(phi_A),
            'phi_deg': float(np.degrees(phi_A)),
            'phi_err_rad': float(phi_A_err),
            'phi_err_deg': float(np.degrees(phi_A_err)),
        },
        'channel_B': {
            'r': float(r_B),
            'r_err': float(r_B_err),
            'phi_rad': float(phi_B),
            'phi_deg': float(np.degrees(phi_B)),
            'phi_err_rad': float(phi_B_err),
            'phi_err_deg': float(np.degrees(phi_B_err)),
        },
        'comparison': {
            'delta_r': float(delta_r),
            'delta_r_err': float(delta_r_err),
            'delta_phi_rad': float(delta_phi),
            'delta_phi_deg': float(np.degrees(delta_phi)),
            'delta_phi_err_rad': float(delta_phi_err),
            'delta_phi_err_deg': float(np.degrees(delta_phi_err)),
            'r_significance_sigma': float(r_sig),
            'phi_significance_sigma': float(phi_sig),
            'max_significance_sigma': float(max_sig),
            'verdict': verdict,
        }
    }

    with open(os.path.join(OUT_DIR, 'rank1_test_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {os.path.join(OUT_DIR, 'rank1_test_summary.json')}")

    return all_results

if __name__ == "__main__":
    main()
