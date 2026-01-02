#!/usr/bin/env python3
"""
Fit coherent interference model to digitized CMS data.
Model: I(m) = |BW_6900(m) + R * BW_7100(m)|^2 + background(m)
Where R = r * exp(i*phi) is the complex ratio we want to extract.
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2
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
    'B': (6.9, 7.6),
}

def breit_wigner(m, m0, w):
    """
    Non-relativistic Breit-Wigner amplitude (complex).
    BW(m) = 1 / (m0^2 - m^2 - i*m0*w)
    """
    return 1.0 / (m0**2 - m**2 - 1j * m0 * w)

def interference_model(m, r, phi, norm, bg_a, bg_b, params):
    """
    Coherent interference intensity model.

    I(m) = norm * |BW_6900(m) + R * BW_7100(m)|^2 + background(m)

    Parameters:
    - r: magnitude of complex ratio R
    - phi: phase of complex ratio R (radians)
    - norm: overall normalization
    - bg_a, bg_b: linear background coefficients
    - params: resonance parameters dict
    """
    m0_6900 = params['m6900']
    w_6900 = params['w6900']
    m0_7100 = params['m7100']
    w_7100 = params['w7100']

    # Complex ratio
    R = r * np.exp(1j * phi)

    # Breit-Wigner amplitudes
    bw_6900 = breit_wigner(m, m0_6900, w_6900)
    bw_7100 = breit_wigner(m, m0_7100, w_7100)

    # Coherent sum squared
    amplitude = bw_6900 + R * bw_7100
    signal = norm * np.abs(amplitude)**2

    # Linear background (constrained to be non-negative)
    m_center = 7.0  # Center for numerical stability
    background = bg_a + bg_b * (m - m_center)
    background = np.maximum(background, 0)

    return signal + background

def chi2_objective(params_vec, m, y, yerr, res_params, bg_degree=1):
    """
    Chi-squared objective function.

    params_vec: [r, phi, norm, bg_a, bg_b]
    """
    r = params_vec[0]
    phi = params_vec[1]
    norm = params_vec[2]
    bg_a = params_vec[3]
    bg_b = params_vec[4] if bg_degree >= 1 else 0.0

    # Constraints: r >= 0, norm > 0, bg_a >= 0
    if r < 0 or norm <= 0 or bg_a < -10:
        return 1e10

    y_model = interference_model(m, r, phi, norm, bg_a, bg_b, res_params)

    residuals = (y - y_model) / yerr
    chi2_val = np.sum(residuals**2)

    return chi2_val

def fit_channel(df, channel, bg_degree=1, verbose=True):
    """
    Fit interference model to a channel's data.
    """
    res_params = RESONANCE_PARAMS[channel]
    fit_min, fit_max = FIT_WINDOWS[channel]

    # Filter to fit window
    mask = (df['mass_GeV'] >= fit_min) & (df['mass_GeV'] <= fit_max)
    df_fit = df[mask].copy()

    if len(df_fit) < 5:
        print(f"  Warning: Only {len(df_fit)} points in fit window!")
        # Expand window
        fit_min -= 0.2
        fit_max += 0.2
        mask = (df['mass_GeV'] >= fit_min) & (df['mass_GeV'] <= fit_max)
        df_fit = df[mask].copy()
        print(f"  Expanded window to ({fit_min:.1f}, {fit_max:.1f}): {len(df_fit)} points")

    m = df_fit['mass_GeV'].values
    y = df_fit['count'].values

    # Poisson uncertainty
    yerr = np.sqrt(np.maximum(y, 1))

    if verbose:
        print(f"  Fitting {len(m)} data points in [{fit_min:.2f}, {fit_max:.2f}] GeV")
        print(f"  Resonance params: m6900={res_params['m6900']:.3f}, w6900={res_params['w6900']:.3f}")
        print(f"                    m7100={res_params['m7100']:.3f}, w7100={res_params['w7100']:.3f}")

    # Initial parameter guesses
    # r ~ 0.5-1.5, phi ~ 0 to pi, norm ~ max(y) * 1e6, bg ~ 10
    bounds = [
        (0.01, 5.0),     # r
        (-np.pi, np.pi), # phi
        (1e4, 1e10),     # norm
        (0, 100),        # bg_a
        (-50, 50),       # bg_b
    ]

    # Use differential evolution for global optimization
    result = differential_evolution(
        chi2_objective,
        bounds,
        args=(m, y, yerr, res_params, bg_degree),
        seed=42,
        maxiter=1000,
        tol=1e-7,
        polish=True,
        workers=1
    )

    r_fit = result.x[0]
    phi_fit = result.x[1]
    norm_fit = result.x[2]
    bg_a_fit = result.x[3]
    bg_b_fit = result.x[4]
    chi2_min = result.fun

    dof = len(m) - 5  # 5 free parameters
    chi2_per_dof = chi2_min / max(dof, 1)

    if verbose:
        print(f"\n  Fit results:")
        print(f"    r     = {r_fit:.4f}")
        print(f"    phi   = {phi_fit:.4f} rad ({np.degrees(phi_fit):.2f}°)")
        print(f"    norm  = {norm_fit:.2e}")
        print(f"    bg_a  = {bg_a_fit:.2f}")
        print(f"    bg_b  = {bg_b_fit:.2f}")
        print(f"    chi2  = {chi2_min:.2f}")
        print(f"    dof   = {dof}")
        print(f"    chi2/dof = {chi2_per_dof:.2f}")

    # Estimate uncertainties from Hessian
    uncertainties = estimate_uncertainties(result.x, m, y, yerr, res_params, bg_degree)

    return {
        'r': r_fit,
        'phi': phi_fit,
        'norm': norm_fit,
        'bg_a': bg_a_fit,
        'bg_b': bg_b_fit,
        'chi2': chi2_min,
        'dof': dof,
        'chi2_per_dof': chi2_per_dof,
        'r_err': uncertainties[0],
        'phi_err': uncertainties[1],
        'fit_window': [fit_min, fit_max],
        'm': m,
        'y': y,
        'yerr': yerr,
        'res_params': res_params
    }

def estimate_uncertainties(x_opt, m, y, yerr, res_params, bg_degree):
    """
    Estimate parameter uncertainties from numerical Hessian.
    """
    n_params = len(x_opt)
    eps = 1e-5

    # Compute Hessian numerically
    hess = np.zeros((n_params, n_params))

    f0 = chi2_objective(x_opt, m, y, yerr, res_params, bg_degree)

    for i in range(n_params):
        for j in range(i, n_params):
            x_pp = x_opt.copy()
            x_pm = x_opt.copy()
            x_mp = x_opt.copy()
            x_mm = x_opt.copy()

            h_i = max(abs(x_opt[i]) * eps, eps)
            h_j = max(abs(x_opt[j]) * eps, eps)

            x_pp[i] += h_i; x_pp[j] += h_j
            x_pm[i] += h_i; x_pm[j] -= h_j
            x_mp[i] -= h_i; x_mp[j] += h_j
            x_mm[i] -= h_i; x_mm[j] -= h_j

            f_pp = chi2_objective(x_pp, m, y, yerr, res_params, bg_degree)
            f_pm = chi2_objective(x_pm, m, y, yerr, res_params, bg_degree)
            f_mp = chi2_objective(x_mp, m, y, yerr, res_params, bg_degree)
            f_mm = chi2_objective(x_mm, m, y, yerr, res_params, bg_degree)

            hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h_i * h_j)
            hess[j, i] = hess[i, j]

    # Covariance matrix = 2 * inverse(Hessian)
    try:
        cov = 2.0 * np.linalg.inv(hess)
        uncertainties = np.sqrt(np.diag(np.abs(cov)))
    except np.linalg.LinAlgError:
        uncertainties = np.ones(n_params) * 999

    return uncertainties

def bootstrap_fit(df, channel, n_bootstrap=200, verbose=False):
    """
    Bootstrap uncertainty estimation.
    """
    res_params = RESONANCE_PARAMS[channel]
    fit_min, fit_max = FIT_WINDOWS[channel]

    mask = (df['mass_GeV'] >= fit_min) & (df['mass_GeV'] <= fit_max)
    df_fit = df[mask].copy()

    if len(df_fit) < 5:
        fit_min -= 0.2
        fit_max += 0.2
        mask = (df['mass_GeV'] >= fit_min) & (df['mass_GeV'] <= fit_max)
        df_fit = df[mask].copy()

    m = df_fit['mass_GeV'].values
    y = df_fit['count'].values
    yerr = np.sqrt(np.maximum(y, 1))

    n_points = len(m)

    r_samples = []
    phi_samples = []

    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n_points, size=n_points, replace=True)
        m_boot = m[idx]
        y_boot = y[idx]
        yerr_boot = yerr[idx]

        # Fit
        bounds = [
            (0.01, 5.0),
            (-np.pi, np.pi),
            (1e4, 1e10),
            (0, 100),
            (-50, 50),
        ]

        try:
            result = differential_evolution(
                chi2_objective,
                bounds,
                args=(m_boot, y_boot, yerr_boot, res_params, 1),
                seed=42 + i,
                maxiter=500,
                tol=1e-5,
                polish=True,
                workers=1
            )
            r_samples.append(result.x[0])
            phi_samples.append(result.x[1])
        except:
            pass

        if verbose and (i + 1) % 50 == 0:
            print(f"    Bootstrap: {i+1}/{n_bootstrap}")

    r_samples = np.array(r_samples)
    phi_samples = np.array(phi_samples)

    r_err_boot = np.std(r_samples)
    phi_err_boot = np.std(phi_samples)

    return r_err_boot, phi_err_boot, r_samples, phi_samples

def create_fit_plot(result, channel, output_path):
    """
    Create a plot showing data and best-fit curve.
    """
    m = result['m']
    y = result['y']
    yerr = result['yerr']
    res_params = result['res_params']

    # Generate smooth curve
    m_fine = np.linspace(m.min(), m.max(), 200)
    y_fit = interference_model(m_fine, result['r'], result['phi'],
                               result['norm'], result['bg_a'], result['bg_b'],
                               res_params)

    # Background only
    y_bg = result['bg_a'] + result['bg_b'] * (m_fine - 7.0)
    y_bg = np.maximum(y_bg, 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

    # Main plot
    ax1.errorbar(m, y, yerr=yerr, fmt='ko', markersize=6, capsize=3, label='Data')
    ax1.plot(m_fine, y_fit, 'r-', linewidth=2, label='Fit (interference)')
    ax1.plot(m_fine, y_bg, 'g--', linewidth=1.5, alpha=0.7, label='Background')

    ax1.set_ylabel('Candidates per bin')
    ax1.set_title(f'Channel {channel}: Interference Fit\n'
                  f'r = {result["r"]:.3f} ± {result["r_err"]:.3f}, '
                  f'φ = {result["phi"]:.3f} ± {result["phi_err"]:.3f} rad')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Pull plot
    y_fit_at_data = interference_model(m, result['r'], result['phi'],
                                       result['norm'], result['bg_a'], result['bg_b'],
                                       res_params)
    pulls = (y - y_fit_at_data) / yerr

    ax2.bar(m, pulls, width=0.015, color='blue', alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Mass (GeV)')
    ax2.set_ylabel('Pull')
    ax2.set_ylim(-4, 4)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Saved: {output_path}")

def main():
    print("="*60)
    print("COHERENT INTERFERENCE FIT")
    print("="*60)

    results = {}

    for channel in ['A', 'B']:
        print(f"\n{'='*60}")
        print(f"CHANNEL {channel}")
        print('='*60)

        # Load digitized data
        csv_path = os.path.join(OUT_DIR, f'digitized_{channel}.csv')
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} data points from {csv_path}")

        # Fit
        result = fit_channel(df, channel, bg_degree=1, verbose=True)

        # Bootstrap
        print(f"\n  Running bootstrap ({200} iterations)...")
        r_err_boot, phi_err_boot, r_samples, phi_samples = bootstrap_fit(df, channel, n_bootstrap=200)
        print(f"  Bootstrap uncertainties: r_err={r_err_boot:.4f}, phi_err={phi_err_boot:.4f}")

        result['r_err_boot'] = r_err_boot
        result['phi_err_boot'] = phi_err_boot

        # Create plot
        plot_path = os.path.join(OUT_DIR, f'fit_{channel}_plot.png')
        create_fit_plot(result, channel, plot_path)

        # Save fit results
        fit_output = {
            'r': float(result['r']),
            'phi': float(result['phi']),
            'phi_deg': float(np.degrees(result['phi'])),
            'r_err_cov': float(result['r_err']),
            'phi_err_cov': float(result['phi_err']),
            'r_err_boot': float(r_err_boot),
            'phi_err_boot': float(phi_err_boot),
            'chi2': float(result['chi2']),
            'dof': int(result['dof']),
            'chi2_per_dof': float(result['chi2_per_dof']),
            'fit_window': result['fit_window'],
            'norm': float(result['norm']),
            'bg_a': float(result['bg_a']),
            'bg_b': float(result['bg_b']),
        }

        json_path = os.path.join(OUT_DIR, f'fit_{channel}.json')
        with open(json_path, 'w') as f:
            json.dump(fit_output, f, indent=2)
        print(f"  Saved: {json_path}")

        results[channel] = result

    # Summary
    print("\n" + "="*60)
    print("FIT SUMMARY")
    print("="*60)

    r_A = results['A']['r']
    phi_A = results['A']['phi']
    r_A_err = max(results['A']['r_err'], results['A']['r_err_boot'])
    phi_A_err = max(results['A']['phi_err'], results['A']['phi_err_boot'])

    r_B = results['B']['r']
    phi_B = results['B']['phi']
    r_B_err = max(results['B']['r_err'], results['B']['r_err_boot'])
    phi_B_err = max(results['B']['phi_err'], results['B']['phi_err_boot'])

    print(f"\nChannel A (J/ψ J/ψ):")
    print(f"  R_A = {r_A:.4f} ± {r_A_err:.4f} * exp(i * {phi_A:.4f} ± {phi_A_err:.4f})")
    print(f"      = {r_A:.4f} ± {r_A_err:.4f} * exp(i * {np.degrees(phi_A):.2f}° ± {np.degrees(phi_A_err):.2f}°)")

    print(f"\nChannel B (J/ψ ψ(2S)):")
    print(f"  R_B = {r_B:.4f} ± {r_B_err:.4f} * exp(i * {phi_B:.4f} ± {phi_B_err:.4f})")
    print(f"      = {r_B:.4f} ± {r_B_err:.4f} * exp(i * {np.degrees(phi_B):.2f}° ± {np.degrees(phi_B_err):.2f}°)")

    # Comparison
    delta_r = r_A - r_B
    delta_r_err = np.sqrt(r_A_err**2 + r_B_err**2)

    delta_phi = phi_A - phi_B
    # Wrap to [-pi, pi]
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    delta_phi_err = np.sqrt(phi_A_err**2 + phi_B_err**2)

    print(f"\nRank-1 Bottleneck Test:")
    print(f"  Δr   = r_A - r_B = {delta_r:.4f} ± {delta_r_err:.4f}")
    print(f"  Δφ   = φ_A - φ_B = {delta_phi:.4f} ± {delta_phi_err:.4f} rad")
    print(f"                   = {np.degrees(delta_phi):.2f}° ± {np.degrees(delta_phi_err):.2f}°")

    # Significance
    r_compat_sigma = abs(delta_r) / delta_r_err if delta_r_err > 0 else 0
    phi_compat_sigma = abs(delta_phi) / delta_phi_err if delta_phi_err > 0 else 0

    print(f"\n  |Δr|/σ_r = {r_compat_sigma:.2f}σ")
    print(f"  |Δφ|/σ_φ = {phi_compat_sigma:.2f}σ")

    if r_compat_sigma < 1 and phi_compat_sigma < 1:
        verdict = "COMPATIBLE within 1σ"
    elif r_compat_sigma < 2 and phi_compat_sigma < 2:
        verdict = "COMPATIBLE within 2σ"
    else:
        verdict = "INCONSISTENT (>2σ deviation)"

    print(f"\n  Verdict: {verdict}")

    return results

if __name__ == "__main__":
    main()
