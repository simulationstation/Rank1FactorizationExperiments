#!/usr/bin/env python3
"""
CMS-like interference fit models for rank-1 bottleneck test.

Channel A: 3-BW interference model (BW1=X(6600), BW2=X(6900), BW3=X(7100))
Channel B: 2-BW interference model (X(6900), X(7100))

The key observable is R = c3/c2 (complex ratio of coupling to X(7100) vs X(6900))
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2 as chi2_dist
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

# ============================================================
# Resonance parameters from CMS publications
# ============================================================

# Channel A: CMS-BPH-21-003 (interference model from HEPData Table 1)
PARAMS_A = {
    'bw1': {'m': 6.638, 'w': 0.440},  # X(6600)
    'bw2': {'m': 6.847, 'w': 0.191},  # X(6900)
    'bw3': {'m': 7.134, 'w': 0.097},  # X(7100)
}

# Channel B: CMS-PAS-BPH-22-004 (unconstrained fi23 fit)
PARAMS_B = {
    'bw2': {'m': 6.876, 'w': 0.253},  # X(6900)
    'bw3': {'m': 7.169, 'w': 0.154},  # X(7100)
}

# ============================================================
# Breit-Wigner functions
# ============================================================

def bw_amplitude(m, m0, w):
    """
    Relativistic Breit-Wigner amplitude (complex).
    BW(m) = 1 / (m0^2 - m^2 - i*m0*w)
    """
    return 1.0 / (m0**2 - m**2 - 1j * m0 * w)

def bw_amplitude_normalized(m, m0, w):
    """
    Normalized BW so that peak amplitude ~ 1.
    """
    bw = bw_amplitude(m, m0, w)
    norm = np.abs(bw_amplitude(m0, m0, w))
    return bw / norm

# ============================================================
# Channel A model: 3-BW interference + background
# ============================================================

def model_channel_A(m, params):
    """
    Channel A intensity model:
    I(m) = |c1*BW1 + c2*BW2 + c3*BW3|^2 + background

    params = [c1_re, c1_im, c2_re, c2_im, c3_re, c3_im, bg_a, bg_b, bg_c]

    We fix c2 to be real and positive (phase convention), so c2_im = 0.
    Actually, we'll parameterize as:
    - c2 = amplitude (real, positive)
    - r1 = |c1|/c2, phi1 = arg(c1)
    - r3 = |c3|/c2, phi3 = arg(c3)

    params = [c2_norm, r1, phi1, r3, phi3, bg_a, bg_b, bg_c]
    """
    c2_norm, r1, phi1, r3, phi3, bg_a, bg_b, bg_c = params

    # Complex coefficients
    c1 = r1 * np.exp(1j * phi1)  # relative to c2
    c2 = 1.0  # normalized
    c3 = r3 * np.exp(1j * phi3)  # relative to c2

    # BW amplitudes
    bw1 = bw_amplitude_normalized(m, PARAMS_A['bw1']['m'], PARAMS_A['bw1']['w'])
    bw2 = bw_amplitude_normalized(m, PARAMS_A['bw2']['m'], PARAMS_A['bw2']['w'])
    bw3 = bw_amplitude_normalized(m, PARAMS_A['bw3']['m'], PARAMS_A['bw3']['w'])

    # Coherent sum
    amplitude = c1 * bw1 + c2 * bw2 + c3 * bw3
    signal = c2_norm * np.abs(amplitude)**2

    # Quadratic background
    m_ref = 6.9  # reference mass
    background = bg_a + bg_b * (m - m_ref) + bg_c * (m - m_ref)**2
    background = np.maximum(background, 0)

    return signal + background

def chi2_channel_A(params, m, y, yerr):
    """Chi-squared for Channel A fit."""
    c2_norm, r1, phi1, r3, phi3, bg_a, bg_b, bg_c = params

    # Constraints
    if c2_norm < 0 or r1 < 0 or r3 < 0:
        return 1e20
    if bg_a < -50:
        return 1e20

    y_model = model_channel_A(m, params)

    # Protect against zero errors
    yerr_safe = np.maximum(yerr, 1.0)
    residuals = (y - y_model) / yerr_safe
    return np.sum(residuals**2)

# ============================================================
# Channel B model: 2-BW interference + background
# ============================================================

def model_channel_B(m, params):
    """
    Channel B intensity model:
    I(m) = |c2*BW2 + c3*BW3|^2 + background

    params = [c2_norm, r3, phi3, bg_a, bg_b]

    c2 is normalized to 1 (phase convention).
    """
    c2_norm, r3, phi3, bg_a, bg_b = params

    c2 = 1.0
    c3 = r3 * np.exp(1j * phi3)

    bw2 = bw_amplitude_normalized(m, PARAMS_B['bw2']['m'], PARAMS_B['bw2']['w'])
    bw3 = bw_amplitude_normalized(m, PARAMS_B['bw3']['m'], PARAMS_B['bw3']['w'])

    amplitude = c2 * bw2 + c3 * bw3
    signal = c2_norm * np.abs(amplitude)**2

    m_ref = 8.0
    background = bg_a + bg_b * (m - m_ref)
    background = np.maximum(background, 0)

    return signal + background

def chi2_channel_B(params, m, y, yerr):
    """Chi-squared for Channel B fit."""
    c2_norm, r3, phi3, bg_a, bg_b = params

    if c2_norm < 0 or r3 < 0:
        return 1e20
    if bg_a < -10:
        return 1e20

    y_model = model_channel_B(m, params)
    yerr_safe = np.maximum(yerr, 1.0)
    residuals = (y - y_model) / yerr_safe
    return np.sum(residuals**2)

# ============================================================
# Joint fit with rank-1 constraint
# ============================================================

def chi2_joint_constrained(params_shared, m_A, y_A, yerr_A, m_B, y_B, yerr_B):
    """
    Joint chi-squared with R_A = R_B constraint.

    Shared: r3, phi3 (the ratio R = c3/c2)
    Independent: normalizations, backgrounds, r1/phi1 for channel A
    """
    # Unpack shared ratio
    r3_shared = params_shared[0]
    phi3_shared = params_shared[1]

    # Channel A specific: c2_norm_A, r1, phi1, bg_a_A, bg_b_A, bg_c_A
    c2_norm_A = params_shared[2]
    r1 = params_shared[3]
    phi1 = params_shared[4]
    bg_a_A = params_shared[5]
    bg_b_A = params_shared[6]
    bg_c_A = params_shared[7]

    # Channel B specific: c2_norm_B, bg_a_B, bg_b_B
    c2_norm_B = params_shared[8]
    bg_a_B = params_shared[9]
    bg_b_B = params_shared[10]

    # Construct full parameter vectors
    params_A = [c2_norm_A, r1, phi1, r3_shared, phi3_shared, bg_a_A, bg_b_A, bg_c_A]
    params_B = [c2_norm_B, r3_shared, phi3_shared, bg_a_B, bg_b_B]

    chi2_A = chi2_channel_A(params_A, m_A, y_A, yerr_A)
    chi2_B = chi2_channel_B(params_B, m_B, y_B, yerr_B)

    return chi2_A + chi2_B

# ============================================================
# Fitting functions
# ============================================================

def fit_channel_A(df, fit_window=(6.6, 7.4), verbose=True):
    """Fit Channel A with 3-BW interference model."""
    print("\n" + "="*60)
    print("Fitting Channel A (J/ψ J/ψ)")
    print("="*60)

    # Filter to fit window
    mask = (df['m_center'] >= fit_window[0]) & (df['m_center'] <= fit_window[1])
    df_fit = df[mask].copy()

    m = df_fit['m_center'].values
    y = df_fit['count'].values
    yerr = df_fit['stat_err'].values

    print(f"Fit window: {fit_window[0]:.2f} - {fit_window[1]:.2f} GeV")
    print(f"Data points: {len(m)}")
    print(f"Total counts: {y.sum():.0f}")

    # Initial guesses
    # c2_norm ~ peak height * width^2, r1 ~ 0.5, r3 ~ 0.3
    c2_norm_init = y.max() * 0.02
    r1_init = 0.5
    phi1_init = 0.0
    r3_init = 0.3
    phi3_init = np.pi
    bg_a_init = y.min()
    bg_b_init = 0
    bg_c_init = 0

    x0 = [c2_norm_init, r1_init, phi1_init, r3_init, phi3_init, bg_a_init, bg_b_init, bg_c_init]

    # Bounds
    bounds = [
        (1, 1e5),          # c2_norm
        (0.01, 5),         # r1
        (-np.pi, np.pi),   # phi1
        (0.01, 5),         # r3
        (-np.pi, np.pi),   # phi3
        (-50, 200),        # bg_a
        (-100, 100),       # bg_b
        (-100, 100),       # bg_c
    ]

    # Global optimization
    result = differential_evolution(
        chi2_channel_A, bounds,
        args=(m, y, yerr),
        seed=42,
        maxiter=2000,
        tol=1e-8,
        polish=True,
        workers=1
    )

    params = result.x
    chi2_val = result.fun
    dof = len(m) - len(params)
    chi2_dof = chi2_val / dof if dof > 0 else chi2_val

    # Extract ratio R = c3/c2
    r3_A = params[3]
    phi3_A = params[4]
    R_A = r3_A * np.exp(1j * phi3_A)

    if verbose:
        print(f"\nFit results:")
        print(f"  c2_norm = {params[0]:.4f}")
        print(f"  r1      = {params[1]:.4f}, phi1 = {params[2]:.4f} ({np.degrees(params[2]):.1f}°)")
        print(f"  r3      = {params[3]:.4f}, phi3 = {params[4]:.4f} ({np.degrees(params[4]):.1f}°)")
        print(f"  bg      = {params[5]:.2f} + {params[6]:.2f}*(m-6.9) + {params[7]:.4f}*(m-6.9)^2")
        print(f"\n  χ² = {chi2_val:.2f}, dof = {dof}, χ²/dof = {chi2_dof:.2f}")
        print(f"\n  R_A = {r3_A:.4f} * exp(i * {np.degrees(phi3_A):.1f}°)")

    return {
        'params': params,
        'chi2': chi2_val,
        'dof': dof,
        'chi2_dof': chi2_dof,
        'r3': r3_A,
        'phi3': phi3_A,
        'R': R_A,
        'm': m,
        'y': y,
        'yerr': yerr,
    }

def fit_channel_B(df, fit_window=(7.0, 9.0), verbose=True):
    """Fit Channel B with 2-BW interference model."""
    print("\n" + "="*60)
    print("Fitting Channel B (J/ψ ψ(2S))")
    print("="*60)

    # Filter to fit window
    mask = (df['mass_center_GeV'] >= fit_window[0]) & (df['mass_center_GeV'] <= fit_window[1])
    df_fit = df[mask].copy()

    m = df_fit['mass_center_GeV'].values
    y = df_fit['count'].values
    yerr = df_fit['sigma_count'].values

    print(f"Fit window: {fit_window[0]:.2f} - {fit_window[1]:.2f} GeV")
    print(f"Data points: {len(m)}")
    print(f"Total counts: {y.sum():.1f}")

    # Initial guesses
    c2_norm_init = y.max() * 0.1
    r3_init = 0.5
    phi3_init = np.pi
    bg_a_init = y.min()
    bg_b_init = 0

    x0 = [c2_norm_init, r3_init, phi3_init, bg_a_init, bg_b_init]

    bounds = [
        (0.1, 1e4),        # c2_norm
        (0.01, 5),         # r3
        (-np.pi, np.pi),   # phi3
        (-10, 50),         # bg_a
        (-20, 20),         # bg_b
    ]

    result = differential_evolution(
        chi2_channel_B, bounds,
        args=(m, y, yerr),
        seed=42,
        maxiter=2000,
        tol=1e-8,
        polish=True,
        workers=1
    )

    params = result.x
    chi2_val = result.fun
    dof = len(m) - len(params)
    chi2_dof = chi2_val / dof if dof > 0 else chi2_val

    r3_B = params[1]
    phi3_B = params[2]
    R_B = r3_B * np.exp(1j * phi3_B)

    if verbose:
        print(f"\nFit results:")
        print(f"  c2_norm = {params[0]:.4f}")
        print(f"  r3      = {params[1]:.4f}, phi3 = {params[2]:.4f} ({np.degrees(params[2]):.1f}°)")
        print(f"  bg      = {params[3]:.2f} + {params[4]:.2f}*(m-8.0)")
        print(f"\n  χ² = {chi2_val:.2f}, dof = {dof}, χ²/dof = {chi2_dof:.2f}")
        print(f"\n  R_B = {r3_B:.4f} * exp(i * {np.degrees(phi3_B):.1f}°)")

    return {
        'params': params,
        'chi2': chi2_val,
        'dof': dof,
        'chi2_dof': chi2_dof,
        'r3': r3_B,
        'phi3': phi3_B,
        'R': R_B,
        'm': m,
        'y': y,
        'yerr': yerr,
    }

def fit_joint_constrained(df_A, df_B, fit_window_A=(6.6, 7.4), fit_window_B=(7.0, 9.0), verbose=True):
    """Joint fit with rank-1 constraint: R_A = R_B."""
    print("\n" + "="*60)
    print("Joint Constrained Fit (R_A = R_B)")
    print("="*60)

    # Prepare Channel A data
    mask_A = (df_A['m_center'] >= fit_window_A[0]) & (df_A['m_center'] <= fit_window_A[1])
    df_fit_A = df_A[mask_A].copy()
    m_A = df_fit_A['m_center'].values
    y_A = df_fit_A['count'].values
    yerr_A = df_fit_A['stat_err'].values

    # Prepare Channel B data
    mask_B = (df_B['mass_center_GeV'] >= fit_window_B[0]) & (df_B['mass_center_GeV'] <= fit_window_B[1])
    df_fit_B = df_B[mask_B].copy()
    m_B = df_fit_B['mass_center_GeV'].values
    y_B = df_fit_B['count'].values
    yerr_B = df_fit_B['sigma_count'].values

    print(f"Channel A: {len(m_A)} points")
    print(f"Channel B: {len(m_B)} points")

    # Initial guesses for shared + individual parameters
    # [r3_shared, phi3_shared, c2_norm_A, r1, phi1, bg_a_A, bg_b_A, bg_c_A, c2_norm_B, bg_a_B, bg_b_B]
    x0 = [0.3, np.pi, 50, 0.5, 0, 50, 0, 0, 10, 10, 0]

    bounds = [
        (0.01, 5),         # r3_shared
        (-np.pi, np.pi),   # phi3_shared
        (1, 1e5),          # c2_norm_A
        (0.01, 5),         # r1
        (-np.pi, np.pi),   # phi1
        (-50, 200),        # bg_a_A
        (-100, 100),       # bg_b_A
        (-100, 100),       # bg_c_A
        (0.1, 1e4),        # c2_norm_B
        (-10, 50),         # bg_a_B
        (-20, 20),         # bg_b_B
    ]

    result = differential_evolution(
        chi2_joint_constrained, bounds,
        args=(m_A, y_A, yerr_A, m_B, y_B, yerr_B),
        seed=42,
        maxiter=3000,
        tol=1e-8,
        polish=True,
        workers=1
    )

    params = result.x
    chi2_val = result.fun
    n_params = len(params)
    dof = len(m_A) + len(m_B) - n_params
    chi2_dof = chi2_val / dof if dof > 0 else chi2_val

    r3_shared = params[0]
    phi3_shared = params[1]
    R_shared = r3_shared * np.exp(1j * phi3_shared)

    if verbose:
        print(f"\nConstrained fit results:")
        print(f"  r3_shared  = {r3_shared:.4f}")
        print(f"  phi3_shared = {phi3_shared:.4f} ({np.degrees(phi3_shared):.1f}°)")
        print(f"\n  χ² = {chi2_val:.2f}, dof = {dof}, χ²/dof = {chi2_dof:.2f}")
        print(f"\n  R_shared = {r3_shared:.4f} * exp(i * {np.degrees(phi3_shared):.1f}°)")

    return {
        'params': params,
        'chi2': chi2_val,
        'dof': dof,
        'chi2_dof': chi2_dof,
        'r3': r3_shared,
        'phi3': phi3_shared,
        'R': R_shared,
        'm_A': m_A, 'y_A': y_A, 'yerr_A': yerr_A,
        'm_B': m_B, 'y_B': y_B, 'yerr_B': yerr_B,
    }

# ============================================================
# Plotting functions
# ============================================================

def plot_fit_A(result, output_path):
    """Create fit plot for Channel A."""
    m = result['m']
    y = result['y']
    yerr = result['yerr']
    params = result['params']

    m_fine = np.linspace(m.min()-0.05, m.max()+0.05, 300)
    y_fit = model_channel_A(m_fine, params)

    # Background only
    bg_a, bg_b, bg_c = params[5], params[6], params[7]
    m_ref = 6.9
    bg_only = bg_a + bg_b * (m_fine - m_ref) + bg_c * (m_fine - m_ref)**2
    bg_only = np.maximum(bg_only, 0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True)

    ax = axes[0]
    ax.errorbar(m, y, yerr=yerr, fmt='ko', markersize=4, capsize=2, label='HEPData')
    ax.plot(m_fine, y_fit, 'r-', lw=2, label='3-BW interference fit')
    ax.plot(m_fine, bg_only, 'g--', lw=1.5, alpha=0.7, label='Background')
    ax.set_ylabel('Candidates / 25 MeV')
    ax.set_title(f'Channel A (J/ψ J/ψ): χ²/dof = {result["chi2_dof"]:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pulls
    y_fit_at_data = model_channel_A(m, params)
    pulls = (y - y_fit_at_data) / yerr

    ax = axes[1]
    ax.bar(m, pulls, width=0.02, color='steelblue', alpha=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(2, color='r', ls='--', alpha=0.5)
    ax.axhline(-2, color='r', ls='--', alpha=0.5)
    ax.set_xlabel('m(J/ψ J/ψ) [GeV]')
    ax.set_ylabel('Pull')
    ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_fit_B(result, output_path):
    """Create fit plot for Channel B."""
    m = result['m']
    y = result['y']
    yerr = result['yerr']
    params = result['params']

    m_fine = np.linspace(m.min()-0.05, m.max()+0.05, 300)
    y_fit = model_channel_B(m_fine, params)

    bg_a, bg_b = params[3], params[4]
    m_ref = 8.0
    bg_only = bg_a + bg_b * (m_fine - m_ref)
    bg_only = np.maximum(bg_only, 0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True)

    ax = axes[0]
    ax.errorbar(m, y, yerr=yerr, fmt='ko', markersize=4, capsize=2, label='Digitized data')
    ax.plot(m_fine, y_fit, 'r-', lw=2, label='2-BW interference fit')
    ax.plot(m_fine, bg_only, 'g--', lw=1.5, alpha=0.7, label='Background')
    ax.set_ylabel('Candidates / 40 MeV')
    ax.set_title(f'Channel B (J/ψ ψ(2S)): χ²/dof = {result["chi2_dof"]:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    y_fit_at_data = model_channel_B(m, params)
    pulls = (y - y_fit_at_data) / yerr

    ax = axes[1]
    ax.bar(m, pulls, width=0.035, color='steelblue', alpha=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(2, color='r', ls='--', alpha=0.5)
    ax.axhline(-2, color='r', ls='--', alpha=0.5)
    ax.set_xlabel('m(J/ψ ψ(2S)) [GeV]')
    ax.set_ylabel('Pull')
    ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# ============================================================
# Bootstrap uncertainty estimation
# ============================================================

def bootstrap_channel_A(df, n_boot=300, fit_window=(6.6, 7.4)):
    """Bootstrap for Channel A using Poisson resampling."""
    mask = (df['m_center'] >= fit_window[0]) & (df['m_center'] <= fit_window[1])
    df_fit = df[mask].copy()

    m = df_fit['m_center'].values
    y_orig = df_fit['count'].values
    yerr = df_fit['stat_err'].values

    r3_samples = []
    phi3_samples = []

    bounds = [
        (1, 1e5), (0.01, 5), (-np.pi, np.pi), (0.01, 5), (-np.pi, np.pi),
        (-50, 200), (-100, 100), (-100, 100),
    ]

    for i in range(n_boot):
        # Poisson resample
        y_boot = np.random.poisson(np.maximum(y_orig, 0.1))
        yerr_boot = np.sqrt(np.maximum(y_boot, 1))

        try:
            result = differential_evolution(
                chi2_channel_A, bounds,
                args=(m, y_boot, yerr_boot),
                seed=i, maxiter=500, tol=1e-5, polish=False, workers=1
            )
            r3_samples.append(result.x[3])
            phi3_samples.append(result.x[4])
        except:
            pass

    return np.array(r3_samples), np.array(phi3_samples)

def bootstrap_channel_B(df, n_boot=300, fit_window=(7.0, 9.0)):
    """Bootstrap for Channel B with digitization jitter."""
    mask = (df['mass_center_GeV'] >= fit_window[0]) & (df['mass_center_GeV'] <= fit_window[1])
    df_fit = df[mask].copy()

    m = df_fit['mass_center_GeV'].values
    y_orig = df_fit['count'].values
    yerr = df_fit['sigma_count'].values

    r3_samples = []
    phi3_samples = []

    bounds = [
        (0.1, 1e4), (0.01, 5), (-np.pi, np.pi), (-10, 50), (-20, 20),
    ]

    for i in range(n_boot):
        # Poisson resample + digitization jitter
        y_boot = np.random.poisson(np.maximum(y_orig, 0.1))
        # Add digitization uncertainty (~5% of count)
        y_boot = y_boot + np.random.normal(0, 0.05 * np.maximum(y_boot, 1))
        y_boot = np.maximum(y_boot, 0.1)
        yerr_boot = np.sqrt(np.maximum(y_boot, 1))

        try:
            result = differential_evolution(
                chi2_channel_B, bounds,
                args=(m, y_boot, yerr_boot),
                seed=i, maxiter=500, tol=1e-5, polish=False, workers=1
            )
            r3_samples.append(result.x[1])
            phi3_samples.append(result.x[2])
        except:
            pass

    return np.array(r3_samples), np.array(phi3_samples)

# ============================================================
# Main analysis
# ============================================================

def main():
    print("="*70)
    print("RANK-1 BOTTLENECK TEST v2")
    print("="*70)

    # Load Channel A data (HEPData)
    df_A = pd.read_csv(os.path.join(DATA_DIR, 'hepdata', 'channel_A_mass_spectrum.csv'))
    print(f"\nChannel A: Loaded {len(df_A)} bins from HEPData")

    # Load Channel B data (digitized)
    df_B = pd.read_csv(os.path.join(DATA_DIR, 'derived', 'channel_B_digitized.csv'))
    print(f"Channel B: Loaded {len(df_B)} bins from digitization")

    # ========================================
    # Fit Channel A
    # ========================================
    result_A = fit_channel_A(df_A)
    plot_fit_A(result_A, os.path.join(OUT_DIR, 'fit_A_plot.png'))

    # Save parameters
    with open(os.path.join(OUT_DIR, 'fit_A_params.json'), 'w') as f:
        json.dump({
            'r3': float(result_A['r3']),
            'phi3_rad': float(result_A['phi3']),
            'phi3_deg': float(np.degrees(result_A['phi3'])),
            'chi2': float(result_A['chi2']),
            'dof': int(result_A['dof']),
            'chi2_dof': float(result_A['chi2_dof']),
        }, f, indent=2)

    # ========================================
    # Fit Channel B
    # ========================================
    result_B = fit_channel_B(df_B)
    plot_fit_B(result_B, os.path.join(OUT_DIR, 'fit_B_plot.png'))

    with open(os.path.join(OUT_DIR, 'fit_B_params.json'), 'w') as f:
        json.dump({
            'r3': float(result_B['r3']),
            'phi3_rad': float(result_B['phi3']),
            'phi3_deg': float(np.degrees(result_B['phi3'])),
            'chi2': float(result_B['chi2']),
            'dof': int(result_B['dof']),
            'chi2_dof': float(result_B['chi2_dof']),
        }, f, indent=2)

    # ========================================
    # Constrained joint fit
    # ========================================
    result_joint = fit_joint_constrained(df_A, df_B)

    # ========================================
    # Likelihood ratio test
    # ========================================
    chi2_unconstrained = result_A['chi2'] + result_B['chi2']
    chi2_constrained = result_joint['chi2']

    # Number of parameters:
    # Unconstrained: A has 8, B has 5 = 13 total
    # Constrained: 11 total (r3, phi3 shared)
    n_params_unconstrained = 13
    n_params_constrained = 11
    n_data = len(result_A['m']) + len(result_B['m'])

    dof_unconstrained = n_data - n_params_unconstrained
    dof_constrained = n_data - n_params_constrained

    delta_chi2 = chi2_constrained - chi2_unconstrained
    delta_dof = n_params_unconstrained - n_params_constrained  # = 2 (extra params in unconstrained)

    # p-value: test if the constraint significantly worsens the fit
    # Under H0 (constraint valid), delta_chi2 ~ chi2(delta_dof)
    p_value = 1 - chi2_dist.cdf(max(delta_chi2, 0), delta_dof)

    print("\n" + "="*60)
    print("LIKELIHOOD RATIO TEST")
    print("="*60)
    print(f"χ²_unconstrained = {chi2_unconstrained:.2f} ({n_params_unconstrained} params, {n_data} data pts)")
    print(f"χ²_constrained   = {chi2_constrained:.2f} ({n_params_constrained} params)")
    print(f"Δχ² = {delta_chi2:.2f}, Δ(params) = {delta_dof}")
    print(f"p-value = {p_value:.6f}")

    if delta_chi2 < 0:
        lr_interpretation = "Rank-1 constraint IMPROVES fit (Δχ² < 0)"
    elif p_value > 0.05:
        lr_interpretation = "Rank-1 constraint is ACCEPTABLE (p > 0.05)"
    elif p_value > 0.01:
        lr_interpretation = "Rank-1 constraint shows TENSION (0.01 < p < 0.05)"
    else:
        lr_interpretation = "Rank-1 constraint is REJECTED (p < 0.01)"

    print(f"Interpretation: {lr_interpretation}")

    # ========================================
    # Bootstrap uncertainties
    # ========================================
    print("\n" + "="*60)
    print("BOOTSTRAP UNCERTAINTY ESTIMATION")
    print("="*60)

    print("Running bootstrap for Channel A (200 iterations)...")
    r3_A_boot, phi3_A_boot = bootstrap_channel_A(df_A, n_boot=200)
    r3_A_err = np.std(r3_A_boot)
    phi3_A_err = np.std(phi3_A_boot)
    print(f"  r3_A  = {result_A['r3']:.4f} ± {r3_A_err:.4f}")
    print(f"  phi3_A = {np.degrees(result_A['phi3']):.1f}° ± {np.degrees(phi3_A_err):.1f}°")

    print("Running bootstrap for Channel B (200 iterations)...")
    r3_B_boot, phi3_B_boot = bootstrap_channel_B(df_B, n_boot=200)
    r3_B_err = np.std(r3_B_boot)
    phi3_B_err = np.std(phi3_B_boot)
    print(f"  r3_B  = {result_B['r3']:.4f} ± {r3_B_err:.4f}")
    print(f"  phi3_B = {np.degrees(result_B['phi3']):.1f}° ± {np.degrees(phi3_B_err):.1f}°")

    # ========================================
    # Final comparison
    # ========================================
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    r_A = result_A['r3']
    phi_A = result_A['phi3']
    r_B = result_B['r3']
    phi_B = result_B['phi3']

    delta_r = r_A - r_B
    delta_r_err = np.sqrt(r3_A_err**2 + r3_B_err**2)

    delta_phi = phi_A - phi_B
    delta_phi = (delta_phi + np.pi) % (2*np.pi) - np.pi
    delta_phi_err = np.sqrt(phi3_A_err**2 + phi3_B_err**2)

    r_sig = abs(delta_r) / delta_r_err if delta_r_err > 0 else 0
    phi_sig = abs(delta_phi) / delta_phi_err if delta_phi_err > 0 else 0

    print(f"\nChannel A (J/ψ J/ψ):")
    print(f"  r_A   = {r_A:.4f} ± {r3_A_err:.4f}")
    print(f"  φ_A   = {np.degrees(phi_A):.1f}° ± {np.degrees(phi3_A_err):.1f}°")

    print(f"\nChannel B (J/ψ ψ(2S)):")
    print(f"  r_B   = {r_B:.4f} ± {r3_B_err:.4f}")
    print(f"  φ_B   = {np.degrees(phi_B):.1f}° ± {np.degrees(phi3_B_err):.1f}°")

    print(f"\nComparison:")
    print(f"  Δr    = {delta_r:+.4f} ± {delta_r_err:.4f}  ({r_sig:.2f}σ)")
    print(f"  Δφ    = {np.degrees(delta_phi):+.1f}° ± {np.degrees(delta_phi_err):.1f}°  ({phi_sig:.2f}σ)")

    print(f"\nLikelihood Ratio:")
    print(f"  Λ = Δχ² = {delta_chi2:.2f}")
    print(f"  p-value = {p_value:.4f}")
    print(f"  {lr_interpretation}")

    max_sig = max(r_sig, phi_sig)
    if p_value > 0.05 and max_sig < 2:
        verdict = "RANK-1 CONSTRAINT SUPPORTED"
    elif p_value < 0.01 or max_sig > 3:
        verdict = "RANK-1 CONSTRAINT DISFAVORED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    # Save summary
    summary = {
        'channel_A': {
            'r': float(r_A), 'r_err': float(r3_A_err),
            'phi_deg': float(np.degrees(phi_A)), 'phi_err_deg': float(np.degrees(phi3_A_err)),
            'chi2_dof': float(result_A['chi2_dof']),
        },
        'channel_B': {
            'r': float(r_B), 'r_err': float(r3_B_err),
            'phi_deg': float(np.degrees(phi_B)), 'phi_err_deg': float(np.degrees(phi3_B_err)),
            'chi2_dof': float(result_B['chi2_dof']),
        },
        'comparison': {
            'delta_r': float(delta_r), 'delta_r_err': float(delta_r_err), 'delta_r_sigma': float(r_sig),
            'delta_phi_deg': float(np.degrees(delta_phi)), 'delta_phi_err_deg': float(np.degrees(delta_phi_err)),
            'delta_phi_sigma': float(phi_sig),
        },
        'likelihood_ratio': {
            'chi2_unconstrained': float(chi2_unconstrained),
            'chi2_constrained': float(chi2_constrained),
            'delta_chi2': float(delta_chi2),
            'delta_dof': int(delta_dof),
            'p_value': float(p_value),
        },
        'verdict': verdict,
    }

    with open(os.path.join(OUT_DIR, 'rank1_test_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary

if __name__ == "__main__":
    main()
