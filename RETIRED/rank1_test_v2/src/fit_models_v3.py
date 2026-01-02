#!/usr/bin/env python3
"""
Rank-1 bottleneck test v3: Improved fitting with bootstrap p-value.

Key improvements:
1. Channel B uses vector-extracted data (no pixel centroids)
2. Poisson NLL for fitting (better for low counts)
3. Bootstrap p-value (not Wilks approximation)
4. Combined digitization + Poisson systematic
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

# ============================================================
# Resonance parameters from CMS publications
# ============================================================

PARAMS_A = {
    'bw1': {'m': 6.638, 'w': 0.440},  # X(6600)
    'bw2': {'m': 6.847, 'w': 0.191},  # X(6900)
    'bw3': {'m': 7.134, 'w': 0.097},  # X(7100)
}

PARAMS_B = {
    'bw2': {'m': 6.876, 'w': 0.253},  # X(6900)
    'bw3': {'m': 7.169, 'w': 0.154},  # X(7100)
}

# ============================================================
# Breit-Wigner functions
# ============================================================

def bw_amplitude(m, m0, w):
    """Relativistic Breit-Wigner amplitude."""
    return 1.0 / (m0**2 - m**2 - 1j * m0 * w)

def bw_amplitude_normalized(m, m0, w):
    """Normalized BW with peak amplitude ~ 1."""
    bw = bw_amplitude(m, m0, w)
    norm = np.abs(bw_amplitude(m0, m0, w))
    return bw / norm

# ============================================================
# Channel A model (unchanged from v2)
# ============================================================

def model_channel_A(m, params):
    """3-BW interference + quadratic background."""
    c2_norm, r1, phi1, r3, phi3, bg_a, bg_b, bg_c = params

    c1 = r1 * np.exp(1j * phi1)
    c2 = 1.0
    c3 = r3 * np.exp(1j * phi3)

    bw1 = bw_amplitude_normalized(m, PARAMS_A['bw1']['m'], PARAMS_A['bw1']['w'])
    bw2 = bw_amplitude_normalized(m, PARAMS_A['bw2']['m'], PARAMS_A['bw2']['w'])
    bw3 = bw_amplitude_normalized(m, PARAMS_A['bw3']['m'], PARAMS_A['bw3']['w'])

    amplitude = c1 * bw1 + c2 * bw2 + c3 * bw3
    signal = c2_norm * np.abs(amplitude)**2

    m_ref = 6.9
    background = bg_a + bg_b * (m - m_ref) + bg_c * (m - m_ref)**2
    background = np.maximum(background, 0)

    return signal + background

def chi2_channel_A(params, m, y, yerr):
    """Chi-squared for Channel A."""
    c2_norm, r1, phi1, r3, phi3, bg_a, bg_b, bg_c = params
    if c2_norm < 0 or r1 < 0 or r3 < 0 or bg_a < -50:
        return 1e20
    y_model = model_channel_A(m, params)
    yerr_safe = np.maximum(yerr, 1.0)
    return np.sum(((y - y_model) / yerr_safe)**2)

# ============================================================
# Channel B model with Poisson NLL
# ============================================================

def model_channel_B(m, params):
    """2-BW interference + smooth threshold background."""
    c2_norm, r3, phi3, bg_a, bg_b = params

    c2 = 1.0
    c3 = r3 * np.exp(1j * phi3)

    bw2 = bw_amplitude_normalized(m, PARAMS_B['bw2']['m'], PARAMS_B['bw2']['w'])
    bw3 = bw_amplitude_normalized(m, PARAMS_B['bw3']['m'], PARAMS_B['bw3']['w'])

    amplitude = c2 * bw2 + c3 * bw3
    signal = c2_norm * np.abs(amplitude)**2

    # Smooth threshold + linear background
    m_thr = 7.0
    threshold = np.where(m > m_thr, 1 - np.exp(-(m - m_thr) / 0.2), 0)
    background = (bg_a + bg_b * (m - 8.0)) * threshold
    background = np.maximum(background, 0)

    return signal + background

def poisson_nll_channel_B(params, m, y):
    """Poisson negative log-likelihood for Channel B."""
    c2_norm, r3, phi3, bg_a, bg_b = params
    if c2_norm < 0 or r3 < 0 or bg_a < -10:
        return 1e20

    mu = model_channel_B(m, params)
    mu = np.maximum(mu, 1e-10)  # Avoid log(0)

    # Poisson NLL: sum(mu - y*log(mu))
    nll = np.sum(mu - y * np.log(mu))

    # Add penalty for unreasonable parameters
    if c2_norm > 1e5 or r3 > 10:
        nll += 1e10

    return nll

def chi2_channel_B(params, m, y, yerr):
    """Chi-squared for Channel B (for comparison)."""
    c2_norm, r3, phi3, bg_a, bg_b = params
    if c2_norm < 0 or r3 < 0 or bg_a < -10:
        return 1e20
    y_model = model_channel_B(m, params)
    yerr_safe = np.maximum(yerr, 1.0)
    return np.sum(((y - y_model) / yerr_safe)**2)

# ============================================================
# Joint fit with rank-1 constraint
# ============================================================

def nll_joint_unconstrained(params_all, m_A, y_A, yerr_A, m_B, y_B):
    """Joint NLL with separate R for each channel."""
    # Channel A: c2_norm_A, r1, phi1, r3_A, phi3_A, bg_a_A, bg_b_A, bg_c_A
    # Channel B: c2_norm_B, r3_B, phi3_B, bg_a_B, bg_b_B
    params_A = params_all[:8]
    params_B = params_all[8:]

    chi2_A = chi2_channel_A(params_A, m_A, y_A, yerr_A)
    nll_B = poisson_nll_channel_B(params_B, m_B, y_B)

    return chi2_A + 2 * nll_B  # Factor of 2 to compare with chi2

def nll_joint_constrained(params_shared, m_A, y_A, yerr_A, m_B, y_B):
    """Joint NLL with shared R = c3/c2."""
    r3_shared = params_shared[0]
    phi3_shared = params_shared[1]

    c2_norm_A = params_shared[2]
    r1 = params_shared[3]
    phi1 = params_shared[4]
    bg_a_A = params_shared[5]
    bg_b_A = params_shared[6]
    bg_c_A = params_shared[7]

    c2_norm_B = params_shared[8]
    bg_a_B = params_shared[9]
    bg_b_B = params_shared[10]

    params_A = [c2_norm_A, r1, phi1, r3_shared, phi3_shared, bg_a_A, bg_b_A, bg_c_A]
    params_B = [c2_norm_B, r3_shared, phi3_shared, bg_a_B, bg_b_B]

    chi2_A = chi2_channel_A(params_A, m_A, y_A, yerr_A)
    nll_B = poisson_nll_channel_B(params_B, m_B, y_B)

    return chi2_A + 2 * nll_B

# ============================================================
# Fitting functions
# ============================================================

def fit_channel_A(df, fit_window=(6.6, 7.4), verbose=True):
    """Fit Channel A with 3-BW interference model."""
    if verbose:
        print("\n" + "="*60)
        print("Fitting Channel A (J/psi J/psi)")
        print("="*60)

    mask = (df['m_center'] >= fit_window[0]) & (df['m_center'] <= fit_window[1])
    df_fit = df[mask].copy()

    m = df_fit['m_center'].values
    y = df_fit['count'].values
    yerr = df_fit['stat_err'].values

    if verbose:
        print(f"Fit window: {fit_window[0]:.2f} - {fit_window[1]:.2f} GeV")
        print(f"Data points: {len(m)}")

    bounds = [
        (1, 1e5), (0.01, 5), (-np.pi, np.pi), (0.01, 5), (-np.pi, np.pi),
        (-50, 200), (-100, 100), (-100, 100),
    ]

    result = differential_evolution(
        chi2_channel_A, bounds, args=(m, y, yerr),
        seed=42, maxiter=2000, tol=1e-8, polish=True, workers=1
    )

    params = result.x
    chi2_val = result.fun
    dof = len(m) - len(params)
    chi2_dof = chi2_val / dof if dof > 0 else chi2_val

    if verbose:
        print(f"\nFit results:")
        print(f"  r3 = {params[3]:.4f}, phi3 = {np.degrees(params[4]):.1f}deg")
        print(f"  chi2/dof = {chi2_dof:.2f}")

    return {
        'params': params,
        'chi2': chi2_val,
        'dof': dof,
        'chi2_dof': chi2_dof,
        'r3': params[3],
        'phi3': params[4],
        'm': m, 'y': y, 'yerr': yerr,
    }

def fit_channel_B(df, fit_window=(7.0, 9.0), verbose=True):
    """Fit Channel B with Poisson NLL."""
    if verbose:
        print("\n" + "="*60)
        print("Fitting Channel B (J/psi psi(2S)) with Poisson NLL")
        print("="*60)

    mask = (df['m_center'] >= fit_window[0]) & (df['m_center'] <= fit_window[1])
    df_fit = df[mask].copy()

    m = df_fit['m_center'].values
    y = df_fit['count'].values
    yerr = df_fit['sigma_total'].values

    if verbose:
        print(f"Fit window: {fit_window[0]:.2f} - {fit_window[1]:.2f} GeV")
        print(f"Data points: {len(m)}")
        print(f"Total counts: {y.sum():.0f}")

    bounds = [
        (0.1, 500),        # c2_norm
        (0.01, 5),         # r3
        (-np.pi, np.pi),   # phi3
        (-5, 30),          # bg_a
        (-10, 10),         # bg_b
    ]

    # Fit with Poisson NLL
    result = differential_evolution(
        poisson_nll_channel_B, bounds, args=(m, y),
        seed=42, maxiter=2000, tol=1e-8, polish=True, workers=1
    )

    params = result.x
    nll_val = result.fun

    # Compute chi2 equivalent for comparison
    y_model = model_channel_B(m, params)
    chi2_val = np.sum(((y - y_model) / np.maximum(yerr, 1))**2)
    dof = len(m) - len(params)
    chi2_dof = chi2_val / dof if dof > 0 else chi2_val

    if verbose:
        print(f"\nFit results (Poisson NLL):")
        print(f"  c2_norm = {params[0]:.2f}")
        print(f"  r3 = {params[1]:.4f}, phi3 = {np.degrees(params[2]):.1f}deg")
        print(f"  background = {params[3]:.2f} + {params[4]:.2f}*(m-8)")
        print(f"  Poisson NLL = {nll_val:.2f}")
        print(f"  chi2/dof (for comparison) = {chi2_dof:.2f}")

    return {
        'params': params,
        'nll': nll_val,
        'chi2': chi2_val,
        'dof': dof,
        'chi2_dof': chi2_dof,
        'r3': params[1],
        'phi3': params[2],
        'm': m, 'y': y, 'yerr': yerr,
    }

def fit_joint_unconstrained(result_A, result_B):
    """Combine individual fits for unconstrained joint result."""
    # Total NLL = chi2_A + 2*nll_B
    total = result_A['chi2'] + 2 * result_B['nll']
    n_params = 8 + 5  # 8 for A, 5 for B
    return {
        'total_nll': total,
        'n_params': n_params,
        'r3_A': result_A['r3'],
        'phi3_A': result_A['phi3'],
        'r3_B': result_B['r3'],
        'phi3_B': result_B['phi3'],
    }

def fit_joint_constrained(df_A, df_B, fit_window_A=(6.6, 7.4), fit_window_B=(7.0, 9.0), verbose=True):
    """Joint fit with R_A = R_B constraint."""
    if verbose:
        print("\n" + "="*60)
        print("Joint Constrained Fit (R_A = R_B)")
        print("="*60)

    # Prepare data
    mask_A = (df_A['m_center'] >= fit_window_A[0]) & (df_A['m_center'] <= fit_window_A[1])
    df_fit_A = df_A[mask_A]
    m_A = df_fit_A['m_center'].values
    y_A = df_fit_A['count'].values
    yerr_A = df_fit_A['stat_err'].values

    mask_B = (df_B['m_center'] >= fit_window_B[0]) & (df_B['m_center'] <= fit_window_B[1])
    df_fit_B = df_B[mask_B]
    m_B = df_fit_B['m_center'].values
    y_B = df_fit_B['count'].values

    if verbose:
        print(f"Channel A: {len(m_A)} points, Channel B: {len(m_B)} points")

    # [r3_shared, phi3_shared, c2_norm_A, r1, phi1, bg_a_A, bg_b_A, bg_c_A, c2_norm_B, bg_a_B, bg_b_B]
    bounds = [
        (0.01, 5),         # r3_shared
        (-np.pi, np.pi),   # phi3_shared
        (1, 1e5),          # c2_norm_A
        (0.01, 5),         # r1
        (-np.pi, np.pi),   # phi1
        (-50, 200),        # bg_a_A
        (-100, 100),       # bg_b_A
        (-100, 100),       # bg_c_A
        (0.1, 500),        # c2_norm_B
        (-5, 30),          # bg_a_B
        (-10, 10),         # bg_b_B
    ]

    result = differential_evolution(
        nll_joint_constrained, bounds,
        args=(m_A, y_A, yerr_A, m_B, y_B),
        seed=42, maxiter=3000, tol=1e-8, polish=True, workers=1
    )

    params = result.x
    total_nll = result.fun
    n_params = len(params)

    if verbose:
        print(f"\nConstrained fit:")
        print(f"  r3_shared = {params[0]:.4f}, phi3_shared = {np.degrees(params[1]):.1f}deg")
        print(f"  Total NLL = {total_nll:.2f}")

    return {
        'params': params,
        'total_nll': total_nll,
        'n_params': n_params,
        'r3': params[0],
        'phi3': params[1],
        'm_A': m_A, 'y_A': y_A, 'yerr_A': yerr_A,
        'm_B': m_B, 'y_B': y_B,
    }

# ============================================================
# Bootstrap p-value calculation
# ============================================================

def compute_lambda_single_bootstrap(args):
    """Compute Lambda for a single bootstrap replicate."""
    i, m_A, y_A_orig, yerr_A, m_B, y_B_orig, perturbed_B = args

    np.random.seed(i)

    # Resample Channel A: Poisson
    y_A = np.random.poisson(np.maximum(y_A_orig, 0.1))
    yerr_A_boot = np.sqrt(np.maximum(y_A, 1))

    # Resample Channel B: use perturbed counts if available
    if perturbed_B is not None and i < len(perturbed_B):
        y_B_pert = perturbed_B[i % len(perturbed_B)]
        y_B = np.random.poisson(np.maximum(y_B_pert, 0.1))
    else:
        y_B = np.random.poisson(np.maximum(y_B_orig, 0.1))

    # Fit unconstrained
    bounds_A = [(1, 1e5), (0.01, 5), (-np.pi, np.pi), (0.01, 5), (-np.pi, np.pi),
                (-50, 200), (-100, 100), (-100, 100)]
    bounds_B = [(0.1, 500), (0.01, 5), (-np.pi, np.pi), (-5, 30), (-10, 10)]

    try:
        res_A = differential_evolution(chi2_channel_A, bounds_A, args=(m_A, y_A, yerr_A_boot),
                                       seed=i, maxiter=300, tol=1e-4, polish=False, workers=1)
        res_B = differential_evolution(poisson_nll_channel_B, bounds_B, args=(m_B, y_B),
                                       seed=i, maxiter=300, tol=1e-4, polish=False, workers=1)
        nll_unc = res_A.fun + 2 * res_B.fun

        # Fit constrained
        bounds_con = [(0.01, 5), (-np.pi, np.pi), (1, 1e5), (0.01, 5), (-np.pi, np.pi),
                      (-50, 200), (-100, 100), (-100, 100), (0.1, 500), (-5, 30), (-10, 10)]
        res_con = differential_evolution(nll_joint_constrained, bounds_con,
                                         args=(m_A, y_A, yerr_A_boot, m_B, y_B),
                                         seed=i, maxiter=300, tol=1e-4, polish=False, workers=1)
        nll_con = res_con.fun

        Lambda = nll_con - nll_unc
        return Lambda
    except:
        return np.nan

def bootstrap_pvalue(result_A, result_B, result_con, n_boot=300, perturbed_B=None, verbose=True):
    """Compute bootstrap p-value for the rank-1 constraint."""
    if verbose:
        print("\n" + "="*60)
        print(f"Bootstrap p-value ({n_boot} replicates)")
        print("="*60)

    m_A = result_A['m']
    y_A_orig = result_A['y']
    yerr_A = result_A['yerr']
    m_B = result_B['m']
    y_B_orig = result_B['y']

    # Observed Lambda
    nll_unc = result_A['chi2'] + 2 * result_B['nll']
    nll_con = result_con['total_nll']
    Lambda_obs = nll_con - nll_unc

    if verbose:
        print(f"Observed Lambda = {Lambda_obs:.2f}")

    # Prepare args for parallel computation
    args_list = [(i, m_A, y_A_orig, yerr_A, m_B, y_B_orig, perturbed_B) for i in range(n_boot)]

    # Run bootstrap (parallel)
    n_workers = max(1, cpu_count() - 1)
    if verbose:
        print(f"Running {n_boot} bootstrap fits using {n_workers} workers...")

    with Pool(n_workers) as pool:
        Lambda_boot = list(pool.map(compute_lambda_single_bootstrap, args_list))

    Lambda_boot = np.array([L for L in Lambda_boot if not np.isnan(L)])
    n_valid = len(Lambda_boot)

    if n_valid == 0:
        print("WARNING: No valid bootstrap replicates!")
        return 1.0, Lambda_obs, np.array([])

    # p-value: fraction of bootstrap Lambda >= observed Lambda
    p_value = np.mean(Lambda_boot >= Lambda_obs)

    if verbose:
        print(f"Valid replicates: {n_valid}/{n_boot}")
        print(f"Bootstrap Lambda: mean={np.mean(Lambda_boot):.2f}, std={np.std(Lambda_boot):.2f}")
        print(f"p-value = {p_value:.4f}")

    return p_value, Lambda_obs, Lambda_boot

# ============================================================
# Plotting
# ============================================================

def plot_fit_B_v3(result, output_path):
    """Plot Channel B fit."""
    m = result['m']
    y = result['y']
    yerr = result['yerr']
    params = result['params']

    m_fine = np.linspace(m.min()-0.05, m.max()+0.05, 300)
    y_fit = model_channel_B(m_fine, params)

    # Background
    m_thr = 7.0
    threshold = np.where(m_fine > m_thr, 1 - np.exp(-(m_fine - m_thr) / 0.2), 0)
    bg_only = (params[3] + params[4] * (m_fine - 8.0)) * threshold
    bg_only = np.maximum(bg_only, 0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True)

    ax = axes[0]
    ax.errorbar(m, y, yerr=yerr, fmt='ko', markersize=5, capsize=2, label='Vector extraction')
    ax.plot(m_fine, y_fit, 'r-', lw=2, label='2-BW interference (Poisson NLL)')
    ax.plot(m_fine, bg_only, 'g--', lw=1.5, alpha=0.7, label='Background')
    ax.set_ylabel('Candidates / 40 MeV')
    ax.set_title(f'Channel B (J/psi psi(2S)): chi2/dof = {result["chi2_dof"]:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pulls
    y_fit_at_data = model_channel_B(m, params)
    pulls = (y - y_fit_at_data) / np.maximum(yerr, 1)

    ax = axes[1]
    ax.bar(m, pulls, width=0.035, color='steelblue', alpha=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(2, color='r', ls='--', alpha=0.5)
    ax.axhline(-2, color='r', ls='--', alpha=0.5)
    ax.set_xlabel('m(J/psi psi(2S)) [GeV]')
    ax.set_ylabel('Pull')
    ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_bootstrap_hist(Lambda_boot, Lambda_obs, p_value, output_path):
    """Plot histogram of bootstrap Lambda values."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(Lambda_boot, bins=30, color='steelblue', alpha=0.7, edgecolor='black', label='Bootstrap')
    ax.axvline(Lambda_obs, color='red', lw=2, ls='--', label=f'Observed = {Lambda_obs:.1f}')

    ax.set_xlabel('Lambda = NLL_constrained - NLL_unconstrained')
    ax.set_ylabel('Count')
    ax.set_title(f'Bootstrap Likelihood Ratio Distribution (p = {p_value:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# ============================================================
# Main
# ============================================================

def main():
    print("="*70)
    print("RANK-1 BOTTLENECK TEST v3")
    print("Vector extraction + Poisson NLL + Bootstrap p-value")
    print("="*70)

    # Load data
    df_A = pd.read_csv(os.path.join(DATA_DIR, 'hepdata', 'channel_A_mass_spectrum.csv'))
    print(f"\nChannel A: {len(df_A)} bins from HEPData")

    df_B = pd.read_csv(os.path.join(DATA_DIR, 'derived', 'channel_B_vector_bins.csv'))
    print(f"Channel B: {len(df_B)} bins from vector extraction")

    # Load perturbed counts for bootstrap
    perturbed_path = os.path.join(DATA_DIR, 'derived', 'channel_B_perturbed_counts.npy')
    if os.path.exists(perturbed_path):
        perturbed_B = np.load(perturbed_path)
        print(f"Loaded {len(perturbed_B)} perturbation realizations for bootstrap")
    else:
        perturbed_B = None
        print("No perturbation file found, using Poisson-only bootstrap")

    # Fit Channel A
    result_A = fit_channel_A(df_A)

    # Fit Channel B
    result_B = fit_channel_B(df_B)
    plot_fit_B_v3(result_B, os.path.join(OUT_DIR, 'fit_B_plot_v3.png'))

    # Joint constrained fit
    result_con = fit_joint_constrained(df_A, df_B)

    # Compute unconstrained total NLL
    result_unc = fit_joint_unconstrained(result_A, result_B)

    # Likelihood ratio
    Lambda_obs = result_con['total_nll'] - result_unc['total_nll']

    print("\n" + "="*60)
    print("LIKELIHOOD RATIO")
    print("="*60)
    print(f"Unconstrained NLL: {result_unc['total_nll']:.2f} ({result_unc['n_params']} params)")
    print(f"Constrained NLL: {result_con['total_nll']:.2f} ({result_con['n_params']} params)")
    print(f"Lambda = {Lambda_obs:.2f}")

    # Bootstrap p-value
    p_value, Lambda_obs, Lambda_boot = bootstrap_pvalue(
        result_A, result_B, result_con,
        n_boot=300, perturbed_B=perturbed_B
    )

    if len(Lambda_boot) > 0:
        plot_bootstrap_hist(Lambda_boot, Lambda_obs, p_value, os.path.join(OUT_DIR, 'bootstrap_Lambda_hist.png'))

    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    r_A = result_A['r3']
    phi_A = result_A['phi3']
    r_B = result_B['r3']
    phi_B = result_B['phi3']
    r_shared = result_con['r3']
    phi_shared = result_con['phi3']

    print(f"\nChannel A (J/psi J/psi):")
    print(f"  r_A   = {r_A:.4f}")
    print(f"  phi_A = {np.degrees(phi_A):.1f}deg")
    print(f"  chi2/dof = {result_A['chi2_dof']:.2f}")

    print(f"\nChannel B (J/psi psi(2S)):")
    print(f"  r_B   = {r_B:.4f}")
    print(f"  phi_B = {np.degrees(phi_B):.1f}deg")
    print(f"  chi2/dof = {result_B['chi2_dof']:.2f}")

    print(f"\nShared (constrained):")
    print(f"  r_shared   = {r_shared:.4f}")
    print(f"  phi_shared = {np.degrees(phi_shared):.1f}deg")

    print(f"\nLikelihood ratio test:")
    print(f"  Lambda = {Lambda_obs:.2f}")
    print(f"  Bootstrap p-value = {p_value:.4f}")

    # Verdict
    delta_r = abs(r_A - r_B)
    delta_phi = abs(phi_A - phi_B)
    delta_phi = min(delta_phi, 2*np.pi - delta_phi)

    if p_value > 0.05:
        verdict = "RANK-1 CONSTRAINT SUPPORTED"
    elif p_value > 0.01:
        verdict = "RANK-1 CONSTRAINT SHOWS TENSION"
    else:
        verdict = "RANK-1 CONSTRAINT DISFAVORED"

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    # Save summary
    summary = {
        'channel_A': {
            'r': float(r_A),
            'phi_deg': float(np.degrees(phi_A)),
            'chi2_dof': float(result_A['chi2_dof']),
        },
        'channel_B': {
            'r': float(r_B),
            'phi_deg': float(np.degrees(phi_B)),
            'chi2_dof': float(result_B['chi2_dof']),
        },
        'shared': {
            'r': float(r_shared),
            'phi_deg': float(np.degrees(phi_shared)),
        },
        'likelihood_ratio': {
            'Lambda': float(Lambda_obs),
            'bootstrap_p_value': float(p_value),
            'n_bootstrap': int(len(Lambda_boot)),
        },
        'verdict': verdict,
    }

    with open(os.path.join(OUT_DIR, 'rank1_test_v3_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save vector bins CSV to out folder
    df_B[['m_low', 'm_high', 'm_center', 'count', 'sigma_total']].to_csv(
        os.path.join(OUT_DIR, 'channel_B_vector_bins.csv'), index=False
    )

    return summary

if __name__ == "__main__":
    main()
