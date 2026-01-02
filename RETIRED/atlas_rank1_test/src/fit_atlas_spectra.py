#!/usr/bin/env python3
"""
Fit ATLAS J/psi+psi(2S) spectra with 2-BW interference model.
Uses Poisson NLL and bootstrap p-value for rank-1 constraint test.
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

# ============================================================
# ATLAS resonance parameters
# From arXiv:2509.13101 and arXiv:2304.08962
# ============================================================

# X(6900) parameters (from ATLAS PRL 131 151902)
# Mass ~ 6.87-6.91 GeV depending on model
# Width ~ 0.12-0.22 GeV depending on model
ATLAS_X6900 = {
    'm': 6.905,   # GeV (Model A central value approx)
    'w': 0.180,   # GeV (approximate)
}

# X(7200) parameters
# ATLAS sets upper limit, but we need to include it for interference
# Using approximate values from their Model B
ATLAS_X7200 = {
    'm': 7.22,    # GeV
    'w': 0.100,   # GeV (assumed for upper limit calculation)
}

# ATLAS yield ratio upper limit: N(X7200)/N(X6900) < 0.41 @ 95% CL
# This means r^2 < 0.41, so r < 0.64

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
# Model: 2-BW interference + background
# ============================================================

def model_atlas(m, params, channel='4mu'):
    """
    ATLAS spectrum model:
    I(m) = |c_6900 * BW_6900 + c_7200 * BW_7200|^2 + background

    Parameterization:
    - c_6900 = c_norm (real, positive)
    - c_7200 = c_norm * r * exp(i*phi)

    params = [c_norm, r, phi, bg_a, bg_b, bg_c]
    """
    c_norm, r, phi, bg_a, bg_b, bg_c = params

    c_6900 = 1.0
    c_7200 = r * np.exp(1j * phi)

    bw_6900 = bw_amplitude_normalized(m, ATLAS_X6900['m'], ATLAS_X6900['w'])
    bw_7200 = bw_amplitude_normalized(m, ATLAS_X7200['m'], ATLAS_X7200['w'])

    amplitude = c_6900 * bw_6900 + c_7200 * bw_7200
    signal = c_norm * np.abs(amplitude)**2

    # Background: threshold + polynomial
    # Threshold near 2*M(J/psi) + M(psi(2S)) ~ 6.78 GeV
    m_thr = 6.78
    threshold = np.where(m > m_thr, 1 - np.exp(-(m - m_thr) / 0.15), 0)

    m_ref = 7.5
    background = (bg_a + bg_b * (m - m_ref) + bg_c * (m - m_ref)**2) * threshold
    background = np.maximum(background, 0)

    return signal + background


def poisson_nll(params, m, y, channel='4mu'):
    """Poisson negative log-likelihood."""
    c_norm, r, phi, bg_a, bg_b, bg_c = params

    if c_norm < 0 or r < 0 or bg_a < -5:
        return 1e20

    mu = model_atlas(m, params, channel)
    mu = np.maximum(mu, 1e-10)

    # Poisson NLL
    nll = np.sum(mu - y * np.log(mu))

    return nll


def chi2_func(params, m, y, yerr, channel='4mu'):
    """Chi-squared for comparison."""
    c_norm, r, phi, bg_a, bg_b, bg_c = params
    if c_norm < 0 or r < 0:
        return 1e20
    y_model = model_atlas(m, params, channel)
    yerr_safe = np.maximum(yerr, 1.0)
    return np.sum(((y - y_model) / yerr_safe)**2)


# ============================================================
# Joint fit with rank-1 constraint
# ============================================================

def nll_joint_unconstrained(params_all, m_4mu, y_4mu, m_4mu2pi, y_4mu2pi):
    """Unconstrained joint NLL."""
    # 4mu: c_norm, r, phi, bg_a, bg_b, bg_c
    params_4mu = params_all[:6]
    # 4mu+2pi: c_norm, r, phi, bg_a, bg_b, bg_c
    params_4mu2pi = params_all[6:]

    nll_4mu = poisson_nll(params_4mu, m_4mu, y_4mu, '4mu')
    nll_4mu2pi = poisson_nll(params_4mu2pi, m_4mu2pi, y_4mu2pi, '4mu+2pi')

    return nll_4mu + nll_4mu2pi


def nll_joint_constrained(params_shared, m_4mu, y_4mu, m_4mu2pi, y_4mu2pi):
    """Constrained joint NLL with shared R."""
    # Shared: r, phi
    r_shared = params_shared[0]
    phi_shared = params_shared[1]

    # 4mu specific: c_norm, bg_a, bg_b, bg_c
    c_norm_4mu = params_shared[2]
    bg_a_4mu = params_shared[3]
    bg_b_4mu = params_shared[4]
    bg_c_4mu = params_shared[5]

    # 4mu+2pi specific: c_norm, bg_a, bg_b, bg_c
    c_norm_4mu2pi = params_shared[6]
    bg_a_4mu2pi = params_shared[7]
    bg_b_4mu2pi = params_shared[8]
    bg_c_4mu2pi = params_shared[9]

    params_4mu = [c_norm_4mu, r_shared, phi_shared, bg_a_4mu, bg_b_4mu, bg_c_4mu]
    params_4mu2pi = [c_norm_4mu2pi, r_shared, phi_shared, bg_a_4mu2pi, bg_b_4mu2pi, bg_c_4mu2pi]

    nll_4mu = poisson_nll(params_4mu, m_4mu, y_4mu, '4mu')
    nll_4mu2pi = poisson_nll(params_4mu2pi, m_4mu2pi, y_4mu2pi, '4mu+2pi')

    return nll_4mu + nll_4mu2pi


# ============================================================
# Fitting functions
# ============================================================

def fit_channel(df, channel, verbose=True):
    """Fit a single ATLAS channel."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Fitting ATLAS {channel}")
        print(f"{'='*60}")

    # Filter to bins with data
    df_fit = df[df['has_data']].copy()

    m = df_fit['m_center'].values
    y = df_fit['count'].values
    yerr = df_fit['sigma_total'].values

    if verbose:
        print(f"Data points: {len(m)}")
        print(f"Total counts: {y.sum():.0f}")

    # Bounds: c_norm, r, phi, bg_a, bg_b, bg_c
    bounds = [
        (0.1, 1000),       # c_norm
        (0.01, 2.0),       # r (ATLAS UL is r^2 < 0.41, so r < 0.64)
        (-np.pi, np.pi),   # phi
        (-5, 50),          # bg_a
        (-20, 20),         # bg_b
        (-20, 20),         # bg_c
    ]

    result = differential_evolution(
        poisson_nll, bounds, args=(m, y, channel),
        seed=42, maxiter=2000, tol=1e-8, polish=True, workers=1
    )

    params = result.x
    nll_val = result.fun

    # Compute chi2 for comparison
    y_model = model_atlas(m, params, channel)
    chi2_val = np.sum(((y - y_model) / np.maximum(yerr, 1))**2)
    dof = len(m) - len(params)
    chi2_dof = chi2_val / dof if dof > 0 else chi2_val

    if verbose:
        print(f"\nFit results:")
        print(f"  c_norm = {params[0]:.2f}")
        print(f"  r = {params[1]:.4f}, phi = {np.degrees(params[2]):.1f}deg")
        print(f"  background = {params[3]:.2f} + {params[4]:.2f}*(m-7.5) + {params[5]:.2f}*(m-7.5)^2")
        print(f"  Poisson NLL = {nll_val:.2f}")
        print(f"  chi2/dof = {chi2_dof:.2f}")

    return {
        'params': params,
        'nll': nll_val,
        'chi2': chi2_val,
        'dof': dof,
        'chi2_dof': chi2_dof,
        'r': params[1],
        'phi': params[2],
        'm': m, 'y': y, 'yerr': yerr,
    }


def fit_joint_constrained_wrapper(df_4mu, df_4mu2pi, verbose=True):
    """Joint fit with R_4mu = R_4mu2pi constraint."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Joint Constrained Fit (R_4mu = R_4mu2pi)")
        print(f"{'='*60}")

    df_4mu_fit = df_4mu[df_4mu['has_data']].copy()
    df_4mu2pi_fit = df_4mu2pi[df_4mu2pi['has_data']].copy()

    m_4mu = df_4mu_fit['m_center'].values
    y_4mu = df_4mu_fit['count'].values
    m_4mu2pi = df_4mu2pi_fit['m_center'].values
    y_4mu2pi = df_4mu2pi_fit['count'].values

    if verbose:
        print(f"4mu: {len(m_4mu)} points, 4mu+2pi: {len(m_4mu2pi)} points")

    # Bounds: r, phi, c_norm_4mu, bg_a_4mu, bg_b_4mu, bg_c_4mu, c_norm_4mu2pi, bg_a_4mu2pi, bg_b_4mu2pi, bg_c_4mu2pi
    bounds = [
        (0.01, 2.0),       # r_shared
        (-np.pi, np.pi),   # phi_shared
        (0.1, 1000),       # c_norm_4mu
        (-5, 50),          # bg_a_4mu
        (-20, 20),         # bg_b_4mu
        (-20, 20),         # bg_c_4mu
        (0.1, 1000),       # c_norm_4mu2pi
        (-5, 50),          # bg_a_4mu2pi
        (-20, 20),         # bg_b_4mu2pi
        (-20, 20),         # bg_c_4mu2pi
    ]

    result = differential_evolution(
        nll_joint_constrained, bounds, args=(m_4mu, y_4mu, m_4mu2pi, y_4mu2pi),
        seed=42, maxiter=3000, tol=1e-8, polish=True, workers=1
    )

    params = result.x
    nll_val = result.fun

    if verbose:
        print(f"\nConstrained fit:")
        print(f"  r_shared = {params[0]:.4f}, phi_shared = {np.degrees(params[1]):.1f}deg")
        print(f"  Total NLL = {nll_val:.2f}")

    return {
        'params': params,
        'nll': nll_val,
        'n_params': len(params),
        'r': params[0],
        'phi': params[1],
        'm_4mu': m_4mu, 'y_4mu': y_4mu,
        'm_4mu2pi': m_4mu2pi, 'y_4mu2pi': y_4mu2pi,
    }


# ============================================================
# Bootstrap p-value
# ============================================================

def compute_lambda_single_bootstrap(args):
    """Compute Lambda for one bootstrap replicate."""
    i, m_4mu, y_4mu_orig, m_4mu2pi, y_4mu2pi_orig, perturbed_4mu, perturbed_4mu2pi = args

    np.random.seed(i)

    # Resample with Poisson + digitization jitter
    if perturbed_4mu is not None and i < len(perturbed_4mu):
        y_4mu_pert = perturbed_4mu[i % len(perturbed_4mu)]
        # Only use bins with data
        y_4mu = np.random.poisson(np.maximum(y_4mu_pert, 0.1))
    else:
        y_4mu = np.random.poisson(np.maximum(y_4mu_orig, 0.1))

    if perturbed_4mu2pi is not None and i < len(perturbed_4mu2pi):
        y_4mu2pi_pert = perturbed_4mu2pi[i % len(perturbed_4mu2pi)]
        y_4mu2pi = np.random.poisson(np.maximum(y_4mu2pi_pert, 0.1))
    else:
        y_4mu2pi = np.random.poisson(np.maximum(y_4mu2pi_orig, 0.1))

    # Fit unconstrained
    bounds_single = [(0.1, 1000), (0.01, 2.0), (-np.pi, np.pi), (-5, 50), (-20, 20), (-20, 20)]

    try:
        res_4mu = differential_evolution(poisson_nll, bounds_single, args=(m_4mu, y_4mu, '4mu'),
                                         seed=i, maxiter=300, tol=1e-4, polish=False, workers=1)
        res_4mu2pi = differential_evolution(poisson_nll, bounds_single, args=(m_4mu2pi, y_4mu2pi, '4mu+2pi'),
                                            seed=i, maxiter=300, tol=1e-4, polish=False, workers=1)
        nll_unc = res_4mu.fun + res_4mu2pi.fun

        # Fit constrained
        bounds_con = [(0.01, 2.0), (-np.pi, np.pi), (0.1, 1000), (-5, 50), (-20, 20), (-20, 20),
                      (0.1, 1000), (-5, 50), (-20, 20), (-20, 20)]
        res_con = differential_evolution(nll_joint_constrained, bounds_con,
                                         args=(m_4mu, y_4mu, m_4mu2pi, y_4mu2pi),
                                         seed=i, maxiter=300, tol=1e-4, polish=False, workers=1)
        nll_con = res_con.fun

        Lambda = nll_con - nll_unc
        return Lambda
    except:
        return np.nan


def bootstrap_pvalue(result_4mu, result_4mu2pi, result_con, perturbed_4mu, perturbed_4mu2pi, n_boot=300, verbose=True):
    """Compute bootstrap p-value."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Bootstrap p-value ({n_boot} replicates)")
        print(f"{'='*60}")

    m_4mu = result_4mu['m']
    y_4mu_orig = result_4mu['y']
    m_4mu2pi = result_4mu2pi['m']
    y_4mu2pi_orig = result_4mu2pi['y']

    # Observed Lambda
    nll_unc = result_4mu['nll'] + result_4mu2pi['nll']
    nll_con = result_con['nll']
    Lambda_obs = nll_con - nll_unc

    if verbose:
        print(f"Observed Lambda = {Lambda_obs:.2f}")

    # Filter perturbed counts to only include bins with data
    # Need to match the indices
    df_4mu_temp = pd.read_csv(os.path.join(DATA_DIR, 'derived', '4mu_bins.csv'))
    df_4mu2pi_temp = pd.read_csv(os.path.join(DATA_DIR, 'derived', '4mu+2pi_bins.csv'))

    mask_4mu = df_4mu_temp['has_data'].values
    mask_4mu2pi = df_4mu2pi_temp['has_data'].values

    if perturbed_4mu is not None:
        perturbed_4mu_filtered = perturbed_4mu[:, mask_4mu]
    else:
        perturbed_4mu_filtered = None

    if perturbed_4mu2pi is not None:
        perturbed_4mu2pi_filtered = perturbed_4mu2pi[:, mask_4mu2pi]
    else:
        perturbed_4mu2pi_filtered = None

    args_list = [(i, m_4mu, y_4mu_orig, m_4mu2pi, y_4mu2pi_orig,
                  perturbed_4mu_filtered, perturbed_4mu2pi_filtered) for i in range(n_boot)]

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

    p_value = np.mean(Lambda_boot >= Lambda_obs)

    if verbose:
        print(f"Valid replicates: {n_valid}/{n_boot}")
        print(f"Bootstrap Lambda: mean={np.mean(Lambda_boot):.2f}, std={np.std(Lambda_boot):.2f}")
        print(f"p-value = {p_value:.4f}")

    return p_value, Lambda_obs, Lambda_boot


# ============================================================
# Plotting
# ============================================================

def plot_fit(result, channel, output_path):
    """Plot fit result."""
    m = result['m']
    y = result['y']
    yerr = result['yerr']
    params = result['params']

    m_fine = np.linspace(m.min()-0.1, m.max()+0.1, 300)
    y_fit = model_atlas(m_fine, params, channel)

    # Background only
    m_thr = 6.78
    m_ref = 7.5
    threshold = np.where(m_fine > m_thr, 1 - np.exp(-(m_fine - m_thr) / 0.15), 0)
    bg_only = (params[3] + params[4] * (m_fine - m_ref) + params[5] * (m_fine - m_ref)**2) * threshold
    bg_only = np.maximum(bg_only, 0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True)

    ax = axes[0]
    ax.errorbar(m, y, yerr=yerr, fmt='ko', markersize=5, capsize=2, label='ATLAS data')
    ax.plot(m_fine, y_fit, 'r-', lw=2, label='2-BW interference fit')
    ax.plot(m_fine, bg_only, 'g--', lw=1.5, alpha=0.7, label='Background')
    ax.set_ylabel('Events / 50 MeV')
    ax.set_title(f'ATLAS {channel}: chi2/dof = {result["chi2_dof"]:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pulls
    y_fit_at_data = model_atlas(m, params, channel)
    pulls = (y - y_fit_at_data) / np.maximum(yerr, 1)

    ax = axes[1]
    ax.bar(m, pulls, width=0.04, color='steelblue', alpha=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(2, color='r', ls='--', alpha=0.5)
    ax.axhline(-2, color='r', ls='--', alpha=0.5)
    ax.set_xlabel('m(J/psi + psi(2S)) [GeV]')
    ax.set_ylabel('Pull')
    ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_bootstrap_hist(Lambda_boot, Lambda_obs, p_value, output_path):
    """Plot bootstrap Lambda distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(Lambda_boot, bins=30, color='steelblue', alpha=0.7, edgecolor='black', label='Bootstrap')
    ax.axvline(Lambda_obs, color='red', lw=2, ls='--', label=f'Observed = {Lambda_obs:.1f}')

    ax.set_xlabel('Lambda = NLL_constrained - NLL_unconstrained')
    ax.set_ylabel('Count')
    ax.set_title(f'ATLAS Bootstrap Likelihood Ratio Distribution (p = {p_value:.3f})')
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
    print("ATLAS Rank-1 Bottleneck Test")
    print("="*70)

    # Load extracted data
    df_4mu = pd.read_csv(os.path.join(DATA_DIR, 'derived', '4mu_bins.csv'))
    df_4mu2pi = pd.read_csv(os.path.join(DATA_DIR, 'derived', '4mu+2pi_bins.csv'))

    print(f"\n4mu: {len(df_4mu)} bins, {df_4mu['has_data'].sum()} with data")
    print(f"4mu+2pi: {len(df_4mu2pi)} bins, {df_4mu2pi['has_data'].sum()} with data")

    # Load perturbed counts
    perturbed_4mu = np.load(os.path.join(DATA_DIR, 'derived', '4mu_perturbed.npy'))
    perturbed_4mu2pi = np.load(os.path.join(DATA_DIR, 'derived', '4mu+2pi_perturbed.npy'))

    # Fit individual channels
    result_4mu = fit_channel(df_4mu, '4mu')
    plot_fit(result_4mu, '4mu', os.path.join(OUT_DIR, 'fit_ATLAS_4mu.png'))

    result_4mu2pi = fit_channel(df_4mu2pi, '4mu+2pi')
    plot_fit(result_4mu2pi, '4mu+2pi', os.path.join(OUT_DIR, 'fit_ATLAS_4mu2pi.png'))

    # Joint constrained fit
    result_con = fit_joint_constrained_wrapper(df_4mu, df_4mu2pi)

    # Likelihood ratio
    Lambda_obs = result_con['nll'] - (result_4mu['nll'] + result_4mu2pi['nll'])

    print(f"\n{'='*60}")
    print("LIKELIHOOD RATIO")
    print(f"{'='*60}")
    print(f"Unconstrained NLL: {result_4mu['nll'] + result_4mu2pi['nll']:.2f} (12 params)")
    print(f"Constrained NLL: {result_con['nll']:.2f} (10 params)")
    print(f"Lambda = {Lambda_obs:.2f}")

    # Bootstrap p-value
    p_value, Lambda_obs, Lambda_boot = bootstrap_pvalue(
        result_4mu, result_4mu2pi, result_con,
        perturbed_4mu, perturbed_4mu2pi, n_boot=300
    )

    if len(Lambda_boot) > 0:
        plot_bootstrap_hist(Lambda_boot, Lambda_obs, p_value, os.path.join(OUT_DIR, 'bootstrap_Lambda_ATLAS_hist.png'))

    # Final results
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    r_4mu = result_4mu['r']
    phi_4mu = result_4mu['phi']
    r_4mu2pi = result_4mu2pi['r']
    phi_4mu2pi = result_4mu2pi['phi']
    r_shared = result_con['r']
    phi_shared = result_con['phi']

    print(f"\n4mu channel:")
    print(f"  r = {r_4mu:.4f}, phi = {np.degrees(phi_4mu):.1f}deg")
    print(f"  chi2/dof = {result_4mu['chi2_dof']:.2f}")

    print(f"\n4mu+2pi channel:")
    print(f"  r = {r_4mu2pi:.4f}, phi = {np.degrees(phi_4mu2pi):.1f}deg")
    print(f"  chi2/dof = {result_4mu2pi['chi2_dof']:.2f}")

    print(f"\nShared (constrained):")
    print(f"  r_shared = {r_shared:.4f}, phi_shared = {np.degrees(phi_shared):.1f}deg")

    print(f"\nLikelihood ratio test:")
    print(f"  Lambda = {Lambda_obs:.2f}")
    print(f"  Bootstrap p-value = {p_value:.4f}")

    # Verdict
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
        'channel_4mu': {
            'r': float(r_4mu),
            'phi_deg': float(np.degrees(phi_4mu)),
            'chi2_dof': float(result_4mu['chi2_dof']),
            'nll': float(result_4mu['nll']),
        },
        'channel_4mu2pi': {
            'r': float(r_4mu2pi),
            'phi_deg': float(np.degrees(phi_4mu2pi)),
            'chi2_dof': float(result_4mu2pi['chi2_dof']),
            'nll': float(result_4mu2pi['nll']),
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

    with open(os.path.join(OUT_DIR, 'ATLAS_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save to CSV
    pd.DataFrame([{
        'channel': '4mu', 'r': r_4mu, 'phi_deg': np.degrees(phi_4mu), 'chi2_dof': result_4mu['chi2_dof'],
    }, {
        'channel': '4mu+2pi', 'r': r_4mu2pi, 'phi_deg': np.degrees(phi_4mu2pi), 'chi2_dof': result_4mu2pi['chi2_dof'],
    }, {
        'channel': 'shared', 'r': r_shared, 'phi_deg': np.degrees(phi_shared), 'chi2_dof': np.nan,
    }]).to_csv(os.path.join(OUT_DIR, 'ATLAS_summary.csv'), index=False)

    return summary


if __name__ == "__main__":
    main()
