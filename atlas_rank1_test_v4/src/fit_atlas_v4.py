#!/usr/bin/env python3
"""
ATLAS Rank-1 Test v4 - Publication-grade analysis

Key features:
- Pure Poisson likelihood (no per-bin digitization sigma)
- Correlated nuisance parameters: s_x (x-scale), b_x (x-shift), s_y (y-scale)
- Tight Gaussian priors on nuisances (1% for scales, 0.5% mass width for shift)
- Profile likelihood contours for (r, phi)
- Joint constrained vs unconstrained likelihood ratio test
- Bootstrap p-value (not Wilks approximation)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.special import gammaln
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ATLAS Resonance Parameters (from arXiv:2509.13101)
# =============================================================================
M_X6900 = 6.905  # GeV
W_X6900 = 0.180  # GeV
M_X7200 = 7.220  # GeV (approximate, ATLAS only sets upper limit)
W_X7200 = 0.100  # GeV

# Threshold for J/psi + psi(2S)
M_THRESH = 6.783  # GeV (3.097 + 3.686)

# Nuisance parameter priors (tight constraints)
SIGMA_SX = 0.01   # 1% scale uncertainty
SIGMA_BX = 0.020  # 20 MeV shift uncertainty (~ 0.3% of mass)
SIGMA_SY = 0.01   # 1% y-scale uncertainty


def breit_wigner(m, m0, w):
    """Relativistic Breit-Wigner amplitude (complex)."""
    return 1.0 / (m0**2 - m**2 - 1j * m0 * w)


def threshold_factor(m, m_thresh=M_THRESH):
    """Smooth threshold turn-on."""
    return np.sqrt(np.maximum(0, m - m_thresh))


def model_intensity(m, params, apply_nuisance=True):
    """
    Signal + background model.

    params: [c_norm, r, phi, b0, b1, b2, s_x, b_x, s_y]
    - c_norm: overall normalization
    - r, phi: complex ratio R = r * exp(i*phi), where R = c_7200/c_6900
    - b0, b1, b2: background polynomial coefficients
    - s_x: x-scale nuisance (1 = nominal)
    - b_x: x-shift nuisance (0 = nominal)
    - s_y: y-scale nuisance (1 = nominal)
    """
    c_norm, r, phi, b0, b1, b2 = params[:6]

    if apply_nuisance and len(params) >= 9:
        s_x, b_x, s_y = params[6:9]
    else:
        s_x, b_x, s_y = 1.0, 0.0, 1.0

    # Apply nuisance transformations to mass
    m_eff = s_x * m + b_x

    # Phase space
    ps = threshold_factor(m_eff)

    # Complex amplitudes
    c_6900 = c_norm
    c_7200 = c_norm * r * np.exp(1j * phi)

    bw_6900 = breit_wigner(m_eff, M_X6900, W_X6900)
    bw_7200 = breit_wigner(m_eff, M_X7200, W_X7200)

    # Coherent sum
    amplitude = c_6900 * bw_6900 + c_7200 * bw_7200
    signal = ps * np.abs(amplitude)**2

    # Background: smooth polynomial above threshold
    m_ref = 7.5  # reference mass
    dm = m_eff - m_ref
    background = ps * np.maximum(0, b0 + b1 * dm + b2 * dm**2)

    # Total intensity with y-scale nuisance
    intensity = s_y * (signal + background)

    return np.maximum(1e-10, intensity)


def poisson_nll(params, m_centers, counts, bin_width):
    """
    Poisson negative log-likelihood.

    NLL = sum_i [ mu_i - n_i * log(mu_i) + log(n_i!) ]

    Plus Gaussian penalty terms for nuisance parameters.
    """
    mu = model_intensity(m_centers, params, apply_nuisance=True) * bin_width

    # Poisson NLL (ignoring constant log(n!) term)
    nll = np.sum(mu - counts * np.log(np.maximum(1e-10, mu)))

    # Add Gaussian priors for nuisance parameters
    if len(params) >= 9:
        s_x, b_x, s_y = params[6:9]
        nll += 0.5 * ((s_x - 1.0) / SIGMA_SX)**2
        nll += 0.5 * (b_x / SIGMA_BX)**2
        nll += 0.5 * ((s_y - 1.0) / SIGMA_SY)**2

    return nll


def fit_channel(m_centers, counts, bin_width, shared_R=None, verbose=False):
    """
    Fit a single channel.

    If shared_R is provided as (r, phi), those values are fixed.

    Returns: best-fit params, NLL
    """
    # Parameter bounds
    if shared_R is None:
        # Unconstrained fit: all parameters free
        bounds = [
            (1, 100),      # c_norm
            (0, 1.5),      # r
            (-np.pi, np.pi),  # phi
            (0, 200),      # b0
            (-50, 50),     # b1
            (-20, 20),     # b2
            (0.97, 1.03),  # s_x (within 3 sigma)
            (-0.06, 0.06), # b_x (within 3 sigma)
            (0.97, 1.03),  # s_y (within 3 sigma)
        ]

        def objective(p):
            return poisson_nll(p, m_centers, counts, bin_width)
    else:
        # Constrained fit: r and phi fixed
        r_fixed, phi_fixed = shared_R
        bounds = [
            (1, 100),      # c_norm
            (0, 200),      # b0
            (-50, 50),     # b1
            (-20, 20),     # b2
            (0.97, 1.03),  # s_x
            (-0.06, 0.06), # b_x
            (0.97, 1.03),  # s_y
        ]

        def objective(p):
            # Expand to full parameter set
            full_p = [p[0], r_fixed, phi_fixed, p[1], p[2], p[3], p[4], p[5], p[6]]
            return poisson_nll(full_p, m_centers, counts, bin_width)

    # Global optimization with differential evolution
    result = differential_evolution(objective, bounds, seed=42, maxiter=1000,
                                    tol=1e-7, polish=True, workers=1)

    if shared_R is None:
        best_params = result.x
    else:
        p = result.x
        best_params = np.array([p[0], r_fixed, phi_fixed, p[1], p[2], p[3], p[4], p[5], p[6]])

    return best_params, result.fun


def compute_profile_contour(m_centers, counts, bin_width, r_grid, phi_grid,
                            best_nll, best_params, delta_chi2):
    """
    Compute profile likelihood contour for (r, phi).

    For each (r, phi) point, minimize over all other parameters.
    Returns boolean mask where 2*(NLL - best_NLL) < delta_chi2.
    """
    contour = np.zeros((len(r_grid), len(phi_grid)), dtype=bool)
    nll_surface = np.full((len(r_grid), len(phi_grid)), np.inf)

    for i, r in enumerate(r_grid):
        for j, phi in enumerate(phi_grid):
            # Fix r and phi, minimize over other params
            bounds = [
                (1, 100),      # c_norm
                (0, 200),      # b0
                (-50, 50),     # b1
                (-20, 20),     # b2
                (0.97, 1.03),  # s_x
                (-0.06, 0.06), # b_x
                (0.97, 1.03),  # s_y
            ]

            def objective(p):
                full_p = [p[0], r, phi, p[1], p[2], p[3], p[4], p[5], p[6]]
                return poisson_nll(full_p, m_centers, counts, bin_width)

            # Start from best-fit values
            x0 = [best_params[0], best_params[3], best_params[4], best_params[5],
                  best_params[6], best_params[7], best_params[8]]

            try:
                result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
                nll_surface[i, j] = result.fun
                delta_nll = 2 * (result.fun - best_nll)
                contour[i, j] = delta_nll < delta_chi2
            except:
                pass

    return contour, nll_surface


def joint_constrained_fit(data_4mu, data_4mu2pi):
    """
    Perform joint fit with shared (r, phi) across both channels.

    Returns: shared (r, phi), total NLL
    """
    m_4mu, counts_4mu, bw_4mu = data_4mu
    m_4mu2pi, counts_4mu2pi, bw_4mu2pi = data_4mu2pi

    # Joint parameters: [c1, c2, r_shared, phi_shared, b0_1, b1_1, b2_1, b0_2, b1_2, b2_2,
    #                    sx_1, bx_1, sy_1, sx_2, bx_2, sy_2]
    bounds = [
        (1, 100),      # c_norm for 4mu
        (1, 100),      # c_norm for 4mu2pi
        (0, 1.5),      # r_shared
        (-np.pi, np.pi),  # phi_shared
        (0, 200), (-50, 50), (-20, 20),  # background 4mu
        (0, 200), (-50, 50), (-20, 20),  # background 4mu2pi
        (0.97, 1.03), (-0.06, 0.06), (0.97, 1.03),  # nuisances 4mu
        (0.97, 1.03), (-0.06, 0.06), (0.97, 1.03),  # nuisances 4mu2pi
    ]

    def objective(p):
        # 4mu params
        p1 = [p[0], p[2], p[3], p[4], p[5], p[6], p[10], p[11], p[12]]
        # 4mu2pi params
        p2 = [p[1], p[2], p[3], p[7], p[8], p[9], p[13], p[14], p[15]]

        nll1 = poisson_nll(p1, m_4mu, counts_4mu, bw_4mu)
        nll2 = poisson_nll(p2, m_4mu2pi, counts_4mu2pi, bw_4mu2pi)

        return nll1 + nll2

    result = differential_evolution(objective, bounds, seed=42, maxiter=2000,
                                    tol=1e-7, polish=True, workers=1)

    r_shared = result.x[2]
    phi_shared = result.x[3]
    total_nll = result.fun

    return (r_shared, phi_shared), total_nll, result.x


def bootstrap_replicate(args):
    """Single bootstrap replicate for parallel execution."""
    seed, m_4mu, counts_4mu, bw_4mu, m_4mu2pi, counts_4mu2pi, bw_4mu2pi = args

    np.random.seed(seed)

    # Poisson resample
    counts_4mu_boot = np.random.poisson(counts_4mu)
    counts_4mu2pi_boot = np.random.poisson(counts_4mu2pi)

    try:
        # Unconstrained fits
        params_4mu, nll_4mu = fit_channel(m_4mu, counts_4mu_boot, bw_4mu)
        params_4mu2pi, nll_4mu2pi = fit_channel(m_4mu2pi, counts_4mu2pi_boot, bw_4mu2pi)
        nll_unconstrained = nll_4mu + nll_4mu2pi

        # Constrained fit
        data_4mu = (m_4mu, counts_4mu_boot, bw_4mu)
        data_4mu2pi = (m_4mu2pi, counts_4mu2pi_boot, bw_4mu2pi)
        _, nll_constrained, _ = joint_constrained_fit(data_4mu, data_4mu2pi)

        Lambda = 2 * (nll_constrained - nll_unconstrained)
        return Lambda
    except:
        return np.nan


def main():
    """Main analysis pipeline."""

    print("="*70)
    print("ATLAS Rank-1 Test v4 - Publication-grade Analysis")
    print("="*70)
    print()

    # Prepare clean data
    exec(open("/home/primary/DarkBItParticleColiderPredictions/atlas_rank1_test_v4/src/prepare_clean_data.py").read())

    # Load clean data
    df_4mu = pd.read_csv("/home/primary/DarkBItParticleColiderPredictions/atlas_rank1_test_v4/data/derived/4mu_bins.csv")
    df_4mu2pi = pd.read_csv("/home/primary/DarkBItParticleColiderPredictions/atlas_rank1_test_v4/data/derived/4mu+2pi_bins.csv")

    m_4mu = df_4mu['m_center'].values
    counts_4mu = df_4mu['count'].values
    bw_4mu = df_4mu['m_high'].values[0] - df_4mu['m_low'].values[0]

    m_4mu2pi = df_4mu2pi['m_center'].values
    counts_4mu2pi = df_4mu2pi['count'].values
    bw_4mu2pi = df_4mu2pi['m_high'].values[0] - df_4mu2pi['m_low'].values[0]

    print(f"4mu: {len(counts_4mu)} bins, {counts_4mu.sum()} total counts")
    print(f"4mu+2pi: {len(counts_4mu2pi)} bins, {counts_4mu2pi.sum()} total counts")
    print()

    # =========================================================================
    # Fit each channel independently (unconstrained)
    # =========================================================================
    print("="*70)
    print("Unconstrained Fits")
    print("="*70)

    params_4mu, nll_4mu = fit_channel(m_4mu, counts_4mu, bw_4mu, verbose=True)
    print(f"\n4mu channel:")
    print(f"  r = {params_4mu[1]:.4f}")
    print(f"  phi = {np.degrees(params_4mu[2]):.1f} deg")
    print(f"  NLL = {nll_4mu:.2f}")
    print(f"  Nuisances: s_x={params_4mu[6]:.4f}, b_x={params_4mu[7]*1000:.1f} MeV, s_y={params_4mu[8]:.4f}")

    params_4mu2pi, nll_4mu2pi = fit_channel(m_4mu2pi, counts_4mu2pi, bw_4mu2pi, verbose=True)
    print(f"\n4mu+2pi channel:")
    print(f"  r = {params_4mu2pi[1]:.4f}")
    print(f"  phi = {np.degrees(params_4mu2pi[2]):.1f} deg")
    print(f"  NLL = {nll_4mu2pi:.2f}")
    print(f"  Nuisances: s_x={params_4mu2pi[6]:.4f}, b_x={params_4mu2pi[7]*1000:.1f} MeV, s_y={params_4mu2pi[8]:.4f}")

    nll_unconstrained = nll_4mu + nll_4mu2pi

    # =========================================================================
    # Joint constrained fit
    # =========================================================================
    print("\n" + "="*70)
    print("Joint Constrained Fit (shared r, phi)")
    print("="*70)

    data_4mu = (m_4mu, counts_4mu, bw_4mu)
    data_4mu2pi = (m_4mu2pi, counts_4mu2pi, bw_4mu2pi)

    (r_shared, phi_shared), nll_constrained, joint_params = joint_constrained_fit(data_4mu, data_4mu2pi)

    print(f"\nShared parameters:")
    print(f"  r_shared = {r_shared:.4f}")
    print(f"  phi_shared = {np.degrees(phi_shared):.1f} deg")
    print(f"  Total NLL = {nll_constrained:.2f}")

    # =========================================================================
    # Likelihood ratio test
    # =========================================================================
    Lambda = 2 * (nll_constrained - nll_unconstrained)
    print(f"\n" + "="*70)
    print("Likelihood Ratio Test")
    print("="*70)
    print(f"NLL unconstrained: {nll_unconstrained:.2f}")
    print(f"NLL constrained: {nll_constrained:.2f}")
    print(f"Lambda = 2 * delta(NLL) = {Lambda:.3f}")

    # =========================================================================
    # Profile likelihood contours
    # =========================================================================
    print("\n" + "="*70)
    print("Computing Profile Likelihood Contours...")
    print("="*70)

    # Grid for (r, phi)
    r_grid = np.linspace(0.01, 1.4, 50)
    phi_grid = np.linspace(-np.pi, np.pi, 50)

    # 68% and 95% confidence levels (2D chi-square with 2 dof)
    DELTA_CHI2_68 = 2.30
    DELTA_CHI2_95 = 5.99

    print("  Computing 4mu contour...")
    contour_4mu_68, nll_surface_4mu = compute_profile_contour(
        m_4mu, counts_4mu, bw_4mu, r_grid, phi_grid, nll_4mu, params_4mu, DELTA_CHI2_68)
    contour_4mu_95, _ = compute_profile_contour(
        m_4mu, counts_4mu, bw_4mu, r_grid, phi_grid, nll_4mu, params_4mu, DELTA_CHI2_95)

    print("  Computing 4mu+2pi contour...")
    contour_4mu2pi_68, nll_surface_4mu2pi = compute_profile_contour(
        m_4mu2pi, counts_4mu2pi, bw_4mu2pi, r_grid, phi_grid, nll_4mu2pi, params_4mu2pi, DELTA_CHI2_68)
    contour_4mu2pi_95, _ = compute_profile_contour(
        m_4mu2pi, counts_4mu2pi, bw_4mu2pi, r_grid, phi_grid, nll_4mu2pi, params_4mu2pi, DELTA_CHI2_95)

    # Check if shared point lies in contours
    r_idx = np.argmin(np.abs(r_grid - r_shared))
    phi_idx = np.argmin(np.abs(phi_grid - phi_shared))

    in_4mu_68 = contour_4mu_68[r_idx, phi_idx]
    in_4mu_95 = contour_4mu_95[r_idx, phi_idx]
    in_4mu2pi_68 = contour_4mu2pi_68[r_idx, phi_idx]
    in_4mu2pi_95 = contour_4mu2pi_95[r_idx, phi_idx]

    print(f"\nShared point (r={r_shared:.3f}, phi={np.degrees(phi_shared):.1f} deg):")
    print(f"  In 4mu 68% contour: {in_4mu_68}")
    print(f"  In 4mu 95% contour: {in_4mu_95}")
    print(f"  In 4mu+2pi 68% contour: {in_4mu2pi_68}")
    print(f"  In 4mu+2pi 95% contour: {in_4mu2pi_95}")

    # =========================================================================
    # Bootstrap p-value
    # =========================================================================
    N_BOOTSTRAP = 300
    print(f"\n" + "="*70)
    print(f"Bootstrap p-value ({N_BOOTSTRAP} replicates)")
    print("="*70)

    n_workers = max(1, cpu_count() - 1)
    print(f"Using {n_workers} workers...")

    args_list = [(seed, m_4mu, counts_4mu, bw_4mu, m_4mu2pi, counts_4mu2pi, bw_4mu2pi)
                 for seed in range(N_BOOTSTRAP)]

    with Pool(n_workers) as pool:
        Lambda_bootstrap = pool.map(bootstrap_replicate, args_list)

    Lambda_bootstrap = np.array([x for x in Lambda_bootstrap if not np.isnan(x)])
    n_valid = len(Lambda_bootstrap)

    # p-value: fraction of bootstrap Lambda >= observed Lambda
    p_value = np.mean(Lambda_bootstrap >= Lambda)

    print(f"\nValid replicates: {n_valid}/{N_BOOTSTRAP}")
    print(f"Bootstrap Lambda: mean={np.mean(Lambda_bootstrap):.2f}, std={np.std(Lambda_bootstrap):.2f}")
    print(f"Observed Lambda = {Lambda:.3f}")
    print(f"p-value = {p_value:.4f}")

    # =========================================================================
    # Verdict
    # =========================================================================
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if p_value > 0.05:
        verdict = "RANK-1 CONSTRAINT SUPPORTED"
    else:
        verdict = "RANK-1 CONSTRAINT REJECTED"

    print(f"\n{verdict}")
    print(f"Bootstrap p-value = {p_value:.3f} {'>' if p_value > 0.05 else '<='} 0.05")

    # =========================================================================
    # Chi-square goodness of fit (sanity check)
    # =========================================================================
    def compute_chi2(m, counts, params, bw):
        mu = model_intensity(m, params, apply_nuisance=True) * bw
        # Pearson chi-square (only bins with counts > 0)
        mask = counts > 0
        chi2 = np.sum((counts[mask] - mu[mask])**2 / mu[mask])
        dof = np.sum(mask) - 6  # 6 physics params + 3 nuisances - effectively 6
        return chi2, dof

    chi2_4mu, dof_4mu = compute_chi2(m_4mu, counts_4mu, params_4mu, bw_4mu)
    chi2_4mu2pi, dof_4mu2pi = compute_chi2(m_4mu2pi, counts_4mu2pi, params_4mu2pi, bw_4mu2pi)

    print(f"\nGoodness of fit (Pearson chi-square):")
    print(f"  4mu: chi2/dof = {chi2_4mu:.1f}/{dof_4mu} = {chi2_4mu/dof_4mu:.2f}")
    print(f"  4mu+2pi: chi2/dof = {chi2_4mu2pi:.1f}/{dof_4mu2pi} = {chi2_4mu2pi/dof_4mu2pi:.2f}")

    # =========================================================================
    # Save results
    # =========================================================================
    results = {
        "channel_4mu": {
            "r": float(params_4mu[1]),
            "phi_deg": float(np.degrees(params_4mu[2])),
            "nll": float(nll_4mu),
            "chi2_dof": float(chi2_4mu / dof_4mu),
            "nuisances": {
                "s_x": float(params_4mu[6]),
                "b_x_MeV": float(params_4mu[7] * 1000),
                "s_y": float(params_4mu[8])
            }
        },
        "channel_4mu2pi": {
            "r": float(params_4mu2pi[1]),
            "phi_deg": float(np.degrees(params_4mu2pi[2])),
            "nll": float(nll_4mu2pi),
            "chi2_dof": float(chi2_4mu2pi / dof_4mu2pi),
            "nuisances": {
                "s_x": float(params_4mu2pi[6]),
                "b_x_MeV": float(params_4mu2pi[7] * 1000),
                "s_y": float(params_4mu2pi[8])
            }
        },
        "shared": {
            "r": float(r_shared),
            "phi_deg": float(np.degrees(phi_shared))
        },
        "contour_check": {
            "shared_in_4mu_68": bool(in_4mu_68),
            "shared_in_4mu_95": bool(in_4mu_95),
            "shared_in_4mu2pi_68": bool(in_4mu2pi_68),
            "shared_in_4mu2pi_95": bool(in_4mu2pi_95)
        },
        "likelihood_ratio": {
            "Lambda": float(Lambda),
            "nll_unconstrained": float(nll_unconstrained),
            "nll_constrained": float(nll_constrained),
            "bootstrap_p_value": float(p_value),
            "n_bootstrap": n_valid
        },
        "verdict": verdict
    }

    with open("/home/primary/DarkBItParticleColiderPredictions/atlas_rank1_test_v4/out/ATLAS_v4_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # =========================================================================
    # Generate plots
    # =========================================================================

    # Fit plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (m, counts, params, bw, name) in zip(axes, [
        (m_4mu, counts_4mu, params_4mu, bw_4mu, "4μ"),
        (m_4mu2pi, counts_4mu2pi, params_4mu2pi, bw_4mu2pi, "4μ+2π")
    ]):
        m_fine = np.linspace(m.min(), m.max(), 200)
        mu_fine = model_intensity(m_fine, params, apply_nuisance=True) * bw

        ax.errorbar(m, counts, yerr=np.sqrt(np.maximum(1, counts)), fmt='ko',
                    capsize=2, markersize=4, label='ATLAS data')
        ax.plot(m_fine, mu_fine, 'r-', lw=2, label='Fit')
        ax.axvline(M_X6900, color='blue', linestyle='--', alpha=0.5, label=f'X(6900)')
        ax.axvline(M_X7200, color='green', linestyle='--', alpha=0.5, label=f'X(7200)')
        ax.set_xlabel(r'$m(J/\psi\psi(2S))$ [GeV]')
        ax.set_ylabel('Events / 50 MeV')
        ax.set_title(f'ATLAS {name} Channel\nr={params[1]:.3f}, φ={np.degrees(params[2]):.0f}°')
        ax.legend(loc='upper right')
        ax.set_xlim(6.8, 9.5)

    plt.tight_layout()
    plt.savefig("/home/primary/DarkBItParticleColiderPredictions/atlas_rank1_test_v4/out/fit_plots_v4.png", dpi=150)
    plt.close()

    # Contour plot
    fig, ax = plt.subplots(figsize=(8, 6))

    phi_grid_deg = np.degrees(phi_grid)

    # Plot contours
    ax.contour(phi_grid_deg, r_grid, contour_4mu_68, levels=[0.5], colors='blue',
               linestyles='-', linewidths=2)
    ax.contour(phi_grid_deg, r_grid, contour_4mu_95, levels=[0.5], colors='blue',
               linestyles='--', linewidths=1)
    ax.contour(phi_grid_deg, r_grid, contour_4mu2pi_68, levels=[0.5], colors='red',
               linestyles='-', linewidths=2)
    ax.contour(phi_grid_deg, r_grid, contour_4mu2pi_95, levels=[0.5], colors='red',
               linestyles='--', linewidths=1)

    # Plot best-fit points
    ax.plot(np.degrees(params_4mu[2]), params_4mu[1], 'bo', markersize=10, label='4μ best fit')
    ax.plot(np.degrees(params_4mu2pi[2]), params_4mu2pi[1], 'ro', markersize=10, label='4μ+2π best fit')
    ax.plot(np.degrees(phi_shared), r_shared, 'g*', markersize=15, label='Shared (constrained)')

    ax.set_xlabel(r'$\phi$ [degrees]')
    ax.set_ylabel(r'$r = |g_{7200}/g_{6900}|$')
    ax.set_title('Profile Likelihood Contours\n(solid=68%, dashed=95%)')
    ax.legend(loc='upper right')
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 1.4)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/primary/DarkBItParticleColiderPredictions/atlas_rank1_test_v4/out/contour_plot_v4.png", dpi=150)
    plt.close()

    # Bootstrap histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(Lambda_bootstrap, bins=30, density=True, alpha=0.7, color='steelblue',
            edgecolor='black', label='Bootstrap')
    ax.axvline(Lambda, color='red', linewidth=2, linestyle='--',
               label=f'Observed Λ = {Lambda:.2f}')
    ax.axvline(np.mean(Lambda_bootstrap), color='green', linewidth=2, linestyle=':',
               label=f'Bootstrap mean = {np.mean(Lambda_bootstrap):.2f}')
    ax.set_xlabel(r'$\Lambda = 2\Delta(\mathrm{NLL})$')
    ax.set_ylabel('Density')
    ax.set_title(f'Bootstrap Distribution of Λ\np-value = {p_value:.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig("/home/primary/DarkBItParticleColiderPredictions/atlas_rank1_test_v4/out/bootstrap_hist_v4.png", dpi=150)
    plt.close()

    print("\n" + "="*70)
    print("Output files saved to atlas_rank1_test_v4/out/")
    print("="*70)

    return results


if __name__ == "__main__":
    main()
