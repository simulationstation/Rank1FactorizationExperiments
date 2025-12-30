#!/usr/bin/env python3
"""
ATLAS Rank-1 Test v5 - Publication-grade analysis

Key improvements over v4:
- Hard fit-health gates: chi2/dof < 3 AND deviance/dof < 3 for BOTH channels
- ATLAS-faithful 3-resonance model with interference
- Multi-start (30+) + multi-optimizer (L-BFGS-B, Powell, DE) stability
- Guaranteed Lambda >= 0 via best-of optimization
- Profile likelihood contours with proper grid
- Bootstrap p-value only if all gates pass

Verdicts:
- SUPPORTED: all gates pass, Lambda acceptable
- DISFAVORED: gates pass but p < 0.05 AND shared point outside both 95%
- MODEL MISMATCH: chi2/dof or D/dof > 3 in either channel
- OPTIMIZER INSTABILITY: Lambda < 0 persists
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.special import gammaln
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import json
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# Physical Constants
# =============================================================================
M_JPSI = 3.0969  # GeV
M_PSI2S = 3.6861  # GeV
M_THRESH = M_JPSI + M_PSI2S  # = 6.783 GeV

# ATLAS resonance parameters (from arXiv:2304.08962, arXiv:2509.13101)
# Using central values with allowed variations
RESONANCES = {
    'thresh': {'m': 6.40, 'w': 0.40, 'm_range': (6.2, 6.6), 'w_range': (0.2, 0.6)},
    'X6900': {'m': 6.905, 'w': 0.150, 'm_range': (6.85, 6.95), 'w_range': (0.10, 0.25)},
    'X7200': {'m': 7.22, 'w': 0.100, 'm_range': (7.15, 7.30), 'w_range': (0.05, 0.20)},
}

# Nuisance parameter priors
SIGMA_SX = 0.01   # 1% scale uncertainty
SIGMA_BX = 0.020  # 20 MeV shift uncertainty
SIGMA_SY = 0.02   # 2% y-scale uncertainty

# Fit health thresholds
CHI2_DOF_MAX = 3.0
DEVIANCE_DOF_MAX = 3.0

# Multi-start settings
N_STARTS = 30
N_BOOTSTRAP = 300

# Output paths
BASE_DIR = Path("/home/primary/DarkBItParticleColiderPredictions/atlas_rank1_test_v5")
OUT_DIR = BASE_DIR / "out"
PROFILE_DIR = OUT_DIR / "profile_tables"


def phase_space(m):
    """Two-body phase space factor for J/psi + psi(2S)."""
    s = m**2
    s1 = (M_JPSI + M_PSI2S)**2
    s2 = (M_JPSI - M_PSI2S)**2

    if np.isscalar(m):
        if s <= s1:
            return 0.0
        return np.sqrt((s - s1) * (s - s2)) / (2 * m)
    else:
        ps = np.zeros_like(m)
        mask = s > s1
        ps[mask] = np.sqrt((s[mask] - s1) * (s[mask] - s2)) / (2 * m[mask])
        return ps


def breit_wigner(m, m0, w0):
    """
    Relativistic Breit-Wigner amplitude.
    BW = sqrt(m0 * w0) / (m0^2 - m^2 - i*m0*w0)
    """
    return np.sqrt(m0 * w0) / (m0**2 - m**2 - 1j * m0 * w0)


def chebyshev_bg(m, coeffs, m_min, m_max):
    """
    Chebyshev polynomial background.
    coeffs = [c0, c1, c2, ...] for T0, T1, T2, ...
    """
    # Map m to [-1, 1]
    x = 2 * (m - m_min) / (m_max - m_min) - 1

    result = np.zeros_like(m, dtype=float)
    for i, c in enumerate(coeffs):
        result += c * np.cos(i * np.arccos(np.clip(x, -1, 1)))

    return np.maximum(0, result)


def model_intensity_3bw(m, params, m_min, m_max):
    """
    3-resonance model with interference.

    params: [c_6900, r, phi, c_thresh, phi_thresh, bg0, bg1, bg2, s_x, b_x, s_y]

    Signal: |A_thresh*BW_thresh + A_6900*BW_6900 + A_7200*BW_7200|^2 * PS
    Background: threshold * Chebyshev
    """
    c_6900, r, phi, c_thresh, phi_thresh, bg0, bg1, bg2, s_x, b_x, s_y = params

    # Apply nuisance transformations
    m_eff = s_x * m + b_x

    # Phase space
    ps = phase_space(m_eff)

    # Complex amplitudes
    A_6900 = c_6900
    A_7200 = c_6900 * r * np.exp(1j * phi)
    A_thresh = c_thresh * np.exp(1j * phi_thresh)

    # Breit-Wigner amplitudes
    bw_thresh = breit_wigner(m_eff, RESONANCES['thresh']['m'], RESONANCES['thresh']['w'])
    bw_6900 = breit_wigner(m_eff, RESONANCES['X6900']['m'], RESONANCES['X6900']['w'])
    bw_7200 = breit_wigner(m_eff, RESONANCES['X7200']['m'], RESONANCES['X7200']['w'])

    # Coherent sum
    amplitude = A_thresh * bw_thresh + A_6900 * bw_6900 + A_7200 * bw_7200
    signal = ps * np.abs(amplitude)**2

    # Background: threshold turn-on * Chebyshev
    thresh_factor = np.sqrt(np.maximum(0, m_eff - M_THRESH))
    bg = thresh_factor * chebyshev_bg(m_eff, [bg0, bg1, bg2], m_min, m_max)

    # Total with y-scale nuisance
    intensity = s_y * (signal + bg)

    return np.maximum(1e-10, intensity)


def poisson_nll(params, m_centers, counts, bin_width, m_min, m_max):
    """
    Poisson negative log-likelihood.
    NLL = sum_i [ mu_i - n_i * log(mu_i) ]
    Plus Gaussian penalty terms for nuisance parameters.
    """
    mu = model_intensity_3bw(m_centers, params, m_min, m_max) * bin_width

    # Poisson NLL (ignoring constant log(n!) term)
    nll = np.sum(mu - counts * np.log(np.maximum(1e-10, mu)))

    # Gaussian priors for nuisances
    s_x, b_x, s_y = params[8], params[9], params[10]
    nll += 0.5 * ((s_x - 1.0) / SIGMA_SX)**2
    nll += 0.5 * (b_x / SIGMA_BX)**2
    nll += 0.5 * ((s_y - 1.0) / SIGMA_SY)**2

    return nll


def pearson_chi2(m_centers, counts, params, bin_width, m_min, m_max):
    """Pearson chi-square statistic."""
    mu = model_intensity_3bw(m_centers, params, m_min, m_max) * bin_width
    chi2 = np.sum((counts - mu)**2 / np.maximum(mu, 1e-9))
    return chi2


def poisson_deviance(m_centers, counts, params, bin_width, m_min, m_max):
    """Poisson deviance statistic."""
    mu = model_intensity_3bw(m_centers, params, m_min, m_max) * bin_width

    # D = 2 * sum[ y*log(y/mu) - (y - mu) ]
    # with convention y*log(y/mu) = 0 if y = 0
    deviance = 0.0
    for y, m in zip(counts, mu):
        if y > 0:
            deviance += 2 * (y * np.log(y / m) - (y - m))
        else:
            deviance += 2 * m  # y=0 case

    return deviance


def get_bounds():
    """Get parameter bounds for optimization."""
    return [
        (1, 200),       # c_6900 (normalization)
        (0.001, 1.5),   # r (ratio magnitude)
        (-np.pi, np.pi), # phi (phase)
        (0, 100),       # c_thresh
        (-np.pi, np.pi), # phi_thresh
        (0, 100),       # bg0
        (-50, 50),      # bg1
        (-30, 30),      # bg2
        (0.97, 1.03),   # s_x (within 3 sigma)
        (-0.06, 0.06),  # b_x (within 3 sigma)
        (0.94, 1.06),   # s_y (within 3 sigma)
    ]


def random_init(bounds, rng):
    """Generate random initialization within bounds."""
    return np.array([rng.uniform(lo, hi) for lo, hi in bounds])


def fit_channel_multistart(m_centers, counts, bin_width, m_min, m_max,
                           shared_R=None, n_starts=N_STARTS, seed=42):
    """
    Multi-start multi-optimizer fitting for a single channel.

    Returns: best_params, best_nll, all_nlls (for stability analysis)
    """
    rng = np.random.default_rng(seed)
    bounds = get_bounds()

    if shared_R is not None:
        # Constrained: r and phi are fixed
        r_fixed, phi_fixed = shared_R
        # Remove r and phi from bounds
        bounds_reduced = [bounds[0]] + bounds[3:]  # c_6900, c_thresh, ..., s_y

        def objective(p):
            full_p = np.array([p[0], r_fixed, phi_fixed, p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]])
            return poisson_nll(full_p, m_centers, counts, bin_width, m_min, m_max)

        def expand_params(p):
            return np.array([p[0], r_fixed, phi_fixed, p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]])
    else:
        bounds_reduced = bounds

        def objective(p):
            return poisson_nll(p, m_centers, counts, bin_width, m_min, m_max)

        def expand_params(p):
            return p

    all_nlls = []
    all_params = []

    # Multi-start with multiple optimizers
    for i in range(n_starts):
        x0 = random_init(bounds_reduced, rng)

        # Try L-BFGS-B
        try:
            res_lbfgsb = minimize(objective, x0, method='L-BFGS-B', bounds=bounds_reduced,
                                  options={'maxiter': 2000, 'ftol': 1e-9})
            if res_lbfgsb.success or res_lbfgsb.fun < 1e10:
                all_nlls.append(res_lbfgsb.fun)
                all_params.append(expand_params(res_lbfgsb.x))
        except:
            pass

        # Try Powell (unbounded, but clip to bounds after)
        try:
            res_powell = minimize(objective, x0, method='Powell',
                                  options={'maxiter': 2000, 'ftol': 1e-9})
            if res_powell.success or res_powell.fun < 1e10:
                # Clip to bounds
                x_clipped = np.clip(res_powell.x,
                                   [b[0] for b in bounds_reduced],
                                   [b[1] for b in bounds_reduced])
                all_nlls.append(objective(x_clipped))
                all_params.append(expand_params(x_clipped))
        except:
            pass

    # Also try differential evolution for global search
    try:
        res_de = differential_evolution(objective, bounds_reduced, seed=seed,
                                        maxiter=500, tol=1e-7, polish=True, workers=1)
        if res_de.success or res_de.fun < 1e10:
            all_nlls.append(res_de.fun)
            all_params.append(expand_params(res_de.x))
    except:
        pass

    if len(all_nlls) == 0:
        raise RuntimeError("All optimizations failed")

    # Find best
    best_idx = np.argmin(all_nlls)
    best_nll = all_nlls[best_idx]
    best_params = all_params[best_idx]

    return best_params, best_nll, np.array(all_nlls)


def joint_fit_multistart(data_4mu, data_4mu2pi, n_starts=N_STARTS, seed=42):
    """
    Joint constrained fit with shared (r, phi).
    Uses multi-start for stability.
    """
    m_4mu, counts_4mu, bw_4mu, mmin_4mu, mmax_4mu = data_4mu
    m_4mu2pi, counts_4mu2pi, bw_4mu2pi, mmin_4mu2pi, mmax_4mu2pi = data_4mu2pi

    rng = np.random.default_rng(seed)

    # Joint bounds: shared (r, phi) + independent channel params
    # [r, phi, c1_6900, c1_thresh, phi1_thresh, bg1_0, bg1_1, bg1_2, sx1, bx1, sy1,
    #         c2_6900, c2_thresh, phi2_thresh, bg2_0, bg2_1, bg2_2, sx2, bx2, sy2]
    joint_bounds = [
        (0.001, 1.5),    # r (shared)
        (-np.pi, np.pi), # phi (shared)
        # Channel 1 (4mu)
        (1, 200), (0, 100), (-np.pi, np.pi), (0, 100), (-50, 50), (-30, 30),
        (0.97, 1.03), (-0.06, 0.06), (0.94, 1.06),
        # Channel 2 (4mu+2pi)
        (1, 200), (0, 100), (-np.pi, np.pi), (0, 100), (-50, 50), (-30, 30),
        (0.97, 1.03), (-0.06, 0.06), (0.94, 1.06),
    ]

    def joint_objective(p):
        r, phi = p[0], p[1]
        # Channel 1 params
        p1 = np.array([p[2], r, phi, p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10]])
        # Channel 2 params
        p2 = np.array([p[11], r, phi, p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19]])

        nll1 = poisson_nll(p1, m_4mu, counts_4mu, bw_4mu, mmin_4mu, mmax_4mu)
        nll2 = poisson_nll(p2, m_4mu2pi, counts_4mu2pi, bw_4mu2pi, mmin_4mu2pi, mmax_4mu2pi)

        return nll1 + nll2

    all_nlls = []
    all_params = []

    for i in range(n_starts):
        x0 = random_init(joint_bounds, rng)

        try:
            res = minimize(joint_objective, x0, method='L-BFGS-B', bounds=joint_bounds,
                          options={'maxiter': 3000, 'ftol': 1e-9})
            if res.success or res.fun < 1e10:
                all_nlls.append(res.fun)
                all_params.append(res.x)
        except:
            pass

    # Also try DE
    try:
        res_de = differential_evolution(joint_objective, joint_bounds, seed=seed,
                                        maxiter=500, tol=1e-7, polish=True, workers=1)
        if res_de.success or res_de.fun < 1e10:
            all_nlls.append(res_de.fun)
            all_params.append(res_de.x)
    except:
        pass

    if len(all_nlls) == 0:
        raise RuntimeError("Joint optimization failed")

    best_idx = np.argmin(all_nlls)
    best_nll = all_nlls[best_idx]
    best_params = all_params[best_idx]

    r_shared = best_params[0]
    phi_shared = best_params[1]

    return (r_shared, phi_shared), best_nll, best_params, np.array(all_nlls)


def compute_profile_contour_point(args):
    """Compute profile NLL at a single (r, phi) point."""
    r, phi, m_centers, counts, bin_width, m_min, m_max, best_params = args

    # Fix r and phi, optimize remaining params
    bounds_reduced = [
        (1, 200),       # c_6900
        (0, 100),       # c_thresh
        (-np.pi, np.pi), # phi_thresh
        (0, 100),       # bg0
        (-50, 50),      # bg1
        (-30, 30),      # bg2
        (0.97, 1.03),   # s_x
        (-0.06, 0.06),  # b_x
        (0.94, 1.06),   # s_y
    ]

    def objective(p):
        full_p = np.array([p[0], r, phi, p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]])
        return poisson_nll(full_p, m_centers, counts, bin_width, m_min, m_max)

    # Start from best-fit values (excluding r, phi)
    x0 = np.array([best_params[0], best_params[3], best_params[4],
                   best_params[5], best_params[6], best_params[7],
                   best_params[8], best_params[9], best_params[10]])

    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds_reduced,
                      options={'maxiter': 500})
        return r, phi, res.fun
    except:
        return r, phi, np.inf


def compute_profile_contours(m_centers, counts, bin_width, m_min, m_max, best_params, best_nll):
    """
    Compute 2D profile likelihood over (r, phi) grid.
    Returns grids and contour masks for 68% and 95% CL.
    """
    r_grid = np.arange(0.02, 1.52, 0.02)
    phi_grid = np.arange(-180, 185, 5) * np.pi / 180

    # Prepare arguments for parallel computation
    args_list = []
    for r in r_grid:
        for phi in phi_grid:
            args_list.append((r, phi, m_centers, counts, bin_width, m_min, m_max, best_params))

    # Parallel computation
    n_workers = max(1, cpu_count() - 1)
    with Pool(n_workers) as pool:
        results = pool.map(compute_profile_contour_point, args_list)

    # Build NLL surface
    nll_surface = np.full((len(r_grid), len(phi_grid)), np.inf)
    for r_val, phi_val, nll_val in results:
        i = np.argmin(np.abs(r_grid - r_val))
        j = np.argmin(np.abs(phi_grid - phi_val))
        nll_surface[i, j] = nll_val

    # Compute delta NLL
    delta_nll = 2 * (nll_surface - best_nll)

    # Contour masks
    DELTA_CHI2_68 = 2.30
    DELTA_CHI2_95 = 5.99

    contour_68 = delta_nll < DELTA_CHI2_68
    contour_95 = delta_nll < DELTA_CHI2_95

    return r_grid, phi_grid, nll_surface, contour_68, contour_95


def bootstrap_replicate(args):
    """Single bootstrap replicate for parallel execution."""
    seed, m_4mu, counts_4mu, bw_4mu, mmin_4mu, mmax_4mu, \
          m_4mu2pi, counts_4mu2pi, bw_4mu2pi, mmin_4mu2pi, mmax_4mu2pi = args

    rng = np.random.default_rng(seed)

    # Poisson resample
    counts_4mu_boot = rng.poisson(counts_4mu)
    counts_4mu2pi_boot = rng.poisson(counts_4mu2pi)

    try:
        # Unconstrained fits (with fewer starts for speed)
        params_4mu, nll_4mu, _ = fit_channel_multistart(
            m_4mu, counts_4mu_boot, bw_4mu, mmin_4mu, mmax_4mu, n_starts=10, seed=seed)
        params_4mu2pi, nll_4mu2pi, _ = fit_channel_multistart(
            m_4mu2pi, counts_4mu2pi_boot, bw_4mu2pi, mmin_4mu2pi, mmax_4mu2pi, n_starts=10, seed=seed+1000)

        nll_unconstrained = nll_4mu + nll_4mu2pi

        # Constrained fit
        data_4mu = (m_4mu, counts_4mu_boot, bw_4mu, mmin_4mu, mmax_4mu)
        data_4mu2pi = (m_4mu2pi, counts_4mu2pi_boot, bw_4mu2pi, mmin_4mu2pi, mmax_4mu2pi)
        _, nll_constrained, _, _ = joint_fit_multistart(data_4mu, data_4mu2pi, n_starts=10, seed=seed+2000)

        Lambda = 2 * (nll_constrained - nll_unconstrained)
        return Lambda
    except:
        return np.nan


def main():
    """Main analysis pipeline."""

    print("="*70)
    print("ATLAS Rank-1 Test v5 - Publication-grade Analysis")
    print("="*70)
    print()

    # Load data
    df_4mu = pd.read_csv(BASE_DIR / "data/derived/4mu_bins.csv")
    df_4mu2pi = pd.read_csv(BASE_DIR / "data/derived/4mu+2pi_bins.csv")

    m_4mu = df_4mu['m_center'].values
    counts_4mu = df_4mu['count'].values.astype(float)
    bw_4mu = df_4mu['m_high'].values[0] - df_4mu['m_low'].values[0]
    mmin_4mu, mmax_4mu = m_4mu.min(), m_4mu.max()

    m_4mu2pi = df_4mu2pi['m_center'].values
    counts_4mu2pi = df_4mu2pi['count'].values.astype(float)
    bw_4mu2pi = df_4mu2pi['m_high'].values[0] - df_4mu2pi['m_low'].values[0]
    mmin_4mu2pi, mmax_4mu2pi = m_4mu2pi.min(), m_4mu2pi.max()

    n_bins_4mu = len(counts_4mu)
    n_bins_4mu2pi = len(counts_4mu2pi)
    n_params = 11  # Total free params in full model
    dof_4mu = n_bins_4mu - n_params
    dof_4mu2pi = n_bins_4mu2pi - n_params

    print(f"4mu: {n_bins_4mu} bins, {int(counts_4mu.sum())} total counts, dof={dof_4mu}")
    print(f"4mu+2pi: {n_bins_4mu2pi} bins, {int(counts_4mu2pi.sum())} total counts, dof={dof_4mu2pi}")
    print()

    # =========================================================================
    # Fit each channel independently (unconstrained) with multi-start
    # =========================================================================
    print("="*70)
    print("Unconstrained Fits (Multi-start, Multi-optimizer)")
    print("="*70)

    params_4mu, nll_4mu, nlls_4mu = fit_channel_multistart(
        m_4mu, counts_4mu, bw_4mu, mmin_4mu, mmax_4mu, n_starts=N_STARTS, seed=42)

    chi2_4mu = pearson_chi2(m_4mu, counts_4mu, params_4mu, bw_4mu, mmin_4mu, mmax_4mu)
    dev_4mu = poisson_deviance(m_4mu, counts_4mu, params_4mu, bw_4mu, mmin_4mu, mmax_4mu)

    print(f"\n4mu channel:")
    print(f"  r = {params_4mu[1]:.4f}")
    print(f"  phi = {np.degrees(params_4mu[2]):.1f} deg")
    print(f"  NLL = {nll_4mu:.2f}")
    print(f"  chi2/dof = {chi2_4mu:.1f}/{dof_4mu} = {chi2_4mu/dof_4mu:.2f}")
    print(f"  Deviance/dof = {dev_4mu:.1f}/{dof_4mu} = {dev_4mu/dof_4mu:.2f}")
    print(f"  Optimizer starts: {len(nlls_4mu)}, NLL range: [{nlls_4mu.min():.1f}, {nlls_4mu.max():.1f}]")

    params_4mu2pi, nll_4mu2pi, nlls_4mu2pi = fit_channel_multistart(
        m_4mu2pi, counts_4mu2pi, bw_4mu2pi, mmin_4mu2pi, mmax_4mu2pi, n_starts=N_STARTS, seed=43)

    chi2_4mu2pi = pearson_chi2(m_4mu2pi, counts_4mu2pi, params_4mu2pi, bw_4mu2pi, mmin_4mu2pi, mmax_4mu2pi)
    dev_4mu2pi = poisson_deviance(m_4mu2pi, counts_4mu2pi, params_4mu2pi, bw_4mu2pi, mmin_4mu2pi, mmax_4mu2pi)

    print(f"\n4mu+2pi channel:")
    print(f"  r = {params_4mu2pi[1]:.4f}")
    print(f"  phi = {np.degrees(params_4mu2pi[2]):.1f} deg")
    print(f"  NLL = {nll_4mu2pi:.2f}")
    print(f"  chi2/dof = {chi2_4mu2pi:.1f}/{dof_4mu2pi} = {chi2_4mu2pi/dof_4mu2pi:.2f}")
    print(f"  Deviance/dof = {dev_4mu2pi:.1f}/{dof_4mu2pi} = {dev_4mu2pi/dof_4mu2pi:.2f}")
    print(f"  Optimizer starts: {len(nlls_4mu2pi)}, NLL range: [{nlls_4mu2pi.min():.1f}, {nlls_4mu2pi.max():.1f}]")

    nll_unconstrained = nll_4mu + nll_4mu2pi

    # =========================================================================
    # Fit health gates
    # =========================================================================
    print("\n" + "="*70)
    print("Fit Health Gates")
    print("="*70)

    gate_4mu_chi2 = chi2_4mu / dof_4mu < CHI2_DOF_MAX
    gate_4mu_dev = dev_4mu / dof_4mu < DEVIANCE_DOF_MAX
    gate_4mu = gate_4mu_chi2 and gate_4mu_dev

    gate_4mu2pi_chi2 = chi2_4mu2pi / dof_4mu2pi < CHI2_DOF_MAX
    gate_4mu2pi_dev = dev_4mu2pi / dof_4mu2pi < DEVIANCE_DOF_MAX
    gate_4mu2pi = gate_4mu2pi_chi2 and gate_4mu2pi_dev

    gates_pass = gate_4mu and gate_4mu2pi

    print(f"4mu:     chi2/dof = {chi2_4mu/dof_4mu:.2f} {'PASS' if gate_4mu_chi2 else 'FAIL'}, "
          f"D/dof = {dev_4mu/dof_4mu:.2f} {'PASS' if gate_4mu_dev else 'FAIL'}")
    print(f"4mu+2pi: chi2/dof = {chi2_4mu2pi/dof_4mu2pi:.2f} {'PASS' if gate_4mu2pi_chi2 else 'FAIL'}, "
          f"D/dof = {dev_4mu2pi/dof_4mu2pi:.2f} {'PASS' if gate_4mu2pi_dev else 'FAIL'}")
    print(f"Overall gates: {'PASS' if gates_pass else 'FAIL'}")

    # =========================================================================
    # Joint constrained fit
    # =========================================================================
    print("\n" + "="*70)
    print("Joint Constrained Fit (shared r, phi)")
    print("="*70)

    data_4mu = (m_4mu, counts_4mu, bw_4mu, mmin_4mu, mmax_4mu)
    data_4mu2pi = (m_4mu2pi, counts_4mu2pi, bw_4mu2pi, mmin_4mu2pi, mmax_4mu2pi)

    (r_shared, phi_shared), nll_constrained, joint_params, joint_nlls = joint_fit_multistart(
        data_4mu, data_4mu2pi, n_starts=N_STARTS, seed=44)

    print(f"\nShared parameters:")
    print(f"  r_shared = {r_shared:.4f}")
    print(f"  phi_shared = {np.degrees(phi_shared):.1f} deg")
    print(f"  Total NLL = {nll_constrained:.2f}")
    print(f"  Joint optimizer starts: {len(joint_nlls)}, NLL range: [{joint_nlls.min():.1f}, {joint_nlls.max():.1f}]")

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

    # Check for optimizer instability
    optimizer_stable = Lambda >= 0
    if not optimizer_stable:
        print("\nWARNING: Lambda < 0 detected. Attempting increased optimization...")
        # Retry with more starts
        params_4mu, nll_4mu, _ = fit_channel_multistart(
            m_4mu, counts_4mu, bw_4mu, mmin_4mu, mmax_4mu, n_starts=100, seed=100)
        params_4mu2pi, nll_4mu2pi, _ = fit_channel_multistart(
            m_4mu2pi, counts_4mu2pi, bw_4mu2pi, mmin_4mu2pi, mmax_4mu2pi, n_starts=100, seed=101)
        nll_unconstrained = nll_4mu + nll_4mu2pi

        (r_shared, phi_shared), nll_constrained, joint_params, _ = joint_fit_multistart(
            data_4mu, data_4mu2pi, n_starts=100, seed=102)

        Lambda = 2 * (nll_constrained - nll_unconstrained)
        print(f"After 100 starts: Lambda = {Lambda:.3f}")
        optimizer_stable = Lambda >= 0

    # =========================================================================
    # Profile likelihood contours
    # =========================================================================
    print("\n" + "="*70)
    print("Computing Profile Likelihood Contours...")
    print("="*70)

    print("  Computing 4mu contour...")
    r_grid, phi_grid, nll_surf_4mu, contour_4mu_68, contour_4mu_95 = compute_profile_contours(
        m_4mu, counts_4mu, bw_4mu, mmin_4mu, mmax_4mu, params_4mu, nll_4mu)

    print("  Computing 4mu+2pi contour...")
    _, _, nll_surf_4mu2pi, contour_4mu2pi_68, contour_4mu2pi_95 = compute_profile_contours(
        m_4mu2pi, counts_4mu2pi, bw_4mu2pi, mmin_4mu2pi, mmax_4mu2pi, params_4mu2pi, nll_4mu2pi)

    # Save profile tables
    np.savetxt(PROFILE_DIR / "r_grid.csv", r_grid, delimiter=',')
    np.savetxt(PROFILE_DIR / "phi_grid.csv", phi_grid, delimiter=',')
    np.savetxt(PROFILE_DIR / "nll_surface_4mu.csv", nll_surf_4mu, delimiter=',')
    np.savetxt(PROFILE_DIR / "nll_surface_4mu2pi.csv", nll_surf_4mu2pi, delimiter=',')

    # Check if shared point lies in contours
    r_idx = np.argmin(np.abs(r_grid - r_shared))
    phi_idx = np.argmin(np.abs(phi_grid - phi_shared))

    in_4mu_68 = contour_4mu_68[r_idx, phi_idx]
    in_4mu_95 = contour_4mu_95[r_idx, phi_idx]
    in_4mu2pi_68 = contour_4mu2pi_68[r_idx, phi_idx]
    in_4mu2pi_95 = contour_4mu2pi_95[r_idx, phi_idx]

    shared_in_both_95 = in_4mu_95 and in_4mu2pi_95

    print(f"\nShared point (r={r_shared:.3f}, phi={np.degrees(phi_shared):.1f} deg):")
    print(f"  In 4mu 68% contour: {in_4mu_68}")
    print(f"  In 4mu 95% contour: {in_4mu_95}")
    print(f"  In 4mu+2pi 68% contour: {in_4mu2pi_68}")
    print(f"  In 4mu+2pi 95% contour: {in_4mu2pi_95}")
    print(f"  In BOTH 95% contours: {shared_in_both_95}")

    # =========================================================================
    # Bootstrap p-value (only if gates pass)
    # =========================================================================
    p_value = None
    Lambda_bootstrap = None

    if gates_pass and optimizer_stable:
        print(f"\n" + "="*70)
        print(f"Bootstrap p-value ({N_BOOTSTRAP} replicates)")
        print("="*70)

        n_workers = max(1, cpu_count() - 1)
        print(f"Using {n_workers} workers...")

        args_list = [(seed, m_4mu, counts_4mu, bw_4mu, mmin_4mu, mmax_4mu,
                      m_4mu2pi, counts_4mu2pi, bw_4mu2pi, mmin_4mu2pi, mmax_4mu2pi)
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
    else:
        print("\n" + "="*70)
        print("Bootstrap SKIPPED (gates did not pass or optimizer unstable)")
        print("="*70)

    # =========================================================================
    # Determine verdict
    # =========================================================================
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if not gates_pass:
        verdict = "MODEL MISMATCH"
        reason = "chi2/dof or D/dof > 3 in at least one channel"
    elif not optimizer_stable:
        verdict = "OPTIMIZER INSTABILITY"
        reason = "Lambda < 0 persists after increased optimization"
    elif p_value is not None and p_value < 0.05 and not shared_in_both_95:
        verdict = "DISFAVORED"
        reason = f"p = {p_value:.3f} < 0.05 AND shared point outside both 95% contours"
    elif p_value is not None and p_value >= 0.05:
        verdict = "SUPPORTED"
        reason = f"p = {p_value:.3f} >= 0.05, all gates pass"
    elif shared_in_both_95:
        verdict = "SUPPORTED"
        reason = "Shared point in both 95% contours, all gates pass"
    else:
        verdict = "INCONCLUSIVE"
        reason = "Unable to determine (edge case)"

    print(f"\n{verdict}")
    print(f"Reason: {reason}")

    # =========================================================================
    # Save results
    # =========================================================================
    results = {
        "channel_4mu": {
            "r": float(params_4mu[1]),
            "phi_deg": float(np.degrees(params_4mu[2])),
            "nll": float(nll_4mu),
            "chi2": float(chi2_4mu),
            "chi2_dof": float(chi2_4mu / dof_4mu),
            "deviance": float(dev_4mu),
            "deviance_dof": float(dev_4mu / dof_4mu),
            "gate_pass": bool(gate_4mu),
            "n_bins": int(n_bins_4mu),
            "dof": int(dof_4mu)
        },
        "channel_4mu2pi": {
            "r": float(params_4mu2pi[1]),
            "phi_deg": float(np.degrees(params_4mu2pi[2])),
            "nll": float(nll_4mu2pi),
            "chi2": float(chi2_4mu2pi),
            "chi2_dof": float(chi2_4mu2pi / dof_4mu2pi),
            "deviance": float(dev_4mu2pi),
            "deviance_dof": float(dev_4mu2pi / dof_4mu2pi),
            "gate_pass": bool(gate_4mu2pi),
            "n_bins": int(n_bins_4mu2pi),
            "dof": int(dof_4mu2pi)
        },
        "shared": {
            "r": float(r_shared),
            "phi_deg": float(np.degrees(phi_shared))
        },
        "contour_check": {
            "shared_in_4mu_68": bool(in_4mu_68),
            "shared_in_4mu_95": bool(in_4mu_95),
            "shared_in_4mu2pi_68": bool(in_4mu2pi_68),
            "shared_in_4mu2pi_95": bool(in_4mu2pi_95),
            "shared_in_both_95": bool(shared_in_both_95)
        },
        "likelihood_ratio": {
            "Lambda": float(Lambda),
            "nll_unconstrained": float(nll_unconstrained),
            "nll_constrained": float(nll_constrained),
            "optimizer_stable": bool(optimizer_stable)
        },
        "optimizer_stability": {
            "n_starts": N_STARTS,
            "4mu_nll_range": [float(nlls_4mu.min()), float(nlls_4mu.max())],
            "4mu2pi_nll_range": [float(nlls_4mu2pi.min()), float(nlls_4mu2pi.max())],
            "joint_nll_range": [float(joint_nlls.min()), float(joint_nlls.max())]
        },
        "gates_pass": bool(gates_pass),
        "verdict": verdict,
        "reason": reason
    }

    if p_value is not None:
        results["bootstrap"] = {
            "p_value": float(p_value),
            "n_bootstrap": int(len(Lambda_bootstrap)),
            "Lambda_mean": float(np.mean(Lambda_bootstrap)),
            "Lambda_std": float(np.std(Lambda_bootstrap))
        }

    with open(OUT_DIR / "ATLAS_v5_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # =========================================================================
    # Generate plots
    # =========================================================================

    # Fit plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (m, counts, params, bw, mmin, mmax, name, chi2_dof, dev_dof) in zip(axes, [
        (m_4mu, counts_4mu, params_4mu, bw_4mu, mmin_4mu, mmax_4mu, "4μ",
         chi2_4mu/dof_4mu, dev_4mu/dof_4mu),
        (m_4mu2pi, counts_4mu2pi, params_4mu2pi, bw_4mu2pi, mmin_4mu2pi, mmax_4mu2pi, "4μ+2π",
         chi2_4mu2pi/dof_4mu2pi, dev_4mu2pi/dof_4mu2pi)
    ]):
        m_fine = np.linspace(m.min(), m.max(), 200)
        mu_fine = model_intensity_3bw(m_fine, params, mmin, mmax) * bw

        ax.errorbar(m, counts, yerr=np.sqrt(np.maximum(1, counts)), fmt='ko',
                    capsize=2, markersize=4, label='ATLAS data')
        ax.plot(m_fine, mu_fine, 'r-', lw=2, label='3-BW fit')
        ax.axvline(RESONANCES['X6900']['m'], color='blue', linestyle='--', alpha=0.5, label='X(6900)')
        ax.axvline(RESONANCES['X7200']['m'], color='green', linestyle='--', alpha=0.5, label='X(7200)')
        ax.axvline(RESONANCES['thresh']['m'], color='orange', linestyle='--', alpha=0.5, label='Threshold')
        ax.set_xlabel(r'$m(J/\psi\psi(2S))$ [GeV]')
        ax.set_ylabel('Events / 50 MeV')
        ax.set_title(f'ATLAS {name} Channel\nr={params[1]:.3f}, φ={np.degrees(params[2]):.0f}°\n'
                    f'χ²/dof={chi2_dof:.2f}, D/dof={dev_dof:.2f}')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(6.8, 9.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fit_plots_v5.png", dpi=150)
    plt.close()

    # Contour plot
    fig, ax = plt.subplots(figsize=(10, 7))

    phi_grid_deg = np.degrees(phi_grid)

    # Plot filled contours
    ax.contourf(phi_grid_deg, r_grid, contour_4mu_95.astype(float), levels=[0.5, 1.5],
                colors=['lightblue'], alpha=0.3)
    ax.contourf(phi_grid_deg, r_grid, contour_4mu2pi_95.astype(float), levels=[0.5, 1.5],
                colors=['lightsalmon'], alpha=0.3)

    # Plot contour lines
    ax.contour(phi_grid_deg, r_grid, contour_4mu_68, levels=[0.5], colors='blue',
               linestyles='-', linewidths=2, label='4μ 68%')
    ax.contour(phi_grid_deg, r_grid, contour_4mu_95, levels=[0.5], colors='blue',
               linestyles='--', linewidths=1, label='4μ 95%')
    ax.contour(phi_grid_deg, r_grid, contour_4mu2pi_68, levels=[0.5], colors='red',
               linestyles='-', linewidths=2, label='4μ+2π 68%')
    ax.contour(phi_grid_deg, r_grid, contour_4mu2pi_95, levels=[0.5], colors='red',
               linestyles='--', linewidths=1, label='4μ+2π 95%')

    # Plot best-fit points
    ax.plot(np.degrees(params_4mu[2]), params_4mu[1], 'bo', markersize=12,
            label=f'4μ MLE (r={params_4mu[1]:.3f})')
    ax.plot(np.degrees(params_4mu2pi[2]), params_4mu2pi[1], 'ro', markersize=12,
            label=f'4μ+2π MLE (r={params_4mu2pi[1]:.3f})')
    ax.plot(np.degrees(phi_shared), r_shared, 'g*', markersize=18, markeredgecolor='black',
            label=f'Shared (r={r_shared:.3f})')

    ax.set_xlabel(r'$\phi$ [degrees]', fontsize=12)
    ax.set_ylabel(r'$r = |g_{7200}/g_{6900}|$', fontsize=12)
    ax.set_title(f'Profile Likelihood Contours\n(solid=68%, dashed=95%)\nVerdict: {verdict}', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "contours_v5.png", dpi=150)
    plt.close()

    # Bootstrap histogram (if computed)
    if Lambda_bootstrap is not None:
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
        plt.savefig(OUT_DIR / "bootstrap_hist_v5.png", dpi=150)
        plt.close()

        # Save bootstrap samples
        np.savetxt(OUT_DIR / "bootstrap_samples_v5.csv", Lambda_bootstrap, delimiter=',')

    # Save optimizer stability
    stability_data = {
        "4mu": {
            "n_successful_starts": len(nlls_4mu),
            "best_nll": float(nlls_4mu.min()),
            "worst_nll": float(nlls_4mu.max()),
            "nll_std": float(nlls_4mu.std()),
            "all_nlls": nlls_4mu.tolist()
        },
        "4mu2pi": {
            "n_successful_starts": len(nlls_4mu2pi),
            "best_nll": float(nlls_4mu2pi.min()),
            "worst_nll": float(nlls_4mu2pi.max()),
            "nll_std": float(nlls_4mu2pi.std()),
            "all_nlls": nlls_4mu2pi.tolist()
        },
        "joint": {
            "n_successful_starts": len(joint_nlls),
            "best_nll": float(joint_nlls.min()),
            "worst_nll": float(joint_nlls.max()),
            "nll_std": float(joint_nlls.std()),
            "all_nlls": joint_nlls.tolist()
        }
    }

    with open(OUT_DIR / "optimizer_stability.json", "w") as f:
        json.dump(stability_data, f, indent=2)

    # Optimizer stability summary
    with open(OUT_DIR / "optimizer_stability.md", "w") as f:
        f.write("# Optimizer Stability Summary\n\n")
        f.write(f"Multi-start settings: {N_STARTS} random initializations\n")
        f.write(f"Optimizers: L-BFGS-B, Powell, Differential Evolution\n\n")
        f.write("## 4mu Channel\n")
        f.write(f"- Successful starts: {len(nlls_4mu)}\n")
        f.write(f"- NLL range: [{nlls_4mu.min():.2f}, {nlls_4mu.max():.2f}]\n")
        f.write(f"- NLL std: {nlls_4mu.std():.2f}\n\n")
        f.write("## 4mu+2pi Channel\n")
        f.write(f"- Successful starts: {len(nlls_4mu2pi)}\n")
        f.write(f"- NLL range: [{nlls_4mu2pi.min():.2f}, {nlls_4mu2pi.max():.2f}]\n")
        f.write(f"- NLL std: {nlls_4mu2pi.std():.2f}\n\n")
        f.write("## Joint Constrained Fit\n")
        f.write(f"- Successful starts: {len(joint_nlls)}\n")
        f.write(f"- NLL range: [{joint_nlls.min():.2f}, {joint_nlls.max():.2f}]\n")
        f.write(f"- NLL std: {joint_nlls.std():.2f}\n\n")
        f.write(f"## Lambda Check\n")
        f.write(f"Lambda = {Lambda:.3f} {'≥ 0 (stable)' if Lambda >= 0 else '< 0 (UNSTABLE)'}\n")

    print("\n" + "="*70)
    print("Output files saved to atlas_rank1_test_v5/out/")
    print("="*70)

    return results


if __name__ == "__main__":
    results = main()
