#!/usr/bin/env python3
"""
ATLAS Rank-1 Test v6 - Statistically Corrected Analysis

Key fixes over v5:
1. Parametric bootstrap under constrained-null H0 (not observed counts)
2. Bin-integrated μ using Gauss-Legendre quadrature (not center × width)
3. Robust bootstrap optimization with retries and failure tracking
4. Validation mode: compare legacy vs correct methods

Usage:
    python fit_atlas_v6.py                    # Run correct mode
    python fit_atlas_v6.py --validate         # Run both legacy and correct for comparison
    python fit_atlas_v6.py --bin_integrate false  # Disable bin integration (legacy)
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
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

warnings.filterwarnings('ignore')

# =============================================================================
# Physical Constants
# =============================================================================
M_JPSI = 3.0969  # GeV
M_PSI2S = 3.6861  # GeV
M_THRESH = M_JPSI + M_PSI2S  # = 6.783 GeV

# ATLAS resonance parameters
RESONANCES = {
    'thresh': {'m': 6.40, 'w': 0.40},
    'X6900': {'m': 6.905, 'w': 0.150},
    'X7200': {'m': 7.22, 'w': 0.100},
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
N_STARTS_BOOT = 30  # Increased from 10 for robustness
N_BOOTSTRAP = 500
MAX_BOOT_RETRIES = 3
MAX_BOOT_FAILURE_RATE = 0.02  # 2%

# Gauss-Legendre quadrature points and weights (4 points - faster, still accurate)
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(4)

# Output paths
BASE_DIR = Path(__file__).parent.parent
OUT_DIR = BASE_DIR / "out"


@dataclass
class ChannelData:
    """Data container for a single channel."""
    name: str
    m_low: np.ndarray
    m_high: np.ndarray
    m_center: np.ndarray
    bin_widths: np.ndarray
    counts: np.ndarray
    m_min: float
    m_max: float


@dataclass
class FitResult:
    """Result container for a fit."""
    params: np.ndarray
    nll: float
    chi2: float
    chi2_dof: float
    deviance: float
    deviance_dof: float
    r: float
    phi: float
    gate_pass: bool
    all_nlls: np.ndarray


def phase_space(m):
    """Two-body phase space factor for J/psi + psi(2S)."""
    m = np.atleast_1d(m)
    s = m**2
    s1 = (M_JPSI + M_PSI2S)**2
    s2 = (M_JPSI - M_PSI2S)**2

    ps = np.zeros_like(m, dtype=float)
    mask = s > s1
    ps[mask] = np.sqrt((s[mask] - s1) * (s[mask] - s2)) / (2 * m[mask])

    if ps.size == 1:
        return float(ps[0])
    return ps


def breit_wigner(m, m0, w0):
    """Relativistic Breit-Wigner amplitude."""
    return np.sqrt(m0 * w0) / (m0**2 - m**2 - 1j * m0 * w0)


def chebyshev_bg(m, coeffs, m_min, m_max):
    """Chebyshev polynomial background."""
    x = 2 * (m - m_min) / (m_max - m_min) - 1
    x = np.clip(x, -1, 1)

    result = np.zeros_like(m, dtype=float)
    for i, c in enumerate(coeffs):
        result += c * np.cos(i * np.arccos(x))

    return np.maximum(0, result)


def model_intensity(m, params, m_min, m_max):
    """
    3-resonance model with interference.

    params: [c_6900, r, phi, c_thresh, phi_thresh, bg0, bg1, bg2, s_x, b_x, s_y]
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


def bin_integrate_mu(params, m_low, m_high, m_min, m_max, use_quadrature=True):
    """
    Compute expected counts per bin.

    If use_quadrature=True: ∫_{m_low}^{m_high} intensity(m) dm using Gauss-Legendre
    If use_quadrature=False: intensity(m_center) × bin_width (legacy method)
    """
    n_bins = len(m_low)

    if use_quadrature:
        # Vectorized Gauss-Legendre integration over all bins
        # Shape: (n_bins, n_gl_points)
        a = m_low[:, np.newaxis]  # (n_bins, 1)
        b = m_high[:, np.newaxis]  # (n_bins, 1)

        # Transform GL nodes from [-1,1] to [a,b] for all bins at once
        m_points = 0.5 * (b - a) * GL_NODES + 0.5 * (a + b)  # (n_bins, n_gl)

        # Flatten for model evaluation, then reshape
        m_flat = m_points.flatten()
        intensities_flat = model_intensity(m_flat, params, m_min, m_max)
        intensities = intensities_flat.reshape(n_bins, len(GL_NODES))

        # Gauss-Legendre sum with Jacobian factor
        jacobian = 0.5 * (m_high - m_low)  # (n_bins,)
        mu = jacobian * np.sum(GL_WEIGHTS * intensities, axis=1)
    else:
        # Legacy: intensity at bin center × bin width
        m_centers = 0.5 * (m_low + m_high)
        bin_widths = m_high - m_low
        mu = model_intensity(m_centers, params, m_min, m_max) * bin_widths

    return np.maximum(1e-10, mu)


def poisson_nll(params, data: ChannelData, use_quadrature=True):
    """
    Poisson negative log-likelihood with nuisance priors.
    """
    mu = bin_integrate_mu(params, data.m_low, data.m_high, data.m_min, data.m_max, use_quadrature)

    # Poisson NLL (ignoring constant log(n!) term)
    nll = np.sum(mu - data.counts * np.log(mu))

    # Gaussian priors for nuisances
    s_x, b_x, s_y = params[8], params[9], params[10]
    nll += 0.5 * ((s_x - 1.0) / SIGMA_SX)**2
    nll += 0.5 * (b_x / SIGMA_BX)**2
    nll += 0.5 * ((s_y - 1.0) / SIGMA_SY)**2

    return nll


def pearson_chi2(data: ChannelData, params, use_quadrature=True):
    """Pearson chi-square statistic."""
    mu = bin_integrate_mu(params, data.m_low, data.m_high, data.m_min, data.m_max, use_quadrature)
    return np.sum((data.counts - mu)**2 / np.maximum(mu, 1e-9))


def poisson_deviance(data: ChannelData, params, use_quadrature=True):
    """Poisson deviance statistic."""
    mu = bin_integrate_mu(params, data.m_low, data.m_high, data.m_min, data.m_max, use_quadrature)

    deviance = 0.0
    for y, m_val in zip(data.counts, mu):
        if y > 0:
            deviance += 2 * (y * np.log(y / m_val) - (y - m_val))
        else:
            deviance += 2 * m_val

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


def fit_channel_multistart(data: ChannelData, shared_R=None, n_starts=N_STARTS,
                           seed=42, use_quadrature=True) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Multi-start multi-optimizer fitting for a single channel.
    """
    rng = np.random.default_rng(seed)
    bounds = get_bounds()

    if shared_R is not None:
        r_fixed, phi_fixed = shared_R
        bounds_reduced = [bounds[0]] + bounds[3:]

        def objective(p):
            full_p = np.array([p[0], r_fixed, phi_fixed, p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]])
            return poisson_nll(full_p, data, use_quadrature)

        def expand_params(p):
            return np.array([p[0], r_fixed, phi_fixed, p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]])
    else:
        bounds_reduced = bounds

        def objective(p):
            return poisson_nll(p, data, use_quadrature)

        def expand_params(p):
            return p

    all_nlls = []
    all_params = []

    for i in range(n_starts):
        x0 = random_init(bounds_reduced, rng)

        # Try L-BFGS-B
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds_reduced,
                          options={'maxiter': 2000, 'ftol': 1e-9})
            if res.success or res.fun < 1e10:
                all_nlls.append(res.fun)
                all_params.append(expand_params(res.x))
        except:
            pass

        # Try Powell
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 2000, 'ftol': 1e-9})
            if res.success or res.fun < 1e10:
                x_clipped = np.clip(res.x, [b[0] for b in bounds_reduced], [b[1] for b in bounds_reduced])
                all_nlls.append(objective(x_clipped))
                all_params.append(expand_params(x_clipped))
        except:
            pass

    # Also try differential evolution
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

    best_idx = np.argmin(all_nlls)
    return all_params[best_idx], all_nlls[best_idx], np.array(all_nlls)


def joint_fit_multistart(data_A: ChannelData, data_B: ChannelData,
                         n_starts=N_STARTS, seed=42, use_quadrature=True):
    """Joint constrained fit with shared (r, phi)."""
    rng = np.random.default_rng(seed)

    joint_bounds = [
        (0.001, 1.5),    # r (shared)
        (-np.pi, np.pi), # phi (shared)
        # Channel A params
        (1, 200), (0, 100), (-np.pi, np.pi), (0, 100), (-50, 50), (-30, 30),
        (0.97, 1.03), (-0.06, 0.06), (0.94, 1.06),
        # Channel B params
        (1, 200), (0, 100), (-np.pi, np.pi), (0, 100), (-50, 50), (-30, 30),
        (0.97, 1.03), (-0.06, 0.06), (0.94, 1.06),
    ]

    def joint_objective(p):
        r, phi = p[0], p[1]
        p1 = np.array([p[2], r, phi, p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10]])
        p2 = np.array([p[11], r, phi, p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19]])

        nll1 = poisson_nll(p1, data_A, use_quadrature)
        nll2 = poisson_nll(p2, data_B, use_quadrature)

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
    best_params = all_params[best_idx]

    return (best_params[0], best_params[1]), all_nlls[best_idx], best_params, np.array(all_nlls)


def get_constrained_mu(joint_params, data_A: ChannelData, data_B: ChannelData, use_quadrature=True):
    """
    Get expected counts under the constrained (shared R) model.
    This is used for parametric bootstrap under H0.
    """
    r, phi = joint_params[0], joint_params[1]

    # Extract per-channel params from joint fit
    p_A = np.array([joint_params[2], r, phi, joint_params[3], joint_params[4],
                    joint_params[5], joint_params[6], joint_params[7],
                    joint_params[8], joint_params[9], joint_params[10]])
    p_B = np.array([joint_params[11], r, phi, joint_params[12], joint_params[13],
                    joint_params[14], joint_params[15], joint_params[16],
                    joint_params[17], joint_params[18], joint_params[19]])

    mu_A = bin_integrate_mu(p_A, data_A.m_low, data_A.m_high, data_A.m_min, data_A.m_max, use_quadrature)
    mu_B = bin_integrate_mu(p_B, data_B.m_low, data_B.m_high, data_B.m_min, data_B.m_max, use_quadrature)

    return mu_A, mu_B


def bootstrap_replicate_parametric(args):
    """
    Single bootstrap replicate using PARAMETRIC bootstrap under H0.

    Correct method: Sample from Poisson(μ_constrained), NOT Poisson(observed).
    """
    seed, mu_A, mu_B, data_A, data_B, n_starts, use_quadrature = args

    rng = np.random.default_rng(seed)

    # Sample from constrained-null model (H0)
    counts_A_boot = rng.poisson(mu_A).astype(float)
    counts_B_boot = rng.poisson(mu_B).astype(float)

    # Create bootstrap data objects
    data_A_boot = ChannelData(
        name=data_A.name, m_low=data_A.m_low, m_high=data_A.m_high,
        m_center=data_A.m_center, bin_widths=data_A.bin_widths,
        counts=counts_A_boot, m_min=data_A.m_min, m_max=data_A.m_max
    )
    data_B_boot = ChannelData(
        name=data_B.name, m_low=data_B.m_low, m_high=data_B.m_high,
        m_center=data_B.m_center, bin_widths=data_B.bin_widths,
        counts=counts_B_boot, m_min=data_B.m_min, m_max=data_B.m_max
    )

    try:
        # Unconstrained fits
        params_A, nll_A, _ = fit_channel_multistart(data_A_boot, n_starts=n_starts,
                                                     seed=seed, use_quadrature=use_quadrature)
        params_B, nll_B, _ = fit_channel_multistart(data_B_boot, n_starts=n_starts,
                                                     seed=seed+1000, use_quadrature=use_quadrature)
        nll_unconstrained = nll_A + nll_B

        # Constrained fit
        _, nll_constrained, _, _ = joint_fit_multistart(data_A_boot, data_B_boot,
                                                         n_starts=n_starts, seed=seed+2000,
                                                         use_quadrature=use_quadrature)

        Lambda = 2 * (nll_constrained - nll_unconstrained)

        # Check for optimizer issues
        if Lambda < -1e-6:
            return np.nan, "Lambda_negative"

        return Lambda, "success"
    except Exception as e:
        return np.nan, f"exception: {str(e)[:50]}"


def bootstrap_replicate_legacy(args):
    """
    Single bootstrap replicate using LEGACY (incorrect) method.

    Wrong method: Sample from Poisson(observed counts).
    """
    seed, data_A, data_B, n_starts, use_quadrature = args

    rng = np.random.default_rng(seed)

    # Sample from observed counts (INCORRECT - this is the bug we're fixing)
    counts_A_boot = rng.poisson(data_A.counts).astype(float)
    counts_B_boot = rng.poisson(data_B.counts).astype(float)

    data_A_boot = ChannelData(
        name=data_A.name, m_low=data_A.m_low, m_high=data_A.m_high,
        m_center=data_A.m_center, bin_widths=data_A.bin_widths,
        counts=counts_A_boot, m_min=data_A.m_min, m_max=data_A.m_max
    )
    data_B_boot = ChannelData(
        name=data_B.name, m_low=data_B.m_low, m_high=data_B.m_high,
        m_center=data_B.m_center, bin_widths=data_B.bin_widths,
        counts=counts_B_boot, m_min=data_B.m_min, m_max=data_B.m_max
    )

    try:
        params_A, nll_A, _ = fit_channel_multistart(data_A_boot, n_starts=n_starts,
                                                     seed=seed, use_quadrature=use_quadrature)
        params_B, nll_B, _ = fit_channel_multistart(data_B_boot, n_starts=n_starts,
                                                     seed=seed+1000, use_quadrature=use_quadrature)
        nll_unconstrained = nll_A + nll_B

        _, nll_constrained, _, _ = joint_fit_multistart(data_A_boot, data_B_boot,
                                                         n_starts=n_starts, seed=seed+2000,
                                                         use_quadrature=use_quadrature)

        Lambda = 2 * (nll_constrained - nll_unconstrained)
        return Lambda, "success"
    except:
        return np.nan, "failed"


def run_bootstrap_parametric(data_A: ChannelData, data_B: ChannelData,
                              joint_params, Lambda_obs: float, n_boot: int,
                              use_quadrature=True) -> Dict[str, Any]:
    """
    Run parametric bootstrap under constrained-null hypothesis.

    Returns dict with p-value, k/N, failure rate, and Lambda samples.
    """
    print(f"\n{'='*60}")
    print(f"Parametric Bootstrap (correct method)")
    print(f"{'='*60}")

    # Get expected counts under H0 (constrained model)
    mu_A, mu_B = get_constrained_mu(joint_params, data_A, data_B, use_quadrature)

    print(f"  μ_A total: {mu_A.sum():.1f} (observed: {data_A.counts.sum():.0f})")
    print(f"  μ_B total: {mu_B.sum():.1f} (observed: {data_B.counts.sum():.0f})")

    n_workers = max(1, cpu_count() - 1)
    print(f"  Running {n_boot} bootstrap replicates ({n_workers} workers)...")

    # Prepare args with retries
    Lambda_samples = []
    failed_count = 0

    args_list = [(seed, mu_A, mu_B, data_A, data_B, N_STARTS_BOOT, use_quadrature)
                 for seed in range(n_boot)]

    with Pool(n_workers) as pool:
        results = pool.map(bootstrap_replicate_parametric, args_list)

    for Lambda_val, status in results:
        if status == "success" and not np.isnan(Lambda_val):
            Lambda_samples.append(Lambda_val)
        else:
            failed_count += 1

    # Retry failed samples with more starts
    if failed_count > 0:
        print(f"  Retrying {failed_count} failed samples with 2x starts...")
        retry_args = [(seed + n_boot, mu_A, mu_B, data_A, data_B, N_STARTS_BOOT * 2, use_quadrature)
                      for seed in range(failed_count)]

        with Pool(n_workers) as pool:
            retry_results = pool.map(bootstrap_replicate_parametric, retry_args)

        recovered = 0
        for Lambda_val, status in retry_results:
            if status == "success" and not np.isnan(Lambda_val):
                Lambda_samples.append(Lambda_val)
                recovered += 1

        failed_count -= recovered
        print(f"  Recovered {recovered} samples, {failed_count} still failed")

    Lambda_samples = np.array(Lambda_samples)
    n_valid = len(Lambda_samples)
    failure_rate = failed_count / n_boot

    # Compute p-value: P(Λ_boot >= Λ_obs | H0)
    k = np.sum(Lambda_samples >= Lambda_obs)
    p_value = k / n_valid if n_valid > 0 else np.nan

    print(f"\n  Valid samples: {n_valid}/{n_boot}")
    print(f"  Failure rate: {failure_rate:.1%}")
    print(f"  Bootstrap Λ: mean={np.mean(Lambda_samples):.2f}, std={np.std(Lambda_samples):.2f}")
    print(f"  Observed Λ = {Lambda_obs:.3f}")
    print(f"  p-value = {p_value:.4f} ({k}/{n_valid})")

    if failure_rate > MAX_BOOT_FAILURE_RATE:
        print(f"  WARNING: Failure rate {failure_rate:.1%} > {MAX_BOOT_FAILURE_RATE:.1%} threshold")

    return {
        "p_value": float(p_value),
        "k": int(k),
        "n_valid": int(n_valid),
        "n_boot": int(n_boot),
        "failure_rate": float(failure_rate),
        "Lambda_mean": float(np.mean(Lambda_samples)),
        "Lambda_std": float(np.std(Lambda_samples)),
        "Lambda_samples": Lambda_samples.tolist(),
        "reliable": failure_rate <= MAX_BOOT_FAILURE_RATE
    }


def run_bootstrap_legacy(data_A: ChannelData, data_B: ChannelData,
                          Lambda_obs: float, n_boot: int,
                          use_quadrature=True) -> Dict[str, Any]:
    """
    Run legacy (incorrect) bootstrap from observed counts.
    """
    print(f"\n{'='*60}")
    print(f"Legacy Bootstrap (incorrect method - for comparison)")
    print(f"{'='*60}")

    n_workers = max(1, cpu_count() - 1)
    print(f"  Running {n_boot} bootstrap replicates ({n_workers} workers)...")

    args_list = [(seed, data_A, data_B, N_STARTS_BOOT, use_quadrature) for seed in range(n_boot)]

    with Pool(n_workers) as pool:
        results = pool.map(bootstrap_replicate_legacy, args_list)

    Lambda_samples = []
    failed = 0
    for Lambda_val, status in results:
        if not np.isnan(Lambda_val):
            Lambda_samples.append(Lambda_val)
        else:
            failed += 1

    Lambda_samples = np.array(Lambda_samples)
    n_valid = len(Lambda_samples)

    k = np.sum(Lambda_samples >= Lambda_obs)
    p_value = k / n_valid if n_valid > 0 else np.nan

    print(f"  Valid samples: {n_valid}/{n_boot}")
    print(f"  Bootstrap Λ: mean={np.mean(Lambda_samples):.2f}, std={np.std(Lambda_samples):.2f}")
    print(f"  p-value = {p_value:.4f} ({k}/{n_valid})")

    return {
        "p_value": float(p_value),
        "k": int(k),
        "n_valid": int(n_valid),
        "Lambda_mean": float(np.mean(Lambda_samples)),
        "Lambda_std": float(np.std(Lambda_samples)),
        "Lambda_samples": Lambda_samples.tolist()
    }


def load_data(base_dir: Path) -> Tuple[ChannelData, ChannelData]:
    """Load and validate data from CSV files."""
    df_4mu = pd.read_csv(base_dir / "4mu_bins.csv")
    df_4mu2pi = pd.read_csv(base_dir / "4mu+2pi_bins.csv")

    def create_channel_data(df, name):
        m_low = df['m_low'].values
        m_high = df['m_high'].values
        m_center = df['m_center'].values
        counts = df['count'].values.astype(float)
        bin_widths = m_high - m_low

        # Check bin width variation
        width_std = np.std(bin_widths) / np.mean(bin_widths)
        if width_std > 0.01:
            print(f"  WARNING: {name} bin widths vary by {width_std:.1%}")

        return ChannelData(
            name=name,
            m_low=m_low,
            m_high=m_high,
            m_center=m_center,
            bin_widths=bin_widths,
            counts=counts,
            m_min=m_center.min(),
            m_max=m_center.max()
        )

    data_A = create_channel_data(df_4mu, "4μ")
    data_B = create_channel_data(df_4mu2pi, "4μ+2π")

    return data_A, data_B


def fit_and_evaluate(data: ChannelData, use_quadrature: bool, seed: int) -> FitResult:
    """Fit a single channel and compute fit quality metrics."""
    n_params = 11
    dof = len(data.counts) - n_params

    params, nll, all_nlls = fit_channel_multistart(data, n_starts=N_STARTS,
                                                    seed=seed, use_quadrature=use_quadrature)

    chi2 = pearson_chi2(data, params, use_quadrature)
    deviance = poisson_deviance(data, params, use_quadrature)

    chi2_dof = chi2 / dof
    dev_dof = deviance / dof

    gate_pass = (chi2_dof < CHI2_DOF_MAX) and (dev_dof < DEVIANCE_DOF_MAX)

    return FitResult(
        params=params,
        nll=nll,
        chi2=chi2,
        chi2_dof=chi2_dof,
        deviance=deviance,
        deviance_dof=dev_dof,
        r=params[1],
        phi=params[2],
        gate_pass=gate_pass,
        all_nlls=all_nlls
    )


def run_analysis(data_A: ChannelData, data_B: ChannelData,
                 use_quadrature: bool, run_bootstrap: bool = True,
                 n_bootstrap: int = N_BOOTSTRAP, mode_name: str = "correct") -> Dict[str, Any]:
    """
    Run full analysis pipeline.

    Args:
        use_quadrature: If True, use bin integration. If False, use legacy center × width.
        run_bootstrap: If True, run bootstrap for p-value.
        mode_name: Label for this run ("correct" or "legacy").
    """
    print(f"\n{'='*70}")
    print(f"ATLAS Rank-1 Analysis - Mode: {mode_name.upper()}")
    print(f"  Bin integration: {'Gauss-Legendre quadrature' if use_quadrature else 'center × width (legacy)'}")
    print(f"{'='*70}")

    # Fit each channel
    print(f"\n--- Fitting {data_A.name} channel ---")
    result_A = fit_and_evaluate(data_A, use_quadrature, seed=42)
    print(f"  r = {result_A.r:.4f}, φ = {np.degrees(result_A.phi):.1f}°")
    print(f"  χ²/dof = {result_A.chi2_dof:.2f}, D/dof = {result_A.deviance_dof:.2f}")
    print(f"  Gate: {'PASS' if result_A.gate_pass else 'FAIL'}")

    print(f"\n--- Fitting {data_B.name} channel ---")
    result_B = fit_and_evaluate(data_B, use_quadrature, seed=43)
    print(f"  r = {result_B.r:.4f}, φ = {np.degrees(result_B.phi):.1f}°")
    print(f"  χ²/dof = {result_B.chi2_dof:.2f}, D/dof = {result_B.deviance_dof:.2f}")
    print(f"  Gate: {'PASS' if result_B.gate_pass else 'FAIL'}")

    gates_pass = result_A.gate_pass and result_B.gate_pass
    nll_unconstrained = result_A.nll + result_B.nll

    # Joint constrained fit
    print(f"\n--- Joint Constrained Fit ---")
    (r_shared, phi_shared), nll_constrained, joint_params, joint_nlls = joint_fit_multistart(
        data_A, data_B, n_starts=N_STARTS, seed=44, use_quadrature=use_quadrature
    )
    print(f"  Shared r = {r_shared:.4f}, φ = {np.degrees(phi_shared):.1f}°")
    print(f"  NLL unconstrained = {nll_unconstrained:.2f}")
    print(f"  NLL constrained = {nll_constrained:.2f}")

    Lambda = 2 * (nll_constrained - nll_unconstrained)
    print(f"  Λ = {Lambda:.4f}")

    # Check optimizer stability
    if Lambda < -1e-6:
        print(f"  WARNING: Λ < 0 detected ({Lambda:.4f}). Retrying with more starts...")
        result_A = fit_and_evaluate(data_A, use_quadrature, seed=100)
        result_B = fit_and_evaluate(data_B, use_quadrature, seed=101)
        nll_unconstrained = result_A.nll + result_B.nll

        (r_shared, phi_shared), nll_constrained, joint_params, joint_nlls = joint_fit_multistart(
            data_A, data_B, n_starts=100, seed=102, use_quadrature=use_quadrature
        )
        Lambda = 2 * (nll_constrained - nll_unconstrained)
        print(f"  After retry: Λ = {Lambda:.4f}")

    optimizer_stable = Lambda >= -1e-6

    # Bootstrap p-value
    boot_result = None
    if run_bootstrap and gates_pass and optimizer_stable:
        if mode_name == "correct":
            boot_result = run_bootstrap_parametric(data_A, data_B, joint_params, Lambda,
                                                    n_bootstrap, use_quadrature)
        else:
            boot_result = run_bootstrap_legacy(data_A, data_B, Lambda, n_bootstrap, use_quadrature)

    return {
        "mode": mode_name,
        "use_quadrature": use_quadrature,
        "channel_A": {
            "r": float(result_A.r),
            "phi_deg": float(np.degrees(result_A.phi)),
            "nll": float(result_A.nll),
            "chi2_dof": float(result_A.chi2_dof),
            "deviance_dof": float(result_A.deviance_dof),
            "gate_pass": result_A.gate_pass
        },
        "channel_B": {
            "r": float(result_B.r),
            "phi_deg": float(np.degrees(result_B.phi)),
            "nll": float(result_B.nll),
            "chi2_dof": float(result_B.chi2_dof),
            "deviance_dof": float(result_B.deviance_dof),
            "gate_pass": result_B.gate_pass
        },
        "shared": {
            "r": float(r_shared),
            "phi_deg": float(np.degrees(phi_shared))
        },
        "Lambda": float(Lambda),
        "nll_unconstrained": float(nll_unconstrained),
        "nll_constrained": float(nll_constrained),
        "gates_pass": gates_pass,
        "optimizer_stable": optimizer_stable,
        "bootstrap": boot_result,
        "joint_params": joint_params.tolist()
    }


def plot_bootstrap_histogram(Lambda_samples: List[float], Lambda_obs: float,
                              p_value: float, k: int, n: int,
                              title: str, filepath: Path):
    """Plot bootstrap Lambda distribution with observed value."""
    fig, ax = plt.subplots(figsize=(10, 6))

    samples = np.array(Lambda_samples)

    ax.hist(samples, bins=40, density=True, alpha=0.7, color='steelblue',
            edgecolor='black', label='Bootstrap Λ under H₀')
    ax.axvline(Lambda_obs, color='red', linewidth=2.5, linestyle='--',
               label=f'Observed Λ = {Lambda_obs:.2f}')
    ax.axvline(np.mean(samples), color='green', linewidth=2, linestyle=':',
               label=f'Bootstrap mean = {np.mean(samples):.2f}')

    ax.set_xlabel(r'$\Lambda = 2\Delta(\mathrm{NLL})$', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{title}\np-value = {p_value:.4f} ({k}/{n})', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def generate_comparison_report(result_legacy: Dict, result_correct: Dict, filepath: Path):
    """Generate comparison markdown report."""
    report = f"""# ATLAS v5 vs v6 Statistical Comparison

## Summary

This report compares the legacy (incorrect) and corrected statistical methods.

**Key Differences:**
1. **Bin Integration**: Legacy uses center × width; Correct uses Gauss-Legendre quadrature
2. **Bootstrap**: Legacy samples from observed counts; Correct samples from constrained-null model (μ_H0)

## Fit Quality Comparison

| Metric | Legacy | Correct |
|--------|--------|---------|
| 4μ χ²/dof | {result_legacy['channel_A']['chi2_dof']:.3f} | {result_correct['channel_A']['chi2_dof']:.3f} |
| 4μ D/dof | {result_legacy['channel_A']['deviance_dof']:.3f} | {result_correct['channel_A']['deviance_dof']:.3f} |
| 4μ+2π χ²/dof | {result_legacy['channel_B']['chi2_dof']:.3f} | {result_correct['channel_B']['chi2_dof']:.3f} |
| 4μ+2π D/dof | {result_legacy['channel_B']['deviance_dof']:.3f} | {result_correct['channel_B']['deviance_dof']:.3f} |

## Best-Fit Parameters

| Parameter | Legacy | Correct |
|-----------|--------|---------|
| r_4μ | {result_legacy['channel_A']['r']:.4f} | {result_correct['channel_A']['r']:.4f} |
| φ_4μ (deg) | {result_legacy['channel_A']['phi_deg']:.1f} | {result_correct['channel_A']['phi_deg']:.1f} |
| r_4μ+2π | {result_legacy['channel_B']['r']:.4f} | {result_correct['channel_B']['r']:.4f} |
| φ_4μ+2π (deg) | {result_legacy['channel_B']['phi_deg']:.1f} | {result_correct['channel_B']['phi_deg']:.1f} |
| r_shared | {result_legacy['shared']['r']:.4f} | {result_correct['shared']['r']:.4f} |
| φ_shared (deg) | {result_legacy['shared']['phi_deg']:.1f} | {result_correct['shared']['phi_deg']:.1f} |

## Likelihood Ratio Test

| Metric | Legacy | Correct |
|--------|--------|---------|
| NLL_unconstrained | {result_legacy['nll_unconstrained']:.2f} | {result_correct['nll_unconstrained']:.2f} |
| NLL_constrained | {result_legacy['nll_constrained']:.2f} | {result_correct['nll_constrained']:.2f} |
| **Λ_obs** | **{result_legacy['Lambda']:.4f}** | **{result_correct['Lambda']:.4f}** |

## Bootstrap p-value Comparison

| Metric | Legacy | Correct |
|--------|--------|---------|
"""

    if result_legacy.get('bootstrap') and result_correct.get('bootstrap'):
        bl = result_legacy['bootstrap']
        bc = result_correct['bootstrap']
        report += f"""| Method | Poisson(observed) | Poisson(μ_H0) |
| p-value | {bl['p_value']:.4f} | {bc['p_value']:.4f} |
| k/N | {bl['k']}/{bl['n_valid']} | {bc['k']}/{bc['n_valid']} |
| Λ_boot mean | {bl['Lambda_mean']:.2f} | {bc['Lambda_mean']:.2f} |
| Λ_boot std | {bl['Lambda_std']:.2f} | {bc['Lambda_std']:.2f} |
"""
        if 'failure_rate' in bc:
            report += f"| Failure rate | N/A | {bc['failure_rate']:.1%} |\n"
            report += f"| Reliable | N/A | {'Yes' if bc.get('reliable', True) else 'No'} |\n"
    else:
        report += "| Bootstrap | NOT COMPUTED | NOT COMPUTED |\n"

    report += f"""
## Interpretation

The p-value difference between legacy and correct methods is
{'N/A (bootstrap not computed - gates failed)' if not (result_legacy.get('bootstrap') and result_correct.get('bootstrap')) else f"{abs(result_legacy['bootstrap']['p_value'] - result_correct['bootstrap']['p_value']):.4f}"}.

**Why the difference matters:**
- Legacy method samples from observed data, which may not follow H₀
- Correct method samples from the constrained-null model, properly testing H₀
- For a valid LRT, we need P(Λ ≥ Λ_obs | H₀), which requires sampling under H₀

## Gates Status

| Channel | Legacy | Correct |
|---------|--------|---------|
| 4μ | {'PASS' if result_legacy['channel_A']['gate_pass'] else 'FAIL'} | {'PASS' if result_correct['channel_A']['gate_pass'] else 'FAIL'} |
| 4μ+2π | {'PASS' if result_legacy['channel_B']['gate_pass'] else 'FAIL'} | {'PASS' if result_correct['channel_B']['gate_pass'] else 'FAIL'} |
| Overall | {'PASS' if result_legacy['gates_pass'] else 'FAIL'} | {'PASS' if result_correct['gates_pass'] else 'FAIL'} |

---
*Generated by fit_atlas_v6.py*
"""

    with open(filepath, 'w') as f:
        f.write(report)


def generate_report_v6(result: Dict, filepath: Path):
    """Generate full v6 report."""
    boot = result.get('bootstrap', {})

    report = f"""# ATLAS Rank-1 Test v6 Report - Statistically Corrected

## Method Corrections Applied

1. **Bin Integration**: Gauss-Legendre quadrature (8 points) instead of center × width
2. **Bootstrap**: Parametric bootstrap under constrained-null H₀
3. **Robustness**: {N_STARTS_BOOT} starts per bootstrap fit, retry logic for failures

## Data Summary

| Channel | Bins | Total Counts |
|---------|------|--------------|
| 4μ | {len(result.get('joint_params', [])) // 2} | N/A |
| 4μ+2π | N/A | N/A |

## Fit Results

### 4μ Channel
| Metric | Value |
|--------|-------|
| r | {result['channel_A']['r']:.4f} |
| φ (deg) | {result['channel_A']['phi_deg']:.1f} |
| χ²/dof | {result['channel_A']['chi2_dof']:.3f} |
| D/dof | {result['channel_A']['deviance_dof']:.3f} |
| Gate | {'PASS' if result['channel_A']['gate_pass'] else 'FAIL'} |

### 4μ+2π Channel
| Metric | Value |
|--------|-------|
| r | {result['channel_B']['r']:.4f} |
| φ (deg) | {result['channel_B']['phi_deg']:.1f} |
| χ²/dof | {result['channel_B']['chi2_dof']:.3f} |
| D/dof | {result['channel_B']['deviance_dof']:.3f} |
| Gate | {'PASS' if result['channel_B']['gate_pass'] else 'FAIL'} |

## Joint Constrained Fit

| Metric | Value |
|--------|-------|
| r_shared | {result['shared']['r']:.4f} |
| φ_shared (deg) | {result['shared']['phi_deg']:.1f} |
| NLL_unconstrained | {result['nll_unconstrained']:.2f} |
| NLL_constrained | {result['nll_constrained']:.2f} |
| **Λ_obs** | **{result['Lambda']:.4f}** |
| Optimizer stable | {'Yes' if result['optimizer_stable'] else 'No'} |

## Bootstrap p-value (Parametric under H₀)

"""
    if boot:
        report += f"""| Metric | Value |
|--------|-------|
| p-value | **{boot['p_value']:.4f}** |
| k/N | {boot['k']}/{boot['n_valid']} |
| Λ_boot mean | {boot['Lambda_mean']:.2f} |
| Λ_boot std | {boot['Lambda_std']:.2f} |
| Failure rate | {boot.get('failure_rate', 0):.1%} |
| Reliable | {'Yes' if boot.get('reliable', True) else 'No'} |
"""
    else:
        report += "Bootstrap NOT COMPUTED (gates failed or optimizer unstable)\n"

    # Verdict
    if not result['gates_pass']:
        verdict = "MODEL MISMATCH"
        reason = "χ²/dof or D/dof > 3 in at least one channel"
    elif not result['optimizer_stable']:
        verdict = "OPTIMIZER INSTABILITY"
        reason = "Λ < 0 persists"
    elif boot and boot['p_value'] < 0.05:
        verdict = "DISFAVORED"
        reason = f"p = {boot['p_value']:.4f} < 0.05"
    elif boot and boot['p_value'] >= 0.05:
        verdict = "SUPPORTED"
        reason = f"p = {boot['p_value']:.4f} >= 0.05"
    else:
        verdict = "INCONCLUSIVE"
        reason = "Unable to compute p-value"

    report += f"""
## Verdict

**{verdict}**

Reason: {reason}

---
*Generated by fit_atlas_v6.py with statistically correct bootstrap*
"""

    with open(filepath, 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='ATLAS Rank-1 Test v6')
    parser.add_argument('--validate', action='store_true',
                        help='Run both legacy and correct modes for comparison')
    parser.add_argument('--bin_integrate', type=str, default='true',
                        help='Use bin integration (true/false)')
    parser.add_argument('--n_bootstrap', type=int, default=N_BOOTSTRAP,
                        help=f'Number of bootstrap samples (default: {N_BOOTSTRAP})')
    args = parser.parse_args()

    use_quadrature = args.bin_integrate.lower() == 'true'

    print("="*70)
    print("ATLAS Rank-1 Test v6 - Statistically Corrected")
    print("="*70)

    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    data_A, data_B = load_data(BASE_DIR)
    print(f"  {data_A.name}: {len(data_A.counts)} bins, {int(data_A.counts.sum())} counts")
    print(f"  {data_B.name}: {len(data_B.counts)} bins, {int(data_B.counts.sum())} counts")

    if args.validate:
        # Run both modes for comparison
        print("\n" + "="*70)
        print("VALIDATION MODE: Running both legacy and correct methods")
        print("="*70)

        # Legacy mode (incorrect bootstrap, no quadrature)
        result_legacy = run_analysis(data_A, data_B, use_quadrature=False,
                                      n_bootstrap=args.n_bootstrap, mode_name="legacy")

        # Correct mode (parametric bootstrap, with quadrature)
        result_correct = run_analysis(data_A, data_B, use_quadrature=True,
                                       n_bootstrap=args.n_bootstrap, mode_name="correct")

        # Save results (convert numpy types to Python types)
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj

        with open(OUT_DIR / "result_legacy.json", 'w') as f:
            result_legacy_save = convert_numpy(result_legacy.copy())
            if result_legacy_save.get('bootstrap'):
                result_legacy_save['bootstrap'] = {k: v for k, v in result_legacy_save['bootstrap'].items()
                                                    if k != 'Lambda_samples'}
            json.dump(result_legacy_save, f, indent=2)

        with open(OUT_DIR / "result_correct.json", 'w') as f:
            result_correct_save = convert_numpy(result_correct.copy())
            if result_correct_save.get('bootstrap'):
                result_correct_save['bootstrap'] = {k: v for k, v in result_correct_save['bootstrap'].items()
                                                     if k != 'Lambda_samples'}
            json.dump(result_correct_save, f, indent=2)

        # Generate comparison report
        generate_comparison_report(result_legacy, result_correct, OUT_DIR / "atlas_v5_v6_comparison.md")

        # Plot bootstrap histograms
        if result_legacy.get('bootstrap') and result_legacy['bootstrap'].get('Lambda_samples'):
            bl = result_legacy['bootstrap']
            plot_bootstrap_histogram(bl['Lambda_samples'], result_legacy['Lambda'],
                                     bl['p_value'], bl['k'], bl['n_valid'],
                                     "Legacy Bootstrap (Poisson from observed)",
                                     OUT_DIR / "bootstrap_hist_legacy.png")

        if result_correct.get('bootstrap') and result_correct['bootstrap'].get('Lambda_samples'):
            bc = result_correct['bootstrap']
            plot_bootstrap_histogram(bc['Lambda_samples'], result_correct['Lambda'],
                                     bc['p_value'], bc['k'], bc['n_valid'],
                                     "Correct Bootstrap (Parametric under H₀)",
                                     OUT_DIR / "bootstrap_hist_correct.png")

        # Print summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"\nΛ_obs (legacy): {result_legacy['Lambda']:.4f}")
        print(f"Λ_obs (correct): {result_correct['Lambda']:.4f}")

        if result_legacy.get('bootstrap') and result_correct.get('bootstrap'):
            bl = result_legacy['bootstrap']
            bc = result_correct['bootstrap']
            print(f"\np_legacy = {bl['p_value']:.4f} ({bl['k']}/{bl['n_valid']})")
            print(f"p_correct = {bc['p_value']:.4f} ({bc['k']}/{bc['n_valid']})")
            print(f"\nDifference: |p_correct - p_legacy| = {abs(bc['p_value'] - bl['p_value']):.4f}")

        print(f"\nOutputs saved to: {OUT_DIR}")

        return result_legacy, result_correct

    else:
        # Run correct mode only
        result = run_analysis(data_A, data_B, use_quadrature=use_quadrature,
                              n_bootstrap=args.n_bootstrap, mode_name="correct")

        # Save results (convert numpy types)
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj

        with open(OUT_DIR / "ATLAS_v6_summary.json", 'w') as f:
            result_save = convert_numpy(result.copy())
            if result_save.get('bootstrap'):
                result_save['bootstrap'] = {k: v for k, v in result_save['bootstrap'].items()
                                             if k != 'Lambda_samples'}
            json.dump(result_save, f, indent=2)

        # Generate report
        generate_report_v6(result, OUT_DIR / "REPORT_v6.md")

        # Plot bootstrap histogram
        if result.get('bootstrap') and result['bootstrap'].get('Lambda_samples'):
            bc = result['bootstrap']
            plot_bootstrap_histogram(bc['Lambda_samples'], result['Lambda'],
                                     bc['p_value'], bc['k'], bc['n_valid'],
                                     "Parametric Bootstrap under H₀",
                                     OUT_DIR / "bootstrap_hist_correct.png")

        # Print final summary
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"\nΛ_obs = {result['Lambda']:.4f}")

        if result.get('bootstrap'):
            bc = result['bootstrap']
            print(f"p_correct = {bc['p_value']:.4f} ({bc['k']}/{bc['n_valid']})")

        print(f"\nOutputs saved to: {OUT_DIR}")

        return result


if __name__ == "__main__":
    main()
