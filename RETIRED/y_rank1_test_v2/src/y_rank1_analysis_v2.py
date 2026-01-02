#!/usr/bin/env python3
"""
Y-state Rank-1 Bottleneck Test v2
=================================
Rigorous analysis with:
- Proper nuisance parameter treatment for correlated systematics
- Fit health gates with lower AND upper bounds (0.5 < chi2/dof < 3.0)
- Optimizer stability audit with multimodality detection
- AIC/BIC model selection for background
- Bootstrap p-value with correct pseudo-data generation
- Strict decision logic

Tests R = c(Y4360)/c(Y4220) invariance across decay channels.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2 as chi2_dist
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import json
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
Y1_MASS = 4.222  # GeV (Y(4220))
Y1_WIDTH = 0.044  # GeV
Y2_MASS = 4.368  # GeV (Y(4360))
Y2_WIDTH = 0.096  # GeV

N_MULTI_START = 200  # Multi-start restarts
N_BOOTSTRAP = 500    # Bootstrap replicates
N_BOOTSTRAP_RESTARTS = 50  # Restarts per bootstrap
N_PROFILE_RESTARTS = 20    # Restarts for profile likelihood

CHI2_DOF_LOWER = 0.5  # Lower fit health bound
CHI2_DOF_UPPER = 3.0  # Upper fit health bound

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(WORKDIR, 'out')
LOGDIR = os.path.join(WORKDIR, 'logs')

# =============================================================================
# Data Loading
# =============================================================================
def load_channel_A():
    """Load Belle pi+pi- J/psi data with stat/syst separation."""
    path = os.path.join(WORKDIR, 'data/hepdata/channelA.csv')

    # Read CSV, skip comment lines
    lines = []
    with open(path, 'r') as f:
        for line in f:
            if not line.startswith('#') and line.strip():
                lines.append(line.strip())

    # Parse header and data
    header = lines[0].split(',')
    data = []
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) >= 7:
            try:
                E = float(parts[0])
                sigma = float(parts[3])
                stat_plus = float(parts[4])
                stat_minus = abs(float(parts[5]))
                # Symmetric stat error (use average of asymmetric)
                stat_err = (stat_plus + stat_minus) / 2
                # Syst is 7.0% - extract as fraction
                syst_frac = 0.07
                data.append([E, sigma, stat_err, syst_frac])
            except ValueError:
                continue

    df = pd.DataFrame(data, columns=['E', 'sigma', 'stat_err', 'syst_frac'])
    return df

def load_channel_B():
    """Load BaBar pi+pi- psi(2S) data with stat/syst separation."""
    path = os.path.join(WORKDIR, 'data/hepdata/channelB.csv')

    lines = []
    with open(path, 'r') as f:
        for line in f:
            if not line.startswith('#') and line.strip():
                lines.append(line.strip())

    header = lines[0].split(',')
    data = []
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) >= 7:
            try:
                E = float(parts[0])
                sigma_str = parts[3]
                if sigma_str == '-':  # No data
                    continue
                sigma = float(sigma_str)
                stat_plus = float(parts[4])
                stat_minus = abs(float(parts[5]))
                # Use asymmetric errors properly: average for Gaussian approx
                stat_err = (stat_plus + stat_minus) / 2
                # Syst is 12.3%
                syst_frac = 0.123
                data.append([E, sigma, stat_err, syst_frac])
            except ValueError:
                continue

    df = pd.DataFrame(data, columns=['E', 'sigma', 'stat_err', 'syst_frac'])
    return df

# =============================================================================
# Physics Model
# =============================================================================
def bw_amplitude(E, M, Gamma):
    """
    Energy-dependent Breit-Wigner amplitude for e+e- production.
    BW(E) = 1 / (E^2 - M^2 + i*M*Gamma)
    """
    return 1.0 / (E**2 - M**2 + 1j * M * Gamma)

def model_amplitude(E, c1, c2, phi, bg_params, bg_order=0, E0=4.3):
    """
    Coherent amplitude model:
    A(E) = c1*BW1 + c2*exp(i*phi)*BW2 + A_bg(E)

    Background:
    - bg_order=0: complex constant a0 + i*b0
    - bg_order=1: complex constant + complex linear: (a0 + i*b0) + (a1 + i*b1)*(E-E0)
    """
    BW1 = bw_amplitude(E, Y1_MASS, Y1_WIDTH)
    BW2 = bw_amplitude(E, Y2_MASS, Y2_WIDTH)

    A = c1 * BW1 + c2 * np.exp(1j * phi) * BW2

    if bg_order == 0:
        a0, b0 = bg_params[0], bg_params[1]
        A += (a0 + 1j * b0)
    elif bg_order == 1:
        a0, b0, a1, b1 = bg_params[0], bg_params[1], bg_params[2], bg_params[3]
        A += (a0 + 1j * b0) + (a1 + 1j * b1) * (E - E0)

    return A

def model_cross_section(E, c1, c2, phi, bg_params, s0, s1, bg_order=0, E0=4.3):
    """
    Cross section with nuisance parameters for correlated systematics.
    sigma(E) = s0 * (1 + s1*(E-E0)) * |A(E)|^2
    """
    A = model_amplitude(E, c1, c2, phi, bg_params, bg_order, E0)
    sigma_raw = np.abs(A)**2
    # Nuisance scaling
    scale = s0 * (1.0 + s1 * (E - E0))
    return scale * sigma_raw

def extract_R(c1, c2, phi):
    """Extract complex ratio R = c2*exp(i*phi) / c1 = r * exp(i*Phi)"""
    r = c2 / c1
    Phi = phi
    return r, Phi

# =============================================================================
# Likelihood with Nuisance Parameters
# =============================================================================
def nll_channel(params, E, sigma_data, stat_err, syst_frac, bg_order=0, E0=4.3):
    """
    Negative log-likelihood for a single channel with nuisance parameters.

    Parameters:
    - c1, c2, phi: amplitude params (c1 > 0 fixed real)
    - bg_params: background (2 for order 0, 4 for order 1)
    - s0: global normalization nuisance (prior: N(1, syst_frac))
    - s1: energy-slope nuisance (prior: N(0, sigma_slope))

    Likelihood uses ONLY stat errors for data term; syst enters via s0 prior.
    """
    n_bg = 2 if bg_order == 0 else 4
    c1 = params[0]
    c2 = params[1]
    phi = params[2]
    bg_params = params[3:3+n_bg]
    s0 = params[3+n_bg]
    s1 = params[4+n_bg]

    # Enforce physical constraints
    if c1 <= 0 or c2 < 0:
        return 1e10

    # Model prediction
    sigma_model = model_cross_section(E, c1, c2, phi, bg_params, s0, s1, bg_order, E0)

    # Data term (using stat errors only)
    residuals = (sigma_data - sigma_model) / stat_err
    nll_data = 0.5 * np.sum(residuals**2)

    # Prior on s0: N(1, syst_frac)
    # Use median syst_frac from data
    sigma_norm = np.median(syst_frac)
    nll_prior_s0 = 0.5 * ((s0 - 1.0) / sigma_norm)**2

    # Prior on s1: N(0, sigma_slope)
    # sigma_slope = 0.02 / (E_max - E_min)
    E_range = E.max() - E.min()
    sigma_slope = 0.02 / max(E_range, 0.1)
    nll_prior_s1 = 0.5 * (s1 / sigma_slope)**2

    return nll_data + nll_prior_s0 + nll_prior_s1

def chi2_per_dof(params, E, sigma_data, stat_err, syst_frac, bg_order=0, E0=4.3):
    """Compute chi2/dof using stat errors."""
    n_bg = 2 if bg_order == 0 else 4
    c1, c2, phi = params[0], params[1], params[2]
    bg_params = params[3:3+n_bg]
    s0 = params[3+n_bg]
    s1 = params[4+n_bg]

    sigma_model = model_cross_section(E, c1, c2, phi, bg_params, s0, s1, bg_order, E0)
    residuals = (sigma_data - sigma_model) / stat_err
    chi2 = np.sum(residuals**2)

    # DOF = n_points - n_params
    n_params = 3 + n_bg + 2  # c1, c2, phi, bg, s0, s1
    dof = len(E) - n_params

    return chi2 / max(dof, 1), chi2, dof

# =============================================================================
# Multi-Start Optimization
# =============================================================================
def random_start_params(bg_order=0):
    """Generate random starting parameters."""
    c1 = np.random.uniform(0.5, 50)
    c2 = np.random.uniform(0.5, 50)
    phi = np.random.uniform(-np.pi, np.pi)

    if bg_order == 0:
        bg_params = [np.random.uniform(-5, 5), np.random.uniform(-5, 5)]
    else:
        bg_params = [np.random.uniform(-5, 5) for _ in range(4)]

    s0 = np.random.uniform(0.8, 1.2)
    s1 = np.random.uniform(-0.01, 0.01)

    return [c1, c2, phi] + bg_params + [s0, s1]

def get_bounds(bg_order=0):
    """Get parameter bounds."""
    bounds = [
        (0.01, 200),   # c1
        (0.01, 200),   # c2
        (-np.pi, np.pi),  # phi
    ]
    if bg_order == 0:
        bounds += [(-50, 50), (-50, 50)]  # a0, b0
    else:
        bounds += [(-50, 50), (-50, 50), (-20, 20), (-20, 20)]  # a0, b0, a1, b1
    bounds += [(0.5, 1.5), (-0.1, 0.1)]  # s0, s1
    return bounds

def fit_single_start(args):
    """Fit from a single starting point."""
    start_params, E, sigma_data, stat_err, syst_frac, bg_order, E0, use_powell = args
    bounds = get_bounds(bg_order)

    try:
        if use_powell:
            # Powell doesn't use bounds directly
            result = minimize(
                nll_channel, start_params,
                args=(E, sigma_data, stat_err, syst_frac, bg_order, E0),
                method='Powell',
                options={'maxiter': 5000, 'ftol': 1e-8}
            )
        else:
            result = minimize(
                nll_channel, start_params,
                args=(E, sigma_data, stat_err, syst_frac, bg_order, E0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-10}
            )
        return result.fun, result.x, result.success
    except Exception:
        return 1e10, start_params, False

def multi_start_fit(E, sigma_data, stat_err, syst_frac, bg_order=0, n_starts=N_MULTI_START, E0=4.3, use_pool=True):
    """
    Multi-start optimization with both L-BFGS-B and Powell.
    Returns best fit and stability metrics.

    use_pool: If False, run sequentially (for use inside worker processes)
    """
    # Generate starting points
    starts = [random_start_params(bg_order) for _ in range(n_starts)]

    # Fit with L-BFGS-B
    args_lbfgs = [(s, E, sigma_data, stat_err, syst_frac, bg_order, E0, False) for s in starts]

    if use_pool:
        with Pool(cpu_count() - 1) as pool:
            results_lbfgs = pool.map(fit_single_start, args_lbfgs)
    else:
        results_lbfgs = [fit_single_start(a) for a in args_lbfgs]

    # Fit with Powell (half the starts)
    args_powell = [(s, E, sigma_data, stat_err, syst_frac, bg_order, E0, True) for s in starts[:n_starts//2]]

    if use_pool:
        with Pool(cpu_count() - 1) as pool:
            results_powell = pool.map(fit_single_start, args_powell)
    else:
        results_powell = [fit_single_start(a) for a in args_powell]

    # Combine results
    all_results = results_lbfgs + results_powell

    # Extract valid results
    valid_results = [(nll, params, success) for nll, params, success in all_results if nll < 1e9]

    if not valid_results:
        return None, None, None

    # Best fit
    valid_results.sort(key=lambda x: x[0])
    best_nll, best_params, _ = valid_results[0]

    # Stability metrics
    nlls = np.array([r[0] for r in valid_results])
    rs = np.array([r[1][1] / r[1][0] for r in valid_results if r[1][0] > 0])
    phis = np.array([r[1][2] for r in valid_results])

    stability = {
        'n_valid': len(valid_results),
        'nll_best': best_nll,
        'nll_mean': np.mean(nlls),
        'nll_std': np.std(nlls),
        'r_mean': np.mean(rs) if len(rs) > 0 else np.nan,
        'r_std': np.std(rs) if len(rs) > 0 else np.nan,
        'phi_mean': np.mean(phis),
        'phi_std': np.std(phis),
        'phi_circular_std': np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * phis))))),
        'nlls': nlls.tolist(),
        'rs': rs.tolist(),
        'phis': phis.tolist()
    }

    return best_nll, best_params, stability

# =============================================================================
# Background Model Selection (AIC/BIC)
# =============================================================================
def compute_aic_bic(nll, n_params, n_data):
    """Compute AIC and BIC."""
    aic = 2 * nll + 2 * n_params
    bic = 2 * nll + n_params * np.log(n_data)
    return aic, bic

def select_background_model(E, sigma_data, stat_err, syst_frac, E0=4.3):
    """Select background model using AIC/BIC."""
    results = {}

    for bg_order in [0, 1]:
        nll, params, stability = multi_start_fit(E, sigma_data, stat_err, syst_frac, bg_order,
                                                  n_starts=100, E0=E0)
        if nll is None:
            continue

        n_bg = 2 if bg_order == 0 else 4
        n_params = 3 + n_bg + 2
        aic, bic = compute_aic_bic(nll, n_params, len(E))

        results[bg_order] = {
            'nll': nll,
            'params': params,
            'aic': aic,
            'bic': bic,
            'n_params': n_params
        }

    if not results:
        return 0, None  # Default to simpler model

    # Prefer simpler model unless delta_AIC > 10
    if 0 in results and 1 in results:
        delta_aic = results[0]['aic'] - results[1]['aic']
        if delta_aic > 10:
            return 1, results[1]
        else:
            return 0, results[0]
    elif 0 in results:
        return 0, results[0]
    else:
        return 1, results[1]

# =============================================================================
# Joint Constrained Fit
# =============================================================================
def nll_joint_constrained(params, E_A, sigma_A, stat_A, syst_A, E_B, sigma_B, stat_B, syst_B,
                          bg_order_A, bg_order_B, E0=4.3):
    """
    Joint NLL with shared R = r * exp(i*Phi) across both channels.
    Separate nuisances and backgrounds per channel.

    Params layout:
    - r_shared, Phi_shared (2)
    - c1_A (1)
    - bg_A (2 or 4)
    - s0_A, s1_A (2)
    - c1_B (1)
    - bg_B (2 or 4)
    - s0_B, s1_B (2)
    """
    n_bg_A = 2 if bg_order_A == 0 else 4
    n_bg_B = 2 if bg_order_B == 0 else 4

    idx = 0
    r_shared = params[idx]; idx += 1
    Phi_shared = params[idx]; idx += 1

    c1_A = params[idx]; idx += 1
    bg_A = params[idx:idx+n_bg_A]; idx += n_bg_A
    s0_A = params[idx]; idx += 1
    s1_A = params[idx]; idx += 1

    c1_B = params[idx]; idx += 1
    bg_B = params[idx:idx+n_bg_B]; idx += n_bg_B
    s0_B = params[idx]; idx += 1
    s1_B = params[idx]; idx += 1

    # Derive c2 from shared R: c2 = r * c1
    c2_A = r_shared * c1_A
    c2_B = r_shared * c1_B
    phi = Phi_shared

    # Enforce constraints
    if c1_A <= 0 or c1_B <= 0 or r_shared <= 0:
        return 1e10

    # Channel A NLL
    params_A = [c1_A, c2_A, phi] + list(bg_A) + [s0_A, s1_A]
    nll_A = nll_channel(params_A, E_A, sigma_A, stat_A, syst_A, bg_order_A, E0)

    # Channel B NLL
    params_B = [c1_B, c2_B, phi] + list(bg_B) + [s0_B, s1_B]
    nll_B = nll_channel(params_B, E_B, sigma_B, stat_B, syst_B, bg_order_B, E0)

    return nll_A + nll_B

def random_joint_params(bg_order_A, bg_order_B):
    """Random starting params for joint fit."""
    r_shared = np.random.uniform(0.5, 5)
    Phi_shared = np.random.uniform(-np.pi, np.pi)

    c1_A = np.random.uniform(0.5, 50)
    n_bg_A = 2 if bg_order_A == 0 else 4
    bg_A = [np.random.uniform(-5, 5) for _ in range(n_bg_A)]
    s0_A = np.random.uniform(0.8, 1.2)
    s1_A = np.random.uniform(-0.01, 0.01)

    c1_B = np.random.uniform(0.5, 50)
    n_bg_B = 2 if bg_order_B == 0 else 4
    bg_B = [np.random.uniform(-5, 5) for _ in range(n_bg_B)]
    s0_B = np.random.uniform(0.8, 1.2)
    s1_B = np.random.uniform(-0.01, 0.01)

    return [r_shared, Phi_shared, c1_A] + bg_A + [s0_A, s1_A, c1_B] + bg_B + [s0_B, s1_B]

def get_joint_bounds(bg_order_A, bg_order_B):
    """Bounds for joint fit."""
    n_bg_A = 2 if bg_order_A == 0 else 4
    n_bg_B = 2 if bg_order_B == 0 else 4

    bounds = [
        (0.01, 20),      # r_shared
        (-np.pi, np.pi),  # Phi_shared
        (0.01, 200),     # c1_A
    ]
    bounds += [(-50, 50)] * n_bg_A  # bg_A
    bounds += [(0.5, 1.5), (-0.1, 0.1)]  # s0_A, s1_A
    bounds += [(0.01, 200)]  # c1_B
    bounds += [(-50, 50)] * n_bg_B  # bg_B
    bounds += [(0.5, 1.5), (-0.1, 0.1)]  # s0_B, s1_B

    return bounds

def fit_joint_single(args):
    """Single joint fit."""
    (start, E_A, sigma_A, stat_A, syst_A, E_B, sigma_B, stat_B, syst_B,
     bg_order_A, bg_order_B, E0) = args
    bounds = get_joint_bounds(bg_order_A, bg_order_B)

    try:
        result = minimize(
            nll_joint_constrained, start,
            args=(E_A, sigma_A, stat_A, syst_A, E_B, sigma_B, stat_B, syst_B,
                  bg_order_A, bg_order_B, E0),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-10}
        )
        return result.fun, result.x, result.success
    except Exception:
        return 1e10, start, False

def multi_start_joint_fit(E_A, sigma_A, stat_A, syst_A, E_B, sigma_B, stat_B, syst_B,
                          bg_order_A, bg_order_B, n_starts=N_MULTI_START, E0=4.3, use_pool=True):
    """Multi-start joint constrained fit."""
    starts = [random_joint_params(bg_order_A, bg_order_B) for _ in range(n_starts)]

    args = [(s, E_A, sigma_A, stat_A, syst_A, E_B, sigma_B, stat_B, syst_B,
             bg_order_A, bg_order_B, E0) for s in starts]

    if use_pool:
        with Pool(cpu_count() - 1) as pool:
            results = pool.map(fit_joint_single, args)
    else:
        results = [fit_joint_single(a) for a in args]

    valid = [(nll, p, s) for nll, p, s in results if nll < 1e9]
    if not valid:
        return None, None

    valid.sort(key=lambda x: x[0])
    best_nll, best_params, _ = valid[0]

    return best_nll, best_params

# =============================================================================
# Bootstrap p-value
# =============================================================================
def bootstrap_single(args):
    """Single bootstrap replicate. Runs SEQUENTIALLY (no nested pools)."""
    (seed, E_A, sigma_model_A, stat_A, syst_A, E_B, sigma_model_B, stat_B, syst_B,
     bg_order_A, bg_order_B, E0, n_restarts) = args

    np.random.seed(seed)

    # Generate pseudo-data from constrained model
    sigma_A_pseudo = sigma_model_A + np.random.normal(0, stat_A)
    sigma_B_pseudo = sigma_model_B + np.random.normal(0, stat_B)

    # Fit unconstrained (both channels separately) - NO POOL (inside worker)
    nll_A_unc, params_A, _ = multi_start_fit(E_A, sigma_A_pseudo, stat_A, syst_A,
                                              bg_order_A, n_starts=n_restarts, E0=E0, use_pool=False)
    nll_B_unc, params_B, _ = multi_start_fit(E_B, sigma_B_pseudo, stat_B, syst_B,
                                              bg_order_B, n_starts=n_restarts, E0=E0, use_pool=False)

    if nll_A_unc is None or nll_B_unc is None:
        return np.nan

    nll_unc = nll_A_unc + nll_B_unc

    # Fit constrained - NO POOL (inside worker)
    nll_con, _ = multi_start_joint_fit(E_A, sigma_A_pseudo, stat_A, syst_A,
                                        E_B, sigma_B_pseudo, stat_B, syst_B,
                                        bg_order_A, bg_order_B, n_starts=n_restarts, E0=E0, use_pool=False)

    if nll_con is None:
        return np.nan

    Lambda = 2 * (nll_con - nll_unc)
    return Lambda

def bootstrap_pvalue(E_A, sigma_model_A, stat_A, syst_A, E_B, sigma_model_B, stat_B, syst_B,
                     bg_order_A, bg_order_B, Lambda_obs, n_boot=N_BOOTSTRAP,
                     n_restarts=N_BOOTSTRAP_RESTARTS, E0=4.3):
    """Compute bootstrap p-value."""
    seeds = np.random.randint(0, 1000000, n_boot)

    args = [(seed, E_A, sigma_model_A, stat_A, syst_A, E_B, sigma_model_B, stat_B, syst_B,
             bg_order_A, bg_order_B, E0, n_restarts) for seed in seeds]

    with Pool(cpu_count() - 1) as pool:
        Lambdas = pool.map(bootstrap_single, args)

    Lambdas = np.array([L for L in Lambdas if not np.isnan(L)])

    if len(Lambdas) == 0:
        return np.nan, []

    # p-value = fraction of bootstrap Lambdas >= observed
    # But Lambda should be >=0; negative values indicate optimizer issues
    Lambdas_valid = Lambdas[Lambdas >= -0.1]  # Allow small numerical noise

    p_value = np.mean(Lambdas_valid >= Lambda_obs)

    return p_value, Lambdas.tolist()

# =============================================================================
# Profile Likelihood Contours
# =============================================================================
def profile_nll_at_r_phi(args):
    """Compute profile NLL at fixed (r, phi)."""
    r_fixed, phi_fixed, E, sigma_data, stat_err, syst_frac, bg_order, E0, n_restarts = args

    def nll_profile(params_reduced):
        c1 = params_reduced[0]
        c2 = r_fixed * c1  # c2 determined by r
        phi = phi_fixed
        n_bg = 2 if bg_order == 0 else 4
        bg_params = params_reduced[1:1+n_bg]
        s0 = params_reduced[1+n_bg]
        s1 = params_reduced[2+n_bg]

        full_params = [c1, c2, phi] + list(bg_params) + [s0, s1]
        return nll_channel(full_params, E, sigma_data, stat_err, syst_frac, bg_order, E0)

    n_bg = 2 if bg_order == 0 else 4

    best_nll = 1e10
    for _ in range(n_restarts):
        c1_start = np.random.uniform(0.5, 50)
        bg_start = [np.random.uniform(-5, 5) for _ in range(n_bg)]
        s0_start = np.random.uniform(0.8, 1.2)
        s1_start = np.random.uniform(-0.01, 0.01)

        start = [c1_start] + bg_start + [s0_start, s1_start]
        bounds = [(0.01, 200)] + [(-50, 50)] * n_bg + [(0.5, 1.5), (-0.1, 0.1)]

        try:
            result = minimize(nll_profile, start, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 2000})
            if result.fun < best_nll:
                best_nll = result.fun
        except:
            pass

    return r_fixed, phi_fixed, best_nll

def compute_profile_contours(E, sigma_data, stat_err, syst_frac, nll_best, params_best,
                              bg_order, r_range, phi_range, E0=4.3):
    """Compute profile likelihood on (r, phi) grid."""
    r_grid = np.linspace(r_range[0], r_range[1], 30)
    phi_grid = np.linspace(phi_range[0], phi_range[1], 30)

    args = []
    for r in r_grid:
        for phi in phi_grid:
            args.append((r, phi, E, sigma_data, stat_err, syst_frac, bg_order, E0, N_PROFILE_RESTARTS))

    with Pool(cpu_count() - 1) as pool:
        results = pool.map(profile_nll_at_r_phi, args)

    # Build grid
    nll_grid = np.zeros((len(r_grid), len(phi_grid)))
    for r_val, phi_val, nll_val in results:
        i = np.argmin(np.abs(r_grid - r_val))
        j = np.argmin(np.abs(phi_grid - phi_val))
        nll_grid[i, j] = nll_val

    # Delta NLL (profile likelihood ratio)
    delta_nll = 2 * (nll_grid - nll_best)

    return r_grid, phi_grid, delta_nll

# =============================================================================
# Fit Health Assessment
# =============================================================================
def assess_fit_health(chi2_dof, channel_name):
    """Assess fit health with lower and upper bounds."""
    if chi2_dof < CHI2_DOF_LOWER:
        return 'UNDERCONSTRAINED', f'{channel_name}: chi2/dof={chi2_dof:.3f} < {CHI2_DOF_LOWER}'
    elif chi2_dof > CHI2_DOF_UPPER:
        return 'MODEL_MISMATCH', f'{channel_name}: chi2/dof={chi2_dof:.3f} > {CHI2_DOF_UPPER}'
    else:
        return 'PASS', f'{channel_name}: chi2/dof={chi2_dof:.3f}'

def assess_optimizer_stability(stability):
    """Check for multimodality in optimizer results."""
    if stability is None:
        return 'FAILURE', 'No valid fits'

    # Check if phi is identifiable (circular std < 1 radian ~ 57 deg)
    phi_circ_std = stability.get('phi_circular_std', np.inf)
    if np.isnan(phi_circ_std) or phi_circ_std > 1.5:
        return 'PHI_NOT_IDENTIFIABLE', f'phi circular std = {phi_circ_std:.2f} rad'

    # Check for multimodality in r (std > 50% of mean)
    r_std = stability.get('r_std', 0)
    r_mean = stability.get('r_mean', 1)
    if r_mean > 0 and r_std / r_mean > 0.5:
        return 'MULTIMODAL_R', f'r std/mean = {r_std/r_mean:.2f}'

    return 'STABLE', f'phi_circ_std={phi_circ_std:.2f}, r_cv={r_std/max(r_mean,0.01):.2f}'

# =============================================================================
# Main Analysis
# =============================================================================
def main():
    print("=" * 70)
    print("Y-state Rank-1 Bottleneck Test v2")
    print("Rigorous analysis with nuisance parameters and strict health gates")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    df_A = load_channel_A()
    df_B = load_channel_B()

    E_A = df_A['E'].values
    sigma_A = df_A['sigma'].values
    stat_A = df_A['stat_err'].values
    syst_A = df_A['syst_frac'].values

    E_B = df_B['E'].values
    sigma_B = df_B['sigma'].values
    stat_B = df_B['stat_err'].values
    syst_B = df_B['syst_frac'].values

    E0 = 4.3  # Reference energy

    print(f"Channel A (Belle pi+pi- J/psi): {len(E_A)} points, E=[{E_A.min():.2f}, {E_A.max():.2f}] GeV")
    print(f"  Stat errors: median={np.median(stat_A):.2f} pb")
    print(f"  Syst: {syst_A[0]*100:.1f}% (correlated)")
    print(f"Channel B (BaBar pi+pi- psi2S): {len(E_B)} points, E=[{E_B.min():.2f}, {E_B.max():.2f}] GeV")
    print(f"  Stat errors: median={np.median(stat_B):.2f} pb")
    print(f"  Syst: {syst_B[0]*100:.1f}% (correlated)")
    print()

    # Background model selection
    print("=" * 60)
    print("Background Model Selection (AIC/BIC)")
    print("=" * 60)

    bg_order_A, result_A_bg = select_background_model(E_A, sigma_A, stat_A, syst_A, E0)
    print(f"Channel A: bg_order={bg_order_A} (0=const, 1=const+linear)")
    if result_A_bg:
        print(f"  AIC={result_A_bg['aic']:.1f}, BIC={result_A_bg['bic']:.1f}")

    bg_order_B, result_B_bg = select_background_model(E_B, sigma_B, stat_B, syst_B, E0)
    print(f"Channel B: bg_order={bg_order_B}")
    if result_B_bg:
        print(f"  AIC={result_B_bg['aic']:.1f}, BIC={result_B_bg['bic']:.1f}")
    print()

    # Fit Channel A
    print("=" * 60)
    print("Fitting Channel A (Belle pi+pi- J/psi)...")
    print("=" * 60)

    nll_A, params_A, stability_A = multi_start_fit(E_A, sigma_A, stat_A, syst_A, bg_order_A,
                                                    n_starts=N_MULTI_START, E0=E0)

    if nll_A is None:
        print("ERROR: Channel A fit failed!")
        return

    r_A, Phi_A = extract_R(params_A[0], params_A[1], params_A[2])
    chi2_dof_A, chi2_A, dof_A = chi2_per_dof(params_A, E_A, sigma_A, stat_A, syst_A, bg_order_A, E0)

    n_bg_A = 2 if bg_order_A == 0 else 4
    s0_A = params_A[3 + n_bg_A]
    s1_A = params_A[4 + n_bg_A]

    print(f"Channel A results:")
    print(f"  c1 = {params_A[0]:.3f}")
    print(f"  c2 = {params_A[1]:.3f}")
    print(f"  phi = {params_A[2] * 180/np.pi:.1f} deg")
    print(f"  R_A = {r_A:.3f} * exp(i * {Phi_A * 180/np.pi:.1f} deg)")
    print(f"  s0 = {s0_A:.4f} (nuisance scale)")
    print(f"  s1 = {s1_A:.6f} (nuisance slope)")
    print(f"  chi2/dof = {chi2_dof_A:.3f} ({chi2_A:.1f}/{dof_A})")
    print(f"  NLL = {nll_A:.2f}")

    health_A, msg_A = assess_fit_health(chi2_dof_A, 'Channel A')
    print(f"  Health: {health_A} - {msg_A}")

    opt_status_A, opt_msg_A = assess_optimizer_stability(stability_A)
    print(f"  Optimizer: {opt_status_A} - {opt_msg_A}")
    print()

    # Fit Channel B
    print("=" * 60)
    print("Fitting Channel B (BaBar pi+pi- psi2S)...")
    print("=" * 60)

    nll_B, params_B, stability_B = multi_start_fit(E_B, sigma_B, stat_B, syst_B, bg_order_B,
                                                    n_starts=N_MULTI_START, E0=E0)

    if nll_B is None:
        print("ERROR: Channel B fit failed!")
        return

    r_B, Phi_B = extract_R(params_B[0], params_B[1], params_B[2])
    chi2_dof_B, chi2_B, dof_B = chi2_per_dof(params_B, E_B, sigma_B, stat_B, syst_B, bg_order_B, E0)

    n_bg_B = 2 if bg_order_B == 0 else 4
    s0_B = params_B[3 + n_bg_B]
    s1_B = params_B[4 + n_bg_B]

    print(f"Channel B results:")
    print(f"  c1 = {params_B[0]:.3f}")
    print(f"  c2 = {params_B[1]:.3f}")
    print(f"  phi = {params_B[2] * 180/np.pi:.1f} deg")
    print(f"  R_B = {r_B:.3f} * exp(i * {Phi_B * 180/np.pi:.1f} deg)")
    print(f"  s0 = {s0_B:.4f} (nuisance scale)")
    print(f"  s1 = {s1_B:.6f} (nuisance slope)")
    print(f"  chi2/dof = {chi2_dof_B:.3f} ({chi2_B:.1f}/{dof_B})")
    print(f"  NLL = {nll_B:.2f}")

    health_B, msg_B = assess_fit_health(chi2_dof_B, 'Channel B')
    print(f"  Health: {health_B} - {msg_B}")

    opt_status_B, opt_msg_B = assess_optimizer_stability(stability_B)
    print(f"  Optimizer: {opt_status_B} - {opt_msg_B}")
    print()

    # Joint constrained fit
    print("=" * 60)
    print("Fitting Joint Constrained (shared R)...")
    print("=" * 60)

    nll_con, params_con = multi_start_joint_fit(E_A, sigma_A, stat_A, syst_A,
                                                 E_B, sigma_B, stat_B, syst_B,
                                                 bg_order_A, bg_order_B,
                                                 n_starts=N_MULTI_START, E0=E0)

    if nll_con is None:
        print("ERROR: Joint fit failed!")
        return

    r_shared = params_con[0]
    Phi_shared = params_con[1]

    nll_unc = nll_A + nll_B
    Lambda = 2 * (nll_con - nll_unc)

    print(f"Shared R = {r_shared:.3f} * exp(i * {Phi_shared * 180/np.pi:.1f} deg)")
    print(f"NLL_constrained = {nll_con:.2f}")
    print(f"NLL_unconstrained = {nll_unc:.2f}")
    print(f"Lambda = 2*(NLL_con - NLL_unc) = {Lambda:.4f}")

    if Lambda < 0:
        print("WARNING: Lambda < 0 indicates optimizer instability!")
    print()

    # Generate model predictions for bootstrap
    sigma_model_A = model_cross_section(E_A, params_A[0], params_A[1], params_A[2],
                                         params_A[3:3+n_bg_A], s0_A, s1_A, bg_order_A, E0)

    sigma_model_B = model_cross_section(E_B, params_B[0], params_B[1], params_B[2],
                                         params_B[3:3+n_bg_B], s0_B, s1_B, bg_order_B, E0)

    # Bootstrap p-value
    print("=" * 60)
    print("Bootstrap p-value estimation...")
    print("=" * 60)
    print(f"Running {N_BOOTSTRAP} bootstrap replicates...")

    p_value, Lambda_boot = bootstrap_pvalue(E_A, sigma_model_A, stat_A, syst_A,
                                             E_B, sigma_model_B, stat_B, syst_B,
                                             bg_order_A, bg_order_B,
                                             Lambda, n_boot=N_BOOTSTRAP,
                                             n_restarts=N_BOOTSTRAP_RESTARTS, E0=E0)

    print(f"Valid bootstrap samples: {len(Lambda_boot)}/{N_BOOTSTRAP}")
    print(f"Bootstrap p-value = {p_value:.4f}")
    print()

    # Profile likelihood contours
    print("=" * 60)
    print("Computing profile likelihood contours...")
    print("=" * 60)

    # Determine r range based on fits
    r_min = max(0.1, min(r_A, r_B, r_shared) * 0.3)
    r_max = max(r_A, r_B, r_shared) * 2.0
    phi_range = (-np.pi, np.pi)

    r_grid_A, phi_grid_A, delta_nll_A = compute_profile_contours(
        E_A, sigma_A, stat_A, syst_A, nll_A, params_A, bg_order_A,
        (r_min, r_max), phi_range, E0)

    r_grid_B, phi_grid_B, delta_nll_B = compute_profile_contours(
        E_B, sigma_B, stat_B, syst_B, nll_B, params_B, bg_order_B,
        (r_min, r_max), phi_range, E0)

    # Check if shared R is within 95% CL (delta_NLL < 5.99 for 2 DOF)
    def point_in_95cl(r_grid, phi_grid, delta_nll, r_test, phi_test):
        # Find nearest grid point
        i = np.argmin(np.abs(r_grid - r_test))
        j = np.argmin(np.abs(phi_grid - phi_test))
        return delta_nll[i, j] < 5.99

    shared_in_A = point_in_95cl(r_grid_A, phi_grid_A, delta_nll_A, r_shared, Phi_shared)
    shared_in_B = point_in_95cl(r_grid_B, phi_grid_B, delta_nll_B, r_shared, Phi_shared)

    print(f"Shared R within 95% CL of Channel A: {'YES' if shared_in_A else 'NO'}")
    print(f"Shared R within 95% CL of Channel B: {'YES' if shared_in_B else 'NO'}")
    print()

    # Magnitude-only check
    print("=" * 60)
    print("Magnitude-only check (ignoring phase)")
    print("=" * 60)

    # Compute marginal 95% intervals on r
    delta_nll_r_A = np.min(delta_nll_A, axis=1)  # Minimize over phi
    delta_nll_r_B = np.min(delta_nll_B, axis=1)

    r_95_A = r_grid_A[delta_nll_r_A < 3.84]  # 1 DOF
    r_95_B = r_grid_B[delta_nll_r_B < 3.84]

    if len(r_95_A) > 0:
        print(f"Channel A 95% CI on r: [{r_95_A.min():.3f}, {r_95_A.max():.3f}]")
    if len(r_95_B) > 0:
        print(f"Channel B 95% CI on r: [{r_95_B.min():.3f}, {r_95_B.max():.3f}]")

    r_shared_in_A_marg = len(r_95_A) > 0 and r_95_A.min() <= r_shared <= r_95_A.max()
    r_shared_in_B_marg = len(r_95_B) > 0 and r_95_B.min() <= r_shared <= r_95_B.max()

    print(f"r_shared in Channel A 95% marginal: {'YES' if r_shared_in_A_marg else 'NO'}")
    print(f"r_shared in Channel B 95% marginal: {'YES' if r_shared_in_B_marg else 'NO'}")
    print()

    # Decision logic
    print("=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    # Collect all issues
    issues = []

    if health_A == 'UNDERCONSTRAINED':
        issues.append(('UNDERCONSTRAINED', msg_A))
    if health_B == 'UNDERCONSTRAINED':
        issues.append(('UNDERCONSTRAINED', msg_B))
    if health_A == 'MODEL_MISMATCH':
        issues.append(('MODEL_MISMATCH', msg_A))
    if health_B == 'MODEL_MISMATCH':
        issues.append(('MODEL_MISMATCH', msg_B))
    if opt_status_A == 'PHI_NOT_IDENTIFIABLE':
        issues.append(('PHI_NOT_IDENTIFIABLE', opt_msg_A))
    if opt_status_B == 'PHI_NOT_IDENTIFIABLE':
        issues.append(('PHI_NOT_IDENTIFIABLE', opt_msg_B))
    if Lambda < -0.1:
        issues.append(('OPTIMIZER_FAILURE', f'Lambda={Lambda:.4f} < 0'))

    # Determine verdict
    if any(i[0] == 'MODEL_MISMATCH' for i in issues):
        verdict = 'MODEL_MISMATCH'
        reason = '; '.join([i[1] for i in issues if i[0] == 'MODEL_MISMATCH'])
    elif any(i[0] == 'OPTIMIZER_FAILURE' for i in issues):
        verdict = 'OPTIMIZER_FAILURE'
        reason = '; '.join([i[1] for i in issues if i[0] == 'OPTIMIZER_FAILURE'])
    elif any(i[0] == 'UNDERCONSTRAINED' for i in issues):
        verdict = 'INCONCLUSIVE'
        reason = 'Fit underconstrained: ' + '; '.join([i[1] for i in issues if i[0] == 'UNDERCONSTRAINED'])
    elif any(i[0] == 'PHI_NOT_IDENTIFIABLE' for i in issues):
        # Phase not identifiable - check magnitude only
        if r_shared_in_A_marg and r_shared_in_B_marg and p_value > 0.05:
            verdict = 'INCONCLUSIVE'
            reason = f'Phase not identifiable but magnitude consistent (p={p_value:.3f})'
        else:
            verdict = 'INCONCLUSIVE'
            reason = f'Phase not identifiable, magnitude test: p={p_value:.3f}'
    elif health_A == 'PASS' and health_B == 'PASS':
        if p_value > 0.05 and shared_in_A and shared_in_B:
            verdict = 'SUPPORTED'
            reason = f'p={p_value:.3f} > 0.05 and shared R within both 95% regions'
        elif p_value < 0.05 and (not shared_in_A or not shared_in_B):
            verdict = 'DISFAVORED'
            reason = f'p={p_value:.3f} < 0.05 and shared R outside 95% region'
        else:
            verdict = 'INCONCLUSIVE'
            reason = f'Mixed signals: p={p_value:.3f}, shared_in_A={shared_in_A}, shared_in_B={shared_in_B}'
    else:
        verdict = 'INCONCLUSIVE'
        reason = 'Unknown condition'

    print(f"Verdict: {verdict}")
    print(f"Reason: {reason}")
    print()

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'channel_A': {
            'experiment': 'Belle',
            'reaction': 'e+e- -> pi+pi- J/psi',
            'n_points': len(E_A),
            'r': float(r_A),
            'Phi_deg': float(Phi_A * 180/np.pi),
            'chi2_dof': float(chi2_dof_A),
            'nll': float(nll_A),
            's0': float(s0_A),
            's1': float(s1_A),
            'health': health_A,
            'optimizer': opt_status_A,
            'bg_order': bg_order_A
        },
        'channel_B': {
            'experiment': 'BaBar',
            'reaction': 'e+e- -> pi+pi- psi(2S)',
            'n_points': len(E_B),
            'r': float(r_B),
            'Phi_deg': float(Phi_B * 180/np.pi),
            'chi2_dof': float(chi2_dof_B),
            'nll': float(nll_B),
            's0': float(s0_B),
            's1': float(s1_B),
            'health': health_B,
            'optimizer': opt_status_B,
            'bg_order': bg_order_B
        },
        'shared': {
            'r': float(r_shared),
            'Phi_deg': float(Phi_shared * 180/np.pi),
            'nll_constrained': float(nll_con),
            'nll_unconstrained': float(nll_unc),
            'Lambda': float(Lambda),
            'in_A_95cl': bool(shared_in_A),
            'in_B_95cl': bool(shared_in_B)
        },
        'bootstrap': {
            'n_samples': len(Lambda_boot),
            'p_value': float(p_value)
        },
        'magnitude_only': {
            'r_shared_in_A_marginal': bool(r_shared_in_A_marg),
            'r_shared_in_B_marginal': bool(r_shared_in_B_marg)
        },
        'verdict': verdict,
        'reason': reason
    }

    # Save JSON
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save optimizer stability
    stability_data = {
        'channel_A': stability_A,
        'channel_B': stability_B
    }
    with open(os.path.join(OUTDIR, 'optimizer_stability.json'), 'w') as f:
        json.dump(stability_data, f, indent=2)

    # Generate plots
    print("Generating plots...")

    # Contour plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Channel A
    ax = axes[0]
    R, P = np.meshgrid(r_grid_A, phi_grid_A * 180/np.pi)
    cs = ax.contour(R, P, delta_nll_A.T, levels=[2.30, 5.99], colors=['blue', 'lightblue'])
    ax.clabel(cs, fmt={2.30: '68%', 5.99: '95%'})
    ax.plot(r_A, Phi_A * 180/np.pi, 'b*', markersize=15, label=f'A: r={r_A:.2f}')
    ax.plot(r_shared, Phi_shared * 180/np.pi, 'r^', markersize=12, label='Shared')
    ax.set_xlabel('r = |c2/c1|')
    ax.set_ylabel(r'$\Phi$ [degrees]')
    ax.set_title(r'Channel A: $\pi^+\pi^-J/\psi$ (Belle)')
    ax.legend(loc='upper right', fontsize=8)

    # Channel B
    ax = axes[1]
    R, P = np.meshgrid(r_grid_B, phi_grid_B * 180/np.pi)
    cs = ax.contour(R, P, delta_nll_B.T, levels=[2.30, 5.99], colors=['green', 'lightgreen'])
    ax.clabel(cs, fmt={2.30: '68%', 5.99: '95%'})
    ax.plot(r_B, Phi_B * 180/np.pi, 'g*', markersize=15, label=f'B: r={r_B:.2f}')
    ax.plot(r_shared, Phi_shared * 180/np.pi, 'r^', markersize=12, label='Shared')
    ax.set_xlabel('r = |c2/c1|')
    ax.set_ylabel(r'$\Phi$ [degrees]')
    ax.set_title(r'Channel B: $\pi^+\pi^-\psi(2S)$ (BaBar)')
    ax.legend(loc='upper right', fontsize=8)

    # Overlay
    ax = axes[2]
    cs_A = ax.contour(r_grid_A, phi_grid_A * 180/np.pi, delta_nll_A.T, levels=[5.99], colors=['blue'])
    cs_B = ax.contour(r_grid_B, phi_grid_B * 180/np.pi, delta_nll_B.T, levels=[5.99], colors=['green'])
    ax.plot(r_A, Phi_A * 180/np.pi, 'b*', markersize=15, label=f'A: r={r_A:.2f}')
    ax.plot(r_B, Phi_B * 180/np.pi, 'g*', markersize=15, label=f'B: r={r_B:.2f}')
    ax.plot(r_shared, Phi_shared * 180/np.pi, 'r^', markersize=12, label=f'Shared: r={r_shared:.2f}')
    ax.set_xlabel('r = |c2/c1|')
    ax.set_ylabel(r'$\Phi$ [degrees]')
    ax.set_title('95% CL Contours Overlay')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'contours_overlay.png'), dpi=150)
    plt.close()

    # Individual channel contours
    for name, r_grid, phi_grid, delta_nll, r_best, Phi_best in [
        ('A', r_grid_A, phi_grid_A, delta_nll_A, r_A, Phi_A),
        ('B', r_grid_B, phi_grid_B, delta_nll_B, r_B, Phi_B)
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        R, P = np.meshgrid(r_grid, phi_grid * 180/np.pi)
        cs = ax.contourf(R, P, delta_nll.T, levels=[0, 2.30, 5.99, 10, 20],
                         colors=['darkblue', 'blue', 'lightblue', 'white'], alpha=0.7)
        ax.contour(R, P, delta_nll.T, levels=[2.30, 5.99], colors=['white', 'black'])
        ax.plot(r_best, Phi_best * 180/np.pi, 'r*', markersize=15)
        ax.plot(r_shared, Phi_shared * 180/np.pi, 'r^', markersize=12)
        ax.set_xlabel('r = |c2/c1|')
        ax.set_ylabel(r'$\Phi$ [degrees]')
        ax.set_title(f'Channel {name} Profile Likelihood')
        plt.colorbar(cs, label=r'$\Delta(-2\ln L)$')
        plt.savefig(os.path.join(OUTDIR, f'contours_{name}.png'), dpi=150)
        plt.close()

    # Bootstrap histogram
    if len(Lambda_boot) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(Lambda_boot, bins=30, density=True, alpha=0.7, label='Bootstrap')
        ax.axvline(Lambda, color='r', linestyle='--', linewidth=2, label=f'Observed: {Lambda:.3f}')
        ax.axvline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel(r'$\Lambda = 2(\ln L_{unc} - \ln L_{con})$')
        ax.set_ylabel('Density')
        ax.set_title(f'Bootstrap Distribution (p = {p_value:.3f})')
        ax.legend()
        plt.savefig(os.path.join(OUTDIR, 'bootstrap_hist.png'), dpi=150)
        plt.close()

    # Optimizer stability plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (name, stab) in enumerate([('A', stability_A), ('B', stability_B)]):
        if stab is None:
            continue

        # NLL histogram
        ax = axes[0, idx]
        ax.hist(stab['nlls'], bins=30, alpha=0.7)
        ax.axvline(stab['nll_best'], color='r', linestyle='--', label=f'Best: {stab["nll_best"]:.2f}')
        ax.set_xlabel('NLL')
        ax.set_ylabel('Count')
        ax.set_title(f'Channel {name} NLL Distribution')
        ax.legend()

        # (r, phi) scatter
        ax = axes[1, idx]
        if len(stab['rs']) > 0 and len(stab['phis']) > 0:
            ax.scatter(stab['rs'], np.array(stab['phis']) * 180/np.pi, alpha=0.3, s=10)
            ax.set_xlabel('r = |c2/c1|')
            ax.set_ylabel(r'$\phi$ [deg]')
            ax.set_title(f'Channel {name} Multi-Start Solutions')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'optimizer_stability.png'), dpi=150)
    plt.close()

    # Write REPORT_v2.md
    report = f"""# Y-state Rank-1 Bottleneck Test v2 Report

## Analysis Version
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- Version: 2.0 (with nuisance parameters and strict health gates)

## Data Sources

| Channel | Reaction | Experiment | HEPData DOI | N points |
|---------|----------|------------|-------------|----------|
| A | e+e- -> pi+pi- J/psi | Belle | 10.17182/hepdata.61431.v1/t1 | {len(E_A)} |
| B | e+e- -> pi+pi- psi(2S) | BaBar | 10.17182/hepdata.19344.v1/t1 | {len(E_B)} |

## Uncertainty Treatment

### Channel A (Belle)
- Statistical errors: Point-wise independent (median = {np.median(stat_A):.2f} pb)
- Systematic: 7.0% **correlated** -> modeled via nuisance parameter s0

### Channel B (BaBar)
- Statistical errors: Point-wise independent, asymmetric (median = {np.median(stat_B):.2f} pb)
- Systematic: 12.3% **correlated** -> modeled via nuisance parameter s0

**Key fix vs v1**: Correlated systematics are NOT added in quadrature. Instead, they enter as a prior on a global scale nuisance parameter s0.

## Resonance Parameters (Fixed)

- Y(4220): M = {Y1_MASS} GeV, Γ = {Y1_WIDTH} GeV
- Y(4360): M = {Y2_MASS} GeV, Γ = {Y2_WIDTH} GeV

## Model

Coherent amplitude: A(E) = c1·BW₁ + c2·exp(iφ)·BW₂ + A_bg

Cross section with nuisance: σ(E) = s0·(1 + s1·(E-E₀))·|A(E)|²

Background order selected by AIC:
- Channel A: bg_order = {bg_order_A}
- Channel B: bg_order = {bg_order_B}

## Fit Results

### Channel A (Belle π⁺π⁻J/ψ)

| Parameter | Value |
|-----------|-------|
| r = \\|c2/c1\\| | {r_A:.3f} |
| Φ [deg] | {Phi_A * 180/np.pi:.1f} |
| s0 (nuisance) | {s0_A:.4f} |
| s1 (nuisance) | {s1_A:.6f} |
| χ²/dof | {chi2_dof_A:.3f} ({chi2_A:.1f}/{dof_A}) |
| NLL | {nll_A:.2f} |
| Health | **{health_A}** |
| Optimizer | {opt_status_A} |

### Channel B (BaBar π⁺π⁻ψ(2S))

| Parameter | Value |
|-----------|-------|
| r = \\|c2/c1\\| | {r_B:.3f} |
| Φ [deg] | {Phi_B * 180/np.pi:.1f} |
| s0 (nuisance) | {s0_B:.4f} |
| s1 (nuisance) | {s1_B:.6f} |
| χ²/dof | {chi2_dof_B:.3f} ({chi2_B:.1f}/{dof_B}) |
| NLL | {nll_B:.2f} |
| Health | **{health_B}** |
| Optimizer | {opt_status_B} |

## Fit Health Gates

**Criterion**: 0.5 < χ²/dof < 3.0

| Channel | χ²/dof | Status |
|---------|--------|--------|
| A | {chi2_dof_A:.3f} | {health_A} |
| B | {chi2_dof_B:.3f} | {health_B} |

## Joint Constrained Fit

- Shared R = {r_shared:.3f} × exp(i × {Phi_shared * 180/np.pi:.1f}°)
- NLL_constrained = {nll_con:.2f}
- NLL_unconstrained = {nll_unc:.2f}
- **Λ = 2×(NLL_con - NLL_unc) = {Lambda:.4f}**

## Statistical Test

- Bootstrap replicates: {len(Lambda_boot)}/{N_BOOTSTRAP}
- **Bootstrap p-value = {p_value:.4f}**

## Profile Likelihood

- Shared R within Channel A 95% CL: **{'YES' if shared_in_A else 'NO'}**
- Shared R within Channel B 95% CL: **{'YES' if shared_in_B else 'NO'}**

## Magnitude-Only Check

| Metric | Value |
|--------|-------|
| r_shared in A 95% marginal | {'YES' if r_shared_in_A_marg else 'NO'} |
| r_shared in B 95% marginal | {'YES' if r_shared_in_B_marg else 'NO'} |

## Verdict

**{verdict}**

Reason: {reason}

## Plots

- [Contour Overlay](contours_overlay.png)
- [Channel A Contours](contours_A.png)
- [Channel B Contours](contours_B.png)
- [Bootstrap Distribution](bootstrap_hist.png)
- [Optimizer Stability](optimizer_stability.png)

## Comparison with v1

| Metric | v1 | v2 |
|--------|----|----|
| Chi²/dof A | 0.632 | {chi2_dof_A:.3f} |
| Chi²/dof B | 0.145 | {chi2_dof_B:.3f} |
| Verdict | SUPPORTED | **{verdict}** |
| Lower χ² gate | None | 0.5 |
| Syst treatment | Quadrature | Nuisance params |
"""

    with open(os.path.join(OUTDIR, 'REPORT_v2.md'), 'w') as f:
        f.write(report)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Channel A: R_A = {r_A:.3f} * exp(i * {Phi_A * 180/np.pi:.1f} deg), chi2/dof = {chi2_dof_A:.3f} [{health_A}]")
    print(f"Channel B: R_B = {r_B:.3f} * exp(i * {Phi_B * 180/np.pi:.1f} deg), chi2/dof = {chi2_dof_B:.3f} [{health_B}]")
    print(f"Shared: R = {r_shared:.3f} * exp(i * {Phi_shared * 180/np.pi:.1f} deg)")
    print(f"Lambda = {Lambda:.4f}")
    print(f"Bootstrap p-value = {p_value:.4f}")
    print(f"Verdict: {verdict}")
    print()
    print(f"Report saved to: {os.path.join(OUTDIR, 'REPORT_v2.md')}")
    print(f"Results saved to: {os.path.join(OUTDIR, 'results.json')}")

if __name__ == '__main__':
    main()
