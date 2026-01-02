#!/usr/bin/env python3
"""
Generic Rank-1 Bottleneck Test Runner
======================================

This script runs a rank-1 bottleneck test for a single model configuration.
It follows the rigorous standards established in the CMS/ATLAS tests:
- No plot digitization (HEPData numeric tables only)
- Constrained vs unconstrained likelihood-ratio test
- Bootstrap p-value estimation
- Optimizer stability audit
- Fit health gates

Usage:
    python run_model.py --model <model_name>
    python run_model.py --config <path_to_config.yaml>
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2 as chi2_dist
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import json
import os
import sys
import argparse
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTS
# ==============================================================================

BASE_DIR = Path(__file__).parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
RUNS_DIR = BASE_DIR / "runs"
OUT_DIR = BASE_DIR / "out"
REQUESTS_DIR = OUT_DIR / "requests"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for d in [RUNS_DIR, OUT_DIR, REQUESTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Default optimization settings
DEFAULT_N_STARTS = 300
DEFAULT_N_BOOTSTRAP = 800
DEFAULT_N_WORKERS = max(1, cpu_count() - 1)

# Contour grid defaults
R_GRID = np.linspace(0.01, 5, 50)
PHI_GRID = np.linspace(-np.pi, np.pi, 72)

# ==============================================================================
# HEPDATA INTEGRATION
# ==============================================================================

def fetch_hepdata_record(record_id):
    """
    Fetch HEPData record metadata.

    Args:
        record_id: INSPIRE record ID (e.g., "ins892684")

    Returns:
        dict: Record metadata including available tables
    """
    url = f"https://www.hepdata.net/record/{record_id}?format=json"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  ERROR: Could not fetch HEPData record {record_id}: {e}")
        return None


def download_hepdata_table(record_id, table_number, output_path):
    """
    Download a specific table from HEPData as CSV.

    Args:
        record_id: INSPIRE record ID
        table_number: Table number (1-indexed)
        output_path: Path to save CSV

    Returns:
        bool: True if successful
    """
    url = f"https://www.hepdata.net/download/table/{record_id}/Table%20{table_number}/csv"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, 'w') as f:
            f.write(response.text)

        return True
    except Exception as e:
        print(f"  ERROR: Could not download table {table_number}: {e}")
        return False


def parse_hepdata_csv(filepath):
    """
    Parse HEPData CSV file into standardized format.

    Handles multiple HEPData CSV formats including:
    - SQRT(S) [GEV],SIG [NB],error +,error -
    - E,sigma,stat+,stat-,syst+,syst-

    Returns:
        DataFrame with columns: E_gev, y, stat, syst (optional), total_err
    """
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')

    # Find header row (contains column names)
    header_idx = None
    for i, line in enumerate(lines):
        # Skip comment lines
        if line.startswith('#'):
            continue
        # Check for typical header keywords
        line_upper = line.upper()
        if 'SQRT' in line_upper or ('GEV' in line_upper and ',' in line):
            header_idx = i
            break

    if header_idx is None:
        # Try to find first non-comment line
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#'):
                header_idx = i
                break

    if header_idx is None:
        raise ValueError(f"Could not find header in {filepath}")

    # Parse header to understand column layout
    header_line = lines[header_idx]
    header_parts = header_line.split(',')

    # Determine column indices based on header
    e_col = 0  # Energy is typically first
    y_col = 1  # Value is typically second
    err_plus_col = 2  # error + is typically third
    err_minus_col = 3  # error - is typically fourth

    # Parse data
    data = []
    for line in lines[header_idx + 1:]:
        if not line.strip() or line.startswith('#'):
            continue

        parts = line.split(',')
        if len(parts) < 3:
            continue

        try:
            E = float(parts[e_col])
            y = float(parts[y_col])

            # Handle asymmetric errors (err+, err-)
            if len(parts) >= 4:
                err_plus = abs(float(parts[err_plus_col]))
                err_minus = abs(float(parts[err_minus_col]))
                stat = (err_plus + err_minus) / 2
            elif len(parts) >= 3:
                stat = abs(float(parts[2]))
            else:
                stat = 0.1 * abs(y) if y != 0 else 0.01

            # Systematic error if available (columns 4-5 or 5-6)
            syst = 0.0
            if len(parts) >= 6:
                try:
                    syst_plus = abs(float(parts[4]))
                    syst_minus = abs(float(parts[5]))
                    syst = (syst_plus + syst_minus) / 2
                except:
                    pass

            if y >= 0 and stat > 0:  # Valid data point
                data.append([E, y, stat, syst])

        except (ValueError, IndexError):
            continue

    if len(data) == 0:
        raise ValueError(f"No valid data points parsed from {filepath}")

    df = pd.DataFrame(data, columns=['E_gev', 'y', 'stat', 'syst'])
    df['total_err'] = np.sqrt(df['stat']**2 + df['syst']**2)

    return df


# ==============================================================================
# PHYSICS MODEL
# ==============================================================================

def breit_wigner(E, M, Gamma):
    """Relativistic Breit-Wigner amplitude."""
    s = E**2
    return 1.0 / (s - M**2 + 1j * M * Gamma)


def coherent_amplitude(E, c1, c2, phi, bg_re, bg_im, M1, W1, M2, W2,
                       bg1_re=0, bg1_im=0, E0=1.8, bg_order=0):
    """
    Coherent sum of two BW resonances plus complex background.

    A(E) = c1*BW1 + c2*exp(iφ)*BW2 + background

    Args:
        E: Energy array
        c1, c2: Amplitude coefficients (c1 real-positive)
        phi: Relative phase
        bg_re, bg_im: Constant background (real/imag)
        M1, W1, M2, W2: BW parameters
        bg1_re, bg1_im: Linear background (optional)
        E0: Reference energy for background
        bg_order: 0 = constant, 1 = linear
    """
    E = np.atleast_1d(E)

    BW1 = breit_wigner(E, M1, W1)
    BW2 = breit_wigner(E, M2, W2)

    # Resonance amplitude
    A_res = c1 * BW1 + c2 * np.exp(1j * phi) * BW2

    # Background
    bg = bg_re + 1j * bg_im
    if bg_order >= 1:
        bg = bg + (bg1_re + 1j * bg1_im) * (E - E0)

    return A_res + bg


def cross_section_model(E, params, M1, W1, M2, W2, bg_order=0):
    """
    Cross section model with nuisance scaling.

    σ(E) = s0 * |A(E)|²

    params layout (bg_order=0):
        [c1, c2, phi, bg_re, bg_im, s0]

    params layout (bg_order=1):
        [c1, c2, phi, bg_re, bg_im, bg1_re, bg1_im, s0]
    """
    c1 = params[0]
    c2 = params[1]
    phi = params[2]
    bg_re = params[3]
    bg_im = params[4]

    if bg_order == 0:
        bg1_re, bg1_im = 0, 0
        s0 = params[5]
    else:
        bg1_re = params[5]
        bg1_im = params[6]
        s0 = params[7]

    A = coherent_amplitude(E, c1, c2, phi, bg_re, bg_im, M1, W1, M2, W2,
                          bg1_re, bg1_im, bg_order=bg_order)
    sigma = s0 * np.abs(A)**2

    return np.maximum(sigma, 1e-10)


def gaussian_nll(params, E_data, y_data, y_err, syst_frac, M1, W1, M2, W2, bg_order):
    """
    Gaussian negative log-likelihood for cross section data.

    NLL = 0.5 * sum((y - model)² / err²) + prior(s0)
    """
    model = cross_section_model(E_data, params, M1, W1, M2, W2, bg_order)

    chi2 = np.sum((y_data - model)**2 / y_err**2)

    # Nuisance prior on scale factor
    s0 = params[-1]
    prior = (s0 - 1)**2 / (2 * syst_frac**2)

    return 0.5 * chi2 + prior


def poisson_nll(params, E_data, counts, bin_width, M1, W1, M2, W2, bg_order):
    """
    Poisson negative log-likelihood for count data.

    NLL = sum(mu - n*log(mu)) + priors
    """
    mu = cross_section_model(E_data, params, M1, W1, M2, W2, bg_order) * bin_width
    mu = np.maximum(mu, 1e-10)

    nll = np.sum(mu - counts * np.log(mu))

    # Nuisance prior
    s0 = params[-1]
    prior = (s0 - 1)**2 / (2 * 0.05**2)  # 5% scale uncertainty

    return nll + prior


# ==============================================================================
# OPTIMIZATION
# ==============================================================================

def get_bounds(bg_order):
    """Get parameter bounds for optimization."""
    bounds = [
        (0.01, 20),       # c1 (amplitude)
        (0.01, 20),       # c2 (amplitude)
        (-np.pi, np.pi),  # phi (phase)
        (-5, 5),          # bg_re
        (-5, 5),          # bg_im
    ]
    if bg_order >= 1:
        bounds.extend([(-2, 2), (-2, 2)])  # bg1_re, bg1_im
    bounds.append((0.5, 1.5))  # s0 (nuisance scale)

    return bounds


def fit_channel(E_data, y_data, y_err, syst_frac, M1, W1, M2, W2,
                bg_order=0, likelihood_type='gaussian', n_starts=DEFAULT_N_STARTS):
    """
    Fit a single channel using multi-start optimization.

    Returns:
        best_params, best_nll, all_results
    """
    bounds = get_bounds(bg_order)

    if likelihood_type == 'gaussian':
        def objective(p):
            return gaussian_nll(p, E_data, y_data, y_err, syst_frac, M1, W1, M2, W2, bg_order)
    else:
        # For Poisson, y_data should be counts
        bin_width = (E_data[1] - E_data[0]) if len(E_data) > 1 else 0.01
        def objective(p):
            return poisson_nll(p, E_data, y_data, bin_width, M1, W1, M2, W2, bg_order)

    # Primary: Differential Evolution (global)
    try:
        result_de = differential_evolution(
            objective, bounds,
            maxiter=2000, tol=1e-8, seed=42,
            workers=1, updating='deferred', polish=True
        )
        best_params = result_de.x
        best_nll = result_de.fun
    except Exception as e:
        print(f"  DE failed: {e}")
        best_params = None
        best_nll = np.inf

    all_results = []
    if best_params is not None:
        all_results.append((best_params.copy(), best_nll))

    # Multi-start L-BFGS-B refinement
    rng = np.random.default_rng(42)
    for i in range(min(100, n_starts)):
        x0 = [rng.uniform(b[0], b[1]) for b in bounds]
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000})
            if np.isfinite(res.fun):
                all_results.append((res.x.copy(), res.fun))
                if res.fun < best_nll:
                    best_params = res.x.copy()
                    best_nll = res.fun
        except:
            pass

    # Also try Powell
    for i in range(min(50, n_starts // 2)):
        x0 = [rng.uniform(b[0], b[1]) for b in bounds]
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 1000})
            if np.isfinite(res.fun):
                # Clip to bounds
                x_clipped = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
                nll_clipped = objective(x_clipped)
                all_results.append((x_clipped.copy(), nll_clipped))
                if nll_clipped < best_nll:
                    best_params = x_clipped.copy()
                    best_nll = nll_clipped
        except:
            pass

    return best_params, best_nll, all_results


def extract_R(params):
    """Extract R = c2*exp(iφ)/c1 from parameters."""
    c1 = params[0]
    c2 = params[1]
    phi = params[2]
    r = c2 / max(c1, 1e-10)
    return r, phi


def compute_chi2_dof(params, E_data, y_data, y_err, M1, W1, M2, W2, bg_order):
    """Compute chi-squared per degree of freedom."""
    model = cross_section_model(E_data, params, M1, W1, M2, W2, bg_order)
    chi2 = np.sum((y_data - model)**2 / y_err**2)
    n_params = 6 if bg_order == 0 else 8
    dof = max(1, len(E_data) - n_params)
    return chi2, dof, chi2 / dof


# ==============================================================================
# JOINT CONSTRAINED FIT
# ==============================================================================

def joint_nll(shared_params, fixed_A, fixed_B,
              E_A, y_A, err_A, syst_A, M1, W1, M2, W2, bg_order_A, likelihood_A,
              E_B, y_B, err_B, syst_B, bg_order_B, likelihood_B):
    """
    Joint NLL with shared R = (r, phi).

    shared_params = [r_shared, phi_shared]
    """
    r_shared = shared_params[0]
    phi_shared = shared_params[1]

    # Reconstruct params for A
    c1_A = fixed_A[0]
    c2_A = r_shared * c1_A
    params_A = [c1_A, c2_A, phi_shared] + list(fixed_A[3:])

    # Reconstruct params for B
    c1_B = fixed_B[0]
    c2_B = r_shared * c1_B
    params_B = [c1_B, c2_B, phi_shared] + list(fixed_B[3:])

    if likelihood_A == 'gaussian':
        nll_A = gaussian_nll(params_A, E_A, y_A, err_A, syst_A, M1, W1, M2, W2, bg_order_A)
    else:
        bin_width_A = (E_A[1] - E_A[0]) if len(E_A) > 1 else 0.01
        nll_A = poisson_nll(params_A, E_A, y_A, bin_width_A, M1, W1, M2, W2, bg_order_A)

    if likelihood_B == 'gaussian':
        nll_B = gaussian_nll(params_B, E_B, y_B, err_B, syst_B, M1, W1, M2, W2, bg_order_B)
    else:
        bin_width_B = (E_B[1] - E_B[0]) if len(E_B) > 1 else 0.01
        nll_B = poisson_nll(params_B, E_B, y_B, bin_width_B, M1, W1, M2, W2, bg_order_B)

    return nll_A + nll_B


def fit_joint_constrained(params_A, params_B, config,
                          E_A, y_A, err_A, syst_A,
                          E_B, y_B, err_B, syst_B,
                          M1, W1, M2, W2):
    """Fit with shared R constraint."""
    r_A, phi_A = extract_R(params_A)
    r_B, phi_B = extract_R(params_B)

    bg_order = config['background'].get('order', 0)
    likelihood = config['likelihood_type']

    bounds = [(0.01, 10), (-np.pi, np.pi)]

    best_nll = np.inf
    best_params = [(r_A + r_B) / 2, (phi_A + phi_B) / 2]

    rng = np.random.default_rng(42)
    for i in range(100):
        x0 = [rng.uniform(0.1, 5), rng.uniform(-np.pi, np.pi)]
        try:
            result = minimize(
                joint_nll, x0,
                args=(params_A, params_B, E_A, y_A, err_A, syst_A,
                      M1, W1, M2, W2, bg_order, likelihood,
                      E_B, y_B, err_B, syst_B, bg_order, likelihood),
                method='L-BFGS-B', bounds=bounds
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except:
            pass

    return best_params, best_nll


# ==============================================================================
# BOOTSTRAP
# ==============================================================================

def bootstrap_worker(args):
    """Worker for parallel bootstrap."""
    (seed, E_A, y_A, err_A, syst_A, bg_order_A, likelihood_A,
     E_B, y_B, err_B, syst_B, bg_order_B, likelihood_B,
     model_A, model_B, params_A, params_B,
     M1, W1, M2, W2) = args

    rng = np.random.default_rng(seed)

    # Generate pseudo-data from constrained model
    if likelihood_A == 'gaussian':
        y_A_boot = model_A + rng.normal(0, err_A)
        y_A_boot = np.maximum(y_A_boot, 0)
    else:
        y_A_boot = rng.poisson(model_A)

    if likelihood_B == 'gaussian':
        y_B_boot = model_B + rng.normal(0, err_B)
        y_B_boot = np.maximum(y_B_boot, 0)
    else:
        y_B_boot = rng.poisson(model_B)

    bounds_A = get_bounds(bg_order_A)
    bounds_B = get_bounds(bg_order_B)

    try:
        # Fit individual channels
        if likelihood_A == 'gaussian':
            def obj_A(p):
                return gaussian_nll(p, E_A, y_A_boot, err_A, syst_A, M1, W1, M2, W2, bg_order_A)
        else:
            bin_width_A = (E_A[1] - E_A[0]) if len(E_A) > 1 else 0.01
            def obj_A(p):
                return poisson_nll(p, E_A, y_A_boot, bin_width_A, M1, W1, M2, W2, bg_order_A)

        if likelihood_B == 'gaussian':
            def obj_B(p):
                return gaussian_nll(p, E_B, y_B_boot, err_B, syst_B, M1, W1, M2, W2, bg_order_B)
        else:
            bin_width_B = (E_B[1] - E_B[0]) if len(E_B) > 1 else 0.01
            def obj_B(p):
                return poisson_nll(p, E_B, y_B_boot, bin_width_B, M1, W1, M2, W2, bg_order_B)

        res_A = minimize(obj_A, params_A, method='L-BFGS-B', bounds=bounds_A,
                        options={'maxiter': 500})
        res_B = minimize(obj_B, params_B, method='L-BFGS-B', bounds=bounds_B,
                        options={'maxiter': 500})

        if not (np.isfinite(res_A.fun) and np.isfinite(res_B.fun)):
            return None

        nll_unc = res_A.fun + res_B.fun

        # Constrained fit
        r_A, phi_A = extract_R(res_A.x)
        r_B, phi_B = extract_R(res_B.x)

        bounds_shared = [(0.01, 10), (-np.pi, np.pi)]

        def joint_obj(shared):
            r, phi = shared
            c1_A, c2_A = res_A.x[0], r * res_A.x[0]
            c1_B, c2_B = res_B.x[0], r * res_B.x[0]
            p_A = [c1_A, c2_A, phi] + list(res_A.x[3:])
            p_B = [c1_B, c2_B, phi] + list(res_B.x[3:])
            return obj_A(p_A) + obj_B(p_B)

        x0 = [(r_A + r_B) / 2, (phi_A + phi_B) / 2]
        res_con = minimize(joint_obj, x0, method='L-BFGS-B', bounds=bounds_shared)

        Lambda = 2 * (res_con.fun - nll_unc)
        return Lambda

    except Exception:
        return None


def run_bootstrap(E_A, y_A, err_A, syst_A, bg_order_A, likelihood_A,
                  E_B, y_B, err_B, syst_B, bg_order_B, likelihood_B,
                  params_A, params_B, shared_R,
                  M1, W1, M2, W2, n_bootstrap=DEFAULT_N_BOOTSTRAP):
    """Run bootstrap to estimate p-value."""

    r_shared, phi_shared = shared_R

    # Get constrained model predictions
    c1_A, c2_A = params_A[0], r_shared * params_A[0]
    params_A_con = [c1_A, c2_A, phi_shared] + list(params_A[3:])
    model_A = cross_section_model(E_A, params_A_con, M1, W1, M2, W2, bg_order_A)

    c1_B, c2_B = params_B[0], r_shared * params_B[0]
    params_B_con = [c1_B, c2_B, phi_shared] + list(params_B[3:])
    model_B = cross_section_model(E_B, params_B_con, M1, W1, M2, W2, bg_order_B)

    args_list = [
        (i, E_A, y_A, err_A, syst_A, bg_order_A, likelihood_A,
         E_B, y_B, err_B, syst_B, bg_order_B, likelihood_B,
         model_A, model_B, params_A, params_B, M1, W1, M2, W2)
        for i in range(n_bootstrap)
    ]

    print(f"  Running {n_bootstrap} bootstrap replicates with {DEFAULT_N_WORKERS} workers...")

    with Pool(DEFAULT_N_WORKERS) as pool:
        results = pool.map(bootstrap_worker, args_list)

    valid = [r for r in results if r is not None]
    print(f"  Valid bootstrap samples: {len(valid)}/{n_bootstrap}")

    return np.array(valid)


# ==============================================================================
# PROFILE LIKELIHOOD CONTOURS
# ==============================================================================

def compute_profile_contour(E_data, y_data, y_err, syst_frac, bg_order, likelihood,
                            best_params, M1, W1, M2, W2, r_grid, phi_grid):
    """Compute profile likelihood over (r, φ) grid."""
    nll_grid = np.zeros((len(r_grid), len(phi_grid)))
    bounds = get_bounds(bg_order)

    if likelihood == 'gaussian':
        def base_nll(p):
            return gaussian_nll(p, E_data, y_data, y_err, syst_frac, M1, W1, M2, W2, bg_order)
    else:
        bin_width = (E_data[1] - E_data[0]) if len(E_data) > 1 else 0.01
        def base_nll(p):
            return poisson_nll(p, E_data, y_data, bin_width, M1, W1, M2, W2, bg_order)

    for i, r in enumerate(r_grid):
        for j, phi in enumerate(phi_grid):
            c1 = best_params[0]
            c2 = r * c1

            def nll_fixed_R(other_params):
                full_params = [c1, c2, phi] + list(other_params)
                return base_nll(full_params)

            other_bounds = bounds[3:]
            other_x0 = list(best_params[3:])

            try:
                res = minimize(nll_fixed_R, other_x0, method='L-BFGS-B',
                              bounds=other_bounds, options={'maxiter': 200})
                nll_grid[i, j] = res.fun
            except:
                nll_grid[i, j] = np.inf

    return nll_grid


def check_point_in_contour(delta_chi2_grid, r_grid, phi_grid, r, phi, level=5.99):
    """Check if point (r, phi) is within contour level."""
    r_idx = np.argmin(np.abs(r_grid - r))
    phi_idx = np.argmin(np.abs(phi_grid - phi))
    return delta_chi2_grid[r_idx, phi_idx] < level


# ==============================================================================
# OPTIMIZER STABILITY AUDIT
# ==============================================================================

def optimizer_stability_audit(all_results_A, all_results_B, r_grid, phi_grid):
    """Audit optimizer stability and check for multimodality."""

    nlls_A = np.array([r[1] for r in all_results_A if r is not None])
    nlls_B = np.array([r[1] for r in all_results_B if r is not None])

    if len(nlls_A) == 0 or len(nlls_B) == 0:
        return False, False, {}

    best_A = np.min(nlls_A)
    best_B = np.min(nlls_B)

    # Near-optimal solutions (ΔNLL < 2)
    near_opt_A = [(r[0], r[1]) for r in all_results_A if r is not None and r[1] - best_A < 2]
    near_opt_B = [(r[0], r[1]) for r in all_results_B if r is not None and r[1] - best_B < 2]

    r_values_A = [extract_R(p[0])[0] for p in near_opt_A]
    phi_values_A = [extract_R(p[0])[1] for p in near_opt_A]
    r_values_B = [extract_R(p[0])[0] for p in near_opt_B]
    phi_values_B = [extract_R(p[0])[1] for p in near_opt_B]

    # Check for multimodality
    r_range_A = max(r_values_A) / max(min(r_values_A), 0.01) if r_values_A else 1
    r_range_B = max(r_values_B) / max(min(r_values_B), 0.01) if r_values_B else 1

    phi_std_A = np.std(phi_values_A) if len(phi_values_A) > 1 else 0
    phi_std_B = np.std(phi_values_B) if len(phi_values_B) > 1 else 0

    phase_identifiable = (phi_std_A < 1.0) and (phi_std_B < 1.0)  # < ~60 degrees
    r_stable = (r_range_A < 10) and (r_range_B < 10)

    details = {
        'channel_A': {
            'n_near_optimal': len(near_opt_A),
            'r_range': [float(min(r_values_A)) if r_values_A else 0,
                       float(max(r_values_A)) if r_values_A else 0],
            'phi_std_deg': float(np.degrees(phi_std_A)),
            'nll_range': [float(best_A), float(nlls_A.max())] if len(nlls_A) > 0 else [0, 0],
            'nll_std': float(nlls_A.std()) if len(nlls_A) > 1 else 0
        },
        'channel_B': {
            'n_near_optimal': len(near_opt_B),
            'r_range': [float(min(r_values_B)) if r_values_B else 0,
                       float(max(r_values_B)) if r_values_B else 0],
            'phi_std_deg': float(np.degrees(phi_std_B)),
            'nll_range': [float(best_B), float(nlls_B.max())] if len(nlls_B) > 0 else [0, 0],
            'nll_std': float(nlls_B.std()) if len(nlls_B) > 1 else 0
        }
    }

    return phase_identifiable, r_stable, details


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_fits(E_A, y_A, err_A, params_A, E_B, y_B, err_B, params_B,
              M1, W1, M2, W2, bg_order, model_name, out_dir):
    """Plot data and best-fit models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (E, y, err, params, name) in zip(axes, [
        (E_A, y_A, err_A, params_A, "Channel A"),
        (E_B, y_B, err_B, params_B, "Channel B")
    ]):
        E_fine = np.linspace(E.min(), E.max(), 200)
        model_fine = cross_section_model(E_fine, params, M1, W1, M2, W2, bg_order)

        ax.errorbar(E, y, yerr=err, fmt='ko', capsize=2, markersize=4, label='Data')
        ax.plot(E_fine, model_fine, 'r-', lw=2, label='Best fit')
        ax.axvline(M1, color='blue', linestyle='--', alpha=0.5, label=f'BW1 ({M1:.3f} GeV)')
        ax.axvline(M2, color='green', linestyle='--', alpha=0.5, label=f'BW2 ({M2:.3f} GeV)')

        r, phi = extract_R(params)
        chi2, dof, chi2_dof = compute_chi2_dof(params, E, y, err, M1, W1, M2, W2, bg_order)

        ax.set_xlabel(r'$\sqrt{s}$ [GeV]')
        ax.set_ylabel(r'$\sigma$ [nb]')
        ax.set_title(f'{name}\nr={r:.3f}, φ={np.degrees(phi):.0f}°, χ²/dof={chi2_dof:.2f}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(model_name, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'fit_plots.png', dpi=150)
    plt.close()


def plot_contours(nll_A, nll_B, r_grid, phi_grid, R_A, R_B, R_shared,
                  model_name, verdict, out_dir):
    """Plot profile likelihood contours."""
    delta_chi2_A = 2 * (nll_A - np.min(nll_A))
    delta_chi2_B = 2 * (nll_B - np.min(nll_B))

    phi_deg = np.degrees(phi_grid)
    levels = [2.30, 5.99]  # 68%, 95%

    fig, ax = plt.subplots(figsize=(10, 8))

    # Contours
    CS_A = ax.contour(phi_deg, r_grid, delta_chi2_A, levels=levels,
                      colors=['blue', 'lightblue'], linestyles='-')
    CS_B = ax.contour(phi_deg, r_grid, delta_chi2_B, levels=levels,
                      colors=['green', 'lightgreen'], linestyles='--')

    # Best-fit points
    r_A, phi_A = R_A
    r_B, phi_B = R_B
    r_shared, phi_shared = R_shared

    ax.plot(np.degrees(phi_A), r_A, 'bo', markersize=10,
            label=f'A: r={r_A:.2f}, φ={np.degrees(phi_A):.0f}°')
    ax.plot(np.degrees(phi_B), r_B, 'go', markersize=10,
            label=f'B: r={r_B:.2f}, φ={np.degrees(phi_B):.0f}°')
    ax.plot(np.degrees(phi_shared), r_shared, 'r*', markersize=15,
            label=f'Shared: r={r_shared:.2f}, φ={np.degrees(phi_shared):.0f}°')

    ax.set_xlabel('φ [deg]', fontsize=12)
    ax.set_ylabel('r = |c₂/c₁|', fontsize=12)
    ax.set_title(f'{model_name}\nProfile Likelihood Contours (solid=68%, dashed=95%)\nVerdict: {verdict}')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-180, 180)

    plt.tight_layout()
    plt.savefig(out_dir / 'contours.png', dpi=150)
    plt.close()

    return delta_chi2_A, delta_chi2_B


def plot_bootstrap(Lambda_boot, Lambda_obs, p_value, out_dir):
    """Plot bootstrap distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(Lambda_boot, bins=30, density=True, alpha=0.7, color='steelblue',
            edgecolor='black', label='Bootstrap')
    ax.axvline(Lambda_obs, color='red', linewidth=2, linestyle='--',
               label=f'Observed Λ = {Lambda_obs:.2f}')
    ax.axvline(np.mean(Lambda_boot), color='green', linewidth=2, linestyle=':',
               label=f'Bootstrap mean = {np.mean(Lambda_boot):.2f}')

    ax.set_xlabel(r'$\Lambda = 2\Delta(\mathrm{NLL})$')
    ax.set_ylabel('Density')
    ax.set_title(f'Bootstrap Distribution\np-value = {p_value:.4f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / 'bootstrap_hist.png', dpi=150)
    plt.close()


# ==============================================================================
# EMAIL REQUEST GENERATOR
# ==============================================================================

def generate_email_request(model_name, config, missing_channels, out_dir):
    """Generate email request for missing data."""
    email = f"""Subject: HEPData Data Request - {model_name}

Dear Data Authors,

I am conducting a systematic rank-1 bottleneck test analysis and would like to request
numeric data tables for the following publication:

Paper: {config.get('paper_ref', 'N/A')}
HEPData DOI: {config.get('hepdata_doi', 'N/A')}

Specifically, I need numeric tables for:

"""
    for ch in missing_channels:
        email += f"- {ch}\n"

    email += """
The test requires cross-section data as a function of center-of-mass energy with:
- Energy values (GeV)
- Cross section values
- Statistical uncertainties
- Systematic uncertainties (if available)

A CSV or plain text format would be ideal.

Thank you for your consideration.

Best regards,
[Your name]
"""

    filepath = out_dir / f"{model_name}_EMAIL.txt"
    with open(filepath, 'w') as f:
        f.write(email)

    return filepath


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_model(config):
    """
    Run complete rank-1 bottleneck test for a single model.

    Args:
        config: Model configuration dict (from YAML)

    Returns:
        dict: Results including verdict
    """
    model_name = config['name']
    display_name = config.get('display_name', model_name)

    print("=" * 70)
    print(f"RANK-1 BOTTLENECK TEST: {display_name}")
    print("=" * 70)
    print()

    # Check status
    if config.get('status') == 'needs_verification':
        print("  STATUS: needs_verification")
        print("  This model requires manual verification of HEPData tables.")
        return {
            'model': model_name,
            'verdict': 'NO DATA',
            'reason': 'Model requires manual verification of HEPData tables',
            'status': 'needs_verification'
        }

    # Setup output directory
    run_dir = RUNS_DIR / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)
    out_dir = run_dir

    # =========================================================================
    # DATA ACQUISITION
    # =========================================================================
    print("Step 1: Data Acquisition (HEPData)")
    print("-" * 50)

    record_id = config['hepdata_record']

    # Download tables
    table_A = config['channel_A'].get('table_number')
    table_B = config['channel_B'].get('table_number')

    if table_A is None or table_B is None:
        print("  ERROR: Table numbers not specified")
        email_path = generate_email_request(model_name, config,
                                           ['Channel A', 'Channel B'], REQUESTS_DIR)
        return {
            'model': model_name,
            'verdict': 'NO DATA',
            'reason': 'Table numbers not specified in config',
            'email_request': str(email_path)
        }

    csv_A = data_dir / f"table{table_A}.csv"
    csv_B = data_dir / f"table{table_B}.csv"

    success_A = download_hepdata_table(record_id, table_A, csv_A)
    success_B = download_hepdata_table(record_id, table_B, csv_B)

    if not success_A or not success_B:
        missing = []
        if not success_A:
            missing.append(config['channel_A'].get('table_name', f'Table {table_A}'))
        if not success_B:
            missing.append(config['channel_B'].get('table_name', f'Table {table_B}'))

        email_path = generate_email_request(model_name, config, missing, REQUESTS_DIR)
        print(f"  Generated email request: {email_path}")

        return {
            'model': model_name,
            'verdict': 'NO DATA',
            'reason': f'Could not download tables: {missing}',
            'email_request': str(email_path)
        }

    # Parse data
    try:
        df_A = parse_hepdata_csv(csv_A)
        df_B = parse_hepdata_csv(csv_B)
    except Exception as e:
        print(f"  ERROR: Could not parse CSV: {e}")
        return {
            'model': model_name,
            'verdict': 'NO DATA',
            'reason': f'CSV parsing failed: {e}'
        }

    # Filter to energy range
    E_range = config.get('energy_range', {})
    E_min = E_range.get('min_gev', 0)
    E_max = E_range.get('max_gev', 10)

    df_A = df_A[(df_A['E_gev'] >= E_min) & (df_A['E_gev'] <= E_max)]
    df_B = df_B[(df_B['E_gev'] >= E_min) & (df_B['E_gev'] <= E_max)]

    # Filter positive values
    df_A = df_A[df_A['y'] > 0]
    df_B = df_B[df_B['y'] > 0]

    if len(df_A) < 5 or len(df_B) < 5:
        print(f"  ERROR: Insufficient data points (A: {len(df_A)}, B: {len(df_B)})")
        return {
            'model': model_name,
            'verdict': 'NO DATA',
            'reason': f'Insufficient data points: A={len(df_A)}, B={len(df_B)}'
        }

    E_A = df_A['E_gev'].values
    y_A = df_A['y'].values
    err_A = df_A['total_err'].values

    E_B = df_B['E_gev'].values
    y_B = df_B['y'].values
    err_B = df_B['total_err'].values

    print(f"  Channel A: {len(E_A)} points, E=[{E_A.min():.2f}, {E_A.max():.2f}] GeV")
    print(f"  Channel B: {len(E_B)} points, E=[{E_B.min():.2f}, {E_B.max():.2f}] GeV")

    # =========================================================================
    # SETUP PARAMETERS
    # =========================================================================

    res = config['resonance_pair']
    M1 = res['BW1']['mass_gev']
    W1 = res['BW1']['width_gev']
    M2 = res['BW2']['mass_gev']
    W2 = res['BW2']['width_gev']

    syst_A = config['systematics'].get('correlated_scale_A', 0.10)
    syst_B = config['systematics'].get('correlated_scale_B', 0.10)

    bg_order = config['background'].get('order', 0)
    likelihood = config['likelihood_type']

    n_starts = config['optimizer'].get('n_starts', DEFAULT_N_STARTS)
    n_bootstrap = config['optimizer'].get('n_bootstrap', DEFAULT_N_BOOTSTRAP)

    chi2_min = config['fit_health_gates'].get('chi2_dof_min', 0.5)
    chi2_max = config['fit_health_gates'].get('chi2_dof_max', 3.0)

    print()
    print(f"Resonances: {res['BW1']['name']} ({M1:.3f} GeV) / {res['BW2']['name']} ({M2:.3f} GeV)")
    print(f"Likelihood: {likelihood}")
    print(f"Background: order {bg_order}")

    # =========================================================================
    # FIT CHANNELS
    # =========================================================================
    print()
    print("Step 2: Unconstrained Fits")
    print("-" * 50)

    print("  Fitting Channel A...")
    params_A, nll_A, all_results_A = fit_channel(
        E_A, y_A, err_A, syst_A, M1, W1, M2, W2,
        bg_order, likelihood, n_starts
    )

    if params_A is None:
        return {
            'model': model_name,
            'verdict': 'OPTIMIZER FAILURE',
            'reason': 'Channel A optimization failed'
        }

    chi2_A, dof_A, chi2_dof_A = compute_chi2_dof(params_A, E_A, y_A, err_A, M1, W1, M2, W2, bg_order)
    r_A, phi_A = extract_R(params_A)

    health_A = "PASS" if chi2_min < chi2_dof_A < chi2_max else \
               "UNDERCONSTRAINED" if chi2_dof_A < chi2_min else "POOR FIT"

    print(f"    r_A = {r_A:.4f}, φ_A = {np.degrees(phi_A):.1f}°")
    print(f"    χ²/dof = {chi2_dof_A:.3f} ({chi2_A:.1f}/{dof_A}) -> {health_A}")

    print("  Fitting Channel B...")
    params_B, nll_B, all_results_B = fit_channel(
        E_B, y_B, err_B, syst_B, M1, W1, M2, W2,
        bg_order, likelihood, n_starts
    )

    if params_B is None:
        return {
            'model': model_name,
            'verdict': 'OPTIMIZER FAILURE',
            'reason': 'Channel B optimization failed'
        }

    chi2_B, dof_B, chi2_dof_B = compute_chi2_dof(params_B, E_B, y_B, err_B, M1, W1, M2, W2, bg_order)
    r_B, phi_B = extract_R(params_B)

    health_B = "PASS" if chi2_min < chi2_dof_B < chi2_max else \
               "UNDERCONSTRAINED" if chi2_dof_B < chi2_min else "POOR FIT"

    print(f"    r_B = {r_B:.4f}, φ_B = {np.degrees(phi_B):.1f}°")
    print(f"    χ²/dof = {chi2_dof_B:.3f} ({chi2_B:.1f}/{dof_B}) -> {health_B}")

    nll_unc = nll_A + nll_B

    # =========================================================================
    # FIT HEALTH GATES
    # =========================================================================
    print()
    print("Step 3: Fit Health Gates")
    print("-" * 50)

    gates_pass = (health_A == "PASS") and (health_B == "PASS")
    print(f"  Channel A: {health_A}")
    print(f"  Channel B: {health_B}")
    print(f"  Overall: {'PASS' if gates_pass else 'FAIL'}")

    # =========================================================================
    # OPTIMIZER STABILITY
    # =========================================================================
    print()
    print("Step 4: Optimizer Stability Audit")
    print("-" * 50)

    phase_identifiable, r_stable, stability_details = optimizer_stability_audit(
        all_results_A, all_results_B, R_GRID, PHI_GRID
    )

    print(f"  Phase identifiable: {phase_identifiable}")
    print(f"  r stable: {r_stable}")

    # Save stability report
    with open(out_dir / 'optimizer_stability.json', 'w') as f:
        json.dump(stability_details, f, indent=2)

    # =========================================================================
    # JOINT CONSTRAINED FIT
    # =========================================================================
    print()
    print("Step 5: Joint Constrained Fit")
    print("-" * 50)

    shared_R, nll_con = fit_joint_constrained(
        params_A, params_B, config,
        E_A, y_A, err_A, syst_A,
        E_B, y_B, err_B, syst_B,
        M1, W1, M2, W2
    )

    Lambda = 2 * (nll_con - nll_unc)

    print(f"  Shared: r = {shared_R[0]:.4f}, φ = {np.degrees(shared_R[1]):.1f}°")
    print(f"  NLL_unconstrained = {nll_unc:.2f}")
    print(f"  NLL_constrained = {nll_con:.2f}")
    print(f"  Λ = {Lambda:.4f}")

    # Check Lambda >= 0
    optimizer_stable = Lambda >= 0
    if not optimizer_stable:
        print("  WARNING: Λ < 0 detected, retrying with more starts...")
        # Retry with more starts
        params_A, nll_A, all_results_A = fit_channel(
            E_A, y_A, err_A, syst_A, M1, W1, M2, W2,
            bg_order, likelihood, n_starts * 3
        )
        params_B, nll_B, all_results_B = fit_channel(
            E_B, y_B, err_B, syst_B, M1, W1, M2, W2,
            bg_order, likelihood, n_starts * 3
        )
        nll_unc = nll_A + nll_B

        shared_R, nll_con = fit_joint_constrained(
            params_A, params_B, config,
            E_A, y_A, err_A, syst_A,
            E_B, y_B, err_B, syst_B,
            M1, W1, M2, W2
        )
        Lambda = 2 * (nll_con - nll_unc)
        optimizer_stable = Lambda >= 0
        print(f"  After retry: Λ = {Lambda:.4f}")

    # =========================================================================
    # PROFILE LIKELIHOOD CONTOURS
    # =========================================================================
    print()
    print("Step 6: Profile Likelihood Contours")
    print("-" * 50)

    nll_grid_A = compute_profile_contour(
        E_A, y_A, err_A, syst_A, bg_order, likelihood,
        params_A, M1, W1, M2, W2, R_GRID, PHI_GRID
    )

    nll_grid_B = compute_profile_contour(
        E_B, y_B, err_B, syst_B, bg_order, likelihood,
        params_B, M1, W1, M2, W2, R_GRID, PHI_GRID
    )

    delta_chi2_A = 2 * (nll_grid_A - np.min(nll_grid_A))
    delta_chi2_B = 2 * (nll_grid_B - np.min(nll_grid_B))

    in_A_95 = check_point_in_contour(delta_chi2_A, R_GRID, PHI_GRID, shared_R[0], shared_R[1], 5.99)
    in_B_95 = check_point_in_contour(delta_chi2_B, R_GRID, PHI_GRID, shared_R[0], shared_R[1], 5.99)
    in_A_68 = check_point_in_contour(delta_chi2_A, R_GRID, PHI_GRID, shared_R[0], shared_R[1], 2.30)
    in_B_68 = check_point_in_contour(delta_chi2_B, R_GRID, PHI_GRID, shared_R[0], shared_R[1], 2.30)

    print(f"  Shared in A 95%: {in_A_95}")
    print(f"  Shared in B 95%: {in_B_95}")

    # =========================================================================
    # BOOTSTRAP P-VALUE
    # =========================================================================
    p_boot = None
    Lambda_boot = None

    if gates_pass and optimizer_stable:
        print()
        print("Step 7: Bootstrap P-value")
        print("-" * 50)

        Lambda_boot = run_bootstrap(
            E_A, y_A, err_A, syst_A, bg_order, likelihood,
            E_B, y_B, err_B, syst_B, bg_order, likelihood,
            params_A, params_B, shared_R,
            M1, W1, M2, W2, n_bootstrap
        )

        if len(Lambda_boot) > 0:
            p_boot = float(np.mean(Lambda_boot >= Lambda))
            print(f"  Bootstrap p-value = {p_boot:.4f}")

            plot_bootstrap(Lambda_boot, Lambda, p_boot, out_dir)
    else:
        print()
        print("Step 7: Bootstrap SKIPPED (gates failed or optimizer unstable)")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print()
    print("Step 8: VERDICT")
    print("=" * 50)

    if not gates_pass:
        verdict = "MODEL MISMATCH"
        reason = f"Fit health gates failed: A={health_A}, B={health_B}"
    elif not optimizer_stable:
        verdict = "OPTIMIZER FAILURE"
        reason = "Λ < 0 persists after increased optimization"
    elif not phase_identifiable and r_stable:
        # Magnitude-only test
        if p_boot is not None and p_boot < 0.05 and (not in_A_95 or not in_B_95):
            verdict = "DISFAVORED"
            reason = f"Magnitude-only: p={p_boot:.4f} < 0.05, shared outside 95%"
        elif p_boot is not None and p_boot >= 0.05:
            verdict = "SUPPORTED"
            reason = f"Magnitude-only: p={p_boot:.4f} >= 0.05"
        else:
            verdict = "INCONCLUSIVE"
            reason = "Phase multimodal, insufficient statistics"
    elif not phase_identifiable and not r_stable:
        verdict = "INCONCLUSIVE"
        reason = "Neither phase nor magnitude identifiable"
    elif p_boot is not None and p_boot >= 0.05 and in_A_95 and in_B_95:
        verdict = "SUPPORTED"
        reason = f"p={p_boot:.4f} >= 0.05, shared in both 95% contours"
    elif p_boot is not None and p_boot < 0.05 and (not in_A_95 or not in_B_95):
        verdict = "DISFAVORED"
        reason = f"p={p_boot:.4f} < 0.05, shared outside 95% contour"
    else:
        verdict = "INCONCLUSIVE"
        reason = "Mixed evidence"

    print(f"  {verdict}")
    print(f"  {reason}")

    # =========================================================================
    # GENERATE OUTPUTS
    # =========================================================================
    print()
    print("Generating outputs...")

    # Plots
    plot_fits(E_A, y_A, err_A, params_A, E_B, y_B, err_B, params_B,
              M1, W1, M2, W2, bg_order, display_name, out_dir)

    plot_contours(nll_grid_A, nll_grid_B, R_GRID, PHI_GRID,
                  (r_A, phi_A), (r_B, phi_B), shared_R,
                  display_name, verdict, out_dir)

    # Results JSON
    results = {
        'model': model_name,
        'display_name': display_name,
        'paper_ref': config.get('paper_ref'),
        'hepdata_doi': config.get('hepdata_doi'),

        'channel_A': {
            'name': config['channel_A'].get('table_name', 'Channel A'),
            'n_points': int(len(E_A)),
            'E_range': [float(E_A.min()), float(E_A.max())],
            'r': float(r_A),
            'phi_deg': float(np.degrees(phi_A)),
            'chi2_dof': float(chi2_dof_A),
            'health': health_A,
            'nll': float(nll_A)
        },

        'channel_B': {
            'name': config['channel_B'].get('table_name', 'Channel B'),
            'n_points': int(len(E_B)),
            'E_range': [float(E_B.min()), float(E_B.max())],
            'r': float(r_B),
            'phi_deg': float(np.degrees(phi_B)),
            'chi2_dof': float(chi2_dof_B),
            'health': health_B,
            'nll': float(nll_B)
        },

        'shared': {
            'r': float(shared_R[0]),
            'phi_deg': float(np.degrees(shared_R[1]))
        },

        'contour_check': {
            'shared_in_A_68': bool(in_A_68),
            'shared_in_A_95': bool(in_A_95),
            'shared_in_B_68': bool(in_B_68),
            'shared_in_B_95': bool(in_B_95),
            'shared_in_both_95': bool(in_A_95 and in_B_95)
        },

        'likelihood_ratio': {
            'Lambda': float(Lambda),
            'nll_unconstrained': float(nll_unc),
            'nll_constrained': float(nll_con),
            'optimizer_stable': bool(optimizer_stable)
        },

        'optimizer_stability': {
            'phase_identifiable': bool(phase_identifiable),
            'r_stable': bool(r_stable),
            'n_starts': n_starts
        },

        'gates_pass': bool(gates_pass),
        'verdict': verdict,
        'reason': reason
    }

    if p_boot is not None:
        results['bootstrap'] = {
            'p_value': float(p_boot),
            'n_bootstrap': int(len(Lambda_boot)),
            'Lambda_mean': float(np.mean(Lambda_boot)),
            'Lambda_std': float(np.std(Lambda_boot))
        }

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # REPORT.md
    report = generate_report(results, config, stability_details)
    with open(out_dir / 'REPORT.md', 'w') as f:
        f.write(report)

    print(f"  Saved to: {out_dir}")
    print()

    return results


def generate_report(results, config, stability_details):
    """Generate markdown report."""

    r = results

    report = f"""# Rank-1 Bottleneck Test Report: {r['display_name']}

## Data Source

- **Paper**: {r.get('paper_ref', 'N/A')}
- **HEPData DOI**: {r.get('hepdata_doi', 'N/A')}
- **Channel A**: {r['channel_A']['name']}
- **Channel B**: {r['channel_B']['name']}

## Resonance Parameters (Fixed)

| Resonance | Mass (GeV) | Width (GeV) |
|-----------|------------|-------------|
| {config['resonance_pair']['BW1']['name']} | {config['resonance_pair']['BW1']['mass_gev']:.3f} | {config['resonance_pair']['BW1']['width_gev']:.3f} |
| {config['resonance_pair']['BW2']['name']} | {config['resonance_pair']['BW2']['mass_gev']:.3f} | {config['resonance_pair']['BW2']['width_gev']:.3f} |

## Data Summary

| Channel | Points | Energy Range (GeV) |
|---------|--------|-------------------|
| A | {r['channel_A']['n_points']} | {r['channel_A']['E_range'][0]:.2f} - {r['channel_A']['E_range'][1]:.2f} |
| B | {r['channel_B']['n_points']} | {r['channel_B']['E_range'][0]:.2f} - {r['channel_B']['E_range'][1]:.2f} |

## Fit Results

### Channel A

| Parameter | Value |
|-----------|-------|
| r = |c2/c1| | {r['channel_A']['r']:.4f} |
| phi [deg] | {r['channel_A']['phi_deg']:.1f} |
| chi2/dof | {r['channel_A']['chi2_dof']:.3f} |
| Health | **{r['channel_A']['health']}** |

### Channel B

| Parameter | Value |
|-----------|-------|
| r = |c2/c1| | {r['channel_B']['r']:.4f} |
| phi [deg] | {r['channel_B']['phi_deg']:.1f} |
| chi2/dof | {r['channel_B']['chi2_dof']:.3f} |
| Health | **{r['channel_B']['health']}** |

## Optimizer Stability

| Metric | Channel A | Channel B |
|--------|-----------|-----------|
| Phase identifiable | {r['optimizer_stability']['phase_identifiable']} | - |
| r stable | {r['optimizer_stability']['r_stable']} | - |

## Joint Constrained Fit

| Metric | Value |
|--------|-------|
| Shared r | {r['shared']['r']:.4f} |
| Shared phi [deg] | {r['shared']['phi_deg']:.1f} |
| NLL_unconstrained | {r['likelihood_ratio']['nll_unconstrained']:.2f} |
| NLL_constrained | {r['likelihood_ratio']['nll_constrained']:.2f} |
| **Lambda** | **{r['likelihood_ratio']['Lambda']:.4f}** |

## Contour Analysis

| Check | Result |
|-------|--------|
| Shared in A 95% | {'YES' if r['contour_check']['shared_in_A_95'] else 'NO'} |
| Shared in B 95% | {'YES' if r['contour_check']['shared_in_B_95'] else 'NO'} |
| Shared in BOTH 95% | {'YES' if r['contour_check']['shared_in_both_95'] else 'NO'} |

"""

    if 'bootstrap' in r:
        report += f"""## Bootstrap P-value

| Metric | Value |
|--------|-------|
| p-value | **{r['bootstrap']['p_value']:.4f}** |
| n_bootstrap | {r['bootstrap']['n_bootstrap']} |
| Lambda mean | {r['bootstrap']['Lambda_mean']:.2f} |
| Lambda std | {r['bootstrap']['Lambda_std']:.2f} |

"""

    report += f"""## Verdict

**{r['verdict']}**

{r['reason']}

## Output Files

- `fit_plots.png` - Data and best-fit overlay
- `contours.png` - Profile likelihood contours
- `bootstrap_hist.png` - Bootstrap distribution (if computed)
- `optimizer_stability.json` - Stability audit data
- `results.json` - Full results

---
*Generated by model_sweep rank-1 bottleneck test framework*
"""

    return report


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run rank-1 bottleneck test for a single model')
    parser.add_argument('--model', type=str, help='Model name from models.yaml')
    parser.add_argument('--config', type=str, help='Path to model config YAML')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    elif args.model:
        with open(CONFIGS_DIR / 'models.yaml', 'r') as f:
            all_configs = yaml.safe_load(f)

        config = None
        for m in all_configs['models']:
            if m['name'] == args.model:
                config = m
                break

        if config is None:
            print(f"ERROR: Model '{args.model}' not found in models.yaml")
            sys.exit(1)
    else:
        print("ERROR: Must specify --model or --config")
        sys.exit(1)

    results = run_model(config)

    print()
    print("=" * 70)
    print(f"FINAL VERDICT: {results['verdict']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
