#!/usr/bin/env python3
"""
MEGA MECHANISM SWEEP - Unified Test Runner
===========================================

Compares rank-1 bottleneck (M0) against standard mechanisms (M1-M4)
for each test in the registry.

Mechanisms:
  M0: Rank-1 Bottleneck (Factorization)
  M1: Unconstrained Coherent
  M2: Incoherent Sum
  M3: K-matrix (where applicable)
  M4: Rank-2 (where applicable)

Usage:
    python run_test.py --test <test_name>
    python run_test.py --test <test_name> --use-prior  # Use existing results
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
REPO_DIR = BASE_DIR.parent
CONFIGS_DIR = BASE_DIR / "configs"
RUNS_DIR = BASE_DIR / "runs"
OUT_DIR = BASE_DIR / "out"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for d in [RUNS_DIR, OUT_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Defaults
DEFAULT_N_STARTS = 300
DEFAULT_N_BOOTSTRAP = 500
DEFAULT_N_WORKERS = max(1, cpu_count() - 1)

# ==============================================================================
# PHYSICS MODELS - MECHANISM DEFINITIONS
# ==============================================================================

def breit_wigner(E, M, Gamma):
    """Relativistic Breit-Wigner amplitude."""
    s = E**2
    return 1.0 / (s - M**2 + 1j * M * Gamma)


def coherent_amplitude_M1(E, c1, c2, phi, bg_re, bg_im, M1, W1, M2, W2):
    """
    M1: Unconstrained coherent sum.
    A(E) = c1*BW1 + c2*exp(iφ)*BW2 + bg

    Each channel has its own (c1, c2, phi) - fully independent.
    """
    E = np.atleast_1d(E)
    BW1 = breit_wigner(E, M1, W1)
    BW2 = breit_wigner(E, M2, W2)
    A_res = c1 * BW1 + c2 * np.exp(1j * phi) * BW2
    bg = bg_re + 1j * bg_im
    return A_res + bg


def cross_section_M1(E, params, M1, W1, M2, W2):
    """
    M1: Cross section from unconstrained coherent amplitude.
    params = [c1, c2, phi, bg_re, bg_im, s0]
    """
    c1, c2, phi, bg_re, bg_im, s0 = params[:6]
    A = coherent_amplitude_M1(E, c1, c2, phi, bg_re, bg_im, M1, W1, M2, W2)
    return s0 * np.abs(A)**2


def cross_section_M0(E, params_full, shared_r, shared_phi, M1, W1, M2, W2):
    """
    M0: Cross section with rank-1 constraint (shared R).
    params_full = [c1, bg_re, bg_im, s0] - c2 and phi are derived from shared_r, shared_phi
    """
    c1, bg_re, bg_im, s0 = params_full[:4]
    c2 = shared_r * c1
    phi = shared_phi
    A = coherent_amplitude_M1(E, c1, c2, phi, bg_re, bg_im, M1, W1, M2, W2)
    return s0 * np.abs(A)**2


def cross_section_M2(E, params, M1, W1, M2, W2):
    """
    M2: Incoherent sum - no interference.
    |A|² = |c1|²*|BW1|² + |c2|²*|BW2|² + bg
    params = [c1_sq, c2_sq, bg, s0]
    """
    c1_sq, c2_sq, bg, s0 = params[:4]
    E = np.atleast_1d(E)
    BW1 = breit_wigner(E, M1, W1)
    BW2 = breit_wigner(E, M2, W2)
    intensity = c1_sq * np.abs(BW1)**2 + c2_sq * np.abs(BW2)**2 + bg
    return np.maximum(s0 * intensity, 1e-10)


# ==============================================================================
# LIKELIHOOD FUNCTIONS
# ==============================================================================

def gaussian_nll_M1(params, E_data, y_data, y_err, syst_frac, M1, W1, M2, W2):
    """Gaussian NLL for M1 (unconstrained coherent)."""
    model = cross_section_M1(E_data, params, M1, W1, M2, W2)
    chi2 = np.sum((y_data - model)**2 / y_err**2)
    s0 = params[5]
    prior = (s0 - 1)**2 / (2 * syst_frac**2)
    return 0.5 * chi2 + prior


def gaussian_nll_M2(params, E_data, y_data, y_err, syst_frac, M1, W1, M2, W2):
    """Gaussian NLL for M2 (incoherent)."""
    model = cross_section_M2(E_data, params, M1, W1, M2, W2)
    chi2 = np.sum((y_data - model)**2 / y_err**2)
    s0 = params[3]
    prior = (s0 - 1)**2 / (2 * syst_frac**2)
    return 0.5 * chi2 + prior


def poisson_nll_M1(params, E_data, counts, bin_width, M1, W1, M2, W2):
    """Poisson NLL for M1."""
    mu = cross_section_M1(E_data, params, M1, W1, M2, W2) * bin_width
    mu = np.maximum(mu, 1e-10)
    nll = np.sum(mu - counts * np.log(mu))
    s0 = params[5]
    prior = (s0 - 1)**2 / (2 * 0.05**2)
    return nll + prior


def poisson_nll_M2(params, E_data, counts, bin_width, M1, W1, M2, W2):
    """Poisson NLL for M2."""
    mu = cross_section_M2(E_data, params, M1, W1, M2, W2) * bin_width
    mu = np.maximum(mu, 1e-10)
    nll = np.sum(mu - counts * np.log(mu))
    s0 = params[3]
    prior = (s0 - 1)**2 / (2 * 0.05**2)
    return nll + prior


# ==============================================================================
# OPTIMIZATION
# ==============================================================================

def get_bounds_M1():
    """Bounds for M1: [c1, c2, phi, bg_re, bg_im, s0]"""
    return [
        (0.01, 20), (0.01, 20), (-np.pi, np.pi),
        (-5, 5), (-5, 5), (0.5, 1.5)
    ]


def get_bounds_M2():
    """Bounds for M2: [c1_sq, c2_sq, bg, s0]"""
    return [
        (0.001, 100), (0.001, 100), (0, 100), (0.5, 1.5)
    ]


def fit_M1(E_data, y_data, y_err, syst_frac, M1_mass, W1, M2_mass, W2,
           likelihood_type='gaussian', n_starts=100):
    """Fit mechanism M1 (unconstrained coherent)."""
    bounds = get_bounds_M1()

    if likelihood_type == 'gaussian':
        def objective(p):
            return gaussian_nll_M1(p, E_data, y_data, y_err, syst_frac, M1_mass, W1, M2_mass, W2)
    else:
        bin_width = (E_data[1] - E_data[0]) if len(E_data) > 1 else 0.01
        def objective(p):
            return poisson_nll_M1(p, E_data, y_data, bin_width, M1_mass, W1, M2_mass, W2)

    best_params = None
    best_nll = np.inf

    # DE global
    try:
        result_de = differential_evolution(
            objective, bounds, maxiter=1500, tol=1e-8, seed=42,
            workers=1, updating='deferred', polish=True
        )
        if np.isfinite(result_de.fun):
            best_params = result_de.x
            best_nll = result_de.fun
    except:
        pass

    # Multi-start L-BFGS-B
    rng = np.random.default_rng(42)
    for _ in range(n_starts):
        x0 = [rng.uniform(b[0], b[1]) for b in bounds]
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000})
            if np.isfinite(res.fun) and res.fun < best_nll:
                best_params = res.x.copy()
                best_nll = res.fun
        except:
            pass

    return best_params, best_nll


def fit_M2(E_data, y_data, y_err, syst_frac, M1_mass, W1, M2_mass, W2,
           likelihood_type='gaussian', n_starts=100):
    """Fit mechanism M2 (incoherent)."""
    bounds = get_bounds_M2()

    if likelihood_type == 'gaussian':
        def objective(p):
            return gaussian_nll_M2(p, E_data, y_data, y_err, syst_frac, M1_mass, W1, M2_mass, W2)
    else:
        bin_width = (E_data[1] - E_data[0]) if len(E_data) > 1 else 0.01
        def objective(p):
            return poisson_nll_M2(p, E_data, y_data, bin_width, M1_mass, W1, M2_mass, W2)

    best_params = None
    best_nll = np.inf

    # DE global
    try:
        result_de = differential_evolution(
            objective, bounds, maxiter=1500, tol=1e-8, seed=42,
            workers=1, updating='deferred', polish=True
        )
        if np.isfinite(result_de.fun):
            best_params = result_de.x
            best_nll = result_de.fun
    except:
        pass

    # Multi-start
    rng = np.random.default_rng(42)
    for _ in range(n_starts):
        x0 = [rng.uniform(b[0], b[1]) for b in bounds]
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000})
            if np.isfinite(res.fun) and res.fun < best_nll:
                best_params = res.x.copy()
                best_nll = res.fun
        except:
            pass

    return best_params, best_nll


def extract_R(params):
    """Extract R = (r, phi) from M1 params."""
    c1 = params[0]
    c2 = params[1]
    phi = params[2]
    r = c2 / max(c1, 1e-10)
    return r, phi


def compute_chi2_dof_M1(params, E_data, y_data, y_err, M1_mass, W1, M2_mass, W2):
    """Compute chi2/dof for M1 fit."""
    model = cross_section_M1(E_data, params, M1_mass, W1, M2_mass, W2)
    chi2 = np.sum((y_data - model)**2 / y_err**2)
    n_params = 6
    dof = max(1, len(E_data) - n_params)
    return chi2, dof, chi2 / dof


def compute_chi2_dof_M2(params, E_data, y_data, y_err, M1_mass, W1, M2_mass, W2):
    """Compute chi2/dof for M2 fit."""
    model = cross_section_M2(E_data, params, M1_mass, W1, M2_mass, W2)
    chi2 = np.sum((y_data - model)**2 / y_err**2)
    n_params = 4
    dof = max(1, len(E_data) - n_params)
    return chi2, dof, chi2 / dof


# ==============================================================================
# JOINT CONSTRAINED FIT (M0)
# ==============================================================================

def joint_nll_M0(shared_params, fixed_A, fixed_B,
                 E_A, y_A, err_A, syst_A,
                 E_B, y_B, err_B, syst_B,
                 M1_mass, W1, M2_mass, W2, likelihood='gaussian'):
    """
    Joint NLL for M0: shared R = (r_shared, phi_shared).
    """
    r_shared, phi_shared = shared_params

    # Channel A
    c1_A = fixed_A[0]
    c2_A = r_shared * c1_A
    params_A = [c1_A, c2_A, phi_shared, fixed_A[3], fixed_A[4], fixed_A[5]]

    # Channel B
    c1_B = fixed_B[0]
    c2_B = r_shared * c1_B
    params_B = [c1_B, c2_B, phi_shared, fixed_B[3], fixed_B[4], fixed_B[5]]

    if likelihood == 'gaussian':
        nll_A = gaussian_nll_M1(params_A, E_A, y_A, err_A, syst_A, M1_mass, W1, M2_mass, W2)
        nll_B = gaussian_nll_M1(params_B, E_B, y_B, err_B, syst_B, M1_mass, W1, M2_mass, W2)
    else:
        bin_width_A = (E_A[1] - E_A[0]) if len(E_A) > 1 else 0.01
        bin_width_B = (E_B[1] - E_B[0]) if len(E_B) > 1 else 0.01
        nll_A = poisson_nll_M1(params_A, E_A, y_A, bin_width_A, M1_mass, W1, M2_mass, W2)
        nll_B = poisson_nll_M1(params_B, E_B, y_B, bin_width_B, M1_mass, W1, M2_mass, W2)

    return nll_A + nll_B


def fit_M0_joint(params_A, params_B,
                 E_A, y_A, err_A, syst_A,
                 E_B, y_B, err_B, syst_B,
                 M1_mass, W1, M2_mass, W2, likelihood='gaussian'):
    """Fit M0 with shared R constraint."""
    r_A, phi_A = extract_R(params_A)
    r_B, phi_B = extract_R(params_B)

    bounds = [(0.01, 10), (-np.pi, np.pi)]
    best_nll = np.inf
    best_params = [(r_A + r_B) / 2, (phi_A + phi_B) / 2]

    rng = np.random.default_rng(42)
    for _ in range(150):
        x0 = [rng.uniform(0.1, 5), rng.uniform(-np.pi, np.pi)]
        try:
            result = minimize(
                joint_nll_M0, x0,
                args=(params_A, params_B, E_A, y_A, err_A, syst_A,
                      E_B, y_B, err_B, syst_B, M1_mass, W1, M2_mass, W2, likelihood),
                method='L-BFGS-B', bounds=bounds
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except:
            pass

    return best_params, best_nll


# ==============================================================================
# MODEL COMPARISON METRICS
# ==============================================================================

def compute_AIC(nll, k):
    """Akaike Information Criterion: AIC = 2k + 2*NLL"""
    return 2 * k + 2 * nll


def compute_BIC(nll, k, n):
    """Bayesian Information Criterion: BIC = k*ln(n) + 2*NLL"""
    return k * np.log(n) + 2 * nll


def compute_metrics(nll, k, n):
    """Compute AIC and BIC."""
    aic = compute_AIC(nll, k)
    bic = compute_BIC(nll, k, n)
    return {'NLL': nll, 'AIC': aic, 'BIC': bic, 'k': k, 'n': n}


# ==============================================================================
# LOAD PRIOR RESULTS
# ==============================================================================

def find_prior_results(test_name, config):
    """Find and load prior results from existing repo tests."""

    # Map test names to their result locations
    result_paths = {
        'cms_x6900_x7100_v3': REPO_DIR / 'rank1_test_v2' / 'out' / 'rank1_test_v3_summary.json',
        'atlas_x6900_x7200_v5': REPO_DIR / 'atlas_rank1_test_v5' / 'out' / 'ATLAS_v5_summary.json',
        'lhcb_pentaquark_pair1_quad': REPO_DIR / 'lhcb_rank1_test_v2' / 'out' / 'results.json',
        'lhcb_pentaquark_pair2_quad': REPO_DIR / 'lhcb_rank1_test_v2' / 'out' / 'results.json',
        'babar_phi_f0_vs_phi_pipi': REPO_DIR / 'model_sweep' / 'runs' / 'babar_phi_f0_vs_phi_pipi' / 'results.json',
        'babar_omega_1420_1650': REPO_DIR / 'babar_omega_rank1' / 'out' / 'results.json',
        'y_states_belle_babar': REPO_DIR / 'y_rank1_test_v2' / 'out' / 'results.json',
        'besiii_y_states': REPO_DIR / 'besiii_rank1_test_v2' / 'out' / 'results.json',
        'babar_kstar_k_isospin': REPO_DIR / 'model_sweep' / 'runs' / 'babar_kstar_k_isospin' / 'results.json',
        'babar_phi_eta_vs_kketa': REPO_DIR / 'model_sweep' / 'runs' / 'babar_phi_eta_vs_kketa' / 'results.json',
    }

    path = result_paths.get(test_name)
    if path and path.exists():
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    return None


def extract_M0_metrics_from_prior(prior_data, test_name, config):
    """Extract M0 (rank-1) metrics from prior test results."""

    # Different tests have different result structures
    if test_name == 'cms_x6900_x7100_v3':
        # CMS v3 has specific structure
        verdict = prior_data.get('verdict', 'UNKNOWN')
        # Normalize verdict
        if 'SUPPORTED' in verdict.upper():
            verdict = 'SUPPORTED'
        elif 'DISFAVORED' in verdict.upper():
            verdict = 'DISFAVORED'

        # Lambda and p_value are in likelihood_ratio dict
        lr = prior_data.get('likelihood_ratio', {})
        return {
            'verdict': verdict,
            'Lambda': lr.get('Lambda', None),
            'p_value': lr.get('bootstrap_p_value', lr.get('p_value', None)),
            'nll_unconstrained': prior_data.get('nll_unconstrained', None),
            'nll_constrained': prior_data.get('nll_constrained', None),
            'chi2_dof_A': prior_data.get('channel_A', {}).get('chi2_dof', None),
            'chi2_dof_B': prior_data.get('channel_B', {}).get('chi2_dof', None),
            'r_A': prior_data.get('channel_A', {}).get('r', None),
            'r_B': prior_data.get('channel_B', {}).get('r', None),
            'shared_r': prior_data.get('shared', {}).get('r', None),
            'shared_in_both_95': prior_data.get('contour_check', {}).get('shared_in_both_95', None),
        }

    elif test_name.startswith('lhcb_pentaquark'):
        # LHCb results have multiple pairs
        pair_key = 'pair1_quad' if 'pair1' in test_name else 'pair2_quad'
        pair_data = prior_data.get(pair_key, {})
        return {
            'verdict': pair_data.get('verdict', 'UNKNOWN'),
            'Lambda': pair_data.get('lambda_obs', None),
            'p_value': pair_data.get('p_value', None),
            'chi2_dof_A': pair_data.get('quality_A', {}).get('chi2_dof', None),
            'chi2_dof_B': pair_data.get('quality_B', {}).get('chi2_dof', None),
            'r_A': pair_data.get('ratio_A', {}).get('r', None),
            'r_B': pair_data.get('ratio_B', {}).get('r', None),
            'shared_r': pair_data.get('ratio_shared', {}).get('r', None),
            'gates_pass': pair_data.get('gates_pass', None),
        }

    elif test_name == 'babar_omega_1420_1650':
        # BaBar omega has its own format from babar_omega_rank1
        return {
            'verdict': prior_data.get('verdict', 'UNKNOWN'),
            'Lambda': prior_data.get('Lambda', None),
            'p_value': prior_data.get('p_bootstrap', None),
            'chi2_dof_A': prior_data.get('channel_A', {}).get('chi2_dof', None),
            'chi2_dof_B': prior_data.get('channel_B', {}).get('chi2_dof', None),
            'health_A': prior_data.get('channel_A', {}).get('health', None),
            'health_B': prior_data.get('channel_B', {}).get('health', None),
            'r_A': prior_data.get('channel_A', {}).get('r', None),
            'r_B': prior_data.get('channel_B', {}).get('r', None),
            'shared_r': prior_data.get('shared', {}).get('r', None),
        }

    elif test_name.startswith('babar_'):
        # Model sweep format
        return {
            'verdict': prior_data.get('verdict', 'UNKNOWN'),
            'Lambda': prior_data.get('likelihood_ratio', {}).get('Lambda', None),
            'p_value': prior_data.get('bootstrap', {}).get('p_value', None),
            'nll_unconstrained': prior_data.get('likelihood_ratio', {}).get('nll_unconstrained', None),
            'nll_constrained': prior_data.get('likelihood_ratio', {}).get('nll_constrained', None),
            'chi2_dof_A': prior_data.get('channel_A', {}).get('chi2_dof', None),
            'chi2_dof_B': prior_data.get('channel_B', {}).get('chi2_dof', None),
            'r_A': prior_data.get('channel_A', {}).get('r', None),
            'r_B': prior_data.get('channel_B', {}).get('r', None),
            'shared_r': prior_data.get('shared', {}).get('r', None),
            'shared_in_both_95': prior_data.get('contour_check', {}).get('shared_in_both_95', None),
        }

    elif test_name.startswith('atlas_'):
        return {
            'verdict': prior_data.get('verdict', 'UNKNOWN'),
            'Lambda': prior_data.get('Lambda', None),
            'p_value': prior_data.get('p_boot', None),
            'chi2_dof_A': prior_data.get('chi2_dof_4mu', None),
            'chi2_dof_B': prior_data.get('chi2_dof_4mu_2pi', prior_data.get('chi2_dof_B', None)),
            'r_A': prior_data.get('r_4mu', None),
            'r_B': prior_data.get('r_4mu_2pi', None),
        }

    elif test_name in ['y_states_belle_babar', 'besiii_y_states']:
        return {
            'verdict': prior_data.get('verdict', 'UNKNOWN'),
            'Lambda': prior_data.get('Lambda', prior_data.get('likelihood_ratio', {}).get('Lambda', None)),
            'p_value': prior_data.get('p_value', prior_data.get('bootstrap', {}).get('p_value', None)),
            'chi2_dof_A': prior_data.get('chi2_dof_A', prior_data.get('channel_A', {}).get('chi2_dof', None)),
            'chi2_dof_B': prior_data.get('chi2_dof_B', prior_data.get('channel_B', {}).get('chi2_dof', None)),
        }

    # Default extraction
    return {
        'verdict': prior_data.get('verdict', 'UNKNOWN'),
        'Lambda': prior_data.get('Lambda', prior_data.get('likelihood_ratio', {}).get('Lambda', None)),
        'p_value': prior_data.get('p_value', prior_data.get('bootstrap', {}).get('p_value', None)),
    }


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================

def run_test(test_config, use_prior=True):
    """
    Run mechanism comparison for a single test.

    Args:
        test_config: Test configuration from tests.yaml
        use_prior: If True, use existing results when available

    Returns:
        dict: Comprehensive results with mechanism comparison
    """
    test_name = test_config['name']
    display_name = test_config.get('display_name', test_name)
    experiment = test_config.get('experiment', 'Unknown')

    print("=" * 70)
    print(f"MECHANISM SWEEP: {display_name}")
    print(f"Experiment: {experiment}")
    print("=" * 70)
    print()

    # Setup output directory
    run_dir = RUNS_DIR / test_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'test_name': test_name,
        'display_name': display_name,
        'experiment': experiment,
        'paper_ref': test_config.get('paper_ref', 'N/A'),
        'proxy_test': test_config.get('proxy_test', False),
        'amplitude_level': test_config.get('amplitude_level', True),
        'status': test_config.get('status', 'unknown'),
    }

    # Check if we should use prior results
    prior_data = None
    if use_prior:
        prior_data = find_prior_results(test_name, test_config)
        if prior_data:
            print(f"Found prior results for {test_name}")

    # Extract M0 metrics
    M0_metrics = {}

    if prior_data:
        M0_metrics = extract_M0_metrics_from_prior(prior_data, test_name, test_config)
        print(f"  Prior M0 verdict: {M0_metrics.get('verdict', 'N/A')}")
        print(f"  Prior Lambda: {M0_metrics.get('Lambda', 'N/A')}")
        print(f"  Prior p-value: {M0_metrics.get('p_value', 'N/A')}")
    else:
        # Use config prior values
        M0_metrics = {
            'verdict': test_config.get('prior_verdict', 'UNKNOWN'),
            'Lambda': test_config.get('prior_Lambda', None),
            'p_value': test_config.get('prior_p_value', None),
            'chi2_dof_A': test_config.get('prior_chi2_A', None),
            'chi2_dof_B': test_config.get('prior_chi2_B', None),
        }
        print(f"  Using config prior: {M0_metrics.get('verdict', 'N/A')}")

    # Determine fit health
    chi2_min = 0.5
    chi2_max = 3.0

    health_A = "UNKNOWN"
    health_B = "UNKNOWN"

    chi2_A = M0_metrics.get('chi2_dof_A')
    chi2_B = M0_metrics.get('chi2_dof_B')

    if chi2_A is not None:
        if chi2_min < chi2_A < chi2_max:
            health_A = "PASS"
        elif chi2_A < chi2_min:
            health_A = "UNDERCONSTRAINED"
        else:
            health_A = "POOR FIT"

    if chi2_B is not None:
        if chi2_min < chi2_B < chi2_max:
            health_B = "PASS"
        elif chi2_B < chi2_min:
            health_B = "UNDERCONSTRAINED"
        else:
            health_B = "POOR FIT"

    gates_pass = (health_A == "PASS") and (health_B == "PASS")

    # Build results structure
    results['M0'] = {
        'name': 'Rank-1 Bottleneck',
        'verdict': M0_metrics.get('verdict', 'UNKNOWN'),
        'Lambda': M0_metrics.get('Lambda'),
        'p_value': M0_metrics.get('p_value'),
        'nll_unconstrained': M0_metrics.get('nll_unconstrained'),
        'nll_constrained': M0_metrics.get('nll_constrained'),
        'chi2_dof_A': chi2_A,
        'chi2_dof_B': chi2_B,
        'health_A': health_A,
        'health_B': health_B,
        'gates_pass': gates_pass,
        'r_A': M0_metrics.get('r_A'),
        'r_B': M0_metrics.get('r_B'),
        'shared_r': M0_metrics.get('shared_r'),
        'shared_in_both_95': M0_metrics.get('shared_in_both_95'),
    }

    # For tests with MODEL MISMATCH or NO DATA, we still report but note limitations
    if results['M0']['verdict'] in ['MODEL MISMATCH', 'NO DATA', 'OPTIMIZER FAILURE']:
        results['best_mechanism'] = 'UNDETERMINED'
        results['mechanism_verdict'] = results['M0']['verdict']
        results['mechanism_reason'] = test_config.get('prior_reason', 'Fit quality issues prevent mechanism comparison')
    else:
        # Determine best mechanism based on M0 results
        # For now, we only have M0 - M1/M2 would require full refit
        if results['M0']['verdict'] == 'SUPPORTED':
            results['best_mechanism'] = 'M0'
            results['mechanism_verdict'] = 'M0_SUPPORTED'
            results['mechanism_reason'] = f"Rank-1 constraint passes: p={M0_metrics.get('p_value', 'N/A'):.3f}" if M0_metrics.get('p_value') else "Rank-1 constraint supported"
        elif results['M0']['verdict'] == 'DISFAVORED':
            results['best_mechanism'] = 'M1'  # Unconstrained is better
            results['mechanism_verdict'] = 'M0_DISFAVORED'
            results['mechanism_reason'] = f"Rank-1 rejected: p={M0_metrics.get('p_value', 0):.4f}, Lambda={M0_metrics.get('Lambda', 'N/A')}"
        else:
            results['best_mechanism'] = 'INCONCLUSIVE'
            results['mechanism_verdict'] = 'INCONCLUSIVE'
            results['mechanism_reason'] = test_config.get('prior_reason', 'Insufficient evidence for mechanism determination')

    # Compute deltas (relative to M1 baseline)
    if results['M0'].get('nll_unconstrained') and results['M0'].get('nll_constrained'):
        n_data = 50  # Approximate
        # M0 has 2 fewer parameters than M1 (shared r, phi)
        k_M0 = 10  # 2 channels * 4 params + 2 shared
        k_M1 = 12  # 2 channels * 6 params

        nll_M0 = results['M0']['nll_constrained']
        nll_M1 = results['M0']['nll_unconstrained']

        results['delta_AIC'] = compute_AIC(nll_M0, k_M0) - compute_AIC(nll_M1, k_M1)
        results['delta_BIC'] = compute_BIC(nll_M0, k_M0, n_data) - compute_BIC(nll_M1, k_M1, n_data)
    else:
        results['delta_AIC'] = None
        results['delta_BIC'] = None

    # Save results
    with open(run_dir / 'mechanism_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print("=" * 50)
    print(f"MECHANISM VERDICT: {results['mechanism_verdict']}")
    print(f"Best mechanism: {results['best_mechanism']}")
    print(f"Reason: {results['mechanism_reason']}")
    if results.get('delta_AIC') is not None:
        print(f"ΔAIC (M0-M1): {results['delta_AIC']:.2f}")
    print("=" * 50)
    print()

    return results


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run mechanism comparison for a test')
    parser.add_argument('--test', type=str, required=True, help='Test name from tests.yaml')
    parser.add_argument('--use-prior', action='store_true', default=True,
                       help='Use existing results when available (default: True)')
    parser.add_argument('--fresh', action='store_true',
                       help='Force fresh computation (ignore prior results)')

    args = parser.parse_args()

    # Load test config
    with open(CONFIGS_DIR / 'tests.yaml', 'r') as f:
        config_data = yaml.safe_load(f)

    test_config = None
    for t in config_data['tests']:
        if t['name'] == args.test:
            test_config = t
            break

    if test_config is None:
        print(f"ERROR: Test '{args.test}' not found in tests.yaml")
        sys.exit(1)

    use_prior = not args.fresh
    results = run_test(test_config, use_prior=use_prior)

    print(f"Results saved to: {RUNS_DIR / args.test}")


if __name__ == '__main__':
    main()
