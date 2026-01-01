#!/usr/bin/env python3
"""
sim_fit_v3.py - Calibrated fitting for rank-1 bottleneck test

CALIBRATION FIXES (v3):
- Fix A: Y-states bootstrap uses stat-only noise (not total sigma)
- Fix B: Bootstrap uses same restarts as main fit + adaptive multistart
- Fix C: Gates applied before Type-I counting
- Fix D: Fixed background model (no AIC/BIC selection in calibration)
- Fix E: Generator/fitter BW convention match assertion

Key features:
- dof_diff = 2 for complex R constraint
- Bootstrap p-values with nestedness enforcement
- Adaptive multi-start optimizer
- Fit-health gates: chi2/dof in [0.5, 3.0], deviance/dof < 3.0
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist, kstest
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import warnings
import csv
import os

warnings.filterwarnings('ignore')

# Constants
DOF_DIFF = 2  # Degrees of freedom for complex R constraint (magnitude + phase)
DEFAULT_BOOTSTRAP = 200
DEFAULT_STARTS = 120
CHI2_DOF_MIN = 0.5
CHI2_DOF_MAX = 3.0
DEVIANCE_DOF_MAX = 3.0
NESTEDNESS_TOL = 1e-6  # NLL_unc must be <= NLL_con + tol
ADAPTIVE_PATIENCE = 20  # Consecutive starts without improvement
ADAPTIVE_MAX_STARTS = 400  # Hard cap for adaptive multistart
NLL_IMPROVEMENT_THRESHOLD = 1e-3


@dataclass
class FitResult:
    """Container for fit results"""
    nll: float
    chi2: float
    dof: int
    r: float
    phi_deg: float
    r_err: float = 0.0
    phi_err: float = 0.0
    converged: bool = True
    n_restarts: int = 1
    deviance: float = 0.0


@dataclass
class CalibrationTrace:
    """Container for calibration diagnostic trace"""
    test: str
    trial: int
    nll_con: float
    nll_unc: float
    lambda_obs: float
    p_boot: float
    p_wilks: float
    k_exceed: int
    n_boot: int
    chi2_dof_a: float
    chi2_dof_b: float
    dev_dof_a: float
    dev_dof_b: float
    gates: str
    starts_used: int
    optimizer_retries: int
    converged: bool
    rejected: bool
    pass_trial: bool


def breit_wigner(x: np.ndarray, m: float, gamma: float) -> np.ndarray:
    """Simple Breit-Wigner amplitude (complex)."""
    return 1.0 / ((x - m) - 1j * gamma / 2)


def complex_from_polar(r: float, phi_deg: float) -> complex:
    """Convert polar (r, phi_deg) to complex number."""
    phi_rad = np.deg2rad(phi_deg)
    return r * np.exp(1j * phi_rad)


def compute_nll_poisson(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Poisson NLL (ignoring factorial constant)."""
    y_pred = np.maximum(y_pred, 1e-10)
    y_obs_safe = np.maximum(y_obs, 0)
    return np.sum(y_pred - y_obs_safe * np.log(y_pred))


def compute_nll_gaussian(y_obs: np.ndarray, y_pred: np.ndarray,
                         sigma: np.ndarray) -> float:
    """Gaussian NLL."""
    sigma = np.maximum(sigma, 1e-10)
    return 0.5 * np.sum(((y_obs - y_pred) / sigma)**2)


def compute_chi2(y_obs: np.ndarray, y_pred: np.ndarray,
                 sigma: np.ndarray, data_type: str) -> float:
    """Compute chi-squared."""
    if data_type == 'poisson':
        var = np.maximum(y_pred, 1)
        return np.sum((y_obs - y_pred)**2 / var)
    else:
        sigma = np.maximum(sigma, 1e-10)
        return np.sum(((y_obs - y_pred) / sigma)**2)


def compute_deviance(y_obs: np.ndarray, y_pred: np.ndarray, data_type: str) -> float:
    """Compute deviance for fit health assessment."""
    if data_type == 'poisson':
        y_obs_safe = np.maximum(y_obs, 0.5)
        y_pred_safe = np.maximum(y_pred, 0.5)
        return 2 * np.sum(y_obs_safe * np.log(y_obs_safe / y_pred_safe) - (y_obs - y_pred))
    else:
        return 0.0


class BWParams:
    """Breit-Wigner parameters"""
    def __init__(self, m1=0.0, m2=1.0, gamma1=0.10, gamma2=0.08):
        self.m1 = m1
        self.m2 = m2
        self.gamma1 = gamma1
        self.gamma2 = gamma2


def intensity_shape(x: np.ndarray, R: complex, bw_params: BWParams) -> np.ndarray:
    """Compute intensity shape (unnormalized)."""
    bw1 = breit_wigner(x, bw_params.m1, bw_params.gamma1)
    bw2 = breit_wigner(x, bw_params.m2, bw_params.gamma2)
    return np.abs(bw1 + R * bw2)**2


def fit_channel_adaptive(x: np.ndarray, y_obs: np.ndarray, sigma: np.ndarray,
                         bw_params: BWParams, data_type: str,
                         n_starts: int = DEFAULT_STARTS,
                         adaptive: bool = True,
                         stat_error: np.ndarray = None) -> Dict:
    """
    Fit single channel with adaptive multistart.
    Parameters: r, phi, scale, b0, b1

    For Gaussian with systematics:
    - Use sigma for NLL computation (includes syst)
    - Use stat_error for chi2 computation (stat-only noise)
    """
    # For chi2 computation, use stat_error if available (Gaussian with systematics)
    chi2_sigma = stat_error if stat_error is not None else sigma

    def model(params):
        r, phi, scale, b0, b1 = params
        if r <= 0 or scale <= 0:
            return np.full_like(x, 1e10, dtype=float)
        R = complex_from_polar(r, phi)
        signal = intensity_shape(x, R, bw_params)
        signal_max = np.max(signal)
        if signal_max > 0:
            signal = signal / signal_max
        bg = b0 + b1 * (x - np.mean(x))
        return scale * signal + np.maximum(bg, 0)

    # FIX: Use stat_error for NLL when available (Gaussian+syst case)
    # This ensures NLL reflects actual noise level, not inflated sigma
    nll_sigma = stat_error if stat_error is not None else sigma

    def objective(params):
        y_pred = model(params)
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return 1e15
        if data_type == 'poisson':
            return compute_nll_poisson(y_obs, y_pred)
        else:
            return compute_nll_gaussian(y_obs, y_pred, nll_sigma)

    best_nll = np.inf
    best_params = None
    starts_used = 0
    no_improvement_count = 0

    scale_init = np.max(y_obs) * 0.7
    b0_init = max(np.min(y_obs) * 0.3, 0.1)
    bounds = [(0.01, 5.0), (-180, 180),
              (0.1, scale_init * 15),
              (0, b0_init * 10), (-5, 5)]

    max_starts = ADAPTIVE_MAX_STARTS if adaptive else n_starts

    for i in range(max_starts):
        r0 = np.random.uniform(0.2, 2.0)
        phi0 = np.random.uniform(-180, 180)
        scale0 = scale_init * np.random.uniform(0.3, 3.0)
        b0_0 = b0_init * np.random.uniform(0.3, 3.0)
        b1_0 = np.random.uniform(-1.0, 1.0)
        x0 = [r0, phi0, scale0, b0_0, b1_0]

        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 500})
            if result.fun > 1e10 or not result.success:
                result_powell = minimize(objective, x0, method='Powell',
                                       options={'maxiter': 1000})
                if result_powell.fun < result.fun:
                    result = result_powell

            starts_used += 1

            if result.fun < best_nll - NLL_IMPROVEMENT_THRESHOLD:
                best_nll = result.fun
                best_params = result.x
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Adaptive stopping
            if adaptive and i >= n_starts and no_improvement_count >= ADAPTIVE_PATIENCE:
                break

        except Exception:
            continue

    if best_params is None:
        return {'converged': False, 'nll': 1e10, 'starts_used': starts_used}

    r, phi, scale, b0, b1 = best_params
    y_pred = model(best_params)
    # Use chi2_sigma (stat_error if available) for chi2 computation
    chi2_val = compute_chi2(y_obs, y_pred, chi2_sigma, data_type)
    deviance = compute_deviance(y_obs, y_pred, data_type)
    dof = len(x) - 5

    return {
        'converged': True,
        'nll': best_nll,
        'chi2': chi2_val,
        'deviance': deviance,
        'dof': dof,
        'chi2_dof': chi2_val / max(dof, 1),
        'deviance_dof': deviance / max(dof, 1) if data_type == 'poisson' else 0.0,
        'r': r,
        'phi': phi,
        'scale': scale,
        'b0': b0,
        'b1': b1,
        'y_pred': y_pred,
        'starts_used': starts_used
    }


def fit_joint_unconstrained(dataset: Dict, n_starts: int = DEFAULT_STARTS,
                            adaptive: bool = True) -> Dict:
    """Fit both channels with separate R_A, R_B."""
    bw_params = dataset['bw_params']
    ch_a = dataset['channelA']
    ch_b = dataset['channelB']

    # Pass stat_error for chi2 computation (Gaussian with systematics)
    fit_a = fit_channel_adaptive(
        ch_a['x'], ch_a['y'], ch_a['sigma'],
        bw_params, ch_a['type'], n_starts, adaptive,
        stat_error=ch_a.get('stat_error')
    )
    fit_b = fit_channel_adaptive(
        ch_b['x'], ch_b['y'], ch_b['sigma'],
        bw_params, ch_b['type'], n_starts, adaptive,
        stat_error=ch_b.get('stat_error')
    )

    if not fit_a['converged'] or not fit_b['converged']:
        return {'converged': False, 'nll': 1e10}

    return {
        'converged': True,
        'nll': fit_a['nll'] + fit_b['nll'],
        'r_a': fit_a['r'],
        'phi_a': fit_a['phi'],
        'r_b': fit_b['r'],
        'phi_b': fit_b['phi'],
        'chi2_a': fit_a['chi2'],
        'chi2_b': fit_b['chi2'],
        'deviance_a': fit_a['deviance'],
        'deviance_b': fit_b['deviance'],
        'dof_a': fit_a['dof'],
        'dof_b': fit_b['dof'],
        'chi2_dof_a': fit_a['chi2_dof'],
        'chi2_dof_b': fit_b['chi2_dof'],
        'deviance_dof_a': fit_a.get('deviance_dof', 0),
        'deviance_dof_b': fit_b.get('deviance_dof', 0),
        'y_pred_a': fit_a.get('y_pred'),
        'y_pred_b': fit_b.get('y_pred'),
        'starts_used': fit_a.get('starts_used', 0) + fit_b.get('starts_used', 0)
    }


def fit_joint_constrained(dataset: Dict, n_starts: int = DEFAULT_STARTS,
                          adaptive: bool = True) -> Dict:
    """Fit both channels with SHARED R (rank-1 constraint)."""
    bw_params = dataset['bw_params']
    ch_a = dataset['channelA']
    ch_b = dataset['channelB']

    x_a, y_a, sigma_a = ch_a['x'], ch_a['y'], ch_a['sigma']
    x_b, y_b, sigma_b = ch_b['x'], ch_b['y'], ch_b['sigma']
    type_a, type_b = ch_a['type'], ch_b['type']

    # For chi2 AND NLL computation, use stat_error if available (Gaussian with systematics)
    # FIX: This ensures NLL reflects actual noise level, not inflated sigma
    nll_sigma_a = ch_a.get('stat_error', sigma_a)
    nll_sigma_b = ch_b.get('stat_error', sigma_b)
    chi2_sigma_a = nll_sigma_a
    chi2_sigma_b = nll_sigma_b

    def model_a(r, phi, scale, b0, b1):
        R = complex_from_polar(r, phi)
        signal = intensity_shape(x_a, R, bw_params)
        signal_max = np.max(signal)
        if signal_max > 0:
            signal = signal / signal_max
        bg = b0 + b1 * (x_a - np.mean(x_a))
        return scale * signal + np.maximum(bg, 0)

    def model_b(r, phi, scale, b0, b1):
        R = complex_from_polar(r, phi)
        signal = intensity_shape(x_b, R, bw_params)
        signal_max = np.max(signal)
        if signal_max > 0:
            signal = signal / signal_max
        bg = b0 + b1 * (x_b - np.mean(x_b))
        return scale * signal + np.maximum(bg, 0)

    def objective(params):
        r, phi, sA, b0A, b1A, sB, b0B, b1B = params
        if r <= 0 or sA <= 0 or sB <= 0:
            return 1e15

        y_pred_a = model_a(r, phi, sA, b0A, b1A)
        y_pred_b = model_b(r, phi, sB, b0B, b1B)

        if np.any(np.isnan(y_pred_a)) or np.any(np.isnan(y_pred_b)):
            return 1e15

        if type_a == 'poisson':
            nll_a = compute_nll_poisson(y_a, y_pred_a)
        else:
            nll_a = compute_nll_gaussian(y_a, y_pred_a, nll_sigma_a)

        if type_b == 'poisson':
            nll_b = compute_nll_poisson(y_b, y_pred_b)
        else:
            nll_b = compute_nll_gaussian(y_b, y_pred_b, nll_sigma_b)

        return nll_a + nll_b

    best_nll = np.inf
    best_params = None
    starts_used = 0
    no_improvement_count = 0

    scale_a_init = max(np.max(y_a) * 0.7, 1.0)
    scale_b_init = max(np.max(y_b) * 0.7, 1.0)
    b0_a_init = max(np.min(y_a) * 0.3, 0.1)
    b0_b_init = max(np.min(y_b) * 0.3, 0.1)

    bounds = [(0.01, 5.0), (-180, 180),
              (0.1, scale_a_init * 15), (0, b0_a_init * 10), (-5, 5),
              (0.1, scale_b_init * 15), (0, b0_b_init * 10), (-5, 5)]

    max_starts = ADAPTIVE_MAX_STARTS if adaptive else n_starts

    for i in range(max_starts):
        r0 = np.random.uniform(0.2, 2.0)
        phi0 = np.random.uniform(-180, 180)
        sA0 = scale_a_init * np.random.uniform(0.3, 3.0)
        sB0 = scale_b_init * np.random.uniform(0.3, 3.0)
        b0A0 = b0_a_init * np.random.uniform(0.3, 3.0)
        b0B0 = b0_b_init * np.random.uniform(0.3, 3.0)
        b1A0 = np.random.uniform(-1.0, 1.0)
        b1B0 = np.random.uniform(-1.0, 1.0)
        x0 = [r0, phi0, sA0, b0A0, b1A0, sB0, b0B0, b1B0]

        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 500})
            if result.fun > 1e10 or not result.success:
                result_powell = minimize(objective, x0, method='Powell',
                                       options={'maxiter': 1000})
                if result_powell.fun < result.fun:
                    result = result_powell

            starts_used += 1

            if result.fun < best_nll - NLL_IMPROVEMENT_THRESHOLD:
                best_nll = result.fun
                best_params = result.x
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if adaptive and i >= n_starts and no_improvement_count >= ADAPTIVE_PATIENCE:
                break

        except Exception:
            continue

    if best_params is None:
        return {'converged': False, 'nll': 1e10, 'starts_used': starts_used}

    r, phi, sA, b0A, b1A, sB, b0B, b1B = best_params
    y_pred_a = model_a(r, phi, sA, b0A, b1A)
    y_pred_b = model_b(r, phi, sB, b0B, b1B)

    # Use chi2_sigma (stat_error if available) for chi2 computation
    chi2_a = compute_chi2(y_a, y_pred_a, chi2_sigma_a, type_a)
    chi2_b = compute_chi2(y_b, y_pred_b, chi2_sigma_b, type_b)
    deviance_a = compute_deviance(y_a, y_pred_a, type_a)
    deviance_b = compute_deviance(y_b, y_pred_b, type_b)

    dof_a = len(x_a) - 4
    dof_b = len(x_b) - 4

    return {
        'converged': True,
        'nll': best_nll,
        'r': r,
        'phi': phi,
        'chi2_a': chi2_a,
        'chi2_b': chi2_b,
        'deviance_a': deviance_a,
        'deviance_b': deviance_b,
        'dof_a': dof_a,
        'dof_b': dof_b,
        'chi2_dof_a': chi2_a / max(dof_a, 1),
        'chi2_dof_b': chi2_b / max(dof_b, 1),
        'deviance_dof_a': deviance_a / max(dof_a, 1) if type_a == 'poisson' else 0.0,
        'deviance_dof_b': deviance_b / max(dof_b, 1) if type_b == 'poisson' else 0.0,
        'y_pred_a': y_pred_a,
        'y_pred_b': y_pred_b,
        'starts_used': starts_used
    }


def compute_lambda(nll_constrained: float, nll_unconstrained: float) -> float:
    """Compute test statistic Lambda = 2*(NLL_con - NLL_unc)."""
    return max(0, 2 * (nll_constrained - nll_unconstrained))


def compute_wilks_pvalue(Lambda: float, dof_diff: int = DOF_DIFF) -> float:
    """Compute asymptotic p-value using Wilks' theorem."""
    if np.isnan(Lambda) or Lambda < 0:
        return np.nan
    return 1 - chi2_dist.cdf(Lambda, dof_diff)


def check_fit_health(fit_unc: Dict, fit_con: Dict,
                     chi2_min: float = CHI2_DOF_MIN,
                     chi2_max: float = CHI2_DOF_MAX,
                     deviance_max: float = DEVIANCE_DOF_MAX) -> Tuple[str, Dict]:
    """Check fit health using chi2/dof and deviance/dof gates."""
    chi2_dof_a = fit_unc.get('chi2_dof_a', 0)
    chi2_dof_b = fit_unc.get('chi2_dof_b', 0)
    deviance_dof_a = fit_unc.get('deviance_dof_a', 0)
    deviance_dof_b = fit_unc.get('deviance_dof_b', 0)

    metrics = {
        'chi2_dof_a': chi2_dof_a,
        'chi2_dof_b': chi2_dof_b,
        'deviance_dof_a': deviance_dof_a,
        'deviance_dof_b': deviance_dof_b
    }

    if chi2_dof_a < chi2_min or chi2_dof_b < chi2_min:
        return 'UNDERCONSTRAINED', metrics

    if chi2_dof_a > chi2_max or chi2_dof_b > chi2_max:
        return 'MISMATCH', metrics

    if deviance_dof_a > deviance_max or deviance_dof_b > deviance_max:
        return 'MISMATCH', metrics

    return 'PASS', metrics


def generate_bootstrap_data_gaussian(ch_data: Dict, y_pred: np.ndarray,
                                     rng: np.random.Generator) -> Dict:
    """
    FIX A: Generate bootstrap data for Gaussian channel using STAT-ONLY noise.
    Hold nuisances fixed at MLE values.
    """
    stat_error = ch_data.get('stat_error', ch_data['sigma'])

    # Generate with stat-only noise around the predicted values
    y_boot = rng.normal(y_pred, stat_error)

    return {
        'x': ch_data['x'],
        'y': y_boot,
        'sigma': ch_data['sigma'],  # Keep original sigma for fitting
        'stat_error': stat_error,
        'type': 'gaussian'
    }


def generate_bootstrap_data_poisson(ch_data: Dict, y_pred: np.ndarray,
                                    rng: np.random.Generator) -> Dict:
    """Generate bootstrap data for Poisson channel."""
    y_pred_safe = np.maximum(y_pred, 0.1)
    y_boot = rng.poisson(y_pred_safe)
    sigma_boot = np.sqrt(np.maximum(y_boot, 1))

    return {
        'x': ch_data['x'],
        'y': y_boot,
        'sigma': sigma_boot,
        'type': 'poisson'
    }


def generate_bootstrap_data(ch_data: Dict, y_pred: np.ndarray,
                            rng: np.random.Generator) -> Dict:
    """Generic bootstrap data generator - dispatches by data type."""
    if ch_data['type'] == 'gaussian':
        return generate_bootstrap_data_gaussian(ch_data, y_pred, rng)
    else:
        return generate_bootstrap_data_poisson(ch_data, y_pred, rng)


def bootstrap_pvalue_calibrated(dataset: Dict, observed_lambda: float,
                                fit_con: Dict, fit_unc: Dict,
                                n_bootstrap: int = DEFAULT_BOOTSTRAP,
                                n_starts: int = DEFAULT_STARTS) -> Tuple[float, np.ndarray, int, int]:
    """
    FIX A+B: Compute calibrated bootstrap p-value.

    - Uses stat-only noise for Gaussian (Fix A)
    - Uses same optimizer settings as main fit (Fix B)
    - Enforces nestedness per replicate (Fix B)

    Returns:
        (p_value, bootstrap_lambdas, n_exceeded, n_optimizer_fails)
    """
    bw_params = dataset['bw_params']
    ch_a = dataset['channelA']
    ch_b = dataset['channelB']

    # Get predicted values from constrained fit for bootstrap generation
    y_pred_a = fit_con.get('y_pred_a')
    y_pred_b = fit_con.get('y_pred_b')

    if y_pred_a is None or y_pred_b is None:
        return np.nan, np.array([]), 0, 0

    bootstrap_lambdas = []
    n_exceeded = 0
    n_optimizer_fails = 0

    for b in range(n_bootstrap):
        rng = np.random.default_rng(b + 54321)

        # FIX A: Generate bootstrap data appropriately for each data type
        if ch_a['type'] == 'gaussian':
            boot_a = generate_bootstrap_data_gaussian(ch_a, y_pred_a, rng)
        else:
            boot_a = generate_bootstrap_data_poisson(ch_a, y_pred_a, rng)

        if ch_b['type'] == 'gaussian':
            boot_b = generate_bootstrap_data_gaussian(ch_b, y_pred_b, rng)
        else:
            boot_b = generate_bootstrap_data_poisson(ch_b, y_pred_b, rng)

        boot_dataset = {
            'channelA': boot_a,
            'channelB': boot_b,
            'bw_params': bw_params
        }

        # FIX B: Use same optimizer settings as main fit
        fit_unc_b = fit_joint_unconstrained(boot_dataset, n_starts=n_starts, adaptive=True)
        fit_con_b = fit_joint_constrained(boot_dataset, n_starts=n_starts, adaptive=True)

        if not fit_unc_b['converged'] or not fit_con_b['converged']:
            n_optimizer_fails += 1
            continue

        # FIX B: Enforce nestedness
        if fit_unc_b['nll'] > fit_con_b['nll'] + NESTEDNESS_TOL:
            # Retry with higher budget
            fit_unc_b = fit_joint_unconstrained(boot_dataset, n_starts=n_starts*2, adaptive=True)
            if fit_unc_b['nll'] > fit_con_b['nll'] + NESTEDNESS_TOL:
                n_optimizer_fails += 1
                continue

        lambda_b = compute_lambda(fit_con_b['nll'], fit_unc_b['nll'])
        bootstrap_lambdas.append(lambda_b)

        if lambda_b >= observed_lambda:
            n_exceeded += 1

    bootstrap_lambdas = np.array(bootstrap_lambdas)

    if len(bootstrap_lambdas) == 0:
        return np.nan, bootstrap_lambdas, 0, n_optimizer_fails

    p_value = n_exceeded / len(bootstrap_lambdas)
    return p_value, bootstrap_lambdas, n_exceeded, n_optimizer_fails


def run_calibration_trial(dataset: Dict, n_bootstrap: int = DEFAULT_BOOTSTRAP,
                          n_starts: int = DEFAULT_STARTS) -> Dict:
    """
    Run a single calibration trial with full diagnostics.

    Returns dict with all trace information.
    """
    # Main fits
    fit_unc = fit_joint_unconstrained(dataset, n_starts=n_starts, adaptive=True)
    fit_con = fit_joint_constrained(dataset, n_starts=n_starts, adaptive=True)

    if not fit_unc['converged'] or not fit_con['converged']:
        return {
            'converged': False,
            'nll_con': np.nan,
            'nll_unc': np.nan,
            'lambda_obs': np.nan,
            'p_boot': np.nan,
            'p_wilks': np.nan,
            'gates': 'FIT_FAILED',
            'pass_trial': False
        }

    # Enforce nestedness on main fit
    optimizer_retries = 0
    if fit_unc['nll'] > fit_con['nll'] + NESTEDNESS_TOL:
        fit_unc = fit_joint_unconstrained(dataset, n_starts=n_starts*2, adaptive=True)
        optimizer_retries += 1

    nll_con = fit_con['nll']
    nll_unc = fit_unc['nll']
    lambda_obs = compute_lambda(nll_con, nll_unc)
    p_wilks = compute_wilks_pvalue(lambda_obs, DOF_DIFF)

    # Check gates
    gates, health_metrics = check_fit_health(fit_unc, fit_con)

    # FIX C: Only compute p_boot for PASS trials
    if gates == 'PASS':
        p_boot, boot_lambdas, k_exceed, n_opt_fail = bootstrap_pvalue_calibrated(
            dataset, lambda_obs, fit_con, fit_unc,
            n_bootstrap=n_bootstrap, n_starts=n_starts
        )
    else:
        p_boot = np.nan
        boot_lambdas = np.array([])
        k_exceed = 0
        n_opt_fail = 0

    pass_trial = (gates == 'PASS')
    rejected = (pass_trial and not np.isnan(p_boot) and p_boot < 0.05)

    return {
        'converged': True,
        'nll_con': nll_con,
        'nll_unc': nll_unc,
        'lambda_obs': lambda_obs,
        'p_boot': p_boot,
        'p_wilks': p_wilks,
        'k_exceed': k_exceed,
        'n_boot': len(boot_lambdas),
        'chi2_dof_a': health_metrics['chi2_dof_a'],
        'chi2_dof_b': health_metrics['chi2_dof_b'],
        'dev_dof_a': health_metrics['deviance_dof_a'],
        'dev_dof_b': health_metrics['deviance_dof_b'],
        'gates': gates,
        'starts_used': fit_unc.get('starts_used', 0) + fit_con.get('starts_used', 0),
        'optimizer_retries': optimizer_retries + n_opt_fail,
        'pass_trial': pass_trial,
        'rejected': rejected,
        'bootstrap_lambdas': boot_lambdas.tolist() if len(boot_lambdas) > 0 else []
    }


def sanity_injection_check(test_config: Dict) -> Tuple[bool, str]:
    """
    FIX E: Verify generator and fitter use same BW convention.
    Generate noiseless M0 data and check Lambda ~ 0.
    """
    from sim_generate import generate_dataset_M0, BWParams as GenBWParams

    # Create a deterministic "noiseless" dataset
    bw_cfg = test_config.get('bw_params', {})
    R_cfg = test_config['R_true']
    R_true = complex_from_polar(R_cfg['r'], R_cfg['phi_deg'])

    # Use the generator to create expected values
    rng = np.random.default_rng(99999)
    dataset = generate_dataset_M0(test_config, scale_factor=10.0, rng=rng)

    # Replace observed with true (noiseless)
    dataset['channelA']['y'] = dataset['channelA']['y_true'].copy()
    dataset['channelB']['y'] = dataset['channelB']['y_true'].copy()

    # Fit
    fit_unc = fit_joint_unconstrained(dataset, n_starts=50, adaptive=False)
    fit_con = fit_joint_constrained(dataset, n_starts=50, adaptive=False)

    if not fit_unc['converged'] or not fit_con['converged']:
        return False, "Sanity fit failed to converge"

    lambda_sanity = compute_lambda(fit_con['nll'], fit_unc['nll'])

    # Lambda should be very small for noiseless M0 data
    if lambda_sanity > 1.0:
        return False, f"Sanity check failed: Lambda = {lambda_sanity:.3f} > 1.0 (model mismatch)"

    return True, f"Sanity check passed: Lambda = {lambda_sanity:.4f}"


def write_trace_csv(traces: List[Dict], filepath: str):
    """Write calibration traces to CSV."""
    if not traces:
        return

    fieldnames = ['test', 'trial', 'nll_con', 'nll_unc', 'lambda_obs', 'p_boot', 'p_wilks',
                  'k_exceed', 'n_boot', 'chi2_dof_a', 'chi2_dof_b', 'dev_dof_a', 'dev_dof_b',
                  'gates', 'starts_used', 'optimizer_retries', 'converged', 'rejected', 'pass_trial']

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(traces)


def compute_calibration_summary(traces: List[Dict]) -> Dict:
    """
    FIX C: Compute calibration summary with gates applied before Type-I counting.
    """
    pass_trials = [t for t in traces if t.get('pass_trial', False)]
    all_trials = [t for t in traces if t.get('converged', False)]

    n_total = len(all_trials)
    n_pass = len(pass_trials)
    pass_rate = n_pass / n_total if n_total > 0 else 0.0

    # Type-I only over PASS trials
    if n_pass > 0:
        n_rejected = sum(1 for t in pass_trials if t.get('rejected', False))
        type_i = n_rejected / n_pass

        # Lambda statistics
        lambda_obs = [t['lambda_obs'] for t in pass_trials if not np.isnan(t.get('lambda_obs', np.nan))]
        p_boots = [t['p_boot'] for t in pass_trials if not np.isnan(t.get('p_boot', np.nan))]

        # Bootstrap lambda mean (from individual trials)
        boot_lambda_means = []
        for t in pass_trials:
            boot_lambdas = t.get('bootstrap_lambdas', [])
            if boot_lambdas:
                boot_lambda_means.append(np.mean(boot_lambdas))

        # KS test of p_boot against Uniform(0,1)
        if len(p_boots) >= 10:
            ks_stat, ks_pval = kstest(p_boots, 'uniform')
        else:
            ks_stat, ks_pval = np.nan, np.nan

        return {
            'n_total': n_total,
            'n_pass': n_pass,
            'pass_rate': pass_rate,
            'n_rejected': n_rejected,
            'type_i': type_i,
            'type_i_se': np.sqrt(type_i * (1 - type_i) / n_pass) if n_pass > 0 else 0,
            'lambda_obs_mean': np.mean(lambda_obs) if lambda_obs else np.nan,
            'lambda_obs_median': np.median(lambda_obs) if lambda_obs else np.nan,
            'lambda_obs_std': np.std(lambda_obs) if lambda_obs else np.nan,
            'lambda_boot_mean': np.mean(boot_lambda_means) if boot_lambda_means else np.nan,
            'p_boot_mean': np.mean(p_boots) if p_boots else np.nan,
            'p_boot_median': np.median(p_boots) if p_boots else np.nan,
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'p_boots': p_boots,
            'lambda_obs_list': lambda_obs
        }
    else:
        return {
            'n_total': n_total,
            'n_pass': 0,
            'pass_rate': 0.0,
            'type_i': np.nan,
            'type_i_se': np.nan,
            'lambda_obs_mean': np.nan,
            'lambda_boot_mean': np.nan,
            'ks_stat': np.nan,
            'ks_pval': np.nan,
            'p_boots': [],
            'lambda_obs_list': []
        }


if __name__ == "__main__":
    print("sim_fit_v3.py - Calibrated fitter loaded")
    print(f"  DOF_DIFF = {DOF_DIFF}")
    print(f"  DEFAULT_STARTS = {DEFAULT_STARTS}")
    print(f"  DEFAULT_BOOTSTRAP = {DEFAULT_BOOTSTRAP}")
    print(f"  ADAPTIVE_MAX_STARTS = {ADAPTIVE_MAX_STARTS}")
