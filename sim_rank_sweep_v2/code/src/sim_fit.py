#!/usr/bin/env python3
"""
sim_fit.py - Publication-grade fitting for rank-1 bottleneck test

Key features:
- dof_diff = 2 for complex R constraint
- Bootstrap p-values (default 200 replicates)
- Multi-start optimizer (default 80 starts)
- Fit-health gates: chi2/dof in [0.5, 3.0], deviance/dof < 3.0
- Identifiability diagnostics (multimodality detection)
- Powell fallback for robustness
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import warnings
from sim_generate import BWParams, breit_wigner, complex_from_polar

warnings.filterwarnings('ignore')

# Constants
DOF_DIFF = 2  # Degrees of freedom for complex R constraint (magnitude + phase)
DEFAULT_BOOTSTRAP = 200
DEFAULT_STARTS = 80
CHI2_DOF_MIN = 0.5
CHI2_DOF_MAX = 3.0
DEVIANCE_DOF_MAX = 3.0


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
class FullFitResult:
    """Container for complete rank-1 test results"""
    converged: bool
    Lambda: float
    p_boot: float
    p_wilks: float
    verdict: str
    gates: str

    # Fit details
    fit_constrained: Dict = field(default_factory=dict)
    fit_unconstrained: Dict = field(default_factory=dict)

    # Health metrics
    chi2_dof_a: float = 0.0
    chi2_dof_b: float = 0.0
    deviance_dof_a: float = 0.0
    deviance_dof_b: float = 0.0

    # Identifiability
    identifiable: bool = True
    ident_reason: str = ""
    r_spread: float = 0.0
    phi_spread: float = 0.0

    # Bootstrap details
    n_bootstrap: int = 0
    bootstrap_lambdas: List = field(default_factory=list)


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
    """
    Compute deviance for fit health assessment.
    For Poisson: 2 * sum(y_obs * log(y_obs/y_pred) - (y_obs - y_pred))
    For Gaussian: chi2 (deviance = chi2 for Gaussian)
    """
    if data_type == 'poisson':
        y_obs_safe = np.maximum(y_obs, 0.5)
        y_pred_safe = np.maximum(y_pred, 0.5)
        return 2 * np.sum(y_obs_safe * np.log(y_obs_safe / y_pred_safe) - (y_obs - y_pred))
    else:
        return 0.0  # Deviance equals chi2 for Gaussian, handled separately


def intensity_shape(x: np.ndarray, R: complex, bw_params: BWParams) -> np.ndarray:
    """
    Compute intensity shape (unnormalized).
    I(x) = |BW1(x) + R * BW2(x)|^2
    """
    bw1 = breit_wigner(x, bw_params.m1, bw_params.gamma1)
    bw2 = breit_wigner(x, bw_params.m2, bw_params.gamma2)
    return np.abs(bw1 + R * bw2)**2


def fit_channel_with_bg(x: np.ndarray, y_obs: np.ndarray, sigma: np.ndarray,
                        bw_params: BWParams, data_type: str,
                        n_restarts: int = DEFAULT_STARTS,
                        stat_error: np.ndarray = None) -> Dict:
    """
    Fit single channel with signal + polynomial background.
    Parameters: r, phi, scale, b0, b1
    Uses multi-start optimization with L-BFGS-B + Powell fallback.
    """
    def model(params):
        r, phi, scale, b0, b1 = params
        if r <= 0 or scale <= 0:
            return np.full_like(x, 1e10)
        R = complex_from_polar(r, phi)
        signal = intensity_shape(x, R, bw_params)
        signal_max = np.max(signal)
        if signal_max > 0:
            signal = signal / signal_max
        bg = b0 + b1 * (x - np.mean(x))
        return scale * signal + np.maximum(bg, 0)

    def objective(params):
        y_pred = model(params)
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return 1e15
        if data_type == 'poisson':
            return compute_nll_poisson(y_obs, y_pred)
        else:
            return compute_nll_gaussian(y_obs, y_pred, sigma)

    best_nll = np.inf
    best_params = None
    nll_list = []

    # Estimate initial scale
    scale_init = np.max(y_obs) * 0.7
    b0_init = max(np.min(y_obs) * 0.3, 0.1)

    # Wider bounds for multi-modal search
    bounds = [(0.01, 5.0), (-180, 180),
              (0.1, scale_init * 15),
              (0, b0_init * 10), (-5, 5)]

    for i in range(n_restarts):
        r0 = np.random.uniform(0.2, 2.0)
        phi0 = np.random.uniform(-180, 180)
        scale0 = scale_init * np.random.uniform(0.3, 3.0)
        b0_0 = b0_init * np.random.uniform(0.3, 3.0)
        b1_0 = np.random.uniform(-1.0, 1.0)

        x0 = [r0, phi0, scale0, b0_0, b1_0]

        try:
            # Try L-BFGS-B first
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 500})

            # Powell fallback if L-BFGS-B gives poor result
            if result.fun > 1e10 or not result.success:
                result_powell = minimize(objective, x0, method='Powell',
                                       options={'maxiter': 1000})
                if result_powell.fun < result.fun:
                    result = result_powell

            nll_list.append(result.fun)
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except Exception:
            continue

    if best_params is None:
        return {'converged': False, 'nll': 1e10}

    r, phi, scale, b0, b1 = best_params
    y_pred = model(best_params)
    # Use stat_error for chi2 if available (Gaussian with correlated systematics)
    chi2_sigma = stat_error if stat_error is not None else sigma
    chi2_val = compute_chi2(y_obs, y_pred, chi2_sigma, data_type)
    deviance = compute_deviance(y_obs, y_pred, data_type)
    dof = len(x) - 5  # 5 free parameters

    # Compute NLL spread for identifiability check
    nll_spread = np.std(nll_list) if len(nll_list) > 1 else 0.0

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
        'nll_spread': nll_spread
    }


def fit_joint_unconstrained(dataset: Dict, n_restarts: int = DEFAULT_STARTS) -> Dict:
    """
    Fit both channels with separate R_A, R_B.
    Each channel has: r, phi, scale, b0, b1 (10 total params).
    """
    bw_params = dataset['bw_params']
    ch_a = dataset['channelA']
    ch_b = dataset['channelB']

    # Fit each channel independently
    # Use stat_error for chi2 if available (Gaussian with correlated systematics)
    fit_a = fit_channel_with_bg(
        ch_a['x'], ch_a['y'], ch_a['sigma'],
        bw_params, ch_a['type'], n_restarts,
        stat_error=ch_a.get('stat_error')
    )

    fit_b = fit_channel_with_bg(
        ch_b['x'], ch_b['y'], ch_b['sigma'],
        bw_params, ch_b['type'], n_restarts,
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
        'y_pred_b': fit_b.get('y_pred')
    }


def fit_joint_constrained(dataset: Dict, n_restarts: int = DEFAULT_STARTS) -> Dict:
    """
    Fit both channels with SHARED R (rank-1 constraint).
    Parameters: r, phi (shared), scale_A, b0_A, b1_A, scale_B, b0_B, b1_B (8 total).
    This is the key rank-1 factorization constraint.
    """
    bw_params = dataset['bw_params']
    ch_a = dataset['channelA']
    ch_b = dataset['channelB']

    x_a, y_a, sigma_a = ch_a['x'], ch_a['y'], ch_a['sigma']
    x_b, y_b, sigma_b = ch_b['x'], ch_b['y'], ch_b['sigma']
    type_a, type_b = ch_a['type'], ch_b['type']

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
            nll_a = compute_nll_gaussian(y_a, y_pred_a, sigma_a)

        if type_b == 'poisson':
            nll_b = compute_nll_poisson(y_b, y_pred_b)
        else:
            nll_b = compute_nll_gaussian(y_b, y_pred_b, sigma_b)

        return nll_a + nll_b

    best_nll = np.inf
    best_params = None
    all_fits = []  # Store all converged fits for identifiability check

    scale_a_init = max(np.max(y_a) * 0.7, 1.0)
    scale_b_init = max(np.max(y_b) * 0.7, 1.0)
    b0_a_init = max(np.min(y_a) * 0.3, 0.1)
    b0_b_init = max(np.min(y_b) * 0.3, 0.1)

    bounds = [(0.01, 5.0), (-180, 180),
              (0.1, scale_a_init * 15), (0, b0_a_init * 10), (-5, 5),
              (0.1, scale_b_init * 15), (0, b0_b_init * 10), (-5, 5)]

    for i in range(n_restarts):
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
            # L-BFGS-B primary
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 500})

            # Powell fallback
            if result.fun > 1e10 or not result.success:
                result_powell = minimize(objective, x0, method='Powell',
                                       options={'maxiter': 1000})
                if result_powell.fun < result.fun:
                    result = result_powell

            if result.fun < 1e10:
                all_fits.append((result.fun, result.x[0], result.x[1]))

            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except Exception:
            continue

    if best_params is None:
        return {'converged': False, 'nll': 1e10}

    r, phi, sA, b0A, b1A, sB, b0B, b1B = best_params

    y_pred_a = model_a(r, phi, sA, b0A, b1A)
    y_pred_b = model_b(r, phi, sB, b0B, b1B)

    # Use stat_error for chi2 if available (Gaussian with correlated systematics)
    chi2_sigma_a = ch_a.get('stat_error', sigma_a)
    chi2_sigma_b = ch_b.get('stat_error', sigma_b)
    chi2_a = compute_chi2(y_a, y_pred_a, chi2_sigma_a, type_a)
    chi2_b = compute_chi2(y_b, y_pred_b, chi2_sigma_b, type_b)
    deviance_a = compute_deviance(y_a, y_pred_a, type_a)
    deviance_b = compute_deviance(y_b, y_pred_b, type_b)

    # DoF: shared r,phi (2) + per-channel scale,b0,b1 (3 each) = 8 total
    # For chi2: n_bins - n_params_per_channel
    dof_a = len(x_a) - 4  # r, phi shared; scale, b0, b1 = 3 local + 1 effective for shared
    dof_b = len(x_b) - 4

    # Compute multimodality metrics from all fits
    r_spread = 0.0
    phi_spread = 0.0
    nll_threshold = 2.0  # fits within this NLL of best are considered equivalent

    if len(all_fits) > 1:
        good_fits = [(f[1], f[2]) for f in all_fits if f[0] < best_nll + nll_threshold]
        if len(good_fits) > 1:
            rs = np.array([f[0] for f in good_fits])
            phis = np.array([f[1] for f in good_fits])
            r_spread = np.std(rs) / np.mean(rs) if np.mean(rs) > 0 else 0.0
            phi_spread = np.std(phis)

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
        'r_spread': r_spread,
        'phi_spread': phi_spread,
        'n_good_fits': len([f for f in all_fits if f[0] < best_nll + nll_threshold])
    }


def compute_lambda(nll_constrained: float, nll_unconstrained: float) -> float:
    """
    Compute test statistic Lambda = 2*(NLL_constrained - NLL_unconstrained).
    Enforces Lambda >= 0 (constrained should never be better than unconstrained).
    """
    return max(0, 2 * (nll_constrained - nll_unconstrained))


def compute_wilks_pvalue(Lambda: float, dof_diff: int = DOF_DIFF) -> float:
    """
    Compute asymptotic p-value using Wilks' theorem.
    Lambda ~ chi2(dof_diff) under H0.
    """
    if np.isnan(Lambda) or Lambda < 0:
        return np.nan
    return 1 - chi2_dist.cdf(Lambda, dof_diff)


def check_fit_health(fit_unc: Dict, fit_con: Dict,
                     chi2_min: float = CHI2_DOF_MIN,
                     chi2_max: float = CHI2_DOF_MAX,
                     deviance_max: float = DEVIANCE_DOF_MAX) -> Tuple[str, Dict]:
    """
    Check fit health using chi2/dof and deviance/dof gates.

    Returns:
        Tuple of (gate_status, metrics_dict)
        gate_status in ['PASS', 'UNDERCONSTRAINED', 'MISMATCH']
    """
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

    # Check underconstrained (chi2/dof too low)
    if chi2_dof_a < chi2_min or chi2_dof_b < chi2_min:
        return 'UNDERCONSTRAINED', metrics

    # Check model mismatch (chi2/dof or deviance/dof too high)
    if chi2_dof_a > chi2_max or chi2_dof_b > chi2_max:
        return 'MISMATCH', metrics

    # Deviance check for Poisson data
    if deviance_dof_a > deviance_max or deviance_dof_b > deviance_max:
        return 'MISMATCH', metrics

    return 'PASS', metrics


def check_identifiability(fit_con: Dict,
                          r_spread_max: float = 0.3,
                          phi_spread_max: float = 90.0) -> Tuple[bool, str, Dict]:
    """
    Check for phase identifiability (multimodality).

    Returns:
        Tuple of (identifiable, reason, metrics)
    """
    r_spread = fit_con.get('r_spread', 0.0)
    phi_spread = fit_con.get('phi_spread', 0.0)
    n_good = fit_con.get('n_good_fits', 1)

    metrics = {
        'r_spread': r_spread,
        'phi_spread': phi_spread,
        'n_equivalent_minima': n_good
    }

    if phi_spread > phi_spread_max:
        return False, 'phase_multimodal', metrics

    if r_spread > r_spread_max:
        return False, 'amplitude_multimodal', metrics

    return True, 'identifiable', metrics


def bootstrap_pvalue(dataset: Dict, observed_lambda: float,
                     n_bootstrap: int = DEFAULT_BOOTSTRAP,
                     n_restarts: int = 20) -> Tuple[float, np.ndarray]:
    """
    Compute bootstrap p-value under H0 (rank-1 is true).

    Generates bootstrap replicates from the null model (shared R)
    and computes the fraction that exceed the observed Lambda.
    """
    from sim_generate import generate_channel_data

    # Get best-fit shared R from constrained fit
    fit_con = fit_joint_constrained(dataset, n_restarts=n_restarts)
    if not fit_con['converged']:
        return np.nan, np.array([])

    r_h0 = fit_con['r']
    phi_h0 = fit_con['phi']
    R_h0 = complex_from_polar(r_h0, phi_h0)

    bw_params = dataset['bw_params']
    ch_a_data = dataset['channelA']
    ch_b_data = dataset['channelB']

    bootstrap_lambdas = []

    for b in range(n_bootstrap):
        rng = np.random.default_rng(b + 12345)

        # Config for bootstrap samples - match observed data characteristics
        ch_a_config = {
            'type': ch_a_data['type'],
            'mean_count_scale': np.mean(ch_a_data['y']),
            'bg_level': 0.3,
            'error_frac_min': 0.03,
            'error_frac_max': 0.06
        }
        ch_b_config = {
            'type': ch_b_data['type'],
            'mean_count_scale': np.mean(ch_b_data['y']),
            'bg_level': 0.3,
            'error_frac_min': 0.04,
            'error_frac_max': 0.08
        }

        boot_a = generate_channel_data(ch_a_data['x'], R_h0, bw_params, ch_a_config, 1.0, rng)
        boot_b = generate_channel_data(ch_b_data['x'], R_h0, bw_params, ch_b_config, 1.0, rng)

        boot_a['x'] = ch_a_data['x']
        boot_b['x'] = ch_b_data['x']

        boot_dataset = {
            'channelA': boot_a,
            'channelB': boot_b,
            'bw_params': bw_params
        }

        # Fewer restarts for bootstrap (speed)
        fit_unc_b = fit_joint_unconstrained(boot_dataset, n_restarts=10)
        fit_con_b = fit_joint_constrained(boot_dataset, n_restarts=10)

        if fit_unc_b['converged'] and fit_con_b['converged']:
            lambda_b = compute_lambda(fit_con_b['nll'], fit_unc_b['nll'])
            bootstrap_lambdas.append(lambda_b)

    bootstrap_lambdas = np.array(bootstrap_lambdas)

    if len(bootstrap_lambdas) == 0:
        return np.nan, bootstrap_lambdas

    p_value = np.mean(bootstrap_lambdas >= observed_lambda)
    return p_value, bootstrap_lambdas


def determine_verdict(p_boot: float, gates: str, identifiable: bool) -> str:
    """
    Determine verdict based on p-value, fit health, and identifiability.

    Verdicts:
    - NOT_REJECTED: p >= 0.05 and gates PASS and identifiable
    - DISFAVORED: p < 0.05 and gates PASS
    - INCONCLUSIVE: gates != PASS or not identifiable
    - MODEL_MISMATCH: gates == MISMATCH
    """
    if gates == 'MISMATCH':
        return 'MODEL_MISMATCH'

    if gates == 'UNDERCONSTRAINED':
        return 'INCONCLUSIVE'

    if not identifiable:
        return 'INCONCLUSIVE'

    if np.isnan(p_boot):
        return 'INCONCLUSIVE'

    if p_boot < 0.05:
        return 'DISFAVORED'

    return 'NOT_REJECTED'


def run_full_fit(dataset: Dict,
                 n_bootstrap: int = DEFAULT_BOOTSTRAP,
                 n_restarts: int = DEFAULT_STARTS,
                 chi2_min: float = CHI2_DOF_MIN,
                 chi2_max: float = CHI2_DOF_MAX,
                 deviance_max: float = DEVIANCE_DOF_MAX) -> FullFitResult:
    """
    Run complete publication-grade fitting procedure.

    Steps:
    1. Fit unconstrained (separate R_A, R_B)
    2. Fit constrained (shared R)
    3. Compute Lambda test statistic
    4. Check fit-health gates
    5. Check identifiability
    6. Compute bootstrap p-value
    7. Determine verdict

    Returns:
        FullFitResult with all metrics
    """
    # Unconstrained fit
    fit_unc = fit_joint_unconstrained(dataset, n_restarts=n_restarts)

    # Constrained fit
    fit_con = fit_joint_constrained(dataset, n_restarts=n_restarts)

    if not fit_unc['converged'] or not fit_con['converged']:
        return FullFitResult(
            converged=False,
            Lambda=np.nan,
            p_boot=np.nan,
            p_wilks=np.nan,
            verdict='INCONCLUSIVE',
            gates='FIT_FAILED'
        )

    # Compute Lambda (dof_diff = 2 for complex R)
    Lambda = compute_lambda(fit_con['nll'], fit_unc['nll'])

    # Wilks p-value (asymptotic reference)
    p_wilks = compute_wilks_pvalue(Lambda, DOF_DIFF)

    # Check fit-health gates
    gates, health_metrics = check_fit_health(fit_unc, fit_con, chi2_min, chi2_max, deviance_max)

    # Check identifiability
    identifiable, ident_reason, ident_metrics = check_identifiability(fit_con)

    # Bootstrap p-value (primary inference)
    if gates == 'PASS':
        p_boot, boot_lambdas = bootstrap_pvalue(dataset, Lambda, n_bootstrap=n_bootstrap)
    else:
        p_boot = np.nan
        boot_lambdas = np.array([])

    # Determine verdict
    verdict = determine_verdict(p_boot, gates, identifiable)

    return FullFitResult(
        converged=True,
        Lambda=Lambda,
        p_boot=p_boot,
        p_wilks=p_wilks,
        verdict=verdict,
        gates=gates,
        fit_constrained=fit_con,
        fit_unconstrained=fit_unc,
        chi2_dof_a=health_metrics['chi2_dof_a'],
        chi2_dof_b=health_metrics['chi2_dof_b'],
        deviance_dof_a=health_metrics['deviance_dof_a'],
        deviance_dof_b=health_metrics['deviance_dof_b'],
        identifiable=identifiable,
        ident_reason=ident_reason,
        r_spread=ident_metrics['r_spread'],
        phi_spread=ident_metrics['phi_spread'],
        n_bootstrap=len(boot_lambdas),
        bootstrap_lambdas=boot_lambdas.tolist() if len(boot_lambdas) > 0 else []
    )


def run_quick_fit(dataset: Dict, n_restarts: int = 20) -> Dict:
    """
    Quick fit without bootstrap (for sweep iterations).
    Returns Lambda, gates, basic metrics.
    """
    fit_unc = fit_joint_unconstrained(dataset, n_restarts=n_restarts)
    fit_con = fit_joint_constrained(dataset, n_restarts=n_restarts)

    if not fit_unc['converged'] or not fit_con['converged']:
        return {
            'converged': False,
            'Lambda': np.nan,
            'gates': 'FIT_FAILED',
            'p_wilks': np.nan
        }

    Lambda = compute_lambda(fit_con['nll'], fit_unc['nll'])
    p_wilks = compute_wilks_pvalue(Lambda, DOF_DIFF)
    gates, metrics = check_fit_health(fit_unc, fit_con)
    identifiable, ident_reason, _ = check_identifiability(fit_con)

    return {
        'converged': True,
        'Lambda': Lambda,
        'p_wilks': p_wilks,
        'gates': gates,
        'identifiable': identifiable,
        'ident_reason': ident_reason,
        'chi2_dof_a': metrics['chi2_dof_a'],
        'chi2_dof_b': metrics['chi2_dof_b'],
        'deviance_dof_a': metrics['deviance_dof_a'],
        'deviance_dof_b': metrics['deviance_dof_b'],
        'r_shared': fit_con.get('r', np.nan),
        'phi_shared': fit_con.get('phi', np.nan),
        'r_a': fit_unc.get('r_a', np.nan),
        'phi_a': fit_unc.get('phi_a', np.nan),
        'r_b': fit_unc.get('r_b', np.nan),
        'phi_b': fit_unc.get('phi_b', np.nan)
    }


if __name__ == "__main__":
    import json
    import os
    from sim_generate import generate_dataset

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'tests_top3.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    test = config['tests'][0]  # Y-states
    print(f"Testing publication-grade fit for: {test['name']}")
    print(f"dof_diff = {DOF_DIFF} (complex R constraint)")

    # Generate M0 data (rank-1 true)
    dataset = generate_dataset(test, 'M0', scale_factor=2.0, seed=42)

    print("\n--- Quick fit (no bootstrap) ---")
    result = run_quick_fit(dataset, n_restarts=30)
    print(f"  Converged: {result['converged']}")
    print(f"  Lambda: {result['Lambda']:.3f}")
    print(f"  p_wilks: {result['p_wilks']:.4f}")
    print(f"  Gates: {result['gates']}")
    print(f"  Identifiable: {result['identifiable']}")
    print(f"  chi2/dof A: {result['chi2_dof_a']:.2f}")
    print(f"  chi2/dof B: {result['chi2_dof_b']:.2f}")

    print("\n--- Full fit (with bootstrap) ---")
    full_result = run_full_fit(dataset, n_bootstrap=50, n_restarts=30)
    print(f"  Converged: {full_result.converged}")
    print(f"  Lambda: {full_result.Lambda:.3f}")
    print(f"  p_boot: {full_result.p_boot:.4f}")
    print(f"  p_wilks: {full_result.p_wilks:.4f}")
    print(f"  Gates: {full_result.gates}")
    print(f"  Verdict: {full_result.verdict}")
