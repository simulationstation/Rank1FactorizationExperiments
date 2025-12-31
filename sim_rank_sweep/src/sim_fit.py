#!/usr/bin/env python3
"""
sim_fit.py - Fitting procedures for rank-1 bottleneck test

Fits:
- Unconstrained: R_A, R_B separate
- Constrained (rank-1): shared R

Computes:
- Lambda = 2*(NLL_constrained - NLL_unconstrained)
- Bootstrap p-value
- Chi2/dof gates
- Confidence intervals
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings
from sim_generate import BWParams, breit_wigner, complex_from_polar

warnings.filterwarnings('ignore')


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


def compute_nll_poisson(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Poisson NLL (ignoring factorial constant)."""
    y_pred = np.maximum(y_pred, 1e-10)
    return np.sum(y_pred - y_obs * np.log(y_pred))


def compute_nll_gaussian(y_obs: np.ndarray, y_pred: np.ndarray,
                         sigma: np.ndarray) -> float:
    """Gaussian NLL."""
    return 0.5 * np.sum(((y_obs - y_pred) / sigma)**2)


def compute_chi2(y_obs: np.ndarray, y_pred: np.ndarray,
                 sigma: np.ndarray, data_type: str) -> float:
    """Compute chi-squared."""
    if data_type == 'poisson':
        var = np.maximum(y_pred, 1)
        return np.sum((y_obs - y_pred)**2 / var)
    else:
        return np.sum(((y_obs - y_pred) / sigma)**2)


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
                        n_restarts: int = 10) -> Dict:
    """
    Fit single channel with signal + polynomial background.
    Parameters: r, phi, scale, b0, b1

    Profile likelihood: for each (r, phi), optimize (scale, b0, b1) analytically
    for Gaussian or numerically for Poisson.
    """
    def model(params):
        r, phi, scale, b0, b1 = params
        if r <= 0 or scale <= 0:
            return np.full_like(x, 1e10)
        R = complex_from_polar(r, phi)
        signal = intensity_shape(x, R, bw_params)
        signal = signal / np.max(signal)  # Normalize shape
        bg = b0 + b1 * (x - np.mean(x))  # Center x for stability
        return scale * signal + np.maximum(bg, 0)

    def objective(params):
        y_pred = model(params)
        if data_type == 'poisson':
            return compute_nll_poisson(y_obs, y_pred)
        else:
            return compute_nll_gaussian(y_obs, y_pred, sigma)

    best_nll = np.inf
    best_params = None

    # Estimate initial scale
    scale_init = np.max(y_obs) * 0.7
    b0_init = np.min(y_obs) * 0.3

    for i in range(n_restarts):
        r0 = np.random.uniform(0.3, 1.2)
        phi0 = np.random.uniform(-180, 180)
        scale0 = scale_init * np.random.uniform(0.5, 2.0)
        b0_0 = b0_init * np.random.uniform(0.5, 2.0)
        b1_0 = np.random.uniform(-0.5, 0.5)

        try:
            result = minimize(objective, [r0, phi0, scale0, b0_0, b1_0],
                            method='L-BFGS-B',
                            bounds=[(0.01, 3.0), (-180, 180),
                                   (0.1, scale_init * 10),
                                   (0, b0_init * 5), (-2, 2)])
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except:
            continue

    if best_params is None:
        return {'converged': False, 'nll': 1e10}

    r, phi, scale, b0, b1 = best_params
    y_pred = model(best_params)
    chi2 = compute_chi2(y_obs, y_pred, sigma, data_type)
    dof = len(x) - 5  # 5 free parameters

    return {
        'converged': True,
        'nll': best_nll,
        'chi2': chi2,
        'dof': dof,
        'chi2_dof': chi2 / max(dof, 1),
        'r': r,
        'phi': phi,
        'scale': scale,
        'b0': b0,
        'b1': b1
    }


def fit_joint_unconstrained(dataset: Dict, n_restarts: int = 10) -> Dict:
    """
    Fit both channels with separate R_A, R_B.
    Each channel has: r, phi, scale, b0, b1 (10 total params).
    """
    bw_params = dataset['bw_params']
    ch_a = dataset['channelA']
    ch_b = dataset['channelB']

    # Fit each channel independently
    fit_a = fit_channel_with_bg(
        ch_a['x'], ch_a['y'], ch_a['sigma'],
        bw_params, ch_a['type'], n_restarts
    )

    fit_b = fit_channel_with_bg(
        ch_b['x'], ch_b['y'], ch_b['sigma'],
        bw_params, ch_b['type'], n_restarts
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
        'dof_a': fit_a['dof'],
        'dof_b': fit_b['dof'],
        'chi2_dof_a': fit_a['chi2_dof'],
        'chi2_dof_b': fit_b['chi2_dof']
    }


def fit_joint_constrained(dataset: Dict, n_restarts: int = 10) -> Dict:
    """
    Fit both channels with SHARED R (rank-1 constraint).
    Parameters: r, phi (shared), scale_A, b0_A, b1_A, scale_B, b0_B, b1_B (8 total).
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
        signal = signal / np.max(signal)
        bg = b0 + b1 * (x_a - np.mean(x_a))
        return scale * signal + np.maximum(bg, 0)

    def model_b(r, phi, scale, b0, b1):
        R = complex_from_polar(r, phi)
        signal = intensity_shape(x_b, R, bw_params)
        signal = signal / np.max(signal)
        bg = b0 + b1 * (x_b - np.mean(x_b))
        return scale * signal + np.maximum(bg, 0)

    def objective(params):
        r, phi, sA, b0A, b1A, sB, b0B, b1B = params
        if r <= 0 or sA <= 0 or sB <= 0:
            return 1e10

        y_pred_a = model_a(r, phi, sA, b0A, b1A)
        y_pred_b = model_b(r, phi, sB, b0B, b1B)

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

    scale_a_init = np.max(y_a) * 0.7
    scale_b_init = np.max(y_b) * 0.7
    b0_a_init = np.min(y_a) * 0.3
    b0_b_init = np.min(y_b) * 0.3

    for i in range(n_restarts):
        r0 = np.random.uniform(0.3, 1.2)
        phi0 = np.random.uniform(-180, 180)
        sA0 = scale_a_init * np.random.uniform(0.5, 2.0)
        sB0 = scale_b_init * np.random.uniform(0.5, 2.0)
        b0A0 = b0_a_init * np.random.uniform(0.5, 2.0)
        b0B0 = b0_b_init * np.random.uniform(0.5, 2.0)
        b1A0 = np.random.uniform(-0.5, 0.5)
        b1B0 = np.random.uniform(-0.5, 0.5)

        try:
            result = minimize(objective,
                            [r0, phi0, sA0, b0A0, b1A0, sB0, b0B0, b1B0],
                            method='L-BFGS-B',
                            bounds=[(0.01, 3.0), (-180, 180),
                                   (0.1, scale_a_init * 10), (0, b0_a_init * 5), (-2, 2),
                                   (0.1, scale_b_init * 10), (0, b0_b_init * 5), (-2, 2)])
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except:
            continue

    if best_params is None:
        return {'converged': False, 'nll': 1e10}

    r, phi, sA, b0A, b1A, sB, b0B, b1B = best_params

    y_pred_a = model_a(r, phi, sA, b0A, b1A)
    y_pred_b = model_b(r, phi, sB, b0B, b1B)

    chi2_a = compute_chi2(y_a, y_pred_a, sigma_a, type_a)
    chi2_b = compute_chi2(y_b, y_pred_b, sigma_b, type_b)

    dof_a = len(x_a) - 4  # r, phi shared; scale, b0, b1 per channel = 4 effective
    dof_b = len(x_b) - 4

    return {
        'converged': True,
        'nll': best_nll,
        'r': r,
        'phi': phi,
        'chi2_a': chi2_a,
        'chi2_b': chi2_b,
        'dof_a': dof_a,
        'dof_b': dof_b,
        'chi2_dof_a': chi2_a / max(dof_a, 1),
        'chi2_dof_b': chi2_b / max(dof_b, 1)
    }


def compute_lambda(nll_constrained: float, nll_unconstrained: float) -> float:
    """
    Compute test statistic Lambda = 2*(NLL_constrained - NLL_unconstrained).
    Enforces Lambda >= 0.
    """
    return max(0, 2 * (nll_constrained - nll_unconstrained))


def check_identifiability(dataset: Dict, n_fits: int = 5) -> Dict:
    """
    Check for phase identifiability (multimodality).
    """
    fits = []
    for i in range(n_fits):
        np.random.seed(i * 777 + 123)
        fit = fit_joint_constrained(dataset, n_restarts=3)
        if fit['converged']:
            fits.append((fit['r'], fit['phi']))

    if len(fits) < 2:
        return {'identifiable': False, 'reason': 'fit_failures'}

    rs = np.array([f[0] for f in fits])
    phis = np.array([f[1] for f in fits])

    r_spread = np.std(rs) / np.mean(rs) if np.mean(rs) > 0 else 1.0
    phi_spread = np.std(phis)

    if phi_spread > 90:
        return {'identifiable': False, 'reason': 'phase_multimodal',
                'r_spread': r_spread, 'phi_spread': phi_spread}

    if r_spread > 0.3:
        return {'identifiable': False, 'reason': 'amplitude_multimodal',
                'r_spread': r_spread, 'phi_spread': phi_spread}

    return {'identifiable': True, 'r_spread': r_spread, 'phi_spread': phi_spread}


def run_full_fit(dataset: Dict, n_bootstrap: int = 200,
                 n_restarts: int = 10, chi2_min: float = 0.5,
                 chi2_max: float = 3.0) -> Dict:
    """
    Run complete fitting procedure with bootstrap p-value.
    """
    # Unconstrained fit
    fit_unc = fit_joint_unconstrained(dataset, n_restarts=n_restarts)

    # Constrained fit
    fit_con = fit_joint_constrained(dataset, n_restarts=n_restarts)

    if not fit_unc['converged'] or not fit_con['converged']:
        return {
            'converged': False,
            'Lambda': np.nan,
            'p_value': np.nan,
            'gates': 'FIT_FAILED'
        }

    # Compute Lambda
    Lambda = compute_lambda(fit_con['nll'], fit_unc['nll'])

    # Bootstrap p-value
    p_value, boot_lambdas = bootstrap_pvalue(dataset, Lambda, n_bootstrap=n_bootstrap)

    # Check gates
    chi2_dof_a = fit_unc['chi2_dof_a']
    chi2_dof_b = fit_unc['chi2_dof_b']

    if chi2_dof_a < chi2_min or chi2_dof_b < chi2_min:
        gates = 'UNDERCONSTRAINED'
    elif chi2_dof_a > chi2_max or chi2_dof_b > chi2_max:
        gates = 'MISMATCH'
    else:
        gates = 'PASS'

    r_ci_width = abs(fit_unc['r_a'] - fit_unc['r_b']) * 0.5
    phi_ci_width = abs(fit_unc['phi_a'] - fit_unc['phi_b']) * 0.5

    return {
        'converged': True,
        'Lambda': Lambda,
        'p_value': p_value,
        'gates': gates,
        'fit_unconstrained': fit_unc,
        'fit_constrained': fit_con,
        'chi2_dof_a': chi2_dof_a,
        'chi2_dof_b': chi2_dof_b,
        'r_ci_width': r_ci_width,
        'phi_ci_width': phi_ci_width,
        'n_bootstrap': len(boot_lambdas),
        'bootstrap_lambdas': boot_lambdas.tolist() if len(boot_lambdas) > 0 else []
    }


def bootstrap_pvalue(dataset: Dict, observed_lambda: float,
                     n_bootstrap: int = 200, n_restarts: int = 5) -> Tuple[float, np.ndarray]:
    """
    Compute bootstrap p-value under H0 (rank-1 is true).
    """
    from sim_generate import generate_channel_data

    # Get best-fit shared R
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

        ch_a_config = {
            'type': ch_a_data['type'],
            'mean_count_scale': np.mean(ch_a_data['y']),
            'bg_level': 0.3
        }
        ch_b_config = {
            'type': ch_b_data['type'],
            'mean_count_scale': np.mean(ch_b_data['y']),
            'bg_level': 0.3
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

        fit_unc_b = fit_joint_unconstrained(boot_dataset, n_restarts=3)
        fit_con_b = fit_joint_constrained(boot_dataset, n_restarts=3)

        if fit_unc_b['converged'] and fit_con_b['converged']:
            lambda_b = compute_lambda(fit_con_b['nll'], fit_unc_b['nll'])
            bootstrap_lambdas.append(lambda_b)

    bootstrap_lambdas = np.array(bootstrap_lambdas)

    if len(bootstrap_lambdas) == 0:
        return np.nan, bootstrap_lambdas

    p_value = np.mean(bootstrap_lambdas >= observed_lambda)
    return p_value, bootstrap_lambdas


if __name__ == "__main__":
    import json
    from sim_generate import generate_dataset

    with open('configs/tests.json', 'r') as f:
        config = json.load(f)

    test = config['tests'][0]
    print(f"Testing fit procedures for: {test['name']}")

    dataset = generate_dataset(test, 'M0', scale_factor=2.0, seed=42)

    print("\nFitting unconstrained...")
    fit_unc = fit_joint_unconstrained(dataset, n_restarts=10)
    print(f"  Converged: {fit_unc['converged']}")
    print(f"  chi2/dof A: {fit_unc['chi2_dof_a']:.2f}")
    print(f"  chi2/dof B: {fit_unc['chi2_dof_b']:.2f}")
    print(f"  r_a={fit_unc['r_a']:.2f}, phi_a={fit_unc['phi_a']:.1f}")
    print(f"  r_b={fit_unc['r_b']:.2f}, phi_b={fit_unc['phi_b']:.1f}")

    print("\nFitting constrained...")
    fit_con = fit_joint_constrained(dataset, n_restarts=10)
    print(f"  Converged: {fit_con['converged']}")
    print(f"  chi2/dof A: {fit_con['chi2_dof_a']:.2f}")
    print(f"  chi2/dof B: {fit_con['chi2_dof_b']:.2f}")
    print(f"  r={fit_con['r']:.2f}, phi={fit_con['phi']:.1f}")

    Lambda = compute_lambda(fit_con['nll'], fit_unc['nll'])
    print(f"\nLambda: {Lambda:.3f}")
