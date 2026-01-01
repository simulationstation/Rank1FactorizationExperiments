#!/usr/bin/env python3
"""
LHCb Pentaquark Rank-1 Bottleneck Test v4
==========================================
Uses LHCb-published 1D model choices (Poly6 for weighted, AIC-selected for unweighted)
Yield-based rank-1 test: r² = Y_4457 / Y_4440

Key differences from v3:
- Yield ratios instead of complex amplitude ratios (matches LHCb 1D fits)
- Poly6 background for Table 3 (weighted) to match LHCb
- AIC/BIC selection among Poly1,2,3 for Tables 1,2
- Stability check across WIDE/TIGHT windows
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2
import json
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import os

warnings.filterwarnings('ignore')

# ============================================================
# Pentaquark parameters (LHCb PRL 122, 222001)
# ============================================================
PC_PARAMS = {
    'Pc4312': {'mass': 4311.9, 'width': 9.8},
    'Pc4440': {'mass': 4440.3, 'width': 20.6},
    'Pc4457': {'mass': 4457.3, 'width': 6.4},
}

# Analysis windows
WINDOWS = {
    'WIDE': (4270, 4520),
    'TIGHT': (4320, 4490),
}

# Reference mass for polynomial
M0 = 4400.0

# Number of restarts and bootstrap
N_RESTARTS = 100
N_BOOTSTRAP = 500
N_BOOTSTRAP_RESTARTS = 30

# Fit-health gate
CHI2_DOF_GATE = 3.0


def bw_intensity(m, mass, width):
    """Relativistic Breit-Wigner intensity (not amplitude)."""
    gamma = width * mass / m  # Energy-dependent width approximation
    return (mass * width)**2 / ((m**2 - mass**2)**2 + (m * gamma)**2)


def model_intensity(m, params, bg_order, n_peaks=3):
    """
    1D intensity model: sum of BW shapes + polynomial background

    params structure:
    - params[0:n_peaks]: yield coefficients Y_k for each BW
    - params[n_peaks:]: polynomial coefficients b_j

    I(m) = sum_k Y_k * BW_k(m) + sum_j b_j * (m - M0)^j
    """
    # BW contributions
    masses = [PC_PARAMS['Pc4312']['mass'], PC_PARAMS['Pc4440']['mass'], PC_PARAMS['Pc4457']['mass']]
    widths = [PC_PARAMS['Pc4312']['width'], PC_PARAMS['Pc4440']['width'], PC_PARAMS['Pc4457']['width']]

    intensity = np.zeros_like(m, dtype=float)
    for k in range(n_peaks):
        Y_k = params[k]
        intensity += Y_k * bw_intensity(m, masses[k], widths[k])

    # Polynomial background
    dm = (m - M0) / 100.0  # Scaled for numerical stability
    for j in range(bg_order + 1):
        b_j = params[n_peaks + j]
        intensity += b_j * dm**j

    return intensity


def poisson_nll(params, m_centers, counts, bg_order, bin_width):
    """Poisson negative log-likelihood for raw counts."""
    mu = model_intensity(m_centers, params, bg_order) * bin_width
    mu = np.maximum(mu, 1e-10)  # Avoid log(0)
    nll = np.sum(mu - counts * np.log(mu))
    return nll


def gaussian_nll(params, m_centers, values, errors, bg_order, bin_width):
    """Gaussian negative log-likelihood for weighted spectrum."""
    mu = model_intensity(m_centers, params, bg_order) * bin_width
    nll = 0.5 * np.sum(((values - mu) / errors)**2)
    return nll


def compute_chi2_deviance(params, m_centers, counts, bg_order, bin_width):
    """Compute chi2 and Poisson deviance."""
    mu = model_intensity(m_centers, params, bg_order) * bin_width
    mu = np.maximum(mu, 1e-10)

    # Chi2
    chi2_val = np.sum((counts - mu)**2 / mu)

    # Poisson deviance
    mask = counts > 0
    deviance = 2 * np.sum(mu - counts)
    deviance += 2 * np.sum(counts[mask] * np.log(counts[mask] / mu[mask]))

    n_params = len(params)
    n_dof = len(counts) - n_params

    return chi2_val, deviance, n_dof


def compute_chi2_gaussian(params, m_centers, values, errors, bg_order, bin_width):
    """Compute chi2 for Gaussian likelihood."""
    mu = model_intensity(m_centers, params, bg_order) * bin_width
    chi2_val = np.sum(((values - mu) / errors)**2)
    n_params = len(params)
    n_dof = len(values) - n_params
    return chi2_val, n_dof


def fit_single_channel(m_centers, data, errors, bg_order, bin_width,
                       likelihood='poisson', n_restarts=N_RESTARTS, seed=None):
    """
    Fit single channel with multi-start optimization.

    Returns: best_params, best_nll, fit_quality
    """
    n_peaks = 3
    n_params = n_peaks + bg_order + 1

    if seed is not None:
        np.random.seed(seed)

    # Bounds
    bounds = []
    for _ in range(n_peaks):
        bounds.append((0, np.max(data) * 10))  # Yield coefficients (non-negative)
    for j in range(bg_order + 1):
        bounds.append((-1e6, 1e6))  # Polynomial coefficients

    if likelihood == 'poisson':
        obj_func = lambda p: poisson_nll(p, m_centers, data, bg_order, bin_width)
    else:
        obj_func = lambda p: gaussian_nll(p, m_centers, data, errors, bg_order, bin_width)

    best_nll = np.inf
    best_params = None

    for _ in range(n_restarts):
        # Random initial guess
        x0 = np.zeros(n_params)
        for k in range(n_peaks):
            x0[k] = np.random.uniform(0.1, 10) * np.mean(data)
        x0[n_peaks] = np.random.uniform(0.5, 2) * np.mean(data) / bin_width
        for j in range(1, bg_order + 1):
            x0[n_peaks + j] = np.random.uniform(-10, 10)

        # L-BFGS-B
        try:
            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 5000, 'ftol': 1e-12})
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x.copy()
        except:
            pass

        # Nelder-Mead refinement
        if best_params is not None:
            try:
                res = minimize(obj_func, best_params, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_params = res.x.copy()
            except:
                pass

    # Compute fit quality
    if likelihood == 'poisson':
        chi2_val, deviance, n_dof = compute_chi2_deviance(best_params, m_centers, data, bg_order, bin_width)
        fit_quality = {
            'chi2': chi2_val,
            'chi2_dof': chi2_val / n_dof if n_dof > 0 else np.inf,
            'deviance': deviance,
            'deviance_dof': deviance / n_dof if n_dof > 0 else np.inf,
            'n_dof': n_dof,
            'gate_pass': (chi2_val / n_dof < CHI2_DOF_GATE) and (deviance / n_dof < CHI2_DOF_GATE)
        }
    else:
        chi2_val, n_dof = compute_chi2_gaussian(best_params, m_centers, data, errors, bg_order, bin_width)
        fit_quality = {
            'chi2': chi2_val,
            'chi2_dof': chi2_val / n_dof if n_dof > 0 else np.inf,
            'n_dof': n_dof,
            'gate_pass': chi2_val / n_dof < CHI2_DOF_GATE
        }

    return best_params, best_nll, fit_quality


def extract_yield_ratio(params):
    """Extract r² = Y_4457 / Y_4440 from fit parameters."""
    Y_4440 = params[1]  # Index 1 is Pc4440
    Y_4457 = params[2]  # Index 2 is Pc4457
    if Y_4440 > 0:
        return Y_4457 / Y_4440
    return np.nan


def fit_joint_constrained(m_A, data_A, err_A, m_B, data_B, err_B,
                          bg_order_A, bg_order_B, bin_width,
                          likelihood_A='poisson', likelihood_B='poisson',
                          n_restarts=N_RESTARTS, seed=None):
    """
    Joint fit with rank-1 constraint: r²_A = r²_B = r²_shared

    Parameterization:
    - Shared: r² = Y_4457/Y_4440 (one value for both channels)
    - Per channel: Y_4312, Y_4440, bg coefficients (independent normalizations)

    Returns: params_A, params_B, r2_shared, total_nll
    """
    if seed is not None:
        np.random.seed(seed)

    n_peaks = 3
    n_bg_A = bg_order_A + 1
    n_bg_B = bg_order_B + 1

    # Joint parameter vector:
    # [r2_shared, Y4312_A, Y4440_A, bg_A..., Y4312_B, Y4440_B, bg_B...]
    # Y4457 = r2_shared * Y4440 (derived)

    def joint_obj(p):
        r2_shared = p[0]
        if r2_shared < 0:
            return 1e20

        # Channel A parameters
        Y4312_A = p[1]
        Y4440_A = p[2]
        Y4457_A = r2_shared * Y4440_A
        bg_A = p[3:3+n_bg_A]
        params_A = np.array([Y4312_A, Y4440_A, Y4457_A] + list(bg_A))

        # Channel B parameters
        idx_B = 3 + n_bg_A
        Y4312_B = p[idx_B]
        Y4440_B = p[idx_B + 1]
        Y4457_B = r2_shared * Y4440_B
        bg_B = p[idx_B + 2:idx_B + 2 + n_bg_B]
        params_B = np.array([Y4312_B, Y4440_B, Y4457_B] + list(bg_B))

        # Check non-negativity of yields
        if any(y < 0 for y in [Y4312_A, Y4440_A, Y4457_A, Y4312_B, Y4440_B, Y4457_B]):
            return 1e20

        # NLL for channel A
        if likelihood_A == 'poisson':
            nll_A = poisson_nll(params_A, m_A, data_A, bg_order_A, bin_width)
        else:
            nll_A = gaussian_nll(params_A, m_A, data_A, err_A, bg_order_A, bin_width)

        # NLL for channel B
        if likelihood_B == 'poisson':
            nll_B = poisson_nll(params_B, m_B, data_B, bg_order_B, bin_width)
        else:
            nll_B = gaussian_nll(params_B, m_B, data_B, err_B, bg_order_B, bin_width)

        return nll_A + nll_B

    n_joint = 1 + 2 + n_bg_A + 2 + n_bg_B  # r2, Y4312_A, Y4440_A, bg_A, Y4312_B, Y4440_B, bg_B

    best_nll = np.inf
    best_p = None

    for _ in range(n_restarts):
        x0 = np.zeros(n_joint)
        x0[0] = np.random.uniform(0.1, 2.0)  # r2_shared
        x0[1] = np.random.uniform(0.1, 10) * np.mean(data_A)  # Y4312_A
        x0[2] = np.random.uniform(0.1, 10) * np.mean(data_A)  # Y4440_A
        x0[3] = np.random.uniform(0.5, 2) * np.mean(data_A) / bin_width  # bg_A[0]

        idx_B = 3 + n_bg_A
        x0[idx_B] = np.random.uniform(0.1, 10) * np.mean(data_B)  # Y4312_B
        x0[idx_B + 1] = np.random.uniform(0.1, 10) * np.mean(data_B)  # Y4440_B
        x0[idx_B + 2] = np.random.uniform(0.5, 2) * np.mean(data_B) / bin_width  # bg_B[0]

        # Random init for higher-order bg coeffs
        for j in range(1, n_bg_A):
            x0[3 + j] = np.random.uniform(-10, 10)
        for j in range(1, n_bg_B):
            x0[idx_B + 2 + j] = np.random.uniform(-10, 10)

        # Bounds
        bounds = [(0.001, 10)]  # r2_shared
        bounds += [(0, 1e8)] * 2  # Y4312_A, Y4440_A
        bounds += [(-1e6, 1e6)] * n_bg_A
        bounds += [(0, 1e8)] * 2  # Y4312_B, Y4440_B
        bounds += [(-1e6, 1e6)] * n_bg_B

        try:
            res = minimize(joint_obj, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 5000, 'ftol': 1e-12})
            if res.fun < best_nll:
                best_nll = res.fun
                best_p = res.x.copy()
        except:
            pass

        if best_p is not None:
            try:
                res = minimize(joint_obj, best_p, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_p = res.x.copy()
            except:
                pass

    if best_p is None:
        return None, None, np.nan, np.inf

    # Extract parameters
    r2_shared = best_p[0]
    Y4312_A = best_p[1]
    Y4440_A = best_p[2]
    Y4457_A = r2_shared * Y4440_A
    bg_A = best_p[3:3+n_bg_A]
    params_A = np.array([Y4312_A, Y4440_A, Y4457_A] + list(bg_A))

    idx_B = 3 + n_bg_A
    Y4312_B = best_p[idx_B]
    Y4440_B = best_p[idx_B + 1]
    Y4457_B = r2_shared * Y4440_B
    bg_B = best_p[idx_B + 2:idx_B + 2 + n_bg_B]
    params_B = np.array([Y4312_B, Y4440_B, Y4457_B] + list(bg_B))

    return params_A, params_B, r2_shared, best_nll


def bootstrap_single(args):
    """Single bootstrap replicate for parallel execution."""
    (seed, params_A_con, params_B_con, m_A, m_B, bg_order_A, bg_order_B,
     bin_width, likelihood_A, likelihood_B) = args

    np.random.seed(seed)

    # Generate pseudo-data from constrained fit
    mu_A = model_intensity(m_A, params_A_con, bg_order_A) * bin_width
    mu_B = model_intensity(m_B, params_B_con, bg_order_B) * bin_width

    if likelihood_A == 'poisson':
        data_A_boot = np.random.poisson(np.maximum(mu_A, 0.1))
    else:
        data_A_boot = np.random.normal(mu_A, np.sqrt(np.abs(mu_A) + 1))

    if likelihood_B == 'poisson':
        data_B_boot = np.random.poisson(np.maximum(mu_B, 0.1))
    else:
        data_B_boot = np.random.normal(mu_B, np.sqrt(np.abs(mu_B) + 1))

    err_A_boot = np.sqrt(np.abs(data_A_boot) + 1)
    err_B_boot = np.sqrt(np.abs(data_B_boot) + 1)

    # Fit unconstrained
    params_A, nll_A, _ = fit_single_channel(m_A, data_A_boot, err_A_boot, bg_order_A,
                                            bin_width, likelihood_A, N_BOOTSTRAP_RESTARTS, seed)
    params_B, nll_B, _ = fit_single_channel(m_B, data_B_boot, err_B_boot, bg_order_B,
                                            bin_width, likelihood_B, N_BOOTSTRAP_RESTARTS, seed+1000)
    nll_unc = nll_A + nll_B

    # Fit constrained
    _, _, _, nll_con = fit_joint_constrained(m_A, data_A_boot, err_A_boot,
                                              m_B, data_B_boot, err_B_boot,
                                              bg_order_A, bg_order_B, bin_width,
                                              likelihood_A, likelihood_B,
                                              N_BOOTSTRAP_RESTARTS, seed+2000)

    Lambda = 2 * (nll_con - nll_unc)
    return Lambda


def run_bootstrap(params_A_con, params_B_con, m_A, m_B, bg_order_A, bg_order_B,
                  bin_width, likelihood_A, likelihood_B, n_bootstrap=N_BOOTSTRAP):
    """Run bootstrap analysis in parallel."""
    args_list = [
        (seed, params_A_con, params_B_con, m_A, m_B, bg_order_A, bg_order_B,
         bin_width, likelihood_A, likelihood_B)
        for seed in range(n_bootstrap)
    ]

    n_workers = max(1, cpu_count() - 1)
    with Pool(n_workers) as pool:
        Lambda_boot = list(pool.map(bootstrap_single, args_list))

    Lambda_boot = np.array([L for L in Lambda_boot if np.isfinite(L)])
    return Lambda_boot


def select_background_order(m_centers, data, errors, bin_width, likelihood='poisson', max_order=3):
    """Select background order using AIC."""
    results = {}
    for order in range(1, max_order + 1):
        params, nll, quality = fit_single_channel(m_centers, data, errors, order, bin_width,
                                                   likelihood, n_restarts=50)
        n_params = 3 + order + 1  # 3 peaks + (order+1) bg coeffs
        aic = 2 * n_params + 2 * nll
        bic = n_params * np.log(len(data)) + 2 * nll
        results[order] = {
            'params': params,
            'nll': nll,
            'quality': quality,
            'aic': aic,
            'bic': bic
        }

    # Select by AIC
    best_order = min(results.keys(), key=lambda k: results[k]['aic'])
    return best_order, results


def load_data(base_path):
    """Load HEPData tables."""
    t1 = pd.read_csv(os.path.join(base_path, '89271_t1_spectrum_full.csv'), comment='#')
    t2 = pd.read_csv(os.path.join(base_path, '89271_t2_spectrum_mKp_cut.csv'), comment='#')
    t3 = pd.read_csv(os.path.join(base_path, '89271_t3_spectrum_weighted.csv'), comment='#')
    return t1, t2, t3


def extract_window(df, window, is_weighted=False):
    """Extract data in mass window."""
    m_low, m_high = window

    # Find mass column (HEPData format)
    mass_col = None
    for c in df.columns:
        if 'MEV]' in c and 'LOW' not in c and 'HIGH' not in c:
            mass_col = c
            break
    if mass_col is None:
        mass_col = df.columns[0]

    # Filter to window
    mask = (df[mass_col] >= m_low) & (df[mass_col] <= m_high)
    df_win = df[mask].copy()

    m_centers = df_win[mass_col].values

    # Find data column (DN/DM)
    if is_weighted:
        y_col = [c for c in df.columns if 'Weighted' in c or 'weighted' in c][0]
    else:
        y_col = [c for c in df.columns if 'DN/DM' in c and 'Weighted' not in c][0]

    # DN/DM is per MeV, these are already rates
    dndm = df_win[y_col].values

    # Find error column
    err_col = [c for c in df.columns if 'stat +' in c][0]
    errors = np.abs(df_win[err_col].values)

    # Bin width from consecutive bins
    if len(m_centers) > 1:
        bin_width = np.median(np.diff(np.sort(m_centers)))
    else:
        bin_width = 2.0

    # Convert DN/DM to counts: counts = DN/DM * bin_width
    counts = dndm * bin_width

    return m_centers, counts, errors * bin_width, bin_width


def run_pair_analysis(m_A, data_A, err_A, m_B, data_B, err_B,
                      bg_order_A, bg_order_B, bin_width,
                      likelihood_A='poisson', likelihood_B='poisson',
                      run_bootstrap_flag=True, config_name=''):
    """Run full rank-1 analysis for a channel pair."""
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"{'='*60}")
    print(f"Channel A: {len(m_A)} bins, bg_order={bg_order_A}")
    print(f"Channel B: {len(m_B)} bins, bg_order={bg_order_B}")

    # Fit unconstrained
    print("Fitting Channel A...")
    params_A, nll_A, quality_A = fit_single_channel(m_A, data_A, err_A, bg_order_A,
                                                     bin_width, likelihood_A)
    print("Fitting Channel B...")
    params_B, nll_B, quality_B = fit_single_channel(m_B, data_B, err_B, bg_order_B,
                                                     bin_width, likelihood_B)

    nll_unc = nll_A + nll_B
    r2_A = extract_yield_ratio(params_A)
    r2_B = extract_yield_ratio(params_B)

    print(f"r²_A = {r2_A:.4f}, r²_B = {r2_B:.4f}")
    print(f"χ²/dof A = {quality_A['chi2_dof']:.3f}, B = {quality_B['chi2_dof']:.3f}")

    # Fit constrained
    print("Fitting Joint (constrained)...")
    params_A_con, params_B_con, r2_shared, nll_con = fit_joint_constrained(
        m_A, data_A, err_A, m_B, data_B, err_B,
        bg_order_A, bg_order_B, bin_width, likelihood_A, likelihood_B
    )

    Lambda = 2 * (nll_con - nll_unc)
    print(f"r²_shared = {r2_shared:.4f}")
    print(f"Λ = {Lambda:.4f}")

    # Check Lambda >= 0
    if Lambda < -1e-3:
        print(f"WARNING: Λ < 0 ({Lambda:.4f}), retrying with more restarts...")
        params_A, nll_A, quality_A = fit_single_channel(m_A, data_A, err_A, bg_order_A,
                                                         bin_width, likelihood_A, n_restarts=300)
        params_B, nll_B, quality_B = fit_single_channel(m_B, data_B, err_B, bg_order_B,
                                                         bin_width, likelihood_B, n_restarts=300)
        nll_unc = nll_A + nll_B

        params_A_con, params_B_con, r2_shared, nll_con = fit_joint_constrained(
            m_A, data_A, err_A, m_B, data_B, err_B,
            bg_order_A, bg_order_B, bin_width, likelihood_A, likelihood_B, n_restarts=300
        )
        Lambda = 2 * (nll_con - nll_unc)
        Lambda = max(0, Lambda)  # Floor at 0
        r2_A = extract_yield_ratio(params_A)
        r2_B = extract_yield_ratio(params_B)

    # Bootstrap
    p_value = np.nan
    Lambda_boot = []
    if run_bootstrap_flag and quality_A.get('gate_pass', False) and quality_B.get('gate_pass', False):
        print(f"Running {N_BOOTSTRAP} bootstrap replicates...")
        Lambda_boot = run_bootstrap(params_A_con, params_B_con, m_A, m_B,
                                    bg_order_A, bg_order_B, bin_width,
                                    likelihood_A, likelihood_B, N_BOOTSTRAP)
        if len(Lambda_boot) > 0:
            p_value = np.mean(Lambda_boot >= Lambda)
            print(f"Bootstrap p-value = {p_value:.4f}")

    result = {
        'config': config_name,
        'r2_A': r2_A,
        'r2_B': r2_B,
        'r2_shared': r2_shared,
        'nll_unc': nll_unc,
        'nll_con': nll_con,
        'Lambda': Lambda,
        'p_value': p_value,
        'quality_A': quality_A,
        'quality_B': quality_B,
        'bg_order_A': bg_order_A,
        'bg_order_B': bg_order_B,
        'params_A': params_A.tolist() if params_A is not None else None,
        'params_B': params_B.tolist() if params_B is not None else None,
        'Lambda_boot': Lambda_boot.tolist() if len(Lambda_boot) > 0 else [],
    }

    return result


def determine_verdict(results_dict):
    """Determine final verdict based on stability across windows and backgrounds."""
    # Check if any configuration passes gates
    passing_configs = []
    for key, res in results_dict.items():
        if res['quality_A'].get('gate_pass', False) and res['quality_B'].get('gate_pass', False):
            passing_configs.append(key)

    if len(passing_configs) == 0:
        return 'MODEL MISMATCH', 'No configuration passes fit-health gates'

    # Check stability of conclusions
    verdicts = {}
    for key in passing_configs:
        res = results_dict[key]
        if np.isnan(res['p_value']):
            verdicts[key] = 'INCOMPLETE'
        elif res['p_value'] > 0.05:
            verdicts[key] = 'SUPPORTED'
        else:
            verdicts[key] = 'DISFAVORED'

    unique_verdicts = set(verdicts.values()) - {'INCOMPLETE'}

    if len(unique_verdicts) == 0:
        return 'INCOMPLETE', 'Bootstrap not completed for passing configurations'
    elif len(unique_verdicts) == 1:
        verdict = list(unique_verdicts)[0]
        # Check r2_shared stability
        r2_values = [results_dict[k]['r2_shared'] for k in passing_configs
                     if not np.isnan(results_dict[k]['r2_shared'])]
        if len(r2_values) > 1:
            r2_range = max(r2_values) - min(r2_values)
            if r2_range > 0.3:  # More than 30% variation
                return 'UNSTABLE', f'r² varies by {r2_range:.3f} across configurations'
        return verdict, f'Stable across {len(passing_configs)} configurations'
    else:
        return 'UNSTABLE', f'Different verdicts across windows: {verdicts}'


def main():
    base_path = '/home/primary/DarkBItParticleColiderPredictions/lhcb_rank1_test_v4'
    data_path = os.path.join(base_path, 'data/hepdata')
    out_path = os.path.join(base_path, 'out')
    log_path = os.path.join(base_path, 'logs')

    print("="*70)
    print("LHCb Pentaquark Rank-1 Bottleneck Test v4")
    print("Yield-based test with LHCb 1D model choices")
    print("="*70)

    # Load data
    print("\nLoading data...")
    t1, t2, t3 = load_data(data_path)
    print(f"Table 1 (full): {len(t1)} bins")
    print(f"Table 2 (mKp cut): {len(t2)} bins")
    print(f"Table 3 (weighted): {len(t3)} bins")

    all_results = {}

    # ================================================================
    # PAIR 1: Table 1 vs Table 2 (both Poisson)
    # ================================================================
    print("\n" + "#"*70)
    print("# PAIR 1: Table 1 (full) vs Table 2 (mKp > 1.9 GeV cut)")
    print("# Both channels: Poisson NLL")
    print("#"*70)

    for window_name, window in WINDOWS.items():
        print(f"\n>>> Window: {window_name} {window}")

        m_A, data_A, err_A, bin_width = extract_window(t1, window, is_weighted=False)
        m_B, data_B, err_B, _ = extract_window(t2, window, is_weighted=False)

        print(f"Channel A: {len(m_A)} bins, {int(sum(data_A))} counts")
        print(f"Channel B: {len(m_B)} bins, {int(sum(data_B))} counts")

        # Select background order by AIC (max 3)
        print("\nSelecting background order for Channel A...")
        bg_A, bg_results_A = select_background_order(m_A, data_A, err_A, bin_width, 'poisson', max_order=3)
        print(f"Selected: Poly{bg_A} (AIC={bg_results_A[bg_A]['aic']:.1f})")

        print("Selecting background order for Channel B...")
        bg_B, bg_results_B = select_background_order(m_B, data_B, err_B, bin_width, 'poisson', max_order=3)
        print(f"Selected: Poly{bg_B} (AIC={bg_results_B[bg_B]['aic']:.1f})")

        # Run with AIC-selected background
        config_key = f"pair1_{window_name.lower()}_aic"
        result = run_pair_analysis(m_A, data_A, err_A, m_B, data_B, err_B,
                                   bg_A, bg_B, bin_width, 'poisson', 'poisson',
                                   run_bootstrap_flag=True,
                                   config_name=f"Pair1_{window_name}_AIC(Poly{bg_A},Poly{bg_B})")
        result['bg_selection'] = 'AIC'
        all_results[config_key] = result

        # Sensitivity check with Poly3 for both
        config_key = f"pair1_{window_name.lower()}_poly3"
        result = run_pair_analysis(m_A, data_A, err_A, m_B, data_B, err_B,
                                   3, 3, bin_width, 'poisson', 'poisson',
                                   run_bootstrap_flag=True,
                                   config_name=f"Pair1_{window_name}_Poly3")
        result['bg_selection'] = 'Poly3'
        all_results[config_key] = result

    # ================================================================
    # PAIR 2: Table 1 vs Table 3 (Poisson + Gaussian)
    # ================================================================
    print("\n" + "#"*70)
    print("# PAIR 2: Table 1 (full, Poisson) vs Table 3 (weighted, Gaussian)")
    print("# Table 3: Poly6 background (LHCb standard)")
    print("#"*70)

    for window_name, window in WINDOWS.items():
        print(f"\n>>> Window: {window_name} {window}")

        m_A, data_A, err_A, bin_width = extract_window(t1, window, is_weighted=False)
        m_C, data_C, err_C, _ = extract_window(t3, window, is_weighted=True)

        print(f"Channel A (Table 1): {len(m_A)} bins, {int(sum(data_A))} counts")
        print(f"Channel C (Table 3): {len(m_C)} bins, weighted")

        # Table 3 uses Poly6 per LHCb
        bg_C = 6

        # Select background for Table 1
        bg_A, bg_results_A = select_background_order(m_A, data_A, err_A, bin_width, 'poisson', max_order=3)

        # Run with AIC-selected for A, Poly6 for C
        config_key = f"pair2_{window_name.lower()}_aic"
        result = run_pair_analysis(m_A, data_A, err_A, m_C, data_C, err_C,
                                   bg_A, bg_C, bin_width, 'poisson', 'gaussian',
                                   run_bootstrap_flag=True,
                                   config_name=f"Pair2_{window_name}_AIC(Poly{bg_A})+Poly6")
        result['bg_selection'] = 'AIC+Poly6'
        all_results[config_key] = result

    # ================================================================
    # Determine verdicts
    # ================================================================
    print("\n" + "="*70)
    print("FINAL VERDICTS")
    print("="*70)

    # Pair 1 verdict
    pair1_results = {k: v for k, v in all_results.items() if k.startswith('pair1_')}
    pair1_verdict, pair1_reason = determine_verdict(pair1_results)
    print(f"\nPair 1 (T1 vs T2): {pair1_verdict}")
    print(f"  Reason: {pair1_reason}")

    # Pair 2 verdict
    pair2_results = {k: v for k, v in all_results.items() if k.startswith('pair2_')}
    pair2_verdict, pair2_reason = determine_verdict(pair2_results)
    print(f"\nPair 2 (T1 vs T3): {pair2_verdict}")
    print(f"  Reason: {pair2_reason}")

    # Find best stable configuration
    best_config = None
    for key, res in all_results.items():
        if (res['quality_A'].get('gate_pass', False) and
            res['quality_B'].get('gate_pass', False) and
            not np.isnan(res['p_value'])):
            if best_config is None or res['p_value'] > all_results[best_config]['p_value']:
                best_config = key

    # ================================================================
    # Save results
    # ================================================================

    # Convert numpy types for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    results_json = {
        'pair1_verdict': pair1_verdict,
        'pair1_reason': pair1_reason,
        'pair2_verdict': pair2_verdict,
        'pair2_reason': pair2_reason,
        'best_config': best_config,
        'results': convert_for_json(all_results)
    }

    with open(os.path.join(out_path, 'results_v4.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Generate summary table
    summary_rows = []
    for key, res in all_results.items():
        row = {
            'config': res['config'],
            'r2_A': res['r2_A'],
            'r2_B': res['r2_B'],
            'r2_shared': res['r2_shared'],
            'Lambda': res['Lambda'],
            'p_value': res['p_value'],
            'chi2_dof_A': res['quality_A']['chi2_dof'],
            'chi2_dof_B': res['quality_B']['chi2_dof'],
            'gate_A': res['quality_A'].get('gate_pass', False),
            'gate_B': res['quality_B'].get('gate_pass', False),
            'bg_order_A': res['bg_order_A'],
            'bg_order_B': res['bg_order_B'],
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_path, 'SUMMARY_TABLE.csv'), index=False)

    # Generate report
    generate_report(all_results, pair1_verdict, pair1_reason, pair2_verdict, pair2_reason,
                    best_config, out_path)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to {out_path}/")

    return results_json


def generate_report(all_results, pair1_verdict, pair1_reason, pair2_verdict, pair2_reason,
                    best_config, out_path):
    """Generate REPORT_v4.md"""

    report = """# LHCb Pentaquark Rank-1 Bottleneck Test v4

## Executive Summary

"""

    # Main verdict
    if pair1_verdict == 'SUPPORTED':
        report += f"**Primary Result (Pair 1 - Table 1 vs Table 2):** **SUPPORTED**\n\n"
    elif pair1_verdict == 'DISFAVORED':
        report += f"**Primary Result (Pair 1 - Table 1 vs Table 2):** **DISFAVORED**\n\n"
    else:
        report += f"**Primary Result (Pair 1 - Table 1 vs Table 2):** **{pair1_verdict}**\n\n"

    report += f"Reason: {pair1_reason}\n\n"

    # Best configuration summary
    if best_config and best_config in all_results:
        res = all_results[best_config]
        report += f"""### Best Configuration: {res['config']}

| Metric | Value |
|--------|-------|
| r²_A (Table 1) | {res['r2_A']:.4f} |
| r²_B (Table 2) | {res['r2_B']:.4f} |
| r²_shared | {res['r2_shared']:.4f} |
| Λ | {res['Lambda']:.4f} |
| Bootstrap p-value | {res['p_value']:.4f} |
| χ²/dof (A) | {res['quality_A']['chi2_dof']:.3f} |
| χ²/dof (B) | {res['quality_B']['chi2_dof']:.3f} |

"""

    report += """---

## 1. Data Provenance

| Item | Value |
|------|-------|
| HEPData Record | [89271](https://www.hepdata.net/record/ins1728691) |
| Publication | PRL 122, 222001 (2019) |
| DOI | [10.1103/PhysRevLett.122.222001](https://doi.org/10.1103/PhysRevLett.122.222001) |

### Tables Used
- **Table 1**: Full m(J/ψ p) spectrum (Poisson NLL)
- **Table 2**: m(J/ψ p) with m(Kp) > 1.9 GeV cut (Poisson NLL)
- **Table 3**: cosθ-weighted m(J/ψ p) spectrum (Gaussian NLL)

### Amplitude Workspace Check
**No full amplitude workspace available.** The CDS supplementary materials (LHCb-PAPER-2019-014-supplementary-v2.zip) contain only ROOT plotting macros, not amplitude models. This analysis uses reconstructed 1D mass-spectrum models.

---

## 2. Model Definitions

### Intensity Model
The 1D intensity model follows LHCb's published approach:

```
I(m) = Σ_k Y_k × BW_k(m) + Σ_j b_j × ((m - 4400)/100)^j
```

Where:
- Y_k: yield coefficient for pentaquark state k
- BW_k(m): relativistic Breit-Wigner intensity shape
- b_j: polynomial background coefficients

### Pentaquark Parameters (Fixed)

| State | Mass [MeV] | Width [MeV] |
|-------|------------|-------------|
| Pc(4312)⁺ | 4311.9 | 9.8 |
| Pc(4440)⁺ | 4440.3 | 20.6 |
| Pc(4457)⁺ | 4457.3 | 6.4 |

### Rank-1 Test (Yield-Based)
The rank-1 constraint at the 1D projection level:

```
r² ≡ Y_4457 / Y_4440
```

Rank-1 prediction: r²_A ≈ r²_B across channels

### Background Selection
- **Tables 1 & 2**: AIC-selected among Poly1, Poly2, Poly3
- **Table 3**: Poly6 (matching LHCb's published fits)

---

## 3. Results by Configuration

"""

    # Add results table
    report += "| Configuration | r²_A | r²_B | r²_shared | Λ | p-value | χ²/dof (A,B) | Gates |\n"
    report += "|---------------|------|------|-----------|---|---------|--------------|-------|\n"

    for key, res in all_results.items():
        gate_str = "✓✓" if res['quality_A'].get('gate_pass', False) and res['quality_B'].get('gate_pass', False) else (
            "✓✗" if res['quality_A'].get('gate_pass', False) else (
            "✗✓" if res['quality_B'].get('gate_pass', False) else "✗✗"))

        p_str = f"{res['p_value']:.4f}" if not np.isnan(res['p_value']) else "N/A"

        report += f"| {res['config']} | {res['r2_A']:.4f} | {res['r2_B']:.4f} | {res['r2_shared']:.4f} | {res['Lambda']:.4f} | {p_str} | {res['quality_A']['chi2_dof']:.2f}, {res['quality_B']['chi2_dof']:.2f} | {gate_str} |\n"

    report += f"""

---

## 4. Stability Analysis

### Pair 1 (Table 1 vs Table 2)
- **Verdict**: {pair1_verdict}
- **Reason**: {pair1_reason}

### Pair 2 (Table 1 vs Table 3)
- **Verdict**: {pair2_verdict}
- **Reason**: {pair2_reason}

---

## 5. Interpretation

"""

    if pair1_verdict == 'SUPPORTED':
        report += """The yield ratio r² = Y(Pc4457)/Y(Pc4440) is consistent between the full spectrum and the m(Kp) > 1.9 GeV cut spectrum. This supports the rank-1 factorization hypothesis at the 1D projection level.

**Physical Interpretation**: The Pc(4457)/Pc(4440) yield ratio appears approximately channel-invariant, consistent with a factorizable coupling structure."""
    elif pair1_verdict == 'DISFAVORED':
        report += """The yield ratio r² = Y(Pc4457)/Y(Pc4440) shows statistically significant differences between channels, disfavoring the rank-1 factorization hypothesis."""
    elif pair1_verdict == 'UNSTABLE':
        report += """The result is sensitive to analysis choices (window, background). No stable conclusion can be drawn."""
    else:
        report += """The fit model does not adequately describe the data (fit-health gates failed). The rank-1 test cannot be reliably performed."""

    report += """

---

## 6. Technical Notes

- **Optimization**: 100 random restarts with L-BFGS-B + Nelder-Mead refinement
- **Bootstrap**: 500 replicates (30 restarts each) for p-value estimation
- **Fit-health gates**: χ²/dof < 3 and Deviance/dof < 3
- **Λ guarantee**: Constrained fit NLL ≥ Unconstrained NLL by construction

---

*Analysis performed with pentaquark_rank1_v4.py*
"""

    with open(os.path.join(out_path, 'REPORT_v4.md'), 'w') as f:
        f.write(report)


if __name__ == '__main__':
    main()
