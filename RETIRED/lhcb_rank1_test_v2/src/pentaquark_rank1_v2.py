#!/usr/bin/env python3
"""
LHCb Pentaquark Rank-1 Bottleneck Test v2

Corrected version with:
1. Restricted mass window [4270, 4520] MeV (pentaquark region only)
2. Poisson NLL for raw count spectra (Tables 1, 2)
3. Gaussian NLL for weighted spectrum (Table 3)
4. Two separate channel pairs tested independently

Pair 1: Table 1 (full) vs Table 2 (mKp cut) - both Poisson
Pair 2: Table 1 (full) vs Table 3 (weighted) - Poisson vs Gaussian
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
import json
import multiprocessing as mp
from functools import partial
import os
import warnings
warnings.filterwarnings('ignore')

# Pentaquark parameters from LHCb PRL 122, 222001 (2019)
PC_PARAMS = {
    'Pc4312': {'mass': 4311.9, 'width': 9.8},
    'Pc4440': {'mass': 4440.3, 'width': 20.6},
    'Pc4457': {'mass': 4457.3, 'width': 6.4},
}

# Mass windows
WINDOW_DEFAULT = (4270, 4520)  # Main analysis window
WINDOW_TIGHT = (4320, 4490)    # Robustness check (doublet focus)


def load_hepdata_csv(filepath):
    """Load HEPData CSV, skipping comment lines."""
    df = pd.read_csv(filepath, comment='#')
    df.columns = ['m_center', 'm_low', 'm_high', 'yield', 'stat_up', 'stat_down']
    df['stat'] = df['stat_up'].abs()  # Symmetric errors
    return df


def apply_window(df, window):
    """Apply mass window cut."""
    mask = (df['m_center'] >= window[0]) & (df['m_center'] <= window[1])
    return df[mask].copy().reset_index(drop=True)


def breit_wigner(m, mass, width):
    """Non-relativistic Breit-Wigner amplitude."""
    return 1.0 / ((m - mass) - 1j * width / 2)


def model_amplitude(m, params):
    """
    Coherent sum of 3 BW amplitudes.

    Parameterization with c2 (Pc4440) as real positive reference:
    params = [c2, r1, phi1, r3, phi3]
    where:
      c1 = c2 * r1 * exp(i*phi1)  [Pc4312]
      c2 = c2 (real positive)     [Pc4440 - reference]
      c3 = c2 * r3 * exp(i*phi3)  [Pc4457]
    """
    c2_mag = params[0]  # |c2| (real positive reference)
    r1, phi1 = params[1], params[2]  # Pc4312 relative to Pc4440
    r3, phi3 = params[3], params[4]  # Pc4457 relative to Pc4440

    c1 = c2_mag * r1 * np.exp(1j * phi1)
    c2 = c2_mag
    c3 = c2_mag * r3 * np.exp(1j * phi3)

    bw1 = breit_wigner(m, PC_PARAMS['Pc4312']['mass'], PC_PARAMS['Pc4312']['width'])
    bw2 = breit_wigner(m, PC_PARAMS['Pc4440']['mass'], PC_PARAMS['Pc4440']['width'])
    bw3 = breit_wigner(m, PC_PARAMS['Pc4457']['mass'], PC_PARAMS['Pc4457']['width'])

    return c1 * bw1 + c2 * bw2 + c3 * bw3


def background_linear(m, b0, b1, m_ref=4400):
    """Linear background: b0 + b1*(m - m_ref)"""
    return b0 + b1 * (m - m_ref)


def background_quadratic(m, b0, b1, b2, m_ref=4400):
    """Quadratic background: b0 + b1*(m-m_ref) + b2*(m-m_ref)^2"""
    x = m - m_ref
    return b0 + b1 * x + b2 * x**2


def full_model(m, signal_params, bg_params, scale, bg_type='linear'):
    """
    Full model: scale * |M(m)|^2 + B(m)

    signal_params = [c2, r1, phi1, r3, phi3]
    bg_params = [b0, b1] or [b0, b1, b2]
    """
    amp = model_amplitude(m, signal_params)
    signal = scale * np.abs(amp)**2

    if bg_type == 'linear':
        bg = background_linear(m, bg_params[0], bg_params[1])
    else:
        bg = background_quadratic(m, bg_params[0], bg_params[1], bg_params[2])

    return signal + bg


def poisson_nll(mu, y):
    """Poisson negative log-likelihood (per bin)."""
    mu = np.maximum(mu, 1e-10)
    # -ln L = sum(mu - y*ln(mu) + ln(y!))
    return np.sum(mu - y * np.log(mu) + gammaln(y + 1))


def gaussian_nll(mu, y, sigma):
    """Gaussian negative log-likelihood (per bin)."""
    sigma = np.maximum(sigma, 1e-10)
    # -ln L = 0.5 * sum((y-mu)^2/sigma^2 + ln(2*pi*sigma^2))
    return 0.5 * np.sum((y - mu)**2 / sigma**2 + np.log(2 * np.pi * sigma**2))


def poisson_deviance(mu, y):
    """Poisson deviance: D = 2 * sum(y*ln(y/mu) - (y-mu))"""
    mu = np.maximum(mu, 1e-10)
    y_safe = np.maximum(y, 1e-10)

    deviance = 2 * np.sum(
        np.where(y > 0, y * np.log(y_safe / mu), 0) - (y - mu)
    )
    return deviance


def pearson_chi2(mu, y):
    """Pearson chi-squared."""
    mu = np.maximum(mu, 1e-10)
    return np.sum((y - mu)**2 / mu)


def gaussian_chi2(mu, y, sigma):
    """Gaussian chi-squared."""
    sigma = np.maximum(sigma, 1e-10)
    return np.sum((y - mu)**2 / sigma**2)


class ChannelFitter:
    """Fitter for a single channel."""

    def __init__(self, m, y, sigma=None, likelihood='poisson', bg_type='linear'):
        self.m = np.array(m)
        self.y = np.array(y)
        self.sigma = np.array(sigma) if sigma is not None else np.sqrt(np.maximum(y, 1))
        self.likelihood = likelihood
        self.bg_type = bg_type
        self.n_bg_params = 2 if bg_type == 'linear' else 3

    def nll(self, params):
        """Compute negative log-likelihood."""
        # params = [c2, r1, phi1, r3, phi3, scale, b0, b1, (b2)]
        signal_params = params[:5]
        scale = params[5]
        bg_params = params[6:6+self.n_bg_params]

        mu = full_model(self.m, signal_params, bg_params, scale, self.bg_type)

        # Ensure non-negative
        if np.any(mu < 0):
            return 1e15

        if self.likelihood == 'poisson':
            return poisson_nll(mu, self.y)
        else:
            return gaussian_nll(mu, self.y, self.sigma)

    def fit(self, n_restarts=30, verbose=False):
        """Multi-start optimization."""
        n_params = 6 + self.n_bg_params

        best_nll = np.inf
        best_params = None

        for restart in range(n_restarts):
            np.random.seed(restart * 7919 + 42)

            # Initialize
            c2_init = np.random.uniform(50, 500)
            r1_init = np.random.uniform(0.1, 2.0)
            phi1_init = np.random.uniform(-np.pi, np.pi)
            r3_init = np.random.uniform(0.1, 2.0)
            phi3_init = np.random.uniform(-np.pi, np.pi)
            scale_init = np.random.uniform(0.5, 2.0)
            b0_init = np.mean(self.y) * np.random.uniform(0.3, 0.7)
            b1_init = np.random.uniform(-0.5, 0.5)

            x0 = [c2_init, r1_init, phi1_init, r3_init, phi3_init, scale_init, b0_init, b1_init]
            if self.bg_type == 'quadratic':
                x0.append(np.random.uniform(-0.01, 0.01))

            # Bounds
            bounds = [
                (1, 5000),      # c2
                (0.01, 10),     # r1
                (-np.pi, np.pi), # phi1
                (0.01, 10),     # r3
                (-np.pi, np.pi), # phi3
                (0.001, 100),   # scale
                (0, None),      # b0 >= 0
                (None, None),   # b1
            ]
            if self.bg_type == 'quadratic':
                bounds.append((None, None))  # b2

            try:
                result = minimize(self.nll, x0, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 2000, 'ftol': 1e-12})

                if result.fun < best_nll:
                    best_nll = result.fun
                    best_params = result.x.copy()

                # Nelder-Mead refinement
                result2 = minimize(self.nll, result.x, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-10})

                if result2.fun < best_nll:
                    best_nll = result2.fun
                    best_params = result2.x.copy()

            except Exception:
                continue

        if best_params is None:
            raise RuntimeError("All optimizations failed")

        return best_params, best_nll

    def compute_fit_quality(self, params):
        """Compute fit quality metrics."""
        signal_params = params[:5]
        scale = params[5]
        bg_params = params[6:6+self.n_bg_params]

        mu = full_model(self.m, signal_params, bg_params, scale, self.bg_type)
        n_params = 6 + self.n_bg_params
        n_dof = len(self.y) - n_params

        if self.likelihood == 'poisson':
            deviance = poisson_deviance(mu, self.y)
            chi2 = pearson_chi2(mu, self.y)
            return {
                'deviance': deviance,
                'deviance_dof': deviance / n_dof,
                'chi2': chi2,
                'chi2_dof': chi2 / n_dof,
                'n_dof': n_dof,
                'mu': mu
            }
        else:
            chi2 = gaussian_chi2(mu, self.y, self.sigma)
            return {
                'chi2': chi2,
                'chi2_dof': chi2 / n_dof,
                'n_dof': n_dof,
                'mu': mu
            }

    def extract_ratio(self, params):
        """Extract R = c3/c2 ratio."""
        r3, phi3 = params[3], params[4]
        return {'r': r3, 'phi': phi3, 'phi_deg': np.degrees(phi3)}


class JointFitter:
    """Joint fitter for two channels with shared ratio constraint."""

    def __init__(self, fitter_A, fitter_B):
        self.fitter_A = fitter_A
        self.fitter_B = fitter_B
        self.n_bg_A = fitter_A.n_bg_params
        self.n_bg_B = fitter_B.n_bg_params

    def nll(self, params):
        """
        Joint NLL with shared (r1, phi1, r3, phi3).

        params = [c2_A, r1, phi1, r3, phi3, scale_A, bg_A..., c2_B, scale_B, bg_B...]
        """
        # Shared signal shape params
        r1, phi1 = params[1], params[2]
        r3, phi3 = params[3], params[4]

        # Channel A
        c2_A = params[0]
        scale_A = params[5]
        bg_A = params[6:6+self.n_bg_A]
        params_A = [c2_A, r1, phi1, r3, phi3, scale_A] + list(bg_A)

        # Channel B
        idx_B = 6 + self.n_bg_A
        c2_B = params[idx_B]
        scale_B = params[idx_B + 1]
        bg_B = params[idx_B + 2:idx_B + 2 + self.n_bg_B]
        params_B = [c2_B, r1, phi1, r3, phi3, scale_B] + list(bg_B)

        return self.fitter_A.nll(params_A) + self.fitter_B.nll(params_B)

    def fit(self, init_A, init_B, n_restarts=30):
        """Multi-start joint optimization."""
        best_nll = np.inf
        best_params = None

        for restart in range(n_restarts):
            np.random.seed(restart * 7919 + 1000)

            # Initialize from single-channel fits with perturbation
            perturb = 1 + 0.1 * np.random.randn()

            # Shared params from average
            r1_init = 0.5 * (init_A[1] + init_B[1]) * perturb
            phi1_init = 0.5 * (init_A[2] + init_B[2]) + 0.1 * np.random.randn()
            r3_init = 0.5 * (init_A[3] + init_B[3]) * perturb
            phi3_init = 0.5 * (init_A[4] + init_B[4]) + 0.1 * np.random.randn()

            # Channel A params
            c2_A = init_A[0] * perturb
            scale_A = init_A[5] * perturb
            bg_A = init_A[6:6+self.n_bg_A] * perturb

            # Channel B params
            c2_B = init_B[0] * perturb
            scale_B = init_B[5] * perturb
            bg_B = init_B[6:6+self.n_bg_B] * perturb

            x0 = [c2_A, r1_init, phi1_init, r3_init, phi3_init, scale_A] + \
                 list(bg_A) + [c2_B, scale_B] + list(bg_B)

            # Bounds
            bounds = [
                (1, 5000),       # c2_A
                (0.01, 10),      # r1 shared
                (-np.pi, np.pi), # phi1 shared
                (0.01, 10),      # r3 shared
                (-np.pi, np.pi), # phi3 shared
                (0.001, 100),    # scale_A
            ]
            bounds += [(0, None)] + [(None, None)] * (self.n_bg_A - 1)  # bg_A
            bounds += [
                (1, 5000),       # c2_B
                (0.001, 100),    # scale_B
            ]
            bounds += [(0, None)] + [(None, None)] * (self.n_bg_B - 1)  # bg_B

            try:
                result = minimize(self.nll, x0, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 3000, 'ftol': 1e-12})

                if result.fun < best_nll:
                    best_nll = result.fun
                    best_params = result.x.copy()

                result2 = minimize(self.nll, result.x, method='Nelder-Mead',
                                 options={'maxiter': 5000})

                if result2.fun < best_nll:
                    best_nll = result2.fun
                    best_params = result2.x.copy()

            except Exception:
                continue

        if best_params is None:
            raise RuntimeError("Joint optimization failed")

        return best_params, best_nll

    def extract_shared_ratio(self, params):
        """Extract shared R = c3/c2."""
        return {'r': params[3], 'phi': params[4], 'phi_deg': np.degrees(params[4])}


def bootstrap_single_poisson_poisson(args):
    """Bootstrap replicate for Pair 1 (both Poisson)."""
    seed, m_A, mu_A, m_B, mu_B, bg_type = args
    np.random.seed(seed)

    # Poisson resample
    y_A_boot = np.random.poisson(np.maximum(mu_A, 0.1))
    y_B_boot = np.random.poisson(np.maximum(mu_B, 0.1))

    try:
        fitter_A = ChannelFitter(m_A, y_A_boot, likelihood='poisson', bg_type=bg_type)
        fitter_B = ChannelFitter(m_B, y_B_boot, likelihood='poisson', bg_type=bg_type)

        params_A, nll_A = fitter_A.fit(n_restarts=10)
        params_B, nll_B = fitter_B.fit(n_restarts=10)
        nll_unc = nll_A + nll_B

        joint = JointFitter(fitter_A, fitter_B)
        params_joint, nll_con = joint.fit(params_A, params_B, n_restarts=10)

        return -2 * (nll_con - nll_unc)
    except Exception:
        return np.nan


def bootstrap_single_poisson_gaussian(args):
    """Bootstrap replicate for Pair 2 (Poisson + Gaussian)."""
    seed, m_A, mu_A, m_B, mu_B, sigma_B, bg_type = args
    np.random.seed(seed)

    # Poisson resample for A
    y_A_boot = np.random.poisson(np.maximum(mu_A, 0.1))

    # Gaussian resample for B
    y_B_boot = np.random.normal(mu_B, sigma_B)

    try:
        fitter_A = ChannelFitter(m_A, y_A_boot, likelihood='poisson', bg_type=bg_type)
        fitter_B = ChannelFitter(m_B, y_B_boot, sigma=sigma_B, likelihood='gaussian', bg_type=bg_type)

        params_A, nll_A = fitter_A.fit(n_restarts=10)
        params_B, nll_B = fitter_B.fit(n_restarts=10)
        nll_unc = nll_A + nll_B

        joint = JointFitter(fitter_A, fitter_B)
        params_joint, nll_con = joint.fit(params_A, params_B, n_restarts=10)

        return -2 * (nll_con - nll_unc)
    except Exception:
        return np.nan


def run_pair_analysis(df_A, df_B, pair_name, window, likelihood_B='poisson',
                      sigma_B=None, bg_type='linear', n_bootstrap=300):
    """Run complete analysis for one pair."""
    print(f"\n{'='*60}")
    print(f"PAIR: {pair_name}")
    print(f"Window: {window[0]}-{window[1]} MeV, Background: {bg_type}")
    print(f"{'='*60}")

    # Apply window
    df_A_win = apply_window(df_A, window)
    df_B_win = apply_window(df_B, window)

    m_A = df_A_win['m_center'].values
    y_A = df_A_win['yield'].values
    m_B = df_B_win['m_center'].values
    y_B = df_B_win['yield'].values

    if sigma_B is not None:
        sig_B = apply_window(sigma_B, window)['stat'].values
    else:
        sig_B = None

    print(f"Channel A: {len(m_A)} bins, {y_A.sum():.0f} total counts")
    print(f"Channel B: {len(m_B)} bins, {y_B.sum():.0f} total counts")

    # Create fitters
    fitter_A = ChannelFitter(m_A, y_A, likelihood='poisson', bg_type=bg_type)
    fitter_B = ChannelFitter(m_B, y_B, sigma=sig_B, likelihood=likelihood_B, bg_type=bg_type)

    # Unconstrained fits
    print("\nFitting Channel A...")
    params_A, nll_A = fitter_A.fit(n_restarts=30)
    quality_A = fitter_A.compute_fit_quality(params_A)
    ratio_A = fitter_A.extract_ratio(params_A)

    print("Fitting Channel B...")
    params_B, nll_B = fitter_B.fit(n_restarts=30)
    quality_B = fitter_B.compute_fit_quality(params_B)
    ratio_B = fitter_B.extract_ratio(params_B)

    nll_unc = nll_A + nll_B

    # Constrained fit
    print("Fitting Joint (constrained)...")
    joint = JointFitter(fitter_A, fitter_B)
    params_joint, nll_con = joint.fit(params_A, params_B, n_restarts=30)
    ratio_shared = joint.extract_shared_ratio(params_joint)

    # Likelihood ratio
    lambda_obs = -2 * (nll_con - nll_unc)

    print(f"\n--- Results ---")
    print(f"Channel A: R = {ratio_A['r']:.4f} * exp(i * {ratio_A['phi_deg']:.1f}°)")
    print(f"Channel B: R = {ratio_B['r']:.4f} * exp(i * {ratio_B['phi_deg']:.1f}°)")
    print(f"Shared:    R = {ratio_shared['r']:.4f} * exp(i * {ratio_shared['phi_deg']:.1f}°)")

    # Fit health
    print(f"\n--- Fit Quality ---")
    if likelihood_B == 'poisson':
        print(f"Channel A: χ²/dof = {quality_A['chi2_dof']:.3f}, deviance/dof = {quality_A['deviance_dof']:.3f}")
        print(f"Channel B: χ²/dof = {quality_B['chi2_dof']:.3f}, deviance/dof = {quality_B['deviance_dof']:.3f}")
        gate_A = quality_A['deviance_dof'] < 3
        gate_B = quality_B['deviance_dof'] < 3
    else:
        print(f"Channel A: χ²/dof = {quality_A['chi2_dof']:.3f}, deviance/dof = {quality_A['deviance_dof']:.3f}")
        print(f"Channel B: χ²/dof = {quality_B['chi2_dof']:.3f} (Gaussian)")
        gate_A = quality_A['deviance_dof'] < 3
        gate_B = quality_B['chi2_dof'] < 3

    gates_pass = gate_A and gate_B
    print(f"Gates pass: A={gate_A}, B={gate_B}")

    print(f"\n--- Likelihood Ratio ---")
    print(f"NLL unconstrained: {nll_unc:.2f}")
    print(f"NLL constrained:   {nll_con:.2f}")
    print(f"Λ = -2ΔlnL = {lambda_obs:.3f}")

    # Bootstrap
    print(f"\nRunning {n_bootstrap} bootstrap replicates...")
    n_workers = max(1, mp.cpu_count() - 1)

    if likelihood_B == 'poisson':
        args_list = [(seed, m_A, quality_A['mu'], m_B, quality_B['mu'], bg_type)
                     for seed in range(n_bootstrap)]
        with mp.Pool(n_workers) as pool:
            lambda_boots = list(pool.imap(bootstrap_single_poisson_poisson, args_list))
    else:
        args_list = [(seed, m_A, quality_A['mu'], m_B, quality_B['mu'], sig_B, bg_type)
                     for seed in range(n_bootstrap)]
        with mp.Pool(n_workers) as pool:
            lambda_boots = list(pool.imap(bootstrap_single_poisson_gaussian, args_list))

    lambda_boots = np.array([l for l in lambda_boots if not np.isnan(l)])
    n_valid = len(lambda_boots)
    n_exceed = np.sum(lambda_boots >= lambda_obs)
    p_value = n_exceed / n_valid if n_valid > 0 else np.nan

    print(f"Valid replicates: {n_valid}/{n_bootstrap}")
    print(f"Λ_boot median: {np.median(lambda_boots):.3f}")
    print(f"Λ_boot 95th: {np.percentile(lambda_boots, 95):.3f}")
    print(f"p-value: {p_value:.4f}")

    # Verdict
    if not gates_pass:
        verdict = "MODEL MISMATCH"
        reason = "Fit-health gates failed"
    elif p_value < 0.05:
        verdict = "DISFAVORED"
        reason = f"Rank-1 constraint rejected (p = {p_value:.4f})"
    else:
        verdict = "SUPPORTED"
        reason = f"Rank-1 constraint not rejected (p = {p_value:.4f})"

    print(f"\n>>> VERDICT: {verdict} <<<")
    print(f"Reason: {reason}")

    return {
        'pair_name': pair_name,
        'window': window,
        'bg_type': bg_type,
        'ratio_A': ratio_A,
        'ratio_B': ratio_B,
        'ratio_shared': ratio_shared,
        'nll_unc': nll_unc,
        'nll_con': nll_con,
        'lambda_obs': lambda_obs,
        'quality_A': {k: float(v) if isinstance(v, (np.floating, float)) else v
                      for k, v in quality_A.items() if k != 'mu'},
        'quality_B': {k: float(v) if isinstance(v, (np.floating, float)) else v
                      for k, v in quality_B.items() if k != 'mu'},
        'gates_pass': bool(gates_pass),
        'n_bootstrap': n_valid,
        'lambda_boot_median': float(np.median(lambda_boots)),
        'lambda_boot_95': float(np.percentile(lambda_boots, 95)),
        'p_value': float(p_value),
        'verdict': verdict,
        'reason': reason,
        'mu_A': quality_A['mu'],
        'mu_B': quality_B['mu'],
        'm_A': m_A,
        'm_B': m_B,
        'y_A': y_A,
        'y_B': y_B,
        'sigma_B': sig_B
    }


def make_fit_plots(results, output_path):
    """Generate fit diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Channel A
        ax1, ax2 = axes[0]

        ax1.errorbar(results['m_A'], results['y_A'],
                    yerr=np.sqrt(np.maximum(results['y_A'], 1)),
                    fmt='ko', ms=3, alpha=0.6, label='Data')
        ax1.plot(results['m_A'], results['mu_A'], 'r-', lw=2, label='Fit')

        for pc, params in PC_PARAMS.items():
            ax1.axvline(params['mass'], color='blue', ls='--', alpha=0.5, lw=1)

        ax1.set_ylabel('Events / 2 MeV')
        ax1.set_title(f'Channel A - {results["pair_name"]}')
        ax1.legend()
        ax1.set_xlim(results['window'])

        # Residuals A
        residuals_A = (results['y_A'] - results['mu_A']) / np.sqrt(np.maximum(results['mu_A'], 1))
        ax2.plot(results['m_A'], residuals_A, 'ko', ms=3, alpha=0.6)
        ax2.axhline(0, color='r', ls='-')
        ax2.axhline(2, color='gray', ls='--', alpha=0.5)
        ax2.axhline(-2, color='gray', ls='--', alpha=0.5)
        ax2.set_ylabel('Pull')
        ax2.set_ylim(-4, 4)
        ax2.set_xlim(results['window'])

        # Channel B
        ax3, ax4 = axes[1]

        if results['sigma_B'] is not None:
            yerr_B = results['sigma_B']
        else:
            yerr_B = np.sqrt(np.maximum(results['y_B'], 1))

        ax3.errorbar(results['m_B'], results['y_B'], yerr=yerr_B,
                    fmt='ko', ms=3, alpha=0.6, label='Data')
        ax3.plot(results['m_B'], results['mu_B'], 'r-', lw=2, label='Fit')

        for pc, params in PC_PARAMS.items():
            ax3.axvline(params['mass'], color='blue', ls='--', alpha=0.5, lw=1)

        ax3.set_xlabel('m(J/ψ p) [MeV]')
        ax3.set_ylabel('Events / 2 MeV')
        ax3.set_title(f'Channel B - {results["pair_name"]}')
        ax3.legend()
        ax3.set_xlim(results['window'])

        # Residuals B
        residuals_B = (results['y_B'] - results['mu_B']) / np.maximum(yerr_B, 1)
        ax4.plot(results['m_B'], residuals_B, 'ko', ms=3, alpha=0.6)
        ax4.axhline(0, color='r', ls='-')
        ax4.axhline(2, color='gray', ls='--', alpha=0.5)
        ax4.axhline(-2, color='gray', ls='--', alpha=0.5)
        ax4.set_xlabel('m(J/ψ p) [MeV]')
        ax4.set_ylabel('Pull')
        ax4.set_ylim(-4, 4)
        ax4.set_xlim(results['window'])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Plot error: {e}")


def main():
    """Main analysis."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'hepdata')
    out_dir = os.path.join(base_dir, 'out')
    logs_dir = os.path.join(base_dir, 'logs')

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 70)
    print("LHCb Pentaquark Rank-1 Bottleneck Test v2")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df_t1 = load_hepdata_csv(os.path.join(data_dir, '89271_t1_spectrum_full.csv'))
    df_t2 = load_hepdata_csv(os.path.join(data_dir, '89271_t2_spectrum_mKp_cut.csv'))
    df_t3 = load_hepdata_csv(os.path.join(data_dir, '89271_t3_spectrum_weighted.csv'))

    print(f"Table 1 (full): {len(df_t1)} bins")
    print(f"Table 2 (mKp cut): {len(df_t2)} bins")
    print(f"Table 3 (weighted): {len(df_t3)} bins")

    all_results = {}

    # ===========================================
    # PAIR 1: Table 1 vs Table 2 (both Poisson)
    # ===========================================
    print("\n" + "="*70)
    print("PAIR 1: Table 1 (full) vs Table 2 (mKp > 1.9 GeV cut)")
    print("Both channels: Poisson NLL")
    print("="*70)

    # Main analysis
    results_p1_main = run_pair_analysis(
        df_t1, df_t2, "Pair1_T1vsT2",
        window=WINDOW_DEFAULT,
        likelihood_B='poisson',
        bg_type='linear',
        n_bootstrap=300
    )
    all_results['pair1_main'] = results_p1_main
    make_fit_plots(results_p1_main, os.path.join(out_dir, 'fit_plots_pair1.png'))

    # Robustness: tighter window
    print("\n--- Robustness: Tight window ---")
    results_p1_tight = run_pair_analysis(
        df_t1, df_t2, "Pair1_T1vsT2_tight",
        window=WINDOW_TIGHT,
        likelihood_B='poisson',
        bg_type='linear',
        n_bootstrap=100
    )
    all_results['pair1_tight'] = results_p1_tight

    # Robustness: quadratic background
    print("\n--- Robustness: Quadratic background ---")
    results_p1_quad = run_pair_analysis(
        df_t1, df_t2, "Pair1_T1vsT2_quad",
        window=WINDOW_DEFAULT,
        likelihood_B='poisson',
        bg_type='quadratic',
        n_bootstrap=100
    )
    all_results['pair1_quad'] = results_p1_quad

    # ===========================================
    # PAIR 2: Table 1 vs Table 3 (Poisson + Gaussian)
    # ===========================================
    print("\n" + "="*70)
    print("PAIR 2: Table 1 (full) vs Table 3 (cos θ weighted)")
    print("Channel A: Poisson NLL, Channel B: Gaussian NLL")
    print("="*70)

    # Main analysis
    results_p2_main = run_pair_analysis(
        df_t1, df_t3, "Pair2_T1vsT3",
        window=WINDOW_DEFAULT,
        likelihood_B='gaussian',
        sigma_B=df_t3,
        bg_type='linear',
        n_bootstrap=300
    )
    all_results['pair2_main'] = results_p2_main
    make_fit_plots(results_p2_main, os.path.join(out_dir, 'fit_plots_pair2.png'))

    # Robustness: tighter window
    print("\n--- Robustness: Tight window ---")
    results_p2_tight = run_pair_analysis(
        df_t1, df_t3, "Pair2_T1vsT3_tight",
        window=WINDOW_TIGHT,
        likelihood_B='gaussian',
        sigma_B=df_t3,
        bg_type='linear',
        n_bootstrap=100
    )
    all_results['pair2_tight'] = results_p2_tight

    # Robustness: quadratic background
    print("\n--- Robustness: Quadratic background ---")
    results_p2_quad = run_pair_analysis(
        df_t1, df_t3, "Pair2_T1vsT3_quad",
        window=WINDOW_DEFAULT,
        likelihood_B='gaussian',
        sigma_B=df_t3,
        bg_type='quadratic',
        n_bootstrap=100
    )
    all_results['pair2_quad'] = results_p2_quad

    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\nPair 1 (T1 vs T2 - both Poisson):")
    print(f"  Main: {results_p1_main['verdict']} (p={results_p1_main['p_value']:.4f})")
    print(f"  Tight window: {results_p1_tight['verdict']} (p={results_p1_tight['p_value']:.4f})")
    print(f"  Quadratic bg: {results_p1_quad['verdict']} (p={results_p1_quad['p_value']:.4f})")

    print("\nPair 2 (T1 vs T3 - Poisson/Gaussian):")
    print(f"  Main: {results_p2_main['verdict']} (p={results_p2_main['p_value']:.4f})")
    print(f"  Tight window: {results_p2_tight['verdict']} (p={results_p2_tight['p_value']:.4f})")
    print(f"  Quadratic bg: {results_p2_quad['verdict']} (p={results_p2_quad['p_value']:.4f})")

    # Save results JSON
    results_json = {}
    for key, res in all_results.items():
        results_json[key] = {
            'pair_name': res['pair_name'],
            'window': list(res['window']),
            'bg_type': res['bg_type'],
            'ratio_A': {k: float(v) for k, v in res['ratio_A'].items()},
            'ratio_B': {k: float(v) for k, v in res['ratio_B'].items()},
            'ratio_shared': {k: float(v) for k, v in res['ratio_shared'].items()},
            'lambda_obs': float(res['lambda_obs']),
            'p_value': float(res['p_value']),
            'n_bootstrap': int(res['n_bootstrap']),
            'gates_pass': res['gates_pass'],
            'quality_A': res['quality_A'],
            'quality_B': res['quality_B'],
            'verdict': res['verdict'],
            'reason': res['reason']
        }

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Generate REPORT.md
    report = f"""# LHCb Pentaquark Rank-1 Bottleneck Test v2

## Executive Summary

### Pair 1 (Table 1 vs Table 2)
- **Channels**: Full spectrum vs m(Kp) > 1.9 GeV cut
- **Likelihood**: Both Poisson NLL
- **Verdict**: **{results_p1_main['verdict']}**

### Pair 2 (Table 1 vs Table 3)
- **Channels**: Full spectrum vs cos θ_Pc weighted
- **Likelihood**: Poisson (A) + Gaussian (B)
- **Verdict**: **{results_p2_main['verdict']}**

## Data Provenance

| Item | Value |
|------|-------|
| HEPData Record | [89271](https://www.hepdata.net/record/ins1728691) |
| Publication | PRL 122, 222001 (2019) |
| DOI | [10.1103/PhysRevLett.122.222001](https://doi.org/10.1103/PhysRevLett.122.222001) |

### Tables Used
- **Table 1**: Full m(J/ψ p) spectrum (raw counts)
- **Table 2**: m(J/ψ p) with m(Kp) > 1.9 GeV cut (raw counts)
- **Table 3**: cos θ_Pc weighted m(J/ψ p) spectrum (weighted, Gaussian errors)

## Analysis Configuration

- **Mass window**: {WINDOW_DEFAULT[0]}-{WINDOW_DEFAULT[1]} MeV (pentaquark region)
- **Model**: Coherent 3-BW (Pc4312, Pc4440, Pc4457) + linear background
- **Reference**: c₂ (Pc4440) fixed real positive
- **Test ratio**: R = c₃/c₂ = c(Pc4457)/c(Pc4440)

### Pentaquark Parameters (fixed)

| State | Mass [MeV] | Width [MeV] |
|-------|------------|-------------|
| Pc(4312)⁺ | 4311.9 | 9.8 |
| Pc(4440)⁺ | 4440.3 | 20.6 |
| Pc(4457)⁺ | 4457.3 | 6.4 |

## Pair 1 Results: Table 1 vs Table 2

### Fit Quality

| Channel | χ²/dof | Deviance/dof | Gate (<3) |
|---------|--------|--------------|-----------|
| A (full) | {results_p1_main['quality_A']['chi2_dof']:.3f} | {results_p1_main['quality_A']['deviance_dof']:.3f} | {'✓' if results_p1_main['quality_A']['deviance_dof'] < 3 else '✗'} |
| B (mKp cut) | {results_p1_main['quality_B']['chi2_dof']:.3f} | {results_p1_main['quality_B']['deviance_dof']:.3f} | {'✓' if results_p1_main['quality_B']['deviance_dof'] < 3 else '✗'} |

### Amplitude Ratios R = c(4457)/c(4440)

| Channel | |R| | arg(R) [°] |
|---------|-----|-----------|
| A (full) | {results_p1_main['ratio_A']['r']:.4f} | {results_p1_main['ratio_A']['phi_deg']:.1f} |
| B (mKp cut) | {results_p1_main['ratio_B']['r']:.4f} | {results_p1_main['ratio_B']['phi_deg']:.1f} |
| Shared | {results_p1_main['ratio_shared']['r']:.4f} | {results_p1_main['ratio_shared']['phi_deg']:.1f} |

### Likelihood Ratio Test

| Quantity | Value |
|----------|-------|
| Λ = -2ΔlnL | {results_p1_main['lambda_obs']:.3f} |
| Bootstrap p-value | {results_p1_main['p_value']:.4f} |
| Valid replicates | {results_p1_main['n_bootstrap']}/300 |

### Robustness Checks

| Variant | Verdict | p-value |
|---------|---------|---------|
| Main (4270-4520, linear) | {results_p1_main['verdict']} | {results_p1_main['p_value']:.4f} |
| Tight (4320-4490, linear) | {results_p1_tight['verdict']} | {results_p1_tight['p_value']:.4f} |
| Quadratic bg | {results_p1_quad['verdict']} | {results_p1_quad['p_value']:.4f} |

### Verdict: **{results_p1_main['verdict']}**
{results_p1_main['reason']}

---

## Pair 2 Results: Table 1 vs Table 3

### Fit Quality

| Channel | χ²/dof | Deviance/dof | Gate (<3) |
|---------|--------|--------------|-----------|
| A (full, Poisson) | {results_p2_main['quality_A']['chi2_dof']:.3f} | {results_p2_main['quality_A']['deviance_dof']:.3f} | {'✓' if results_p2_main['quality_A']['deviance_dof'] < 3 else '✗'} |
| B (weighted, Gaussian) | {results_p2_main['quality_B']['chi2_dof']:.3f} | - | {'✓' if results_p2_main['quality_B']['chi2_dof'] < 3 else '✗'} |

### Amplitude Ratios R = c(4457)/c(4440)

| Channel | |R| | arg(R) [°] |
|---------|-----|-----------|
| A (full) | {results_p2_main['ratio_A']['r']:.4f} | {results_p2_main['ratio_A']['phi_deg']:.1f} |
| B (weighted) | {results_p2_main['ratio_B']['r']:.4f} | {results_p2_main['ratio_B']['phi_deg']:.1f} |
| Shared | {results_p2_main['ratio_shared']['r']:.4f} | {results_p2_main['ratio_shared']['phi_deg']:.1f} |

### Likelihood Ratio Test

| Quantity | Value |
|----------|-------|
| Λ = -2ΔlnL | {results_p2_main['lambda_obs']:.3f} |
| Bootstrap p-value | {results_p2_main['p_value']:.4f} |
| Valid replicates | {results_p2_main['n_bootstrap']}/300 |

### Robustness Checks

| Variant | Verdict | p-value |
|---------|---------|---------|
| Main (4270-4520, linear) | {results_p2_main['verdict']} | {results_p2_main['p_value']:.4f} |
| Tight (4320-4490, linear) | {results_p2_tight['verdict']} | {results_p2_tight['p_value']:.4f} |
| Quadratic bg | {results_p2_quad['verdict']} | {results_p2_quad['p_value']:.4f} |

### Verdict: **{results_p2_main['verdict']}**
{results_p2_main['reason']}

---

## Conclusion

The rank-1 factorization hypothesis for the LHCb pentaquark amplitude ratios was tested using two independent channel pairs from HEPData record 89271.

**Pair 1** (selection channels): {results_p1_main['verdict']}
**Pair 2** (weighted projection): {results_p2_main['verdict']}

---
*Analysis: Poisson NLL for raw counts, Gaussian NLL for weighted data*
*Optimization: Multi-start (30 restarts), L-BFGS-B + Nelder-Mead refinement*
*Bootstrap: 300 replicates (main), 100 replicates (robustness)*
"""

    with open(os.path.join(out_dir, 'REPORT.md'), 'w') as f:
        f.write(report)

    # Commands log
    commands = """# Commands for LHCb Pentaquark Rank-1 Test v2

# 1. Copy data
cp lhcb_rank1_test/data/hepdata/89271_t*.csv lhcb_rank1_test_v2/data/hepdata/

# 2. Run analysis
nohup python3 -u src/pentaquark_rank1_v2.py > logs/analysis.log 2>&1 &

# 3. Monitor
tail -f logs/analysis.log
"""

    with open(os.path.join(logs_dir, 'COMMANDS.txt'), 'w') as f:
        f.write(commands)

    print(f"\nOutputs saved to {out_dir}/")

    return all_results


if __name__ == '__main__':
    results = main()
