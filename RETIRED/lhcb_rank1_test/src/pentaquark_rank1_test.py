#!/usr/bin/env python3
"""
LHCb Pentaquark Rank-1 Bottleneck Test

Tests whether the coupling ratios between Pc states are channel-invariant
across two distinct spectral projections (unweighted vs cos θ_Pc weighted).

This is analogous to testing whether the coupling matrix factorizes as:
    c_{i,α} = a_i * k_α

Uses Poisson NLL on binned data with fit-health gates.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.special import gammaln
from scipy.stats import chi2
import json
import warnings
import multiprocessing as mp
from functools import partial
import os

warnings.filterwarnings('ignore')

# Pentaquark parameters from LHCb 2019 (PRL 122, 222001)
# Fixed masses and widths for stability
PC_PARAMS = {
    'Pc4312': {'mass': 4311.9, 'width': 9.8},
    'Pc4440': {'mass': 4440.3, 'width': 20.6},
    'Pc4457': {'mass': 4457.3, 'width': 6.4},
}

# Fit window (focus on pentaquark region)
FIT_WINDOW = (4250, 4550)  # MeV

def load_hepdata_csv(filepath):
    """Load HEPData CSV, skipping comment lines."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the header line (first line not starting with #)
    data_lines = [l for l in lines if not l.startswith('#')]

    from io import StringIO
    df = pd.read_csv(StringIO(''.join(data_lines)))

    # Standardize column names
    df.columns = ['m_center', 'm_low', 'm_high', 'yield', 'stat_up', 'stat_down']

    # Convert yield per MeV to counts (bin width = 2 MeV)
    df['counts'] = df['yield'] * 2.0  # DN/DM * bin_width
    df['counts'] = np.maximum(df['counts'], 0)  # Ensure non-negative

    return df


def breit_wigner_amplitude(m, mass, width):
    """Relativistic Breit-Wigner amplitude (complex)."""
    return 1.0 / (mass**2 - m**2 - 1j * mass * width)


def model_amplitude(m, params, pc_names=['Pc4312', 'Pc4440', 'Pc4457']):
    """
    Coherent sum of Breit-Wigner amplitudes.

    params = [re_0, im_0, re_1, im_1, re_2, im_2, ...]
    """
    amp = 0.0 + 0.0j
    for i, pc in enumerate(pc_names):
        c_re = params[2*i]
        c_im = params[2*i + 1]
        c = c_re + 1j * c_im
        amp += c * breit_wigner_amplitude(m, PC_PARAMS[pc]['mass'], PC_PARAMS[pc]['width'])
    return amp


def background_model(m, bg_params, m_threshold=4034):
    """
    Smooth threshold + polynomial background.
    bg_params = [norm, p0, p1, p2]
    """
    # Phase space factor (approximate)
    q = np.sqrt(np.maximum((m - m_threshold) / 1000.0, 1e-10))

    # Polynomial in (m - 4300)/100
    x = (m - 4300) / 100.0
    poly = bg_params[1] + bg_params[2] * x + bg_params[3] * x**2

    return bg_params[0] * q * np.exp(-poly)


def full_model(m, signal_params, bg_params, pc_names=['Pc4312', 'Pc4440', 'Pc4457']):
    """Signal + background model."""
    amp = model_amplitude(m, signal_params, pc_names)
    signal = np.abs(amp)**2
    bg = background_model(m, bg_params)
    return signal + bg


def poisson_nll(params, m_vals, counts, n_signal_params, pc_names):
    """
    Poisson negative log-likelihood.

    -ln L = sum_i [ mu_i - y_i * ln(mu_i) + ln(y_i!) ]
    """
    signal_params = params[:n_signal_params]
    bg_params = params[n_signal_params:]

    mu = full_model(m_vals, signal_params, bg_params, pc_names)
    mu = np.maximum(mu, 1e-10)  # Prevent log(0)

    # Poisson NLL (dropping constant term)
    nll = np.sum(mu - counts * np.log(mu))

    # Add penalty for negative predictions
    if np.any(mu < 0):
        nll += 1e10

    return nll


def compute_fit_quality(m_vals, counts, mu_pred):
    """Compute chi2/dof and Poisson deviance/dof."""
    n_bins = len(counts)

    # Pearson chi-squared
    chi2_stat = np.sum((counts - mu_pred)**2 / np.maximum(mu_pred, 1))

    # Poisson deviance
    mask = counts > 0
    deviance = 2 * np.sum(
        counts[mask] * np.log(counts[mask] / np.maximum(mu_pred[mask], 1e-10))
        - (counts[mask] - mu_pred[mask])
    )
    # Add contribution from zero-count bins
    deviance += 2 * np.sum(mu_pred[~mask])

    return chi2_stat, deviance


def fit_single_channel(m_vals, counts, n_restarts=30, pc_names=['Pc4312', 'Pc4440', 'Pc4457']):
    """
    Fit a single channel with multi-start optimization.

    Returns best-fit parameters and diagnostics.
    """
    n_pc = len(pc_names)
    n_signal_params = 2 * n_pc  # re, im for each
    n_bg_params = 4
    n_params = n_signal_params + n_bg_params

    best_nll = np.inf
    best_params = None

    # Multi-start optimization
    for restart in range(n_restarts):
        np.random.seed(restart * 7919)

        # Random initialization
        signal_init = np.random.randn(n_signal_params) * 100
        bg_init = [np.sum(counts) / len(counts), 1.0, 0.01, 0.001]
        bg_init = np.array(bg_init) * (1 + 0.2 * np.random.randn(4))

        x0 = np.concatenate([signal_init, bg_init])

        # Bounds
        bounds = [(None, None)] * n_signal_params  # Signal params unbounded
        bounds += [(1, None), (None, None), (None, None), (None, None)]  # bg params

        try:
            # L-BFGS-B
            result = minimize(
                poisson_nll, x0,
                args=(m_vals, counts, n_signal_params, pc_names),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10}
            )

            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()

            # Nelder-Mead refinement
            result2 = minimize(
                poisson_nll, result.x,
                args=(m_vals, counts, n_signal_params, pc_names),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )

            if result2.fun < best_nll:
                best_nll = result2.fun
                best_params = result2.x.copy()

        except Exception as e:
            continue

    if best_params is None:
        raise RuntimeError("All optimization attempts failed")

    # Extract couplings
    signal_params = best_params[:n_signal_params]
    bg_params = best_params[n_signal_params:]

    couplings = {}
    for i, pc in enumerate(pc_names):
        c_re = signal_params[2*i]
        c_im = signal_params[2*i + 1]
        couplings[pc] = {'re': c_re, 'im': c_im, 'mag': np.sqrt(c_re**2 + c_im**2),
                         'phase': np.arctan2(c_im, c_re)}

    # Compute predictions
    mu_pred = full_model(m_vals, signal_params, bg_params, pc_names)

    # Fit quality
    n_dof = len(counts) - n_params
    chi2_stat, deviance = compute_fit_quality(m_vals, counts, mu_pred)

    return {
        'nll': best_nll,
        'params': best_params,
        'signal_params': signal_params,
        'bg_params': bg_params,
        'couplings': couplings,
        'mu_pred': mu_pred,
        'chi2': chi2_stat,
        'chi2_dof': chi2_stat / n_dof,
        'deviance': deviance,
        'deviance_dof': deviance / n_dof,
        'n_dof': n_dof
    }


def compute_ratio(couplings, pc1='Pc4457', pc2='Pc4440'):
    """Compute complex amplitude ratio c_1/c_2."""
    c1 = couplings[pc1]['re'] + 1j * couplings[pc1]['im']
    c2 = couplings[pc2]['re'] + 1j * couplings[pc2]['im']

    if np.abs(c2) < 1e-10:
        return {'mag': np.nan, 'phase': np.nan, 're': np.nan, 'im': np.nan}

    ratio = c1 / c2
    return {
        'mag': np.abs(ratio),
        'phase': np.angle(ratio),
        're': np.real(ratio),
        'im': np.imag(ratio)
    }


def constrained_nll(params, m_A, counts_A, m_B, counts_B,
                    n_signal_params, pc_names, ratio_constraint):
    """
    Joint NLL with constraint that ratio R_A = R_B.

    ratio_constraint = (pc1, pc2) - which ratio to constrain

    Parameterization:
    - Channel A: [re_0, im_0, ..., bg_A]
    - Channel B: shares ratio, different overall scale and background
    """
    pc1, pc2 = ratio_constraint
    n_pc = len(pc_names)

    # Parse parameters
    # Channel A signal params
    signal_A = params[:n_signal_params]
    # Channel A background
    bg_A = params[n_signal_params:n_signal_params+4]
    # Channel B: overall scale, phase offset, background
    # We parameterize B's couplings as: c_B = scale_B * exp(i*phi_B) * c_A * (k_B / k_A)
    # But for rank-1, the ratios must be the same
    # Simpler: parameterize B couplings directly but constrain ratio
    scale_B = params[n_signal_params+4]
    phase_offset = params[n_signal_params+5]
    bg_B = params[n_signal_params+6:n_signal_params+10]

    # Construct channel B couplings with same ratio
    # c_B,i = scale_B * exp(i*phase_offset) * c_A,i
    signal_B = np.zeros(n_signal_params)
    for i in range(n_pc):
        c_A_re = signal_A[2*i]
        c_A_im = signal_A[2*i + 1]
        c_A = c_A_re + 1j * c_A_im
        c_B = scale_B * np.exp(1j * phase_offset) * c_A
        signal_B[2*i] = np.real(c_B)
        signal_B[2*i + 1] = np.imag(c_B)

    # NLL for channel A
    nll_A = poisson_nll(
        np.concatenate([signal_A, bg_A]),
        m_A, counts_A, n_signal_params, pc_names
    )

    # NLL for channel B
    nll_B = poisson_nll(
        np.concatenate([signal_B, bg_B]),
        m_B, counts_B, n_signal_params, pc_names
    )

    return nll_A + nll_B


def fit_constrained(m_A, counts_A, m_B, counts_B,
                    unconstrained_A, unconstrained_B,
                    n_restarts=30, pc_names=['Pc4312', 'Pc4440', 'Pc4457']):
    """
    Fit both channels jointly with rank-1 constraint.
    """
    n_pc = len(pc_names)
    n_signal_params = 2 * n_pc

    # Total params: signal_A (6) + bg_A (4) + scale_B (1) + phase_B (1) + bg_B (4) = 16
    n_params = n_signal_params + 4 + 2 + 4

    best_nll = np.inf
    best_params = None

    for restart in range(n_restarts):
        np.random.seed(restart * 7919 + 1000)

        # Initialize from unconstrained fits
        signal_A_init = unconstrained_A['signal_params'] * (1 + 0.1 * np.random.randn(n_signal_params))
        bg_A_init = unconstrained_A['bg_params'] * (1 + 0.1 * np.random.randn(4))

        # Estimate scale and phase from unconstrained fits
        c_A_ref = unconstrained_A['couplings']['Pc4440']['re'] + 1j * unconstrained_A['couplings']['Pc4440']['im']
        c_B_ref = unconstrained_B['couplings']['Pc4440']['re'] + 1j * unconstrained_B['couplings']['Pc4440']['im']

        if np.abs(c_A_ref) > 1e-10:
            scale_init = np.abs(c_B_ref / c_A_ref)
            phase_init = np.angle(c_B_ref / c_A_ref)
        else:
            scale_init = 1.0
            phase_init = 0.0

        scale_init *= (1 + 0.2 * np.random.randn())
        phase_init += 0.2 * np.random.randn()

        bg_B_init = unconstrained_B['bg_params'] * (1 + 0.1 * np.random.randn(4))

        x0 = np.concatenate([signal_A_init, bg_A_init, [scale_init, phase_init], bg_B_init])

        # Bounds
        bounds = [(None, None)] * n_signal_params  # Signal A
        bounds += [(1, None), (None, None), (None, None), (None, None)]  # bg_A
        bounds += [(0.01, 100), (-np.pi, np.pi)]  # scale_B, phase_B
        bounds += [(1, None), (None, None), (None, None), (None, None)]  # bg_B

        try:
            result = minimize(
                constrained_nll, x0,
                args=(m_A, counts_A, m_B, counts_B, n_signal_params, pc_names, ('Pc4457', 'Pc4440')),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-10}
            )

            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()

            # Nelder-Mead refinement
            result2 = minimize(
                constrained_nll, result.x,
                args=(m_A, counts_A, m_B, counts_B, n_signal_params, pc_names, ('Pc4457', 'Pc4440')),
                method='Nelder-Mead',
                options={'maxiter': 5000}
            )

            if result2.fun < best_nll:
                best_nll = result2.fun
                best_params = result2.x.copy()

        except Exception:
            continue

    if best_params is None:
        raise RuntimeError("Constrained optimization failed")

    # Extract results
    signal_A = best_params[:n_signal_params]
    bg_A = best_params[n_signal_params:n_signal_params+4]
    scale_B = best_params[n_signal_params+4]
    phase_B = best_params[n_signal_params+5]
    bg_B = best_params[n_signal_params+6:]

    # Compute channel B signal
    signal_B = np.zeros(n_signal_params)
    for i in range(n_pc):
        c_A = signal_A[2*i] + 1j * signal_A[2*i + 1]
        c_B = scale_B * np.exp(1j * phase_B) * c_A
        signal_B[2*i] = np.real(c_B)
        signal_B[2*i + 1] = np.imag(c_B)

    # Predictions
    mu_A = full_model(m_A, signal_A, bg_A, pc_names)
    mu_B = full_model(m_B, signal_B, bg_B, pc_names)

    # Extract couplings for both channels
    couplings_A = {}
    couplings_B = {}
    for i, pc in enumerate(pc_names):
        c_A = signal_A[2*i] + 1j * signal_A[2*i + 1]
        c_B = signal_B[2*i] + 1j * signal_B[2*i + 1]
        couplings_A[pc] = {'re': np.real(c_A), 'im': np.imag(c_A),
                          'mag': np.abs(c_A), 'phase': np.angle(c_A)}
        couplings_B[pc] = {'re': np.real(c_B), 'im': np.imag(c_B),
                          'mag': np.abs(c_B), 'phase': np.angle(c_B)}

    # Compute shared ratio
    ratio_shared = compute_ratio(couplings_A, 'Pc4457', 'Pc4440')

    return {
        'nll': best_nll,
        'params': best_params,
        'couplings_A': couplings_A,
        'couplings_B': couplings_B,
        'ratio_shared': ratio_shared,
        'mu_A': mu_A,
        'mu_B': mu_B,
        'scale_B': scale_B,
        'phase_B': phase_B
    }


def bootstrap_single(args):
    """Single bootstrap replicate - for parallel execution."""
    seed, m_A, mu_A, m_B, mu_B, n_signal_params, pc_names = args

    np.random.seed(seed)

    # Generate Poisson samples from constrained fit predictions
    counts_A_boot = np.random.poisson(np.maximum(mu_A, 0.1))
    counts_B_boot = np.random.poisson(np.maximum(mu_B, 0.1))

    try:
        # Fit unconstrained
        fit_A = fit_single_channel(m_A, counts_A_boot, n_restarts=10, pc_names=pc_names)
        fit_B = fit_single_channel(m_B, counts_B_boot, n_restarts=10, pc_names=pc_names)
        nll_unconstrained = fit_A['nll'] + fit_B['nll']

        # Fit constrained
        fit_const = fit_constrained(m_A, counts_A_boot, m_B, counts_B_boot,
                                    fit_A, fit_B, n_restarts=10, pc_names=pc_names)
        nll_constrained = fit_const['nll']

        lambda_boot = -2 * (nll_constrained - nll_unconstrained)
        return lambda_boot

    except Exception:
        return np.nan


def run_bootstrap(m_A, mu_A_constrained, m_B, mu_B_constrained,
                  n_replicates=300, pc_names=['Pc4312', 'Pc4440', 'Pc4457']):
    """
    Parametric bootstrap from constrained fit.
    """
    n_signal_params = 2 * len(pc_names)

    # Prepare arguments for parallel execution
    args_list = [
        (seed, m_A, mu_A_constrained, m_B, mu_B_constrained, n_signal_params, pc_names)
        for seed in range(n_replicates)
    ]

    # Use multiprocessing
    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Running {n_replicates} bootstrap replicates with {n_workers} workers...")

    with mp.Pool(n_workers) as pool:
        lambda_boots = list(pool.imap(bootstrap_single, args_list))

    lambda_boots = np.array([l for l in lambda_boots if not np.isnan(l)])

    return lambda_boots


def make_fit_plot(m_vals, counts, mu_pred, channel_name, output_path, couplings):
    """Generate fit diagnostic plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                        gridspec_kw={'height_ratios': [3, 1]})

        # Main plot
        ax1.errorbar(m_vals, counts, yerr=np.sqrt(np.maximum(counts, 1)),
                     fmt='ko', ms=2, alpha=0.5, label='Data')
        ax1.plot(m_vals, mu_pred, 'r-', lw=1.5, label='Fit')

        # Mark pentaquark positions
        for pc, params in PC_PARAMS.items():
            ax1.axvline(params['mass'], color='blue', ls='--', alpha=0.5, lw=1)
            ax1.text(params['mass'], ax1.get_ylim()[1]*0.9, pc.replace('Pc', 'Pc(')+')',
                    rotation=90, va='top', fontsize=8)

        ax1.set_ylabel('Events / 2 MeV')
        ax1.set_title(f'Channel {channel_name}: m(J/ψ p) Spectrum Fit')
        ax1.legend()
        ax1.set_xlim(FIT_WINDOW)

        # Residuals
        residuals = (counts - mu_pred) / np.sqrt(np.maximum(mu_pred, 1))
        ax2.plot(m_vals, residuals, 'ko', ms=2, alpha=0.5)
        ax2.axhline(0, color='r', ls='-')
        ax2.axhline(2, color='gray', ls='--', alpha=0.5)
        ax2.axhline(-2, color='gray', ls='--', alpha=0.5)
        ax2.set_xlabel('m(J/ψ p) [MeV]')
        ax2.set_ylabel('Pull')
        ax2.set_xlim(FIT_WINDOW)
        ax2.set_ylim(-5, 5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved plot: {output_path}")

    except Exception as e:
        print(f"Could not generate plot: {e}")


def main():
    """Main analysis routine."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'hepdata')
    out_dir = os.path.join(base_dir, 'out')
    logs_dir = os.path.join(base_dir, 'logs')

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 60)
    print("LHCb Pentaquark Rank-1 Bottleneck Test")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df_A = load_hepdata_csv(os.path.join(data_dir, '89271_t1_spectrum_full.csv'))
    df_B = load_hepdata_csv(os.path.join(data_dir, '89271_t3_spectrum_weighted.csv'))

    print(f"Channel A (unweighted): {len(df_A)} bins, {df_A['counts'].sum():.0f} total events")
    print(f"Channel B (weighted):   {len(df_B)} bins, {df_B['counts'].sum():.0f} total events")

    # Apply fit window
    mask_A = (df_A['m_center'] >= FIT_WINDOW[0]) & (df_A['m_center'] <= FIT_WINDOW[1])
    mask_B = (df_B['m_center'] >= FIT_WINDOW[0]) & (df_B['m_center'] <= FIT_WINDOW[1])

    m_A = df_A.loc[mask_A, 'm_center'].values
    counts_A = df_A.loc[mask_A, 'counts'].values

    m_B = df_B.loc[mask_B, 'm_center'].values
    counts_B = df_B.loc[mask_B, 'counts'].values

    print(f"\nFit window: {FIT_WINDOW[0]}-{FIT_WINDOW[1]} MeV")
    print(f"Channel A: {len(m_A)} bins in window")
    print(f"Channel B: {len(m_B)} bins in window")

    pc_names = ['Pc4312', 'Pc4440', 'Pc4457']

    # Unconstrained fits
    print("\n" + "=" * 40)
    print("STEP 1: Unconstrained Fits")
    print("=" * 40)

    print("\nFitting Channel A (unweighted)...")
    fit_A = fit_single_channel(m_A, counts_A, n_restarts=30, pc_names=pc_names)

    print("\nFitting Channel B (weighted)...")
    fit_B = fit_single_channel(m_B, counts_B, n_restarts=30, pc_names=pc_names)

    # Fit health check
    print("\n--- Fit Quality Check ---")
    print(f"Channel A: χ²/dof = {fit_A['chi2_dof']:.3f}, deviance/dof = {fit_A['deviance_dof']:.3f}")
    print(f"Channel B: χ²/dof = {fit_B['chi2_dof']:.3f}, deviance/dof = {fit_B['deviance_dof']:.3f}")

    gate_pass_A = (fit_A['chi2_dof'] < 3) and (fit_A['deviance_dof'] < 3)
    gate_pass_B = (fit_B['chi2_dof'] < 3) and (fit_B['deviance_dof'] < 3)

    if not gate_pass_A:
        print("WARNING: Channel A fails fit-health gate!")
    if not gate_pass_B:
        print("WARNING: Channel B fails fit-health gate!")

    gates_pass = gate_pass_A and gate_pass_B

    # Extract ratios
    ratio_A = compute_ratio(fit_A['couplings'], 'Pc4457', 'Pc4440')
    ratio_B = compute_ratio(fit_B['couplings'], 'Pc4457', 'Pc4440')

    print("\n--- Amplitude Ratios R = c(4457)/c(4440) ---")
    print(f"Channel A: |R| = {ratio_A['mag']:.4f}, arg(R) = {np.degrees(ratio_A['phase']):.1f}°")
    print(f"Channel B: |R| = {ratio_B['mag']:.4f}, arg(R) = {np.degrees(ratio_B['phase']):.1f}°")

    # Generate plots
    make_fit_plot(m_A, counts_A, fit_A['mu_pred'], 'A (unweighted)',
                  os.path.join(out_dir, 'fit_A.png'), fit_A['couplings'])
    make_fit_plot(m_B, counts_B, fit_B['mu_pred'], 'B (cos θ weighted)',
                  os.path.join(out_dir, 'fit_B.png'), fit_B['couplings'])

    # Save individual fit results
    fit_A_json = {
        'nll': fit_A['nll'],
        'chi2_dof': fit_A['chi2_dof'],
        'deviance_dof': fit_A['deviance_dof'],
        'n_dof': fit_A['n_dof'],
        'couplings': {k: {kk: float(vv) for kk, vv in v.items()}
                      for k, v in fit_A['couplings'].items()},
        'ratio_4457_4440': {k: float(v) for k, v in ratio_A.items()},
        'bg_params': fit_A['bg_params'].tolist()
    }

    fit_B_json = {
        'nll': fit_B['nll'],
        'chi2_dof': fit_B['chi2_dof'],
        'deviance_dof': fit_B['deviance_dof'],
        'n_dof': fit_B['n_dof'],
        'couplings': {k: {kk: float(vv) for kk, vv in v.items()}
                      for k, v in fit_B['couplings'].items()},
        'ratio_4457_4440': {k: float(v) for k, v in ratio_B.items()},
        'bg_params': fit_B['bg_params'].tolist()
    }

    with open(os.path.join(out_dir, 'fit_A.json'), 'w') as f:
        json.dump(fit_A_json, f, indent=2)
    with open(os.path.join(out_dir, 'fit_B.json'), 'w') as f:
        json.dump(fit_B_json, f, indent=2)

    # Constrained fit
    print("\n" + "=" * 40)
    print("STEP 2: Constrained Fit (Rank-1 Test)")
    print("=" * 40)

    print("\nFitting with R_A = R_B constraint...")
    fit_constrained_result = fit_constrained(
        m_A, counts_A, m_B, counts_B,
        fit_A, fit_B, n_restarts=30, pc_names=pc_names
    )

    nll_unconstrained = fit_A['nll'] + fit_B['nll']
    nll_constrained = fit_constrained_result['nll']

    # Likelihood ratio
    lambda_obs = -2 * (nll_constrained - nll_unconstrained)

    print(f"\nNLL unconstrained (A+B): {nll_unconstrained:.2f}")
    print(f"NLL constrained:         {nll_constrained:.2f}")
    print(f"Λ = -2ΔlnL:              {lambda_obs:.3f}")
    print(f"Shared ratio |R|: {fit_constrained_result['ratio_shared']['mag']:.4f}")

    # Bootstrap
    print("\n" + "=" * 40)
    print("STEP 3: Bootstrap p-value")
    print("=" * 40)

    lambda_boots = run_bootstrap(
        m_A, fit_constrained_result['mu_A'],
        m_B, fit_constrained_result['mu_B'],
        n_replicates=300, pc_names=pc_names
    )

    n_valid = len(lambda_boots)
    n_exceed = np.sum(lambda_boots >= lambda_obs)
    p_value = n_exceed / n_valid if n_valid > 0 else np.nan

    print(f"\nBootstrap results:")
    print(f"  Valid replicates: {n_valid}/300")
    print(f"  Λ_obs = {lambda_obs:.3f}")
    print(f"  Λ_boot median = {np.median(lambda_boots):.3f}")
    print(f"  Λ_boot 95th = {np.percentile(lambda_boots, 95):.3f}")
    print(f"  p-value = {p_value:.4f} ({n_exceed}/{n_valid})")

    # Determine verdict
    print("\n" + "=" * 40)
    print("VERDICT")
    print("=" * 40)

    if not gates_pass:
        verdict = "MODEL MISMATCH"
        verdict_reason = "Fit-health gates failed (χ²/dof or deviance/dof > 3)"
    elif p_value < 0.05:
        verdict = "DISFAVORED"
        verdict_reason = f"Rank-1 constraint rejected at p = {p_value:.4f} < 0.05"
    else:
        verdict = "SUPPORTED"
        verdict_reason = f"Rank-1 constraint not rejected (p = {p_value:.4f})"

    print(f"\n>>> {verdict} <<<")
    print(f"Reason: {verdict_reason}")

    # Save comprehensive results
    results = {
        'data_source': {
            'hepdata_record': 89271,
            'inspire_id': 1728691,
            'doi': '10.1103/PhysRevLett.122.222001',
            'url': 'https://www.hepdata.net/record/ins1728691',
            'channel_A': 'Table 1 - Full m(J/ψ p) spectrum (unweighted)',
            'channel_B': 'Table 3 - cos θ_Pc weighted m(J/ψ p) spectrum'
        },
        'fit_window': list(FIT_WINDOW),
        'pentaquark_params': PC_PARAMS,
        'unconstrained': {
            'channel_A': fit_A_json,
            'channel_B': fit_B_json,
            'nll_total': nll_unconstrained
        },
        'constrained': {
            'nll': nll_constrained,
            'ratio_shared': {k: float(v) for k, v in fit_constrained_result['ratio_shared'].items()},
            'scale_B': float(fit_constrained_result['scale_B']),
            'phase_B_deg': float(np.degrees(fit_constrained_result['phase_B']))
        },
        'likelihood_ratio_test': {
            'lambda_obs': float(lambda_obs),
            'n_bootstrap': n_valid,
            'lambda_boot_median': float(np.median(lambda_boots)),
            'lambda_boot_95': float(np.percentile(lambda_boots, 95)),
            'p_value': float(p_value)
        },
        'fit_health': {
            'A_chi2_dof': float(fit_A['chi2_dof']),
            'A_deviance_dof': float(fit_A['deviance_dof']),
            'B_chi2_dof': float(fit_B['chi2_dof']),
            'B_deviance_dof': float(fit_B['deviance_dof']),
            'gates_pass': bool(gates_pass)
        },
        'verdict': verdict,
        'verdict_reason': verdict_reason
    }

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Generate REPORT.md
    report = f"""# LHCb Pentaquark Rank-1 Bottleneck Test

## Executive Summary
**Verdict: {verdict}**

{verdict_reason}

## Data Provenance

| Item | Value |
|------|-------|
| HEPData Record | [89271](https://www.hepdata.net/record/ins1728691) |
| INSPIRE ID | 1728691 |
| Publication | PRL 122, 222001 (2019) |
| DOI | [10.1103/PhysRevLett.122.222001](https://doi.org/10.1103/PhysRevLett.122.222001) |

### Channel Definitions

- **Channel A**: Table 1 - Full m(J/ψ p) invariant mass spectrum (unweighted)
- **Channel B**: Table 3 - cos θ_Pc weighted m(J/ψ p) spectrum

Both spectra from Λ_b → J/ψ p K⁻ decay at √s = 7, 8, 13 TeV.

## Pentaquark Family

Testing the 2019 LHCb narrow pentaquark triple:

| State | Mass [MeV] | Width [MeV] |
|-------|------------|-------------|
| Pc(4312)⁺ | 4311.9 ± 0.7 | 9.8 ± 2.7 |
| Pc(4440)⁺ | 4440.3 ± 1.3 | 20.6 ± 4.9 |
| Pc(4457)⁺ | 4457.3 ± 0.6 | 6.4 ± 2.0 |

## Fit Quality

### Fit Health Gates (threshold: < 3.0)

| Channel | χ²/dof | Deviance/dof | Pass |
|---------|--------|--------------|------|
| A (unweighted) | {fit_A['chi2_dof']:.3f} | {fit_A['deviance_dof']:.3f} | {'✓' if gate_pass_A else '✗'} |
| B (weighted) | {fit_B['chi2_dof']:.3f} | {fit_B['deviance_dof']:.3f} | {'✓' if gate_pass_B else '✗'} |

**Gates overall: {'PASS' if gates_pass else 'FAIL'}**

## Amplitude Ratio Results

Testing ratio R = c(Pc4457)/c(Pc4440)

### Unconstrained Fits

| Channel | |R| | arg(R) [°] |
|---------|-----|-----------|
| A | {ratio_A['mag']:.4f} | {np.degrees(ratio_A['phase']):.1f} |
| B | {ratio_B['mag']:.4f} | {np.degrees(ratio_B['phase']):.1f} |

### Constrained Fit (Rank-1)

| Parameter | Value |
|-----------|-------|
| |R_shared| | {fit_constrained_result['ratio_shared']['mag']:.4f} |
| arg(R_shared) [°] | {np.degrees(fit_constrained_result['ratio_shared']['phase']):.1f} |
| scale_B | {fit_constrained_result['scale_B']:.4f} |
| phase_B [°] | {np.degrees(fit_constrained_result['phase_B']):.1f} |

## Likelihood Ratio Test

| Quantity | Value |
|----------|-------|
| NLL (unconstrained A+B) | {nll_unconstrained:.2f} |
| NLL (constrained) | {nll_constrained:.2f} |
| **Λ = -2ΔlnL** | **{lambda_obs:.3f}** |

## Bootstrap p-value

- Parametric bootstrap from constrained fit
- {n_valid} valid replicates out of 300

| Statistic | Value |
|-----------|-------|
| Λ_obs | {lambda_obs:.3f} |
| Λ_boot (median) | {np.median(lambda_boots):.3f} |
| Λ_boot (95th percentile) | {np.percentile(lambda_boots, 95):.3f} |
| **p-value** | **{p_value:.4f}** |

## Conclusion

**{verdict}**

The rank-1 factorization hypothesis states that the coupling matrix c_{{i,α}} (where i indexes pentaquark states and α indexes channels) factorizes as c_{{i,α}} = a_i · k_α. This implies channel-invariant amplitude ratios.

{'The data are consistent with this factorization.' if verdict == 'SUPPORTED' else 'The data show significant deviation from rank-1 factorization.' if verdict == 'DISFAVORED' else 'The fit quality is insufficient to draw conclusions.'}

## Files Generated

- `out/fit_A.json` - Channel A fit parameters
- `out/fit_B.json` - Channel B fit parameters
- `out/fit_A.png` - Channel A fit plot
- `out/fit_B.png` - Channel B fit plot
- `out/results.json` - Complete results

---
*Analysis performed using Poisson NLL with multi-start optimization (30 restarts)*
*Bootstrap: 300 replicates with parallel processing*
"""

    with open(os.path.join(out_dir, 'REPORT.md'), 'w') as f:
        f.write(report)

    print(f"\nResults saved to {out_dir}/")
    print(f"  - REPORT.md")
    print(f"  - results.json")
    print(f"  - fit_A.json, fit_B.json")
    print(f"  - fit_A.png, fit_B.png")

    # Log commands
    commands = """# Commands executed for LHCb Pentaquark Rank-1 Test

# 1. Data acquisition
curl -sL "https://www.hepdata.net/download/table/ins1728691/Table%201/csv" -o data/hepdata/89271_t1_spectrum_full.csv
curl -sL "https://www.hepdata.net/download/table/ins1728691/Table%202/csv" -o data/hepdata/89271_t2_spectrum_mKp_cut.csv
curl -sL "https://www.hepdata.net/download/table/ins1728691/Table%203/csv" -o data/hepdata/89271_t3_spectrum_weighted.csv

# 2. Run analysis
python3 src/pentaquark_rank1_test.py
"""

    with open(os.path.join(logs_dir, 'COMMANDS.txt'), 'w') as f:
        f.write(commands)

    return results


if __name__ == '__main__':
    results = main()
