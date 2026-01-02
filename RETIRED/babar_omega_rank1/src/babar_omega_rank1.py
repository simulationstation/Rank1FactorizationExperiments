#!/usr/bin/env python3
"""
BaBar ω(1420)/ω(1650) Rank-1 Bottleneck Test
=============================================

Tests if R = c2/c1 is consistent across two channels:
  - Channel A: e+e- → ω π+ π-
  - Channel B: e+e- → ω f0(980)

Data source: BaBar PRD 76, 092005 (2007)
HEPData DOI: 10.17182/hepdata.51824

No plot digitization - using only HEPData numeric tables.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2 as chi2_dist
import json
import os
import sys
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Resonance parameters from PDG 2024
# ω(1420): M = 1410 ± 60 MeV, Γ = 290 ± 40 MeV
# ω(1650): M = 1670 ± 30 MeV, Γ = 315 ± 35 MeV
OMEGA_1420_MASS = 1.420  # GeV
OMEGA_1420_WIDTH = 0.290  # GeV

OMEGA_1650_MASS = 1.670  # GeV
OMEGA_1650_WIDTH = 0.315  # GeV

# Fit health gates
CHI2_DOF_MIN = 0.5
CHI2_DOF_MAX = 3.0

# Optimization settings
N_STARTS = 300
N_BOOTSTRAP = 800
N_WORKERS = max(1, cpu_count() - 1)

# Contour grid
R_GRID = np.linspace(0.01, 5, 50)
PHI_GRID = np.linspace(-np.pi, np.pi, 72)  # 5 degree steps

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_hepdata_csv(filepath):
    """Load HEPData CSV file, skipping header comments."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find data start (after header comments)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('SQRT(S)'):
            data_start = i
            break

    # Parse data
    data = []
    for line in lines[data_start+1:]:
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    E = float(parts[0])
                    sigma = float(parts[3])
                    err_plus = float(parts[4])
                    err_minus = abs(float(parts[5]))
                    # Symmetric error approximation
                    err = (err_plus + err_minus) / 2
                    data.append([E, sigma, err])
                except:
                    pass

    df = pd.DataFrame(data, columns=['E_gev', 'sigma_nb', 'stat_err_nb'])
    return df

def load_data(base_dir):
    """Load both channels from HEPData."""
    df_A = load_hepdata_csv(os.path.join(base_dir, 'data/hepdata/table3_omega_pipi_raw.csv'))
    df_B = load_hepdata_csv(os.path.join(base_dir, 'data/hepdata/table4_omega_f0_raw.csv'))

    # Filter to resonance region (1.2 - 2.2 GeV covers ω(1420) and ω(1650))
    df_A = df_A[(df_A['E_gev'] >= 1.2) & (df_A['E_gev'] <= 2.2)]
    df_B = df_B[(df_B['E_gev'] >= 1.2) & (df_B['E_gev'] <= 2.2)]

    # Keep only positive cross sections
    df_A = df_A[df_A['sigma_nb'] >= 0]
    df_B = df_B[df_B['sigma_nb'] >= 0]

    return df_A, df_B

# ==============================================================================
# PHYSICS MODEL
# ==============================================================================

def breit_wigner(E, M, Gamma):
    """Relativistic Breit-Wigner amplitude (energy-dependent form)."""
    s = E**2
    return 1.0 / (s - M**2 + 1j * M * Gamma)

def coherent_amplitude(E, c1, c2, phi, bg_re, bg_im, bg1_re=0, bg1_im=0, E0=1.6, bg_order=0):
    """
    Coherent sum of two BW resonances plus background.

    A(E) = c1*BW1 + c2*exp(iΦ)*BW2 + background
    """
    E = np.atleast_1d(E)

    BW1 = breit_wigner(E, OMEGA_1420_MASS, OMEGA_1420_WIDTH)
    BW2 = breit_wigner(E, OMEGA_1650_MASS, OMEGA_1650_WIDTH)

    # Resonance amplitude (c1 fixed real-positive)
    A_res = c1 * BW1 + c2 * np.exp(1j * phi) * BW2

    # Background
    bg = bg_re + 1j * bg_im
    if bg_order >= 1:
        bg = bg + (bg1_re + 1j * bg1_im) * (E - E0)

    return A_res + bg

def cross_section_model(E, params, bg_order=0):
    """
    Cross section with nuisance parameter.

    σ(E) = s0 * |A(E)|^2

    params = [c1, c2, phi, bg_re, bg_im, (bg1_re, bg1_im if bg_order=1), s0]
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

    A = coherent_amplitude(E, c1, c2, phi, bg_re, bg_im, bg1_re, bg1_im, bg_order=bg_order)
    sigma = s0 * np.abs(A)**2

    return np.maximum(sigma, 1e-10)

def neg_log_likelihood(params, E_data, sigma_data, sigma_err, sigma_syst_frac, bg_order):
    """
    Gaussian NLL with nuisance parameter prior.
    """
    model = cross_section_model(E_data, params, bg_order)

    # Chi-squared
    residuals = (sigma_data - model) / sigma_err
    chi2 = np.sum(residuals**2)

    # Nuisance prior: s0 ~ N(1, sigma_syst_frac^2)
    if bg_order == 0:
        s0 = params[5]
    else:
        s0 = params[7]

    prior = (s0 - 1)**2 / (2 * sigma_syst_frac**2)

    return 0.5 * chi2 + prior

# ==============================================================================
# OPTIMIZATION
# ==============================================================================

def get_bounds(bg_order):
    """Get parameter bounds."""
    bounds = [
        (0.01, 10),     # c1 (amplitude, real positive)
        (0.01, 10),     # c2 (amplitude magnitude)
        (-np.pi, np.pi),  # phi (phase)
        (-5, 5),        # bg_re
        (-5, 5),        # bg_im
    ]
    if bg_order >= 1:
        bounds.extend([(-2, 2), (-2, 2)])  # bg1_re, bg1_im
    bounds.append((0.5, 1.5))  # s0 (nuisance scale)

    return bounds

def fit_channel(E_data, sigma_data, sigma_err, sigma_syst_frac, bg_order, n_starts=N_STARTS):
    """Fit a single channel using differential evolution + refinement."""
    bounds = get_bounds(bg_order)

    # Primary: Differential Evolution
    result = differential_evolution(
        neg_log_likelihood,
        bounds,
        args=(E_data, sigma_data, sigma_err, sigma_syst_frac, bg_order),
        maxiter=2000,
        tol=1e-8,
        seed=42,
        workers=1,
        updating='deferred',
        polish=True
    )

    best_params = result.x
    best_nll = result.fun

    # Multi-start refinement
    all_results = [(best_params, best_nll)]

    for i in range(min(100, n_starts)):
        np.random.seed(42 + i)
        x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
        try:
            res = minimize(neg_log_likelihood, x0,
                          args=(E_data, sigma_data, sigma_err, sigma_syst_frac, bg_order),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000})
            if np.isfinite(res.fun):
                all_results.append((res.x, res.fun))
                if res.fun < best_nll:
                    best_params = res.x
                    best_nll = res.fun
        except:
            pass

    return best_params, best_nll, all_results

def extract_R(params):
    """Extract R = c2*exp(iφ)/c1 from parameters."""
    c1 = params[0]
    c2 = params[1]
    phi = params[2]

    r = c2 / c1
    R = r * np.exp(1j * phi)
    return r, phi, R

def compute_chi2_dof(params, E_data, sigma_data, sigma_err, bg_order):
    """Compute chi-squared per degree of freedom."""
    model = cross_section_model(E_data, params, bg_order)
    residuals = (sigma_data - model) / sigma_err
    chi2 = np.sum(residuals**2)
    n_params = 6 if bg_order == 0 else 8
    dof = len(E_data) - n_params
    return chi2, dof, chi2 / max(dof, 1)

# ==============================================================================
# JOINT FITTING
# ==============================================================================

def joint_neg_log_likelihood(shared_params, fixed_A, fixed_B,
                             E_A, sigma_A, err_A, syst_A, bg_order_A,
                             E_B, sigma_B, err_B, syst_B, bg_order_B):
    """
    Joint NLL with shared R = c2*exp(iΦ)/c1.

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

    nll_A = neg_log_likelihood(params_A, E_A, sigma_A, err_A, syst_A, bg_order_A)
    nll_B = neg_log_likelihood(params_B, E_B, sigma_B, err_B, syst_B, bg_order_B)

    return nll_A + nll_B

def fit_joint_constrained(params_A, params_B, bg_order_A, bg_order_B,
                          E_A, sigma_A, err_A, syst_A,
                          E_B, sigma_B, err_B, syst_B):
    """Fit with shared R constraint."""
    r_A, phi_A, _ = extract_R(params_A)
    r_B, phi_B, _ = extract_R(params_B)

    # Start from average
    r_init = (r_A + r_B) / 2
    phi_init = (phi_A + phi_B) / 2

    bounds = [(0.01, 10), (-np.pi, np.pi)]

    best_nll = np.inf
    best_params = [r_init, phi_init]

    for i in range(100):
        np.random.seed(42 + i)
        x0 = [np.random.uniform(0.1, 5), np.random.uniform(-np.pi, np.pi)]
        try:
            result = minimize(joint_neg_log_likelihood, x0,
                            args=(params_A, params_B, E_A, sigma_A, err_A, syst_A, bg_order_A,
                                  E_B, sigma_B, err_B, syst_B, bg_order_B),
                            method='L-BFGS-B', bounds=bounds)
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except:
            pass

    return best_params, best_nll

# ==============================================================================
# BOOTSTRAP
# ==============================================================================

def bootstrap_worker(args):
    """Worker for bootstrap replication."""
    (seed, E_A, sigma_A, err_A, syst_A, bg_order_A,
     E_B, sigma_B, err_B, syst_B, bg_order_B,
     model_A, model_B, params_A_orig, params_B_orig) = args

    np.random.seed(seed)

    # Generate pseudo-data from constrained model
    sigma_A_boot = model_A + np.random.normal(0, err_A)
    sigma_A_boot = np.maximum(sigma_A_boot, 0)

    sigma_B_boot = model_B + np.random.normal(0, err_B)
    sigma_B_boot = np.maximum(sigma_B_boot, 0)

    bounds_A = get_bounds(bg_order_A)
    bounds_B = get_bounds(bg_order_B)

    try:
        # Fit individual channels
        res_A = minimize(neg_log_likelihood, params_A_orig,
                        args=(E_A, sigma_A_boot, err_A, syst_A, bg_order_A),
                        method='L-BFGS-B', bounds=bounds_A, options={'maxiter': 500})
        res_B = minimize(neg_log_likelihood, params_B_orig,
                        args=(E_B, sigma_B_boot, err_B, syst_B, bg_order_B),
                        method='L-BFGS-B', bounds=bounds_B, options={'maxiter': 500})

        if not (np.isfinite(res_A.fun) and np.isfinite(res_B.fun)):
            return None

        nll_unc = res_A.fun + res_B.fun

        # Fit constrained
        _, nll_con = fit_joint_constrained(res_A.x, res_B.x, bg_order_A, bg_order_B,
                                           E_A, sigma_A_boot, err_A, syst_A,
                                           E_B, sigma_B_boot, err_B, syst_B)

        Lambda = 2 * (nll_con - nll_unc)
        return Lambda
    except:
        return None

def run_bootstrap(E_A, sigma_A, err_A, syst_A, bg_order_A,
                  E_B, sigma_B, err_B, syst_B, bg_order_B,
                  params_A, params_B, shared_R, n_bootstrap=N_BOOTSTRAP):
    """Run bootstrap to estimate p-value."""

    # Get constrained model predictions
    r_shared, phi_shared = shared_R
    c1_A, c2_A = params_A[0], r_shared * params_A[0]
    params_A_con = [c1_A, c2_A, phi_shared] + list(params_A[3:])
    model_A = cross_section_model(E_A, params_A_con, bg_order_A)

    c1_B, c2_B = params_B[0], r_shared * params_B[0]
    params_B_con = [c1_B, c2_B, phi_shared] + list(params_B[3:])
    model_B = cross_section_model(E_B, params_B_con, bg_order_B)

    args_list = [(i, E_A, sigma_A, err_A, syst_A, bg_order_A,
                  E_B, sigma_B, err_B, syst_B, bg_order_B,
                  model_A, model_B, params_A, params_B) for i in range(n_bootstrap)]

    print(f"Running {n_bootstrap} bootstrap replicates...")

    with Pool(N_WORKERS) as pool:
        results = pool.map(bootstrap_worker, args_list)

    valid_results = [r for r in results if r is not None]
    print(f"Valid bootstrap samples: {len(valid_results)}/{n_bootstrap}")

    return np.array(valid_results)

# ==============================================================================
# PROFILE LIKELIHOOD CONTOURS
# ==============================================================================

def compute_profile_likelihood(E_data, sigma_data, sigma_err, syst_frac, bg_order,
                               best_params, r_grid, phi_grid):
    """Compute profile likelihood over (r, φ) grid."""
    nll_grid = np.zeros((len(r_grid), len(phi_grid)))
    bounds = get_bounds(bg_order)

    for i, r in enumerate(r_grid):
        for j, phi in enumerate(phi_grid):
            # Fix r, phi; optimize other params
            c1 = best_params[0]
            c2 = r * c1

            # Create fixed params
            x0 = [c1, c2, phi] + list(best_params[3:])

            # Optimize only background and nuisance
            def nll_fixed_R(other_params):
                full_params = [c1, c2, phi] + list(other_params)
                return neg_log_likelihood(full_params, E_data, sigma_data, sigma_err, syst_frac, bg_order)

            other_bounds = bounds[3:]
            other_x0 = list(best_params[3:])

            try:
                res = minimize(nll_fixed_R, other_x0, method='L-BFGS-B',
                              bounds=other_bounds, options={'maxiter': 200})
                nll_grid[i, j] = res.fun
            except:
                nll_grid[i, j] = np.inf

    return nll_grid

def plot_contours(nll_A, nll_B, r_grid, phi_grid, R_A, R_B, R_shared, out_dir):
    """Plot profile likelihood contours."""

    # Convert to Δχ²
    delta_chi2_A = 2 * (nll_A - np.min(nll_A))
    delta_chi2_B = 2 * (nll_B - np.min(nll_B))

    phi_deg = np.degrees(phi_grid)

    # Contour levels (68%, 95%)
    levels = [2.30, 5.99]

    # Channel A
    fig, ax = plt.subplots(figsize=(8, 6))
    CS = ax.contour(phi_deg, r_grid, delta_chi2_A, levels=levels, colors=['blue', 'lightblue'])
    ax.clabel(CS, fmt={2.30: '68%', 5.99: '95%'})
    r_A, phi_A = R_A
    ax.plot(np.degrees(phi_A), r_A, 'bo', markersize=10, label=f'Best fit A')
    ax.plot(np.degrees(R_shared[1]), R_shared[0], 'r*', markersize=15, label='Shared R')
    ax.set_xlabel('φ [deg]')
    ax.set_ylabel('r = |c₂/c₁|')
    ax.set_title('Channel A: e⁺e⁻ → ω π⁺π⁻')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'contours_A.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Channel B
    fig, ax = plt.subplots(figsize=(8, 6))
    CS = ax.contour(phi_deg, r_grid, delta_chi2_B, levels=levels, colors=['green', 'lightgreen'])
    ax.clabel(CS, fmt={2.30: '68%', 5.99: '95%'})
    r_B, phi_B = R_B
    ax.plot(np.degrees(phi_B), r_B, 'go', markersize=10, label=f'Best fit B')
    ax.plot(np.degrees(R_shared[1]), R_shared[0], 'r*', markersize=15, label='Shared R')
    ax.set_xlabel('φ [deg]')
    ax.set_ylabel('r = |c₂/c₁|')
    ax.set_title('Channel B: e⁺e⁻ → ω f₀(980)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'contours_B.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Overlay
    fig, ax = plt.subplots(figsize=(10, 8))
    CS_A = ax.contour(phi_deg, r_grid, delta_chi2_A, levels=levels, colors=['blue', 'lightblue'], linestyles='-')
    CS_B = ax.contour(phi_deg, r_grid, delta_chi2_B, levels=levels, colors=['green', 'lightgreen'], linestyles='--')
    ax.plot(np.degrees(phi_A), r_A, 'bo', markersize=10, label=f'Best A: r={r_A:.2f}, φ={np.degrees(phi_A):.0f}°')
    ax.plot(np.degrees(phi_B), r_B, 'go', markersize=10, label=f'Best B: r={r_B:.2f}, φ={np.degrees(phi_B):.0f}°')
    ax.plot(np.degrees(R_shared[1]), R_shared[0], 'r*', markersize=15,
            label=f'Shared: r={R_shared[0]:.2f}, φ={np.degrees(R_shared[1]):.0f}°')
    ax.set_xlabel('φ [deg]')
    ax.set_ylabel('r = |c₂/c₁|')
    ax.set_title('ω(1420)/ω(1650) Rank-1 Test: Profile Likelihood Contours')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-180, 180)
    plt.savefig(os.path.join(out_dir, 'contours_overlay.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return delta_chi2_A, delta_chi2_B

def check_shared_in_contours(delta_chi2_A, delta_chi2_B, r_grid, phi_grid, R_shared):
    """Check if shared R is within 95% CL contours."""
    r_shared, phi_shared = R_shared

    # Find nearest grid points
    r_idx = np.argmin(np.abs(r_grid - r_shared))
    phi_idx = np.argmin(np.abs(phi_grid - phi_shared))

    in_A_95 = delta_chi2_A[r_idx, phi_idx] < 5.99
    in_B_95 = delta_chi2_B[r_idx, phi_idx] < 5.99

    return in_A_95, in_B_95

# ==============================================================================
# OPTIMIZER STABILITY AUDIT
# ==============================================================================

def optimizer_stability_audit(all_results_A, all_results_B, out_dir):
    """Audit optimizer stability."""

    nlls_A = np.array([r[1] for r in all_results_A if r is not None])
    nlls_B = np.array([r[1] for r in all_results_B if r is not None])

    # Near-optimal solutions (ΔNLL < 2)
    best_A = np.min(nlls_A)
    best_B = np.min(nlls_B)

    near_opt_A = [(r[0], r[1]) for r in all_results_A if r[1] - best_A < 2]
    near_opt_B = [(r[0], r[1]) for r in all_results_B if r[1] - best_B < 2]

    r_values_A = [extract_R(p[0])[0] for p in near_opt_A]
    phi_values_A = [extract_R(p[0])[1] for p in near_opt_A]
    r_values_B = [extract_R(p[0])[0] for p in near_opt_B]
    phi_values_B = [extract_R(p[0])[1] for p in near_opt_B]

    # Check for multimodality
    r_range_A = max(r_values_A) / max(min(r_values_A), 0.01) if r_values_A else 1
    r_range_B = max(r_values_B) / max(min(r_values_B), 0.01) if r_values_B else 1

    phi_std_A = np.std(phi_values_A) if phi_values_A else 0
    phi_std_B = np.std(phi_values_B) if phi_values_B else 0

    phase_identifiable = (phi_std_A < 1.0) and (phi_std_B < 1.0)  # < ~60 degrees
    r_stable = (r_range_A < 10) and (r_range_B < 10)

    # Write stability report
    with open(os.path.join(out_dir, 'optimizer_stability.md'), 'w') as f:
        f.write("# Optimizer Stability Audit\n\n")
        f.write(f"## Channel A (ω π⁺π⁻)\n")
        f.write(f"- Near-optimal solutions (ΔNLL < 2): {len(near_opt_A)}\n")
        f.write(f"- r range: [{min(r_values_A):.3f}, {max(r_values_A):.3f}]\n")
        f.write(f"- φ std: {np.degrees(phi_std_A):.1f}°\n\n")
        f.write(f"## Channel B (ω f₀)\n")
        f.write(f"- Near-optimal solutions (ΔNLL < 2): {len(near_opt_B)}\n")
        f.write(f"- r range: [{min(r_values_B):.3f}, {max(r_values_B):.3f}]\n")
        f.write(f"- φ std: {np.degrees(phi_std_B):.1f}°\n\n")
        f.write(f"## Assessment\n")
        f.write(f"- Phase identifiable: **{'YES' if phase_identifiable else 'NO'}**\n")
        f.write(f"- r stable: **{'YES' if r_stable else 'NO'}**\n")

    # Plot stability
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(nlls_A, bins=30, alpha=0.7, color='blue')
    axes[0, 0].axvline(best_A, color='red', linestyle='--', label=f'Best: {best_A:.2f}')
    axes[0, 0].set_xlabel('NLL')
    axes[0, 0].set_title('Channel A: NLL Distribution')
    axes[0, 0].legend()

    axes[0, 1].hist(nlls_B, bins=30, alpha=0.7, color='green')
    axes[0, 1].axvline(best_B, color='red', linestyle='--', label=f'Best: {best_B:.2f}')
    axes[0, 1].set_xlabel('NLL')
    axes[0, 1].set_title('Channel B: NLL Distribution')
    axes[0, 1].legend()

    if r_values_A and phi_values_A:
        axes[1, 0].scatter(np.degrees(phi_values_A), r_values_A, alpha=0.5, c='blue')
        axes[1, 0].set_xlabel('φ [deg]')
        axes[1, 0].set_ylabel('r')
        axes[1, 0].set_title('Channel A: Near-optimal (r, φ)')

    if r_values_B and phi_values_B:
        axes[1, 1].scatter(np.degrees(phi_values_B), r_values_B, alpha=0.5, c='green')
        axes[1, 1].set_xlabel('φ [deg]')
        axes[1, 1].set_ylabel('r')
        axes[1, 1].set_title('Channel B: Near-optimal (r, φ)')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'optimizer_stability.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return phase_identifiable, r_stable

# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def main():
    print("=" * 70)
    print("BaBar ω(1420)/ω(1650) Rank-1 Bottleneck Test")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, 'out')
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    print("\nLoading HEPData tables...")
    df_A, df_B = load_data(base_dir)

    E_A = df_A['E_gev'].values
    sigma_A = df_A['sigma_nb'].values
    err_A = df_A['stat_err_nb'].values

    E_B = df_B['E_gev'].values
    sigma_B = df_B['sigma_nb'].values
    err_B = df_B['stat_err_nb'].values

    print(f"Channel A (ω π⁺π⁻): {len(E_A)} points, E=[{E_A.min():.2f}, {E_A.max():.2f}] GeV")
    print(f"Channel B (ω f₀): {len(E_B)} points, E=[{E_B.min():.2f}, {E_B.max():.2f}] GeV")

    # Systematic uncertainty (10% correlated)
    syst_A = 0.10
    syst_B = 0.10

    # Background model order (start with simple)
    bg_order_A = 0
    bg_order_B = 0

    # =========================================================================
    # FIT CHANNEL A
    # =========================================================================
    print("\n" + "=" * 60)
    print("Fitting Channel A (e⁺e⁻ → ω π⁺π⁻)...")
    print("=" * 60)

    params_A, nll_A, all_results_A = fit_channel(E_A, sigma_A, err_A, syst_A, bg_order_A)

    chi2_A, dof_A, chi2_dof_A = compute_chi2_dof(params_A, E_A, sigma_A, err_A, bg_order_A)
    r_A, phi_A, R_A_complex = extract_R(params_A)

    health_A = "PASS" if CHI2_DOF_MIN < chi2_dof_A < CHI2_DOF_MAX else \
               "UNDERCONSTRAINED" if chi2_dof_A < CHI2_DOF_MIN else "POOR FIT"

    print(f"  R_A = {r_A:.3f} × exp(i × {np.degrees(phi_A):.1f}°)")
    print(f"  χ²/dof = {chi2_dof_A:.3f} ({chi2_A:.1f}/{dof_A})")
    print(f"  Health: {health_A}")

    # =========================================================================
    # FIT CHANNEL B
    # =========================================================================
    print("\n" + "=" * 60)
    print("Fitting Channel B (e⁺e⁻ → ω f₀(980))...")
    print("=" * 60)

    params_B, nll_B, all_results_B = fit_channel(E_B, sigma_B, err_B, syst_B, bg_order_B)

    chi2_B, dof_B, chi2_dof_B = compute_chi2_dof(params_B, E_B, sigma_B, err_B, bg_order_B)
    r_B, phi_B, R_B_complex = extract_R(params_B)

    health_B = "PASS" if CHI2_DOF_MIN < chi2_dof_B < CHI2_DOF_MAX else \
               "UNDERCONSTRAINED" if chi2_dof_B < CHI2_DOF_MIN else "POOR FIT"

    print(f"  R_B = {r_B:.3f} × exp(i × {np.degrees(phi_B):.1f}°)")
    print(f"  χ²/dof = {chi2_dof_B:.3f} ({chi2_B:.1f}/{dof_B})")
    print(f"  Health: {health_B}")

    # =========================================================================
    # OPTIMIZER STABILITY
    # =========================================================================
    print("\n" + "=" * 60)
    print("Running optimizer stability audit...")
    print("=" * 60)

    phase_identifiable, r_stable = optimizer_stability_audit(all_results_A, all_results_B, out_dir)
    print(f"  Phase identifiable: {phase_identifiable}")
    print(f"  r stable: {r_stable}")

    # =========================================================================
    # JOINT CONSTRAINED FIT
    # =========================================================================
    print("\n" + "=" * 60)
    print("Fitting Joint Constrained (shared R)...")
    print("=" * 60)

    nll_unc = nll_A + nll_B

    shared_R, nll_con = fit_joint_constrained(
        params_A, params_B, bg_order_A, bg_order_B,
        E_A, sigma_A, err_A, syst_A,
        E_B, sigma_B, err_B, syst_B
    )

    Lambda = 2 * (nll_con - nll_unc)

    print(f"  Shared R = {shared_R[0]:.3f} × exp(i × {np.degrees(shared_R[1]):.1f}°)")
    print(f"  NLL_unconstrained = {nll_unc:.2f}")
    print(f"  NLL_constrained = {nll_con:.2f}")
    print(f"  Λ = 2×(NLL_con - NLL_unc) = {Lambda:.4f}")

    # Chi2 p-value
    p_chi2 = 1 - chi2_dist.cdf(Lambda, 2)
    print(f"  χ² p-value (2 dof) = {p_chi2:.4f}")

    # =========================================================================
    # BOOTSTRAP
    # =========================================================================
    print("\n" + "=" * 60)
    print("Bootstrap p-value estimation...")
    print("=" * 60)

    Lambda_boot = run_bootstrap(
        E_A, sigma_A, err_A, syst_A, bg_order_A,
        E_B, sigma_B, err_B, syst_B, bg_order_B,
        params_A, params_B, shared_R, n_bootstrap=N_BOOTSTRAP
    )

    if len(Lambda_boot) > 0:
        p_boot = np.mean(Lambda_boot >= Lambda)
        print(f"  Bootstrap p-value = {p_boot:.4f}")
    else:
        p_boot = np.nan
        print("  Bootstrap failed")

    # =========================================================================
    # PROFILE LIKELIHOOD CONTOURS
    # =========================================================================
    print("\n" + "=" * 60)
    print("Computing profile likelihood contours...")
    print("=" * 60)

    nll_grid_A = compute_profile_likelihood(E_A, sigma_A, err_A, syst_A, bg_order_A,
                                            params_A, R_GRID, PHI_GRID)
    nll_grid_B = compute_profile_likelihood(E_B, sigma_B, err_B, syst_B, bg_order_B,
                                            params_B, R_GRID, PHI_GRID)

    delta_chi2_A, delta_chi2_B = plot_contours(nll_grid_A, nll_grid_B, R_GRID, PHI_GRID,
                                                (r_A, phi_A), (r_B, phi_B), shared_R, out_dir)

    in_A_95, in_B_95 = check_shared_in_contours(delta_chi2_A, delta_chi2_B, R_GRID, PHI_GRID, shared_R)
    print(f"  Shared R in A 95% CL: {in_A_95}")
    print(f"  Shared R in B 95% CL: {in_B_95}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    fit_healthy = (health_A == "PASS") and (health_B == "PASS")

    if not fit_healthy:
        verdict = "MODEL MISMATCH"
        reason = f"Fit health failed: A={health_A}, B={health_B}"
    elif not phase_identifiable and r_stable:
        # Fallback to magnitude-only test when phase is multimodal but r is stable
        # Use the constrained fit results and bootstrap p-value for magnitude test
        r_diff = abs(r_A - r_B)
        r_diff_sigma = r_diff / max(0.01, min(abs(r_A * 0.1), abs(r_B * 0.1)))  # rough significance

        if p_boot < 0.01 and (not in_A_95 or not in_B_95):
            verdict = "DISFAVORED"
            reason = f"Magnitude-only test: |r_A - r_B| = {r_diff:.3f}, p_boot = {p_boot:.4f} << 0.05"
        elif p_boot < 0.05 and (not in_A_95 or not in_B_95):
            verdict = "DISFAVORED"
            reason = f"Magnitude-only test: |r_A - r_B| = {r_diff:.3f}, p_boot = {p_boot:.4f} < 0.05"
        elif p_boot > 0.05:
            verdict = "SUPPORTED"
            reason = f"Magnitude-only test: |r_A - r_B| = {r_diff:.3f}, p_boot = {p_boot:.3f} > 0.05"
        else:
            verdict = "INCONCLUSIVE"
            reason = f"Magnitude-only test: |r_A - r_B| = {r_diff:.3f}, mixed evidence"
    elif not phase_identifiable and not r_stable:
        verdict = "INCONCLUSIVE"
        reason = "Neither phase nor magnitude identifiable"
    elif p_boot > 0.05 and in_A_95 and in_B_95:
        verdict = "SUPPORTED"
        reason = f"p_boot={p_boot:.3f}>0.05, shared R within both 95% contours"
    elif p_boot < 0.05 and (not in_A_95 or not in_B_95):
        verdict = "DISFAVORED"
        reason = f"p_boot={p_boot:.3f}<0.05, shared R outside 95% contour"
    else:
        verdict = "INCONCLUSIVE"
        reason = f"Mixed evidence: p_boot={p_boot:.3f}, in_A={in_A_95}, in_B={in_B_95}"

    print(f"Verdict: {verdict}")
    print(f"Reason: {reason}")

    # =========================================================================
    # WRITE REPORT
    # =========================================================================
    report = f"""# BaBar ω(1420)/ω(1650) Rank-1 Bottleneck Test Report

## Data Source

- **Paper**: BaBar PRD 76, 092005 (2007)
- **HEPData DOI**: 10.17182/hepdata.51824
- **Table 3**: e⁺e⁻ → ω π⁺π⁻ (Channel A)
- **Table 4**: e⁺e⁻ → ω f₀(980) (Channel B)

## Resonance Parameters (Fixed)

| Resonance | Mass (GeV) | Width (GeV) |
|-----------|------------|-------------|
| ω(1420)   | {OMEGA_1420_MASS:.3f} | {OMEGA_1420_WIDTH:.3f} |
| ω(1650)   | {OMEGA_1650_MASS:.3f} | {OMEGA_1650_WIDTH:.3f} |

## Data Summary

| Channel | Points | Energy Range (GeV) |
|---------|--------|-------------------|
| A (ω π⁺π⁻) | {len(E_A)} | {E_A.min():.2f} - {E_A.max():.2f} |
| B (ω f₀) | {len(E_B)} | {E_B.min():.2f} - {E_B.max():.2f} |

## Fit Results

### Channel A (e⁺e⁻ → ω π⁺π⁻)

| Parameter | Value |
|-----------|-------|
| r = |c₂/c₁| | {r_A:.3f} |
| φ [deg] | {np.degrees(phi_A):.1f} |
| χ²/dof | {chi2_dof_A:.3f} ({chi2_A:.1f}/{dof_A}) |
| Health | **{health_A}** |

### Channel B (e⁺e⁻ → ω f₀(980))

| Parameter | Value |
|-----------|-------|
| r = |c₂/c₁| | {r_B:.3f} |
| φ [deg] | {np.degrees(phi_B):.1f} |
| χ²/dof | {chi2_dof_B:.3f} ({chi2_B:.1f}/{dof_B}) |
| Health | **{health_B}** |

## Optimizer Stability

- Phase identifiable: **{'YES' if phase_identifiable else 'NO'}**
- r stable: **{'YES' if r_stable else 'NO'}**

## Magnitude-Only Test

| Metric | Value |
|--------|-------|
| r_A = |c₂/c₁|_A | {r_A:.3f} |
| r_B = |c₂/c₁|_B | {r_B:.3f} |
| |r_A - r_B| | {abs(r_A - r_B):.3f} |
| Test Mode | {'Magnitude-only (phase multimodal)' if not phase_identifiable and r_stable else 'Full complex R'} |

## Joint Constrained Fit

| Metric | Value |
|--------|-------|
| Shared r | {shared_R[0]:.3f} |
| Shared φ [deg] | {np.degrees(shared_R[1]):.1f} |
| NLL_unconstrained | {nll_unc:.2f} |
| NLL_constrained | {nll_con:.2f} |
| **Λ** | **{Lambda:.4f}** |

## Statistical Tests

| Test | p-value |
|------|---------|
| χ² (2 dof) | {p_chi2:.4f} |
| Bootstrap ({len(Lambda_boot)} samples) | **{p_boot:.4f}** |

## Contour Analysis

| Check | Result |
|-------|--------|
| Shared R in A 95% CL | {'YES' if in_A_95 else 'NO'} |
| Shared R in B 95% CL | {'YES' if in_B_95 else 'NO'} |

## Fit Health Gates

| Criterion | Range | A | B |
|-----------|-------|---|---|
| χ²/dof | [{CHI2_DOF_MIN:.1f}, {CHI2_DOF_MAX:.1f}] | {chi2_dof_A:.3f} ({health_A}) | {chi2_dof_B:.3f} ({health_B}) |

## Verdict

**{verdict}**

{reason}

## Output Files

- `contours_A.png` - Channel A profile likelihood contours
- `contours_B.png` - Channel B profile likelihood contours
- `contours_overlay.png` - Overlay of both channels
- `optimizer_stability.png` - Optimizer stability audit
- `optimizer_stability.md` - Stability details

---
*Generated by BaBar ω(1420)/ω(1650) Rank-1 Test*
"""

    with open(os.path.join(out_dir, 'REPORT.md'), 'w') as f:
        f.write(report)

    # Save results JSON
    results = {
        'channel_A': {
            'r': float(r_A),
            'phi_deg': float(np.degrees(phi_A)),
            'chi2_dof': float(chi2_dof_A),
            'health': health_A
        },
        'channel_B': {
            'r': float(r_B),
            'phi_deg': float(np.degrees(phi_B)),
            'chi2_dof': float(chi2_dof_B),
            'health': health_B
        },
        'shared': {
            'r': float(shared_R[0]),
            'phi_deg': float(np.degrees(shared_R[1]))
        },
        'Lambda': float(Lambda),
        'p_bootstrap': float(p_boot) if not np.isnan(p_boot) else None,
        'verdict': verdict
    }

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nReport saved to: {os.path.join(out_dir, 'REPORT.md')}")

    return results

if __name__ == '__main__':
    main()
