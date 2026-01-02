#!/usr/bin/env python3
"""
Y-state Rank-1 Bottleneck Test
Test whether complex amplitude ratio R = c(Y2)/c(Y1) is invariant across:
  Channel A: e+e- -> pi+pi- J/psi (Belle)
  Channel B: e+e- -> pi+pi- psi(2S) (BaBar)

Uses coherent BW line-shape model with Y(4220) and Y(4360).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import os
import warnings
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Physical constants (PDG 2023 values for Y states)
# Y(4220) - formerly called Y(4260)
M_Y1 = 4.222  # GeV (central value)
G_Y1 = 0.044  # GeV (width)

# Y(4360)
M_Y2 = 4.368  # GeV
G_Y2 = 0.096  # GeV

# Reference energy for background
E0 = 4.3  # GeV

# Fit settings
N_RESTARTS = 150
N_BOOTSTRAP = 500
N_BOOTSTRAP_RESTARTS = 50
CHI2_DOF_GATE = 3.0


def load_channel_A(base_path):
    """Load Belle pi+pi- J/psi data."""
    path = os.path.join(base_path, 'belle_pipijpsi_raw.csv')
    df = pd.read_csv(path, comment='#')

    # Extract columns
    E = df['SQRT(S) [GEV]'].values
    sigma = df['SIG [PB]'].values
    stat_err = df['error +'].values  # Symmetric errors
    syst_pct = df['sys,overall uncertainty +'].str.replace('%', '').astype(float).values / 100
    syst_err = sigma * syst_pct

    # Total error in quadrature
    total_err = np.sqrt(stat_err**2 + syst_err**2)

    # Filter out invalid points
    valid = (sigma > 0) & (total_err > 0)

    return E[valid], sigma[valid], total_err[valid]


def load_channel_B(base_path):
    """Load BaBar pi+pi- psi(2S) data."""
    path = os.path.join(base_path, 'babar_pipipsi2s_raw.csv')
    df = pd.read_csv(path, comment='#')

    # Extract columns
    E = df['SQRT(S) [GEV]'].values
    sigma_raw = df['SIG [PB]'].values
    stat_err_plus = df['error +'].values

    # Handle dashes (missing data)
    valid = []
    E_clean = []
    sigma_clean = []
    err_clean = []

    for i in range(len(E)):
        try:
            s = float(sigma_raw[i])
            e = float(stat_err_plus[i])
            if s > 0 and e > 0:
                E_clean.append(E[i])
                sigma_clean.append(s)
                # Add 12.3% systematic in quadrature
                syst = s * 0.123
                err_clean.append(np.sqrt(e**2 + syst**2))
        except (ValueError, TypeError):
            continue

    return np.array(E_clean), np.array(sigma_clean), np.array(err_clean)


def bw_amplitude(E, M, Gamma):
    """
    Energy-dependent Breit-Wigner amplitude for e+e- production.
    BW(E) = 1 / (E^2 - M^2 + i*M*Gamma)
    """
    return 1.0 / (E**2 - M**2 + 1j * M * Gamma)


def model_cross_section(E, params, bg_order=1):
    """
    Coherent two-resonance model:
    sigma(E) = |A(E)|^2
    A(E) = c1*BW1 + c2*exp(i*phi)*BW2 + (a0 + a1*(E-E0))

    params: [c1, c2, phi, a0_re, a0_im, (a1_re, a1_im if bg_order>=1)]
    """
    c1 = params[0]  # Real, positive (absorbs overall phase)
    c2 = params[1]
    phi = params[2]

    # Background parameters
    a0 = params[3] + 1j * params[4]
    if bg_order >= 1 and len(params) > 5:
        a1 = params[5] + 1j * params[6]
    else:
        a1 = 0.0

    # Breit-Wigner amplitudes
    BW1 = bw_amplitude(E, M_Y1, G_Y1)
    BW2 = bw_amplitude(E, M_Y2, G_Y2)

    # Total amplitude
    A = c1 * BW1 + c2 * np.exp(1j * phi) * BW2 + a0 + a1 * (E - E0)

    return np.abs(A)**2


def gaussian_nll(params, E, sigma, err, bg_order=1):
    """Gaussian negative log-likelihood for cross section fits."""
    model = model_cross_section(E, params, bg_order)

    # Protect against numerical issues
    if np.any(model < 0) or np.any(~np.isfinite(model)):
        return 1e10

    residuals = (sigma - model) / err
    nll = 0.5 * np.sum(residuals**2)

    return nll


def fit_channel(E, sigma, err, bg_order=1, n_restarts=N_RESTARTS):
    """
    Fit single channel with multi-start optimization.
    Returns best parameters and NLL.
    """
    n_bg_params = 2 if bg_order == 0 else 4  # a0_re, a0_im, (a1_re, a1_im)
    n_params = 3 + n_bg_params  # c1, c2, phi + background

    best_nll = np.inf
    best_params = None

    # Bounds
    bounds = [
        (1e-3, 1e4),  # c1 > 0
        (1e-3, 1e4),  # c2 > 0
        (-np.pi, np.pi),  # phi
        (-100, 100),  # a0_re
        (-100, 100),  # a0_im
    ]
    if bg_order >= 1:
        bounds.extend([(-50, 50), (-50, 50)])  # a1_re, a1_im

    for _ in range(n_restarts):
        # Random initialization
        x0 = [
            np.random.uniform(10, 500),  # c1
            np.random.uniform(10, 500),  # c2
            np.random.uniform(-np.pi, np.pi),  # phi
            np.random.uniform(-10, 10),  # a0_re
            np.random.uniform(-10, 10),  # a0_im
        ]
        if bg_order >= 1:
            x0.extend([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])

        try:
            # L-BFGS-B
            res = minimize(gaussian_nll, x0, args=(E, sigma, err, bg_order),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-10})
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x.copy()

            # Powell (no bounds, but use result from L-BFGS-B as starting point)
            res2 = minimize(gaussian_nll, res.x, args=(E, sigma, err, bg_order),
                           method='Powell',
                           options={'maxiter': 2000, 'ftol': 1e-10})
            if res2.fun < best_nll:
                best_nll = res2.fun
                best_params = res2.x.copy()
        except:
            continue

    return best_params, best_nll


def extract_complex_ratio(params):
    """
    Extract R = c2*exp(i*phi) / c1 from fit parameters.
    Returns (r, Phi) where R = r * exp(i*Phi).
    """
    c1 = params[0]
    c2 = params[1]
    phi = params[2]

    r = c2 / c1
    Phi = phi  # Since c1 is real-positive

    return r, Phi


def joint_nll_constrained(params_joint, E_A, sigma_A, err_A, E_B, sigma_B, err_B, bg_order=1):
    """
    Joint NLL with shared complex ratio R = r * exp(i*Phi).

    params_joint: [r, Phi, c1_A, a0A_re, a0A_im, a1A_re, a1A_im,
                          c1_B, a0B_re, a0B_im, a1B_re, a1B_im]
    """
    r = params_joint[0]
    Phi = params_joint[1]

    # Channel A parameters
    c1_A = params_joint[2]
    c2_A = r * c1_A
    params_A = [c1_A, c2_A, Phi] + list(params_joint[3:7])

    # Channel B parameters
    c1_B = params_joint[7]
    c2_B = r * c1_B
    params_B = [c1_B, c2_B, Phi] + list(params_joint[8:12])

    nll_A = gaussian_nll(params_A, E_A, sigma_A, err_A, bg_order)
    nll_B = gaussian_nll(params_B, E_B, sigma_B, err_B, bg_order)

    return nll_A + nll_B


def fit_joint_constrained(E_A, sigma_A, err_A, E_B, sigma_B, err_B,
                          params_A_init, params_B_init, bg_order=1, n_restarts=N_RESTARTS):
    """
    Fit both channels jointly with shared R constraint.
    """
    best_nll = np.inf
    best_params = None

    # Get initial R estimate from unconstrained fits
    r_A, Phi_A = extract_complex_ratio(params_A_init)
    r_B, Phi_B = extract_complex_ratio(params_B_init)
    r_init = (r_A + r_B) / 2
    Phi_init = (Phi_A + Phi_B) / 2

    # Bounds for joint fit
    bounds = [
        (0.01, 10),     # r
        (-np.pi, np.pi),  # Phi
        (1e-3, 1e4),    # c1_A
        (-100, 100),    # a0A_re
        (-100, 100),    # a0A_im
        (-50, 50),      # a1A_re
        (-50, 50),      # a1A_im
        (1e-3, 1e4),    # c1_B
        (-100, 100),    # a0B_re
        (-100, 100),    # a0B_im
        (-50, 50),      # a1B_re
        (-50, 50),      # a1B_im
    ]

    for i in range(n_restarts):
        # Initialize from unconstrained fits with perturbation
        if i == 0:
            x0 = [r_init, Phi_init,
                  params_A_init[0], params_A_init[3], params_A_init[4],
                  params_A_init[5] if len(params_A_init) > 5 else 0,
                  params_A_init[6] if len(params_A_init) > 6 else 0,
                  params_B_init[0], params_B_init[3], params_B_init[4],
                  params_B_init[5] if len(params_B_init) > 5 else 0,
                  params_B_init[6] if len(params_B_init) > 6 else 0]
        else:
            # Random perturbation
            x0 = [
                np.random.uniform(0.1, 5),
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(10, 500),
                np.random.uniform(-10, 10),
                np.random.uniform(-10, 10),
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(10, 500),
                np.random.uniform(-10, 10),
                np.random.uniform(-10, 10),
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
            ]

        try:
            res = minimize(joint_nll_constrained, x0,
                          args=(E_A, sigma_A, err_A, E_B, sigma_B, err_B, bg_order),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 3000, 'ftol': 1e-10})
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x.copy()

            # Refine with Powell
            res2 = minimize(joint_nll_constrained, res.x,
                           args=(E_A, sigma_A, err_A, E_B, sigma_B, err_B, bg_order),
                           method='Powell',
                           options={'maxiter': 3000, 'ftol': 1e-10})
            if res2.fun < best_nll:
                best_nll = res2.fun
                best_params = res2.x.copy()
        except:
            continue

    return best_params, best_nll


def compute_chi2_dof(params, E, sigma, err, bg_order=1):
    """Compute chi^2/dof for fit quality assessment."""
    model = model_cross_section(E, params, bg_order)
    residuals = (sigma - model) / err
    chi2 = np.sum(residuals**2)
    n_params = len(params)
    dof = len(E) - n_params
    return chi2 / dof if dof > 0 else np.inf


def bootstrap_worker(args):
    """Worker function for parallel bootstrap."""
    (E_A, sigma_A, err_A, E_B, sigma_B, err_B,
     params_A, params_B, params_joint, bg_order, seed) = args

    np.random.seed(seed)

    # Generate pseudo-data from constrained model
    r_shared = params_joint[0]
    Phi_shared = params_joint[1]

    # Channel A model
    c1_A = params_joint[2]
    c2_A = r_shared * c1_A
    params_A_con = [c1_A, c2_A, Phi_shared] + list(params_joint[3:7])
    model_A = model_cross_section(E_A, params_A_con, bg_order)
    sigma_A_boot = np.random.normal(model_A, err_A)
    sigma_A_boot = np.maximum(sigma_A_boot, 0.1)  # Keep positive

    # Channel B model
    c1_B = params_joint[7]
    c2_B = r_shared * c1_B
    params_B_con = [c1_B, c2_B, Phi_shared] + list(params_joint[8:12])
    model_B = model_cross_section(E_B, params_B_con, bg_order)
    sigma_B_boot = np.random.normal(model_B, err_B)
    sigma_B_boot = np.maximum(sigma_B_boot, 0.1)

    # Refit unconstrained
    params_A_boot, nll_A_boot = fit_channel(E_A, sigma_A_boot, err_A, bg_order, N_BOOTSTRAP_RESTARTS)
    params_B_boot, nll_B_boot = fit_channel(E_B, sigma_B_boot, err_B, bg_order, N_BOOTSTRAP_RESTARTS)

    if params_A_boot is None or params_B_boot is None:
        return np.nan

    nll_unc_boot = nll_A_boot + nll_B_boot

    # Refit constrained
    params_joint_boot, nll_con_boot = fit_joint_constrained(
        E_A, sigma_A_boot, err_A, E_B, sigma_B_boot, err_B,
        params_A_boot, params_B_boot, bg_order, N_BOOTSTRAP_RESTARTS
    )

    if params_joint_boot is None:
        return np.nan

    Lambda_boot = 2 * (nll_con_boot - nll_unc_boot)
    return Lambda_boot


def run_bootstrap(E_A, sigma_A, err_A, E_B, sigma_B, err_B,
                  params_A, params_B, params_joint, bg_order=1, n_bootstrap=N_BOOTSTRAP):
    """Run parallel bootstrap to estimate p-value."""
    print(f"Running {n_bootstrap} bootstrap replicates...")

    n_workers = max(1, cpu_count() - 1)

    args_list = [
        (E_A, sigma_A, err_A, E_B, sigma_B, err_B,
         params_A, params_B, params_joint, bg_order, seed)
        for seed in range(n_bootstrap)
    ]

    with Pool(n_workers) as pool:
        Lambda_boots = pool.map(bootstrap_worker, args_list)

    return np.array([l for l in Lambda_boots if np.isfinite(l)])


def compute_profile_likelihood(E, sigma, err, bg_order=1,
                               r_range=(0.1, 3.0), phi_range=(-np.pi, np.pi),
                               r_steps=40, phi_steps=36):
    """
    Compute 2D profile likelihood over (r, Phi).
    At each grid point, optimize remaining parameters.
    """
    r_vals = np.linspace(r_range[0], r_range[1], r_steps)
    phi_vals = np.linspace(phi_range[0], phi_range[1], phi_steps)

    nll_grid = np.zeros((r_steps, phi_steps))

    # First get unconstrained optimum for initialization
    params_opt, nll_opt = fit_channel(E, sigma, err, bg_order, N_RESTARTS // 2)

    for i, r in enumerate(r_vals):
        for j, phi in enumerate(phi_vals):
            # Fix r and phi, optimize c1 and background
            def profile_nll(params_reduced):
                c1 = params_reduced[0]
                c2 = r * c1
                full_params = [c1, c2, phi] + list(params_reduced[1:])
                return gaussian_nll(full_params, E, sigma, err, bg_order)

            # Bounds for reduced parameters
            bounds_reduced = [
                (1e-3, 1e4),  # c1
                (-100, 100),  # a0_re
                (-100, 100),  # a0_im
                (-50, 50),    # a1_re
                (-50, 50),    # a1_im
            ]

            best_profile_nll = np.inf
            for _ in range(20):  # Fewer restarts for profile
                x0 = [
                    np.random.uniform(10, 500),
                    np.random.uniform(-10, 10),
                    np.random.uniform(-10, 10),
                    np.random.uniform(-5, 5),
                    np.random.uniform(-5, 5),
                ]
                try:
                    res = minimize(profile_nll, x0, method='L-BFGS-B',
                                  bounds=bounds_reduced, options={'maxiter': 1000})
                    if res.fun < best_profile_nll:
                        best_profile_nll = res.fun
                except:
                    continue

            nll_grid[i, j] = best_profile_nll

    # Convert to Delta chi^2
    delta_chi2 = 2 * (nll_grid - np.min(nll_grid))

    return r_vals, phi_vals, delta_chi2


def plot_contours(r_vals, phi_vals, delta_chi2_A, delta_chi2_B,
                  r_A, phi_A, r_B, phi_B, r_shared, phi_shared, out_path):
    """Plot profile likelihood contours for both channels."""

    # 68% and 95% CL for 2 parameters
    levels = [2.30, 5.99]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Channel A
    ax = axes[0]
    R, PHI = np.meshgrid(r_vals, phi_vals, indexing='ij')
    cs = ax.contour(R, PHI * 180/np.pi, delta_chi2_A, levels=levels, colors=['blue', 'lightblue'])
    ax.plot(r_A, phi_A * 180/np.pi, 'b*', markersize=15, label='Channel A optimum')
    ax.plot(r_shared, phi_shared * 180/np.pi, 'r^', markersize=12, label='Shared R')
    ax.set_xlabel('r = |c2/c1|', fontsize=12)
    ax.set_ylabel(r'$\Phi$ [degrees]', fontsize=12)
    ax.set_title(r'Channel A: $\pi^+\pi^- J/\psi$ (Belle)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Channel B
    ax = axes[1]
    cs = ax.contour(R, PHI * 180/np.pi, delta_chi2_B, levels=levels, colors=['green', 'lightgreen'])
    ax.plot(r_B, phi_B * 180/np.pi, 'g*', markersize=15, label='Channel B optimum')
    ax.plot(r_shared, phi_shared * 180/np.pi, 'r^', markersize=12, label='Shared R')
    ax.set_xlabel('r = |c2/c1|', fontsize=12)
    ax.set_ylabel(r'$\Phi$ [degrees]', fontsize=12)
    ax.set_title(r'Channel B: $\pi^+\pi^- \psi(2S)$ (BaBar)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Overlay
    ax = axes[2]
    ax.contour(R, PHI * 180/np.pi, delta_chi2_A, levels=[5.99], colors=['blue'],
               linestyles=['-'], linewidths=2)
    ax.contour(R, PHI * 180/np.pi, delta_chi2_B, levels=[5.99], colors=['green'],
               linestyles=['-'], linewidths=2)
    ax.plot(r_A, phi_A * 180/np.pi, 'b*', markersize=15, label=f'A: r={r_A:.2f}, $\Phi$={phi_A*180/np.pi:.1f}')
    ax.plot(r_B, phi_B * 180/np.pi, 'g*', markersize=15, label=f'B: r={r_B:.2f}, $\Phi$={phi_B*180/np.pi:.1f}')
    ax.plot(r_shared, phi_shared * 180/np.pi, 'r^', markersize=12,
            label=f'Shared: r={r_shared:.2f}, $\Phi$={phi_shared*180/np.pi:.1f}')
    ax.set_xlabel('r = |c2/c1|', fontsize=12)
    ax.set_ylabel(r'$\Phi$ [degrees]', fontsize=12)
    ax.set_title('95% CL Contours Overlay', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved contours to {out_path}")


def main():
    print("=" * 70)
    print("Y-state Rank-1 Bottleneck Test")
    print("Testing R = c(Y4360)/c(Y4220) invariance across decay channels")
    print("=" * 70)

    # Paths
    base_path = '/home/primary/DarkBItParticleColiderPredictions/y_rank1_test'
    data_path = os.path.join(base_path, 'data/hepdata')
    out_path = os.path.join(base_path, 'out')

    # Load data
    print("\nLoading data...")
    E_A, sigma_A, err_A = load_channel_A(data_path)
    E_B, sigma_B, err_B = load_channel_B(data_path)

    print(f"Channel A (Belle pi+pi- J/psi): {len(E_A)} points, E=[{E_A.min():.2f}, {E_A.max():.2f}] GeV")
    print(f"Channel B (BaBar pi+pi- psi2S): {len(E_B)} points, E=[{E_B.min():.2f}, {E_B.max():.2f}] GeV")

    # Fit parameters
    bg_order = 1  # Linear complex background

    # Fit Channel A
    print("\n" + "=" * 60)
    print("Fitting Channel A (Belle pi+pi- J/psi)...")
    print("=" * 60)
    params_A, nll_A = fit_channel(E_A, sigma_A, err_A, bg_order, N_RESTARTS)
    r_A, phi_A = extract_complex_ratio(params_A)
    chi2_dof_A = compute_chi2_dof(params_A, E_A, sigma_A, err_A, bg_order)

    print(f"Channel A results:")
    print(f"  c1 = {params_A[0]:.2f}")
    print(f"  c2 = {params_A[1]:.2f}")
    print(f"  phi = {params_A[2]*180/np.pi:.1f} deg")
    print(f"  R_A = {r_A:.3f} * exp(i * {phi_A*180/np.pi:.1f} deg)")
    print(f"  chi2/dof = {chi2_dof_A:.3f}")
    print(f"  NLL = {nll_A:.2f}")

    # Fit Channel B
    print("\n" + "=" * 60)
    print("Fitting Channel B (BaBar pi+pi- psi2S)...")
    print("=" * 60)
    params_B, nll_B = fit_channel(E_B, sigma_B, err_B, bg_order, N_RESTARTS)
    r_B, phi_B = extract_complex_ratio(params_B)
    chi2_dof_B = compute_chi2_dof(params_B, E_B, sigma_B, err_B, bg_order)

    print(f"Channel B results:")
    print(f"  c1 = {params_B[0]:.2f}")
    print(f"  c2 = {params_B[1]:.2f}")
    print(f"  phi = {params_B[2]*180/np.pi:.1f} deg")
    print(f"  R_B = {r_B:.3f} * exp(i * {phi_B*180/np.pi:.1f} deg)")
    print(f"  chi2/dof = {chi2_dof_B:.3f}")
    print(f"  NLL = {nll_B:.2f}")

    # Check fit health gates
    gate_A = chi2_dof_A < CHI2_DOF_GATE
    gate_B = chi2_dof_B < CHI2_DOF_GATE

    print(f"\nFit health gates (chi2/dof < {CHI2_DOF_GATE}):")
    print(f"  Channel A: {'PASS' if gate_A else 'FAIL'} ({chi2_dof_A:.2f})")
    print(f"  Channel B: {'PASS' if gate_B else 'FAIL'} ({chi2_dof_B:.2f})")

    # Unconstrained total NLL
    nll_unc = nll_A + nll_B

    # Fit joint constrained
    print("\n" + "=" * 60)
    print("Fitting Joint Constrained (shared R)...")
    print("=" * 60)
    params_joint, nll_con = fit_joint_constrained(
        E_A, sigma_A, err_A, E_B, sigma_B, err_B,
        params_A, params_B, bg_order, N_RESTARTS
    )

    r_shared = params_joint[0]
    phi_shared = params_joint[1]

    print(f"Shared R = {r_shared:.3f} * exp(i * {phi_shared*180/np.pi:.1f} deg)")
    print(f"NLL_constrained = {nll_con:.2f}")
    print(f"NLL_unconstrained = {nll_unc:.2f}")

    # Likelihood ratio
    Lambda = 2 * (nll_con - nll_unc)
    print(f"\nLambda = 2*(NLL_con - NLL_unc) = {Lambda:.4f}")

    if Lambda < 0:
        print("WARNING: Lambda < 0 indicates optimizer failure. Retrying with more restarts...")
        # Retry with more restarts
        params_joint, nll_con = fit_joint_constrained(
            E_A, sigma_A, err_A, E_B, sigma_B, err_B,
            params_A, params_B, bg_order, N_RESTARTS * 2
        )
        r_shared = params_joint[0]
        phi_shared = params_joint[1]
        Lambda = 2 * (nll_con - nll_unc)
        print(f"Retry: Lambda = {Lambda:.4f}")

    # Bootstrap p-value
    print("\n" + "=" * 60)
    print("Bootstrap p-value estimation...")
    print("=" * 60)

    Lambda_boots = run_bootstrap(E_A, sigma_A, err_A, E_B, sigma_B, err_B,
                                  params_A, params_B, params_joint, bg_order, N_BOOTSTRAP)

    p_value = np.mean(Lambda_boots >= Lambda) if len(Lambda_boots) > 0 else np.nan
    print(f"Valid bootstrap samples: {len(Lambda_boots)}/{N_BOOTSTRAP}")
    print(f"Bootstrap p-value = {p_value:.4f}")

    # Profile likelihood contours
    print("\n" + "=" * 60)
    print("Computing profile likelihood contours...")
    print("=" * 60)

    r_vals_A, phi_vals_A, delta_chi2_A = compute_profile_likelihood(
        E_A, sigma_A, err_A, bg_order, r_range=(0.1, 3.0), phi_range=(-np.pi, np.pi)
    )

    r_vals_B, phi_vals_B, delta_chi2_B = compute_profile_likelihood(
        E_B, sigma_B, err_B, bg_order, r_range=(0.1, 3.0), phi_range=(-np.pi, np.pi)
    )

    # Check if shared point is within 95% CL of both channels
    # Find nearest grid point
    i_A = np.argmin(np.abs(r_vals_A - r_shared))
    j_A = np.argmin(np.abs(phi_vals_A - phi_shared))
    i_B = np.argmin(np.abs(r_vals_B - r_shared))
    j_B = np.argmin(np.abs(phi_vals_B - phi_shared))

    in_95_A = delta_chi2_A[i_A, j_A] < 5.99
    in_95_B = delta_chi2_B[i_B, j_B] < 5.99

    print(f"Shared R within 95% CL of Channel A: {'YES' if in_95_A else 'NO'}")
    print(f"Shared R within 95% CL of Channel B: {'YES' if in_95_B else 'NO'}")

    # Plot contours
    contour_path = os.path.join(out_path, 'contours_overlay.png')
    plot_contours(r_vals_A, phi_vals_A, delta_chi2_A, delta_chi2_B,
                  r_A, phi_A, r_B, phi_B, r_shared, phi_shared, contour_path)

    # Determine verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    if not gate_A or not gate_B:
        verdict = "MODEL_MISMATCH"
        reason = "Fit health gate failed"
    elif Lambda < 0:
        verdict = "OPTIMIZER_FAILURE"
        reason = "Lambda < 0 persists"
    elif p_value > 0.05 and in_95_A and in_95_B:
        verdict = "SUPPORTED"
        reason = f"p={p_value:.3f} > 0.05 and shared R within both 95% regions"
    elif p_value < 0.05:
        verdict = "DISFAVORED"
        reason = f"p={p_value:.3f} < 0.05"
    else:
        verdict = "INCONCLUSIVE"
        reason = "Mixed results"

    print(f"Verdict: {verdict}")
    print(f"Reason: {reason}")

    # Write REPORT.md
    report_path = os.path.join(out_path, 'REPORT.md')
    with open(report_path, 'w') as f:
        f.write("# Y-state Rank-1 Bottleneck Test Report\n\n")
        f.write("## Data Sources\n\n")
        f.write("| Channel | Reaction | Experiment | HEPData Record | DOI |\n")
        f.write("|---------|----------|------------|----------------|-----|\n")
        f.write(f"| A | e+e- -> pi+pi- J/psi | Belle | ins1225975 | 10.17182/hepdata.61431.v1/t1 |\n")
        f.write(f"| B | e+e- -> pi+pi- psi(2S) | BaBar | ins729388 | 10.17182/hepdata.19344.v1/t1 |\n\n")

        f.write("## Resonance Parameters (Fixed)\n\n")
        f.write(f"- Y(4220): M = {M_Y1} GeV, Gamma = {G_Y1} GeV\n")
        f.write(f"- Y(4360): M = {M_Y2} GeV, Gamma = {G_Y2} GeV\n\n")

        f.write("## Model\n\n")
        f.write("Coherent amplitude: A(E) = c1*BW(Y1) + c2*exp(i*phi)*BW(Y2) + background\n\n")
        f.write(f"Background: Complex constant + linear term (bg_order = {bg_order})\n\n")

        f.write("## Unconstrained Fit Results\n\n")
        f.write("| Channel | r = |c2/c1| | Phi [deg] | chi2/dof | NLL |\n")
        f.write("|---------|------------|-----------|----------|-----|\n")
        f.write(f"| A | {r_A:.3f} | {phi_A*180/np.pi:.1f} | {chi2_dof_A:.3f} | {nll_A:.2f} |\n")
        f.write(f"| B | {r_B:.3f} | {phi_B*180/np.pi:.1f} | {chi2_dof_B:.3f} | {nll_B:.2f} |\n\n")

        f.write("## Constrained Fit (Shared R)\n\n")
        f.write(f"- R_shared = {r_shared:.3f} * exp(i * {phi_shared*180/np.pi:.1f} deg)\n")
        f.write(f"- NLL_constrained = {nll_con:.2f}\n")
        f.write(f"- NLL_unconstrained = {nll_unc:.2f}\n\n")

        f.write("## Statistical Test\n\n")
        f.write(f"- Lambda = 2*(NLL_con - NLL_unc) = {Lambda:.4f}\n")
        f.write(f"- Bootstrap p-value = {p_value:.4f} ({len(Lambda_boots)} valid replicates)\n\n")

        f.write("## Fit Health\n\n")
        f.write(f"- Channel A chi2/dof = {chi2_dof_A:.3f} ({'PASS' if gate_A else 'FAIL'})\n")
        f.write(f"- Channel B chi2/dof = {chi2_dof_B:.3f} ({'PASS' if gate_B else 'FAIL'})\n\n")

        f.write("## Profile Likelihood\n\n")
        f.write(f"- Shared R within Channel A 95% CL: {'YES' if in_95_A else 'NO'}\n")
        f.write(f"- Shared R within Channel B 95% CL: {'YES' if in_95_B else 'NO'}\n\n")

        f.write("## Verdict\n\n")
        f.write(f"**{verdict}**\n\n")
        f.write(f"Reason: {reason}\n\n")

        f.write("## Plots\n\n")
        f.write("![Contours](contours_overlay.png)\n")

    print(f"\nReport saved to: {report_path}")
    print(f"Contours saved to: {contour_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Channel A (Belle pi+pi- J/psi): R_A = {r_A:.3f} * exp(i * {phi_A*180/np.pi:.1f} deg)")
    print(f"Channel B (BaBar pi+pi- psi2S): R_B = {r_B:.3f} * exp(i * {phi_B*180/np.pi:.1f} deg)")
    print(f"Shared: R = {r_shared:.3f} * exp(i * {phi_shared*180/np.pi:.1f} deg)")
    print(f"Lambda = {Lambda:.4f}")
    print(f"Bootstrap p-value = {p_value:.4f}")
    print(f"Verdict: {verdict}")


if __name__ == '__main__':
    main()
