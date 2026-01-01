#!/usr/bin/env python3
"""
BESIII Y-Sector Rank-1 Test v3 - Fixed
Key fixes:
1. Data-driven resonance positions (not literature priors)
2. Constrained fit first, then seed unconstrained
3. Proper s = E² in BW formula
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

BASE = "/home/primary/DarkBItParticleColiderPredictions/besiii_y_rank1_v2"
OUT = f"{BASE}/out"

@dataclass
class DataPoint:
    E: float
    sigma: float
    stat_err: float

# Load data directly
DATA_A = [  # ππJ/ψ in [4.01, 4.60]
    DataPoint(4.0076, 40.3, 5.9), DataPoint(4.0330, 36.4, 10.5), DataPoint(4.0854, 71.9, 3.7),
    DataPoint(4.1285, 104.3, 4.7), DataPoint(4.1574, 127.9, 4.9), DataPoint(4.1780, 110.7, 3.9),
    DataPoint(4.1889, 104.6, 3.5), DataPoint(4.1989, 100.9, 3.0), DataPoint(4.2092, 103.9, 2.5),
    DataPoint(4.2187, 109.9, 2.8), DataPoint(4.2263, 116.9, 2.1), DataPoint(4.2358, 88.6, 2.4),
    DataPoint(4.2439, 79.8, 2.5), DataPoint(4.2580, 75.9, 1.5), DataPoint(4.2668, 72.7, 2.2),
    DataPoint(4.2780, 65.9, 2.1), DataPoint(4.2879, 69.9, 2.0), DataPoint(4.3079, 62.9, 1.7),
    DataPoint(4.3121, 60.5, 2.3), DataPoint(4.3374, 56.7, 2.0), DataPoint(4.3583, 52.3, 1.6),
    DataPoint(4.3774, 51.7, 1.5), DataPoint(4.3964, 48.3, 1.0), DataPoint(4.4156, 47.9, 1.3),
    DataPoint(4.4367, 46.8, 1.4), DataPoint(4.4671, 44.1, 2.6), DataPoint(4.5271, 32.3, 2.2),
    DataPoint(4.5745, 26.2, 1.2),
]

DATA_B = [  # ππh_c in [4.01, 4.60]
    DataPoint(4.0854, 11.0, 3.5), DataPoint(4.1285, 15.2, 3.3), DataPoint(4.1574, 19.4, 3.1),
    DataPoint(4.1780, 19.0, 2.7), DataPoint(4.1889, 20.5, 2.4), DataPoint(4.1989, 22.6, 2.2),
    DataPoint(4.2092, 25.8, 1.8), DataPoint(4.2187, 27.8, 2.0), DataPoint(4.2263, 34.8, 1.6),
    DataPoint(4.2358, 25.5, 1.7), DataPoint(4.2439, 23.3, 1.8), DataPoint(4.2580, 19.9, 1.2),
    DataPoint(4.2668, 17.6, 1.5), DataPoint(4.2780, 14.0, 1.4), DataPoint(4.2879, 13.7, 1.3),
    DataPoint(4.3079, 16.1, 1.2), DataPoint(4.3121, 15.9, 1.5), DataPoint(4.3374, 19.5, 1.4),
    DataPoint(4.3583, 20.7, 1.2), DataPoint(4.3774, 21.5, 1.1), DataPoint(4.3964, 19.6, 0.8),
    DataPoint(4.4156, 17.3, 0.9), DataPoint(4.4367, 14.2, 1.0), DataPoint(4.4671, 10.2, 1.6),
    DataPoint(4.5271, 5.4, 1.3),
]

def breit_wigner(s, M, Gamma):
    """BW amplitude with s = E²."""
    M2 = M**2
    return M * Gamma / (M2 - s + 1j * M * Gamma)

def model_A(E, M1, G1, M2, G2, c1, r, phi, scale):
    """2-resonance model for Channel A (simplified for shared subspace test)."""
    s = E**2
    c2 = r * np.exp(1j * phi) * c1
    A = c1 * breit_wigner(s, M1, G1) + c2 * breit_wigner(s, M2, G2)
    return scale * np.abs(A)**2

def model_B(E, M1, G1, M2, G2, c1, r, phi, scale):
    """2-resonance model for Channel B."""
    s = E**2
    c2 = r * np.exp(1j * phi) * c1
    A = c1 * breit_wigner(s, M1, G1) + c2 * breit_wigner(s, M2, G2)
    return scale * np.abs(A)**2

def chi2(data, model_vals):
    return sum(((p.sigma - model_vals[i]) / p.stat_err)**2 for i, p in enumerate(data))

def nll(data, model_vals):
    return 0.5 * chi2(data, model_vals)

def fit_unconstrained(data_A, data_B, n_starts=300):
    """Fit both channels with independent R values."""
    E_A = np.array([p.E for p in data_A])
    E_B = np.array([p.E for p in data_B])

    def objective(params):
        M1, G1, M2, G2, c1_A, r_A, phi_A, scale_A, c1_B, r_B, phi_B, scale_B = params

        if G1 <= 0 or G2 <= 0 or c1_A <= 0 or c1_B <= 0 or r_A < 0 or r_B < 0:
            return 1e10
        if scale_A <= 0 or scale_B <= 0:
            return 1e10

        try:
            sigma_A = model_A(E_A, M1, G1, M2, G2, c1_A, r_A, phi_A, scale_A)
            sigma_B = model_B(E_B, M1, G1, M2, G2, c1_B, r_B, phi_B, scale_B)

            if np.any(~np.isfinite(sigma_A)) or np.any(~np.isfinite(sigma_B)):
                return 1e10
            if np.any(sigma_A < 0) or np.any(sigma_B < 0):
                return 1e10

            return nll(data_A, sigma_A) + nll(data_B, sigma_B)
        except:
            return 1e10

    # Use data to estimate initial values
    # ππJ/ψ peak at ~4.16, ππh_c peaks at ~4.22 and ~4.38
    # Shared Y1 ~ 4.20 (compromise), Y2 ~ 4.35

    bounds = [
        (4.15, 4.25),   # M1
        (0.02, 0.15),   # G1
        (4.28, 4.45),   # M2
        (0.05, 0.30),   # G2
        (0.1, 50),      # c1_A
        (0.1, 10),      # r_A
        (-np.pi, np.pi), # phi_A
        (0.1, 100),     # scale_A
        (0.1, 50),      # c1_B
        (0.1, 10),      # r_B
        (-np.pi, np.pi), # phi_B
        (0.1, 100),     # scale_B
    ]

    best_nll = np.inf
    best_params = None

    for i in range(n_starts):
        x0 = [
            np.random.uniform(4.18, 4.24),
            np.random.uniform(0.04, 0.10),
            np.random.uniform(4.32, 4.42),
            np.random.uniform(0.08, 0.20),
            np.random.uniform(1, 20),
            np.random.uniform(0.3, 3),
            np.random.uniform(-2, 2),
            np.random.uniform(1, 50),
            np.random.uniform(0.5, 10),
            np.random.uniform(0.3, 3),
            np.random.uniform(-2, 2),
            np.random.uniform(1, 30),
        ]

        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except:
            pass

    if best_params is None:
        return np.inf, None

    return best_nll, best_params

def fit_constrained(data_A, data_B, n_starts=300):
    """Fit both channels with shared R value."""
    E_A = np.array([p.E for p in data_A])
    E_B = np.array([p.E for p in data_B])

    def objective(params):
        M1, G1, M2, G2, r_shared, phi_shared, c1_A, scale_A, c1_B, scale_B = params

        if G1 <= 0 or G2 <= 0 or c1_A <= 0 or c1_B <= 0 or r_shared < 0:
            return 1e10
        if scale_A <= 0 or scale_B <= 0:
            return 1e10

        try:
            sigma_A = model_A(E_A, M1, G1, M2, G2, c1_A, r_shared, phi_shared, scale_A)
            sigma_B = model_B(E_B, M1, G1, M2, G2, c1_B, r_shared, phi_shared, scale_B)

            if np.any(~np.isfinite(sigma_A)) or np.any(~np.isfinite(sigma_B)):
                return 1e10
            if np.any(sigma_A < 0) or np.any(sigma_B < 0):
                return 1e10

            return nll(data_A, sigma_A) + nll(data_B, sigma_B)
        except:
            return 1e10

    bounds = [
        (4.15, 4.25),   # M1
        (0.02, 0.15),   # G1
        (4.28, 4.45),   # M2
        (0.05, 0.30),   # G2
        (0.1, 10),      # r_shared
        (-np.pi, np.pi), # phi_shared
        (0.1, 50),      # c1_A
        (0.1, 100),     # scale_A
        (0.1, 50),      # c1_B
        (0.1, 100),     # scale_B
    ]

    best_nll = np.inf
    best_params = None

    for i in range(n_starts):
        x0 = [
            np.random.uniform(4.18, 4.24),
            np.random.uniform(0.04, 0.10),
            np.random.uniform(4.32, 4.42),
            np.random.uniform(0.08, 0.20),
            np.random.uniform(0.3, 3),
            np.random.uniform(-2, 2),
            np.random.uniform(1, 20),
            np.random.uniform(1, 50),
            np.random.uniform(0.5, 10),
            np.random.uniform(1, 30),
        ]

        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except:
            pass

    if best_params is None:
        return np.inf, None

    return best_nll, best_params

def fit_unconstrained_seeded(data_A, data_B, con_params, n_starts=200):
    """Fit unconstrained, seeded from constrained solution."""
    E_A = np.array([p.E for p in data_A])
    E_B = np.array([p.E for p in data_B])

    M1, G1, M2, G2, r_shared, phi_shared, c1_A, scale_A, c1_B, scale_B = con_params

    def objective(params):
        M1, G1, M2, G2, c1_A, r_A, phi_A, scale_A, c1_B, r_B, phi_B, scale_B = params

        if G1 <= 0 or G2 <= 0 or c1_A <= 0 or c1_B <= 0 or r_A < 0 or r_B < 0:
            return 1e10
        if scale_A <= 0 or scale_B <= 0:
            return 1e10

        try:
            sigma_A = model_A(E_A, M1, G1, M2, G2, c1_A, r_A, phi_A, scale_A)
            sigma_B = model_B(E_B, M1, G1, M2, G2, c1_B, r_B, phi_B, scale_B)

            if np.any(~np.isfinite(sigma_A)) or np.any(~np.isfinite(sigma_B)):
                return 1e10
            if np.any(sigma_A < 0) or np.any(sigma_B < 0):
                return 1e10

            return nll(data_A, sigma_A) + nll(data_B, sigma_B)
        except:
            return 1e10

    bounds = [
        (4.15, 4.25), (0.02, 0.15), (4.28, 4.45), (0.05, 0.30),
        (0.1, 50), (0.1, 10), (-np.pi, np.pi), (0.1, 100),
        (0.1, 50), (0.1, 10), (-np.pi, np.pi), (0.1, 100),
    ]

    # Start from constrained solution (convert to unconstrained params)
    x0_seed = [M1, G1, M2, G2, c1_A, r_shared, phi_shared, scale_A,
               c1_B, r_shared, phi_shared, scale_B]

    best_nll = np.inf
    best_params = None

    # First try from seeded point
    try:
        res = minimize(objective, x0_seed, method='L-BFGS-B', bounds=bounds)
        if res.fun < best_nll:
            best_nll = res.fun
            best_params = res.x
    except:
        pass

    # Then try random starts
    for i in range(n_starts):
        # Perturb around seed
        x0 = [
            x0_seed[0] + np.random.uniform(-0.02, 0.02),
            x0_seed[1] + np.random.uniform(-0.02, 0.02),
            x0_seed[2] + np.random.uniform(-0.03, 0.03),
            x0_seed[3] + np.random.uniform(-0.05, 0.05),
            x0_seed[4] * np.random.uniform(0.5, 2),
            x0_seed[5] * np.random.uniform(0.5, 2),
            x0_seed[6] + np.random.uniform(-1, 1),
            x0_seed[7] * np.random.uniform(0.5, 2),
            x0_seed[8] * np.random.uniform(0.5, 2),
            x0_seed[9] * np.random.uniform(0.5, 2),
            x0_seed[10] + np.random.uniform(-1, 1),
            x0_seed[11] * np.random.uniform(0.5, 2),
        ]
        x0 = [max(bounds[i][0], min(bounds[i][1], x0[i])) for i in range(len(x0))]

        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except:
            pass

    return best_nll, best_params

# Bootstrap worker
_boot_data = {}

def _init_boot(data_A, data_B, con_params):
    global _boot_data
    _boot_data = {'data_A': data_A, 'data_B': data_B, 'con_params': con_params}

def _boot_worker(seed):
    np.random.seed(seed)
    d = _boot_data
    E_A = np.array([p.E for p in d['data_A']])
    E_B = np.array([p.E for p in d['data_B']])

    cp = d['con_params']
    M1, G1, M2, G2, r_shared, phi_shared, c1_A, scale_A, c1_B, scale_B = cp

    sigma_A = model_A(E_A, M1, G1, M2, G2, c1_A, r_shared, phi_shared, scale_A)
    sigma_B = model_B(E_B, M1, G1, M2, G2, c1_B, r_shared, phi_shared, scale_B)

    pseudo_A = [DataPoint(p.E, max(0.1, np.random.normal(sigma_A[i], p.stat_err)), p.stat_err)
                for i, p in enumerate(d['data_A'])]
    pseudo_B = [DataPoint(p.E, max(0.1, np.random.normal(sigma_B[i], p.stat_err)), p.stat_err)
                for i, p in enumerate(d['data_B'])]

    try:
        nll_con, con_p = fit_constrained(pseudo_A, pseudo_B, n_starts=80)
        if con_p is None:
            return 0.0
        nll_unc, _ = fit_unconstrained_seeded(pseudo_A, pseudo_B, con_p, n_starts=80)
        Lambda = 2 * (nll_con - nll_unc)
        return max(0, Lambda)
    except:
        return 0.0

def main():
    print("=" * 60)
    print("BESIII Y-Sector Rank-1 Test v3 (Fixed)")
    print("=" * 60)

    data_A = [p for p in DATA_A if 4.08 <= p.E <= 4.55]  # Restrict to overlap region
    data_B = [p for p in DATA_B if 4.08 <= p.E <= 4.55]

    print(f"\nData: {len(data_A)} (A), {len(data_B)} (B) points in [4.08, 4.55] GeV")

    print("\n1) Fitting constrained (shared R)...")
    nll_con, con_params = fit_constrained(data_A, data_B, n_starts=400)
    print(f"   NLL_con = {nll_con:.2f}")
    M1, G1, M2, G2, r_shared, phi_shared = con_params[:6]
    print(f"   M1 = {M1*1000:.1f} MeV, G1 = {G1*1000:.1f} MeV")
    print(f"   M2 = {M2*1000:.1f} MeV, G2 = {G2*1000:.1f} MeV")
    print(f"   R_shared = {r_shared:.3f} exp(i {phi_shared:.3f})")

    print("\n2) Fitting unconstrained (seeded from constrained)...")
    nll_unc, unc_params = fit_unconstrained_seeded(data_A, data_B, con_params, n_starts=300)
    print(f"   NLL_unc = {nll_unc:.2f}")
    r_A = unc_params[5]
    phi_A = unc_params[6]
    r_B = unc_params[9]
    phi_B = unc_params[10]
    print(f"   R_A = {r_A:.3f} exp(i {phi_A:.3f})")
    print(f"   R_B = {r_B:.3f} exp(i {phi_B:.3f})")

    Lambda = 2 * (nll_con - nll_unc)
    Lambda = max(0, Lambda)
    print(f"\n3) Lambda = 2*(NLL_con - NLL_unc) = {Lambda:.2f}")

    if nll_con < nll_unc:
        print("   WARNING: NLL_con < NLL_unc - unconstrained didn't find global min")
        Lambda = 0

    # Chi2
    E_A = np.array([p.E for p in data_A])
    E_B = np.array([p.E for p in data_B])
    sigma_A = model_A(E_A, M1, G1, M2, G2, con_params[6], r_shared, phi_shared, con_params[7])
    sigma_B = model_B(E_B, M1, G1, M2, G2, con_params[8], r_shared, phi_shared, con_params[9])

    chi2_A = chi2(data_A, sigma_A)
    chi2_B = chi2(data_B, sigma_B)
    ndof_A = len(data_A) - 5  # M1,G1,M2,G2,c1,scale shared; r,phi shared
    ndof_B = len(data_B) - 5

    print(f"\n4) Fit health:")
    print(f"   Channel A: chi2/dof = {chi2_A/ndof_A:.2f}")
    print(f"   Channel B: chi2/dof = {chi2_B/ndof_B:.2f}")

    health_A = 0.5 < chi2_A/ndof_A < 3.0
    health_B = 0.5 < chi2_B/ndof_B < 3.0

    print(f"\n5) Running bootstrap (500 replicates)...")
    n_workers = max(1, cpu_count() - 1)
    seeds = list(range(42, 42 + 500))

    with Pool(n_workers, initializer=_init_boot, initargs=(data_A, data_B, con_params)) as pool:
        Lambda_boot = np.array(list(pool.map(_boot_worker, seeds)))

    p_value = np.mean(Lambda_boot >= Lambda)
    print(f"   p-value = {p_value:.3f}")

    # Plots
    print("\n6) Generating plots...")

    E_fine = np.linspace(4.08, 4.55, 200)
    sigma_A_fit = model_A(E_fine, M1, G1, M2, G2, con_params[6], r_shared, phi_shared, con_params[7])
    sigma_B_fit = model_B(E_fine, M1, G1, M2, G2, con_params[8], r_shared, phi_shared, con_params[9])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.errorbar([p.E for p in data_A], [p.sigma for p in data_A],
                 yerr=[p.stat_err for p in data_A], fmt='ko', ms=4, capsize=2)
    ax1.plot(E_fine, sigma_A_fit, 'r-', lw=2)
    ax1.set_xlabel('$\\sqrt{s}$ (GeV)')
    ax1.set_ylabel('$\\sigma$ (pb)')
    ax1.set_title(f'$\\pi\\pi J/\\psi$ ($\\chi^2$/dof = {chi2_A/ndof_A:.2f})')
    ax1.grid(True, alpha=0.3)

    ax2.errorbar([p.E for p in data_B], [p.sigma for p in data_B],
                 yerr=[p.stat_err for p in data_B], fmt='ko', ms=4, capsize=2)
    ax2.plot(E_fine, sigma_B_fit, 'b-', lw=2)
    ax2.set_xlabel('$\\sqrt{s}$ (GeV)')
    ax2.set_ylabel('$\\sigma$ (pb)')
    ax2.set_title(f'$\\pi\\pi h_c$ ($\\chi^2$/dof = {chi2_B/ndof_B:.2f})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fits_v3.png', dpi=150)

    plt.figure(figsize=(8, 5))
    plt.hist(Lambda_boot, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(Lambda, color='red', ls='--', lw=2, label=f'Obs $\\Lambda$ = {Lambda:.2f}')
    plt.xlabel('$\\Lambda$')
    plt.ylabel('Count')
    plt.title(f'Bootstrap: p = {p_value:.3f}')
    plt.legend()
    plt.savefig(f'{OUT}/bootstrap_v3.png', dpi=150)

    # Verdict
    if not health_A or not health_B:
        verdict = "MODEL MISMATCH"
        reason = f"chi2/dof: A={chi2_A/ndof_A:.2f}, B={chi2_B/ndof_B:.2f}"
    elif p_value > 0.05:
        verdict = "SUPPORTED"
        reason = f"p = {p_value:.3f} > 0.05"
    else:
        verdict = "DISFAVORED"
        reason = f"p = {p_value:.3f} < 0.05"

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"Reason: {reason}")
    print(f"R_A = {r_A:.3f} exp(i {phi_A:.3f})")
    print(f"R_B = {r_B:.3f} exp(i {phi_B:.3f})")
    print(f"R_shared = {r_shared:.3f} exp(i {phi_shared:.3f})")
    print(f"Lambda = {Lambda:.2f}, p = {p_value:.3f}")
    print(f"chi2/dof: A = {chi2_A/ndof_A:.2f}, B = {chi2_B/ndof_B:.2f}")

if __name__ == "__main__":
    main()
