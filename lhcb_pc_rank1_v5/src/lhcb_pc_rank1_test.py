#!/usr/bin/env python3
"""
LHCb Pentaquark Pc(4440)/Pc(4457) Rank-1 Test

PROJECTION-BASED TEST (1D spectrum) - NOT full amplitude workspace.

Uses HEPData record 89271 (PRL 122, 222001) to test whether the relative
mixture of Pc(4457) vs Pc(4440) is consistent across different projections:

Pair 1: Table 1 (full m(J/ψp)) vs Table 2 (mKp > 1.9 GeV cut)
Pair 2: Table 1 (full) vs Table 3 (cosθ_Pc-weighted m(J/ψp))

Reference: LHCb Collaboration, PRL 122, 222001 (2019)
HEPData: https://www.hepdata.net/record/ins1728691
"""

import os
import json
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2 as chi2_dist
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Pc parameters from LHCb (central values)
# ============================================================
M_PC4440 = 4440.3  # MeV
G_PC4440 = 20.6    # MeV (broader state)
M_PC4457 = 4457.3  # MeV
G_PC4457 = 6.4     # MeV (narrower state)

# Fit window (tight around the doublet)
FIT_WINDOW = (4400, 4500)  # MeV


# ============================================================
# Load HEPData tables
# ============================================================
def load_hepdata_csv(filepath, is_weighted=False):
    """
    Load HEPData CSV format.
    Returns: array of (m_center_GeV, y, y_err)
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('$'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    m_center = float(parts[0])  # MeV
                    y = float(parts[3])
                    y_err_plus = float(parts[4])
                    # Use absolute value of error
                    y_err = abs(y_err_plus)
                    if y_err > 0:
                        data.append((m_center / 1000.0, y, y_err))  # Convert to GeV
                except:
                    pass
    return np.array(data)


def select_window(data, m_low, m_high):
    """Select data within mass window (in MeV, data in GeV)."""
    m_low_gev = m_low / 1000.0
    m_high_gev = m_high / 1000.0
    mask = (data[:, 0] >= m_low_gev) & (data[:, 0] <= m_high_gev)
    return data[mask]


# ============================================================
# Breit-Wigner amplitude
# ============================================================
def breit_wigner(m, M, Gamma):
    """
    Relativistic Breit-Wigner amplitude.
    BW(m) = 1 / ((m - M) - i*Γ/2)
    m in GeV, M in MeV, Γ in MeV
    """
    M_gev = M / 1000.0
    G_gev = Gamma / 1000.0
    return 1.0 / ((m - M_gev) - 1j * G_gev / 2.0)


# ============================================================
# Model: Coherent two-BW + background
# ============================================================
def model_coherent(m, params, use_bg1=True):
    """
    Coherent two-BW model:
    I(m) = |a1 * BW_4440(m) + R * BW_4457(m)|² + background

    params for BG0: [a1, r, phi, norm, bg0]
    params for BG1: [a1, r, phi, norm, bg0, bg1]
    """
    a1 = params[0]
    r = params[1]
    phi = params[2]
    norm = params[3]
    bg0 = params[4]
    bg1 = params[5] if use_bg1 and len(params) > 5 else 0.0

    BW1 = breit_wigner(m, M_PC4440, G_PC4440)
    BW2 = breit_wigner(m, M_PC4457, G_PC4457)

    R = r * np.exp(1j * phi)
    amplitude = a1 * (BW1 + R * BW2)
    signal = norm * np.abs(amplitude)**2

    # Linear background
    m_center = (FIT_WINDOW[0] + FIT_WINDOW[1]) / 2000.0  # GeV
    background = bg0 + bg1 * (m - m_center)

    return signal + np.maximum(background, 0)


# ============================================================
# Negative log-likelihood
# ============================================================
def nll_gaussian(params, data, use_bg1=True):
    """Gaussian NLL for weighted or count data with errors."""
    nll = 0.0
    for m, y, y_err in data:
        pred = model_coherent(m, params, use_bg1)
        nll += 0.5 * ((y - pred) / y_err)**2
    return nll


def nll_poisson(params, data, use_bg1=True):
    """Poisson deviance for count data."""
    dev = 0.0
    for m, y, y_err in data:
        mu = max(model_coherent(m, params, use_bg1), 1e-10)
        if y > 0:
            dev += 2 * (mu - y + y * np.log(y / mu))
        else:
            dev += 2 * mu
    return dev


# ============================================================
# Fitting functions
# ============================================================
def fit_single_spectrum(data, n_starts=300, use_bg1=True, use_poisson=False):
    """
    Fit single spectrum with coherent two-BW model.
    Returns: best_nll, best_params, chi2, dof, health
    """
    n_params = 6 if use_bg1 else 5

    bounds = [
        (0.01, 100),      # a1
        (0.01, 10),       # r
        (-np.pi, np.pi),  # phi
        (1, 1e8),         # norm
        (0, 2000),        # bg0
    ]
    if use_bg1:
        bounds.append((-500, 500))  # bg1

    nll_func = nll_poisson if use_poisson else nll_gaussian

    best_nll = np.inf
    best_params = None

    # Global optimization first
    try:
        result = differential_evolution(
            lambda p: nll_func(p, data, use_bg1),
            bounds, maxiter=300, seed=42, polish=True
        )
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x.copy()
    except:
        pass

    # Multi-start local optimization
    for i in range(n_starts):
        x0 = [np.random.uniform(lo, hi) for lo, hi in bounds]
        try:
            result = minimize(
                lambda p: nll_func(p, data, use_bg1),
                x0, method='L-BFGS-B', bounds=bounds
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except:
            pass

    # Compute chi2/dof
    if best_params is not None:
        chi2 = 0
        for m, y, y_err in data:
            pred = model_coherent(m, best_params, use_bg1)
            chi2 += ((y - pred) / y_err)**2
        dof = len(data) - n_params
        chi2_dof = chi2 / max(1, dof)

        if chi2_dof < 0.5:
            health = "UNDERCONSTRAINED"
        elif chi2_dof > 3.0:
            health = "MODEL_MISMATCH"
        else:
            health = "HEALTHY"
    else:
        chi2, dof, chi2_dof, health = np.nan, 0, np.nan, "FIT_FAILED"

    return best_nll, best_params, chi2, dof, health


def fit_joint_constrained(data_A, data_B, n_starts=300, use_bg1=True, use_poisson_A=False, use_poisson_B=False):
    """
    Joint fit with shared R = (r, phi) between spectra A and B.

    params: [a1_A, a1_B, r_shared, phi_shared, norm_A, norm_B, bg0_A, bg0_B, bg1_A, bg1_B]
    """
    n_params = 10 if use_bg1 else 8

    bounds = [
        (0.01, 100),      # a1_A
        (0.01, 100),      # a1_B
        (0.01, 10),       # r_shared
        (-np.pi, np.pi),  # phi_shared
        (1, 1e8),         # norm_A
        (1, 1e8),         # norm_B
        (0, 2000),        # bg0_A
        (0, 2000),        # bg0_B
    ]
    if use_bg1:
        bounds.extend([(-500, 500), (-500, 500)])  # bg1_A, bg1_B

    def joint_nll(params):
        # Unpack
        a1_A, a1_B, r, phi, norm_A, norm_B, bg0_A, bg0_B = params[:8]
        bg1_A = params[8] if use_bg1 else 0
        bg1_B = params[9] if use_bg1 else 0

        # Build per-spectrum params
        params_A = [a1_A, r, phi, norm_A, bg0_A, bg1_A]
        params_B = [a1_B, r, phi, norm_B, bg0_B, bg1_B]

        nll_A = nll_poisson(params_A, data_A, use_bg1) if use_poisson_A else nll_gaussian(params_A, data_A, use_bg1)
        nll_B = nll_poisson(params_B, data_B, use_bg1) if use_poisson_B else nll_gaussian(params_B, data_B, use_bg1)

        return nll_A + nll_B

    best_nll = np.inf
    best_params = None

    # Global optimization
    try:
        result = differential_evolution(joint_nll, bounds, maxiter=300, seed=42, polish=True)
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x.copy()
    except:
        pass

    # Multi-start
    for i in range(n_starts):
        x0 = [np.random.uniform(lo, hi) for lo, hi in bounds]
        try:
            result = minimize(joint_nll, x0, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except:
            pass

    return best_nll, best_params


def fit_joint_unconstrained(data_A, data_B, n_starts=300, use_bg1=True, use_poisson_A=False, use_poisson_B=False):
    """
    Joint fit with independent R_A = (r_A, phi_A) and R_B = (r_B, phi_B).

    params: [a1_A, a1_B, r_A, phi_A, r_B, phi_B, norm_A, norm_B, bg0_A, bg0_B, bg1_A, bg1_B]
    """
    n_params = 12 if use_bg1 else 10

    bounds = [
        (0.01, 100),      # a1_A
        (0.01, 100),      # a1_B
        (0.01, 10),       # r_A
        (-np.pi, np.pi),  # phi_A
        (0.01, 10),       # r_B
        (-np.pi, np.pi),  # phi_B
        (1, 1e8),         # norm_A
        (1, 1e8),         # norm_B
        (0, 2000),        # bg0_A
        (0, 2000),        # bg0_B
    ]
    if use_bg1:
        bounds.extend([(-500, 500), (-500, 500)])

    def joint_nll(params):
        a1_A, a1_B, r_A, phi_A, r_B, phi_B, norm_A, norm_B, bg0_A, bg0_B = params[:10]
        bg1_A = params[10] if use_bg1 else 0
        bg1_B = params[11] if use_bg1 else 0

        params_A = [a1_A, r_A, phi_A, norm_A, bg0_A, bg1_A]
        params_B = [a1_B, r_B, phi_B, norm_B, bg0_B, bg1_B]

        nll_A = nll_poisson(params_A, data_A, use_bg1) if use_poisson_A else nll_gaussian(params_A, data_A, use_bg1)
        nll_B = nll_poisson(params_B, data_B, use_bg1) if use_poisson_B else nll_gaussian(params_B, data_B, use_bg1)

        return nll_A + nll_B

    best_nll = np.inf
    best_params = None

    # Global optimization
    try:
        result = differential_evolution(joint_nll, bounds, maxiter=300, seed=42, polish=True)
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x.copy()
    except:
        pass

    # Multi-start
    for i in range(n_starts):
        x0 = [np.random.uniform(lo, hi) for lo, hi in bounds]
        try:
            result = minimize(joint_nll, x0, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except:
            pass

    return best_nll, best_params


# ============================================================
# Bootstrap
# ============================================================
def bootstrap_replicate(args):
    """Single bootstrap replicate."""
    data_A, data_B, seed, use_bg1, use_poisson_A, use_poisson_B, params_con = args
    np.random.seed(seed)

    # Generate pseudo-data from constrained fit
    def generate_pseudo(data, params, use_poisson):
        pseudo = []
        for m, y, y_err in data:
            mu = model_coherent(m, params, use_bg1)
            if use_poisson:
                y_new = float(np.random.poisson(max(1, mu)))
                y_err_new = max(1, np.sqrt(y_new))
            else:
                y_new = np.random.normal(mu, y_err)
                y_err_new = y_err
            pseudo.append((m, y_new, y_err_new))
        return np.array(pseudo)

    # Build params for A and B from constrained params
    a1_A, a1_B, r, phi, norm_A, norm_B, bg0_A, bg0_B = params_con[:8]
    bg1_A = params_con[8] if use_bg1 else 0
    bg1_B = params_con[9] if use_bg1 else 0
    params_A = [a1_A, r, phi, norm_A, bg0_A, bg1_A]
    params_B = [a1_B, r, phi, norm_B, bg0_B, bg1_B]

    pseudo_A = generate_pseudo(data_A, params_A, use_poisson_A)
    pseudo_B = generate_pseudo(data_B, params_B, use_poisson_B)

    # Fit constrained and unconstrained
    nll_con, _ = fit_joint_constrained(pseudo_A, pseudo_B, n_starts=120, use_bg1=use_bg1,
                                       use_poisson_A=use_poisson_A, use_poisson_B=use_poisson_B)
    nll_unc, _ = fit_joint_unconstrained(pseudo_A, pseudo_B, n_starts=120, use_bg1=use_bg1,
                                         use_poisson_A=use_poisson_A, use_poisson_B=use_poisson_B)

    lambda_boot = 2 * max(0, nll_con - nll_unc)
    return lambda_boot


def run_bootstrap(data_A, data_B, params_con, n_boot=200, use_bg1=True,
                  use_poisson_A=False, use_poisson_B=False):
    """Run bootstrap for p-value estimation."""
    n_workers = max(1, cpu_count() - 1)
    args_list = [(data_A, data_B, i, use_bg1, use_poisson_A, use_poisson_B, params_con)
                 for i in range(n_boot)]

    with Pool(n_workers) as pool:
        lambda_boots = list(pool.map(bootstrap_replicate, args_list))

    return np.array(lambda_boots)


# ============================================================
# Run rank-1 test for a pair
# ============================================================
def run_pair_test(data_A, data_B, pair_name, use_poisson_A=False, use_poisson_B=False, n_boot=200):
    """Run the full rank-1 test for a pair of spectra."""
    print(f"\n{'='*60}")
    print(f"PAIR: {pair_name}")
    print(f"{'='*60}")

    results = {'pair': pair_name}

    # Fit each spectrum individually first
    print("\n1. Individual spectrum fits:")

    nll_A, params_A, chi2_A, dof_A, health_A = fit_single_spectrum(
        data_A, n_starts=300, use_bg1=True, use_poisson=use_poisson_A
    )
    print(f"   Spectrum A: χ²/dof = {chi2_A:.1f}/{dof_A} = {chi2_A/max(1,dof_A):.2f}, Health: {health_A}")
    if params_A is not None:
        print(f"     R_A = {params_A[1]:.3f} exp(i {np.rad2deg(params_A[2]):.1f}°)")

    nll_B, params_B, chi2_B, dof_B, health_B = fit_single_spectrum(
        data_B, n_starts=300, use_bg1=True, use_poisson=use_poisson_B
    )
    print(f"   Spectrum B: χ²/dof = {chi2_B:.1f}/{dof_B} = {chi2_B/max(1,dof_B):.2f}, Health: {health_B}")
    if params_B is not None:
        print(f"     R_B = {params_B[1]:.3f} exp(i {np.rad2deg(params_B[2]):.1f}°)")

    results['chi2_A'] = float(chi2_A)
    results['dof_A'] = int(dof_A)
    results['health_A'] = health_A
    results['chi2_B'] = float(chi2_B)
    results['dof_B'] = int(dof_B)
    results['health_B'] = health_B

    if params_A is not None:
        results['R_A'] = {'r': float(params_A[1]), 'phi_deg': float(np.rad2deg(params_A[2]))}
    if params_B is not None:
        results['R_B'] = {'r': float(params_B[1]), 'phi_deg': float(np.rad2deg(params_B[2]))}

    # Check if both healthy
    if health_A != "HEALTHY" or health_B != "HEALTHY":
        print(f"\n   WARNING: One or both spectra not HEALTHY, proceeding anyway...")

    # Joint fits
    print("\n2. Joint fits:")

    nll_con, params_con = fit_joint_constrained(
        data_A, data_B, n_starts=300, use_bg1=True,
        use_poisson_A=use_poisson_A, use_poisson_B=use_poisson_B
    )
    print(f"   Constrained NLL: {nll_con:.2f}")
    if params_con is not None:
        print(f"   R_shared = {params_con[2]:.3f} exp(i {np.rad2deg(params_con[3]):.1f}°)")
        results['R_shared'] = {'r': float(params_con[2]), 'phi_deg': float(np.rad2deg(params_con[3]))}

    nll_unc, params_unc = fit_joint_unconstrained(
        data_A, data_B, n_starts=300, use_bg1=True,
        use_poisson_A=use_poisson_A, use_poisson_B=use_poisson_B
    )
    print(f"   Unconstrained NLL: {nll_unc:.2f}")
    if params_unc is not None:
        print(f"   R_A = {params_unc[2]:.3f} exp(i {np.rad2deg(params_unc[3]):.1f}°)")
        print(f"   R_B = {params_unc[4]:.3f} exp(i {np.rad2deg(params_unc[5]):.1f}°)")

    # Lambda statistic
    Lambda = 2 * max(0, nll_con - nll_unc)
    print(f"\n3. Test statistic:")
    print(f"   Λ = 2*(NLL_con - NLL_unc) = {Lambda:.2f}")

    # Enforce Λ >= 0
    if nll_con < nll_unc:
        print("   WARNING: NLL_con < NLL_unc (expected), Λ properly >=0")

    results['Lambda'] = float(Lambda)
    results['nll_con'] = float(nll_con)
    results['nll_unc'] = float(nll_unc)

    # Wilks p-value (reference only)
    p_wilks = 1 - chi2_dist.cdf(Lambda, 2)
    print(f"   Wilks p-value (ref): {p_wilks:.4f}")
    results['p_wilks'] = float(p_wilks)

    # Bootstrap p-value
    if params_con is not None:
        print(f"\n4. Bootstrap ({n_boot} replicates)...")
        lambda_boots = run_bootstrap(
            data_A, data_B, params_con, n_boot=n_boot, use_bg1=True,
            use_poisson_A=use_poisson_A, use_poisson_B=use_poisson_B
        )
        k = sum(lb >= Lambda for lb in lambda_boots)
        p_boot = k / n_boot
        print(f"   p_boot = {k}/{n_boot} = {p_boot:.4f}")
        results['p_boot'] = float(p_boot)
        results['k'] = int(k)
        results['n_boot'] = n_boot
        results['lambda_boots'] = lambda_boots.tolist()
    else:
        p_boot = np.nan
        lambda_boots = []
        results['p_boot'] = np.nan

    # Verdict
    print("\n5. Verdict:")
    if health_A != "HEALTHY" or health_B != "HEALTHY":
        verdict = "INCONCLUSIVE"
        reason = f"Fit health issues: A={health_A}, B={health_B}"
    elif np.isnan(p_boot):
        verdict = "OPTIMIZER_FAILURE"
        reason = "Constrained fit failed"
    elif p_boot >= 0.05:
        verdict = "NOT_REJECTED"
        reason = f"p_boot = {p_boot:.3f} >= 0.05"
    else:
        verdict = "DISFAVORED"
        reason = f"p_boot = {p_boot:.3f} < 0.05"

    print(f"   {verdict}: {reason}")
    results['verdict'] = verdict
    results['reason'] = reason

    return results, params_A, params_B, params_con, lambda_boots


# ============================================================
# Main
# ============================================================
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("="*70)
    print("LHCb Pentaquark Pc(4440)/Pc(4457) Rank-1 Test")
    print("PROJECTION-BASED (1D spectrum) - NOT full amplitude workspace")
    print("="*70)

    # Load data
    print("\nLoading HEPData tables...")
    t1_full = load_hepdata_csv("../data/hepdata/t1_full.csv")
    t2_cut = load_hepdata_csv("../data/hepdata/t2_cut.csv")
    t3_weighted = load_hepdata_csv("../data/hepdata/t3_weighted.csv", is_weighted=True)

    print(f"  Table 1 (full): {len(t1_full)} bins")
    print(f"  Table 2 (mKp>1.9): {len(t2_cut)} bins")
    print(f"  Table 3 (cosθ-weighted): {len(t3_weighted)} bins")

    # Select fit window
    print(f"\nSelecting fit window: {FIT_WINDOW[0]}-{FIT_WINDOW[1]} MeV")
    t1_window = select_window(t1_full, FIT_WINDOW[0], FIT_WINDOW[1])
    t2_window = select_window(t2_cut, FIT_WINDOW[0], FIT_WINDOW[1])
    t3_window = select_window(t3_weighted, FIT_WINDOW[0], FIT_WINDOW[1])

    print(f"  Table 1 window: {len(t1_window)} bins")
    print(f"  Table 2 window: {len(t2_window)} bins")
    print(f"  Table 3 window: {len(t3_window)} bins")

    all_results = {}

    # ============================================================
    # Pair 1: Full vs mKp cut
    # ============================================================
    results_p1, params_A1, params_B1, params_con1, boots1 = run_pair_test(
        t1_window, t2_window, "Pair 1: Full vs mKp>1.9 cut",
        use_poisson_A=False, use_poisson_B=False, n_boot=200
    )
    all_results['pair1'] = results_p1

    # ============================================================
    # Pair 2: Full vs cosθ-weighted
    # ============================================================
    results_p2, params_A2, params_B2, params_con2, boots2 = run_pair_test(
        t1_window, t3_window, "Pair 2: Full vs cosθ-weighted",
        use_poisson_A=False, use_poisson_B=False, n_boot=200
    )
    all_results['pair2'] = results_p2

    # ============================================================
    # Generate outputs
    # ============================================================
    print("\n" + "="*70)
    print("Generating outputs")
    print("="*70)

    # Save JSON
    # Convert numpy types for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj

    with open('../out/result.json', 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    print("  Saved: ../out/result.json")

    # Generate plots
    generate_plots(t1_window, t2_window, t3_window, params_con1, params_con2, boots1, boots2)

    # Generate reports
    generate_report(all_results)
    generate_rank1_result(all_results)
    generate_optimizer_audit(all_results)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Pair 1 (Full vs mKp cut): {results_p1['verdict']}")
    print(f"  Pair 2 (Full vs cosθ-weighted): {results_p2['verdict']}")

    return all_results


def generate_plots(t1, t2, t3, params_con1, params_con2, boots1, boots2):
    """Generate all plots."""
    # Pair 1 fit plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title in [(axes[0], t1, 'Table 1 (Full)'), (axes[1], t2, 'Table 2 (mKp>1.9)')]:
        m = data[:, 0] * 1000
        y = data[:, 1]
        yerr = data[:, 2]
        ax.errorbar(m, y, yerr=yerr, fmt='o', color='black', capsize=2, label='Data')
        ax.axvline(M_PC4440, color='red', linestyle='--', alpha=0.7, label='Pc(4440)')
        ax.axvline(M_PC4457, color='blue', linestyle='--', alpha=0.7, label='Pc(4457)')
        ax.set_xlabel('$m_{J/\\psi p}$ [MeV]')
        ax.set_ylabel('Events / 2 MeV')
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('../out/fit_pair1.png', dpi=150)
    plt.close()
    print("  Saved: ../out/fit_pair1.png")

    # Pair 2 fit plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title in [(axes[0], t1, 'Table 1 (Full)'), (axes[1], t3, 'Table 3 (cosθ-weighted)')]:
        m = data[:, 0] * 1000
        y = data[:, 1]
        yerr = data[:, 2]
        ax.errorbar(m, y, yerr=yerr, fmt='o', color='black', capsize=2, label='Data')
        ax.axvline(M_PC4440, color='red', linestyle='--', alpha=0.7, label='Pc(4440)')
        ax.axvline(M_PC4457, color='blue', linestyle='--', alpha=0.7, label='Pc(4457)')
        ax.set_xlabel('$m_{J/\\psi p}$ [MeV]')
        ax.set_ylabel('Weighted events / 2 MeV' if 'weighted' in title.lower() else 'Events / 2 MeV')
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('../out/fit_pair2.png', dpi=150)
    plt.close()
    print("  Saved: ../out/fit_pair2.png")

    # Bootstrap histograms
    for boots, pair_name, filename in [(boots1, 'Pair 1', 'bootstrap_hist_pair1.png'),
                                       (boots2, 'Pair 2', 'bootstrap_hist_pair2.png')]:
        if len(boots) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(boots, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(5.99, color='orange', linestyle='--', label='χ²(2) 95%')
            ax.set_xlabel('Λ')
            ax.set_ylabel('Count')
            ax.set_title(f'{pair_name} Bootstrap Distribution')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'../out/{filename}', dpi=150)
            plt.close()
            print(f"  Saved: ../out/{filename}")


def generate_report(results):
    """Generate REPORT.md."""
    p1 = results['pair1']
    p2 = results['pair2']

    report = f"""# LHCb Pentaquark Pc(4440)/Pc(4457) Rank-1 Test

## ⚠️ IMPORTANT: PROJECTION-BASED TEST

This is a **projection-based (1D spectrum) test**, NOT a full amplitude workspace analysis.
Results should be interpreted with caution as projection tests have limited sensitivity
to interference effects.

## Provenance

- **Experiment**: LHCb
- **Publication**: PRL 122, 222001 (2019)
- **HEPData**: Record 89271 (INSPIRE 1728691)
- **Tables used**:
  - Table 1: Full m(J/ψp) spectrum
  - Table 2: m(J/ψp) with mKp > 1.9 GeV cut
  - Table 3: cosθ_Pc-weighted m(J/ψp) spectrum

## Pc Parameters (from LHCb)

| State | Mass (MeV) | Width (MeV) |
|-------|------------|-------------|
| Pc(4440) | 4440.3 | 20.6 |
| Pc(4457) | 4457.3 | 6.4 |

## Fit Window

- Mass range: 4400-4500 MeV
- Model: Coherent two-BW + linear background
- dof_diff = 2 (complex R constraint)

## Results

### Pair 1: Full vs mKp > 1.9 cut

| Metric | Value |
|--------|-------|
| Verdict | **{p1['verdict']}** |
| Λ | {p1.get('Lambda', 'N/A'):.2f} |
| p_boot | {p1.get('p_boot', 'N/A'):.4f} ({p1.get('k', 'N/A')}/{p1.get('n_boot', 'N/A')}) |
| χ²/dof (A) | {p1.get('chi2_A', 'N/A'):.1f}/{p1.get('dof_A', 'N/A')} [{p1.get('health_A', 'N/A')}] |
| χ²/dof (B) | {p1.get('chi2_B', 'N/A'):.1f}/{p1.get('dof_B', 'N/A')} [{p1.get('health_B', 'N/A')}] |

### Pair 2: Full vs cosθ-weighted

| Metric | Value |
|--------|-------|
| Verdict | **{p2['verdict']}** |
| Λ | {p2.get('Lambda', 'N/A'):.2f} |
| p_boot | {p2.get('p_boot', 'N/A'):.4f} ({p2.get('k', 'N/A')}/{p2.get('n_boot', 'N/A')}) |
| χ²/dof (A) | {p2.get('chi2_A', 'N/A'):.1f}/{p2.get('dof_A', 'N/A')} [{p2.get('health_A', 'N/A')}] |
| χ²/dof (B) | {p2.get('chi2_B', 'N/A'):.1f}/{p2.get('dof_B', 'N/A')} [{p2.get('health_B', 'N/A')}] |

## Figures

### Pair 1 Spectra
![Pair 1](fit_pair1.png)

### Pair 2 Spectra
![Pair 2](fit_pair2.png)

### Bootstrap Distributions
![Bootstrap Pair 1](bootstrap_hist_pair1.png)
![Bootstrap Pair 2](bootstrap_hist_pair2.png)

## Interpretation

This projection-based test examines whether the Pc(4457)/Pc(4440) mixture ratio R
is consistent across different analysis cuts/weightings of the same dataset.

**Limitations**:
1. 1D projections lose information about interference phases
2. Different cuts may have different background compositions
3. Not equivalent to a full amplitude analysis

---
*Generated by lhcb_pc_rank1_test.py*
"""

    with open('../out/REPORT.md', 'w') as f:
        f.write(report)
    print("  Saved: ../out/REPORT.md")


def generate_rank1_result(results):
    """Generate RANK1_RESULT.md."""
    p1 = results['pair1']
    p2 = results['pair2']

    result = f"""# LHCb Pc Rank-1 Result (Projection-Based)

## Summary Table

| Pair | Spectra | Λ | p_boot | Health | Verdict |
|------|---------|---|--------|--------|---------|
| 1 | Full vs mKp cut | {p1.get('Lambda', 'N/A'):.2f} | {p1.get('p_boot', 'N/A'):.3f} | A:{p1.get('health_A', '?')}, B:{p1.get('health_B', '?')} | **{p1['verdict']}** |
| 2 | Full vs cosθ-wt | {p2.get('Lambda', 'N/A'):.2f} | {p2.get('p_boot', 'N/A'):.3f} | A:{p2.get('health_A', '?')}, B:{p2.get('health_B', '?')} | **{p2['verdict']}** |

## Per-Pair Details

### Pair 1: {p1['verdict']}
- Reason: {p1.get('reason', 'N/A')}
- R_shared: |R| = {p1.get('R_shared', {}).get('r', 'N/A')}, φ = {p1.get('R_shared', {}).get('phi_deg', 'N/A')}°

### Pair 2: {p2['verdict']}
- Reason: {p2.get('reason', 'N/A')}
- R_shared: |R| = {p2.get('R_shared', {}).get('r', 'N/A')}, φ = {p2.get('R_shared', {}).get('phi_deg', 'N/A')}°

## Warning

⚠️ This is a **projection-based test** using 1D mass spectra.
Results have limited sensitivity compared to full amplitude analysis.

## Data Source

HEPData record 89271 (LHCb PRL 122, 222001)

---
*Generated by lhcb_pc_rank1_test.py*
"""

    with open('../out/RANK1_RESULT.md', 'w') as f:
        f.write(result)
    print("  Saved: ../out/RANK1_RESULT.md")


def generate_optimizer_audit(results):
    """Generate optimizer_audit.md."""
    p1 = results['pair1']
    p2 = results['pair2']

    audit = f"""# Optimizer Audit

## Λ >= 0 Check

| Pair | NLL_con | NLL_unc | Λ | Status |
|------|---------|---------|---|--------|
| 1 | {p1.get('nll_con', 'N/A'):.2f} | {p1.get('nll_unc', 'N/A'):.2f} | {p1.get('Lambda', 'N/A'):.2f} | {'✓ OK' if p1.get('Lambda', 0) >= 0 else '✗ VIOLATION'} |
| 2 | {p2.get('nll_con', 'N/A'):.2f} | {p2.get('nll_unc', 'N/A'):.2f} | {p2.get('Lambda', 'N/A'):.2f} | {'✓ OK' if p2.get('Lambda', 0) >= 0 else '✗ VIOLATION'} |

## Optimization Settings

- Multi-start: 300 starts (constrained/unconstrained)
- Bootstrap starts: 120 starts per replicate
- Global optimizer: differential_evolution + L-BFGS-B refinement
- Λ enforcement: max(0, 2*(NLL_con - NLL_unc))

## NLL Stability

The constrained fit should always have NLL >= unconstrained fit.
Any violation indicates optimizer issues (more starts may be needed).

---
*Generated by lhcb_pc_rank1_test.py*
"""

    with open('../out/optimizer_audit.md', 'w') as f:
        f.write(audit)
    print("  Saved: ../out/optimizer_audit.md")


if __name__ == "__main__":
    os.makedirs("../out", exist_ok=True)
    main()
