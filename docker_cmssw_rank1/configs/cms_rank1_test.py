#!/usr/bin/env python3
"""
CMS Rank-1 Bottleneck Test - Publication-Grade Inference Tool

Tests if the complex coupling ratio R = c2/c1 is shared between two channels.
This is a statistically rigorous implementation with:
- Correct degrees of freedom (dof=2 for complex R constraint)
- Bootstrap p-values as default (with Wilks approximation as reference)
- Fit-health gates including underconstrained detection
- Multi-start optimizer with fallback methods
- Injection/recovery mode for end-to-end validation

Usage:
  python3 cms_rank1_test.py \\
    --channel-a outputs/dijpsi_hist_dijpsi.csv \\
    --channel-b outputs/fourmu_hist_4mu.csv \\
    --output RANK1_RESULT.md \\
    --bootstrap 500 --starts 50

  # Injection mode:
  python3 cms_rank1_test.py --inject --bootstrap 300 --outdir outputs/injection_test

Version: 2.0.0 (Publication Grade)
"""

import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist
from multiprocessing import Pool, cpu_count
import warnings
import os
import sys
import json
from datetime import datetime
from typing import Tuple, Dict, List, Optional

warnings.filterwarnings('ignore')

# ============================================================
# Physical constants for two-resonance model
# ============================================================
M1_DEFAULT = 9.4    # First resonance mass (GeV)
G1_DEFAULT = 0.02   # First resonance width (GeV)
M2_DEFAULT = 10.0   # Second resonance mass (GeV)
G2_DEFAULT = 0.05   # Second resonance width (GeV)

# ============================================================
# Fit health thresholds
# ============================================================
CHI2_DOF_LOW = 0.5    # Below this = UNDERCONSTRAINED
CHI2_DOF_HIGH = 3.0   # Above this = MODEL_MISMATCH
DEVIANCE_DOF_HIGH = 3.0  # Poisson deviance threshold

# ============================================================
# Verdict strings (avoid overclaiming)
# ============================================================
VERDICT_NOT_REJECTED = "NOT_REJECTED"
VERDICT_DISFAVORED = "DISFAVORED"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_MODEL_MISMATCH = "MODEL_MISMATCH"
VERDICT_OPTIMIZER_FAILURE = "OPTIMIZER_FAILURE"


def load_channel_data(csv_file: str) -> np.ndarray:
    """
    Load channel data from CSV.

    Expected format:
        mass_GeV,counts,stat_err
        6.0500,0,1.0000
        ...

    Returns:
        ndarray of shape (N, 3): [mass, counts, error]
    """
    data = []
    with open(csv_file, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                mass = float(parts[0])
                counts = float(parts[1])
                err = float(parts[2])
                # Include bins with zero counts but valid errors
                if err > 0:
                    data.append((mass, counts, err))
    return np.array(data)


def breit_wigner(m: np.ndarray, M: float, Gamma: float) -> np.ndarray:
    """Relativistic Breit-Wigner amplitude."""
    s = m**2
    M2 = M**2
    return M * Gamma / (M2 - s + 1j * M * Gamma)


def model_amplitude(m: np.ndarray, params: np.ndarray, channel: str) -> np.ndarray:
    """
    Two-resonance amplitude model intensity.

    Parameters layout (12 total):
        [0] M1, [1] G1, [2] M2, [3] G2,
        [4] c1_A, [5] r_A, [6] phi_A, [7] scale_A,
        [8] c1_B, [9] r_B, [10] phi_B, [11] scale_B

    For constrained fit: r_A = r_B and phi_A = phi_B
    For unconstrained fit: independent R_A, R_B
    """
    M1, G1, M2, G2 = params[:4]

    if channel == 'A':
        c1 = params[4]
        r = params[5]
        phi = params[6]
        scale = params[7]
    else:  # channel B
        c1 = params[8]
        r = params[9]
        phi = params[10]
        scale = params[11]

    R = r * np.exp(1j * phi)

    BW1 = breit_wigner(m, M1, G1)
    BW2 = breit_wigner(m, M2, G2)

    amp = c1 * (BW1 + R * BW2)
    intensity = np.abs(amp)**2 * scale

    return np.maximum(intensity, 1e-10)  # Prevent exactly zero


def gaussian_nll(params: np.ndarray, data: np.ndarray, channel: str) -> float:
    """Gaussian negative log-likelihood for one channel."""
    nll = 0.0
    masses = data[:, 0]
    pred = model_amplitude(masses, params, channel)

    for i, (m, n, err) in enumerate(data):
        nll += 0.5 * ((n - pred[i]) / err)**2

    return nll


def poisson_nll(params: np.ndarray, data: np.ndarray, channel: str) -> float:
    """Poisson negative log-likelihood for one channel."""
    nll = 0.0
    masses = data[:, 0]
    pred = model_amplitude(masses, params, channel)

    for i, (m, n, err) in enumerate(data):
        mu = max(pred[i], 1e-10)
        # Poisson NLL: mu - n*log(mu) + log(n!)
        # We ignore constant log(n!) terms
        if n > 0:
            nll += mu - n * np.log(mu)
        else:
            nll += mu

    return nll


def nll_total(params: np.ndarray, data_A: np.ndarray, data_B: np.ndarray,
              use_poisson: bool = False) -> float:
    """Total NLL for both channels."""
    nll_func = poisson_nll if use_poisson else gaussian_nll
    return nll_func(params, data_A, 'A') + nll_func(params, data_B, 'B')


def get_bounds() -> List[Tuple[float, float]]:
    """Parameter bounds for optimization."""
    return [
        (7.0, 12.0),    # M1
        (0.001, 0.5),   # G1
        (8.0, 13.0),    # M2
        (0.01, 0.5),    # G2
        (0.01, 1000),   # c1_A
        (0.01, 10),     # r_A
        (-np.pi, np.pi),  # phi_A
        (0.01, 10000),  # scale_A
        (0.01, 1000),   # c1_B
        (0.01, 10),     # r_B
        (-np.pi, np.pi),  # phi_B
        (0.01, 10000),  # scale_B
    ]


def random_init(bounds: List[Tuple[float, float]]) -> np.ndarray:
    """Generate random initial parameters within bounds."""
    return np.array([np.random.uniform(lo, hi) for lo, hi in bounds])


def fit_constrained(
    data_A: np.ndarray,
    data_B: np.ndarray,
    n_starts: int = 50,
    use_poisson: bool = False
) -> Tuple[float, np.ndarray, Dict]:
    """
    Constrained fit: R_A = R_B (shared complex coupling ratio).

    Uses multi-start optimization with fallback methods.

    Returns:
        (best_nll, best_params, audit_info)
    """
    bounds = get_bounds()
    best_nll = np.inf
    best_params = None
    audit = {"n_starts": n_starts, "n_converged": 0, "methods_used": []}

    for i in range(n_starts):
        x0 = random_init(bounds)
        # Enforce constraint: r_B = r_A, phi_B = phi_A
        x0[9] = x0[5]
        x0[10] = x0[6]

        def constrained_nll(x):
            x_full = x.copy()
            x_full[9] = x_full[5]    # r_B = r_A
            x_full[10] = x_full[6]   # phi_B = phi_A
            return nll_total(x_full, data_A, data_B, use_poisson)

        # Try L-BFGS-B first
        try:
            result = minimize(constrained_nll, x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 1000})
            if result.success or result.fun < best_nll:
                audit["n_converged"] += 1
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_params = result.x.copy()
                    best_params[9] = best_params[5]
                    best_params[10] = best_params[6]
                    audit["methods_used"].append("L-BFGS-B")
        except Exception:
            pass

        # Fallback to Powell for some starts
        if i % 5 == 0:
            try:
                # Powell doesn't use bounds, so we clip manually
                def bounded_nll(x):
                    x_clipped = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
                    return constrained_nll(x_clipped)

                result = minimize(bounded_nll, x0, method='Powell',
                                  options={'maxiter': 2000})
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_params = np.clip(result.x, [b[0] for b in bounds], [b[1] for b in bounds])
                    best_params[9] = best_params[5]
                    best_params[10] = best_params[6]
                    audit["methods_used"].append("Powell")
            except Exception:
                pass

    return best_nll, best_params, audit


def fit_unconstrained(
    data_A: np.ndarray,
    data_B: np.ndarray,
    n_starts: int = 50,
    seed_params: Optional[np.ndarray] = None,
    use_poisson: bool = False
) -> Tuple[float, np.ndarray, Dict]:
    """
    Unconstrained fit: R_A and R_B are independent.

    Returns:
        (best_nll, best_params, audit_info)
    """
    bounds = get_bounds()
    best_nll = np.inf
    best_params = None
    audit = {"n_starts": n_starts, "n_converged": 0, "methods_used": []}

    for i in range(n_starts):
        if seed_params is not None and i < 5:
            # Seed from constrained fit with perturbation
            x0 = seed_params.copy()
            x0[9] = x0[5] * np.random.uniform(0.5, 2.0)
            x0[10] = x0[6] + np.random.uniform(-1.0, 1.0)
            x0[10] = np.clip(x0[10], -np.pi, np.pi)
        else:
            x0 = random_init(bounds)

        def unconstrained_nll(x):
            return nll_total(x, data_A, data_B, use_poisson)

        # Try L-BFGS-B
        try:
            result = minimize(unconstrained_nll, x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 1000})
            if result.success or result.fun < best_nll:
                audit["n_converged"] += 1
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_params = result.x.copy()
                    audit["methods_used"].append("L-BFGS-B")
        except Exception:
            pass

        # Fallback to Nelder-Mead for some starts
        if i % 5 == 0:
            try:
                def bounded_nll(x):
                    x_clipped = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
                    return unconstrained_nll(x_clipped)

                result = minimize(bounded_nll, x0, method='Nelder-Mead',
                                  options={'maxiter': 3000})
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_params = np.clip(result.x, [b[0] for b in bounds], [b[1] for b in bounds])
                    audit["methods_used"].append("Nelder-Mead")
            except Exception:
                pass

    return best_nll, best_params, audit


def compute_chi2_dof(params: np.ndarray, data: np.ndarray, channel: str) -> float:
    """Compute Pearson chi2/dof for fit health check."""
    masses = data[:, 0]
    pred = model_amplitude(masses, params, channel)

    chi2 = 0.0
    for i, (m, n, err) in enumerate(data):
        chi2 += ((n - pred[i]) / err)**2

    # Parameters per channel: c1, r, phi, scale (4) + shared M1, G1, M2, G2 (4 shared)
    n_params = 6  # Effective per channel
    dof = max(1, len(data) - n_params)

    return chi2 / dof


def compute_poisson_deviance_dof(params: np.ndarray, data: np.ndarray, channel: str) -> float:
    """Compute Poisson deviance / dof for count data."""
    masses = data[:, 0]
    pred = model_amplitude(masses, params, channel)

    deviance = 0.0
    for i, (m, n, err) in enumerate(data):
        mu = max(pred[i], 1e-10)
        if n > 0:
            deviance += 2 * (n * np.log(n / mu) - (n - mu))
        else:
            deviance += 2 * mu

    n_params = 6
    dof = max(1, len(data) - n_params)

    return deviance / dof


def assess_fit_health(chi2_dof: float, dev_dof: float) -> Tuple[str, str]:
    """
    Assess fit health based on chi2/dof and deviance/dof.

    Returns:
        (status, reason)
    """
    if chi2_dof < CHI2_DOF_LOW:
        return "UNDERCONSTRAINED", f"chi2/dof={chi2_dof:.2f} < {CHI2_DOF_LOW}"
    elif chi2_dof > CHI2_DOF_HIGH:
        return "MODEL_MISMATCH", f"chi2/dof={chi2_dof:.2f} > {CHI2_DOF_HIGH}"
    elif dev_dof > DEVIANCE_DOF_HIGH:
        return "MODEL_MISMATCH", f"deviance/dof={dev_dof:.2f} > {DEVIANCE_DOF_HIGH}"
    else:
        return "HEALTHY", f"chi2/dof={chi2_dof:.2f}, dev/dof={dev_dof:.2f}"


def bootstrap_replicate_poisson(args: Tuple) -> float:
    """
    Single bootstrap replicate using Poisson resampling.

    Generates pseudo-data from the constrained best-fit model,
    refits both models, and returns Lambda.
    """
    params_con, data_A, data_B, seed, n_starts, use_poisson = args
    np.random.seed(seed)

    # Generate pseudo-data from constrained model
    masses_A = data_A[:, 0]
    masses_B = data_B[:, 0]

    mu_A = model_amplitude(masses_A, params_con, 'A')
    mu_B = model_amplitude(masses_B, params_con, 'B')

    # Poisson resampling
    counts_A = np.random.poisson(mu_A)
    counts_B = np.random.poisson(mu_B)

    errors_A = np.sqrt(np.maximum(counts_A, 1))
    errors_B = np.sqrt(np.maximum(counts_B, 1))

    boot_A = np.column_stack([masses_A, counts_A, errors_A])
    boot_B = np.column_stack([masses_B, counts_B, errors_B])

    # Fit constrained
    nll_con, _, _ = fit_constrained(boot_A, boot_B, n_starts=n_starts // 2, use_poisson=use_poisson)

    # Fit unconstrained
    nll_unc, _, _ = fit_unconstrained(boot_A, boot_B, n_starts=n_starts // 2, use_poisson=use_poisson)

    # Ensure proper ordering
    nll_unc = min(nll_unc, nll_con)

    lambda_boot = 2 * (nll_con - nll_unc)
    return max(0, lambda_boot)


def compute_bootstrap_pvalue(
    params_con: np.ndarray,
    data_A: np.ndarray,
    data_B: np.ndarray,
    lambda_obs: float,
    n_bootstrap: int = 500,
    n_workers: Optional[int] = None,
    n_starts: int = 30,
    use_poisson: bool = False,
    seed: int = 42
) -> Tuple[float, int, List[float]]:
    """
    Compute bootstrap p-value.

    Returns:
        (p_value, n_exceedances, lambda_distribution)
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    args_list = [
        (params_con, data_A, data_B, seed + i, n_starts, use_poisson)
        for i in range(n_bootstrap)
    ]

    with Pool(n_workers) as pool:
        lambda_boots = list(pool.map(bootstrap_replicate_poisson, args_list))

    n_exceed = sum(1 for lb in lambda_boots if lb >= lambda_obs)
    p_value = n_exceed / n_bootstrap

    return p_value, n_exceed, lambda_boots


def run_rank1_test(
    data_A: np.ndarray,
    data_B: np.ndarray,
    n_bootstrap: int = 500,
    n_starts: int = 50,
    n_workers: Optional[int] = None,
    use_poisson: bool = False,
    use_wilks_only: bool = False,
    dof_diff: int = 2,
    bootstrap_seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run the full rank-1 bottleneck test.

    Args:
        data_A, data_B: Channel data arrays
        n_bootstrap: Number of bootstrap replicates
        n_starts: Number of optimizer starts
        n_workers: Number of parallel workers
        use_poisson: Use Poisson likelihood (vs Gaussian)
        use_wilks_only: Skip bootstrap, use only Wilks approximation
        dof_diff: Degrees of freedom for constraint (2 for complex R)
        bootstrap_seed: Random seed for bootstrap
        verbose: Print progress

    Returns:
        Dictionary with all results
    """
    if verbose:
        print("=" * 60)
        print("CMS Rank-1 Bottleneck Test (Publication Grade v2.0)")
        print("=" * 60)
        print(f"\nData: {len(data_A)} (A), {len(data_B)} (B) bins")
        print(f"dof_diff: {dof_diff} (complex R constraint)")

    # ============================================================
    # 1) Fit constrained model (shared R)
    # ============================================================
    if verbose:
        print(f"\n1) Fitting constrained model (shared R, {n_starts} starts)...")

    nll_con, params_con, audit_con = fit_constrained(
        data_A, data_B, n_starts=n_starts, use_poisson=use_poisson
    )

    if params_con is None:
        if verbose:
            print("   ERROR: Constrained fit failed!")
        return {"verdict": VERDICT_OPTIMIZER_FAILURE, "reason": "Constrained fit failed"}

    r_shared = params_con[5]
    phi_shared = params_con[6]

    if verbose:
        print(f"   NLL_con = {nll_con:.2f}")
        print(f"   R_shared = {r_shared:.3f} * exp(i * {phi_shared:.3f})")
        print(f"   Converged: {audit_con['n_converged']}/{n_starts}")

    # ============================================================
    # 2) Fit unconstrained model (independent R_A, R_B)
    # ============================================================
    if verbose:
        print(f"\n2) Fitting unconstrained model (independent R, {n_starts} starts)...")

    nll_unc, params_unc, audit_unc = fit_unconstrained(
        data_A, data_B, n_starts=n_starts, seed_params=params_con, use_poisson=use_poisson
    )

    if params_unc is None:
        if verbose:
            print("   ERROR: Unconstrained fit failed!")
        return {"verdict": VERDICT_OPTIMIZER_FAILURE, "reason": "Unconstrained fit failed"}

    r_A = params_unc[5]
    phi_A = params_unc[6]
    r_B = params_unc[9]
    phi_B = params_unc[10]

    if verbose:
        print(f"   NLL_unc = {nll_unc:.2f}")
        print(f"   R_A = {r_A:.3f} * exp(i * {phi_A:.3f})")
        print(f"   R_B = {r_B:.3f} * exp(i * {phi_B:.3f})")
        print(f"   Converged: {audit_unc['n_converged']}/{n_starts}")

    # ============================================================
    # 3) Check optimizer consistency: NLL_unc <= NLL_con
    # ============================================================
    optimizer_ok = True
    if nll_unc > nll_con + 0.01:  # Small tolerance
        if verbose:
            print(f"\n   WARNING: NLL_unc ({nll_unc:.2f}) > NLL_con ({nll_con:.2f})")
            print("   Attempting additional optimization...")

        # Retry with more starts
        nll_unc2, params_unc2, _ = fit_unconstrained(
            data_A, data_B, n_starts=n_starts * 2, seed_params=params_con, use_poisson=use_poisson
        )
        if nll_unc2 < nll_unc:
            nll_unc = nll_unc2
            params_unc = params_unc2
            r_A = params_unc[5]
            phi_A = params_unc[6]
            r_B = params_unc[9]
            phi_B = params_unc[10]

        if nll_unc > nll_con + 0.01:
            if verbose:
                print("   Still violated. Setting NLL_unc = NLL_con")
            nll_unc = nll_con
            optimizer_ok = False

    # ============================================================
    # 4) Compute Lambda statistic
    # ============================================================
    Lambda = 2 * (nll_con - nll_unc)
    Lambda = max(0, Lambda)  # Can't be negative

    if verbose:
        print(f"\n3) Lambda = 2*(NLL_con - NLL_unc) = {Lambda:.2f}")

    # ============================================================
    # 5) Fit health assessment
    # ============================================================
    chi2_A = compute_chi2_dof(params_unc, data_A, 'A')
    chi2_B = compute_chi2_dof(params_unc, data_B, 'B')
    dev_A = compute_poisson_deviance_dof(params_unc, data_A, 'A')
    dev_B = compute_poisson_deviance_dof(params_unc, data_B, 'B')

    health_A, reason_A = assess_fit_health(chi2_A, dev_A)
    health_B, reason_B = assess_fit_health(chi2_B, dev_B)

    if verbose:
        print(f"\n4) Fit health:")
        print(f"   Channel A: chi2/dof={chi2_A:.2f}, dev/dof={dev_A:.2f} -> {health_A}")
        print(f"   Channel B: chi2/dof={chi2_B:.2f}, dev/dof={dev_B:.2f} -> {health_B}")

    # ============================================================
    # 6) Wilks p-value (reference only)
    # ============================================================
    p_wilks = 1 - chi2_dist.cdf(Lambda, df=dof_diff)
    chi2_threshold = chi2_dist.ppf(0.95, df=dof_diff)

    if verbose:
        print(f"\n5) Wilks approximation (dof={dof_diff}, reference only):")
        print(f"   chi2({dof_diff}) 95% threshold: {chi2_threshold:.2f}")
        print(f"   p_wilks = {p_wilks:.4f}")

    # ============================================================
    # 7) Bootstrap p-value (primary)
    # ============================================================
    p_boot = None
    n_exceed = None
    lambda_boots = []

    if not use_wilks_only:
        if verbose:
            print(f"\n6) Bootstrap p-value ({n_bootstrap} replicates, {n_workers or 'auto'} workers)...")

        p_boot, n_exceed, lambda_boots = compute_bootstrap_pvalue(
            params_con, data_A, data_B, Lambda,
            n_bootstrap=n_bootstrap,
            n_workers=n_workers,
            n_starts=max(20, n_starts // 2),
            use_poisson=use_poisson,
            seed=bootstrap_seed
        )

        if verbose:
            print(f"   p_boot = {p_boot:.4f} ({n_exceed}/{n_bootstrap} exceedances)")
            print(f"   Minimum resolvable p = {1/n_bootstrap:.4f}")

    # ============================================================
    # 8) Determine verdict
    # ============================================================
    if health_A == "MODEL_MISMATCH" or health_B == "MODEL_MISMATCH":
        verdict = VERDICT_MODEL_MISMATCH
        reason = f"Channel fit issues: A={health_A}, B={health_B}"
    elif health_A == "UNDERCONSTRAINED" or health_B == "UNDERCONSTRAINED":
        verdict = VERDICT_INCONCLUSIVE
        reason = f"Underconstrained fits: A={reason_A}, B={reason_B}"
    elif not optimizer_ok:
        verdict = VERDICT_INCONCLUSIVE
        reason = "Optimizer instability (NLL_unc > NLL_con)"
    else:
        # Use bootstrap p-value if available, else Wilks
        p_test = p_boot if p_boot is not None else p_wilks

        if p_test < 0.05:
            verdict = VERDICT_DISFAVORED
            reason = f"p={p_test:.4f} < 0.05"
        else:
            verdict = VERDICT_NOT_REJECTED
            reason = f"p={p_test:.4f} >= 0.05"

    if verbose:
        print("\n" + "=" * 60)
        print(f"VERDICT: {verdict}")
        print(f"Reason: {reason}")
        print("=" * 60)

    return {
        'nll_con': nll_con,
        'nll_unc': nll_unc,
        'Lambda': Lambda,
        'dof_diff': dof_diff,
        'chi2_threshold': chi2_threshold,
        'p_wilks': p_wilks,
        'p_boot': p_boot,
        'n_bootstrap': n_bootstrap,
        'n_exceed': n_exceed,
        'min_resolvable_p': 1 / n_bootstrap if n_bootstrap > 0 else None,
        'chi2_A': chi2_A,
        'chi2_B': chi2_B,
        'dev_A': dev_A,
        'dev_B': dev_B,
        'health_A': health_A,
        'health_B': health_B,
        'R_shared': (r_shared, phi_shared),
        'R_A': (r_A, phi_A),
        'R_B': (r_B, phi_B),
        'verdict': verdict,
        'reason': reason,
        'optimizer_ok': optimizer_ok,
        'audit_con': audit_con,
        'audit_unc': audit_unc,
        'lambda_boots': lambda_boots,
    }


def write_result_markdown(results: Dict, output_path: str, channel_a_path: str,
                          channel_b_path: str, is_injection: bool = False):
    """Write results to markdown file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# CMS Rank-1 Bottleneck Test Results\n\n")
        f.write(f"**Version**: Publication Grade v2.0\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if is_injection:
            f.write("> **Note**: This analysis used injected/simulated data for validation.\n\n")

        f.write("## Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| **Verdict** | {results['verdict']} |\n")
        f.write(f"| Reason | {results['reason']} |\n")
        f.write(f"| NLL (constrained) | {results['nll_con']:.2f} |\n")
        f.write(f"| NLL (unconstrained) | {results['nll_unc']:.2f} |\n")
        f.write(f"| Lambda | {results['Lambda']:.2f} |\n")
        f.write(f"| dof_diff | {results['dof_diff']} |\n")
        f.write(f"| chi2({results['dof_diff']}) 95% threshold | {results['chi2_threshold']:.2f} |\n")

        f.write("\n## P-values\n\n")
        f.write("| Method | p-value | Notes |\n")
        f.write("|--------|---------|-------|\n")
        if results['p_boot'] is not None:
            f.write(f"| **Bootstrap (primary)** | {results['p_boot']:.4f} | "
                    f"{results['n_exceed']}/{results['n_bootstrap']} exceedances |\n")
            f.write(f"| Minimum resolvable | {results['min_resolvable_p']:.4f} | 1/N_bootstrap |\n")
        f.write(f"| Wilks (reference) | {results['p_wilks']:.4f} | chi2({results['dof_diff']}) approximation |\n")

        f.write("\n## Fit Health\n\n")
        f.write("| Channel | chi2/dof | deviance/dof | Status |\n")
        f.write("|---------|----------|--------------|--------|\n")
        f.write(f"| A | {results['chi2_A']:.2f} | {results['dev_A']:.2f} | {results['health_A']} |\n")
        f.write(f"| B | {results['chi2_B']:.2f} | {results['dev_B']:.2f} | {results['health_B']} |\n")
        f.write(f"\n*Thresholds: UNDERCONSTRAINED if chi2/dof < {CHI2_DOF_LOW}, "
                f"MODEL_MISMATCH if chi2/dof > {CHI2_DOF_HIGH} or dev/dof > {DEVIANCE_DOF_HIGH}*\n")

        f.write("\n## Coupling Ratios\n\n")
        f.write("```\n")
        f.write(f"R_shared = {results['R_shared'][0]:.4f} * exp(i * {results['R_shared'][1]:.4f})\n")
        f.write(f"R_A      = {results['R_A'][0]:.4f} * exp(i * {results['R_A'][1]:.4f})\n")
        f.write(f"R_B      = {results['R_B'][0]:.4f} * exp(i * {results['R_B'][1]:.4f})\n")
        f.write("```\n")

        f.write("\n## Input Files\n\n")
        f.write(f"- Channel A: `{channel_a_path}`\n")
        f.write(f"- Channel B: `{channel_b_path}`\n")
        f.write(f"- Bootstrap replicates: {results['n_bootstrap']}\n")
        f.write(f"- Optimizer starts: {results['audit_con']['n_starts']}\n")

        f.write("\n## Interpretation Guide\n\n")
        f.write("| Verdict | Meaning |\n")
        f.write("|---------|--------|\n")
        f.write(f"| {VERDICT_NOT_REJECTED} | Data consistent with shared R (rank-1) |\n")
        f.write(f"| {VERDICT_DISFAVORED} | Evidence against shared R |\n")
        f.write(f"| {VERDICT_INCONCLUSIVE} | Cannot draw conclusion (fit issues) |\n")
        f.write(f"| {VERDICT_MODEL_MISMATCH} | Model does not describe data |\n")
        f.write(f"| {VERDICT_OPTIMIZER_FAILURE} | Numerical issues in optimization |\n")


def write_optimizer_audit(results: Dict, output_path: str):
    """Write optimizer audit log."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# Rank-1 Optimizer Audit Log\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Constrained Fit\n\n")
        f.write(f"- Starts: {results['audit_con']['n_starts']}\n")
        f.write(f"- Converged: {results['audit_con']['n_converged']}\n")
        f.write(f"- Methods used: {', '.join(set(results['audit_con']['methods_used']))}\n")
        f.write(f"- Final NLL: {results['nll_con']:.4f}\n")

        f.write("\n## Unconstrained Fit\n\n")
        f.write(f"- Starts: {results['audit_unc']['n_starts']}\n")
        f.write(f"- Converged: {results['audit_unc']['n_converged']}\n")
        f.write(f"- Methods used: {', '.join(set(results['audit_unc']['methods_used']))}\n")
        f.write(f"- Final NLL: {results['nll_unc']:.4f}\n")

        f.write("\n## Consistency Check\n\n")
        f.write(f"- NLL_unc <= NLL_con: {'PASS' if results['optimizer_ok'] else 'FAIL'}\n")
        f.write(f"- Delta NLL: {results['nll_con'] - results['nll_unc']:.4f}\n")

        if results['lambda_boots']:
            f.write("\n## Bootstrap Lambda Distribution\n\n")
            boots = np.array(results['lambda_boots'])
            f.write(f"- Mean: {np.mean(boots):.3f}\n")
            f.write(f"- Std: {np.std(boots):.3f}\n")
            f.write(f"- Median: {np.median(boots):.3f}\n")
            f.write(f"- 95th percentile: {np.percentile(boots, 95):.3f}\n")
            f.write(f"- Max: {np.max(boots):.3f}\n")


def run_injection_test(
    n_trials: int = 50,
    n_bootstrap: int = 300,
    n_starts: int = 50,
    outdir: str = "outputs/injection_test",
    seed: int = 12345,
    verbose: bool = True
) -> Dict:
    """
    Run injection/recovery test for both rank-1 true and false scenarios.

    Returns summary with type-I error rate and power.
    """
    # Import injection module
    try:
        from rank1_injection import generate_injection_scenario, load_config, DEFAULT_CONFIG
    except ImportError:
        # Try relative import
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rank1_injection",
            os.path.join(os.path.dirname(__file__), "rank1_injection.py")
        )
        rank1_injection = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rank1_injection)
        generate_injection_scenario = rank1_injection.generate_injection_scenario
        load_config = rank1_injection.load_config
        DEFAULT_CONFIG = rank1_injection.DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    os.makedirs(outdir, exist_ok=True)

    results_true = []
    results_false = []

    if verbose:
        print("=" * 60)
        print("Rank-1 Injection/Recovery Test")
        print("=" * 60)
        print(f"\nRunning {n_trials} trials for each scenario...")
        print(f"Bootstrap: {n_bootstrap}, Starts: {n_starts}")

    # Rank-1 TRUE scenario
    if verbose:
        print(f"\n--- Scenario: RANK-1 TRUE ---")

    for trial in range(n_trials):
        trial_seed = seed + trial
        trial_dir = os.path.join(outdir, "rank1_true", f"trial_{trial:03d}")

        # Generate injection
        info = generate_injection_scenario("rank1_true", config, trial_dir, seed=trial_seed)

        # Load data
        data_A = load_channel_data(info['csv_A'])
        data_B = load_channel_data(info['csv_B'])

        # Run test
        result = run_rank1_test(
            data_A, data_B,
            n_bootstrap=n_bootstrap,
            n_starts=n_starts,
            use_poisson=True,
            dof_diff=2,
            bootstrap_seed=trial_seed,
            verbose=False
        )

        result['trial'] = trial
        result['scenario'] = 'rank1_true'
        result['true_R_A'] = info['R_A_true']
        result['true_R_B'] = info['R_B_true']
        results_true.append(result)

        if verbose and (trial + 1) % 10 == 0:
            print(f"  Completed {trial + 1}/{n_trials} trials")

    # Rank-1 FALSE scenario
    if verbose:
        print(f"\n--- Scenario: RANK-1 FALSE ---")

    for trial in range(n_trials):
        trial_seed = seed + 10000 + trial
        trial_dir = os.path.join(outdir, "rank1_false", f"trial_{trial:03d}")

        # Generate injection
        info = generate_injection_scenario("rank1_false", config, trial_dir, seed=trial_seed)

        # Load data
        data_A = load_channel_data(info['csv_A'])
        data_B = load_channel_data(info['csv_B'])

        # Run test
        result = run_rank1_test(
            data_A, data_B,
            n_bootstrap=n_bootstrap,
            n_starts=n_starts,
            use_poisson=True,
            dof_diff=2,
            bootstrap_seed=trial_seed,
            verbose=False
        )

        result['trial'] = trial
        result['scenario'] = 'rank1_false'
        result['true_R_A'] = info['R_A_true']
        result['true_R_B'] = info['R_B_true']
        results_false.append(result)

        if verbose and (trial + 1) % 10 == 0:
            print(f"  Completed {trial + 1}/{n_trials} trials")

    # Compute summary statistics
    # Type-I error: False rejection when rank-1 is TRUE
    n_false_rejections = sum(1 for r in results_true if r['verdict'] == VERDICT_DISFAVORED)
    type_I_error = n_false_rejections / n_trials

    # Power: Correct rejection when rank-1 is FALSE
    n_correct_rejections = sum(1 for r in results_false if r['verdict'] == VERDICT_DISFAVORED)
    power = n_correct_rejections / n_trials

    # Other statistics
    n_inconclusive_true = sum(1 for r in results_true
                              if r['verdict'] in [VERDICT_INCONCLUSIVE, VERDICT_MODEL_MISMATCH, VERDICT_OPTIMIZER_FAILURE])
    n_inconclusive_false = sum(1 for r in results_false
                               if r['verdict'] in [VERDICT_INCONCLUSIVE, VERDICT_MODEL_MISMATCH, VERDICT_OPTIMIZER_FAILURE])

    summary = {
        'n_trials': n_trials,
        'n_bootstrap': n_bootstrap,
        'n_starts': n_starts,
        'type_I_error': type_I_error,
        'n_false_rejections': n_false_rejections,
        'power': power,
        'n_correct_rejections': n_correct_rejections,
        'n_inconclusive_true': n_inconclusive_true,
        'n_inconclusive_false': n_inconclusive_false,
        'results_true': results_true,
        'results_false': results_false,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("INJECTION/RECOVERY SUMMARY")
        print("=" * 60)
        print(f"\nRank-1 TRUE scenario ({n_trials} trials):")
        print(f"  Type-I error rate: {type_I_error:.3f} ({n_false_rejections}/{n_trials})")
        print(f"  Inconclusive: {n_inconclusive_true}/{n_trials}")

        print(f"\nRank-1 FALSE scenario ({n_trials} trials):")
        print(f"  Power (rejection rate): {power:.3f} ({n_correct_rejections}/{n_trials})")
        print(f"  Inconclusive: {n_inconclusive_false}/{n_trials}")

    # Write report
    report_path = os.path.join(outdir, "INJECTION_REPORT.md")
    write_injection_report(summary, report_path)

    return summary


def write_injection_report(summary: Dict, output_path: str):
    """Write injection/recovery report to markdown."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# Rank-1 Injection/Recovery Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Trials per scenario: {summary['n_trials']}\n")
        f.write(f"- Bootstrap replicates: {summary['n_bootstrap']}\n")
        f.write(f"- Optimizer starts: {summary['n_starts']}\n")

        f.write("\n## Summary Statistics\n\n")
        f.write("| Metric | Value | Notes |\n")
        f.write("|--------|-------|-------|\n")
        f.write(f"| **Type-I Error Rate** | {summary['type_I_error']:.3f} | "
                f"False rejections when rank-1 TRUE ({summary['n_false_rejections']}/{summary['n_trials']}) |\n")
        f.write(f"| **Power** | {summary['power']:.3f} | "
                f"Correct rejections when rank-1 FALSE ({summary['n_correct_rejections']}/{summary['n_trials']}) |\n")
        f.write(f"| Inconclusive (TRUE) | {summary['n_inconclusive_true']}/{summary['n_trials']} | Fit/optimizer issues |\n")
        f.write(f"| Inconclusive (FALSE) | {summary['n_inconclusive_false']}/{summary['n_trials']} | Fit/optimizer issues |\n")

        f.write("\n## Interpretation\n\n")
        f.write("- **Type-I error** should be ~0.05 for a well-calibrated test\n")
        f.write("- **Power** should be high (>0.80) for the test to be useful\n")
        f.write("- High inconclusive rates indicate fit stability issues\n")

        f.write("\n## Rank-1 TRUE Scenario Details\n\n")
        f.write("True parameters: R_A = R_B (shared complex coupling)\n\n")
        f.write("| Verdict | Count | Fraction |\n")
        f.write("|---------|-------|----------|\n")
        verdicts_true = {}
        for r in summary['results_true']:
            v = r['verdict']
            verdicts_true[v] = verdicts_true.get(v, 0) + 1
        for v, c in sorted(verdicts_true.items()):
            f.write(f"| {v} | {c} | {c/summary['n_trials']:.3f} |\n")

        f.write("\n## Rank-1 FALSE Scenario Details\n\n")
        f.write("True parameters: R_A != R_B (different complex couplings)\n\n")
        f.write("| Verdict | Count | Fraction |\n")
        f.write("|---------|-------|----------|\n")
        verdicts_false = {}
        for r in summary['results_false']:
            v = r['verdict']
            verdicts_false[v] = verdicts_false.get(v, 0) + 1
        for v, c in sorted(verdicts_false.items()):
            f.write(f"| {v} | {c} | {c/summary['n_trials']:.3f} |\n")

        # Lambda distributions
        f.write("\n## Lambda Distributions\n\n")
        lambdas_true = [r['Lambda'] for r in summary['results_true']]
        lambdas_false = [r['Lambda'] for r in summary['results_false']]

        f.write("| Scenario | Mean | Std | Median | 95th pct |\n")
        f.write("|----------|------|-----|--------|----------|\n")
        f.write(f"| Rank-1 TRUE | {np.mean(lambdas_true):.2f} | {np.std(lambdas_true):.2f} | "
                f"{np.median(lambdas_true):.2f} | {np.percentile(lambdas_true, 95):.2f} |\n")
        f.write(f"| Rank-1 FALSE | {np.mean(lambdas_false):.2f} | {np.std(lambdas_false):.2f} | "
                f"{np.median(lambdas_false):.2f} | {np.percentile(lambdas_false, 95):.2f} |\n")


def main():
    parser = argparse.ArgumentParser(
        description="CMS Rank-1 Bottleneck Test (Publication Grade v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on data files
  python3 cms_rank1_test.py --channel-a channelA.csv --channel-b channelB.csv

  # Injection/recovery test
  python3 cms_rank1_test.py --inject --bootstrap 300 --outdir outputs/injection

  # Quick run with Wilks only (no bootstrap)
  python3 cms_rank1_test.py --channel-a A.csv --channel-b B.csv --use-wilks-only
        """
    )

    # Input options
    parser.add_argument("--channel-a", help="Channel A CSV file")
    parser.add_argument("--channel-b", help="Channel B CSV file")

    # Output options
    parser.add_argument("--output", help="Output report file (default: RANK1_RESULT.md in outdir)")
    parser.add_argument("--outdir", default=".", help="Output directory")

    # Bootstrap options
    parser.add_argument("--bootstrap", type=int, default=500,
                        help="Number of bootstrap replicates (default: 500)")
    parser.add_argument("--bootstrap-seed", type=int, default=42,
                        help="Random seed for bootstrap")
    parser.add_argument("--bootstrap-workers", type=int, default=None,
                        help="Number of parallel workers (default: auto)")
    parser.add_argument("--use-wilks-only", action="store_true",
                        help="Skip bootstrap, use only Wilks approximation")

    # Optimizer options
    parser.add_argument("--starts", type=int, default=50,
                        help="Number of optimizer starts (default: 50)")
    parser.add_argument("--use-poisson", action="store_true",
                        help="Use Poisson likelihood instead of Gaussian")

    # DOF option
    parser.add_argument("--dof", type=int, default=2,
                        help="Degrees of freedom for constraint (default: 2 for complex R)")

    # Injection mode
    parser.add_argument("--inject", action="store_true",
                        help="Run injection/recovery test")
    parser.add_argument("--inject-trials", type=int, default=50,
                        help="Number of injection trials per scenario")
    parser.add_argument("--inject-config", help="Path to injection config JSON")

    # Legacy compatibility
    parser.add_argument("--generate-mock", action="store_true",
                        help="[Legacy] Generate mock data")
    parser.add_argument("--mock-dir", default="outputs/rank1_inputs",
                        help="[Legacy] Directory for mock data")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # ============================================================
    # Injection mode
    # ============================================================
    if args.inject:
        summary = run_injection_test(
            n_trials=args.inject_trials,
            n_bootstrap=args.bootstrap,
            n_starts=args.starts,
            outdir=args.outdir,
            seed=args.bootstrap_seed,
            verbose=True
        )
        print(f"\nInjection report written to: {os.path.join(args.outdir, 'INJECTION_REPORT.md')}")
        return

    # ============================================================
    # Legacy mock generation
    # ============================================================
    if args.generate_mock:
        print("WARNING: --generate-mock is deprecated. Use --inject for validation.")
        # Simple mock generation for backwards compatibility
        np.random.seed(45)
        os.makedirs(args.mock_dir, exist_ok=True)

        masses = np.linspace(7, 12, 50)
        signal_A = 500 * np.exp(-0.5 * ((masses - 9.4) / 0.15)**2)
        bkg_A = 100 * np.exp(-(masses - 7) / 2)
        counts_A = np.random.poisson(signal_A + bkg_A)
        err_A = np.sqrt(np.maximum(counts_A, 1))

        signal_B = 300 * np.exp(-0.5 * ((masses - 9.45) / 0.18)**2)
        bkg_B = 80 * np.exp(-(masses - 7) / 2.5)
        counts_B = np.random.poisson(signal_B + bkg_B)
        err_B = np.sqrt(np.maximum(counts_B, 1))

        csv_A = os.path.join(args.mock_dir, "channelA.csv")
        csv_B = os.path.join(args.mock_dir, "channelB.csv")

        with open(csv_A, 'w') as f:
            f.write("mass_GeV,counts,stat_err\n")
            for m, c, e in zip(masses, counts_A, err_A):
                f.write(f"{m:.4f},{c},{e:.4f}\n")

        with open(csv_B, 'w') as f:
            f.write("mass_GeV,counts,stat_err\n")
            for m, c, e in zip(masses, counts_B, err_B):
                f.write(f"{m:.4f},{c},{e:.4f}\n")

        print(f"Generated mock data: {csv_A}, {csv_B}")
        args.channel_a = csv_A
        args.channel_b = csv_B

    # ============================================================
    # Standard analysis mode
    # ============================================================
    if not args.channel_a or not args.channel_b:
        print("ERROR: Must specify --channel-a and --channel-b, or use --inject")
        parser.print_help()
        sys.exit(1)

    # Load data
    data_A = load_channel_data(args.channel_a)
    data_B = load_channel_data(args.channel_b)

    if len(data_A) == 0 or len(data_B) == 0:
        print(f"ERROR: Empty data. A has {len(data_A)} bins, B has {len(data_B)} bins")
        sys.exit(1)

    # Run test
    results = run_rank1_test(
        data_A, data_B,
        n_bootstrap=args.bootstrap,
        n_starts=args.starts,
        n_workers=args.bootstrap_workers,
        use_poisson=args.use_poisson,
        use_wilks_only=args.use_wilks_only,
        dof_diff=args.dof,
        bootstrap_seed=args.bootstrap_seed,
        verbose=True
    )

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.outdir, "RANK1_RESULT.md")

    # Write results
    write_result_markdown(results, output_path, args.channel_a, args.channel_b)
    print(f"\nReport written to: {output_path}")

    # Write optimizer audit
    audit_path = os.path.join(args.outdir, "RANK1_OPTIMIZER_AUDIT.md")
    write_optimizer_audit(results, audit_path)
    print(f"Optimizer audit written to: {audit_path}")


if __name__ == "__main__":
    main()
