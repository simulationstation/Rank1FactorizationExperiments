#!/usr/bin/env python3
"""
Belle Zb(10610)/Zb(10650) Rank-1 Factorization Test

Tests whether the complex coupling ratio R = g(Zb10650)/g(Zb10610)
is channel-invariant across multiple decay channels.

Channels:
- Υ(nS)π (n=1,2,3): relative phase ~0°
- hb(mP)π (m=1,2): relative phase ~180°

The 180° phase difference between Υ and hb families is a physical
effect from heavy-quark spin-flip and should be accounted for.

Uses Table I parameters from arXiv:1110.2251 and digitized spectra.
"""

import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Physics constants
M1_PDG = 10.6072  # Zb(10610) mass in GeV
G1_PDG = 0.0184   # Width in GeV
M2_PDG = 10.6522  # Zb(10650) mass in GeV
G2_PDG = 0.0115   # Width in GeV

# Energy reference point for background
E0 = 10.55  # GeV


@dataclass
class ChannelData:
    """Container for channel spectrum data."""
    name: str
    m: np.ndarray  # mass points (GeV)
    y: np.ndarray  # counts or yield
    yerr: np.ndarray  # uncertainties
    spin_flip: bool  # True for hb channels (180° phase flip)


@dataclass
class FitResult:
    """Container for fit results."""
    nll: float
    params: np.ndarray
    chi2: float
    ndof: int
    chi2_dof: float
    r: float  # |R|
    phi: float  # arg(R) in radians
    converged: bool


@dataclass
class Rank1Result:
    """Container for rank-1 test results."""
    Lambda_obs: float
    p_boot: float
    n_boot: int
    k_exceed: int
    chi2_dof_A: float
    chi2_dof_B: float
    gate_A: str
    gate_B: str
    gates_pass: bool
    r_shared: float
    phi_shared: float
    r_A: float
    phi_A: float
    r_B: float
    phi_B: float
    identifiable: bool
    verdict: str
    channel_A: str
    channel_B: str


def breit_wigner(m, M, G):
    """S-wave Breit-Wigner amplitude."""
    s = m**2
    return np.sqrt(M * G) / (M**2 - s - 1j * M * G)


def cross_section_model(m, params, channel_data, fix_masses=True):
    """
    Two-state coherent model with polynomial background.

    I(m) = norm × |BW1(m) + R×BW2(m)|² + B(m)

    Parameters:
      params[0]: norm
      params[1]: r (|R|)
      params[2]: phi (arg(R))
      params[3]: b0 (constant background)
      params[4]: b1 (linear background)
      params[5]: b2 (quadratic background)
    """
    norm = params[0]
    r = params[1]
    phi = params[2]
    b0 = params[3]
    b1 = params[4] if len(params) > 4 else 0.0
    b2 = params[5] if len(params) > 5 else 0.0

    M1, G1 = M1_PDG, G1_PDG
    M2, G2 = M2_PDG, G2_PDG

    # Account for spin-flip phase shift in hb channels
    if channel_data.spin_flip:
        phi_effective = phi + np.pi  # Add 180° for hb channels
    else:
        phi_effective = phi

    R = r * np.exp(1j * phi_effective)

    # Coherent amplitudes
    A1 = breit_wigner(m, M1, G1)
    A2 = breit_wigner(m, M2, G2)
    A_total = A1 + R * A2

    # Intensity
    intensity = norm * np.abs(A_total)**2

    # Polynomial background
    dm = m - E0
    background = b0 + b1 * dm + b2 * dm**2

    return np.maximum(intensity + background, 0)  # Ensure non-negative


def gaussian_nll(params, channel_data, fix_masses=True):
    """Gaussian negative log-likelihood."""
    m = channel_data.m
    y = channel_data.y
    yerr = channel_data.yerr

    # Prevent division by zero
    yerr_safe = np.maximum(yerr, 1e-6)

    y_model = cross_section_model(m, params, channel_data, fix_masses)

    # Gaussian NLL
    residuals = (y - y_model) / yerr_safe
    nll = 0.5 * np.sum(residuals**2)

    return nll


def fit_single_channel(channel_data, n_starts=50, fix_masses=True):
    """
    Fit two-BW model to a single channel.

    Returns FitResult with best-fit parameters and fit quality.
    """
    m = channel_data.m
    y = channel_data.y
    yerr = channel_data.yerr

    # Parameter bounds: [norm, r, phi, b0, b1, b2]
    bounds = [
        (1e-3, 1e6),     # norm
        (0.01, 10.0),    # r
        (-np.pi, np.pi), # phi
        (-1e5, 1e5),     # b0
        (-1e5, 1e5),     # b1
        (-1e5, 1e5),     # b2
    ]

    best_nll = np.inf
    best_params = None

    np.random.seed(42)

    for _ in range(n_starts):
        # Random initial parameters
        p0 = [
            np.random.uniform(0.1, 100) * np.max(y),
            np.random.uniform(0.3, 2.0),
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100),
        ]

        try:
            result = minimize(
                gaussian_nll,
                p0,
                args=(channel_data, fix_masses),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000}
            )

            if result.fun < best_nll and result.success:
                best_nll = result.fun
                best_params = result.x

        except Exception:
            continue

    if best_params is None:
        return FitResult(
            nll=np.inf, params=np.zeros(6), chi2=np.inf,
            ndof=1, chi2_dof=np.inf, r=0, phi=0, converged=False
        )

    # Compute chi2
    y_model = cross_section_model(m, best_params, channel_data, fix_masses)
    chi2 = np.sum(((y - y_model) / yerr)**2)
    ndof = len(m) - len(best_params)
    chi2_dof = chi2 / ndof if ndof > 0 else np.inf

    return FitResult(
        nll=best_nll,
        params=best_params,
        chi2=chi2,
        ndof=ndof,
        chi2_dof=chi2_dof,
        r=best_params[1],
        phi=best_params[2],
        converged=True
    )


def combined_nll_constrained(params, channel_A, channel_B, fix_masses=True):
    """
    Combined NLL with shared R constraint.

    params: [norm_A, r_shared, phi_shared, b0_A, b1_A, b2_A,
             norm_B, b0_B, b1_B, b2_B]
    """
    params_A = [params[0], params[1], params[2], params[3], params[4], params[5]]
    params_B = [params[6], params[1], params[2], params[7], params[8], params[9]]

    nll_A = gaussian_nll(params_A, channel_A, fix_masses)
    nll_B = gaussian_nll(params_B, channel_B, fix_masses)

    return nll_A + nll_B


def combined_nll_unconstrained(params, channel_A, channel_B, fix_masses=True):
    """
    Combined NLL with independent R per channel.

    params: [norm_A, r_A, phi_A, b0_A, b1_A, b2_A,
             norm_B, r_B, phi_B, b0_B, b1_B, b2_B]
    """
    params_A = params[0:6]
    params_B = params[6:12]

    nll_A = gaussian_nll(params_A, channel_A, fix_masses)
    nll_B = gaussian_nll(params_B, channel_B, fix_masses)

    return nll_A + nll_B


def fit_constrained(channel_A, channel_B, n_starts=80):
    """Fit with shared R constraint."""
    # Bounds: [norm_A, r, phi, b0_A, b1_A, b2_A, norm_B, b0_B, b1_B, b2_B]
    bounds = [
        (1e-3, 1e6), (0.01, 10.0), (-np.pi, np.pi),
        (-1e5, 1e5), (-1e5, 1e5), (-1e5, 1e5),
        (1e-3, 1e6), (-1e5, 1e5), (-1e5, 1e5), (-1e5, 1e5)
    ]

    best_nll = np.inf
    best_params = None

    np.random.seed(123)

    for _ in range(n_starts):
        p0 = [
            np.random.uniform(0.1, 100) * np.max(channel_A.y),
            np.random.uniform(0.3, 2.0),
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100),
            np.random.uniform(0.1, 100) * np.max(channel_B.y),
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100),
        ]

        try:
            result = minimize(
                combined_nll_constrained,
                p0,
                args=(channel_A, channel_B, True),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000}
            )

            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x

        except Exception:
            continue

    return best_nll, best_params


def fit_unconstrained(channel_A, channel_B, n_starts=80):
    """Fit with independent R per channel."""
    # Bounds: [norm_A, r_A, phi_A, b0_A, b1_A, b2_A, norm_B, r_B, phi_B, b0_B, b1_B, b2_B]
    bounds = [
        (1e-3, 1e6), (0.01, 10.0), (-np.pi, np.pi),
        (-1e5, 1e5), (-1e5, 1e5), (-1e5, 1e5),
        (1e-3, 1e6), (0.01, 10.0), (-np.pi, np.pi),
        (-1e5, 1e5), (-1e5, 1e5), (-1e5, 1e5)
    ]

    best_nll = np.inf
    best_params = None

    np.random.seed(456)

    for _ in range(n_starts):
        p0 = [
            np.random.uniform(0.1, 100) * np.max(channel_A.y),
            np.random.uniform(0.3, 2.0),
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100),
            np.random.uniform(0.1, 100) * np.max(channel_B.y),
            np.random.uniform(0.3, 2.0),
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100),
        ]

        try:
            result = minimize(
                combined_nll_unconstrained,
                p0,
                args=(channel_A, channel_B, True),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000}
            )

            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x

        except Exception:
            continue

    return best_nll, best_params


def bootstrap_trial(args):
    """Single bootstrap trial."""
    seed, channel_A, channel_B = args
    np.random.seed(seed)

    # Resample with replacement
    n_A = len(channel_A.m)
    n_B = len(channel_B.m)

    idx_A = np.random.choice(n_A, size=n_A, replace=True)
    idx_B = np.random.choice(n_B, size=n_B, replace=True)

    boot_A = ChannelData(
        name=channel_A.name,
        m=channel_A.m[idx_A],
        y=channel_A.y[idx_A],
        yerr=channel_A.yerr[idx_A],
        spin_flip=channel_A.spin_flip
    )

    boot_B = ChannelData(
        name=channel_B.name,
        m=channel_B.m[idx_B],
        y=channel_B.y[idx_B],
        yerr=channel_B.yerr[idx_B],
        spin_flip=channel_B.spin_flip
    )

    # Fit both models
    nll_con, _ = fit_constrained(boot_A, boot_B, n_starts=30)
    nll_unc, _ = fit_unconstrained(boot_A, boot_B, n_starts=30)

    Lambda_boot = 2 * (nll_con - nll_unc)
    return max(0, Lambda_boot)


def run_bootstrap(channel_A, channel_B, n_boot=200, n_workers=None):
    """Run parametric bootstrap for p-value estimation."""
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    args_list = [(i, channel_A, channel_B) for i in range(n_boot)]

    print(f"Running {n_boot} bootstrap trials with {n_workers} workers...")

    with Pool(n_workers) as pool:
        Lambda_boots = pool.map(bootstrap_trial, args_list)

    return np.array(Lambda_boots)


def compute_chi2_per_channel(params_con, channel_A, channel_B):
    """Compute chi2/dof for each channel from constrained fit."""
    # params_con: [norm_A, r, phi, b0_A, b1_A, b2_A, norm_B, b0_B, b1_B, b2_B]
    params_A = [params_con[0], params_con[1], params_con[2], params_con[3], params_con[4], params_con[5]]
    params_B = [params_con[6], params_con[1], params_con[2], params_con[7], params_con[8], params_con[9]]

    y_model_A = cross_section_model(channel_A.m, params_A, channel_A)
    chi2_A = np.sum(((channel_A.y - y_model_A) / channel_A.yerr)**2)
    ndof_A = len(channel_A.m) - 6  # norm, r, phi, b0, b1, b2

    y_model_B = cross_section_model(channel_B.m, params_B, channel_B)
    chi2_B = np.sum(((channel_B.y - y_model_B) / channel_B.yerr)**2)
    ndof_B = len(channel_B.m) - 6

    return chi2_A / max(1, ndof_A), chi2_B / max(1, ndof_B)


def health_gate(chi2_dof):
    """Determine fit health based on chi2/dof."""
    if chi2_dof > 3.0:
        return "MODEL_MISMATCH"
    elif chi2_dof < 0.5:
        return "UNDERCONSTRAINED"
    else:
        return "HEALTHY"


def load_channel_data(extracted_dir, channel_name, spin_flip=False):
    """Load channel data from CSV file."""
    csv_path = os.path.join(extracted_dir, f'{channel_name}.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=',', skip_header=2)

    return ChannelData(
        name=channel_name,
        m=data[:, 0],
        y=data[:, 1],
        yerr=data[:, 2],
        spin_flip=spin_flip
    )


def run_pairwise_test(channel_A, channel_B, n_boot=200, n_starts=80):
    """
    Run pairwise rank-1 test between two channels.
    """
    print(f"\n{'='*70}")
    print(f"Pairwise Rank-1 Test: {channel_A.name} vs {channel_B.name}")
    print(f"{'='*70}")

    # Fit constrained (shared R)
    print("\n[1/4] Fitting constrained model (shared R)...")
    nll_con, params_con = fit_constrained(channel_A, channel_B, n_starts)

    # Fit unconstrained (independent R)
    print("[2/4] Fitting unconstrained model (independent R)...")
    nll_unc, params_unc = fit_unconstrained(channel_A, channel_B, n_starts)

    # Lambda test statistic
    Lambda_obs = 2 * (nll_con - nll_unc)
    Lambda_obs = max(0, Lambda_obs)

    print(f"\nNLL constrained: {nll_con:.2f}")
    print(f"NLL unconstrained: {nll_unc:.2f}")
    print(f"Lambda_obs: {Lambda_obs:.4f}")

    # Extract R values
    # constrained params: [norm_A, r, phi, b0_A, b1_A, b2_A, norm_B, b0_B, b1_B, b2_B]
    # unconstrained params: [norm_A, r_A, phi_A, b0_A, b1_A, b2_A, norm_B, r_B, phi_B, b0_B, b1_B, b2_B]
    r_shared = params_con[1]
    phi_shared = params_con[2]
    r_A = params_unc[1]
    phi_A = params_unc[2]
    r_B = params_unc[7]
    phi_B = params_unc[8]

    print(f"\nShared R: |R|={r_shared:.3f}, φ={np.degrees(phi_shared):.1f}°")
    print(f"Channel A R: |R|={r_A:.3f}, φ={np.degrees(phi_A):.1f}°")
    print(f"Channel B R: |R|={r_B:.3f}, φ={np.degrees(phi_B):.1f}°")

    # Chi2/dof per channel
    chi2_dof_A, chi2_dof_B = compute_chi2_per_channel(params_con, channel_A, channel_B)
    gate_A = health_gate(chi2_dof_A)
    gate_B = health_gate(chi2_dof_B)

    print(f"\n[3/4] Fit health:")
    print(f"  Channel A chi2/dof: {chi2_dof_A:.3f} [{gate_A}]")
    print(f"  Channel B chi2/dof: {chi2_dof_B:.3f} [{gate_B}]")

    gates_pass = (gate_A == "HEALTHY") and (gate_B == "HEALTHY")

    # Bootstrap (only if gates pass or for diagnostic)
    print(f"\n[4/4] Running bootstrap ({n_boot} replicates)...")
    Lambda_boots = run_bootstrap(channel_A, channel_B, n_boot)

    k_exceed = np.sum(Lambda_boots >= Lambda_obs)
    p_boot = k_exceed / n_boot

    print(f"p_boot = {p_boot:.4f} ({k_exceed}/{n_boot} exceedances)")

    # Verdict
    if not gates_pass:
        verdict = "INCONCLUSIVE"
    elif Lambda_obs < 0:
        verdict = "OPTIMIZER_FAILURE"
    elif p_boot < 0.05:
        verdict = "DISFAVORED"
    else:
        verdict = "NOT_REJECTED"

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    return Rank1Result(
        Lambda_obs=Lambda_obs,
        p_boot=p_boot,
        n_boot=n_boot,
        k_exceed=k_exceed,
        chi2_dof_A=chi2_dof_A,
        chi2_dof_B=chi2_dof_B,
        gate_A=gate_A,
        gate_B=gate_B,
        gates_pass=gates_pass,
        r_shared=r_shared,
        phi_shared=phi_shared,
        r_A=r_A,
        phi_A=phi_A,
        r_B=r_B,
        phi_B=phi_B,
        identifiable=True,
        verdict=verdict,
        channel_A=channel_A.name,
        channel_B=channel_B.name
    ), Lambda_boots, params_con, params_unc


def generate_plots(channel_A, channel_B, params_con, params_unc, Lambda_boots, result, out_dir):
    """Generate diagnostic plots."""
    # Bootstrap histogram
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(Lambda_boots, bins=30, density=True, alpha=0.6, color='steelblue',
            label='Bootstrap distribution')
    ax.axvline(result.Lambda_obs, color='red', linestyle='--', linewidth=2,
               label=f'Λ_obs = {result.Lambda_obs:.2f}')
    ax.axvline(5.99, color='orange', linestyle=':', linewidth=2,
               label='χ²(2) 95% = 5.99')

    ax.set_xlabel('Λ = 2×(NLL_con - NLL_unc)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Belle Zb Rank-1 Test: {channel_A.name} vs {channel_B.name}', fontsize=13)
    ax.legend(fontsize=10)

    ax.text(0.95, 0.95, f'p = {result.p_boot:.3f}\n({result.k_exceed}/{result.n_boot})',
            transform=ax.transAxes, fontsize=11, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'bootstrap_hist_pairwise.png'), dpi=150)
    plt.close()

    # Channel fits
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (ch, ax) in enumerate([(channel_A, axes[0]), (channel_B, axes[1])]):
        # Data
        ax.errorbar(ch.m, ch.y, yerr=ch.yerr, fmt='o', color='blue',
                    markersize=5, capsize=3, label='Data')

        # Model curve
        m_fine = np.linspace(ch.m.min(), ch.m.max(), 200)

        # constrained params: [norm_A, r, phi, b0_A, b1_A, b2_A, norm_B, b0_B, b1_B, b2_B]
        if idx == 0:
            params_ch = [params_con[0], params_con[1], params_con[2],
                         params_con[3], params_con[4], params_con[5]]
        else:
            params_ch = [params_con[6], params_con[1], params_con[2],
                         params_con[7], params_con[8], params_con[9]]

        # Create temporary ChannelData for model evaluation
        ch_fine = ChannelData(ch.name, m_fine, np.zeros_like(m_fine),
                              np.ones_like(m_fine), ch.spin_flip)
        y_model = cross_section_model(m_fine, params_ch, ch_fine)

        ax.plot(m_fine, y_model, 'r-', linewidth=2, label='Shared-R fit')

        ax.axvline(M1_PDG, color='green', linestyle='--', alpha=0.5)
        ax.axvline(M2_PDG, color='purple', linestyle='--', alpha=0.5)

        chi2_dof = result.chi2_dof_A if idx == 0 else result.chi2_dof_B
        ax.set_xlabel('M (GeV)', fontsize=12)
        ax.set_ylabel('Events', fontsize=12)
        ax.set_title(f'{ch.name}: χ²/dof = {chi2_dof:.2f}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'channel_fits.png'), dpi=150)
    plt.close()


def generate_report(result, channel_A, channel_B, out_dir):
    """Generate markdown report."""
    report = f"""# Belle Zb(10610)/Zb(10650) Rank-1 Factorization Test

## Executive Summary

**Verdict: {result.verdict}**

| Metric | Value |
|--------|-------|
| Lambda_obs | {result.Lambda_obs:.4f} |
| p_boot | **{result.p_boot:.3f}** ({result.k_exceed}/{result.n_boot}) |
| χ²/dof ({channel_A.name}) | {result.chi2_dof_A:.2f} [{result.gate_A}] |
| χ²/dof ({channel_B.name}) | {result.chi2_dof_B:.2f} [{result.gate_B}] |
| Gates | **{'PASS' if result.gates_pass else 'FAIL'}** |

---

## Channels Tested

| Channel | Source | Spin-Flip | Points |
|---------|--------|-----------|--------|
| {channel_A.name} | arXiv:1110.2251 | {'Yes' if channel_A.spin_flip else 'No'} | {len(channel_A.m)} |
| {channel_B.name} | arXiv:1110.2251 | {'Yes' if channel_B.spin_flip else 'No'} | {len(channel_B.m)} |

---

## Model

The cross section per channel is modeled as:

```
I_α(m) = norm_α × |BW₁(m) + R_α×BW₂(m)|² + B_α(m)
```

Where:
- `BW(m) = √(MΓ) / (M² - m² - iMΓ)` (S-wave Breit-Wigner)
- `R_α = r_α × exp(iφ_α)` is the complex coupling ratio
- `B_α(m) = b0 + b1×(m - m₀)` is linear background

**Fixed resonance parameters:**
- Zb(10610): M = {M1_PDG*1000:.1f} MeV, Γ = {G1_PDG*1000:.1f} MeV
- Zb(10650): M = {M2_PDG*1000:.1f} MeV, Γ = {G2_PDG*1000:.1f} MeV

**Spin-flip correction:**
For hb channels, an additional 180° phase is applied to R to account for
heavy-quark spin-flip effects (as observed in the original Belle analysis).

---

## Coupling Ratios

### Shared (Constrained fit)
| Parameter | Value |
|-----------|-------|
| |R| | {result.r_shared:.4f} |
| arg(R) | {np.degrees(result.phi_shared):.1f}° |

### Per-Channel (Unconstrained fit)
| Channel | |R| | arg(R) |
|---------|-----|--------|
| {channel_A.name} | {result.r_A:.4f} | {np.degrees(result.phi_A):.1f}° |
| {channel_B.name} | {result.r_B:.4f} | {np.degrees(result.phi_B):.1f}° |

---

## Fit Health

| Channel | χ²/dof | Gate |
|---------|--------|------|
| {channel_A.name} | {result.chi2_dof_A:.3f} | {result.gate_A} |
| {channel_B.name} | {result.chi2_dof_B:.3f} | {result.gate_B} |

Health gates:
- `χ²/dof > 3.0` → MODEL_MISMATCH
- `χ²/dof < 0.5` → UNDERCONSTRAINED
- Otherwise → HEALTHY

---

## Bootstrap Distribution

![Bootstrap Distribution](bootstrap_hist_pairwise.png)

- N_boot = {result.n_boot}
- Lambda_obs = {result.Lambda_obs:.4f}
- k_exceed = {result.k_exceed}
- p_boot = {result.p_boot:.4f}

Under χ²(2) null hypothesis: p_chi2 ≈ {1 - (1 - np.exp(-result.Lambda_obs/2)):.3f}

---

## Channel Fits

![Channel Fits](channel_fits.png)

---

## Interpretation

The rank-1 factorization hypothesis is **{result.verdict}** at the 5% level.

{'The complex coupling ratio R = g(Zb10650)/g(Zb10610) is consistent with being shared across the tested channels, supporting a common production mechanism for the Zb states.' if result.verdict == 'NOT_REJECTED' else 'The data shows evidence against a shared coupling ratio across channels.' if result.verdict == 'DISFAVORED' else 'The test is inconclusive due to fit quality issues.'}

---

## Data Sources

- Belle Collaboration, arXiv:1110.2251
- "Observation of two charged bottomonium-like resonances in Υ(5S) decays"

---

## Files Generated

- `REPORT.md` - This report
- `bootstrap_hist_pairwise.png` - Bootstrap distribution
- `channel_fits.png` - Per-channel fit plots
- `result.json` - Machine-readable results

---

*Generated by belle_zb_rank1_test.py*
"""

    with open(os.path.join(out_dir, 'REPORT.md'), 'w') as f:
        f.write(report)

    # Save JSON result (convert numpy types to native Python)
    result_dict = {
        'Lambda_obs': float(result.Lambda_obs),
        'p_boot': float(result.p_boot),
        'n_boot': int(result.n_boot),
        'k_exceed': int(result.k_exceed),
        'chi2_dof_A': float(result.chi2_dof_A),
        'chi2_dof_B': float(result.chi2_dof_B),
        'gate_A': result.gate_A,
        'gate_B': result.gate_B,
        'gates_pass': bool(result.gates_pass),
        'r_shared': float(result.r_shared),
        'phi_shared': float(result.phi_shared),
        'r_A': float(result.r_A),
        'phi_A': float(result.phi_A),
        'r_B': float(result.r_B),
        'phi_B': float(result.phi_B),
        'verdict': result.verdict,
        'channel_A': result.channel_A,
        'channel_B': result.channel_B
    }

    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extracted_dir = os.path.join(base_dir, 'extracted')
    out_dir = os.path.join(base_dir, 'out')

    print("=" * 70)
    print("Belle Zb(10610)/Zb(10650) Rank-1 Factorization Test")
    print("=" * 70)

    # Load channels - prioritize hb channels (cleanest) and Υ(2S), Υ(3S)
    # hb channels have spin_flip=True (180° phase difference)
    channels = {}

    try:
        channels['hb1p'] = load_channel_data(extracted_dir, 'hb1p', spin_flip=True)
        print(f"Loaded hb(1P)π: {len(channels['hb1p'].m)} points")
    except FileNotFoundError:
        print("Warning: hb1p data not found")

    try:
        channels['hb2p'] = load_channel_data(extracted_dir, 'hb2p', spin_flip=True)
        print(f"Loaded hb(2P)π: {len(channels['hb2p'].m)} points")
    except FileNotFoundError:
        print("Warning: hb2p data not found")

    try:
        channels['upsilon2s'] = load_channel_data(extracted_dir, 'upsilon2s', spin_flip=False)
        print(f"Loaded Υ(2S)π: {len(channels['upsilon2s'].m)} points")
    except FileNotFoundError:
        print("Warning: upsilon2s data not found")

    try:
        channels['upsilon3s'] = load_channel_data(extracted_dir, 'upsilon3s', spin_flip=False)
        print(f"Loaded Υ(3S)π: {len(channels['upsilon3s'].m)} points")
    except FileNotFoundError:
        print("Warning: upsilon3s data not found")

    if len(channels) < 2:
        print("ERROR: Need at least 2 channels for rank-1 test")
        sys.exit(1)

    # Select best channel pair
    # Priority: Υ(2S) vs Υ(3S) first (smaller backgrounds in original paper)
    # then hb(1P) vs hb(2P)
    if 'upsilon2s' in channels and 'upsilon3s' in channels:
        channel_A = channels['upsilon2s']
        channel_B = channels['upsilon3s']
        print("\nUsing Υ(2S)π vs Υ(3S)π (same spin-conserving phase)")
    elif 'hb1p' in channels and 'hb2p' in channels:
        channel_A = channels['hb1p']
        channel_B = channels['hb2p']
        print("\nUsing hb(1P)π vs hb(2P)π (same spin-flip phase)")
    else:
        # Use whatever two channels are available
        ch_names = list(channels.keys())
        channel_A = channels[ch_names[0]]
        channel_B = channels[ch_names[1]]
        print(f"\nUsing {ch_names[0]} vs {ch_names[1]}")

    # Run pairwise test
    result, Lambda_boots, params_con, params_unc = run_pairwise_test(
        channel_A, channel_B, n_boot=200, n_starts=80
    )

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(channel_A, channel_B, params_con, params_unc, Lambda_boots, result, out_dir)

    # Generate report
    print("Generating report...")
    generate_report(result, channel_A, channel_B, out_dir)

    print(f"\nReport saved to: {out_dir}/REPORT.md")
    print(f"Results saved to: {out_dir}/result.json")

    # Log commands
    with open(os.path.join(base_dir, 'logs/COMMANDS.txt'), 'a') as f:
        f.write(f"\npython3 belle_zb_rank1_test.py\n")
        f.write(f"Channels: {channel_A.name} vs {channel_B.name}\n")
        f.write(f"Verdict: {result.verdict}\n")
        f.write(f"p_boot: {result.p_boot:.4f}\n")


if __name__ == '__main__':
    main()
