#!/usr/bin/env python3
"""
Global Multi-Channel Rank-1 Test (NEW)

Implements comprehensive rank-1 tests across >=3 channels simultaneously:

(A) Global Multi-Channel Shared-R Test:
    For a 2-state subspace {X1, X2} across Nc channels:
    - Constrained: Single shared R = g2/g1 across ALL channels (2 dof total)
    - Unconstrained: Independent R_alpha per channel (2*Nc dof total)
    - Test statistic: Lambda = 2*(NLL_con - NLL_unc), dof_diff = 2*(Nc-1)

This extends the existing pairwise tests to provide stronger constraints.

Author: Automated extension of rank-1 harness
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2 as chi2_dist
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================
NESTED_MODEL_TOL = 1e-4
CHI2_DOF_LOW = 0.5
CHI2_DOF_HIGH = 3.0


@dataclass
class ChannelData:
    """Container for channel spectrum data."""
    name: str
    m: np.ndarray      # mass points (GeV)
    y: np.ndarray      # counts or yield
    yerr: np.ndarray   # uncertainties
    spin_flip: bool = False  # True for channels with 180deg phase shift


@dataclass
class MultiChannelResult:
    """Container for multi-channel test results."""
    test_type: str          # "global_shared_R"
    n_channels: int
    channel_names: List[str]
    Lambda_obs: float
    Lambda_raw: float
    dof_diff: int
    p_boot: float
    p_wilks: float
    n_boot: int
    k_exceed: int
    nll_con: float
    nll_unc: float
    R_shared: Tuple[float, float]  # (r, phi_deg)
    R_per_channel: Dict[str, Tuple[float, float]]  # {name: (r, phi_deg)}
    chi2_dof_per_channel: Dict[str, float]
    health_per_channel: Dict[str, str]
    invariant_holds: bool
    overall_health: str
    verdict: str
    reason: str


# =============================================================================
# Core Breit-Wigner Model
# =============================================================================

def breit_wigner(m: np.ndarray, M: float, Gamma: float) -> np.ndarray:
    """S-wave Breit-Wigner amplitude."""
    s = m**2
    return np.sqrt(M * Gamma) / (M**2 - s - 1j * M * Gamma)


def coherent_model(m: np.ndarray, params: List[float], M1: float, G1: float,
                   M2: float, G2: float, spin_flip: bool = False,
                   m_ref: float = 10.55) -> np.ndarray:
    """
    Two-state coherent amplitude model.

    I(m) = norm × |BW1 + R×BW2|² + background

    params: [norm, r, phi, b0, b1, b2]
    """
    norm = params[0]
    r = params[1]
    phi = params[2]
    b0 = params[3]
    b1 = params[4] if len(params) > 4 else 0.0
    b2 = params[5] if len(params) > 5 else 0.0

    # Apply spin-flip phase if needed
    if spin_flip:
        phi = phi + np.pi

    R = r * np.exp(1j * phi)

    BW1 = breit_wigner(m, M1, G1)
    BW2 = breit_wigner(m, M2, G2)

    amplitude = BW1 + R * BW2
    signal = norm * np.abs(amplitude)**2

    # Polynomial background
    dm = m - m_ref
    background = b0 + b1 * dm + b2 * dm**2

    return np.maximum(signal + background, 1e-10)


def gaussian_nll(params: List[float], channel: ChannelData,
                 M1: float, G1: float, M2: float, G2: float,
                 m_ref: float = 10.55) -> float:
    """Gaussian NLL for single channel."""
    y_pred = coherent_model(channel.m, params, M1, G1, M2, G2,
                            channel.spin_flip, m_ref)
    residuals = (channel.y - y_pred) / np.maximum(channel.yerr, 1e-6)
    return 0.5 * np.sum(residuals**2)


# =============================================================================
# Multi-Channel Joint Fitting
# =============================================================================

def multichannel_nll_constrained(params: np.ndarray, channels: List[ChannelData],
                                  M1: float, G1: float, M2: float, G2: float,
                                  m_ref: float = 10.55) -> float:
    """
    Combined NLL with ONE shared R across all channels.

    Parameter layout:
        [r_shared, phi_shared,
         norm_0, b0_0, b1_0, b2_0,
         norm_1, b0_1, b1_1, b2_1,
         ...]

    Total params: 2 + 4*Nc
    """
    r_shared = params[0]
    phi_shared = params[1]

    total_nll = 0.0
    for i, ch in enumerate(channels):
        offset = 2 + i * 4
        ch_params = [
            params[offset],      # norm
            r_shared,            # shared r
            phi_shared,          # shared phi
            params[offset + 1],  # b0
            params[offset + 2],  # b1
            params[offset + 3],  # b2
        ]
        total_nll += gaussian_nll(ch_params, ch, M1, G1, M2, G2, m_ref)

    return total_nll


def multichannel_nll_unconstrained(params: np.ndarray, channels: List[ChannelData],
                                    M1: float, G1: float, M2: float, G2: float,
                                    m_ref: float = 10.55) -> float:
    """
    Combined NLL with independent R per channel.

    Parameter layout:
        [norm_0, r_0, phi_0, b0_0, b1_0, b2_0,
         norm_1, r_1, phi_1, b0_1, b1_1, b2_1,
         ...]

    Total params: 6*Nc
    """
    total_nll = 0.0
    for i, ch in enumerate(channels):
        offset = i * 6
        ch_params = params[offset:offset + 6]
        total_nll += gaussian_nll(ch_params, ch, M1, G1, M2, G2, m_ref)

    return total_nll


def fit_multichannel_constrained(channels: List[ChannelData],
                                  M1: float, G1: float, M2: float, G2: float,
                                  m_ref: float = 10.55,
                                  n_starts: int = 100,
                                  seed: int = 42) -> Tuple[float, Optional[np.ndarray]]:
    """Fit with single shared R across all channels."""
    Nc = len(channels)

    # Bounds: [r_shared, phi_shared, (norm, b0, b1, b2) * Nc]
    bounds = [
        (0.01, 10.0),         # r_shared
        (-np.pi, np.pi),      # phi_shared
    ]
    for ch in channels:
        y_max = np.max(ch.y)
        bounds.extend([
            (1e-3, 1e6 * max(1, y_max)),  # norm
            (-1e5, 1e5),                   # b0
            (-1e5, 1e5),                   # b1
            (-1e5, 1e5),                   # b2
        ])

    best_nll = np.inf
    best_params = None

    rng = np.random.default_rng(seed)

    def nll(p):
        return multichannel_nll_constrained(p, channels, M1, G1, M2, G2, m_ref)

    # Global optimization first
    try:
        result = differential_evolution(nll, bounds, maxiter=200, seed=seed,
                                        polish=True, workers=1)
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x.copy()
    except Exception:
        pass

    # Multi-start local optimization
    for _ in range(n_starts):
        x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
        try:
            result = minimize(nll, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 2000})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    return best_nll, best_params


def fit_multichannel_unconstrained(channels: List[ChannelData],
                                    M1: float, G1: float, M2: float, G2: float,
                                    m_ref: float = 10.55,
                                    n_starts: int = 100,
                                    seed: int = 43,
                                    init_from_constrained: Optional[np.ndarray] = None
                                    ) -> Tuple[float, Optional[np.ndarray]]:
    """Fit with independent R per channel."""
    Nc = len(channels)

    # Bounds: [(norm, r, phi, b0, b1, b2) * Nc]
    bounds = []
    for ch in channels:
        y_max = np.max(ch.y)
        bounds.extend([
            (1e-3, 1e6 * max(1, y_max)),  # norm
            (0.01, 10.0),                  # r
            (-np.pi, np.pi),               # phi
            (-1e5, 1e5),                   # b0
            (-1e5, 1e5),                   # b1
            (-1e5, 1e5),                   # b2
        ])

    best_nll = np.inf
    best_params = None

    rng = np.random.default_rng(seed)

    def nll(p):
        return multichannel_nll_unconstrained(p, channels, M1, G1, M2, G2, m_ref)

    # Initialize from constrained if available (ensures unconstrained <= constrained)
    if init_from_constrained is not None:
        # Convert constrained params to unconstrained format
        # Constrained: [r_shared, phi_shared, (norm, b0, b1, b2) * Nc]
        # Unconstrained: [(norm, r, phi, b0, b1, b2) * Nc]
        r_shared = init_from_constrained[0]
        phi_shared = init_from_constrained[1]
        x0_from_con = []
        for i in range(Nc):
            offset = 2 + i * 4
            x0_from_con.extend([
                init_from_constrained[offset],      # norm
                r_shared,                            # r (same as shared)
                phi_shared,                          # phi (same as shared)
                init_from_constrained[offset + 1],  # b0
                init_from_constrained[offset + 2],  # b1
                init_from_constrained[offset + 3],  # b2
            ])

        try:
            result = minimize(nll, x0_from_con, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 3000})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Global optimization
    try:
        result = differential_evolution(nll, bounds, maxiter=200, seed=seed,
                                        polish=True, workers=1)
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x.copy()
    except Exception:
        pass

    # Multi-start
    for _ in range(n_starts):
        x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
        try:
            result = minimize(nll, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 2000})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    return best_nll, best_params


# =============================================================================
# Bootstrap for Multi-Channel Test
# =============================================================================

def _bootstrap_worker_multichannel(args):
    """Single bootstrap replicate for multi-channel test."""
    channels, seed, M1, G1, M2, G2, m_ref, params_con = args

    rng = np.random.default_rng(seed)
    Nc = len(channels)

    # Extract constrained params
    r_shared = params_con[0]
    phi_shared = params_con[1]

    # Generate pseudo-data for each channel
    pseudo_channels = []
    for i, ch in enumerate(channels):
        offset = 2 + i * 4
        ch_params = [
            params_con[offset],      # norm
            r_shared,
            phi_shared,
            params_con[offset + 1],  # b0
            params_con[offset + 2],  # b1
            params_con[offset + 3],  # b2
        ]
        mu = coherent_model(ch.m, ch_params, M1, G1, M2, G2,
                           ch.spin_flip, m_ref)
        y_new = rng.normal(mu, np.maximum(ch.yerr, 1e-6))
        pseudo_channels.append(ChannelData(
            name=ch.name, m=ch.m, y=y_new, yerr=ch.yerr, spin_flip=ch.spin_flip
        ))

    # Fit constrained
    nll_con, params_con_boot = fit_multichannel_constrained(
        pseudo_channels, M1, G1, M2, G2, m_ref, n_starts=30, seed=seed
    )

    # Fit unconstrained (init from constrained)
    nll_unc, _ = fit_multichannel_unconstrained(
        pseudo_channels, M1, G1, M2, G2, m_ref, n_starts=40, seed=seed + 1000,
        init_from_constrained=params_con_boot
    )

    # Check invariant
    if nll_unc > nll_con + NESTED_MODEL_TOL:
        return {'failed': True, 'violation': nll_unc - nll_con}

    Lambda = max(0.0, 2 * (nll_con - nll_unc))
    return {'failed': False, 'Lambda': Lambda}


def run_bootstrap_multichannel(channels: List[ChannelData], params_con: np.ndarray,
                                M1: float, G1: float, M2: float, G2: float,
                                m_ref: float = 10.55,
                                n_boot: int = 100) -> Dict[str, Any]:
    """Run bootstrap for multi-channel test."""
    n_workers = max(1, cpu_count() - 1)

    args_list = [(channels, i, M1, G1, M2, G2, m_ref, params_con) for i in range(n_boot)]

    lambda_boots = []
    n_failed = 0

    with Pool(n_workers) as pool:
        results = pool.map(_bootstrap_worker_multichannel, args_list)

    for res in results:
        if res['failed']:
            n_failed += 1
        else:
            lambda_boots.append(res['Lambda'])

    return {
        'lambda_boots': lambda_boots,
        'n_valid': len(lambda_boots),
        'n_failed': n_failed,
    }


# =============================================================================
# Main Multi-Channel Test Function
# =============================================================================

def run_global_multichannel_test(
    channels: List[ChannelData],
    M1: float, G1: float, M2: float, G2: float,
    m_ref: float = 10.55,
    n_boot: int = 100,
    n_starts: int = 100
) -> MultiChannelResult:
    """
    Run global multi-channel shared-R rank-1 test.

    Tests whether a single complex ratio R = g2/g1 is shared across
    all Nc channels, vs independent R per channel.

    dof_diff = 2*(Nc - 1) for Nc channels
    """
    Nc = len(channels)
    if Nc < 2:
        raise ValueError("Need at least 2 channels for multi-channel test")

    channel_names = [ch.name for ch in channels]
    dof_diff = 2 * (Nc - 1)

    logger.info(f"Running global {Nc}-channel shared-R test")
    logger.info(f"Channels: {channel_names}")
    logger.info(f"dof_diff = {dof_diff}")

    # Fit constrained (shared R)
    logger.info("Fitting constrained model (shared R)...")
    nll_con, params_con = fit_multichannel_constrained(
        channels, M1, G1, M2, G2, m_ref, n_starts=n_starts
    )

    if params_con is None:
        return MultiChannelResult(
            test_type="global_shared_R",
            n_channels=Nc,
            channel_names=channel_names,
            Lambda_obs=np.nan,
            Lambda_raw=np.nan,
            dof_diff=dof_diff,
            p_boot=np.nan,
            p_wilks=np.nan,
            n_boot=0,
            k_exceed=0,
            nll_con=np.inf,
            nll_unc=np.inf,
            R_shared=(np.nan, np.nan),
            R_per_channel={},
            chi2_dof_per_channel={},
            health_per_channel={},
            invariant_holds=False,
            overall_health="FIT_FAILED",
            verdict="OPTIMIZER_FAILURE",
            reason="Constrained fit failed"
        )

    r_shared = params_con[0]
    phi_shared = params_con[1]

    # Fit unconstrained (independent R per channel)
    logger.info("Fitting unconstrained model (independent R)...")
    nll_unc, params_unc = fit_multichannel_unconstrained(
        channels, M1, G1, M2, G2, m_ref, n_starts=n_starts,
        init_from_constrained=params_con
    )

    if params_unc is None:
        nll_unc = nll_con  # Fallback

    # Check nested invariant
    invariant_holds = nll_unc <= nll_con + NESTED_MODEL_TOL

    # Lambda statistic
    Lambda_raw = 2 * (nll_con - nll_unc)
    Lambda_obs = max(0.0, Lambda_raw) if invariant_holds else np.nan

    # Extract per-channel R from unconstrained fit
    R_per_channel = {}
    if params_unc is not None:
        for i, ch in enumerate(channels):
            offset = i * 6
            r_i = params_unc[offset + 1]
            phi_i = params_unc[offset + 2]
            R_per_channel[ch.name] = (r_i, np.degrees(phi_i))

    # Per-channel chi2/dof (from constrained fit)
    chi2_dof_per_channel = {}
    health_per_channel = {}
    for i, ch in enumerate(channels):
        offset = 2 + i * 4
        ch_params = [
            params_con[offset], r_shared, phi_shared,
            params_con[offset + 1], params_con[offset + 2], params_con[offset + 3]
        ]
        y_pred = coherent_model(ch.m, ch_params, M1, G1, M2, G2,
                               ch.spin_flip, m_ref)
        chi2 = np.sum(((ch.y - y_pred) / np.maximum(ch.yerr, 1e-6))**2)
        ndof = max(1, len(ch.m) - 4)  # 4 free params per channel (norm, b0, b1, b2)
        chi2_dof = chi2 / ndof
        chi2_dof_per_channel[ch.name] = chi2_dof

        if chi2_dof < CHI2_DOF_LOW:
            health_per_channel[ch.name] = "UNDERCONSTRAINED"
        elif chi2_dof > CHI2_DOF_HIGH:
            health_per_channel[ch.name] = "MODEL_MISMATCH"
        else:
            health_per_channel[ch.name] = "HEALTHY"

    # Overall health
    health_values = list(health_per_channel.values())
    if all(h == "HEALTHY" for h in health_values):
        overall_health = "HEALTHY"
    elif any(h == "MODEL_MISMATCH" for h in health_values):
        overall_health = "MODEL_MISMATCH"
    else:
        overall_health = "UNDERCONSTRAINED"

    # p-values
    if not invariant_holds:
        p_wilks = np.nan
        p_boot = np.nan
        k_exceed = 0
        n_boot_actual = 0
        verdict = "OPTIMIZER_FAILURE"
        reason = f"Nested invariant violated: nll_unc ({nll_unc:.2f}) > nll_con ({nll_con:.2f})"
    else:
        # Wilks p-value (reference)
        p_wilks = 1 - chi2_dist.cdf(Lambda_obs, dof_diff)

        # Bootstrap
        if n_boot > 0:
            logger.info(f"Running bootstrap ({n_boot} replicates)...")
            boot_results = run_bootstrap_multichannel(
                channels, params_con, M1, G1, M2, G2, m_ref, n_boot
            )
            lambda_boots = boot_results['lambda_boots']
            n_boot_actual = boot_results['n_valid']

            if n_boot_actual > 0:
                k_exceed = sum(lb >= Lambda_obs for lb in lambda_boots)
                p_boot = (1 + k_exceed) / (1 + n_boot_actual)
            else:
                k_exceed = 0
                p_boot = np.nan
        else:
            k_exceed = 0
            p_boot = np.nan
            n_boot_actual = 0

        # Verdict
        if overall_health != "HEALTHY":
            verdict = "INCONCLUSIVE"
            reason = f"Fit health issues: {health_per_channel}"
        elif np.isnan(p_boot):
            verdict = "NO_BOOTSTRAP"
            reason = "Bootstrap failed"
        elif p_boot >= 0.05:
            verdict = "NOT_REJECTED"
            reason = f"p_boot = {p_boot:.3f} >= 0.05 (global {Nc}-channel test)"
        else:
            verdict = "DISFAVORED"
            reason = f"p_boot = {p_boot:.3f} < 0.05 (global {Nc}-channel test)"

    return MultiChannelResult(
        test_type="global_shared_R",
        n_channels=Nc,
        channel_names=channel_names,
        Lambda_obs=Lambda_obs,
        Lambda_raw=Lambda_raw,
        dof_diff=dof_diff,
        p_boot=p_boot,
        p_wilks=p_wilks,
        n_boot=n_boot_actual if 'n_boot_actual' in dir() else n_boot,
        k_exceed=k_exceed,
        nll_con=nll_con,
        nll_unc=nll_unc,
        R_shared=(r_shared, np.degrees(phi_shared)),
        R_per_channel=R_per_channel,
        chi2_dof_per_channel=chi2_dof_per_channel,
        health_per_channel=health_per_channel,
        invariant_holds=invariant_holds,
        overall_health=overall_health,
        verdict=verdict,
        reason=reason
    )


# =============================================================================
# Report Generation
# =============================================================================

def generate_multichannel_report(result: MultiChannelResult, out_path: str):
    """Generate markdown report for multi-channel test."""

    report = f"""# Global Multi-Channel Rank-1 Test Report

## Summary

| Metric | Value |
|--------|-------|
| Test Type | {result.test_type} |
| Channels | {result.n_channels} |
| **Verdict** | **{result.verdict}** |
| Reason | {result.reason} |

---

## Channels Tested

| Channel | chi2/dof | Health |
|---------|----------|--------|
"""
    for name in result.channel_names:
        chi2 = result.chi2_dof_per_channel.get(name, np.nan)
        health = result.health_per_channel.get(name, "N/A")
        report += f"| {name} | {chi2:.2f} | {health} |\n"

    report += f"""
---

## Test Statistics

| Statistic | Value |
|-----------|-------|
| Lambda_obs | {result.Lambda_obs:.4f} |
| Lambda_raw | {result.Lambda_raw:.4f} |
| dof_diff | {result.dof_diff} |
| NLL constrained | {result.nll_con:.2f} |
| NLL unconstrained | {result.nll_unc:.2f} |
| Nested invariant | {'PASS' if result.invariant_holds else 'FAIL'} |

---

## P-values

| Method | p-value | Details |
|--------|---------|---------|
| **Bootstrap** | {result.p_boot:.4f} | {result.k_exceed}/{result.n_boot} exceedances |
| Wilks (ref) | {result.p_wilks:.4f} | chi2({result.dof_diff}) approximation |

---

## Coupling Ratios

### Shared (Constrained Fit)

| Parameter | Value |
|-----------|-------|
| |R| | {result.R_shared[0]:.4f} |
| arg(R) | {result.R_shared[1]:.1f}deg |

### Per-Channel (Unconstrained Fit)

| Channel | |R| | arg(R) |
|---------|-----|--------|
"""
    for name, (r, phi) in result.R_per_channel.items():
        report += f"| {name} | {r:.4f} | {phi:.1f}deg |\n"

    report += f"""
---

## Interpretation

This global {result.n_channels}-channel test examines whether a **single** complex ratio
R = g(X2)/g(X1) is shared across ALL {result.n_channels} channels simultaneously.

**dof_diff = {result.dof_diff}**: The constrained model has {result.dof_diff} fewer
degrees of freedom than the unconstrained model (2 dof per additional channel constraint).

This is a more stringent test than pairwise comparisons because it requires
consistency across ALL channels simultaneously.

---

## Verdict Key

| Verdict | Meaning |
|---------|---------|
| NOT_REJECTED | p >= 0.05, consistent with shared R |
| DISFAVORED | p < 0.05, evidence against shared R |
| INCONCLUSIVE | Fit health issues |
| OPTIMIZER_FAILURE | Nested invariant violated |

---

*Generated by rank1_multichannel.py*
"""

    with open(out_path, 'w') as f:
        f.write(report)

    # Also save JSON
    json_path = out_path.replace('.md', '.json')
    result_dict = {
        'test_type': result.test_type,
        'n_channels': int(result.n_channels),
        'channel_names': result.channel_names,
        'Lambda_obs': float(result.Lambda_obs) if not np.isnan(result.Lambda_obs) else None,
        'Lambda_raw': float(result.Lambda_raw) if not np.isnan(result.Lambda_raw) else None,
        'dof_diff': int(result.dof_diff),
        'p_boot': float(result.p_boot) if not np.isnan(result.p_boot) else None,
        'p_wilks': float(result.p_wilks) if not np.isnan(result.p_wilks) else None,
        'n_boot': int(result.n_boot),
        'k_exceed': int(result.k_exceed),
        'nll_con': float(result.nll_con),
        'nll_unc': float(result.nll_unc),
        'R_shared': {'r': float(result.R_shared[0]), 'phi_deg': float(result.R_shared[1])},
        'R_per_channel': {k: {'r': float(v[0]), 'phi_deg': float(v[1])}
                          for k, v in result.R_per_channel.items()},
        'chi2_dof_per_channel': {k: float(v) for k, v in result.chi2_dof_per_channel.items()},
        'health_per_channel': result.health_per_channel,
        'invariant_holds': bool(result.invariant_holds),
        'overall_health': result.overall_health,
        'verdict': result.verdict,
        'reason': result.reason,
    }

    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

    logger.info(f"Saved report: {out_path}")
    logger.info(f"Saved JSON: {json_path}")


# =============================================================================
# Convenience: Load Belle Zb Channels
# =============================================================================

def load_belle_zb_channels(extracted_dir: str) -> List[ChannelData]:
    """Load all Belle Zb channels from extracted directory."""
    import os

    channel_specs = [
        ('upsilon2s', False),
        ('upsilon3s', False),
        ('hb1p', True),   # spin_flip
        ('hb2p', True),   # spin_flip
    ]

    channels = []
    for name, spin_flip in channel_specs:
        csv_path = os.path.join(extracted_dir, f'{name}.csv')
        if os.path.exists(csv_path):
            data = np.genfromtxt(csv_path, delimiter=',', skip_header=2)
            ch = ChannelData(
                name=name,
                m=data[:, 0],
                y=data[:, 1],
                yerr=data[:, 2],
                spin_flip=spin_flip
            )
            channels.append(ch)
            logger.info(f"Loaded {name}: {len(ch.m)} points")

    return channels


def load_lhcb_pc_channels(data_dir: str) -> List[ChannelData]:
    """Load LHCb Pc projection tables as channels."""
    import os

    channels = []
    for i in range(1, 5):
        csv_path = os.path.join(data_dir, f't{i}_full.csv' if i == 1 else f't{i}_cut.csv' if i == 2 else f't{i}_weighted.csv' if i == 3 else f't{i}_weight.csv')
        if not os.path.exists(csv_path):
            # Try alternate names
            for alt in [f't{i}_full.csv', f't{i}_cut.csv', f't{i}_weighted.csv', f't{i}_weight.csv']:
                alt_path = os.path.join(data_dir, alt)
                if os.path.exists(alt_path):
                    csv_path = alt_path
                    break

        if os.path.exists(csv_path):
            try:
                data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
                if len(data.shape) == 1:
                    continue
                ch = ChannelData(
                    name=os.path.basename(csv_path).replace('.csv', ''),
                    m=data[:, 0],
                    y=data[:, 1] if data.shape[1] > 1 else np.zeros_like(data[:, 0]),
                    yerr=data[:, 2] if data.shape[1] > 2 else np.ones_like(data[:, 0]),
                    spin_flip=False
                )
                if len(ch.m) >= 5:
                    channels.append(ch)
                    logger.info(f"Loaded {ch.name}: {len(ch.m)} points")
            except Exception as e:
                logger.warning(f"Failed to load {csv_path}: {e}")

    return channels


if __name__ == "__main__":
    import argparse
    import os

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Global Multi-Channel Rank-1 Test")
    parser.add_argument("--dataset", choices=["belle_zb", "lhcb_pc"], default="belle_zb",
                        help="Dataset to test")
    parser.add_argument("--n-boot", type=int, default=100, help="Bootstrap replicates")
    parser.add_argument("--n-starts", type=int, default=80, help="Optimizer starts")
    parser.add_argument("--outdir", default="out", help="Output directory")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.dataset == "belle_zb":
        # Belle Zb(10610)/Zb(10650)
        extracted_dir = os.path.join(base_dir, "belle_zb_rank1/extracted")
        channels = load_belle_zb_channels(extracted_dir)

        # Physics parameters
        M1 = 10.6072  # GeV
        G1 = 0.0184
        M2 = 10.6522
        G2 = 0.0115
        m_ref = 10.55

    elif args.dataset == "lhcb_pc":
        # LHCb Pc(4440)/Pc(4457)
        data_dir = os.path.join(base_dir, "lhcb_pc_rank1_v5/data/hepdata")
        channels = load_lhcb_pc_channels(data_dir)

        # Physics parameters (GeV)
        M1 = 4.440
        G1 = 0.020
        M2 = 4.457
        G2 = 0.006
        m_ref = 4.45

    if len(channels) < 2:
        print(f"ERROR: Need at least 2 channels, found {len(channels)}")
        exit(1)

    print(f"\n{'='*70}")
    print(f"Global Multi-Channel Rank-1 Test: {args.dataset}")
    print(f"{'='*70}")
    print(f"Channels: {[ch.name for ch in channels]}")

    result = run_global_multichannel_test(
        channels, M1, G1, M2, G2, m_ref,
        n_boot=args.n_boot, n_starts=args.n_starts
    )

    print(f"\n{'='*70}")
    print(f"VERDICT: {result.verdict}")
    print(f"Reason: {result.reason}")
    print(f"Lambda = {result.Lambda_obs:.4f}, p_boot = {result.p_boot:.4f}")
    print(f"{'='*70}")

    os.makedirs(args.outdir, exist_ok=True)
    generate_multichannel_report(result, os.path.join(args.outdir, "MULTICHANNEL_REPORT.md"))
