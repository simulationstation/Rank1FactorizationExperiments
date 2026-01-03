"""
Core rank-1 fitting functions.

Implements coherent two-BW model for exotic hadron doublets.
Based on the LHCb pentaquark analysis approach.

FIXED: Enforces nested-model invariant (nll_unc <= nll_con),
       correct Lambda computation, retry logic for optimizer failures.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2 as chi2_dist
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple, List, Optional, Any
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# =============================================================================
# CONSTANTS FOR NUMERICAL TOLERANCE
# =============================================================================
NESTED_MODEL_TOL = 1e-4  # Absolute tolerance for nested model invariant
NESTED_MODEL_REL_TOL = 1e-6  # Relative tolerance


def load_hepdata_csv(filepath: str, mass_col: int = 0, y_col: int = 3,
                     err_col: int = 4, mass_scale: float = 1.0) -> np.ndarray:
    """
    Load HEPData CSV format.

    Args:
        filepath: Path to CSV file
        mass_col: Column index for mass values
        y_col: Column index for y values (counts/cross-section)
        err_col: Column index for error values
        mass_scale: Multiply mass by this (1.0 for MeV, 0.001 for GeV->MeV)

    Returns:
        Array of (m_GeV, y, y_err)
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('$') or not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) > max(mass_col, y_col, err_col):
                try:
                    m = float(parts[mass_col]) * mass_scale
                    y = float(parts[y_col])
                    y_err = abs(float(parts[err_col]))
                    if y_err > 0:
                        data.append((m / 1000.0, y, y_err))  # Convert to GeV
                except (ValueError, IndexError):
                    pass
    return np.array(data) if data else np.array([]).reshape(0, 3)


def select_window(data: np.ndarray, m_low: float, m_high: float) -> np.ndarray:
    """Select data within mass window (m_low, m_high in MeV, data in GeV)."""
    if len(data) == 0:
        return data
    m_low_gev = m_low / 1000.0
    m_high_gev = m_high / 1000.0
    mask = (data[:, 0] >= m_low_gev) & (data[:, 0] <= m_high_gev)
    return data[mask]


# ============================================================================
# VECTORIZED CORE FUNCTIONS
# ============================================================================

@jit(nopython=True, cache=True)
def _breit_wigner_numba(m: np.ndarray, M_gev: float, G_gev: float) -> np.ndarray:
    """Breit-Wigner amplitude (numba-optimized)."""
    result = np.empty(len(m), dtype=np.complex128)
    for i in range(len(m)):
        result[i] = 1.0 / ((m[i] - M_gev) - 1j * G_gev / 2.0)
    return result


@jit(nopython=True, cache=True)
def _model_coherent_numba(m: np.ndarray, a1: float, r: float, phi: float,
                          norm: float, bg0: float, bg1: float,
                          M1_gev: float, G1_gev: float,
                          M2_gev: float, G2_gev: float,
                          m_center: float) -> np.ndarray:
    """Coherent two-BW model (numba-optimized, vectorized)."""
    n = len(m)
    result = np.empty(n, dtype=np.float64)

    # Complex ratio
    R_real = r * np.cos(phi)
    R_imag = r * np.sin(phi)

    for i in range(n):
        # BW1
        denom1_real = m[i] - M1_gev
        denom1_imag = -G1_gev / 2.0
        denom1_sq = denom1_real**2 + denom1_imag**2
        bw1_real = denom1_real / denom1_sq
        bw1_imag = -denom1_imag / denom1_sq

        # BW2
        denom2_real = m[i] - M2_gev
        denom2_imag = -G2_gev / 2.0
        denom2_sq = denom2_real**2 + denom2_imag**2
        bw2_real = denom2_real / denom2_sq
        bw2_imag = -denom2_imag / denom2_sq

        # R * BW2
        rbw2_real = R_real * bw2_real - R_imag * bw2_imag
        rbw2_imag = R_real * bw2_imag + R_imag * bw2_real

        # a1 * (BW1 + R*BW2)
        amp_real = a1 * (bw1_real + rbw2_real)
        amp_imag = a1 * (bw1_imag + rbw2_imag)

        # |amplitude|^2
        signal = norm * (amp_real**2 + amp_imag**2)

        # Background
        background = bg0 + bg1 * (m[i] - m_center)
        if background < 0:
            background = 0.0

        result[i] = signal + background

    return result


@jit(nopython=True, cache=True)
def _nll_gaussian_numba(m: np.ndarray, y: np.ndarray, y_err: np.ndarray,
                        a1: float, r: float, phi: float,
                        norm: float, bg0: float, bg1: float,
                        M1_gev: float, G1_gev: float,
                        M2_gev: float, G2_gev: float,
                        m_center: float) -> float:
    """Vectorized Gaussian NLL (numba-optimized)."""
    pred = _model_coherent_numba(m, a1, r, phi, norm, bg0, bg1,
                                  M1_gev, G1_gev, M2_gev, G2_gev, m_center)
    nll = 0.0
    for i in range(len(m)):
        residual = (y[i] - pred[i]) / y_err[i]
        nll += 0.5 * residual * residual
    return nll


def breit_wigner(m: np.ndarray, M: float, Gamma: float) -> np.ndarray:
    """
    Relativistic Breit-Wigner amplitude (vectorized).
    BW(m) = 1 / ((m - M) - i*Γ/2)
    """
    M_gev = M / 1000.0
    G_gev = Gamma / 1000.0
    return 1.0 / ((m - M_gev) - 1j * G_gev / 2.0)


def model_coherent(m: np.ndarray, params: List[float],
                   M1: float, G1: float, M2: float, G2: float,
                   fit_window: Tuple[float, float], use_bg1: bool = True) -> np.ndarray:
    """
    Coherent two-BW model (vectorized).
    I(m) = |a1 * BW1(m) + R * BW2(m)|² + background
    """
    a1 = params[0]
    r = params[1]
    phi = params[2]
    norm = params[3]
    bg0 = params[4]
    bg1 = params[5] if use_bg1 and len(params) > 5 else 0.0

    m_center = (fit_window[0] + fit_window[1]) / 2000.0
    M1_gev = M1 / 1000.0
    G1_gev = G1 / 1000.0
    M2_gev = M2 / 1000.0
    G2_gev = G2 / 1000.0

    if HAS_NUMBA:
        return _model_coherent_numba(m, a1, r, phi, norm, bg0, bg1,
                                     M1_gev, G1_gev, M2_gev, G2_gev, m_center)
    else:
        # Fallback: numpy vectorized
        BW1 = breit_wigner(m, M1, G1)
        BW2 = breit_wigner(m, M2, G2)
        R = r * np.exp(1j * phi)
        amplitude = a1 * (BW1 + R * BW2)
        signal = norm * np.abs(amplitude)**2
        background = bg0 + bg1 * (m - m_center)
        return signal + np.maximum(background, 0)


def nll_gaussian(params: List[float], data: np.ndarray,
                 M1: float, G1: float, M2: float, G2: float,
                 fit_window: Tuple[float, float], use_bg1: bool = True) -> float:
    """Gaussian NLL (vectorized)."""
    if len(data) == 0:
        return np.inf

    a1 = params[0]
    r = params[1]
    phi = params[2]
    norm = params[3]
    bg0 = params[4]
    bg1 = params[5] if use_bg1 and len(params) > 5 else 0.0

    m_center = (fit_window[0] + fit_window[1]) / 2000.0
    M1_gev = M1 / 1000.0
    G1_gev = G1 / 1000.0
    M2_gev = M2 / 1000.0
    G2_gev = G2 / 1000.0

    m = data[:, 0]
    y = data[:, 1]
    y_err = data[:, 2]

    if HAS_NUMBA:
        return _nll_gaussian_numba(m, y, y_err, a1, r, phi, norm, bg0, bg1,
                                   M1_gev, G1_gev, M2_gev, G2_gev, m_center)
    else:
        # Fallback: numpy vectorized
        pred = model_coherent(m, params, M1, G1, M2, G2, fit_window, use_bg1)
        residuals = (y - pred) / y_err
        return 0.5 * np.sum(residuals**2)


# ============================================================================
# FITTING FUNCTIONS
# ============================================================================

def fit_single_spectrum(data: np.ndarray, M1: float, G1: float, M2: float, G2: float,
                        fit_window: Tuple[float, float], n_starts: int = 100,
                        use_bg1: bool = True) -> Tuple[float, Optional[np.ndarray], float, int, str]:
    """
    Fit single spectrum with coherent two-BW model.

    Returns: (best_nll, best_params, chi2, dof, health)
    """
    if len(data) == 0:
        return np.inf, None, np.nan, 0, "NO_DATA"

    n_params = 6 if use_bg1 else 5

    bounds = [
        (0.01, 100),      # a1
        (0.01, 10),       # r
        (-np.pi, np.pi),  # phi
        (1, 1e8),         # norm
        (0, 2000),        # bg0
    ]
    if use_bg1:
        bounds.append((-500, 500))

    def nll(p):
        return nll_gaussian(p, data, M1, G1, M2, G2, fit_window, use_bg1)

    best_nll = np.inf
    best_params = None

    # Global optimization
    try:
        result = differential_evolution(nll, bounds, maxiter=150, seed=42,
                                        polish=True, updating='deferred',
                                        workers=1)
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x.copy()
    except Exception:
        pass

    # Multi-start local optimization
    rng = np.random.default_rng(42)
    for i in range(n_starts):
        x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
        try:
            result = minimize(nll, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 200})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Compute chi2/dof (vectorized)
    if best_params is not None:
        m = data[:, 0]
        y = data[:, 1]
        y_err = data[:, 2]
        pred = model_coherent(m, best_params, M1, G1, M2, G2, fit_window, use_bg1)
        chi2 = np.sum(((y - pred) / y_err)**2)
        dof = len(data) - n_params
        chi2_dof = chi2 / max(1, dof)

        if chi2_dof < 0.5:
            health = "UNDERCONSTRAINED"
        elif chi2_dof > 3.0:
            health = "MODEL_MISMATCH"
        else:
            health = "HEALTHY"
    else:
        chi2, dof, health = np.nan, 0, "FIT_FAILED"

    return best_nll, best_params, chi2, dof, health


def fit_joint_constrained(data_A: np.ndarray, data_B: np.ndarray,
                          M1: float, G1: float, M2: float, G2: float,
                          fit_window: Tuple[float, float],
                          n_starts: int = 100, use_bg1: bool = True,
                          seed: int = 43) -> Tuple[float, Optional[np.ndarray]]:
    """
    Joint fit with shared R = (r, phi) between spectra A and B.

    Returns: (best_nll, best_params)
    """
    if len(data_A) == 0 or len(data_B) == 0:
        return np.inf, None

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
        bounds.extend([(-500, 500), (-500, 500)])

    def joint_nll(params):
        a1_A, a1_B, r, phi, norm_A, norm_B, bg0_A, bg0_B = params[:8]
        bg1_A = params[8] if use_bg1 else 0
        bg1_B = params[9] if use_bg1 else 0

        params_A = [a1_A, r, phi, norm_A, bg0_A, bg1_A]
        params_B = [a1_B, r, phi, norm_B, bg0_B, bg1_B]

        nll_A = nll_gaussian(params_A, data_A, M1, G1, M2, G2, fit_window, use_bg1)
        nll_B = nll_gaussian(params_B, data_B, M1, G1, M2, G2, fit_window, use_bg1)

        return nll_A + nll_B

    best_nll = np.inf
    best_params = None

    try:
        result = differential_evolution(joint_nll, bounds, maxiter=150, seed=seed,
                                        polish=True, updating='deferred', workers=1)
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x.copy()
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    for i in range(n_starts):
        x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
        try:
            result = minimize(joint_nll, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 200})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    return best_nll, best_params


def fit_joint_unconstrained(data_A: np.ndarray, data_B: np.ndarray,
                            M1: float, G1: float, M2: float, G2: float,
                            fit_window: Tuple[float, float],
                            n_starts: int = 100, use_bg1: bool = True,
                            seed: int = 44,
                            init_from_constrained: Optional[np.ndarray] = None) -> Tuple[float, Optional[np.ndarray]]:
    """
    Joint fit with independent R_A and R_B.

    Args:
        init_from_constrained: If provided, use constrained params to initialize
                               unconstrained fit (R_A = R_B = R_shared).

    Returns: (best_nll, best_params)
    """
    if len(data_A) == 0 or len(data_B) == 0:
        return np.inf, None

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

        nll_A = nll_gaussian(params_A, data_A, M1, G1, M2, G2, fit_window, use_bg1)
        nll_B = nll_gaussian(params_B, data_B, M1, G1, M2, G2, fit_window, use_bg1)

        return nll_A + nll_B

    best_nll = np.inf
    best_params = None

    # If we have constrained solution, use it as starting point
    # This ensures unconstrained can't be worse (it can reach same solution)
    if init_from_constrained is not None:
        # Expand constrained params to unconstrained format
        # Constrained: [a1_A, a1_B, r, phi, norm_A, norm_B, bg0_A, bg0_B, bg1_A?, bg1_B?]
        # Unconstrained: [a1_A, a1_B, r_A, phi_A, r_B, phi_B, norm_A, norm_B, bg0_A, bg0_B, bg1_A?, bg1_B?]
        con = init_from_constrained
        x0_from_con = [
            con[0], con[1],  # a1_A, a1_B
            con[2], con[3],  # r_A = r_shared, phi_A = phi_shared
            con[2], con[3],  # r_B = r_shared, phi_B = phi_shared
            con[4], con[5],  # norm_A, norm_B
            con[6], con[7],  # bg0_A, bg0_B
        ]
        if use_bg1 and len(con) >= 10:
            x0_from_con.extend([con[8], con[9]])  # bg1_A, bg1_B

        try:
            result = minimize(joint_nll, x0_from_con, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 300})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

        # Also try Powell from constrained init
        try:
            result = minimize(joint_nll, x0_from_con, method='Powell',
                            options={'maxiter': 500})
            # Clip to bounds
            clipped = np.clip(result.x, [b[0] for b in bounds], [b[1] for b in bounds])
            val = joint_nll(clipped)
            if val < best_nll:
                best_nll = val
                best_params = clipped.copy()
        except Exception:
            pass

    # Global optimization
    try:
        result = differential_evolution(joint_nll, bounds, maxiter=200, seed=seed,
                                        polish=True, updating='deferred', workers=1)
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x.copy()
    except Exception:
        pass

    # Multi-start local optimization
    rng = np.random.default_rng(seed)
    for i in range(n_starts):
        x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
        try:
            result = minimize(joint_nll, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 200})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Extra random restarts for phase parameters
    for i in range(n_starts // 2):
        x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
        # Random phase restarts
        x0[3] = rng.uniform(-np.pi, np.pi)  # phi_A
        x0[5] = rng.uniform(-np.pi, np.pi)  # phi_B
        try:
            result = minimize(joint_nll, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 200})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    return best_nll, best_params


def check_nested_invariant(nll_con: float, nll_unc: float) -> Tuple[bool, float]:
    """
    Check if nested model invariant holds: nll_unc <= nll_con (within tolerance).

    Returns: (invariant_holds, violation_amount)
    """
    if np.isinf(nll_con) or np.isinf(nll_unc):
        return False, np.inf

    tol = max(NESTED_MODEL_TOL, NESTED_MODEL_REL_TOL * abs(nll_con))
    violation = nll_unc - nll_con

    if violation > tol:
        return False, violation
    return True, violation


def compute_lambda_with_diagnostics(nll_con: float, nll_unc: float) -> Dict[str, Any]:
    """
    Compute Lambda with full diagnostics.

    Returns dict with:
        - Lambda_raw: 2*(nll_con - nll_unc), can be negative
        - Lambda: clamped to 0 only if invariant holds and Lambda_raw is small negative
        - invariant_holds: bool
        - violation: amount by which invariant is violated
    """
    invariant_holds, violation = check_nested_invariant(nll_con, nll_unc)

    Lambda_raw = 2 * (nll_con - nll_unc)

    if not invariant_holds:
        # Invariant violated - this is optimizer failure
        return {
            'Lambda_raw': float(Lambda_raw),
            'Lambda': None,  # Cannot compute valid Lambda
            'invariant_holds': False,
            'violation': float(violation),
        }

    # Invariant holds, clamp small negative values to 0
    Lambda = max(0.0, Lambda_raw)

    return {
        'Lambda_raw': float(Lambda_raw),
        'Lambda': float(Lambda),
        'invariant_holds': True,
        'violation': float(violation),
    }


# ============================================================================
# BOOTSTRAP
# ============================================================================

def _bootstrap_worker(args):
    """Single bootstrap replicate (for parallel execution)."""
    data_A, data_B, seed, use_bg1, M1, G1, M2, G2, fit_window, params_con = args

    rng = np.random.default_rng(seed)

    # Build params for A and B from constrained
    a1_A, a1_B, r, phi, norm_A, norm_B, bg0_A, bg0_B = params_con[:8]
    bg1_A = params_con[8] if use_bg1 else 0
    bg1_B = params_con[9] if use_bg1 else 0
    params_A = [a1_A, r, phi, norm_A, bg0_A, bg1_A]
    params_B = [a1_B, r, phi, norm_B, bg0_B, bg1_B]

    # Generate pseudo-data (vectorized)
    def generate_pseudo(data, params):
        m = data[:, 0]
        y_err = data[:, 2]
        mu = model_coherent(m, params, M1, G1, M2, G2, fit_window, use_bg1)
        y_new = rng.normal(mu, y_err)
        return np.column_stack([m, y_new, y_err])

    pseudo_A = generate_pseudo(data_A, params_A)
    pseudo_B = generate_pseudo(data_B, params_B)

    # Fit constrained
    nll_con, params_con_boot = fit_joint_constrained(
        pseudo_A, pseudo_B, M1, G1, M2, G2, fit_window,
        n_starts=30, use_bg1=use_bg1, seed=seed
    )

    # Fit unconstrained, initialized from constrained
    nll_unc, _ = fit_joint_unconstrained(
        pseudo_A, pseudo_B, M1, G1, M2, G2, fit_window,
        n_starts=40, use_bg1=use_bg1, seed=seed + 1000,
        init_from_constrained=params_con_boot
    )

    # Check invariant
    invariant_holds, violation = check_nested_invariant(nll_con, nll_unc)

    if not invariant_holds:
        # Return special value to indicate failure
        return {'failed': True, 'violation': violation}

    Lambda_raw = 2 * (nll_con - nll_unc)
    Lambda = max(0.0, Lambda_raw)

    return {'failed': False, 'Lambda': Lambda, 'Lambda_raw': Lambda_raw}


def run_bootstrap(data_A: np.ndarray, data_B: np.ndarray, params_con: np.ndarray,
                  M1: float, G1: float, M2: float, G2: float,
                  fit_window: Tuple[float, float],
                  n_boot: int = 100, use_bg1: bool = True,
                  max_retries_per_rep: int = 2) -> Dict[str, Any]:
    """
    Run bootstrap for p-value estimation with invariant checking.

    Returns dict with:
        - lambda_boots: list of valid Lambda values
        - n_valid: number of valid replicates
        - n_failed: number of failed replicates
        - violations: list of violation amounts for failed reps
    """
    n_workers = max(1, cpu_count() - 1)

    lambda_boots = []
    n_failed = 0
    violations = []

    # Run in batches to allow retries
    seed_offset = 0
    remaining = n_boot

    while remaining > 0 and seed_offset < n_boot * (max_retries_per_rep + 1):
        batch_size = min(remaining, n_boot)
        args_list = [(data_A, data_B, seed_offset + i, use_bg1, M1, G1, M2, G2, fit_window, params_con)
                     for i in range(batch_size)]

        with Pool(n_workers) as pool:
            results = list(pool.map(_bootstrap_worker, args_list))

        for res in results:
            if res['failed']:
                n_failed += 1
                violations.append(res['violation'])
            else:
                lambda_boots.append(res['Lambda'])
                remaining -= 1

        seed_offset += batch_size

        # Stop if we've done enough retries
        if seed_offset >= n_boot * (max_retries_per_rep + 1):
            break

    return {
        'lambda_boots': lambda_boots,
        'n_valid': len(lambda_boots),
        'n_failed': n_failed,
        'violations': violations,
    }


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def run_pair_test(data_A: np.ndarray, data_B: np.ndarray,
                  pair_name: str, M1: float, G1: float, M2: float, G2: float,
                  fit_window: Tuple[float, float],
                  n_boot: int = 100, use_bg1: bool = True,
                  n_starts: int = 100) -> Dict[str, Any]:
    """
    Run full rank-1 test for a pair of spectra.

    Returns dict with results including verdict and full diagnostics.
    """
    results = {'pair': pair_name}

    # Individual fits
    nll_A, params_A, chi2_A, dof_A, health_A = fit_single_spectrum(
        data_A, M1, G1, M2, G2, fit_window, n_starts=n_starts, use_bg1=use_bg1
    )
    nll_B, params_B, chi2_B, dof_B, health_B = fit_single_spectrum(
        data_B, M1, G1, M2, G2, fit_window, n_starts=n_starts, use_bg1=use_bg1
    )

    results.update({
        'chi2_A': float(chi2_A) if not np.isnan(chi2_A) else None,
        'dof_A': int(dof_A),
        'health_A': health_A,
        'chi2_B': float(chi2_B) if not np.isnan(chi2_B) else None,
        'dof_B': int(dof_B),
        'health_B': health_B,
    })

    if params_A is not None:
        results['R_A_ind'] = {'r': float(params_A[1]), 'phi_deg': float(np.rad2deg(params_A[2]))}
    if params_B is not None:
        results['R_B_ind'] = {'r': float(params_B[1]), 'phi_deg': float(np.rad2deg(params_B[2]))}

    # Joint constrained fit
    nll_con, params_con = fit_joint_constrained(
        data_A, data_B, M1, G1, M2, G2, fit_window, n_starts=n_starts, use_bg1=use_bg1
    )

    if params_con is not None:
        results['R_shared'] = {'r': float(params_con[2]), 'phi_deg': float(np.rad2deg(params_con[3]))}

    # Joint unconstrained fit, initialized from constrained
    nll_unc, params_unc = fit_joint_unconstrained(
        data_A, data_B, M1, G1, M2, G2, fit_window, n_starts=n_starts, use_bg1=use_bg1,
        init_from_constrained=params_con
    )

    if params_unc is not None:
        results['R_A_unc'] = {'r': float(params_unc[2]), 'phi_deg': float(np.rad2deg(params_unc[3]))}
        results['R_B_unc'] = {'r': float(params_unc[4]), 'phi_deg': float(np.rad2deg(params_unc[5]))}

    # Compute Lambda with diagnostics
    lambda_diag = compute_lambda_with_diagnostics(nll_con, nll_unc)

    results['nll_con'] = float(nll_con) if not np.isinf(nll_con) else None
    results['nll_unc'] = float(nll_unc) if not np.isinf(nll_unc) else None
    results['Lambda_raw'] = lambda_diag['Lambda_raw']
    results['Lambda'] = lambda_diag['Lambda']
    results['invariant_holds'] = lambda_diag['invariant_holds']
    results['invariant_violation'] = lambda_diag['violation']

    # Check for optimizer failure
    if not lambda_diag['invariant_holds']:
        results['verdict'] = 'OPTIMIZER_FAILURE'
        results['reason'] = f"Nested invariant violated: nll_unc ({nll_unc:.4f}) > nll_con ({nll_con:.4f}) by {lambda_diag['violation']:.4f}"
        results['p_wilks'] = None
        results['p_boot'] = None
        return results

    # Wilks p-value (reference)
    Lambda = lambda_diag['Lambda']
    p_wilks = 1 - chi2_dist.cdf(Lambda, 2)
    results['p_wilks'] = float(p_wilks)

    # Bootstrap
    if params_con is not None and n_boot > 0:
        boot_results = run_bootstrap(
            data_A, data_B, params_con, M1, G1, M2, G2, fit_window,
            n_boot=n_boot, use_bg1=use_bg1
        )

        lambda_boots = boot_results['lambda_boots']
        n_valid = boot_results['n_valid']
        n_failed = boot_results['n_failed']

        if n_valid > 0:
            # Conservative p-value estimator
            k = sum(lb >= Lambda for lb in lambda_boots)
            p_boot = (1 + k) / (1 + n_valid)

            results['p_boot'] = float(p_boot)
            results['k'] = int(k)
            results['n_boot_valid'] = n_valid
            results['n_boot_failed'] = n_failed
            results['lambda_boot_mean'] = float(np.mean(lambda_boots))
            results['lambda_boot_std'] = float(np.std(lambda_boots))
            results['lambda_boot_median'] = float(np.median(lambda_boots))
        else:
            p_boot = np.nan
            results['p_boot'] = None
            results['n_boot_valid'] = 0
            results['n_boot_failed'] = n_failed
    else:
        p_boot = np.nan
        results['p_boot'] = None

    # Verdict
    if health_A not in ("HEALTHY", "UNDERCONSTRAINED") or health_B not in ("HEALTHY", "UNDERCONSTRAINED"):
        verdict = "INCONCLUSIVE"
        reason = f"Fit health issues: A={health_A}, B={health_B}"
    elif params_con is None:
        verdict = "OPTIMIZER_FAILURE"
        reason = "Constrained fit failed"
    elif np.isnan(p_boot) or p_boot is None:
        verdict = "NO_BOOTSTRAP"
        reason = "Bootstrap not run or all replicates failed"
    elif p_boot >= 0.05:
        verdict = "NOT_REJECTED"
        reason = f"p_boot = {p_boot:.3f} >= 0.05"
    else:
        verdict = "DISFAVORED"
        reason = f"p_boot = {p_boot:.3f} < 0.05"

    results['verdict'] = verdict
    results['reason'] = reason

    return results


def quick_check(data_A: np.ndarray, data_B: np.ndarray,
                M1: float, G1: float, M2: float, G2: float,
                fit_window: Tuple[float, float],
                use_bg1: bool = True) -> Dict[str, Any]:
    """
    Quick diagnostic check before full bootstrap.

    Runs minimal fits to verify:
    1. Constrained fit succeeds
    2. Unconstrained fit succeeds
    3. Nested invariant holds
    4. Bootstrap pipeline works (3 replicates)

    Returns dict with pass/fail and diagnostics.
    """
    result = {'passed': False, 'checks': {}}

    # Constrained fit
    nll_con, params_con = fit_joint_constrained(
        data_A, data_B, M1, G1, M2, G2, fit_window,
        n_starts=20, use_bg1=use_bg1
    )
    result['checks']['constrained_fit'] = params_con is not None
    if params_con is None:
        result['error'] = "Constrained fit failed"
        return result

    # Unconstrained fit
    nll_unc, params_unc = fit_joint_unconstrained(
        data_A, data_B, M1, G1, M2, G2, fit_window,
        n_starts=30, use_bg1=use_bg1,
        init_from_constrained=params_con
    )
    result['checks']['unconstrained_fit'] = params_unc is not None
    if params_unc is None:
        result['error'] = "Unconstrained fit failed"
        return result

    # Check invariant
    invariant_holds, violation = check_nested_invariant(nll_con, nll_unc)
    result['checks']['invariant_holds'] = invariant_holds
    result['nll_con'] = float(nll_con)
    result['nll_unc'] = float(nll_unc)
    result['violation'] = float(violation)

    if not invariant_holds:
        result['error'] = f"Invariant violated: nll_unc > nll_con by {violation:.4f}"
        return result

    # Quick bootstrap (3 reps)
    boot_results = run_bootstrap(
        data_A, data_B, params_con, M1, G1, M2, G2, fit_window,
        n_boot=3, use_bg1=use_bg1
    )
    result['checks']['bootstrap_works'] = boot_results['n_valid'] >= 2
    result['bootstrap_valid'] = boot_results['n_valid']
    result['bootstrap_failed'] = boot_results['n_failed']

    if boot_results['n_valid'] < 2:
        result['error'] = f"Bootstrap failed: only {boot_results['n_valid']}/3 valid"
        return result

    result['passed'] = True
    return result
