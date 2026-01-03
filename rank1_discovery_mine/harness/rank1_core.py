"""
Core rank-1 fitting functions.

Implements coherent two-BW model for exotic hadron doublets.
Based on the LHCb pentaquark analysis approach.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2 as chi2_dist
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


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


def breit_wigner(m: np.ndarray, M: float, Gamma: float) -> np.ndarray:
    """
    Relativistic Breit-Wigner amplitude.
    BW(m) = 1 / ((m - M) - i*Γ/2)

    Args:
        m: Mass array in GeV
        M: Resonance mass in MeV
        Gamma: Width in MeV
    """
    M_gev = M / 1000.0
    G_gev = Gamma / 1000.0
    return 1.0 / ((m - M_gev) - 1j * G_gev / 2.0)


def model_coherent(m: np.ndarray, params: List[float],
                   M1: float, G1: float, M2: float, G2: float,
                   fit_window: Tuple[float, float], use_bg1: bool = True) -> np.ndarray:
    """
    Coherent two-BW model:
    I(m) = |a1 * BW1(m) + R * BW2(m)|² + background

    Args:
        m: Mass array (GeV)
        params: [a1, r, phi, norm, bg0, bg1?]
        M1, G1: First resonance mass and width (MeV)
        M2, G2: Second resonance mass and width (MeV)
        fit_window: (m_low, m_high) in MeV
        use_bg1: Include linear background term
    """
    a1 = params[0]
    r = params[1]
    phi = params[2]
    norm = params[3]
    bg0 = params[4]
    bg1 = params[5] if use_bg1 and len(params) > 5 else 0.0

    BW1 = breit_wigner(m, M1, G1)
    BW2 = breit_wigner(m, M2, G2)

    R = r * np.exp(1j * phi)
    amplitude = a1 * (BW1 + R * BW2)
    signal = norm * np.abs(amplitude)**2

    m_center = (fit_window[0] + fit_window[1]) / 2000.0  # GeV
    background = bg0 + bg1 * (m - m_center)

    return signal + np.maximum(background, 0)


def nll_gaussian(params: List[float], data: np.ndarray,
                 M1: float, G1: float, M2: float, G2: float,
                 fit_window: Tuple[float, float], use_bg1: bool = True) -> float:
    """Gaussian NLL."""
    nll = 0.0
    for m, y, y_err in data:
        pred = model_coherent(np.array([m]), params, M1, G1, M2, G2, fit_window, use_bg1)[0]
        nll += 0.5 * ((y - pred) / y_err)**2
    return nll


def fit_single_spectrum(data: np.ndarray, M1: float, G1: float, M2: float, G2: float,
                        fit_window: Tuple[float, float], n_starts: int = 200,
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
        result = differential_evolution(nll, bounds, maxiter=200, seed=42, polish=True)
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x.copy()
    except Exception:
        pass

    # Multi-start local optimization
    for i in range(n_starts):
        x0 = [np.random.uniform(lo, hi) for lo, hi in bounds]
        try:
            result = minimize(nll, x0, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Compute chi2/dof
    if best_params is not None:
        chi2 = 0
        for m, y, y_err in data:
            pred = model_coherent(np.array([m]), best_params, M1, G1, M2, G2, fit_window, use_bg1)[0]
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
        chi2, dof, health = np.nan, 0, "FIT_FAILED"

    return best_nll, best_params, chi2, dof, health


def fit_joint_constrained(data_A: np.ndarray, data_B: np.ndarray,
                          M1: float, G1: float, M2: float, G2: float,
                          fit_window: Tuple[float, float],
                          n_starts: int = 200, use_bg1: bool = True) -> Tuple[float, Optional[np.ndarray]]:
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
        result = differential_evolution(joint_nll, bounds, maxiter=200, seed=42, polish=True)
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x.copy()
    except Exception:
        pass

    for i in range(n_starts):
        x0 = [np.random.uniform(lo, hi) for lo, hi in bounds]
        try:
            result = minimize(joint_nll, x0, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    return best_nll, best_params


def fit_joint_unconstrained(data_A: np.ndarray, data_B: np.ndarray,
                            M1: float, G1: float, M2: float, G2: float,
                            fit_window: Tuple[float, float],
                            n_starts: int = 200, use_bg1: bool = True) -> Tuple[float, Optional[np.ndarray]]:
    """
    Joint fit with independent R_A and R_B.

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

    try:
        result = differential_evolution(joint_nll, bounds, maxiter=200, seed=42, polish=True)
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x.copy()
    except Exception:
        pass

    for i in range(n_starts):
        x0 = [np.random.uniform(lo, hi) for lo, hi in bounds]
        try:
            result = minimize(joint_nll, x0, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    return best_nll, best_params


def _bootstrap_worker(args):
    """Single bootstrap replicate (for parallel execution)."""
    data_A, data_B, seed, use_bg1, M1, G1, M2, G2, fit_window, params_con = args
    np.random.seed(seed)

    def generate_pseudo(data, params):
        pseudo = []
        for m, y, y_err in data:
            mu = model_coherent(np.array([m]), params, M1, G1, M2, G2, fit_window, use_bg1)[0]
            y_new = np.random.normal(mu, y_err)
            pseudo.append((m, y_new, y_err))
        return np.array(pseudo)

    # Build params for A and B from constrained
    a1_A, a1_B, r, phi, norm_A, norm_B, bg0_A, bg0_B = params_con[:8]
    bg1_A = params_con[8] if use_bg1 else 0
    bg1_B = params_con[9] if use_bg1 else 0
    params_A = [a1_A, r, phi, norm_A, bg0_A, bg1_A]
    params_B = [a1_B, r, phi, norm_B, bg0_B, bg1_B]

    pseudo_A = generate_pseudo(data_A, params_A)
    pseudo_B = generate_pseudo(data_B, params_B)

    nll_con, _ = fit_joint_constrained(pseudo_A, pseudo_B, M1, G1, M2, G2, fit_window,
                                       n_starts=80, use_bg1=use_bg1)
    nll_unc, _ = fit_joint_unconstrained(pseudo_A, pseudo_B, M1, G1, M2, G2, fit_window,
                                         n_starts=80, use_bg1=use_bg1)

    return 2 * max(0, nll_con - nll_unc)


def run_bootstrap(data_A: np.ndarray, data_B: np.ndarray, params_con: np.ndarray,
                  M1: float, G1: float, M2: float, G2: float,
                  fit_window: Tuple[float, float],
                  n_boot: int = 100, use_bg1: bool = True) -> np.ndarray:
    """Run bootstrap for p-value estimation."""
    n_workers = max(1, cpu_count() - 1)
    args_list = [(data_A, data_B, i, use_bg1, M1, G1, M2, G2, fit_window, params_con)
                 for i in range(n_boot)]

    with Pool(n_workers) as pool:
        lambda_boots = list(pool.map(_bootstrap_worker, args_list))

    return np.array(lambda_boots)


def run_pair_test(data_A: np.ndarray, data_B: np.ndarray,
                  pair_name: str, M1: float, G1: float, M2: float, G2: float,
                  fit_window: Tuple[float, float],
                  n_boot: int = 100, use_bg1: bool = True) -> Dict[str, Any]:
    """
    Run full rank-1 test for a pair of spectra.

    Returns dict with results including verdict.
    """
    results = {'pair': pair_name}

    # Individual fits
    nll_A, params_A, chi2_A, dof_A, health_A = fit_single_spectrum(
        data_A, M1, G1, M2, G2, fit_window, n_starts=200, use_bg1=use_bg1
    )
    nll_B, params_B, chi2_B, dof_B, health_B = fit_single_spectrum(
        data_B, M1, G1, M2, G2, fit_window, n_starts=200, use_bg1=use_bg1
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
        results['R_A'] = {'r': float(params_A[1]), 'phi_deg': float(np.rad2deg(params_A[2]))}
    if params_B is not None:
        results['R_B'] = {'r': float(params_B[1]), 'phi_deg': float(np.rad2deg(params_B[2]))}

    # Joint fits
    nll_con, params_con = fit_joint_constrained(
        data_A, data_B, M1, G1, M2, G2, fit_window, n_starts=200, use_bg1=use_bg1
    )
    nll_unc, params_unc = fit_joint_unconstrained(
        data_A, data_B, M1, G1, M2, G2, fit_window, n_starts=200, use_bg1=use_bg1
    )

    if params_con is not None:
        results['R_shared'] = {'r': float(params_con[2]), 'phi_deg': float(np.rad2deg(params_con[3]))}

    # Lambda statistic
    Lambda = 2 * max(0, nll_con - nll_unc)
    results['Lambda'] = float(Lambda)
    results['nll_con'] = float(nll_con) if not np.isinf(nll_con) else None
    results['nll_unc'] = float(nll_unc) if not np.isinf(nll_unc) else None

    # Wilks p-value (reference)
    p_wilks = 1 - chi2_dist.cdf(Lambda, 2)
    results['p_wilks'] = float(p_wilks)

    # Bootstrap
    if params_con is not None and n_boot > 0:
        lambda_boots = run_bootstrap(
            data_A, data_B, params_con, M1, G1, M2, G2, fit_window,
            n_boot=n_boot, use_bg1=use_bg1
        )
        k = sum(lb >= Lambda for lb in lambda_boots)
        p_boot = k / n_boot
        results['p_boot'] = float(p_boot)
        results['k'] = int(k)
        results['n_boot'] = n_boot
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
        reason = "Bootstrap not run"
    elif p_boot >= 0.05:
        verdict = "NOT_REJECTED"
        reason = f"p_boot = {p_boot:.3f} >= 0.05"
    else:
        verdict = "DISFAVORED"
        reason = f"p_boot = {p_boot:.3f} < 0.05"

    results['verdict'] = verdict
    results['reason'] = reason

    return results
