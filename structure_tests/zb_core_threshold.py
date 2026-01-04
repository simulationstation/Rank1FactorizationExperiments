#!/usr/bin/env python3
"""
Belle Zb core + threshold dressing structure test.

Implements a nested test between:
H0: R_hid == R_open == R_core
H1: R_hid and R_open independent
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json
import math
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.optimize import differential_evolution, minimize
from scipy.stats import chi2 as chi2_dist


M_ZB10610 = 10607.2  # MeV
G_ZB10610 = 18.4     # MeV
M_ZB10650 = 10652.2  # MeV
G_ZB10650 = 11.5     # MeV

M_B = 5.279  # GeV
M_BSTAR = 5.325  # GeV
M_THR1 = M_B + M_BSTAR
M_THR2 = 2.0 * M_BSTAR


@dataclass(frozen=True)
class HiddenRatioDatum:
    name: str
    r: float
    r_err: float
    phi_deg: float
    phi_err_deg: float
    spin_flip: bool


@dataclass(frozen=True)
class OpenSpectrumData:
    name: str
    mass_gev: np.ndarray
    signal: np.ndarray
    signal_err: np.ndarray


@dataclass(frozen=True)
class OpenRatioDatum:
    r: float
    r_err: float
    phi_deg: float
    phi_err_deg: float


@dataclass(frozen=True)
class FitResult:
    nll: float
    params: Optional[np.ndarray]


@dataclass(frozen=True)
class SectorHealth:
    chi2: float
    dof: int
    chi2_dof: float


@dataclass(frozen=True)
class StructureResult:
    status: str
    reason: str
    mode: str
    lambda_stat: Optional[float]
    p_boot: Optional[float]
    p_wilks: Optional[float]
    r_core: Optional[float]
    phi_core: Optional[float]
    r_hid: Optional[float]
    phi_hid: Optional[float]
    r_open: Optional[float]
    phi_open: Optional[float]
    kappa1: Optional[float]
    kappa2: Optional[float]
    hidden_health: Optional[SectorHealth]
    open_health: Optional[SectorHealth]


HIDDEN_RATIO_TABLE: List[HiddenRatioDatum] = [
    HiddenRatioDatum(
        name='Υ(2S)π',
        r=0.86,
        r_err=math.sqrt(0.11**2 + 0.04**2),
        phi_deg=-13.0,
        phi_err_deg=math.sqrt(13**2 + 17**2),
        spin_flip=False,
    ),
    HiddenRatioDatum(
        name='Υ(3S)π',
        r=0.96,
        r_err=math.sqrt(0.14**2 + 0.08**2),
        phi_deg=-9.0,
        phi_err_deg=math.sqrt(19**2 + 11**2),
        spin_flip=False,
    ),
    HiddenRatioDatum(
        name='hb(1P)π',
        r=1.39,
        r_err=math.sqrt(0.37**2 + 0.05**2),
        phi_deg=187.0,
        phi_err_deg=math.sqrt(44**2 + 3**2),
        spin_flip=True,
    ),
    HiddenRatioDatum(
        name='hb(2P)π',
        r=1.60,
        r_err=math.sqrt(0.6**2 + 0.4**2),
        phi_deg=181.0,
        phi_err_deg=math.sqrt(65**2 + 74**2),
        spin_flip=True,
    ),
]


def wrap_to_pi(phi: np.ndarray | float) -> np.ndarray | float:
    return (phi + np.pi) % (2 * np.pi) - np.pi


def _phi_with_spin_flip(phi: float, spin_flip: bool) -> float:
    return phi + (np.pi if spin_flip else 0.0)


def hidden_ratio_nll(r: float, phi: float, data: Iterable[HiddenRatioDatum]) -> float:
    nll = 0.0
    for datum in data:
        phi_obs = math.radians(datum.phi_deg)
        phi_err = math.radians(datum.phi_err_deg)
        phi_pred = _phi_with_spin_flip(phi, datum.spin_flip)
        dphi = wrap_to_pi(phi_pred - phi_obs)
        nll += 0.5 * ((r - datum.r) / datum.r_err) ** 2
        nll += 0.5 * (dphi / phi_err) ** 2
    return nll


def hidden_ratio_chi2(r: float, phi: float, data: Iterable[HiddenRatioDatum]) -> SectorHealth:
    chi2 = 0.0
    n_points = 0
    for datum in data:
        phi_obs = math.radians(datum.phi_deg)
        phi_err = math.radians(datum.phi_err_deg)
        phi_pred = _phi_with_spin_flip(phi, datum.spin_flip)
        dphi = wrap_to_pi(phi_pred - phi_obs)
        chi2 += ((r - datum.r) / datum.r_err) ** 2
        chi2 += (dphi / phi_err) ** 2
        n_points += 2
    dof = max(1, n_points - 2)
    return SectorHealth(chi2=chi2, dof=dof, chi2_dof=chi2 / dof)


def rho_threshold(m: np.ndarray, mthr: float) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, 1.0 - (mthr**2) / (m**2)))


def gamma_dressed(gamma0: float, kappa: float, m: np.ndarray, mthr: float) -> np.ndarray:
    return gamma0 * (1.0 + kappa * rho_threshold(m, mthr))


def breit_wigner_dressed(
    m: np.ndarray,
    mass_mev: float,
    gamma0_mev: float,
    kappa: float,
    mthr: float,
) -> np.ndarray:
    gamma = gamma_dressed(gamma0_mev, kappa, m, mthr)
    return 1.0 / ((m - mass_mev / 1000.0) - 1j * gamma / 2000.0)


def open_spectrum_model(
    m: np.ndarray,
    norm: float,
    r_open: float,
    phi_open: float,
    b0: float,
    b1: float,
    b2: float,
    kappa1: float,
    kappa2: float,
) -> np.ndarray:
    bw1 = breit_wigner_dressed(m, M_ZB10610, G_ZB10610, kappa1, M_THR1)
    bw2 = breit_wigner_dressed(m, M_ZB10650, G_ZB10650, kappa2, M_THR2)
    r_complex = r_open * np.exp(1j * phi_open)
    intensity = norm * np.abs(bw1 + r_complex * bw2) ** 2
    background = b0 + b1 * (m - 10.6) + b2 * (m - 10.6) ** 2
    return intensity + np.maximum(background, 0.0)


def open_spectrum_nll(
    params: np.ndarray,
    data: OpenSpectrumData,
    r_open: float,
    phi_open: float,
) -> float:
    norm, b0, b1, b2, kappa1, kappa2 = params
    pred = open_spectrum_model(
        data.mass_gev, norm, r_open, phi_open, b0, b1, b2, kappa1, kappa2
    )
    return 0.5 * np.sum(((data.signal - pred) / data.signal_err) ** 2)


def open_spectrum_chi2(
    params: np.ndarray,
    data: OpenSpectrumData,
    r_open: float,
    phi_open: float,
) -> SectorHealth:
    norm, b0, b1, b2, kappa1, kappa2 = params
    pred = open_spectrum_model(
        data.mass_gev, norm, r_open, phi_open, b0, b1, b2, kappa1, kappa2
    )
    chi2 = float(np.sum(((data.signal - pred) / data.signal_err) ** 2))
    dof = max(1, len(data.mass_gev) - 8)
    return SectorHealth(chi2=chi2, dof=dof, chi2_dof=chi2 / dof)


def open_ratio_nll(r: float, phi: float, data: OpenRatioDatum) -> float:
    phi_obs = math.radians(data.phi_deg)
    phi_err = math.radians(data.phi_err_deg)
    dphi = wrap_to_pi(phi - phi_obs)
    return 0.5 * ((r - data.r) / data.r_err) ** 2 + 0.5 * (dphi / phi_err) ** 2


def open_ratio_chi2(r: float, phi: float, data: OpenRatioDatum) -> SectorHealth:
    phi_obs = math.radians(data.phi_deg)
    phi_err = math.radians(data.phi_err_deg)
    dphi = wrap_to_pi(phi - phi_obs)
    chi2 = ((r - data.r) / data.r_err) ** 2 + (dphi / phi_err) ** 2
    return SectorHealth(chi2=chi2, dof=1, chi2_dof=chi2)


def load_open_spectrum(csv_path: Path) -> OpenSpectrumData:
    masses: List[float] = []
    signal: List[float] = []
    signal_err: List[float] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or line.startswith("m_low"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            m_center = float(parts[2])
            sig = float(parts[8])
            sig_err = float(parts[9])
            if sig_err <= 0:
                continue
            masses.append(m_center / 1000.0)
            signal.append(sig)
            signal_err.append(sig_err)
    return OpenSpectrumData(
        name="BB*π",
        mass_gev=np.array(masses),
        signal=np.array(signal),
        signal_err=np.array(signal_err),
    )


def load_open_ratio(json_path: Path) -> OpenRatioDatum:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    ob = payload["open_bottom_R"]["Model-2_Solution1"]
    return OpenRatioDatum(
        r=ob["r"],
        r_err=ob["r_err"],
        phi_deg=ob["phi_deg"],
        phi_err_deg=ob["phi_err_deg"],
    )


def assess_health(hidden: SectorHealth, open_: SectorHealth) -> Tuple[str, str]:
    for label, sector in (("hidden", hidden), ("open", open_)):
        if sector.chi2_dof < 0.5:
            return "INCONCLUSIVE", f"{label} chi2/dof below 0.5"
        if sector.chi2_dof > 3.0:
            return "MODEL_MISMATCH", f"{label} chi2/dof above 3.0"
    return "OK", "fit-healthy"


def _fit_constrained(
    hidden_data: List[HiddenRatioDatum],
    open_data: Optional[OpenSpectrumData],
    open_ratio: Optional[OpenRatioDatum],
    n_starts: int,
    seed: int,
) -> FitResult:
    bounds: List[Tuple[float, float]] = [
        (0.01, 10.0),
        (-np.pi, np.pi),
    ]

    if open_data is not None:
        y_max = float(np.max(open_data.signal))
        bounds.extend(
            [
                (1e-3, 1e6 * max(1.0, y_max)),
                (-1e5, 1e5),
                (-1e5, 1e5),
                (-1e5, 1e5),
                (0.0, 5.0),
                (0.0, 5.0),
            ]
        )

    def nll(params: np.ndarray) -> float:
        r_core, phi_core = params[0], params[1]
        total = hidden_ratio_nll(r_core, phi_core, hidden_data)
        if open_data is not None:
            total += open_spectrum_nll(params[2:], open_data, r_core, phi_core)
        if open_ratio is not None:
            total += open_ratio_nll(r_core, phi_core, open_ratio)
        return total

    best_nll = np.inf
    best_params: Optional[np.ndarray] = None
    rng = np.random.default_rng(seed)

    try:
        result = differential_evolution(
            nll,
            bounds,
            maxiter=200,
            seed=seed,
            polish=True,
            workers=1,
        )
        if result.fun < best_nll:
            best_nll = float(result.fun)
            best_params = result.x.copy()
    except Exception:
        pass

    for _ in range(n_starts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        try:
            result = minimize(
                nll,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000},
            )
            if result.fun < best_nll:
                best_nll = float(result.fun)
                best_params = result.x.copy()
        except Exception:
            pass

    return FitResult(nll=best_nll, params=best_params)


def _fit_unconstrained(
    hidden_data: List[HiddenRatioDatum],
    open_data: Optional[OpenSpectrumData],
    open_ratio: Optional[OpenRatioDatum],
    n_starts: int,
    seed: int,
    init_from_constrained: Optional[np.ndarray],
) -> FitResult:
    bounds: List[Tuple[float, float]] = [
        (0.01, 10.0),
        (-np.pi, np.pi),
        (0.01, 10.0),
        (-np.pi, np.pi),
    ]

    if open_data is not None:
        y_max = float(np.max(open_data.signal))
        bounds.extend(
            [
                (1e-3, 1e6 * max(1.0, y_max)),
                (-1e5, 1e5),
                (-1e5, 1e5),
                (-1e5, 1e5),
                (0.0, 5.0),
                (0.0, 5.0),
            ]
        )

    def nll(params: np.ndarray) -> float:
        r_hid, phi_hid, r_open, phi_open = params[0], params[1], params[2], params[3]
        total = hidden_ratio_nll(r_hid, phi_hid, hidden_data)
        if open_data is not None:
            total += open_spectrum_nll(params[4:], open_data, r_open, phi_open)
        if open_ratio is not None:
            total += open_ratio_nll(r_open, phi_open, open_ratio)
        return total

    best_nll = np.inf
    best_params: Optional[np.ndarray] = None
    rng = np.random.default_rng(seed)

    if init_from_constrained is not None:
        r_core, phi_core = init_from_constrained[0], init_from_constrained[1]
        x0 = [r_core, phi_core, r_core, phi_core]
        if open_data is not None:
            x0.extend(init_from_constrained[2:])
        try:
            result = minimize(
                nll,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 3000},
            )
            if result.fun < best_nll:
                best_nll = float(result.fun)
                best_params = result.x.copy()
        except Exception:
            pass

    try:
        result = differential_evolution(
            nll,
            bounds,
            maxiter=200,
            seed=seed,
            polish=True,
            workers=1,
        )
        if result.fun < best_nll:
            best_nll = float(result.fun)
            best_params = result.x.copy()
    except Exception:
        pass

    for _ in range(n_starts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        try:
            result = minimize(
                nll,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000},
            )
            if result.fun < best_nll:
                best_nll = float(result.fun)
                best_params = result.x.copy()
        except Exception:
            pass

    return FitResult(nll=best_nll, params=best_params)


def _bootstrap_worker(args: Tuple[int, List[HiddenRatioDatum], Optional[OpenSpectrumData],
                                 Optional[OpenRatioDatum], np.ndarray, int, int]) -> float:
    seed, hidden_data, open_data, open_ratio, params_con, n_starts, base_seed = args
    rng = np.random.default_rng(seed + base_seed)
    r_core, phi_core = params_con[0], params_con[1]

    boot_hidden: List[HiddenRatioDatum] = []
    for datum in hidden_data:
        r_sample = rng.normal(r_core, datum.r_err)
        phi_mean = _phi_with_spin_flip(phi_core, datum.spin_flip)
        phi_sample = rng.normal(phi_mean, math.radians(datum.phi_err_deg))
        phi_sample = float(wrap_to_pi(phi_sample))
        boot_hidden.append(
            HiddenRatioDatum(
                name=datum.name,
                r=r_sample,
                r_err=datum.r_err,
                phi_deg=math.degrees(phi_sample),
                phi_err_deg=datum.phi_err_deg,
                spin_flip=datum.spin_flip,
            )
        )

    boot_open_data = None
    boot_open_ratio = None
    if open_data is not None:
        norm, b0, b1, b2, kappa1, kappa2 = params_con[2:]
        pred = open_spectrum_model(
            open_data.mass_gev,
            norm,
            r_core,
            phi_core,
            b0,
            b1,
            b2,
            kappa1,
            kappa2,
        )
        noisy = rng.normal(pred, open_data.signal_err)
        boot_open_data = OpenSpectrumData(
            name=open_data.name,
            mass_gev=open_data.mass_gev,
            signal=noisy,
            signal_err=open_data.signal_err,
        )
    if open_ratio is not None:
        r_sample = rng.normal(r_core, open_ratio.r_err)
        phi_sample = rng.normal(phi_core, math.radians(open_ratio.phi_err_deg))
        phi_sample = float(wrap_to_pi(phi_sample))
        boot_open_ratio = OpenRatioDatum(
            r=r_sample,
            r_err=open_ratio.r_err,
            phi_deg=math.degrees(phi_sample),
            phi_err_deg=open_ratio.phi_err_deg,
        )

    con_fit = _fit_constrained(boot_hidden, boot_open_data, boot_open_ratio, n_starts, seed)
    unc_fit = _fit_unconstrained(
        boot_hidden,
        boot_open_data,
        boot_open_ratio,
        n_starts,
        seed + 17,
        con_fit.params,
    )
    return 2.0 * (con_fit.nll - unc_fit.nll)


def bootstrap_lambda(
    hidden_data: List[HiddenRatioDatum],
    open_data: Optional[OpenSpectrumData],
    open_ratio: Optional[OpenRatioDatum],
    params_con: np.ndarray,
    n_boot: int,
    n_starts: int,
    seed: int,
) -> np.ndarray:
    if n_boot <= 0:
        return np.array([])
    workers = min(cpu_count(), max(1, n_boot))
    seeds = list(range(n_boot))
    args = [
        (s, hidden_data, open_data, open_ratio, params_con, n_starts, seed)
        for s in seeds
    ]
    with Pool(processes=workers) as pool:
        stats = pool.map(_bootstrap_worker, args)
    return np.array(stats)


def resolve_open_mode(mode: str, base_dir: Path) -> Tuple[str, Optional[Path], Optional[Path]]:
    spectrum_path = base_dir / "belle_zb_openbottom_rank1" / "extracted" / "bb_star_pi.csv"
    ratio_path = base_dir / "belle_zb_openbottom_rank1" / "out" / "result_table.json"

    if mode == "spectrum":
        return ("spectrum" if spectrum_path.exists() else "missing", spectrum_path, ratio_path)
    if mode == "ratio":
        return ("ratio" if ratio_path.exists() else "missing", spectrum_path, ratio_path)

    if spectrum_path.exists():
        return "spectrum", spectrum_path, ratio_path
    if ratio_path.exists():
        return "ratio", spectrum_path, ratio_path
    return "missing", spectrum_path, ratio_path


def run_structure_test(
    mode: str = "auto",
    n_boot: int = 200,
    n_starts: int = 40,
    seed: int = 42,
    base_dir: Optional[Path] = None,
) -> StructureResult:
    base_dir = base_dir or Path(__file__).resolve().parents[1]
    resolved, spectrum_path, ratio_path = resolve_open_mode(mode, base_dir)

    if resolved == "missing":
        return StructureResult(
            status="SKIPPED",
            reason="open-bottom inputs missing",
            mode=mode,
            lambda_stat=None,
            p_boot=None,
            p_wilks=None,
            r_core=None,
            phi_core=None,
            r_hid=None,
            phi_hid=None,
            r_open=None,
            phi_open=None,
            kappa1=None,
            kappa2=None,
            hidden_health=None,
            open_health=None,
        )

    hidden_data = list(HIDDEN_RATIO_TABLE)
    open_data = load_open_spectrum(spectrum_path) if resolved == "spectrum" else None
    open_ratio = load_open_ratio(ratio_path) if resolved == "ratio" else None

    con_fit = _fit_constrained(hidden_data, open_data, open_ratio, n_starts, seed)
    if con_fit.params is None:
        return StructureResult(
            status="OPTIMIZER_FAILURE",
            reason="constrained fit failed",
            mode=resolved,
            lambda_stat=None,
            p_boot=None,
            p_wilks=None,
            r_core=None,
            phi_core=None,
            r_hid=None,
            phi_hid=None,
            r_open=None,
            phi_open=None,
            kappa1=None,
            kappa2=None,
            hidden_health=None,
            open_health=None,
        )

    unc_fit = _fit_unconstrained(
        hidden_data,
        open_data,
        open_ratio,
        n_starts,
        seed + 1,
        con_fit.params,
    )
    if unc_fit.params is None:
        return StructureResult(
            status="OPTIMIZER_FAILURE",
            reason="unconstrained fit failed",
            mode=resolved,
            lambda_stat=None,
            p_boot=None,
            p_wilks=None,
            r_core=None,
            phi_core=None,
            r_hid=None,
            phi_hid=None,
            r_open=None,
            phi_open=None,
            kappa1=None,
            kappa2=None,
            hidden_health=None,
            open_health=None,
        )

    if unc_fit.nll > con_fit.nll + 1e-4:
        return StructureResult(
            status="OPTIMIZER_FAILURE",
            reason="nested invariant violated",
            mode=resolved,
            lambda_stat=None,
            p_boot=None,
            p_wilks=None,
            r_core=None,
            phi_core=None,
            r_hid=None,
            phi_hid=None,
            r_open=None,
            phi_open=None,
            kappa1=None,
            kappa2=None,
            hidden_health=None,
            open_health=None,
        )

    r_core, phi_core = con_fit.params[0], con_fit.params[1]
    r_hid, phi_hid, r_open, phi_open = (
        unc_fit.params[0],
        unc_fit.params[1],
        unc_fit.params[2],
        unc_fit.params[3],
    )

    hidden_health = hidden_ratio_chi2(r_hid, phi_hid, hidden_data)
    if open_data is not None:
        open_health = open_spectrum_chi2(unc_fit.params[4:], open_data, r_open, phi_open)
        kappa1, kappa2 = unc_fit.params[8], unc_fit.params[9]
    else:
        open_health = open_ratio_chi2(r_open, phi_open, open_ratio)
        kappa1, kappa2 = None, None

    health_status, health_reason = assess_health(hidden_health, open_health)

    lambda_stat = 2.0 * (con_fit.nll - unc_fit.nll)
    p_wilks = 1.0 - chi2_dist.cdf(lambda_stat, df=2)

    if health_status != "OK":
        return StructureResult(
            status=health_status,
            reason=health_reason,
            mode=resolved,
            lambda_stat=lambda_stat,
            p_boot=None,
            p_wilks=p_wilks,
            r_core=r_core,
            phi_core=phi_core,
            r_hid=r_hid,
            phi_hid=phi_hid,
            r_open=r_open,
            phi_open=phi_open,
            kappa1=kappa1,
            kappa2=kappa2,
            hidden_health=hidden_health,
            open_health=open_health,
        )

    stats = bootstrap_lambda(
        hidden_data,
        open_data,
        open_ratio,
        con_fit.params,
        n_boot,
        n_starts=max(10, n_starts // 2),
        seed=seed + 11,
    )
    p_boot = float(np.mean(stats >= lambda_stat)) if len(stats) else None

    verdict = "DISFAVORED" if (p_boot is not None and p_boot < 0.05) else "NOT_REJECTED"
    reason = f"p_boot={p_boot:.3f} (alpha=0.05)" if p_boot is not None else "bootstrap skipped"

    return StructureResult(
        status=verdict,
        reason=reason,
        mode=resolved,
        lambda_stat=lambda_stat,
        p_boot=p_boot,
        p_wilks=p_wilks,
        r_core=r_core,
        phi_core=phi_core,
        r_hid=r_hid,
        phi_hid=phi_hid,
        r_open=r_open,
        phi_open=phi_open,
        kappa1=kappa1,
        kappa2=kappa2,
        hidden_health=hidden_health,
        open_health=open_health,
    )


def result_to_dict(result: StructureResult) -> Dict[str, object]:
    return {
        "status": result.status,
        "reason": result.reason,
        "mode": result.mode,
        "lambda": result.lambda_stat,
        "p_boot": result.p_boot,
        "p_wilks": result.p_wilks,
        "r_core": result.r_core,
        "phi_core": result.phi_core,
        "r_hid": result.r_hid,
        "phi_hid": result.phi_hid,
        "r_open": result.r_open,
        "phi_open": result.phi_open,
        "kappa1": result.kappa1,
        "kappa2": result.kappa2,
        "hidden_health": None if result.hidden_health is None else result.hidden_health.__dict__,
        "open_health": None if result.open_health is None else result.open_health.__dict__,
    }


def render_report(result: StructureResult) -> str:
    def fmt_angle(phi: Optional[float]) -> str:
        if phi is None:
            return "n/a"
        return f"{math.degrees(phi):.1f}°"

    lines = [
        "# Belle Zb core + threshold dressing structure test",
        "",
        f"**Status:** {result.status}",
        f"**Reason:** {result.reason}",
        f"**Mode:** {result.mode}",
        "",
        "## Fit summary",
        f"- R_core = {result.r_core:.3f} @ {fmt_angle(result.phi_core)}" if result.r_core is not None else "- R_core = n/a",
        f"- R_hid = {result.r_hid:.3f} @ {fmt_angle(result.phi_hid)}" if result.r_hid is not None else "- R_hid = n/a",
        f"- R_open = {result.r_open:.3f} @ {fmt_angle(result.phi_open)}" if result.r_open is not None else "- R_open = n/a",
    ]

    if result.kappa1 is not None:
        lines.append(f"- kappa1 = {result.kappa1:.3f}")
        lines.append(f"- kappa2 = {result.kappa2:.3f}")

    if result.hidden_health is not None:
        lines.append("")
        lines.append("## Fit health")
        lines.append(
            f"- Hidden chi2/dof = {result.hidden_health.chi2:.2f}/{result.hidden_health.dof} = {result.hidden_health.chi2_dof:.2f}"
        )
        if result.open_health is not None:
            lines.append(
                f"- Open chi2/dof = {result.open_health.chi2:.2f}/{result.open_health.dof} = {result.open_health.chi2_dof:.2f}"
            )

    lines.extend(
        [
            "",
            "## Test statistic",
            f"- Lambda = {result.lambda_stat:.3f}" if result.lambda_stat is not None else "- Lambda = n/a",
            f"- p_boot = {result.p_boot:.4f}" if result.p_boot is not None else "- p_boot = n/a",
            f"- p_wilks = {result.p_wilks:.4f}" if result.p_wilks is not None else "- p_wilks = n/a",
        ]
    )

    return "\n".join(lines) + "\n"
