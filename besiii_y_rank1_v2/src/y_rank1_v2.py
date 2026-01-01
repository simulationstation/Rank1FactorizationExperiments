#!/usr/bin/env python3
"""
BESIII Y-Sector Rank-1 Bottleneck Test v2
Uses 3 resonances for ππJ/ψ and 2 for ππh_c, tests rank-1 on shared {Y1,Y2} subspace.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
BASE = "/home/primary/DarkBItParticleColiderPredictions/besiii_y_rank1_v2"
EXTRACTED = f"{BASE}/data/extracted"
OUT = f"{BASE}/out"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Data Classes
# ============================================================
@dataclass
class DataPoint:
    E: float
    sigma: float
    stat_err: float
    syst_err: float

    @property
    def total_err(self):
        return np.sqrt(self.stat_err**2 + self.syst_err**2)

# ============================================================
# Load Data
# ============================================================
def load_data(filepath: str, E_min: float = 4.01, E_max: float = 4.60) -> List[DataPoint]:
    """Load CSV data and filter to energy range."""
    points = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            E = float(row['E_GeV'])
            if E_min <= E <= E_max:
                points.append(DataPoint(
                    E=E,
                    sigma=float(row['sigma_pb']),
                    stat_err=float(row['stat_err']),
                    syst_err=float(row.get('syst_err', 0))
                ))
    return points

# ============================================================
# Amplitude Model
# ============================================================
def breit_wigner(E: np.ndarray, M: float, Gamma: float) -> np.ndarray:
    """Complex Breit-Wigner: 1/(E² - M² + iMΓ)"""
    return 1.0 / (E**2 - M**2 + 1j * M * Gamma)

def amplitude_3res(E: np.ndarray, M1: float, G1: float, M2: float, G2: float,
                   M3: float, G3: float, c1: float, c2_re: float, c2_im: float,
                   c3_re: float, c3_im: float, bg_re: float, bg_im: float) -> np.ndarray:
    """3-resonance amplitude for Channel A (ππJ/ψ)."""
    c2 = c2_re + 1j * c2_im
    c3 = c3_re + 1j * c3_im
    bg = bg_re + 1j * bg_im

    return c1 * breit_wigner(E, M1, G1) + c2 * breit_wigner(E, M2, G2) + \
           c3 * breit_wigner(E, M3, G3) + bg

def amplitude_2res(E: np.ndarray, M1: float, G1: float, M2: float, G2: float,
                   c1: float, c2_re: float, c2_im: float,
                   bg_re: float, bg_im: float) -> np.ndarray:
    """2-resonance amplitude for Channel B (ππh_c)."""
    c2 = c2_re + 1j * c2_im
    bg = bg_re + 1j * bg_im

    return c1 * breit_wigner(E, M1, G1) + c2 * breit_wigner(E, M2, G2) + bg

def cross_section(A: np.ndarray, s0: float = 1.0, s1: float = 0.0, E: np.ndarray = None, E0: float = 4.3) -> np.ndarray:
    """Cross section with nuisance parameters."""
    base = np.abs(A)**2
    if E is not None:
        return s0 * (1 + s1 * (E - E0)) * base
    return s0 * base

# ============================================================
# Fitting Functions
# ============================================================
def gaussian_nll(data: List[DataPoint], sigma_model: np.ndarray) -> float:
    """Gaussian NLL using stat errors."""
    nll = 0
    for i, p in enumerate(data):
        nll += 0.5 * ((p.sigma - sigma_model[i]) / p.stat_err)**2
    return nll

def chi2_stat(data: List[DataPoint], sigma_model: np.ndarray) -> float:
    """Chi-squared statistic."""
    return sum(((p.sigma - sigma_model[i]) / p.stat_err)**2 for i, p in enumerate(data))

def fit_channel_A(data: List[DataPoint], M1: float, G1: float, M2: float, G2: float,
                  n_starts: int = 200) -> Tuple[float, Dict]:
    """Fit Channel A (ππJ/ψ) with 3 resonances, shared M1,G1,M2,G2."""
    E = np.array([p.E for p in data])
    E0 = np.mean(E)

    def objective(params):
        M3, G3, c1, c2_re, c2_im, c3_re, c3_im, bg_re, bg_im, s0, s1 = params

        if G3 <= 0 or c1 <= 0 or s0 <= 0:
            return 1e10
        if M3 < 4.3 or M3 > 4.6:
            return 1e10

        A = amplitude_3res(E, M1, G1, M2, G2, M3, G3, c1, c2_re, c2_im, c3_re, c3_im, bg_re, bg_im)
        sigma = cross_section(A, s0, s1, E, E0)

        if np.any(~np.isfinite(sigma)) or np.any(sigma < 0):
            return 1e10

        nll = gaussian_nll(data, sigma)

        # Priors on Y3
        nll += 0.5 * ((M3 - 4.42) / 0.05)**2
        nll += 0.5 * ((G3 - 0.200) / 0.150)**2
        nll += 0.5 * ((s0 - 1.0) / 0.10)**2
        nll += 0.5 * (s1 / 0.02)**2

        return nll

    # Initial guess
    x0 = [4.42, 0.15, 200, 100, 50, 80, 30, 5, 0, 1.0, 0]
    bounds = [
        (4.35, 4.55),   # M3
        (0.05, 0.40),   # G3
        (1, 1000),      # c1
        (-500, 500),    # c2_re
        (-500, 500),    # c2_im
        (-500, 500),    # c3_re
        (-500, 500),    # c3_im
        (-50, 50),      # bg_re
        (-50, 50),      # bg_im
        (0.7, 1.3),     # s0
        (-0.1, 0.1),    # s1
    ]

    best_nll = np.inf
    best_params = None

    for _ in range(n_starts):
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except:
            pass

        x0 = [
            np.random.uniform(4.38, 4.50),
            np.random.uniform(0.10, 0.30),
            np.random.uniform(50, 400),
            np.random.uniform(-200, 200),
            np.random.uniform(-150, 150),
            np.random.uniform(-150, 150),
            np.random.uniform(-100, 100),
            np.random.uniform(-20, 20),
            np.random.uniform(-20, 20),
            np.random.uniform(0.9, 1.1),
            np.random.uniform(-0.03, 0.03),
        ]

    if best_params is None:
        return np.inf, {}

    p = best_params
    params = {
        'M3': p[0], 'G3': p[1], 'c1': p[2],
        'c2_re': p[3], 'c2_im': p[4], 'c3_re': p[5], 'c3_im': p[6],
        'bg_re': p[7], 'bg_im': p[8], 's0': p[9], 's1': p[10]
    }

    # Compute R_A = c2/c1
    c2 = p[3] + 1j * p[4]
    R_A = c2 / p[2]
    params['r_A'] = np.abs(R_A)
    params['phi_A'] = np.angle(R_A)

    return best_nll, params

def fit_channel_B(data: List[DataPoint], M1: float, G1: float, M2: float, G2: float,
                  n_starts: int = 200) -> Tuple[float, Dict]:
    """Fit Channel B (ππh_c) with 2 resonances, shared M1,G1,M2,G2."""
    E = np.array([p.E for p in data])
    E0 = np.mean(E)

    def objective(params):
        c1, c2_re, c2_im, bg_re, bg_im, s0, s1 = params

        if c1 <= 0 or s0 <= 0:
            return 1e10

        A = amplitude_2res(E, M1, G1, M2, G2, c1, c2_re, c2_im, bg_re, bg_im)
        sigma = cross_section(A, s0, s1, E, E0)

        if np.any(~np.isfinite(sigma)) or np.any(sigma < 0):
            return 1e10

        nll = gaussian_nll(data, sigma)
        nll += 0.5 * ((s0 - 1.0) / 0.10)**2
        nll += 0.5 * (s1 / 0.02)**2

        return nll

    x0 = [50, 30, 10, 2, 0, 1.0, 0]
    bounds = [
        (1, 500),       # c1
        (-300, 300),    # c2_re
        (-300, 300),    # c2_im
        (-30, 30),      # bg_re
        (-30, 30),      # bg_im
        (0.7, 1.3),     # s0
        (-0.1, 0.1),    # s1
    ]

    best_nll = np.inf
    best_params = None

    for _ in range(n_starts):
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except:
            pass

        x0 = [
            np.random.uniform(20, 150),
            np.random.uniform(-100, 100),
            np.random.uniform(-80, 80),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(0.9, 1.1),
            np.random.uniform(-0.03, 0.03),
        ]

    if best_params is None:
        return np.inf, {}

    p = best_params
    params = {
        'c1': p[0], 'c2_re': p[1], 'c2_im': p[2],
        'bg_re': p[3], 'bg_im': p[4], 's0': p[5], 's1': p[6]
    }

    c2 = p[1] + 1j * p[2]
    R_B = c2 / p[0]
    params['r_B'] = np.abs(R_B)
    params['phi_B'] = np.angle(R_B)

    return best_nll, params

def fit_joint_constrained(data_A: List[DataPoint], data_B: List[DataPoint],
                          n_starts: int = 500) -> Tuple[float, Dict, Dict, Dict]:
    """Joint fit with shared R = c2/c1 on {Y1,Y2} subspace."""
    E_A = np.array([p.E for p in data_A])
    E_B = np.array([p.E for p in data_B])
    E0_A, E0_B = np.mean(E_A), np.mean(E_B)

    def objective(params):
        (M1, G1, M2, G2, r_shared, phi_shared,
         M3, G3, c1_A, c3_re_A, c3_im_A, bg_re_A, bg_im_A, s0_A, s1_A,
         c1_B, bg_re_B, bg_im_B, s0_B, s1_B) = params

        if G1 <= 0 or G2 <= 0 or G3 <= 0 or r_shared < 0:
            return 1e10
        if c1_A <= 0 or c1_B <= 0 or s0_A <= 0 or s0_B <= 0:
            return 1e10

        # Compute c2 from shared ratio
        c2_A = r_shared * np.exp(1j * phi_shared) * c1_A
        c2_B = r_shared * np.exp(1j * phi_shared) * c1_B
        c3_A = c3_re_A + 1j * c3_im_A
        bg_A = bg_re_A + 1j * bg_im_A
        bg_B = bg_re_B + 1j * bg_im_B

        # Channel A: 3 resonances
        A_A = c1_A * breit_wigner(E_A, M1, G1) + c2_A * breit_wigner(E_A, M2, G2) + \
              c3_A * breit_wigner(E_A, M3, G3) + bg_A
        sigma_A = cross_section(A_A, s0_A, s1_A, E_A, E0_A)

        # Channel B: 2 resonances
        A_B = c1_B * breit_wigner(E_B, M1, G1) + c2_B * breit_wigner(E_B, M2, G2) + bg_B
        sigma_B = cross_section(A_B, s0_B, s1_B, E_B, E0_B)

        if np.any(~np.isfinite(sigma_A)) or np.any(~np.isfinite(sigma_B)):
            return 1e10

        nll = gaussian_nll(data_A, sigma_A) + gaussian_nll(data_B, sigma_B)

        # Priors
        nll += 0.5 * ((M1 - 4.222) / 0.010)**2
        nll += 0.5 * ((G1 - 0.050) / 0.030)**2
        nll += 0.5 * ((M2 - 4.330) / 0.040)**2
        nll += 0.5 * ((G2 - 0.150) / 0.120)**2
        nll += 0.5 * ((M3 - 4.42) / 0.05)**2
        nll += 0.5 * ((G3 - 0.200) / 0.150)**2
        nll += 0.5 * ((s0_A - 1.0) / 0.10)**2
        nll += 0.5 * ((s0_B - 1.0) / 0.10)**2

        return nll

    x0 = [4.222, 0.050, 4.330, 0.150, 0.5, 0.5,
          4.42, 0.15, 200, 50, 30, 5, 0, 1.0, 0,
          50, 2, 0, 1.0, 0]

    bounds = [
        (4.18, 4.28), (0.02, 0.12),   # M1, G1
        (4.25, 4.45), (0.05, 0.35),   # M2, G2
        (0.01, 5.0), (-np.pi, np.pi), # r_shared, phi_shared
        (4.35, 4.55), (0.05, 0.40),   # M3, G3
        (1, 1000), (-500, 500), (-500, 500), (-50, 50), (-50, 50), (0.7, 1.3), (-0.1, 0.1),  # A params
        (1, 500), (-30, 30), (-30, 30), (0.7, 1.3), (-0.1, 0.1),  # B params
    ]

    best_nll = np.inf
    best_params = None

    for _ in range(n_starts):
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except:
            pass

        x0 = [
            np.random.uniform(4.20, 4.25), np.random.uniform(0.03, 0.08),
            np.random.uniform(4.28, 4.40), np.random.uniform(0.08, 0.25),
            np.random.uniform(0.1, 2.0), np.random.uniform(-np.pi, np.pi),
            np.random.uniform(4.38, 4.50), np.random.uniform(0.10, 0.30),
            np.random.uniform(50, 400), np.random.uniform(-150, 150), np.random.uniform(-100, 100),
            np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(0.9, 1.1), np.random.uniform(-0.03, 0.03),
            np.random.uniform(20, 150), np.random.uniform(-10, 10), np.random.uniform(-10, 10),
            np.random.uniform(0.9, 1.1), np.random.uniform(-0.03, 0.03),
        ]

    if best_params is None:
        return np.inf, {}, {}, {}

    p = best_params
    params_shared = {
        'M1': p[0], 'G1': p[1], 'M2': p[2], 'G2': p[3],
        'r_shared': p[4], 'phi_shared': p[5]
    }
    params_A = {
        'M3': p[6], 'G3': p[7], 'c1': p[8],
        'c3_re': p[9], 'c3_im': p[10], 'bg_re': p[11], 'bg_im': p[12],
        's0': p[13], 's1': p[14]
    }
    params_B = {
        'c1': p[15], 'bg_re': p[16], 'bg_im': p[17], 's0': p[18], 's1': p[19]
    }

    return best_nll, params_shared, params_A, params_B

def fit_joint_unconstrained(data_A: List[DataPoint], data_B: List[DataPoint],
                            n_starts: int = 500) -> Tuple[float, Dict, Dict, Dict]:
    """Joint fit with independent R_A and R_B."""
    E_A = np.array([p.E for p in data_A])
    E_B = np.array([p.E for p in data_B])
    E0_A, E0_B = np.mean(E_A), np.mean(E_B)

    def objective(params):
        (M1, G1, M2, G2,
         M3, G3, c1_A, c2_re_A, c2_im_A, c3_re_A, c3_im_A, bg_re_A, bg_im_A, s0_A, s1_A,
         c1_B, c2_re_B, c2_im_B, bg_re_B, bg_im_B, s0_B, s1_B) = params

        if G1 <= 0 or G2 <= 0 or G3 <= 0:
            return 1e10
        if c1_A <= 0 or c1_B <= 0 or s0_A <= 0 or s0_B <= 0:
            return 1e10

        c2_A = c2_re_A + 1j * c2_im_A
        c3_A = c3_re_A + 1j * c3_im_A
        c2_B = c2_re_B + 1j * c2_im_B
        bg_A = bg_re_A + 1j * bg_im_A
        bg_B = bg_re_B + 1j * bg_im_B

        A_A = c1_A * breit_wigner(E_A, M1, G1) + c2_A * breit_wigner(E_A, M2, G2) + \
              c3_A * breit_wigner(E_A, M3, G3) + bg_A
        sigma_A = cross_section(A_A, s0_A, s1_A, E_A, E0_A)

        A_B = c1_B * breit_wigner(E_B, M1, G1) + c2_B * breit_wigner(E_B, M2, G2) + bg_B
        sigma_B = cross_section(A_B, s0_B, s1_B, E_B, E0_B)

        if np.any(~np.isfinite(sigma_A)) or np.any(~np.isfinite(sigma_B)):
            return 1e10

        nll = gaussian_nll(data_A, sigma_A) + gaussian_nll(data_B, sigma_B)

        nll += 0.5 * ((M1 - 4.222) / 0.010)**2
        nll += 0.5 * ((G1 - 0.050) / 0.030)**2
        nll += 0.5 * ((M2 - 4.330) / 0.040)**2
        nll += 0.5 * ((G2 - 0.150) / 0.120)**2
        nll += 0.5 * ((M3 - 4.42) / 0.05)**2
        nll += 0.5 * ((G3 - 0.200) / 0.150)**2
        nll += 0.5 * ((s0_A - 1.0) / 0.10)**2
        nll += 0.5 * ((s0_B - 1.0) / 0.10)**2

        return nll

    x0 = [4.222, 0.050, 4.330, 0.150,
          4.42, 0.15, 200, 100, 50, 50, 30, 5, 0, 1.0, 0,
          50, 30, 10, 2, 0, 1.0, 0]

    bounds = [
        (4.18, 4.28), (0.02, 0.12),
        (4.25, 4.45), (0.05, 0.35),
        (4.35, 4.55), (0.05, 0.40),
        (1, 1000), (-500, 500), (-500, 500), (-500, 500), (-500, 500), (-50, 50), (-50, 50), (0.7, 1.3), (-0.1, 0.1),
        (1, 500), (-300, 300), (-300, 300), (-30, 30), (-30, 30), (0.7, 1.3), (-0.1, 0.1),
    ]

    best_nll = np.inf
    best_params = None

    for _ in range(n_starts):
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except:
            pass

        x0 = [
            np.random.uniform(4.20, 4.25), np.random.uniform(0.03, 0.08),
            np.random.uniform(4.28, 4.40), np.random.uniform(0.08, 0.25),
            np.random.uniform(4.38, 4.50), np.random.uniform(0.10, 0.30),
            np.random.uniform(50, 400), np.random.uniform(-200, 200), np.random.uniform(-150, 150),
            np.random.uniform(-150, 150), np.random.uniform(-100, 100),
            np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(0.9, 1.1), np.random.uniform(-0.03, 0.03),
            np.random.uniform(20, 150), np.random.uniform(-100, 100), np.random.uniform(-80, 80),
            np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(0.9, 1.1), np.random.uniform(-0.03, 0.03),
        ]

    if best_params is None:
        return np.inf, {}, {}, {}

    p = best_params

    r_A = np.abs((p[7] + 1j * p[8]) / p[6])
    phi_A = np.angle((p[7] + 1j * p[8]) / p[6])
    r_B = np.abs((p[16] + 1j * p[17]) / p[15])
    phi_B = np.angle((p[16] + 1j * p[17]) / p[15])

    params_shared = {
        'M1': p[0], 'G1': p[1], 'M2': p[2], 'G2': p[3]
    }
    params_A = {
        'M3': p[4], 'G3': p[5], 'c1': p[6],
        'c2_re': p[7], 'c2_im': p[8], 'c3_re': p[9], 'c3_im': p[10],
        'bg_re': p[11], 'bg_im': p[12], 's0': p[13], 's1': p[14],
        'r_A': r_A, 'phi_A': phi_A
    }
    params_B = {
        'c1': p[15], 'c2_re': p[16], 'c2_im': p[17],
        'bg_re': p[18], 'bg_im': p[19], 's0': p[20], 's1': p[21],
        'r_B': r_B, 'phi_B': phi_B
    }

    return best_nll, params_shared, params_A, params_B

# ============================================================
# Bootstrap (Parallelized)
# ============================================================
_boot_data = {}

def _init_bootstrap(data_A, data_B, params_shared, params_A, params_B):
    global _boot_data
    _boot_data = {
        'data_A': data_A, 'data_B': data_B,
        'params_shared': params_shared,
        'params_A': params_A, 'params_B': params_B,
    }

def _bootstrap_worker(seed):
    np.random.seed(seed)
    d = _boot_data

    E_A = np.array([p.E for p in d['data_A']])
    E_B = np.array([p.E for p in d['data_B']])
    E0_A, E0_B = np.mean(E_A), np.mean(E_B)

    ps = d['params_shared']
    pA = d['params_A']
    pB = d['params_B']

    # Generate model predictions
    c2_A = ps['r_shared'] * np.exp(1j * ps['phi_shared']) * pA['c1']
    c2_B = ps['r_shared'] * np.exp(1j * ps['phi_shared']) * pB['c1']
    c3_A = pA['c3_re'] + 1j * pA['c3_im']
    bg_A = pA['bg_re'] + 1j * pA['bg_im']
    bg_B = pB['bg_re'] + 1j * pB['bg_im']

    A_A = pA['c1'] * breit_wigner(E_A, ps['M1'], ps['G1']) + \
          c2_A * breit_wigner(E_A, ps['M2'], ps['G2']) + \
          c3_A * breit_wigner(E_A, pA['M3'], pA['G3']) + bg_A
    sigma_A_model = cross_section(A_A, pA['s0'], pA['s1'], E_A, E0_A)

    A_B = pB['c1'] * breit_wigner(E_B, ps['M1'], ps['G1']) + \
          c2_B * breit_wigner(E_B, ps['M2'], ps['G2']) + bg_B
    sigma_B_model = cross_section(A_B, pB['s0'], pB['s1'], E_B, E0_B)

    # Generate pseudo-data
    pseudo_A = []
    for i, p in enumerate(d['data_A']):
        sigma_pseudo = max(0.1, np.random.normal(sigma_A_model[i], p.stat_err))
        pseudo_A.append(DataPoint(p.E, sigma_pseudo, p.stat_err, p.syst_err))

    pseudo_B = []
    for i, p in enumerate(d['data_B']):
        sigma_pseudo = max(0.1, np.random.normal(sigma_B_model[i], p.stat_err))
        pseudo_B.append(DataPoint(p.E, sigma_pseudo, p.stat_err, p.syst_err))

    try:
        nll_con, _, _, _ = fit_joint_constrained(pseudo_A, pseudo_B, n_starts=150)
        nll_unc, _, _, _ = fit_joint_unconstrained(pseudo_A, pseudo_B, n_starts=150)
        Lambda = 2 * (nll_con - nll_unc)
        return max(0, Lambda)
    except:
        return 0.0

def run_bootstrap(data_A, data_B, params_shared, params_A, params_B, n_boot=800):
    """Run parallelized bootstrap."""
    n_workers = max(1, cpu_count() - 1)
    seeds = list(range(42, 42 + n_boot))

    print(f"Running {n_boot} bootstrap replicates with {n_workers} workers...")

    with Pool(n_workers, initializer=_init_bootstrap,
              initargs=(data_A, data_B, params_shared, params_A, params_B)) as pool:
        Lambda_boot = list(pool.map(_bootstrap_worker, seeds))

    return np.array(Lambda_boot)

# ============================================================
# Plotting
# ============================================================
def plot_fit_A(data: List[DataPoint], params_shared: Dict, params_A: Dict, savepath: str):
    """Plot Channel A fit with 3 resonances."""
    E = np.array([p.E for p in data])
    sigma = np.array([p.sigma for p in data])
    err = np.array([p.stat_err for p in data])
    E0 = np.mean(E)

    E_fine = np.linspace(E.min(), E.max(), 500)

    c2 = params_shared['r_shared'] * np.exp(1j * params_shared['phi_shared']) * params_A['c1']
    c3 = params_A['c3_re'] + 1j * params_A['c3_im']
    bg = params_A['bg_re'] + 1j * params_A['bg_im']

    A = params_A['c1'] * breit_wigner(E_fine, params_shared['M1'], params_shared['G1']) + \
        c2 * breit_wigner(E_fine, params_shared['M2'], params_shared['G2']) + \
        c3 * breit_wigner(E_fine, params_A['M3'], params_A['G3']) + bg
    sigma_model = cross_section(A, params_A['s0'], params_A['s1'], E_fine, E0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(E, sigma, yerr=err, fmt='ko', markersize=4, capsize=2, label='Data')
    ax.plot(E_fine, sigma_model, 'r-', linewidth=2, label='3-resonance fit')
    ax.set_xlabel('$\\sqrt{s}$ (GeV)', fontsize=12)
    ax.set_ylabel('$\\sigma$ (pb)', fontsize=12)
    ax.set_title('$e^+e^- \\to \\pi^+\\pi^- J/\\psi$ (3 resonances: Y1+Y2+Y3)', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()

def plot_fit_B(data: List[DataPoint], params_shared: Dict, params_B: Dict, savepath: str):
    """Plot Channel B fit with 2 resonances."""
    E = np.array([p.E for p in data])
    sigma = np.array([p.sigma for p in data])
    err = np.array([p.stat_err for p in data])
    E0 = np.mean(E)

    E_fine = np.linspace(E.min(), E.max(), 500)

    c2 = params_shared['r_shared'] * np.exp(1j * params_shared['phi_shared']) * params_B['c1']
    bg = params_B['bg_re'] + 1j * params_B['bg_im']

    A = params_B['c1'] * breit_wigner(E_fine, params_shared['M1'], params_shared['G1']) + \
        c2 * breit_wigner(E_fine, params_shared['M2'], params_shared['G2']) + bg
    sigma_model = cross_section(A, params_B['s0'], params_B['s1'], E_fine, E0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(E, sigma, yerr=err, fmt='ko', markersize=4, capsize=2, label='Data')
    ax.plot(E_fine, sigma_model, 'b-', linewidth=2, label='2-resonance fit')
    ax.set_xlabel('$\\sqrt{s}$ (GeV)', fontsize=12)
    ax.set_ylabel('$\\sigma$ (pb)', fontsize=12)
    ax.set_title('$e^+e^- \\to \\pi^+\\pi^- h_c$ (2 resonances: Y1+Y2)', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()

def plot_bootstrap_hist(Lambda_boot: np.ndarray, Lambda_obs: float, savepath: str):
    """Plot bootstrap distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(Lambda_boot, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(Lambda_obs, color='red', linestyle='--', linewidth=2,
               label=f'Observed $\\Lambda$ = {Lambda_obs:.2f}')
    ax.set_xlabel('$\\Lambda$', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Bootstrap Distribution (Rank-1 on {Y1,Y2} subspace)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()

def plot_contours(r_A, phi_A, r_B, phi_B, r_shared, phi_shared, savepath: str):
    """Plot coupling ratio comparison."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.scatter(phi_A, r_A, s=200, c='blue', marker='o', label=f'$\\pi\\pi J/\\psi$: r={r_A:.2f}')
    ax.scatter(phi_B, r_B, s=200, c='orange', marker='s', label=f'$\\pi\\pi h_c$: r={r_B:.2f}')
    ax.scatter(phi_shared, r_shared, s=300, c='red', marker='*', label=f'Shared: r={r_shared:.2f}')
    ax.set_title('Complex Ratio R = c$_2$/c$_1$ on {Y1,Y2} subspace', fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("BESIII Y-Sector Rank-1 Test v2")
    print("3-resonance model for ππJ/ψ, 2-resonance for ππh_c")
    print("=" * 60)
    print()

    # Load data
    print("Step 1: Loading data...")
    data_A = load_data(f"{EXTRACTED}/pipiJpsi_points.csv")
    data_B = load_data(f"{EXTRACTED}/pipihc_points.csv")
    print(f"  Channel A (ππJ/ψ): {len(data_A)} points in [4.01, 4.60] GeV")
    print(f"  Channel B (ππh_c): {len(data_B)} points in [4.01, 4.60] GeV")
    print()

    # Joint fits
    print("Step 2: Running joint fits (500 starts each)...")
    print("  Unconstrained fit (independent R_A, R_B)...")
    nll_unc, params_shared_unc, params_A_unc, params_B_unc = fit_joint_unconstrained(data_A, data_B, n_starts=500)
    print(f"    NLL_unc = {nll_unc:.2f}")
    print(f"    R_A = {params_A_unc['r_A']:.3f} exp(i {params_A_unc['phi_A']:.3f})")
    print(f"    R_B = {params_B_unc['r_B']:.3f} exp(i {params_B_unc['phi_B']:.3f})")

    print("  Constrained fit (shared R)...")
    nll_con, params_shared_con, params_A_con, params_B_con = fit_joint_constrained(data_A, data_B, n_starts=500)
    print(f"    NLL_con = {nll_con:.2f}")
    print(f"    R_shared = {params_shared_con['r_shared']:.3f} exp(i {params_shared_con['phi_shared']:.3f})")

    Lambda_obs = 2 * (nll_con - nll_unc)
    Lambda_obs = max(0, Lambda_obs)
    print(f"  Λ = 2*(NLL_con - NLL_unc) = {Lambda_obs:.2f}")
    print()

    # Compute chi2/dof
    E_A = np.array([p.E for p in data_A])
    E_B = np.array([p.E for p in data_B])
    E0_A, E0_B = np.mean(E_A), np.mean(E_B)

    # Chi2 for constrained fit
    c2_A = params_shared_con['r_shared'] * np.exp(1j * params_shared_con['phi_shared']) * params_A_con['c1']
    c3_A = params_A_con['c3_re'] + 1j * params_A_con['c3_im']
    bg_A = params_A_con['bg_re'] + 1j * params_A_con['bg_im']

    A_A = params_A_con['c1'] * breit_wigner(E_A, params_shared_con['M1'], params_shared_con['G1']) + \
          c2_A * breit_wigner(E_A, params_shared_con['M2'], params_shared_con['G2']) + \
          c3_A * breit_wigner(E_A, params_A_con['M3'], params_A_con['G3']) + bg_A
    sigma_A_model = cross_section(A_A, params_A_con['s0'], params_A_con['s1'], E_A, E0_A)
    chi2_A = chi2_stat(data_A, sigma_A_model)
    ndof_A = len(data_A) - 11  # 11 params for A
    chi2_dof_A = chi2_A / max(ndof_A, 1)

    c2_B = params_shared_con['r_shared'] * np.exp(1j * params_shared_con['phi_shared']) * params_B_con['c1']
    bg_B = params_B_con['bg_re'] + 1j * params_B_con['bg_im']

    A_B = params_B_con['c1'] * breit_wigner(E_B, params_shared_con['M1'], params_shared_con['G1']) + \
          c2_B * breit_wigner(E_B, params_shared_con['M2'], params_shared_con['G2']) + bg_B
    sigma_B_model = cross_section(A_B, params_B_con['s0'], params_B_con['s1'], E_B, E0_B)
    chi2_B = chi2_stat(data_B, sigma_B_model)
    ndof_B = len(data_B) - 5  # 5 params for B
    chi2_dof_B = chi2_B / max(ndof_B, 1)

    health_A = 0.5 < chi2_dof_A < 3.0
    health_B = 0.5 < chi2_dof_B < 3.0

    print("Step 3: Fit health check...")
    print(f"  Channel A: χ²/dof = {chi2_dof_A:.2f}, Health: {'PASS' if health_A else 'FAIL'}")
    print(f"  Channel B: χ²/dof = {chi2_dof_B:.2f}, Health: {'PASS' if health_B else 'FAIL'}")
    print()

    # Bootstrap
    print("Step 4: Running bootstrap (800 replicates)...")
    Lambda_boot = run_bootstrap(data_A, data_B, params_shared_con, params_A_con, params_B_con, n_boot=800)
    p_value = np.mean(Lambda_boot >= Lambda_obs)
    print(f"  Bootstrap p-value: {p_value:.3f}")
    print()

    # Plots
    print("Step 5: Generating plots...")
    plot_fit_A(data_A, params_shared_con, params_A_con, f"{OUT}/fit_A.png")
    plot_fit_B(data_B, params_shared_con, params_B_con, f"{OUT}/fit_B.png")
    plot_bootstrap_hist(Lambda_boot, Lambda_obs, f"{OUT}/bootstrap_hist.png")
    plot_contours(params_A_unc['r_A'], params_A_unc['phi_A'],
                  params_B_unc['r_B'], params_B_unc['phi_B'],
                  params_shared_con['r_shared'], params_shared_con['phi_shared'],
                  f"{OUT}/contours_overlay.png")
    print(f"  Saved: fit_A.png, fit_B.png, bootstrap_hist.png, contours_overlay.png")
    print()

    # Verdict
    print("Step 6: Determining verdict...")
    if not health_A or not health_B:
        verdict = "MODEL MISMATCH"
        reason = f"Fit health failed: A={chi2_dof_A:.2f}, B={chi2_dof_B:.2f}"
    elif Lambda_obs < 0:
        verdict = "OPTIMIZER FAILURE"
        reason = "Λ < 0"
    elif p_value > 0.05:
        verdict = "SUPPORTED"
        reason = f"p = {p_value:.3f} > 0.05"
    else:
        verdict = "DISFAVORED"
        reason = f"p = {p_value:.3f} < 0.05"

    print(f"  Verdict: {verdict}")
    print(f"  Reason: {reason}")
    print()

    # Report
    print("Step 7: Generating REPORT.md...")
    report = f"""# BESIII Y-Sector Rank-1 Test v2

**Generated**: 2025-12-30
**Model**: 3 resonances for ππJ/ψ, 2 for ππh_c
**Test**: Rank-1 on shared {{Y1, Y2}} subspace

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Verdict** | **{verdict}** |
| Reason | {reason} |
| Λ = 2ΔlnL | {Lambda_obs:.2f} |
| Bootstrap p | {p_value:.3f} |
| Replicates | 800 |

---

## 2. Data

| Channel | Points | Range |
|---------|--------|-------|
| π⁺π⁻ J/ψ | {len(data_A)} | 4.01-4.60 GeV |
| π⁺π⁻ h_c | {len(data_B)} | 4.01-4.60 GeV |

---

## 3. Fit Health

| Channel | χ²/dof | Gate [0.5, 3.0] | Health |
|---------|--------|-----------------|--------|
| π⁺π⁻ J/ψ | {chi2_dof_A:.2f} | {'PASS' if 0.5 < chi2_dof_A < 3.0 else 'FAIL'} | {'PASS' if health_A else 'FAIL'} |
| π⁺π⁻ h_c | {chi2_dof_B:.2f} | {'PASS' if 0.5 < chi2_dof_B < 3.0 else 'FAIL'} | {'PASS' if health_B else 'FAIL'} |

---

## 4. Resonance Parameters (Constrained Fit)

### Shared {Y1, Y2}

| Resonance | Mass (MeV) | Width (MeV) |
|-----------|------------|-------------|
| Y1 | {params_shared_con['M1']*1000:.1f} | {params_shared_con['G1']*1000:.1f} |
| Y2 | {params_shared_con['M2']*1000:.1f} | {params_shared_con['G2']*1000:.1f} |

### Channel A only: Y3

| Resonance | Mass (MeV) | Width (MeV) |
|-----------|------------|-------------|
| Y3 | {params_A_con['M3']*1000:.1f} | {params_A_con['G3']*1000:.1f} |

---

## 5. Coupling Ratios

### On shared {{Y1, Y2}} subspace: R = c₂/c₁

| Source | |R| | arg(R) (rad) | arg(R) (deg) |
|--------|-----|--------------|--------------|
| π⁺π⁻ J/ψ (unconstrained) | {params_A_unc['r_A']:.3f} | {params_A_unc['phi_A']:.3f} | {np.degrees(params_A_unc['phi_A']):.1f}° |
| π⁺π⁻ h_c (unconstrained) | {params_B_unc['r_B']:.3f} | {params_B_unc['phi_B']:.3f} | {np.degrees(params_B_unc['phi_B']):.1f}° |
| **Shared (constrained)** | **{params_shared_con['r_shared']:.3f}** | **{params_shared_con['phi_shared']:.3f}** | **{np.degrees(params_shared_con['phi_shared']):.1f}°** |

---

## 6. Likelihood Ratio Test

| Fit | NLL |
|-----|-----|
| Constrained (R_A = R_B) | {nll_con:.2f} |
| Unconstrained | {nll_unc:.2f} |
| **Λ = 2ΔlnL** | **{Lambda_obs:.2f}** |

---

## 7. Final Verdict

### **{verdict}**

{reason}

### Physical Interpretation
"""

    if verdict == "SUPPORTED":
        report += """
The rank-1 hypothesis is **SUPPORTED** on the shared {Y1, Y2} subspace.
The coupling ratio R = c₂/c₁ is consistent between the two channels,
suggesting the same Y-states contribute with a universal relative coupling.
"""
    elif verdict == "DISFAVORED":
        report += """
The rank-1 hypothesis is **DISFAVORED**. The coupling ratios differ between channels,
suggesting different resonance dynamics or additional structure.
"""
    else:
        report += f"""
The test is **{verdict}**. {reason}
"""

    report += f"""
---

## 8. Output Files

| File | Description |
|------|-------------|
| `out/fit_A.png` | π⁺π⁻ J/ψ fit (3 resonances) |
| `out/fit_B.png` | π⁺π⁻ h_c fit (2 resonances) |
| `out/bootstrap_hist.png` | Bootstrap Λ distribution |
| `out/contours_overlay.png` | Coupling ratio comparison |

---

*Report generated by BESIII Y-sector rank-1 test v2*
"""

    with open(f"{OUT}/REPORT.md", 'w') as f:
        f.write(report)
    print(f"  Saved: {OUT}/REPORT.md")
    print()

    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print(f"Verdict: {verdict}")
    print(f"χ²/dof: A = {chi2_dof_A:.2f}, B = {chi2_dof_B:.2f}")
    print(f"R_A = {params_A_unc['r_A']:.3f} exp(i {params_A_unc['phi_A']:.3f})")
    print(f"R_B = {params_B_unc['r_B']:.3f} exp(i {params_B_unc['phi_B']:.3f})")
    print(f"R_shared = {params_shared_con['r_shared']:.3f} exp(i {params_shared_con['phi_shared']:.3f})")
    print(f"Λ = {Lambda_obs:.2f}, p = {p_value:.3f}")

if __name__ == "__main__":
    main()
