#!/usr/bin/env python3
"""
Rank-1 Injection/Recovery Module

Generates synthetic two-channel spectra from a two-BW interference model
for validation of the rank-1 bottleneck test statistic.

Can generate:
- Rank-1 TRUE scenarios (shared R across channels)
- Rank-1 FALSE scenarios (different R_A != R_B)

Usage:
  python3 rank1_injection.py --scenario rank1_true --outdir outputs/injection
  python3 rank1_injection.py --scenario rank1_false --outdir outputs/injection
  python3 rank1_injection.py --config custom_config.json --outdir outputs/injection
"""

import argparse
import json
import numpy as np
from typing import Tuple, Dict, Optional
import os


# Default physical parameters
DEFAULT_CONFIG = {
    # Resonance 1 (e.g., eta_b or similar)
    "M1": 9.4,      # Mass in GeV
    "Gamma1": 0.02,  # Width in GeV

    # Resonance 2 (second state)
    "M2": 10.0,     # Mass in GeV
    "Gamma2": 0.05,  # Width in GeV

    # Mass range for spectrum
    "mass_min": 6.0,
    "mass_max": 15.0,
    "n_bins": 90,

    # Coupling and scale parameters (per channel)
    "c1_A": 10.0,       # Overall coupling channel A
    "scale_A": 100.0,   # Scale factor A
    "c1_B": 8.0,        # Overall coupling channel B
    "scale_B": 80.0,    # Scale factor B

    # Background parameters (exponential)
    "bkg_A": 50.0,
    "bkg_slope_A": 0.3,
    "bkg_B": 40.0,
    "bkg_slope_B": 0.25,

    # Complex R values for injection
    # Rank-1 TRUE: R_A = R_B
    "R_true": {"r": 0.6, "phi": -0.9},

    # Rank-1 FALSE: R_A != R_B
    "R_A_false": {"r": 0.6, "phi": -0.9},
    "R_B_false": {"r": 0.9, "phi": 0.6},
}


def breit_wigner_amplitude(m: np.ndarray, M: float, Gamma: float) -> np.ndarray:
    """
    Relativistic Breit-Wigner amplitude.

    A(m) = M * Gamma / (M^2 - m^2 + i*M*Gamma)
    """
    s = m**2
    M2 = M**2
    return M * Gamma / (M2 - s + 1j * M * Gamma)


def interference_intensity(
    m: np.ndarray,
    M1: float, Gamma1: float,
    M2: float, Gamma2: float,
    c1: float, R: complex, scale: float
) -> np.ndarray:
    """
    Two-resonance interference intensity.

    I(m) = |c1 * (BW1 + R * BW2)|^2 * scale
    """
    BW1 = breit_wigner_amplitude(m, M1, Gamma1)
    BW2 = breit_wigner_amplitude(m, M2, Gamma2)

    amplitude = c1 * (BW1 + R * BW2)
    intensity = np.abs(amplitude)**2 * scale

    return intensity


def exponential_background(m: np.ndarray, A: float, slope: float, m0: float) -> np.ndarray:
    """Exponential background: A * exp(-slope * (m - m0))"""
    return A * np.exp(-slope * (m - m0))


def generate_spectrum(
    config: Dict,
    R_A: complex,
    R_B: complex,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic spectra for channels A and B.

    Returns:
        masses: bin centers
        counts_A: Poisson-fluctuated counts for channel A
        counts_B: Poisson-fluctuated counts for channel B
        errors_A/B: Statistical errors
    """
    if seed is not None:
        np.random.seed(seed)

    # Mass bins
    edges = np.linspace(config["mass_min"], config["mass_max"], config["n_bins"] + 1)
    masses = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    # Signal intensities
    signal_A = interference_intensity(
        masses,
        config["M1"], config["Gamma1"],
        config["M2"], config["Gamma2"],
        config["c1_A"], R_A, config["scale_A"]
    )

    signal_B = interference_intensity(
        masses,
        config["M1"], config["Gamma1"],
        config["M2"], config["Gamma2"],
        config["c1_B"], R_B, config["scale_B"]
    )

    # Background
    bkg_A = exponential_background(masses, config["bkg_A"], config["bkg_slope_A"], config["mass_min"])
    bkg_B = exponential_background(masses, config["bkg_B"], config["bkg_slope_B"], config["mass_min"])

    # Expected counts (multiply by bin width for rate -> counts)
    mu_A = (signal_A + bkg_A) * bin_width
    mu_B = (signal_B + bkg_B) * bin_width

    # Ensure non-negative
    mu_A = np.maximum(mu_A, 0.1)
    mu_B = np.maximum(mu_B, 0.1)

    # Poisson fluctuations
    counts_A = np.random.poisson(mu_A)
    counts_B = np.random.poisson(mu_B)

    # Statistical errors (sqrt of counts, min 1)
    errors_A = np.sqrt(np.maximum(counts_A, 1))
    errors_B = np.sqrt(np.maximum(counts_B, 1))

    return masses, counts_A, errors_A, counts_B, errors_B, mu_A, mu_B


def write_channel_csv(
    filepath: str,
    masses: np.ndarray,
    counts: np.ndarray,
    errors: np.ndarray
):
    """Write channel data to CSV in expected format."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("mass_GeV,counts,stat_err\n")
        for m, c, e in zip(masses, counts, errors):
            f.write(f"{m:.4f},{c},{e:.4f}\n")


def generate_injection_scenario(
    scenario: str,
    config: Dict,
    outdir: str,
    seed: Optional[int] = None
) -> Dict:
    """
    Generate injection scenario.

    Args:
        scenario: "rank1_true" or "rank1_false"
        config: Configuration dictionary
        outdir: Output directory
        seed: Random seed

    Returns:
        Dictionary with true parameters and file paths
    """
    if scenario == "rank1_true":
        r = config["R_true"]["r"]
        phi = config["R_true"]["phi"]
        R_A = r * np.exp(1j * phi)
        R_B = R_A  # Same R
        is_rank1 = True
    elif scenario == "rank1_false":
        r_A = config["R_A_false"]["r"]
        phi_A = config["R_A_false"]["phi"]
        r_B = config["R_B_false"]["r"]
        phi_B = config["R_B_false"]["phi"]
        R_A = r_A * np.exp(1j * phi_A)
        R_B = r_B * np.exp(1j * phi_B)
        is_rank1 = False
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Generate spectra
    masses, counts_A, errors_A, counts_B, errors_B, mu_A, mu_B = generate_spectrum(
        config, R_A, R_B, seed=seed
    )

    # Write CSVs
    os.makedirs(outdir, exist_ok=True)
    csv_A = os.path.join(outdir, "channelA.csv")
    csv_B = os.path.join(outdir, "channelB.csv")

    write_channel_csv(csv_A, masses, counts_A, errors_A)
    write_channel_csv(csv_B, masses, counts_B, errors_B)

    return {
        "scenario": scenario,
        "is_rank1": is_rank1,
        "R_A_true": {"r": np.abs(R_A), "phi": np.angle(R_A)},
        "R_B_true": {"r": np.abs(R_B), "phi": np.angle(R_B)},
        "csv_A": csv_A,
        "csv_B": csv_B,
        "n_bins": len(masses),
        "total_counts_A": int(np.sum(counts_A)),
        "total_counts_B": int(np.sum(counts_B)),
        "expected_counts_A": float(np.sum(mu_A)),
        "expected_counts_B": float(np.sum(mu_B)),
    }


def run_injection_trials(
    scenario: str,
    config: Dict,
    n_trials: int,
    outdir: str,
    base_seed: int = 12345
) -> Dict:
    """
    Run multiple injection trials for power/type-I error estimation.

    Returns summary statistics.
    """
    results = []

    for trial in range(n_trials):
        seed = base_seed + trial
        trial_dir = os.path.join(outdir, f"trial_{trial:03d}")

        # Generate injection data
        injection_info = generate_injection_scenario(scenario, config, trial_dir, seed=seed)
        injection_info["trial"] = trial
        injection_info["seed"] = seed
        results.append(injection_info)

    return {
        "scenario": scenario,
        "n_trials": n_trials,
        "trials": results,
    }


def load_config(config_path: Optional[str]) -> Dict:
    """Load configuration from JSON file or return defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        # Merge with defaults
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)
        return config
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict, filepath: str):
    """Save configuration to JSON."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Rank-1 Injection/Recovery Module")
    parser.add_argument("--scenario", choices=["rank1_true", "rank1_false"],
                        default="rank1_true", help="Injection scenario")
    parser.add_argument("--outdir", default="outputs/injection",
                        help="Output directory")
    parser.add_argument("--config", help="Path to custom config JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-trials", type=int, default=1,
                        help="Number of trials (for batch mode)")
    parser.add_argument("--save-config", action="store_true",
                        help="Save default config to outdir")

    args = parser.parse_args()

    config = load_config(args.config)

    if args.save_config:
        config_path = os.path.join(args.outdir, "injection_config.json")
        save_config(config, config_path)
        print(f"Saved config to: {config_path}")

    if args.n_trials == 1:
        # Single injection
        info = generate_injection_scenario(
            args.scenario, config, args.outdir, seed=args.seed
        )
        print(f"Generated {args.scenario} injection:")
        print(f"  Channel A: {info['csv_A']} ({info['total_counts_A']} counts)")
        print(f"  Channel B: {info['csv_B']} ({info['total_counts_B']} counts)")
        print(f"  R_A (true): {info['R_A_true']['r']:.3f} * exp(i * {info['R_A_true']['phi']:.3f})")
        print(f"  R_B (true): {info['R_B_true']['r']:.3f} * exp(i * {info['R_B_true']['phi']:.3f})")
        print(f"  Is rank-1: {info['is_rank1']}")
    else:
        # Batch mode
        results = run_injection_trials(
            args.scenario, config, args.n_trials, args.outdir, base_seed=args.seed
        )
        print(f"Generated {args.n_trials} {args.scenario} injection trials in {args.outdir}")


if __name__ == "__main__":
    main()
