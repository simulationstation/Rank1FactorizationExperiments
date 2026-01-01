#!/usr/bin/env python3
"""
CMS Rank-1 Bottleneck Test

Tests if the complex coupling ratio R = c2/c1 is shared between two channels
from CMS 4-muon / di-J/psi mass spectra.

Channel A: FourMu MC or real data selection A
Channel B: Di-J/psi MC or real data selection B

Usage:
  python3 cms_rank1_test.py \
    --channel-a outputs/rank1_inputs/channelA.csv \
    --channel-b outputs/rank1_inputs/channelB.csv \
    --output reports/RANK1_LOCAL_VALIDATION.md
"""

import argparse
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Physical constants for resonance model
M1_INIT = 9.4    # eta_b mass (GeV)
G1_INIT = 0.01   # eta_b width (GeV)
M2_INIT = 10.0   # second resonance (GeV)
G2_INIT = 0.05   # second resonance width (GeV)


def load_channel_data(csv_file):
    """Load channel data from CSV."""
    data = []
    with open(csv_file, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                mass = float(parts[0])
                counts = float(parts[1])
                err = float(parts[2])
                if counts > 0 and err > 0:
                    data.append((mass, counts, err))
    return np.array(data)


def breit_wigner(m, M, Gamma):
    """Breit-Wigner amplitude."""
    s = m**2
    M2 = M**2
    return M * Gamma / (M2 - s + 1j * M * Gamma)


def model_amplitude(m, params, channel):
    """
    Two-resonance amplitude model.

    For constrained fit: R is shared between channels
    For unconstrained fit: R_A and R_B are independent
    """
    M1, G1, M2, G2 = params[:4]

    if channel == 'A':
        c1 = params[4]  # c1_A
        r = params[5]   # |R|
        phi = params[6] # arg(R)
        scale = params[7]  # scale_A
    else:  # channel B
        c1 = params[8]  # c1_B
        r = params[9]   # |R| (same as A for constrained, different for unconstrained)
        phi = params[10] # arg(R)
        scale = params[11]  # scale_B

    R = r * np.exp(1j * phi)

    BW1 = breit_wigner(m, M1, G1)
    BW2 = breit_wigner(m, M2, G2)

    amp = c1 * (BW1 + R * BW2)
    intensity = np.abs(amp)**2 * scale

    return intensity


def nll_channel(params, data, channel):
    """Negative log-likelihood for one channel."""
    nll = 0.0
    for m, n, err in data:
        pred = model_amplitude(m, params, channel)
        # Gaussian likelihood
        nll += 0.5 * ((n - pred) / err)**2
    return nll


def nll_total(params, data_A, data_B):
    """Total NLL for both channels."""
    return nll_channel(params, data_A, 'A') + nll_channel(params, data_B, 'B')


def fit_constrained(data_A, data_B, n_starts=100):
    """
    Constrained fit: R_A = R_B (shared coupling ratio).

    Parameters: [M1, G1, M2, G2, c1_A, r_shared, phi_shared, scale_A, c1_B, r_shared, phi_shared, scale_B]
    """
    bounds = [
        (8.5, 10.5),   # M1
        (0.001, 0.5),  # G1
        (9.5, 11.5),   # M2
        (0.01, 0.5),   # G2
        (0.1, 100),    # c1_A
        (0.1, 10),     # r_shared
        (-np.pi, np.pi),  # phi_shared
        (0.1, 1000),   # scale_A
        (0.1, 100),    # c1_B
        (0.1, 10),     # r_shared (same as element 5)
        (-np.pi, np.pi),  # phi_shared (same as element 6)
        (0.1, 1000),   # scale_B
    ]

    best_nll = np.inf
    best_params = None

    for _ in range(n_starts):
        x0 = []
        for lo, hi in bounds:
            x0.append(np.random.uniform(lo, hi))

        # Enforce constraint: r_B = r_A, phi_B = phi_A
        x0[9] = x0[5]
        x0[10] = x0[6]

        def constrained_nll(x):
            # Copy shared R from A to B
            x_full = x.copy()
            x_full[9] = x_full[5]
            x_full[10] = x_full[6]
            return nll_total(x_full, data_A, data_B)

        try:
            result = minimize(constrained_nll, x0, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
                best_params[9] = best_params[5]
                best_params[10] = best_params[6]
        except:
            pass

    return best_nll, best_params


def fit_unconstrained(data_A, data_B, n_starts=100, seed_params=None):
    """
    Unconstrained fit: R_A and R_B are independent.
    """
    bounds = [
        (8.5, 10.5),   # M1
        (0.001, 0.5),  # G1
        (9.5, 11.5),   # M2
        (0.01, 0.5),   # G2
        (0.1, 100),    # c1_A
        (0.1, 10),     # r_A
        (-np.pi, np.pi),  # phi_A
        (0.1, 1000),   # scale_A
        (0.1, 100),    # c1_B
        (0.1, 10),     # r_B
        (-np.pi, np.pi),  # phi_B
        (0.1, 1000),   # scale_B
    ]

    best_nll = np.inf
    best_params = None

    for i in range(n_starts):
        if seed_params is not None and i == 0:
            x0 = seed_params.copy()
            # Add small perturbation to R_B
            x0[9] = x0[5] * np.random.uniform(0.8, 1.2)
            x0[10] = x0[6] + np.random.uniform(-0.3, 0.3)
        else:
            x0 = []
            for lo, hi in bounds:
                x0.append(np.random.uniform(lo, hi))

        try:
            result = minimize(
                lambda x: nll_total(x, data_A, data_B),
                x0, method='L-BFGS-B', bounds=bounds
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except:
            pass

    return best_nll, best_params


def bootstrap_replicate(args):
    """Single bootstrap replicate."""
    data_A, data_B, seed = args
    np.random.seed(seed)

    # Resample with replacement
    idx_A = np.random.choice(len(data_A), len(data_A), replace=True)
    idx_B = np.random.choice(len(data_B), len(data_B), replace=True)

    boot_A = data_A[idx_A]
    boot_B = data_B[idx_B]

    # Fit constrained
    nll_con, params_con = fit_constrained(boot_A, boot_B, n_starts=30)

    # Fit unconstrained (seeded from constrained)
    nll_unc, params_unc = fit_unconstrained(boot_A, boot_B, n_starts=30, seed_params=params_con)

    # Ensure proper ordering
    if nll_unc > nll_con:
        nll_unc = nll_con

    lambda_boot = 2 * (nll_con - nll_unc)
    return max(0, lambda_boot)


def compute_chi2_dof(params, data, channel):
    """Compute chi2/dof for fit health check."""
    chi2 = 0.0
    for m, n, err in data:
        pred = model_amplitude(m, params, channel)
        chi2 += ((n - pred) / err)**2

    n_params = 6  # per channel: M1, G1, M2, G2, c1, R (complex)
    dof = len(data) - n_params
    return chi2 / max(1, dof)


def run_rank1_test(data_A, data_B, n_bootstrap=200):
    """Run the full rank-1 test."""
    print("=" * 60)
    print("CMS Rank-1 Bottleneck Test")
    print("=" * 60)
    print(f"\nData: {len(data_A)} (A), {len(data_B)} (B) points")

    # Fit constrained
    print("\n1) Fitting constrained (shared R)...")
    nll_con, params_con = fit_constrained(data_A, data_B, n_starts=100)
    print(f"   NLL_con = {nll_con:.2f}")

    r_shared = params_con[5]
    phi_shared = params_con[6]
    print(f"   R_shared = {r_shared:.3f} exp(i {phi_shared:.3f})")

    # Fit unconstrained (seeded from constrained)
    print("\n2) Fitting unconstrained (seeded from constrained)...")
    nll_unc, params_unc = fit_unconstrained(data_A, data_B, n_starts=100, seed_params=params_con)
    print(f"   NLL_unc = {nll_unc:.2f}")

    r_A = params_unc[5]
    phi_A = params_unc[6]
    r_B = params_unc[9]
    phi_B = params_unc[10]
    print(f"   R_A = {r_A:.3f} exp(i {phi_A:.3f})")
    print(f"   R_B = {r_B:.3f} exp(i {phi_B:.3f})")

    # Ensure proper ordering
    if nll_unc > nll_con:
        print("   WARNING: NLL_unc > NLL_con, setting NLL_unc = NLL_con")
        nll_unc = nll_con

    # Lambda
    Lambda = 2 * (nll_con - nll_unc)
    print(f"\n3) Lambda = 2*(NLL_con - NLL_unc) = {Lambda:.2f}")

    # Fit health
    chi2_A = compute_chi2_dof(params_unc, data_A, 'A')
    chi2_B = compute_chi2_dof(params_unc, data_B, 'B')
    print(f"\n4) Fit health:")
    print(f"   Channel A: chi2/dof = {chi2_A:.2f}")
    print(f"   Channel B: chi2/dof = {chi2_B:.2f}")

    health_pass = 0.5 < chi2_A < 5.0 and 0.5 < chi2_B < 5.0

    # Bootstrap
    print(f"\n5) Running bootstrap ({n_bootstrap} replicates)...")

    n_workers = max(1, cpu_count() - 1)
    args_list = [(data_A, data_B, i) for i in range(n_bootstrap)]

    with Pool(n_workers) as pool:
        lambda_boots = list(pool.map(bootstrap_replicate, args_list))

    p_value = np.mean([lb >= Lambda for lb in lambda_boots])
    print(f"   p-value = {p_value:.3f}")

    # Verdict
    print("\n" + "=" * 60)
    if not health_pass:
        verdict = "MODEL MISMATCH"
        reason = f"chi2/dof: A={chi2_A:.2f}, B={chi2_B:.2f}"
    elif Lambda < 3.84:  # chi2(1) 95% threshold
        verdict = "COMPATIBLE"
        reason = f"Lambda={Lambda:.2f} < 3.84"
    elif p_value > 0.05:
        verdict = "COMPATIBLE"
        reason = f"p={p_value:.3f} > 0.05"
    else:
        verdict = "DISFAVORED"
        reason = f"Lambda={Lambda:.2f}, p={p_value:.3f}"

    print(f"VERDICT: {verdict}")
    print(f"Reason: {reason}")
    print("=" * 60)

    return {
        'nll_con': nll_con,
        'nll_unc': nll_unc,
        'Lambda': Lambda,
        'p_value': p_value,
        'chi2_A': chi2_A,
        'chi2_B': chi2_B,
        'R_shared': (r_shared, phi_shared),
        'R_A': (r_A, phi_A),
        'R_B': (r_B, phi_B),
        'verdict': verdict,
        'reason': reason,
        'health_pass': health_pass,
    }


def generate_mock_channels(output_dir):
    """Generate mock channel data for testing."""
    np.random.seed(45)

    # Mass range (di-J/psi region)
    masses = np.linspace(7, 12, 50)

    # Channel A: eta_b peak + background
    signal_A = 500 * np.exp(-0.5 * ((masses - 9.4) / 0.15)**2)
    bkg_A = 100 * np.exp(-(masses - 7) / 2)
    counts_A = np.random.poisson(signal_A + bkg_A)
    err_A = np.sqrt(np.maximum(counts_A, 1))

    # Channel B: similar but slightly different R
    signal_B = 300 * np.exp(-0.5 * ((masses - 9.45) / 0.18)**2)
    bkg_B = 80 * np.exp(-(masses - 7) / 2.5)
    counts_B = np.random.poisson(signal_B + bkg_B)
    err_B = np.sqrt(np.maximum(counts_B, 1))

    # Write CSVs
    csv_A = f"{output_dir}/channelA.csv"
    csv_B = f"{output_dir}/channelB.csv"

    with open(csv_A, 'w') as f:
        f.write("mass_GeV,counts,stat_err\n")
        for m, c, e in zip(masses, counts_A, err_A):
            f.write(f"{m:.4f},{c},{e:.4f}\n")

    with open(csv_B, 'w') as f:
        f.write("mass_GeV,counts,stat_err\n")
        for m, c, e in zip(masses, counts_B, err_B):
            f.write(f"{m:.4f},{c},{e:.4f}\n")

    print(f"Generated mock data: {csv_A}, {csv_B}")
    return csv_A, csv_B


def main():
    parser = argparse.ArgumentParser(description="CMS Rank-1 Bottleneck Test")
    parser.add_argument("--channel-a", help="Channel A CSV file")
    parser.add_argument("--channel-b", help="Channel B CSV file")
    parser.add_argument("--output", default="RANK1_RESULT.md", help="Output report file")
    parser.add_argument("--bootstrap", type=int, default=200, help="Number of bootstrap replicates")
    parser.add_argument("--generate-mock", action="store_true", help="Generate mock data")
    parser.add_argument("--mock-dir", default="outputs/rank1_inputs", help="Directory for mock data")
    args = parser.parse_args()

    if args.generate_mock:
        csv_A, csv_B = generate_mock_channels(args.mock_dir)
        args.channel_a = csv_A
        args.channel_b = csv_B

    if not args.channel_a or not args.channel_b:
        print("ERROR: Must specify --channel-a and --channel-b, or use --generate-mock")
        return

    # Load data
    data_A = load_channel_data(args.channel_a)
    data_B = load_channel_data(args.channel_b)

    # Run test
    results = run_rank1_test(data_A, data_B, n_bootstrap=args.bootstrap)

    # Write report
    with open(args.output, 'w') as f:
        f.write("# CMS Rank-1 Bottleneck Test Results\n\n")
        f.write("## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Verdict** | {results['verdict']} |\n")
        f.write(f"| Reason | {results['reason']} |\n")
        f.write(f"| NLL (constrained) | {results['nll_con']:.2f} |\n")
        f.write(f"| NLL (unconstrained) | {results['nll_unc']:.2f} |\n")
        f.write(f"| Lambda | {results['Lambda']:.2f} |\n")
        f.write(f"| p-value | {results['p_value']:.3f} |\n")
        f.write(f"| chi2/dof (A) | {results['chi2_A']:.2f} |\n")
        f.write(f"| chi2/dof (B) | {results['chi2_B']:.2f} |\n")
        f.write(f"\n## Coupling Ratios\n\n")
        f.write(f"```\n")
        f.write(f"R_shared = {results['R_shared'][0]:.3f} exp(i {results['R_shared'][1]:.3f})\n")
        f.write(f"R_A      = {results['R_A'][0]:.3f} exp(i {results['R_A'][1]:.3f})\n")
        f.write(f"R_B      = {results['R_B'][0]:.3f} exp(i {results['R_B'][1]:.3f})\n")
        f.write(f"```\n")
        f.write(f"\n## Input Files\n\n")
        f.write(f"- Channel A: `{args.channel_a}`\n")
        f.write(f"- Channel B: `{args.channel_b}`\n")
        f.write(f"- Bootstrap replicates: {args.bootstrap}\n")

    print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
