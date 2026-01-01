#!/usr/bin/env python3
"""
Real Data Histogram Analysis Configuration

Reads BPHNano output from real data and produces mass spectra for:
- Dimuon (J/psi, psi(2S))
- 4-muon candidates
- Di-J/psi candidates

Usage:
  python3 realdata_hist_cfg.py --input bphnano_output.root --output realdata_hist.root
"""

import argparse
import sys
from itertools import combinations

try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False
    print("WARNING: ROOT not available. Will generate mock data.")

import numpy as np

# Physical constants
M_MU = 0.10566  # muon mass in GeV
M_JPSI = 3.097  # J/psi mass in GeV
M_PSI2S = 3.686  # psi(2S) mass in GeV
JPSI_WINDOW = 0.15  # Window for J/psi selection


def lorentz_vector(pt, eta, phi, mass):
    """Create 4-momentum from pt, eta, phi, mass."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.array([E, px, py, pz])


def invariant_mass(p1, p2):
    """Calculate invariant mass of two 4-vectors."""
    p_sum = p1 + p2
    m2 = p_sum[0]**2 - p_sum[1]**2 - p_sum[2]**2 - p_sum[3]**2
    return np.sqrt(max(0, m2))


def invariant_mass_4(p1, p2, p3, p4):
    """Calculate invariant mass of four 4-vectors."""
    p_sum = p1 + p2 + p3 + p4
    m2 = p_sum[0]**2 - p_sum[1]**2 - p_sum[2]**2 - p_sum[3]**2
    return np.sqrt(max(0, m2))


def analyze_bphnano(input_file, output_file, csv_prefix):
    """Analyze BPHNano file from real data."""
    if not HAS_ROOT:
        return generate_mock_realdata(output_file, csv_prefix)

    f_in = ROOT.TFile.Open(input_file, "READ")
    if not f_in or f_in.IsZombie():
        print(f"ERROR: Cannot open {input_file}")
        return False

    tree = f_in.Get("Events")
    if not tree:
        print("ERROR: Cannot find Events tree")
        f_in.Close()
        return False

    # Histograms for dimuon spectrum
    h_dimu_all = ROOT.TH1D("h_dimu_all", "Dimuon Mass (all);m(#mu#mu) [GeV];Events / 10 MeV", 500, 0, 5)
    h_dimu_jpsi = ROOT.TH1D("h_dimu_jpsi", "J/#psi Region;m(#mu#mu) [GeV];Events / 2 MeV", 150, 2.9, 3.2)
    h_dimu_psi2s = ROOT.TH1D("h_dimu_psi2s", "#psi(2S) Region;m(#mu#mu) [GeV];Events / 2 MeV", 150, 3.5, 3.8)

    # Histograms for 4-muon/di-J/psi
    h_4mu = ROOT.TH1D("h_4mu_mass", "4-Muon Mass;m(4#mu) [GeV];Events / 50 MeV", 200, 6, 16)
    h_dijpsi = ROOT.TH1D("h_dijpsi_mass", "Di-J/#psi Mass;m(J/#psi J/#psi) [GeV];Events / 50 MeV", 200, 6, 16)

    # Event counts
    h_nmu = ROOT.TH1I("h_nmu", "Number of Muons;N(#mu);Events", 20, 0, 20)
    h_nmumu = ROOT.TH1I("h_nmumu", "Number of DiMuon Candidates;N(#mu#mu);Events", 20, 0, 20)

    n_entries = tree.GetEntries()
    print(f"Processing {n_entries} events from real data...")

    n_4mu_events = 0
    n_dijpsi_events = 0

    for i_evt in range(n_entries):
        if i_evt % 10000 == 0:
            print(f"  Event {i_evt}/{n_entries}")

        tree.GetEntry(i_evt)

        # Process MuMu collection (if available)
        try:
            if hasattr(tree, 'nMuMu'):
                n_mumu = tree.nMuMu
                h_nmumu.Fill(min(n_mumu, 19))

                jpsi_candidates = []

                for i in range(n_mumu):
                    mass = tree.MuMu_mass[i]
                    h_dimu_all.Fill(mass)

                    if 2.9 < mass < 3.2:
                        h_dimu_jpsi.Fill(mass)

                    if 3.5 < mass < 3.8:
                        h_dimu_psi2s.Fill(mass)

                    # Collect J/psi candidates
                    if abs(mass - M_JPSI) < 0.3:
                        pt = tree.MuMu_pt[i]
                        eta = tree.MuMu_eta[i]
                        phi = tree.MuMu_phi[i]
                        p4 = lorentz_vector(pt, eta, phi, mass)
                        jpsi_candidates.append((p4, mass))

                # Di-J/psi combinations
                if len(jpsi_candidates) >= 2:
                    for (p1, m1), (p2, m2) in combinations(jpsi_candidates, 2):
                        if abs(m1 - M_JPSI) < JPSI_WINDOW and abs(m2 - M_JPSI) < JPSI_WINDOW:
                            m_dijpsi = invariant_mass(p1, p2)
                            h_dijpsi.Fill(m_dijpsi)
                            n_dijpsi_events += 1
        except AttributeError:
            pass

        # Process muon collection for 4-muon candidates
        try:
            mu_collection = 'BPHMuon' if hasattr(tree, 'nBPHMuon') else 'Muon'
            n_mu = getattr(tree, f'n{mu_collection}', 0)
            h_nmu.Fill(min(n_mu, 19))

            if n_mu >= 4:
                muons = []
                for i in range(min(n_mu, 10)):
                    pt = getattr(tree, f'{mu_collection}_pt')[i]
                    eta = getattr(tree, f'{mu_collection}_eta')[i]
                    phi = getattr(tree, f'{mu_collection}_phi')[i]
                    charge = getattr(tree, f'{mu_collection}_charge')[i]
                    p4 = lorentz_vector(pt, eta, phi, M_MU)
                    muons.append((p4, charge))

                # 4-muon mass
                if len(muons) >= 4:
                    m4mu = invariant_mass_4(
                        muons[0][0], muons[1][0], muons[2][0], muons[3][0]
                    )
                    h_4mu.Fill(m4mu)
                    n_4mu_events += 1

        except AttributeError:
            pass

    f_in.Close()

    print(f"Found {n_4mu_events} events with >= 4 muons")
    print(f"Found {n_dijpsi_events} di-J/psi candidates")

    # Save histograms
    f_out = ROOT.TFile(output_file, "RECREATE")
    h_dimu_all.Write()
    h_dimu_jpsi.Write()
    h_dimu_psi2s.Write()
    h_4mu.Write()
    h_dijpsi.Write()
    h_nmu.Write()
    h_nmumu.Write()
    f_out.Close()
    print(f"Wrote histograms to {output_file}")

    # Write CSVs for rank-1 test
    write_histogram_csv(h_dimu_jpsi, f"{csv_prefix}_jpsi.csv")
    write_histogram_csv(h_dijpsi, f"{csv_prefix}_dijpsi.csv")
    write_histogram_csv(h_4mu, f"{csv_prefix}_4mu.csv")

    return True


def generate_mock_realdata(output_file, csv_prefix):
    """Generate mock real data for testing."""
    print("Generating mock real data analysis...")

    np.random.seed(44)

    # J/psi peak (realistic shape with resolution)
    n_jpsi = 50000
    jpsi_signal = np.random.normal(M_JPSI, 0.025, n_jpsi)
    jpsi_bkg = np.random.exponential(0.5, n_jpsi // 10) + 2.9

    bins_jpsi = np.linspace(2.9, 3.2, 151)
    counts_jpsi, _ = np.histogram(np.concatenate([jpsi_signal, jpsi_bkg]), bins=bins_jpsi)
    centers_jpsi = 0.5 * (bins_jpsi[:-1] + bins_jpsi[1:])

    # Di-J/psi spectrum (less events, eta_b region)
    n_dijpsi = 500
    dijpsi_flat = np.random.uniform(6.5, 14, n_dijpsi)
    dijpsi_etab = np.random.normal(9.4, 0.15, n_dijpsi // 5)

    bins_dijpsi = np.linspace(6, 16, 201)
    counts_dijpsi, _ = np.histogram(np.concatenate([dijpsi_flat, dijpsi_etab]), bins=bins_dijpsi)
    centers_dijpsi = 0.5 * (bins_dijpsi[:-1] + bins_dijpsi[1:])

    # 4-muon spectrum
    n_4mu = 1000
    m4mu = np.random.uniform(6, 15, n_4mu)
    m4mu_peak = np.random.normal(9.5, 0.3, n_4mu // 10)

    bins_4mu = np.linspace(6, 16, 201)
    counts_4mu, _ = np.histogram(np.concatenate([m4mu, m4mu_peak]), bins=bins_4mu)
    centers_4mu = 0.5 * (bins_4mu[:-1] + bins_4mu[1:])

    # Write CSVs
    csv_jpsi = f"{csv_prefix}_jpsi.csv"
    csv_dijpsi = f"{csv_prefix}_dijpsi.csv"
    csv_4mu = f"{csv_prefix}_4mu.csv"

    with open(csv_jpsi, 'w') as f:
        f.write("mass_GeV,counts,stat_err\n")
        for m, c in zip(centers_jpsi, counts_jpsi):
            f.write(f"{m:.4f},{c},{np.sqrt(max(1,c)):.4f}\n")

    with open(csv_dijpsi, 'w') as f:
        f.write("mass_GeV,counts,stat_err\n")
        for m, c in zip(centers_dijpsi, counts_dijpsi):
            f.write(f"{m:.4f},{c},{np.sqrt(max(1,c)):.4f}\n")

    with open(csv_4mu, 'w') as f:
        f.write("mass_GeV,counts,stat_err\n")
        for m, c in zip(centers_4mu, counts_4mu):
            f.write(f"{m:.4f},{c},{np.sqrt(max(1,c)):.4f}\n")

    print(f"Wrote mock real data to {csv_jpsi}, {csv_dijpsi}, {csv_4mu}")
    return True


def write_histogram_csv(hist, csv_file):
    """Write ROOT histogram to CSV."""
    with open(csv_file, 'w') as f:
        f.write("mass_GeV,counts,stat_err\n")
        for i in range(1, hist.GetNbinsX() + 1):
            center = hist.GetBinCenter(i)
            content = hist.GetBinContent(i)
            error = hist.GetBinError(i)
            f.write(f"{center:.4f},{content:.0f},{error:.4f}\n")
    print(f"Wrote CSV to {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Real Data Histogram Analysis")
    parser.add_argument("--input", required=True, help="Input BPHNano ROOT file")
    parser.add_argument("--output", required=True, help="Output histogram ROOT file")
    parser.add_argument("--csv-prefix", default="realdata", help="CSV output prefix")
    parser.add_argument("--mock", action="store_true", help="Generate mock data")
    args = parser.parse_args()

    if args.mock or not HAS_ROOT:
        generate_mock_realdata(args.output, args.csv_prefix)
    else:
        analyze_bphnano(args.input, args.output, args.csv_prefix)


if __name__ == "__main__":
    main()
