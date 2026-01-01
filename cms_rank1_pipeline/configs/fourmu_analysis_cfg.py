#!/usr/bin/env python3
"""
FourMu Analysis Configuration

Reads NanoAOD or GEN-only ROOT files and produces:
- Dimuon mass histograms
- 4-muon invariant mass histograms
- CSV outputs for rank-1 test

Usage:
  python3 fourmu_analysis_cfg.py --input fourmu_nanoaod.root --output fourmu_hist.root
  python3 fourmu_analysis_cfg.py --input fourmu_gen.root --output fourmu_hist.root --gen-only
"""

import argparse
import sys
from itertools import combinations

# Try to import ROOT - if not available, generate mock data
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False
    print("WARNING: ROOT not available. Will generate mock data for testing.")

import numpy as np

# Physical constants
M_MU = 0.10566  # muon mass in GeV
M_JPSI = 3.097  # J/psi mass in GeV


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


def analyze_nanoaod(input_file, output_file, csv_file):
    """Analyze NanoAOD file with BPH collections."""
    if not HAS_ROOT:
        return generate_mock_analysis(output_file, csv_file)

    f_in = ROOT.TFile.Open(input_file, "READ")
    if not f_in or f_in.IsZombie():
        print(f"ERROR: Cannot open {input_file}")
        return False

    tree = f_in.Get("Events")
    if not tree:
        print("ERROR: Cannot find Events tree")
        f_in.Close()
        return False

    # Create output histograms
    h_dimu = ROOT.TH1D("h_dimu_mass", "Dimuon Mass;m(#mu#mu) [GeV];Events / 0.1 GeV", 150, 0, 15)
    h_4mu = ROOT.TH1D("h_4mu_mass", "4-Muon Mass;m(4#mu) [GeV];Events / 0.5 GeV", 100, 0, 50)
    h_dijpsi = ROOT.TH1D("h_dijpsi_mass", "Di-J/#psi Mass;m(J/#psi J/#psi) [GeV];Events / 0.25 GeV", 100, 0, 25)

    n_entries = tree.GetEntries()
    print(f"Processing {n_entries} events...")

    for i_evt in range(n_entries):
        tree.GetEntry(i_evt)

        # Get muons from BPHMuon collection
        try:
            n_mu = tree.nBPHMuon
            if n_mu < 2:
                continue

            muons = []
            for i in range(min(n_mu, 10)):  # limit to first 10
                pt = tree.BPHMuon_pt[i]
                eta = tree.BPHMuon_eta[i]
                phi = tree.BPHMuon_phi[i]
                charge = tree.BPHMuon_charge[i]
                p4 = lorentz_vector(pt, eta, phi, M_MU)
                muons.append((p4, charge))

            # Dimuon combinations (opposite sign)
            for (p1, q1), (p2, q2) in combinations(muons, 2):
                if q1 * q2 < 0:  # opposite sign
                    m12 = invariant_mass(p1, p2)
                    h_dimu.Fill(m12)

            # 4-muon combinations
            if len(muons) >= 4:
                for combo in combinations(range(len(muons)), 4):
                    m4mu = invariant_mass_4(
                        muons[combo[0]][0], muons[combo[1]][0],
                        muons[combo[2]][0], muons[combo[3]][0]
                    )
                    h_4mu.Fill(m4mu)

                    # Di-dimuon: pair into two opposite-sign dimuons
                    for pairing in [(0,1,2,3), (0,2,1,3), (0,3,1,2)]:
                        i1, i2, i3, i4 = [combo[p] for p in pairing]
                        if muons[i1][1] * muons[i2][1] < 0 and muons[i3][1] * muons[i4][1] < 0:
                            m_pair1 = invariant_mass(muons[i1][0], muons[i2][0])
                            m_pair2 = invariant_mass(muons[i3][0], muons[i4][0])
                            # If both near J/psi mass
                            if abs(m_pair1 - M_JPSI) < 0.3 and abs(m_pair2 - M_JPSI) < 0.3:
                                p_jpsi1 = muons[i1][0] + muons[i2][0]
                                p_jpsi2 = muons[i3][0] + muons[i4][0]
                                m_dijpsi = invariant_mass(p_jpsi1, p_jpsi2)
                                h_dijpsi.Fill(m_dijpsi)

        except AttributeError:
            # Fallback to slimmedMuons for standard NanoAOD
            try:
                n_mu = tree.nMuon
                if n_mu < 2:
                    continue
                muons = []
                for i in range(min(n_mu, 10)):
                    pt = tree.Muon_pt[i]
                    eta = tree.Muon_eta[i]
                    phi = tree.Muon_phi[i]
                    charge = tree.Muon_charge[i]
                    p4 = lorentz_vector(pt, eta, phi, M_MU)
                    muons.append((p4, charge))

                for (p1, q1), (p2, q2) in combinations(muons, 2):
                    if q1 * q2 < 0:
                        h_dimu.Fill(invariant_mass(p1, p2))
            except:
                pass

    f_in.Close()

    # Save histograms
    f_out = ROOT.TFile(output_file, "RECREATE")
    h_dimu.Write()
    h_4mu.Write()
    h_dijpsi.Write()
    f_out.Close()
    print(f"Wrote histograms to {output_file}")

    # Write CSV
    write_histogram_csv(h_dimu, csv_file.replace('.csv', '_dimu.csv'))
    write_histogram_csv(h_4mu, csv_file.replace('.csv', '_4mu.csv'))

    return True


def analyze_gen_only(input_file, output_file, csv_file):
    """Analyze GEN-only file (truth-level muons)."""
    if not HAS_ROOT:
        return generate_mock_analysis(output_file, csv_file)

    f_in = ROOT.TFile.Open(input_file, "READ")
    if not f_in or f_in.IsZombie():
        print(f"ERROR: Cannot open {input_file}")
        return False

    tree = f_in.Get("Events")
    if not tree:
        print("ERROR: Cannot find Events tree")
        f_in.Close()
        return False

    h_dimu = ROOT.TH1D("h_dimu_mass", "Dimuon Mass (GEN);m(#mu#mu) [GeV];Events", 150, 0, 15)
    h_4mu = ROOT.TH1D("h_4mu_mass", "4-Muon Mass (GEN);m(4#mu) [GeV];Events", 100, 0, 50)

    n_entries = tree.GetEntries()
    print(f"Processing {n_entries} GEN events...")

    for i_evt in range(n_entries):
        tree.GetEntry(i_evt)

        try:
            n_gen = tree.nGenPart
            muons = []

            for i in range(n_gen):
                pdg = abs(tree.GenPart_pdgId[i])
                status = tree.GenPart_status[i]

                if pdg == 13 and status == 1:  # final-state muon
                    pt = tree.GenPart_pt[i]
                    eta = tree.GenPart_eta[i]
                    phi = tree.GenPart_phi[i]
                    charge = -1 if tree.GenPart_pdgId[i] > 0 else 1
                    p4 = lorentz_vector(pt, eta, phi, M_MU)
                    muons.append((p4, charge))

            if len(muons) >= 2:
                for (p1, q1), (p2, q2) in combinations(muons, 2):
                    if q1 * q2 < 0:
                        h_dimu.Fill(invariant_mass(p1, p2))

            if len(muons) >= 4:
                # Take first 4 muons
                m4mu = invariant_mass_4(
                    muons[0][0], muons[1][0], muons[2][0], muons[3][0]
                )
                h_4mu.Fill(m4mu)

        except AttributeError:
            pass

    f_in.Close()

    f_out = ROOT.TFile(output_file, "RECREATE")
    h_dimu.Write()
    h_4mu.Write()
    f_out.Close()
    print(f"Wrote histograms to {output_file}")

    write_histogram_csv(h_dimu, csv_file.replace('.csv', '_dimu.csv'))
    write_histogram_csv(h_4mu, csv_file.replace('.csv', '_4mu.csv'))

    return True


def generate_mock_analysis(output_file, csv_file):
    """Generate mock data for testing without ROOT."""
    print("Generating mock FourMu analysis data...")

    np.random.seed(42)

    # Mock dimuon mass spectrum (flat + J/psi peak)
    n_events = 10000
    dimu_flat = np.random.uniform(0.5, 12, n_events)
    dimu_jpsi = np.random.normal(M_JPSI, 0.05, n_events // 5)
    dimu_all = np.concatenate([dimu_flat, dimu_jpsi])

    # Histogram
    bins_dimu = np.linspace(0, 15, 151)
    counts_dimu, _ = np.histogram(dimu_all, bins=bins_dimu)
    centers_dimu = 0.5 * (bins_dimu[:-1] + bins_dimu[1:])

    # Mock 4mu spectrum (broad distribution)
    m4mu = np.random.uniform(5, 40, n_events)
    bins_4mu = np.linspace(0, 50, 101)
    counts_4mu, _ = np.histogram(m4mu, bins=bins_4mu)
    centers_4mu = 0.5 * (bins_4mu[:-1] + bins_4mu[1:])

    # Write CSVs
    csv_dimu = csv_file.replace('.csv', '_dimu.csv')
    csv_4mu = csv_file.replace('.csv', '_4mu.csv')

    with open(csv_dimu, 'w') as f:
        f.write("mass_GeV,counts,stat_err\n")
        for m, c in zip(centers_dimu, counts_dimu):
            f.write(f"{m:.4f},{c},{np.sqrt(max(1,c)):.4f}\n")

    with open(csv_4mu, 'w') as f:
        f.write("mass_GeV,counts,stat_err\n")
        for m, c in zip(centers_4mu, counts_4mu):
            f.write(f"{m:.4f},{c},{np.sqrt(max(1,c)):.4f}\n")

    print(f"Wrote mock data to {csv_dimu} and {csv_4mu}")
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
    parser = argparse.ArgumentParser(description="FourMu Analysis")
    parser.add_argument("--input", required=True, help="Input ROOT file")
    parser.add_argument("--output", required=True, help="Output histogram ROOT file")
    parser.add_argument("--csv", default="fourmu_hist.csv", help="Output CSV file base")
    parser.add_argument("--gen-only", action="store_true", help="Use GEN-only mode")
    parser.add_argument("--mock", action="store_true", help="Generate mock data")
    args = parser.parse_args()

    if args.mock or not HAS_ROOT:
        generate_mock_analysis(args.output, args.csv)
    elif args.gen_only:
        analyze_gen_only(args.input, args.output, args.csv)
    else:
        analyze_nanoaod(args.input, args.output, args.csv)


if __name__ == "__main__":
    main()
