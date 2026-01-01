#!/usr/bin/env python3
"""
Di-J/psi (EtaB→J/ψJ/ψ) Analysis Configuration

Reads NanoAOD or GEN-only ROOT files and produces:
- J/psi mass histograms
- Di-J/psi mass histograms
- 4-muon invariant mass histograms
- CSV outputs for rank-1 test

Usage:
  python3 dijpsi_analysis_cfg.py --input dijpsi_nanoaod.root --output dijpsi_hist.root
  python3 dijpsi_analysis_cfg.py --input dijpsi_gen.root --output dijpsi_hist.root --gen-only
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
    print("WARNING: ROOT not available. Will generate mock data for testing.")

import numpy as np

# Physical constants
M_MU = 0.10566  # muon mass in GeV
M_JPSI = 3.097  # J/psi mass in GeV
M_ETAB = 9.4    # eta_b mass in GeV
JPSI_WINDOW = 0.15  # Window around J/psi mass for selection


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
    """Analyze NanoAOD file for di-J/psi."""
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

    # Create histograms
    h_jpsi = ROOT.TH1D("h_jpsi_mass", "J/#psi Mass;m(#mu#mu) [GeV];Events / 10 MeV", 100, 2.5, 3.5)
    h_dijpsi = ROOT.TH1D("h_dijpsi_mass", "Di-J/#psi Mass;m(J/#psi J/#psi) [GeV];Events / 50 MeV", 180, 6, 15)
    h_4mu = ROOT.TH1D("h_4mu_mass", "4-Muon Mass;m(4#mu) [GeV];Events / 50 MeV", 180, 6, 15)

    n_entries = tree.GetEntries()
    print(f"Processing {n_entries} events...")

    for i_evt in range(n_entries):
        tree.GetEntry(i_evt)

        try:
            # Try MuMu collection first (di-muon candidates from BPHNano)
            if hasattr(tree, 'nMuMu') and tree.nMuMu > 0:
                jpsi_candidates = []
                for i in range(tree.nMuMu):
                    mass = tree.MuMu_mass[i]
                    if abs(mass - M_JPSI) < 0.3:
                        pt = tree.MuMu_pt[i]
                        eta = tree.MuMu_eta[i]
                        phi = tree.MuMu_phi[i]
                        p4 = lorentz_vector(pt, eta, phi, mass)
                        jpsi_candidates.append((p4, mass))
                        h_jpsi.Fill(mass)

                # Di-J/psi
                if len(jpsi_candidates) >= 2:
                    for (p1, m1), (p2, m2) in combinations(jpsi_candidates, 2):
                        if abs(m1 - M_JPSI) < JPSI_WINDOW and abs(m2 - M_JPSI) < JPSI_WINDOW:
                            m_dijpsi = invariant_mass(p1, p2)
                            h_dijpsi.Fill(m_dijpsi)

            # Fallback: reconstruct from muons
            n_mu = getattr(tree, 'nBPHMuon', 0) or getattr(tree, 'nMuon', 0)
            if n_mu >= 4:
                muons = []
                mu_collection = 'BPHMuon' if hasattr(tree, 'nBPHMuon') else 'Muon'

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

                    # Try to pair into two J/psis
                    for pairing in [(0,1,2,3), (0,2,1,3), (0,3,1,2)]:
                        i1, i2, i3, i4 = pairing
                        if i4 < len(muons):
                            if muons[i1][1] * muons[i2][1] < 0 and muons[i3][1] * muons[i4][1] < 0:
                                m1 = invariant_mass(muons[i1][0], muons[i2][0])
                                m2 = invariant_mass(muons[i3][0], muons[i4][0])
                                h_jpsi.Fill(m1)
                                h_jpsi.Fill(m2)
                                if abs(m1 - M_JPSI) < JPSI_WINDOW and abs(m2 - M_JPSI) < JPSI_WINDOW:
                                    p_jpsi1 = muons[i1][0] + muons[i2][0]
                                    p_jpsi2 = muons[i3][0] + muons[i4][0]
                                    m_dijpsi = invariant_mass(p_jpsi1, p_jpsi2)
                                    h_dijpsi.Fill(m_dijpsi)

        except Exception as e:
            pass

    f_in.Close()

    # Save
    f_out = ROOT.TFile(output_file, "RECREATE")
    h_jpsi.Write()
    h_dijpsi.Write()
    h_4mu.Write()
    f_out.Close()
    print(f"Wrote histograms to {output_file}")

    write_histogram_csv(h_jpsi, csv_file.replace('.csv', '_jpsi.csv'))
    write_histogram_csv(h_dijpsi, csv_file.replace('.csv', '_dijpsi.csv'))
    write_histogram_csv(h_4mu, csv_file.replace('.csv', '_4mu.csv'))

    return True


def analyze_gen_only(input_file, output_file, csv_file):
    """Analyze GEN-only file for di-J/psi."""
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

    h_jpsi = ROOT.TH1D("h_jpsi_mass", "J/#psi Mass (GEN);m(#mu#mu) [GeV];Events", 100, 2.5, 3.5)
    h_dijpsi = ROOT.TH1D("h_dijpsi_mass", "Di-J/#psi Mass (GEN);m(J/#psi J/#psi) [GeV];Events", 180, 6, 15)
    h_4mu = ROOT.TH1D("h_4mu_mass", "4-Muon Mass (GEN);m(4#mu) [GeV];Events", 180, 6, 15)
    h_etab = ROOT.TH1D("h_etab_mass", "#eta_b Mass (GEN);m(#eta_b) [GeV];Events", 100, 8, 11)

    n_entries = tree.GetEntries()
    print(f"Processing {n_entries} GEN events...")

    for i_evt in range(n_entries):
        tree.GetEntry(i_evt)

        try:
            n_gen = tree.nGenPart
            muons = []
            jpsis = []
            etab_mass = None

            for i in range(n_gen):
                pdg = tree.GenPart_pdgId[i]
                status = tree.GenPart_status[i]

                # Find eta_b (PDG 35 in this generator)
                if abs(pdg) == 35:
                    etab_mass = tree.GenPart_mass[i]
                    h_etab.Fill(etab_mass)

                # Find J/psi
                if abs(pdg) == 443:
                    pt = tree.GenPart_pt[i]
                    eta = tree.GenPart_eta[i]
                    phi = tree.GenPart_phi[i]
                    mass = tree.GenPart_mass[i]
                    p4 = lorentz_vector(pt, eta, phi, mass)
                    jpsis.append(p4)
                    h_jpsi.Fill(mass)

                # Find final-state muons
                if abs(pdg) == 13 and status == 1:
                    pt = tree.GenPart_pt[i]
                    eta = tree.GenPart_eta[i]
                    phi = tree.GenPart_phi[i]
                    charge = -1 if pdg > 0 else 1
                    p4 = lorentz_vector(pt, eta, phi, M_MU)
                    muons.append((p4, charge))

            # Di-J/psi from truth J/psis
            if len(jpsis) >= 2:
                m_dijpsi = invariant_mass(jpsis[0], jpsis[1])
                h_dijpsi.Fill(m_dijpsi)

            # 4-muon mass
            if len(muons) >= 4:
                m4mu = invariant_mass_4(
                    muons[0][0], muons[1][0], muons[2][0], muons[3][0]
                )
                h_4mu.Fill(m4mu)

        except Exception as e:
            pass

    f_in.Close()

    f_out = ROOT.TFile(output_file, "RECREATE")
    h_jpsi.Write()
    h_dijpsi.Write()
    h_4mu.Write()
    h_etab.Write()
    f_out.Close()
    print(f"Wrote histograms to {output_file}")

    write_histogram_csv(h_jpsi, csv_file.replace('.csv', '_jpsi.csv'))
    write_histogram_csv(h_dijpsi, csv_file.replace('.csv', '_dijpsi.csv'))

    return True


def generate_mock_analysis(output_file, csv_file):
    """Generate mock di-J/psi data for testing."""
    print("Generating mock Di-J/psi analysis data...")

    np.random.seed(43)
    n_events = 5000

    # J/psi peak (narrow)
    jpsi_mass = np.random.normal(M_JPSI, 0.03, n_events * 2)
    bins_jpsi = np.linspace(2.5, 3.5, 101)
    counts_jpsi, _ = np.histogram(jpsi_mass, bins=bins_jpsi)
    centers_jpsi = 0.5 * (bins_jpsi[:-1] + bins_jpsi[1:])

    # Di-J/psi spectrum: eta_b peak at 9.4 GeV + background
    dijpsi_signal = np.random.normal(M_ETAB, 0.1, n_events)
    dijpsi_bkg = np.random.uniform(6.5, 12, n_events // 2)
    dijpsi_all = np.concatenate([dijpsi_signal, dijpsi_bkg])

    bins_dijpsi = np.linspace(6, 15, 181)
    counts_dijpsi, _ = np.histogram(dijpsi_all, bins=bins_dijpsi)
    centers_dijpsi = 0.5 * (bins_dijpsi[:-1] + bins_dijpsi[1:])

    # 4-muon spectrum (similar to di-J/psi)
    counts_4mu = counts_dijpsi.copy()
    centers_4mu = centers_dijpsi.copy()

    # Write CSVs
    csv_jpsi = csv_file.replace('.csv', '_jpsi.csv')
    csv_dijpsi = csv_file.replace('.csv', '_dijpsi.csv')
    csv_4mu = csv_file.replace('.csv', '_4mu.csv')

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

    print(f"Wrote mock data to {csv_jpsi}, {csv_dijpsi}, {csv_4mu}")
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
    parser = argparse.ArgumentParser(description="Di-J/psi Analysis")
    parser.add_argument("--input", required=True, help="Input ROOT file")
    parser.add_argument("--output", required=True, help="Output histogram ROOT file")
    parser.add_argument("--csv", default="dijpsi_hist.csv", help="Output CSV file base")
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
