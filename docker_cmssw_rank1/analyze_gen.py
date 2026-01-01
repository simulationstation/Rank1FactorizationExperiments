#!/usr/bin/env python3
"""
Analyze CMSSW GEN-only ROOT files using uproot.
Produces CSV histograms for rank-1 test.
"""

import numpy as np
import uproot
import sys
from itertools import combinations

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

def analyze_fourmu(input_file, output_prefix):
    """Analyze FourMu GEN file and produce histograms."""
    print(f"Analyzing {input_file}...")

    f = uproot.open(input_file)
    events = f["Events"]

    # Get the branches
    pt = events["recoGenParticles_genParticles__GEN./recoGenParticles_genParticles__GEN.obj/recoGenParticles_genParticles__GEN.obj.m_state.p4Polar_.fCoordinates.fPt"].array()
    eta = events["recoGenParticles_genParticles__GEN./recoGenParticles_genParticles__GEN.obj/recoGenParticles_genParticles__GEN.obj.m_state.p4Polar_.fCoordinates.fEta"].array()
    phi = events["recoGenParticles_genParticles__GEN./recoGenParticles_genParticles__GEN.obj/recoGenParticles_genParticles__GEN.obj.m_state.p4Polar_.fCoordinates.fPhi"].array()
    pdgId = events["recoGenParticles_genParticles__GEN./recoGenParticles_genParticles__GEN.obj/recoGenParticles_genParticles__GEN.obj.m_state.pdgId_"].array()
    status = events["recoGenParticles_genParticles__GEN./recoGenParticles_genParticles__GEN.obj/recoGenParticles_genParticles__GEN.obj.m_state.status_"].array()

    n_events = len(pt)
    print(f"Found {n_events} events")

    dimu_masses = []
    fourmu_masses = []

    for i_evt in range(n_events):
        evt_pt = np.array(pt[i_evt])
        evt_eta = np.array(eta[i_evt])
        evt_phi = np.array(phi[i_evt])
        evt_pdg = np.array(pdgId[i_evt])
        evt_status = np.array(status[i_evt])

        # Select final-state muons (|pdgId| == 13, status == 1)
        mask = (np.abs(evt_pdg) == 13) & (evt_status == 1)
        mu_pt = evt_pt[mask]
        mu_eta = evt_eta[mask]
        mu_phi = evt_phi[mask]
        mu_pdg = evt_pdg[mask]

        n_mu = len(mu_pt)
        if n_mu < 2:
            continue

        # Build muon 4-vectors
        muons = []
        for j in range(n_mu):
            p4 = lorentz_vector(mu_pt[j], mu_eta[j], mu_phi[j], M_MU)
            charge = -1 if mu_pdg[j] > 0 else 1  # mu- has pdgId=13
            muons.append((p4, charge))

        # Dimuon combinations (opposite sign)
        for (p1, q1), (p2, q2) in combinations(muons, 2):
            if q1 * q2 < 0:  # opposite sign
                m = invariant_mass(p1, p2)
                dimu_masses.append(m)

        # 4-muon invariant mass
        if n_mu >= 4:
            m4 = invariant_mass_4(muons[0][0], muons[1][0], muons[2][0], muons[3][0])
            fourmu_masses.append(m4)

    # Create histograms
    bins_dimu = np.linspace(0, 15, 151)
    bins_4mu = np.linspace(0, 50, 101)

    counts_dimu, _ = np.histogram(dimu_masses, bins=bins_dimu)
    counts_4mu, _ = np.histogram(fourmu_masses, bins=bins_4mu)

    centers_dimu = 0.5 * (bins_dimu[:-1] + bins_dimu[1:])
    centers_4mu = 0.5 * (bins_4mu[:-1] + bins_4mu[1:])

    # Write CSVs
    csv_dimu = f"{output_prefix}_dimu.csv"
    csv_4mu = f"{output_prefix}_4mu.csv"

    with open(csv_dimu, 'w') as f:
        f.write("mass_GeV,counts,stat_err\n")
        for m, c in zip(centers_dimu, counts_dimu):
            f.write(f"{m:.4f},{c},{np.sqrt(max(1,c)):.4f}\n")

    with open(csv_4mu, 'w') as f:
        f.write("mass_GeV,counts,stat_err\n")
        for m, c in zip(centers_4mu, counts_4mu):
            f.write(f"{m:.4f},{c},{np.sqrt(max(1,c)):.4f}\n")

    print(f"FourMu analysis complete:")
    print(f"  Dimuon pairs: {len(dimu_masses)}")
    print(f"  4-muon events: {len(fourmu_masses)}")
    print(f"  Wrote {csv_dimu} and {csv_4mu}")

def analyze_dijpsi(input_file, output_prefix):
    """Analyze DiJpsi GEN file and produce histograms."""
    print(f"Analyzing {input_file}...")

    f = uproot.open(input_file)
    events = f["Events"]

    # Get the branches
    pt = events["recoGenParticles_genParticles__GEN./recoGenParticles_genParticles__GEN.obj/recoGenParticles_genParticles__GEN.obj.m_state.p4Polar_.fCoordinates.fPt"].array()
    eta = events["recoGenParticles_genParticles__GEN./recoGenParticles_genParticles__GEN.obj/recoGenParticles_genParticles__GEN.obj.m_state.p4Polar_.fCoordinates.fEta"].array()
    phi = events["recoGenParticles_genParticles__GEN./recoGenParticles_genParticles__GEN.obj/recoGenParticles_genParticles__GEN.obj.m_state.p4Polar_.fCoordinates.fPhi"].array()
    pdgId = events["recoGenParticles_genParticles__GEN./recoGenParticles_genParticles__GEN.obj/recoGenParticles_genParticles__GEN.obj.m_state.pdgId_"].array()
    status = events["recoGenParticles_genParticles__GEN./recoGenParticles_genParticles__GEN.obj/recoGenParticles_genParticles__GEN.obj.m_state.status_"].array()
    mass_arr = events["recoGenParticles_genParticles__GEN./recoGenParticles_genParticles__GEN.obj/recoGenParticles_genParticles__GEN.obj.m_state.p4Polar_.fCoordinates.fM"].array()

    n_events = len(pt)
    print(f"Found {n_events} events")

    jpsi_masses = []
    dijpsi_masses = []
    fourmu_masses = []

    for i_evt in range(n_events):
        evt_pt = np.array(pt[i_evt])
        evt_eta = np.array(eta[i_evt])
        evt_phi = np.array(phi[i_evt])
        evt_pdg = np.array(pdgId[i_evt])
        evt_status = np.array(status[i_evt])
        evt_mass = np.array(mass_arr[i_evt])

        # Find J/psi (pdgId = 443)
        jpsi_mask = (evt_pdg == 443)
        jpsi_pt = evt_pt[jpsi_mask]
        jpsi_eta = evt_eta[jpsi_mask]
        jpsi_phi = evt_phi[jpsi_mask]
        jpsi_m = evt_mass[jpsi_mask]

        n_jpsi = len(jpsi_pt)

        # Record J/psi masses
        for m in jpsi_m:
            jpsi_masses.append(m)

        # Di-J/psi
        if n_jpsi >= 2:
            jpsi_p4s = []
            for j in range(n_jpsi):
                p4 = lorentz_vector(jpsi_pt[j], jpsi_eta[j], jpsi_phi[j], jpsi_m[j])
                jpsi_p4s.append(p4)

            for p1, p2 in combinations(jpsi_p4s, 2):
                m = invariant_mass(p1, p2)
                dijpsi_masses.append(m)

        # Also look at final-state muons for 4mu mass
        mu_mask = (np.abs(evt_pdg) == 13) & (evt_status == 1)
        mu_pt = evt_pt[mu_mask]
        mu_eta = evt_eta[mu_mask]
        mu_phi = evt_phi[mu_mask]

        if len(mu_pt) >= 4:
            muons = [lorentz_vector(mu_pt[j], mu_eta[j], mu_phi[j], M_MU) for j in range(4)]
            m4 = invariant_mass_4(muons[0], muons[1], muons[2], muons[3])
            fourmu_masses.append(m4)

    # Create histograms
    bins_jpsi = np.linspace(2.9, 3.3, 41)  # narrow around J/psi
    bins_dijpsi = np.linspace(6, 15, 91)
    bins_4mu = np.linspace(6, 15, 91)

    counts_jpsi, _ = np.histogram(jpsi_masses, bins=bins_jpsi)
    counts_dijpsi, _ = np.histogram(dijpsi_masses, bins=bins_dijpsi)
    counts_4mu, _ = np.histogram(fourmu_masses, bins=bins_4mu)

    centers_jpsi = 0.5 * (bins_jpsi[:-1] + bins_jpsi[1:])
    centers_dijpsi = 0.5 * (bins_dijpsi[:-1] + bins_dijpsi[1:])
    centers_4mu = 0.5 * (bins_4mu[:-1] + bins_4mu[1:])

    # Write CSVs
    csv_jpsi = f"{output_prefix}_jpsi.csv"
    csv_dijpsi = f"{output_prefix}_dijpsi.csv"
    csv_4mu = f"{output_prefix}_4mu.csv"

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

    print(f"DiJpsi analysis complete:")
    print(f"  J/psi particles: {len(jpsi_masses)}")
    print(f"  Di-J/psi pairs: {len(dijpsi_masses)}")
    print(f"  4-muon events: {len(fourmu_masses)}")
    print(f"  Wrote {csv_jpsi}, {csv_dijpsi}, {csv_4mu}")

if __name__ == "__main__":
    # Analyze FourMu
    analyze_fourmu("outputs/fourmu_gen.root", "outputs/fourmu_hist")

    # Analyze DiJpsi
    analyze_dijpsi("outputs/dijpsi_gen.root", "outputs/dijpsi_hist")

    print("\nAnalysis complete!")
