# Zc(3900) Rank-1 Bottleneck Test - Final Report

**Generated**: 2025-12-30
**Status**: STOPPED - Data unavailable
**Verdict**: **NO DATA**

---

## 1. Executive Summary

This test aimed to determine whether the Zc(3900) charged exotic hadron exhibits rank-1 amplitude factorization across its two observed decay channels:
- Channel A: π± J/ψ
- Channel B: D D̄*

**Result**: Test could not be performed. While π J/ψ invariant mass spectra are available on HEPData, the D D̄* invariant mass spectrum data is not publicly available in numeric form.

---

## 2. Data Sources

### Channel A: π± J/ψ - FOUND ✓

| Property | Value |
|----------|-------|
| HEPData Record | [166173](https://www.hepdata.net/record/ins2922807) |
| Inspire ID | 2922807 |
| Paper | "Partial Wave Analysis of e⁺e⁻ → π⁺π⁻J/ψ" (BESIII, 2025) |
| Observable | M(π± J/ψ) invariant mass spectrum |
| Energy Points | 17 (from 4.127 to 4.358 GeV) |
| Bin Width | 20 MeV |
| Data Type | Efficiency-corrected, sideband-subtracted event counts |

**Tables Downloaded**:
| File | √s (GeV) | Table # |
|------|----------|---------|
| piJpsi_table1.csv | 4.127 | 1 |
| piJpsi_table7.csv | 4.189 | 7 |
| piJpsi_table15.csv | 4.226 | 15 |
| piJpsi_table21.csv | 4.258 | 21 |
| piJpsi_table27.csv | 4.287 | 27 |
| piJpsi_table33.csv | 4.358 | 33 |

### Channel B: D D̄* - NOT FOUND ✗

| Property | Value |
|----------|-------|
| Required Data | M(D D̄*) or M(DD*) invariant mass spectrum |
| Original Paper | PRL 112, 022001 (2014) |
| arXiv | 1310.1163 |
| Inspire ID | 1239819 |
| HEPData Status | **NOT UPLOADED** |
| Supplementary | **NONE** on arXiv or APS |

---

## 3. Search Attempts

### HEPData Query Results

| Query | Results | Outcome |
|-------|---------|---------|
| `Zc 3900 BESIII` | 78 | No D D* mass tables |
| `BESIII D Dstar cross section` | ~10000 | No mass spectra |
| `BESIII Y 4260 D` | 7594 | Only π J/ψ data |
| `e+e- DD* pi recoil mass` | 9985 | No BESIII D D* |
| `Zc 3900 DD* DDbar*` | 2851 | Zc(3900)⁰ only |
| `BESIII D0 Dbar star pi` | 4527 | No mass spectra |

### Records Examined

| Record | Inspire | Experiment | Data Available |
|--------|---------|------------|----------------|
| 166173 | 2922807 | BESIII | M(π J/ψ) ✓ |
| 73771 | 1377204 | BESIII | Cross sections only |
| 18803 | 776519 | BaBar | D⁰D̄⁰, D⁺D⁻ (wrong channel) |
| 61431 | 1225975 | Belle | σ(√s) only |
| 50926 | 756012 | Belle | σ(√s) only |

### Alternative Sources Checked

1. **arXiv 1310.1163**: No supplementary data files
2. **APS PRL 112, 022001**: No supplementary materials
3. **BaBar DD̄ data**: Wrong channel (D⁰D̄⁰, not D D̄*)
4. **Belle data**: Only cross section vs √s, not mass spectra

---

## 4. Physics Context

### The Zc(3900) Exotic State

The Zc(3900)± is a charged charmonium-like exotic hadron that cannot be a conventional cc̄ meson (which must be neutral). It was discovered in 2013 and observed in two decay modes:

| Channel | Mass (MeV) | Width (MeV) | Paper |
|---------|------------|-------------|-------|
| π± J/ψ | 3899.0 ± 3.6 | 46 ± 22 | PRL 110, 252001 (2013) |
| D D̄* | 3883.9 ± 4.5 | 24.8 ± 12 | PRL 112, 022001 (2014) |

### Why Both Channels Are Needed

The rank-1 bottleneck test checks whether the complex amplitude ratio R (between interfering resonance structures) is shared across decay channels:

**Model**: I_α(m) = |BW₁(m) + R_α BW₂(m)|² + B_α(m)

- **M₀ (Null)**: R_A = R_B (universal production mechanism)
- **M₁ (Alternative)**: R_A ≠ R_B (channel-dependent)

Testing this requires invariant mass spectra from BOTH channels. Having only π J/ψ data is insufficient.

---

## 5. Fit Results

| Metric | Value |
|--------|-------|
| M₀ χ²/dof | N/A |
| M₁ χ²/dof | N/A |
| Λ = -2 ln(L₀/L₁) | N/A |
| Bootstrap p-value | N/A |
| Number of replicates | N/A |

**Reason**: Fit not performed - missing D D̄* channel data.

---

## 6. Gates and Quality Checks

| Gate | Status | Value |
|------|--------|-------|
| χ²/dof in [0.5, 3.0] | N/A | - |
| Bootstrap replicates ≥ 500 | N/A | - |
| R magnitude physical | N/A | - |
| Convergence achieved | N/A | - |

---

## 7. Final Verdict

### **NO DATA**

The Zc(3900) rank-1 bottleneck test cannot be performed with currently available public data. The D D̄* invariant mass spectrum from BESIII (or any other experiment) is not available on HEPData or as official supplementary material.

### Classification

| Verdict | Definition | Applies? |
|---------|------------|----------|
| SUPPORTED | Λ < threshold, p > 0.05 | No |
| DISFAVORED | Λ > threshold, p < 0.05 | No |
| INCONCLUSIVE | Poor fit quality or low statistics | No |
| MODEL MISMATCH | Data incompatible with model | No |
| **NO DATA** | Required data unavailable | **Yes** |
| OPTIMIZER FAILURE | Fit did not converge | No |

---

## 8. Recommendations

1. **Contact BESIII Collaboration**: A draft email request has been prepared (see EMAIL_REQUEST.txt)

2. **Monitor HEPData**: Check periodically for new uploads of D D̄* data

3. **Alternative Targets**: Consider other Zc states if their multi-channel data becomes available:
   - Zc(4020): π hc and D*D̄*
   - Zc(4430): π ψ(2S) and D*D̄

4. **Alternative Exotics**: The rank-1 test may be applicable to other exotic candidates with better data availability

---

## 9. Files Generated

```
zc_rank1_test/
├── data/
│   └── hepdata/
│       ├── piJpsi_table1.csv    (4.127 GeV)
│       ├── piJpsi_table7.csv    (4.189 GeV)
│       ├── piJpsi_table15.csv   (4.226 GeV)
│       ├── piJpsi_table21.csv   (4.258 GeV)
│       ├── piJpsi_table27.csv   (4.287 GeV)
│       ├── piJpsi_table33.csv   (4.358 GeV)
│       ├── babar_ddbbar.csv     (wrong channel, reference)
│       └── belle_piJpsi.csv     (σ vs √s only)
├── out/
│   ├── NO_DATA.md
│   ├── EMAIL_REQUEST.txt
│   └── REPORT.md               (this file)
└── logs/
    └── COMMANDS.txt
```

---

## 10. References

1. BESIII Collaboration, "Observation of a charged charmoniumlike structure in e⁺e⁻ → π⁺π⁻J/ψ at √s = 4.26 GeV", PRL 110, 252001 (2013)

2. BESIII Collaboration, "Observation of a Charged Charmoniumlike Structure Zc(3900) in e⁺e⁻ → π⁺π⁻J/ψ at √s = 4.26 GeV", PRL 112, 022001 (2014)

3. BESIII Collaboration, "Partial Wave Analysis of e⁺e⁻ → π⁺π⁻J/ψ and Cross Section Measurement", arXiv:2501.XXXXX (2025), HEPData record 166173

---

*Report generated by automated rank-1 bottleneck test pipeline*
