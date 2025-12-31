# Zc Rank-1 Bottleneck Test v2 - Final Report

**Generated**: 2025-12-30
**Status**: STOPPED - Insufficient data
**Verdict**: **NO DATA**

---

## 1. Executive Summary

This test attempted two paths to obtain numeric data for a Zc-family rank-1 bottleneck test:

| Path | Target States | Channel A | Channel B | Result |
|------|---------------|-----------|-----------|--------|
| A | Zc(4020)/Zc(4025) | π± h_c | D*D̄* | **FAILED** |
| B | Zc(3900)/Zc(3885) | π± J/ψ | D D̄* | **FAILED** |

**Critical finding**: The arXiv source files (LaTeX) do not contain binned mass spectrum data - only fitted resonance parameters and integrated yields. Without binned spectra, the rank-1 amplitude model cannot be tested.

---

## 2. Data Sources Attempted

### 2.1 HEPData Searches

| Query | Records | Relevant Data |
|-------|---------|---------------|
| Zc 4020 BESIII | Limited | No D*D* spectra |
| Zc 4025 | 0 relevant | No HEPData upload |
| BESIII D*Dbar* | Many | No Zc-region mass spectra |
| BESIII pi hc | Many | No binned M(π h_c) |

**Result**: Neither Zc(4020) π h_c nor Zc(4025) D*D* data is on HEPData.

### 2.2 arXiv Source Extraction

| arXiv ID | Paper | LaTeX Tables Found |
|----------|-------|--------------------|
| 1309.1896 | Zc(4020) → π h_c | Cross sections (13 pts), yields (3 pts) |
| 1308.2760 | Zc(4025) → D*D* | Systematics table, resonance params |
| 1310.1163 | Zc(3885) → D D* | Pole mass/width table |

**Result**: All papers contain only:
- Fitted Breit-Wigner parameters (mass, width)
- Total signal yields
- Cross sections

None contain binned M(X) mass spectrum data points.

---

## 3. Extracted Numeric Data

### 3.1 Zc(4020) → π h_c (arXiv:1309.1896)

**Cross sections for e⁺e⁻ → π⁺π⁻ h_c at 13 energies:**

| √s (GeV) | σ (pb) | Stat | Syst |
|----------|--------|------|------|
| 4.190 | 17.7 | ±9.8 | ±1.6 |
| 4.210 | 34.8 | ±9.5 | ±3.2 |
| 4.220 | 41.9 | ±10.7 | ±3.8 |
| 4.230 | 50.2 | ±2.7 | ±4.6 |
| 4.245 | 32.7 | ±10.3 | ±3.0 |
| 4.260 | 41.0 | ±2.8 | ±3.7 |
| 4.310 | 61.9 | ±12.9 | ±5.6 |
| 4.360 | 52.3 | ±3.7 | ±4.8 |
| 4.390 | 41.8 | ±10.8 | ±3.8 |
| 4.420 | 49.4 | ±12.4 | ±4.5 |

**Zc(4020) yields at 3 energies:**

| √s (GeV) | N(Zc) | σ(π Zc → ππhc) pb |
|----------|-------|-------------------|
| 4.230 | 114±25 | 8.7±1.9 |
| 4.260 | 72±17 | 7.4±1.7 |
| 4.360 | 67±15 | 10.3±2.3 |

### 3.2 Zc(4025) → D*D̄* (arXiv:1308.2760)

**Single energy point only (√s = 4.26 GeV):**

| Parameter | Value |
|-----------|-------|
| Mass | (4026.3 ± 2.6 ± 3.7) MeV/c² |
| Width | (24.8 ± 5.6 ± 7.7) MeV |
| N(signal) | 400.9 ± 47.3 events |
| σ(D*D̄*π) | (137 ± 9 ± 15) pb |
| R = σ(Zc)/σ(total) | 0.65 ± 0.09 ± 0.06 |

### 3.3 Zc(3885) → D D̄* (arXiv:1310.1163)

**Single energy point only (√s = 4.26 GeV):**

| Tag | M_pole (MeV/c²) | Γ_pole (MeV) | N_signal |
|-----|-----------------|--------------|----------|
| π⁺D⁰ | 3882.3±1.5 | 24.6±3.3 | 502±41 |
| π⁻D⁺ | 3885.5±1.5 | 24.9±3.2 | 710±54 |

Average: M = 3883.9±1.5±4.2 MeV/c², Γ = 24.8±3.3±11.0 MeV
σ×B = 83.5±6.6±22.0 pb

---

## 4. Why Rank-1 Test Cannot Proceed

### 4.1 For Path A (Zc(4020)/Zc(4025))

The rank-1 test requires fitting:

```
I_α(m) = |BW₁(m) + R_α BW₂(m)|² + B_α(m)
```

to binned mass spectra in BOTH channels. We have:

- **π h_c**: Only total yields at 3 energies, NO M(π h_c) spectrum
- **D*D***: Only yield at 1 energy, NO M(D*D*) spectrum

### 4.2 For Path B (Zc(3900)/Zc(3885))

- **π J/ψ**: Have binned M(π J/ψ) from HEPData (v1 test)
- **D D***: Only yield at 1 energy, NO M(D D*) spectrum

The mismatch in data types prevents comparison.

### 4.3 Alternative "Shared-Coupling" Test

A weaker test comparing energy-dependent yields would require:
- Channel A yields at ≥3 energy points ✓ (π h_c has 3)
- Channel B yields at ≥3 energy points ✗ (D*D* has only 1)

With only 1 D*D* point, no energy-dependent fit is possible.

---

## 5. Fit Results

| Metric | Value |
|--------|-------|
| M₀ χ²/dof | N/A |
| M₁ χ²/dof | N/A |
| Λ = -2 ln(L₀/L₁) | N/A |
| Bootstrap p-value | N/A |

**Reason**: Fit not performed - insufficient data.

---

## 6. Quality Gates

| Gate | Status |
|------|--------|
| Both channels have binned spectra | **FAIL** |
| ≥3 energy points per channel | **FAIL** |
| χ²/dof in [0.5, 3.0] | N/A |
| Bootstrap ≥500 replicates | N/A |

---

## 7. Final Verdict

### **NO DATA**

Neither Path A (Zc(4020)/Zc(4025)) nor Path B (Zc(3900)/Zc(3885)) can be tested due to lack of binned invariant mass spectra in the D-meson decay channels.

| Path | Hidden-charm | Open-charm | Verdict |
|------|--------------|------------|---------|
| A | π h_c (yields only) | D*D* (1 point only) | NO DATA |
| B | π J/ψ (spectra ✓) | D D* (1 point only) | NO DATA |

---

## 8. Files Generated

```
zc_rank1_test_v2/
├── data/
│   ├── arxiv/
│   │   ├── 1308.2760/          (Zc(4025) source)
│   │   ├── 1309.1896/          (Zc(4020) source)
│   │   ├── 1310.1163/          (Zc(3885) source)
│   │   ├── zc4020_pihc_xsec.csv
│   │   ├── zc4020_yields.csv
│   │   ├── zc4025_dstardstar.csv
│   │   └── zc3885_ddstar.csv
│   └── hepdata/                 (empty - no relevant data found)
├── out/
│   ├── NO_DATA.md
│   ├── EMAIL_REQUEST.txt
│   └── REPORT.md               (this file)
└── logs/
    └── COMMANDS.txt
```

---

## 9. Recommendations

1. **Request binned mass spectra from BESIII** (see EMAIL_REQUEST.txt)
   - M(D*D̄*) at √s = 4.26 GeV (and 4.23, 4.36 if available)
   - M(D D̄*) at √s = 4.26 GeV

2. **Monitor HEPData** for future uploads of BESIII Zc analyses

3. **Consider alternative exotics** with better data:
   - LHCb pentaquarks (Pc states) - tested in lhcb_rank1_test_v4/
   - Y-states in e⁺e⁻ line shapes - tested in y_rank1_test_v2/

4. **Await BESIII updates**: New publications may include supplementary data

---

## 10. References

1. BESIII Collaboration, "Observation of Zc(4020) in π⁺π⁻h_c", PRL 111, 242001 (2013), arXiv:1309.1896
2. BESIII Collaboration, "Observation of Zc(4025) in D*D̄*π", PRL 112, 132001 (2014), arXiv:1308.2760
3. BESIII Collaboration, "Observation of Zc(3885) in D D̄*π", PRL 112, 022001 (2014), arXiv:1310.1163
4. BESIII Collaboration, "PWA of π⁺π⁻J/ψ", HEPData 166173 (2025)

---

*Report generated by Zc rank-1 bottleneck test pipeline v2*
