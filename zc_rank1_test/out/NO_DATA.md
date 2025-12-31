# NO DATA AVAILABLE - Zc(3900) Rank-1 Test

Generated: 2025-12-30

## Summary

**STOP CONDITION MET**: Could not obtain numeric tables for BOTH channels (π J/ψ and D D*) required for the rank-1 bottleneck test.

## Data Found

### Channel A: π± J/ψ - FOUND ✓

- **Source**: HEPData record 166173 (inspire:2922807)
- **Paper**: "Partial Wave Analysis of e⁺e⁻ → π⁺π⁻J/ψ and Cross Section Measurement" (BESIII, 2025)
- **Tables**: 17 tables with M(π± J/ψ) invariant mass spectra at energies from 4.127 to 4.358 GeV
- **Format**: Efficiency-corrected, sideband-subtracted event counts in 20 MeV bins
- **Files downloaded**:
  - `data/hepdata/piJpsi_table1.csv` (4.127 GeV)
  - `data/hepdata/piJpsi_table7.csv` (4.189 GeV)
  - `data/hepdata/piJpsi_table15.csv` (4.226 GeV)
  - `data/hepdata/piJpsi_table21.csv` (4.258 GeV)
  - `data/hepdata/piJpsi_table27.csv` (4.287 GeV)
  - `data/hepdata/piJpsi_table33.csv` (4.358 GeV)

### Channel B: D D* - NOT FOUND ✗

- **Required**: M(D D*) or M(D D̄*) invariant mass spectrum at √s ~ 4.26 GeV
- **Status**: No numeric tables found on HEPData

## Search Attempts

### 1. HEPData Searches

| Search Query | Results | Relevant Data |
|--------------|---------|---------------|
| `Zc 3900 BESIII` | 78 | No D D* tables |
| `BESIII D Dstar cross section` | ~10000 | No D D* mass spectra |
| `BESIII Y 4260 D` | 7594 | Only π J/ψ cross sections |
| `e+e- DD* pi recoil mass` | 9985 | No BESIII D D* data |
| `Zc 3900 DD* DDbar*` | 2851 | Zc(3900)⁰ in π⁰π⁰J/ψ only |
| `BESIII D0 Dbar star pi` | 4527 | No D D* invariant mass |
| `D Dstar pi cross section` | 10000 | No Zc region data |

### 2. Records Checked

| Record ID | Inspire | Paper | Data Available |
|-----------|---------|-------|----------------|
| 166173 | 2922807 | BESIII PWA π⁺π⁻J/ψ | M(π J/ψ) ✓ |
| 73771 | 1377204 | BESIII Zc(3900)⁰ π⁰π⁰J/ψ | Only cross sections |
| 18803 | 776519 | BaBar ISR DD̄ | D⁰D̄⁰, D⁺D⁻ (NOT D D*) |
| 61431 | 1225975 | Belle π⁺π⁻J/ψ | σ(√s) only, no mass dist |
| 50926 | 756012 | Belle ISR π⁺π⁻J/ψ | σ(√s) only |

### 3. Original D D* Observation Paper

- **Paper**: PRL 112, 022001 (2014)
- **arXiv**: 1310.1163
- **Inspire**: 1239819
- **HEPData**: NOT UPLOADED
- **Supplementary material**: NONE on arXiv or APS
- **Status**: Original Zc(3885) observation with M(D D̄*) distribution - data not public

### 4. BaBar DD̄ Data (Wrong Channel)

Downloaded `data/hepdata/babar_ddbbar.csv` but this contains:
- e⁺e⁻ → D⁰D̄⁰ and D⁺D⁻ cross sections
- NOT e⁺e⁻ → π D D̄* (which is needed for Zc)
- Mass range covers Zc region but wrong final state

## Why D D* Data is Critical

The Zc(3900) was observed in TWO channels:
1. **π± J/ψ**: Zc(3900)± → π± J/ψ (M = 3899.0 ± 3.6 MeV)
2. **D D***: Zc(3885)± → D D̄* (M = 3883.9 ± 4.5 MeV)

For a rank-1 bottleneck test, we need:
- M(π J/ψ) mass spectrum (available)
- M(D D*) mass spectrum (NOT available on HEPData)

The test would check if the complex amplitude ratio R between two resonance structures is the same across both decay channels.

## Conclusion

Cannot proceed with rank-1 bottleneck test due to missing D D* channel data.

## Next Steps

1. Request data from BESIII collaboration (see EMAIL_REQUEST.txt)
2. Check if future publications add D D* data to HEPData
3. Consider alternative Zc states (Zc(4020), Zc(4430)) if data becomes available
