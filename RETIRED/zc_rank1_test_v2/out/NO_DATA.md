# NO DATA AVAILABLE - Zc Rank-1 Test v2

Generated: 2025-12-30

## Summary

**STOP CONDITION MET**: Could not obtain binned mass spectra for BOTH channels required for rank-1 bottleneck test.

Two paths were attempted:
- **Path A**: Zc(4020)/Zc(4025) → π h_c vs D*D*
- **Path B**: Zc(3900)/Zc(3885) → π J/ψ vs D D*

Both paths failed due to lack of binned invariant mass spectra in the D-meson decay channels.

---

## Path A: Zc(4020)/Zc(4025)

### Channel A: π± h_c - PARTIAL DATA ⚠️

| Source | arXiv:1309.1896 |
|--------|-----------------|
| Paper | PRL 111, 242001 (2013) |
| Observable | σ(e⁺e⁻ → π⁺π⁻ h_c) at 13 energy points |
| Zc(4020) yields | At 3 energy points: 4.23, 4.26, 4.36 GeV |
| **Missing** | Binned M(π± h_c) mass spectrum |

Extracted data:
- `data/arxiv/zc4020_pihc_xsec.csv`: Cross sections at 13 energies
- `data/arxiv/zc4020_yields.csv`: Zc(4020) yields at 3 energies

### Channel B: D*D̄* - INSUFFICIENT ✗

| Source | arXiv:1308.2760 |
|--------|-----------------|
| Paper | PRL 112, 132001 (2014) |
| Observable | Zc(4025) → D*D̄* at √s=4.26 GeV only |
| **Missing** | Binned M(D*D̄*) mass spectrum |
| **Missing** | Multiple energy points |

Extracted data:
- `data/arxiv/zc4025_dstardstar.csv`: Single-point resonance parameters

### Why Path A Fails

1. **No binned mass spectra**: Both papers report only fitted resonance parameters and yields, not the underlying M(π h_c) or M(D*D*) distributions
2. **Only 1 energy point for D*D***: Cannot test energy-dependent coupling with single point
3. **Cannot fit 2-BW coherent model**: Model requires binned mass spectrum data

---

## Path B: Zc(3900)/Zc(3885)

### Channel A: π± J/ψ - FOUND ✓ (from v1)

| Source | HEPData 166173 |
|--------|----------------|
| Paper | BESIII PWA 2025 |
| Observable | Binned M(π± J/ψ) at 17 energy points |
| Status | Downloaded in zc_rank1_test/ |

### Channel B: D D̄* - INSUFFICIENT ✗

| Source | arXiv:1310.1163 |
|--------|-----------------|
| Paper | PRL 112, 022001 (2014) |
| Observable | Zc(3885) → D D̄* at √s=4.26 GeV only |
| **Missing** | Binned M(D D̄*) mass spectrum |

Extracted data:
- `data/arxiv/zc3885_ddstar.csv`: Single-point resonance parameters

### Why Path B Fails

1. **No binned M(D D̄*) mass spectrum**: Only fitted pole mass/width reported
2. **Mismatch in data types**: Cannot compare mass spectrum (π J/ψ) to single yield (D D̄*)
3. **Different analysis method**: π J/ψ from PWA; D D̄* from partial reconstruction

---

## arXiv Source Extraction Summary

| arXiv ID | Paper | Data Type | Binned Spectrum? |
|----------|-------|-----------|------------------|
| 1309.1896 | Zc(4020) π h_c | Cross sections, yields | **NO** |
| 1308.2760 | Zc(4025) D*D* | Resonance params | **NO** |
| 1310.1163 | Zc(3885) D D* | Resonance params | **NO** |

All three papers contain LaTeX tables with:
- Fitted resonance parameters (mass, width)
- Total signal yields
- Cross sections

None contain the actual binned invariant mass distribution data points.

---

## What Would Be Needed

For a rank-1 bottleneck test, we need:

### Option 1: Binned Mass Spectra
- M(π h_c) spectrum at √s ≈ 4.26 GeV
- M(D*D*) spectrum at √s ≈ 4.26 GeV
- Same binning, efficiency-corrected, with uncertainties

### Option 2: Multi-Energy Yields (weaker test)
- Zc → Channel A yields at ≥3 energy points
- Zc → Channel B yields at same ≥3 energy points
- Current status: Only 1 D*D* point, 1 D D* point

---

## Files Created

```
zc_rank1_test_v2/data/arxiv/
├── 1308.2760/               (Zc(4025) paper source)
├── 1309.1896/               (Zc(4020) paper source)
├── 1310.1163/               (Zc(3885) paper source)
├── zc4020_pihc_xsec.csv     (13 energy points)
├── zc4020_yields.csv        (3 energy points)
├── zc4025_dstardstar.csv    (1 energy point)
└── zc3885_ddstar.csv        (1 energy point)
```

---

## Conclusion

Cannot proceed with rank-1 bottleneck test. The arXiv source files do not contain binned mass spectrum data - only fitted parameters and yields.

See EMAIL_REQUEST.txt for data request template.
