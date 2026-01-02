# NO_DATA Report: LHCb Strange Pentaquark (Pcs) Rank-1 Test

**Family:** Strange pentaquarks (Pcs)
**Requested States:** Pcs(4338), Pcs(4459)
**Channel:** J/psi Lambda
**Date:** 2026-01-02
**Branch:** lhcb_pcs_rank1_v1

---

## Executive Summary

**Status: DATA ACQUISITION FAILED**

A rank-1 factorization test for the Pcs(4338)/Pcs(4459) system is **not feasible** with currently available public data. The two states were discovered in different production mechanisms and do not appear together in any single published spectrum.

---

## Data Acquisition Attempts

### Priority 1: HEPData

| Paper | arXiv ID | HEPData Record | Status |
|-------|----------|----------------|--------|
| Pcs(4459) discovery | 2012.10380 | None | FAILED |
| Pcs(4338) discovery | 2210.10346 | None | FAILED |

Extensive search of HEPData confirmed no records exist for either Pcs discovery paper. The only LHCb pentaquark HEPData entry (ins1728691) is for the Pc states in J/psi p, not Pcs in J/psi Lambda.

### Priority 2: Supplementary Materials

Checked LHCb public pages for both analyses:
- Only PDF/PNG figures and ROOT plotting macros available
- **No tabular data** (CSV, YAML, or similar) provided
- Status: FAILED

### Priority 3: PDF Vector Extraction

Both PDFs downloaded and examined:
- `pcs4459_2012.10380.pdf` - Contains m(J/psi Lambda) spectrum in Figure 3
- `pcs4338_2210.10346.pdf` - Contains m(J/psi Lambda) spectrum in Figure 3

PDF extraction is technically possible but **would not resolve the fundamental physics issue** (see below).

---

## Fundamental Physics Issue

### The Two Pcs States Are From Different Production Mechanisms

| State | Production Mode | Parent Decay | Signal Yield | Significance |
|-------|-----------------|--------------|--------------|--------------|
| Pcs(4459) | Ξb⁻ → J/ψ Λ K⁻ | Ξb⁻ decay | ~1750 | 3.1σ |
| Pcs(4338) | B⁻ → J/ψ Λ p̄ | B⁻ decay | ~4400 | >15σ |

### Why This Prevents a Rank-1 Test

A rank-1 factorization test requires:
1. **The same resonance(s)** appearing in **multiple spectra**
2. Different spectra arise from different kinematic selections, decay channels, or experimental conditions
3. The test checks if the coupling ratio R = g₂/g₁ is consistent across spectra

**Problem:** Pcs(4459) and Pcs(4338) are:
- At different masses (4459 MeV vs 4338 MeV) - no overlap
- From different production mechanisms (Ξb vs B decay)
- In different phase space regions
- Neither paper shows both states in the same spectrum

The Pcs(4338) mass region is outside the phase space accessible in Ξb⁻ → J/ψ Λ K⁻, and Pcs(4459) was not observed in B⁻ → J/ψ Λ p̄.

### Comparison to Successful Pc Test

The LHCb Pc(4440)/Pc(4457) rank-1 test (lhcb_pc_rank1_v5) worked because:
- Both Pc states appear in the **same** J/psi p spectrum from Λb → J/ψ p K⁻
- HEPData provides **three different projections** of the same data (full, mKp cut, cosθ-weighted)
- The rank-1 test checks if R is consistent across these projections

For Pcs, we would need either:
1. Both Pcs(4338) and Pcs(4459) visible in the same spectrum, OR
2. The same Pcs state measured in two different decay modes/selections

Neither condition is met in current public data.

---

## Published Parameters (For Reference)

### Pcs(4459)⁰ [arXiv:2012.10380]
- Mass: 4458.8 ± 2.9 (stat) +4.7/-1.1 (syst) MeV
- Width: 17.3 ± 6.5 (stat) +8.0/-5.7 (syst) MeV
- Significance: 3.1σ (including systematics and look-elsewhere effect)
- Quantum numbers: Not determined

### Pcs(4338)⁰ [arXiv:2210.10346]
- Mass: 4338.2 ± 0.7 (stat) ± 0.4 (syst) MeV
- Width: 7.0 ± 1.2 (stat) ± 1.3 (syst) MeV
- Significance: >15σ
- Quantum numbers: J^P = 1/2⁻ preferred

---

## Recommendations

### For Future Work

1. **Monitor HEPData:** If LHCb releases tabular data for either paper, revisit this analysis

2. **Alternative Test Design:** If a combined amplitude analysis is published showing both Pcs states in the same spectrum, a rank-1 test becomes possible

3. **Single-State Test:** Could perform a simpler consistency test on Pcs(4338) alone if efficiency-corrected spectra with different selections become available from arXiv:2210.10346

4. **Cross-Channel Search:** Watch for future LHCb analyses that might observe Pcs(4338) in Ξb decay or Pcs(4459) in B decay

---

## Conclusion

The rank-1 factorization test for the Pcs family cannot proceed due to:
1. No HEPData records for either discovery paper
2. No tabular supplementary data available
3. **Fundamental physics limitation:** The two Pcs states are from different production mechanisms and do not appear together in any published spectrum

This is a **data availability and physics constraint issue**, not an analysis methodology failure.

---

## References

1. LHCb Collaboration, "Evidence for a new structure in the J/ψΛ spectrum", arXiv:2012.10380 (2020)
2. LHCb Collaboration, "Observation of a J/ψΛ resonance consistent with a strange pentaquark", arXiv:2210.10346 (2022)
3. HEPData search: https://www.hepdata.net/search/?q=pentaquark%20LHCb
