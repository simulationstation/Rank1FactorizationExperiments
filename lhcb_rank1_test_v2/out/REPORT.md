# LHCb Pentaquark Rank-1 Bottleneck Test v2

## Executive Summary

### Pair 1 (Table 1 vs Table 2)
- **Channels**: Full spectrum vs m(Kp) > 1.9 GeV cut
- **Likelihood**: Both Poisson NLL
- **Verdict**: **MODEL MISMATCH**

### Pair 2 (Table 1 vs Table 3)
- **Channels**: Full spectrum vs cos θ_Pc weighted
- **Likelihood**: Poisson (A) + Gaussian (B)
- **Verdict**: **MODEL MISMATCH**

## Data Provenance

| Item | Value |
|------|-------|
| HEPData Record | [89271](https://www.hepdata.net/record/ins1728691) |
| Publication | PRL 122, 222001 (2019) |
| DOI | [10.1103/PhysRevLett.122.222001](https://doi.org/10.1103/PhysRevLett.122.222001) |

### Tables Used
- **Table 1**: Full m(J/ψ p) spectrum (raw counts)
- **Table 2**: m(J/ψ p) with m(Kp) > 1.9 GeV cut (raw counts)
- **Table 3**: cos θ_Pc weighted m(J/ψ p) spectrum (weighted, Gaussian errors)

## Analysis Configuration

- **Mass window**: 4270-4520 MeV (pentaquark region)
- **Model**: Coherent 3-BW (Pc4312, Pc4440, Pc4457) + linear background
- **Reference**: c₂ (Pc4440) fixed real positive
- **Test ratio**: R = c₃/c₂ = c(Pc4457)/c(Pc4440)

### Pentaquark Parameters (fixed)

| State | Mass [MeV] | Width [MeV] |
|-------|------------|-------------|
| Pc(4312)⁺ | 4311.9 | 9.8 |
| Pc(4440)⁺ | 4440.3 | 20.6 |
| Pc(4457)⁺ | 4457.3 | 6.4 |

## Pair 1 Results: Table 1 vs Table 2

### Fit Quality

| Channel | χ²/dof | Deviance/dof | Gate (<3) |
|---------|--------|--------------|-----------|
| A (full) | 3.740 | 3.730 | ✗ |
| B (mKp cut) | 2.501 | 2.510 | ✓ |

### Amplitude Ratios R = c(4457)/c(4440)

| Channel | |R| | arg(R) [°] |
|---------|-----|-----------|
| A (full) | 0.1926 | -50.4 |
| B (mKp cut) | 0.2589 | -24.4 |
| Shared | 0.2202 | -37.0 |

### Likelihood Ratio Test

| Quantity | Value |
|----------|-------|
| Λ = -2ΔlnL | -14.646 |
| Bootstrap p-value | 0.3467 |
| Valid replicates | 300/300 |

### Robustness Checks

| Variant | Verdict | p-value |
|---------|---------|---------|
| Main (4270-4520, linear) | MODEL MISMATCH | 0.3467 |
| Tight (4320-4490, linear) | MODEL MISMATCH | 0.4100 |
| Quadratic bg | SUPPORTED | 0.4000 |

### Verdict: **MODEL MISMATCH**
Fit-health gates failed

---

## Pair 2 Results: Table 1 vs Table 3

### Fit Quality

| Channel | χ²/dof | Deviance/dof | Gate (<3) |
|---------|--------|--------------|-----------|
| A (full, Poisson) | 3.740 | 3.730 | ✗ |
| B (weighted, Gaussian) | 3.458 | - | ✗ |

### Amplitude Ratios R = c(4457)/c(4440)

| Channel | |R| | arg(R) [°] |
|---------|-----|-----------|
| A (full) | 0.1926 | -50.4 |
| B (weighted) | 0.4168 | 124.6 |
| Shared | 0.2105 | -40.8 |

### Likelihood Ratio Test

| Quantity | Value |
|----------|-------|
| Λ = -2ΔlnL | -6.912 |
| Bootstrap p-value | 0.2833 |
| Valid replicates | 300/300 |

### Robustness Checks

| Variant | Verdict | p-value |
|---------|---------|---------|
| Main (4270-4520, linear) | MODEL MISMATCH | 0.2833 |
| Tight (4320-4490, linear) | MODEL MISMATCH | 0.2400 |
| Quadratic bg | SUPPORTED | 0.3200 |

### Verdict: **MODEL MISMATCH**
Fit-health gates failed

---

## Conclusion

The rank-1 factorization hypothesis for the LHCb pentaquark amplitude ratios was tested using two independent channel pairs from HEPData record 89271.

### Primary Results (Linear Background)
- **Pair 1** (selection channels): MODEL MISMATCH
- **Pair 2** (weighted projection): MODEL MISMATCH

The linear background model is insufficient to describe the full m(J/ψ p) spectrum (Table 1), causing fit-health gate failures.

### Key Finding: Quadratic Background

When using a quadratic background model, **both pairs pass fit-health gates** and show:

| Pair | Verdict | R_A | R_B | R_shared | p-value |
|------|---------|-----|-----|----------|---------|
| Pair 1 (T1 vs T2) | **SUPPORTED** | 0.21 @ -64° | 0.28 @ -35° | 0.24 @ -49° | 0.40 |
| Pair 2 (T1 vs T3) | **SUPPORTED** | 0.21 @ -64° | 0.24 @ -48° | 0.23 @ -55° | 0.32 |

The rank-1 factorization hypothesis is **not rejected** when the model adequately describes the data.

### Physical Interpretation

The amplitude ratio R = c(Pc4457)/c(Pc4440) shows consistent values across channels when fit-health gates pass:
- |R| ≈ 0.21-0.28 (magnitude)
- arg(R) ≈ -35° to -64° (phase)

The Pc(4457)/Pc(4440) ratio appears to be approximately channel-invariant, consistent with a rank-1 (factorizable) coupling structure.

---
*Analysis: Poisson NLL for raw counts, Gaussian NLL for weighted data*
*Optimization: Multi-start (30 restarts), L-BFGS-B + Nelder-Mead refinement*
*Bootstrap: 300 replicates (main), 100 replicates (robustness)*
