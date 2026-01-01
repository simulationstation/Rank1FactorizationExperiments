# MEGA MECHANISM SWEEP - Summary Report

Generated: 2025-12-30 17:21:35

## Executive Summary

This report compares the rank-1 bottleneck mechanism (M0) against standard mechanisms
(unconstrained coherent, incoherent sum) across 10 experimental tests from
BaBar, BESIII, Belle, CMS, ATLAS, and LHCb.

### Key Findings

| Category | Amplitude-Level | Proxy (Yield) | Total |
|----------|-----------------|---------------|-------|
| M0 SUPPORTED | 1 | 2 | 3 |
| M0 DISFAVORED | 2 | 0 | 2 |
| INCONCLUSIVE | 2 | - | 2 |
| MODEL MISMATCH | 3 | - | 3 |
| Other | 0 | - | 0 |

---

## Mechanism Definitions

| Mechanism | Description |
|-----------|-------------|
| M0 | Rank-1 Bottleneck (Factorization): Couplings factorize: g_{ia} = a_i * c_a. For 2-state, 2-chan... |
| M1 | Unconstrained Coherent: Each channel has independent complex ratios R_a, shared mass... |
| M2 | Incoherent Sum: Intensity = sum |BW_i|^2 + background, no interference |
| M3 | K-matrix P-vector rank-1: F = (I - iKrho)^{-1} P with rank-1 P-vector |
| M4 | Rank-2 Generalization: g_{ia} = a_i*c_a + b_i*d_a (for >2 channels or validating ra... |

---

## Tests Where M0 (Rank-1) is SUPPORTED

### CMS X(6900)/X(7100) Di-J/psi

- **Experiment**: CMS
- **Paper**: CMS-BPH-21-003 / arXiv:2306.07164
- **Lambda**: 1.8247718147333671
- **p-value**: 0.4033333333333333
- **chi2/dof**: A=1.016286225275991, B=2.486865269540029

**Interpretation**: Rank-1 constraint passes: p=0.403

### LHCb Pentaquark Pair1 T1vsT2 (quad bg) **(PROXY)**

- **Experiment**: LHCb
- **Paper**: PRL 122, 222001 (2019) / LHCb-PAPER-2019-014
- **Lambda**: -15.660395232758674
- **p-value**: 0.4
- **chi2/dof**: A=1.6988900222312775, B=1.1429000778378429

**Interpretation**: Rank-1 constraint passes: p=0.400

### LHCb Pentaquark Pair2 T1vsT3 (quad bg) **(PROXY)**

- **Experiment**: LHCb
- **Paper**: PRL 122, 222001 (2019)
- **Lambda**: -5.936433902511453
- **p-value**: 0.32
- **chi2/dof**: A=1.6988900222312775, B=1.390822894927072

**Interpretation**: Rank-1 constraint passes: p=0.320

---

## Tests Where M0 (Rank-1) is DISFAVORED

### BaBar phi-f0(980) vs phi-pi+pi-

- **Experiment**: BaBar
- **Paper**: PRD 86, 012008 (2012)
- **Lambda**: 342.139698270595
- **p-value**: 0.0
- **chi2/dof**: A=1.1413601393415096, B=2.815885552500829
- **ΔAIC (M0-M1)**: 338.1

**Interpretation**: Rank-1 rejected: p=0.0000, Lambda=342.139698270595

### BaBar omega(1420)/omega(1650)

- **Experiment**: BaBar
- **Paper**: PRD 76, 092005 (2007)
- **Lambda**: 958.2697198046197
- **p-value**: 0.0
- **chi2/dof**: A=1.1880750221892287, B=1.169367911389312

**Interpretation**: Rank-1 rejected: p=0.0000, Lambda=958.2697198046197

---

## Tests with INCONCLUSIVE Results

### Y(4220)/Y(4360) Belle vs BaBar

- **Reason**: Channel B underconstrained: chi2/dof=0.36
- **chi2/dof**: A=0.7029855439890009, B=0.3636043216335064

### BESIII Y(4220)/Y(4360)

- **Reason**: Channel B chi2/dof=8.8 >> 3.0
- **chi2/dof**: A=1.6597225608806556, B=8.815631814751953

---

## Tests with MODEL MISMATCH

These tests have fit quality issues that prevent reliable mechanism comparison.

### ATLAS X(6900)/X(7200) Di-Charmonium

- **Health A**: UNKNOWN (chi2/dof = None)
- **Health B**: UNKNOWN (chi2/dof = None)
- **Issue**: 4mu+2pi chi2/dof=9.74 >> 3.0

### BaBar K*(892)K I=0 vs I=1

- **Health A**: UNDERCONSTRAINED (chi2/dof = 0.010358479044889986)
- **Health B**: UNDERCONSTRAINED (chi2/dof = 0.014611237717513341)
- **Issue**: chi2/dof << 0.5 (underconstrained)

### BaBar phi-eta vs K+K-eta

- **Health A**: UNDERCONSTRAINED (chi2/dof = 0.017519963603685088)
- **Health B**: UNDERCONSTRAINED (chi2/dof = 0.02294803805353423)
- **Issue**: chi2/dof << 0.5 (underconstrained)

---

## Top 3 Strongest M0 Supports

Tests with highest statistical confidence for rank-1 mechanism:

1. **CMS X(6900)/X(7100) Di-J/psi**: p=0.403, Λ=1.8247718147333671
2. **LHCb Pentaquark Pair1 T1vsT2 (quad bg) (PROXY)**: p=0.400, Λ=-15.660395232758674
3. **LHCb Pentaquark Pair2 T1vsT3 (quad bg) (PROXY)**: p=0.320, Λ=-5.936433902511453

---

## Top 3 Strongest M0 Rejections

Tests with highest statistical confidence against rank-1 mechanism:

1. **BaBar omega(1420)/omega(1650)**: Λ=958.3, p=0.0000
2. **BaBar phi-f0(980) vs phi-pi+pi-**: Λ=342.1, p=0.0000

---

## Physics Interpretation

### Where Rank-1 Works

The rank-1 bottleneck mechanism is **supported** in amplitude-level tests:

- CMS X(6900)/X(7100) Di-J/psi (CMS)

These results suggest that the coupling structure g_{iα} = a_i * c_α may describe the underlying production mechanism for these exotic states.

Proxy tests (yield-ratio based) also support rank-1:

- LHCb Pentaquark Pair1 T1vsT2 (quad bg) (LHCb)
- LHCb Pentaquark Pair2 T1vsT3 (quad bg) (LHCb)

**Note**: Proxy tests provide indirect evidence as they don't test the full amplitude structure.

### Where Rank-1 Fails

The rank-1 constraint is **rejected** in:

- BaBar phi-f0(980) vs phi-pi+pi- (BaBar): Λ=342.139698270595
- BaBar omega(1420)/omega(1650) (BaBar): Λ=958.2697198046197

These systems likely have more complex production mechanisms that cannot be described by a rank-1 coupling matrix.

### Inconclusive Systems

Several systems require further investigation:

- Y(4220)/Y(4360) Belle vs BaBar: Channel B underconstrained: chi2/dof=0.36
- BESIII Y(4220)/Y(4360): Channel B chi2/dof=8.8 >> 3.0
- ATLAS X(6900)/X(7200) Di-Charmonium: 4mu+2pi chi2/dof=9.74 >> 3.0
- BaBar K*(892)K I=0 vs I=1: chi2/dof << 0.5 (underconstrained)
- BaBar phi-eta vs K+K-eta: chi2/dof << 0.5 (underconstrained)

These may benefit from higher-statistics data or improved amplitude models.

---

## Methodology Notes

1. **M0 (Rank-1)**: Tests whether both channels share a common complex ratio R = c₂·exp(iφ)/c₁
2. **Fit Health Gates**: Require 0.5 < χ²/dof < 3.0 for valid interpretation
3. **Bootstrap p-value**: Parametric bootstrap under H₀ (rank-1 constraint holds)
4. **SUPPORTED**: p ≥ 0.05 AND shared R within both 95% CL contours
5. **DISFAVORED**: p < 0.05 AND shared R outside 95% CL contours
6. **Proxy Tests**: Use yield ratios instead of full amplitude analysis

---

*Generated by mega_mechanism_sweep framework*
