# LHCb Pentaquark Rank-1 Bottleneck Test v3

## Executive Summary

**Primary Result (WIDE window, quadratic background):**

| Metric | Value |
|--------|-------|
| Verdict | **UNSTABLE** |
| R_A | 0.2116 @ -64.0° |
| R_B | 0.2781 @ -34.9° |
| R_shared | 0.2388 @ -49.4° |
| Λ | 9.7399 |
| p-value | 0.0020 |

R varies 0.177 between backgrounds

## Data Provenance

| Item | Value |
|------|-------|
| HEPData Record | [89271](https://www.hepdata.net/record/ins1728691) |
| Publication | PRL 122, 222001 (2019) |
| DOI | [10.1103/PhysRevLett.122.222001](https://doi.org/10.1103/PhysRevLett.122.222001) |

**Tables Used:**
- Table 1: Full m(J/ψ p) spectrum (Poisson)
- Table 2: m(Kp) > 1.9 GeV cut spectrum (Poisson)

## Pentaquark Parameters (Fixed)

| State | Mass [MeV] | Width [MeV] |
|-------|------------|-------------|
| Pc(4312)⁺ | 4311.9 | 9.8 |
| Pc(4440)⁺ | 4440.3 | 20.6 |
| Pc(4457)⁺ | 4457.3 | 6.4 |

## Background Selection (AIC/BIC)

### Wide Window ((4270, 4520))

| Background | AIC | BIC |
|------------|-----|-----|
| Linear | 2682.12 | 2727.37 |
| Quadratic | 2286.58 | 2337.49 |

ΔAIC = -395.54 → **Selected: QUADRATIC**

### Tight Window ((4320, 4490))

| Background | AIC | BIC |
|------------|-----|-----|
| Linear | 1766.40 | 1805.48 |
| Quadratic | 1521.82 | 1565.79 |

ΔAIC = -244.58 → **Selected: QUADRATIC**

## Fit Quality

### Wide Window (quadratic background)

| Channel | χ²/dof | Deviance/dof | Gate (<3) |
|---------|--------|--------------|-----------|
| A | 1.699 | 1.702 | ✓ |
| B | 1.143 | 1.148 | ✓ |

### Tight Window (quadratic background)

| Channel | χ²/dof | Deviance/dof | Gate (<3) |
|---------|--------|--------------|-----------|
| A | 1.074 | 1.069 | ✓ |
| B | 1.152 | 1.151 | ✓ |

## Likelihood Ratio Test

### Wide Window

| Quantity | Value |
|----------|-------|
| NLL_unc | 1125.2904 |
| NLL_con | 1130.1603 |
| Λ = 2*(NLL_con - NLL_unc) | 9.7399 |
| Bootstrap p-value | 0.0020 |
| Valid replicates | 500/500 |
| Λ_boot median | 0.712 |
| Λ_boot 95th | 4.453 |

**Verdict: UNSTABLE**
R varies 0.177 between backgrounds

### Tight Window

| Quantity | Value |
|----------|-------|
| NLL_unc | 742.9088 |
| NLL_con | 744.6996 |
| Λ = 2*(NLL_con - NLL_unc) | 3.5817 |
| Bootstrap p-value | 0.0560 |
| Valid replicates | 500/500 |
| Λ_boot median | 0.616 |
| Λ_boot 95th | 3.762 |

**Verdict: SUPPORTED**
p = 0.0560, gates pass, stable

## Optimizer Audit Summary

All fits verified: Λ >= 0 (see OPTIMIZER_AUDIT.md for details)

## Files Generated

- `out/fit_*.png` - Fit diagnostic plots
- `out/contours_*.png` - Profile likelihood contours
- `out/OPTIMIZER_AUDIT.md` - Optimizer verification
- `out/REPORT_v3.md` - This report

---
*Analysis: Poisson NLL, 80 restarts main fits, 500 bootstrap replicates*
*Background selection: AIC-based with stability check*
