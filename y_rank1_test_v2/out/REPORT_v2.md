# Y-state Rank-1 Bottleneck Test v2 Report

## Analysis Version
- Date: 2025-12-30 12:29
- Version: 2.0 (with nuisance parameters and strict health gates)

## Data Sources

| Channel | Reaction | Experiment | HEPData DOI | N points |
|---------|----------|------------|-------------|----------|
| A | e+e- -> pi+pi- J/psi | Belle | 10.17182/hepdata.61431.v1/t1 | 85 |
| B | e+e- -> pi+pi- psi(2S) | BaBar | 10.17182/hepdata.19344.v1/t1 | 35 |

## Uncertainty Treatment

### Channel A (Belle)
- Statistical errors: Point-wise independent (median = 3.91 pb)
- Systematic: 7.0% **correlated** -> modeled via nuisance parameter s0

### Channel B (BaBar)
- Statistical errors: Point-wise independent, asymmetric (median = 7.00 pb)
- Systematic: 12.3% **correlated** -> modeled via nuisance parameter s0

**Key fix vs v1**: Correlated systematics are NOT added in quadrature. Instead, they enter as a prior on a global scale nuisance parameter s0.

## Resonance Parameters (Fixed)

- Y(4220): M = 4.222 GeV, Γ = 0.044 GeV
- Y(4360): M = 4.368 GeV, Γ = 0.096 GeV

## Model

Coherent amplitude: A(E) = c1·BW₁ + c2·exp(iφ)·BW₂ + A_bg

Cross section with nuisance: σ(E) = s0·(1 + s1·(E-E₀))·|A(E)|²

Background order selected by AIC:
- Channel A: bg_order = 1
- Channel B: bg_order = 0

## Fit Results

### Channel A (Belle π⁺π⁻J/ψ)

| Parameter | Value |
|-----------|-------|
| r = \|c2/c1\| | 1.655 |
| Φ [deg] | -86.1 |
| s0 (nuisance) | 1.0000 |
| s1 (nuisance) | 0.000156 |
| χ²/dof | 0.703 (53.4/76) |
| NLL | 26.71 |
| Health | **PASS** |
| Optimizer | MULTIMODAL_R |

### Channel B (BaBar π⁺π⁻ψ(2S))

| Parameter | Value |
|-----------|-------|
| r = \|c2/c1\| | 4.726 |
| Φ [deg] | 29.1 |
| s0 (nuisance) | 1.0000 |
| s1 (nuisance) | -0.000104 |
| χ²/dof | 0.364 (10.2/28) |
| NLL | 5.09 |
| Health | **UNDERCONSTRAINED** |
| Optimizer | MULTIMODAL_R |

## Fit Health Gates

**Criterion**: 0.5 < χ²/dof < 3.0

| Channel | χ²/dof | Status |
|---------|--------|--------|
| A | 0.703 | PASS |
| B | 0.364 | UNDERCONSTRAINED |

## Joint Constrained Fit

- Shared R = 3.531 × exp(i × -165.6°)
- NLL_constrained = 32.16
- NLL_unconstrained = 31.80
- **Λ = 2×(NLL_con - NLL_unc) = 0.7056**

## Statistical Test

- Bootstrap replicates: 500/500
- **Bootstrap p-value = 0.7320**

## Profile Likelihood

- Shared R within Channel A 95% CL: **YES**
- Shared R within Channel B 95% CL: **YES**

## Magnitude-Only Check

| Metric | Value |
|--------|-------|
| r_shared in A 95% marginal | YES |
| r_shared in B 95% marginal | YES |

## Verdict

**INCONCLUSIVE**

Reason: Fit underconstrained: Channel B: chi2/dof=0.364 < 0.5

## Plots

- [Contour Overlay](contours_overlay.png)
- [Channel A Contours](contours_A.png)
- [Channel B Contours](contours_B.png)
- [Bootstrap Distribution](bootstrap_hist.png)
- [Optimizer Stability](optimizer_stability.png)

## Comparison with v1

| Metric | v1 | v2 |
|--------|----|----|
| Chi²/dof A | 0.632 | 0.703 |
| Chi²/dof B | 0.145 | 0.364 |
| Verdict | SUPPORTED | **INCONCLUSIVE** |
| Lower χ² gate | None | 0.5 |
| Syst treatment | Quadrature | Nuisance params |
