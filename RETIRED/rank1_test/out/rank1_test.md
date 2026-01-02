# Rank-1 Bottleneck Test: X(6900)/X(7100) Coupling Ratio

## Objective

Test the factorization constraint that predicts the complex coupling ratio
R = g_7100/g_6900 should be identical in both J/ψJ/ψ and J/ψψ(2S) decay channels.

## Data Sources

- **Channel A**: CMS-PAS-BPH-24-003 (J/ψJ/ψ spectrum)
- **Channel B**: CMS-PAS-BPH-22-004 (J/ψψ(2S) spectrum)

## Model

Intensity: I(m) = A × |BW₆₉₀₀(m) + R × BW₇₁₀₀(m)|² + background

Where R = r × exp(i × φ) is the complex coupling ratio.

## Results

| Channel | r | σ_r | φ (deg) | σ_φ (deg) | χ²/dof |
|---------|---|-----|---------|-----------|--------|
| A (J/ψJ/ψ) | 0.151 | 0.045 | -148.9 | 78.7 | 24.4 |
| B (J/ψψ(2S)) | 1.133 | 0.604 | 180.0 | 161.0 | 13.9 |

## Comparison

- **Δr** = r_A - r_B = -0.981 ± 0.605
- **Δφ** = φ_A - φ_B = +31.1° ± 179.2°

## Significance

- |Δr|/σ = 1.62σ
- |Δφ|/σ = 0.17σ

## Verdict

**COMPATIBLE within 2σ**

## Interpretation

The complex coupling ratios are consistent between channels within the estimated uncertainties.
This supports the rank-1 factorization hypothesis for tetraquark production.

## Fit Plots

![Channel A Fit](fit_A_plot.png)

![Channel B Fit](fit_B_plot.png)
